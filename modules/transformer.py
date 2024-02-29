# SECTION: Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import math

from .pos_encoding import PositionalEncoding
from .encoder import TransformerEncoder
from .lr_scheduler import CosineWarmupRestartScheduler, REXScheduler
from .optimizers import Lion
#!SECTION

# SECTION: A Decoder-Only Transformer Lightning Module for Causal Language Modeling
# TODO: Add the final list of arguments to the docstring
class CausalTransformer(L.LightningModule):
    def __init__(
        self,
        num_classes,
        max_context_len=1024,
        model_dim=128,
        attention_norm=None,
        learn_temperatures=False,
        positional_temperatures=False,
        lr=0.0001,
        num_steps=-1,
        temperature_lr_scale=1.0,
        num_heads=8,
        num_layers=16,
        dropout=0.3,
        attn_dropout=0.1,
        activation_dropout=0.1,
        ffn_dim=4096,
        use_pos_encoding=True,
    ):
        """CausalTransformer.

        Args:
            input_dim: Hidden dimensionality of the input
            model_dim: Hidden dimensionality to use inside the Transformer
            num_classes: Number of classes to predict per sequence element
            num_heads: Number of heads to use in the Multi-Head Attention blocks
            lr: Learning rate in the optimizer
            max_iters: Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            warmup: Number of warmup steps. Usually between 50 and 500. Needed for CosineWarmupScheduler
            max_context_len: Maximum sequence size this transformer can accept
            num_heads=8: Number of heads to use inside the attention module of EncoderBlocks
            num_layers=16: Number of EncoderBlocks to use in the TransformerEncoder
            dropout=0.3: General dropout proportion applied inside the model
            attn_dropout=0.1: Dropout proportion passed to attention module of EncoderBlocks
            activation_dropout=0.1: Dropout proportion applied to activation of MLP layers in the EncoderBlocks
            ffn_dim=4096: Size of the MLP layers in the EncoderBlocks
            use_pos_encoding=True: Whether or not to use a sinusoidal positional encoding in this network
            use_euclidean_attention=False: Whether or not to use Euclidean Attention instead of classic MultiheadAttention
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self._init_layers()

    def _create_model(self):
        # Store learning rate in a state variable for learning rate tuning
        self.lr = self.hparams.lr

        # Check hparams num steps
        assert self.hparams.num_steps > 0

        # Causal attention mask which ignores tokens beyond the current position
        causal_mask = torch.triu(torch.ones(self.hparams.max_context_len, self.hparams.max_context_len), 1)
        causal_mask *= float('-inf')
        causal_mask = torch.nan_to_num(causal_mask)  # nan_to_num avoids operations that result in NaN instead of -inf
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # Input projection Layer
        # Using nn.Embedding allows us to skip one-hot encoding the token IDs, which introduces unnecessary data type conversions 
        self.input_proj = nn.Embedding(
            self.hparams.num_classes, 
            self.hparams.model_dim
        )
        self.embed_scale = math.sqrt(self.hparams.model_dim)

        # Positional encoding for sequences
        if self.hparams.use_pos_encoding:
            self.positional_encoding = PositionalEncoding(
                d_model=self.hparams.model_dim, 
                max_len=self.hparams.max_context_len
            )
        else:
            self.positional_encoding = nn.Identity()

        # Dropout Module for inputs
        self.dropout = nn.Dropout(self.hparams.dropout)

        # Transformer Decoder
        # NOTE: Not a Typo. In a Decoder-only architecture, the Decoder is architecturally equivalent to an Encoder.
        # NOTE: When normalizing before, an extra LayerNorm is used before feeding data to the output_net
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=self.hparams.ffn_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            attn_dropout=self.hparams.attn_dropout,
            activation_dropout=self.hparams.activation_dropout,
            max_context_len=self.hparams.max_context_len,
            attention_norm=self.hparams.attention_norm,
            learn_temperatures=self.hparams.learn_temperatures,
            positional_temperatures=self.hparams.positional_temperatures
        )
        self.output_norm = nn.LayerNorm(self.hparams.model_dim)  # Decoder blocks normalize before their layers, so we need an extra norm here before our output MLP

        # Output classifier
        self.output_proj = nn.Linear(
            self.hparams.model_dim, 
            self.hparams.num_classes, 
            bias=False  # the nn.Embedding on the input has no bias; neither should the output embedding projection layer -- fairseq does this too
        )

        # Store Q, K convex hull proportions of layers
        self.q_hull_props = {}
        self.k_hull_props = {}
        for i in range(len(self.transformer.layers)):
            self.q_hull_props[i] = []
            self.k_hull_props[i] = []

    def _init_layers(self):
        # NOTE: Initializing linear layers like this overrides any layer-specific initialization, as layers' constructors are called
        #       in self._create_model()
        # LINK: Scaled Initialization from Spike No More: https://arxiv.org/pdf/2312.16903.pdf#page10
        sigma_main = math.sqrt(2/(5*self.hparams.model_dim))
        sigma_proj = sigma_main / math.sqrt(2*self.hparams.num_layers)

        nn.init.normal_(self.input_proj.weight, mean=0, std=sigma_main)
        nn.init.normal_(self.output_proj.weight, mean=0, std=sigma_main)
        self.transformer.init_layers(sigma_main, sigma_proj)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)  # set bias to 0 initially
        #     elif isinstance(m, nn.Embedding):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')  # doesn't seem like it makes sense to use fan_in for the embedding

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen]
            pad_mask: Mask to apply on padding positions in the attention outputs (optional), shape is [Batch, SeqLen]
        """
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]
        if pad_mask is not None: # If supplied, 'pad_mask' will contain -inf at pad positions which should also be ignored
            mask = pad_mask.unsqueeze(-1).expand(-1, -1, seq_len) + mask  # pad_mask.shape --> [Batch, SeqLen, SeqLen] in order to add our causal mask with shape [SeqLen, SeqLen]

        # Project inputs, scale embeddings, apply positional encoding, and apply dropout
        x = self.input_proj(x)
        x *= self.embed_scale # Original Vaswani paper says they scale embedding layers' weights by sqrt(model_dim). Spike No More says it is crucial to avoiding a type of gradient explosion. 
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Send data through the decoder layers and normalize outputs
        x = self.transformer(x, self.q_hull_props, self.k_hull_props, mask=mask)
        x = self.output_norm(x)

        # Project outputs
        x = self.output_proj(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        """Function for extracting the attention matrices of the whole Transformer for a single batch.

        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        # Determine the learning rate for the temperatures' parameter group
        temperature_lr = self.lr * self.hparams.temperature_lr_scale

        # Split the model's params into temperature and non-temperature groups
        all_params = self.named_parameters()
        base_params = []
        temperature_params = []
        for name, param in all_params:  # objects generated by self.named_parameters() are 2-element tuples (str, torch.Tensor)
            if name.endswith('self_attn.temperatures'):
                temperature_params.append(param)
            else:
                base_params.append(param)

        # Create parameter split dictionary object to pass to optimizers
        param_split = [
            {'params': base_params},
            {'params': temperature_params, 'lr': temperature_lr}
        ]

        # Initialize the optimizer
        # This seems to be the closest to fairseq's NAGOptimizer (https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/nag.py)
        # optimizer = optim.SGD(
        #     self.parameters(), 
        #     lr=self.lr,
        #     momentum=0.99,
        #     weight_decay=0.0,
        #     nesterov=True
        # )
        # optimizer = Lion(
        #     param_split,
        #     lr=self.lr, # because 'lr' wasn't set for base_params in 'param_split', it will use this value by default
        #     weight_decay=0.00
        # )
        optimizer = optim.RAdam(
            param_split,
            lr=self.lr,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=1e-4
        )
        # optimizer = optim.AdamW(
        #     param_split,
        #     lr=1e-5, # to match warmup_init_lr
        #     betas=(0.9, 0.95),
        #     weight_decay=1e-4
        # )

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        # self.lr_scheduler = CosineWarmupRestartScheduler(
        #     optimizer,
        #     warmup_updates=5e3,
        #     warmup_init_lr=1e-5,
        #     warmup_end_lr=self.hparams.lr,
        #     min_lr=1e-5,
        #     lr_period_updates=int(1e5 - 5e3),
        #     t_mult=1
        # )
        self.lr_scheduler = REXScheduler(
            optimizer,
            num_steps=self.hparams.num_steps
        )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
#!SECTION
