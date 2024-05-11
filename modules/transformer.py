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
        use_euclidean_attention=None
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
        # Check hparams num steps
        assert self.hparams.num_steps > 0

        # Causal attention mask which ignores tokens beyond the current position
        causal_mask = torch.tril(torch.ones(self.hparams.max_context_len, self.hparams.max_context_len))
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
            positional_temperatures=self.hparams.positional_temperatures,
            use_euclidean_attention=self.hparams.use_euclidean_attention
        )
        self.output_norm = nn.LayerNorm(self.hparams.model_dim)  # Decoder blocks normalize before their layers, so we need an extra norm here before our output MLP

        # Output classifier
        self.output_proj = nn.Linear(
            self.hparams.model_dim, 
            self.hparams.num_classes, 
            bias=False  # the nn.Embedding on the input has no bias; neither should the output embedding projection layer -- fairseq does this too
        )

    def _init_layers(self):
        # NOTE: Initializing linear layers like this overrides any layer-specific initialization, as layers' constructors are called
        #       in self._create_model()
        # LINK: Scaled Initialization from Spike No More: https://arxiv.org/pdf/2312.16903.pdf#page10
        sigma_main = math.sqrt(2/(5*self.hparams.model_dim))
        sigma_proj = sigma_main / math.sqrt(2*self.hparams.num_layers)

        nn.init.normal_(self.input_proj.weight, mean=0, std=sigma_main)
        nn.init.normal_(self.output_proj.weight, mean=0, std=sigma_main)
        self.transformer.init_layers(sigma_main, sigma_proj)

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen]
            pad_mask: Mask to apply on padding positions in the attention outputs (optional), shape is [Batch, SeqLen]
        """
        n_batches, seq_len = x.size()
        mask = self.causal_mask[:seq_len, :seq_len]       # grab causal mask for sequences of this size
        mask = mask.unsqueeze(0).repeat(n_batches, 1, 1)  # add batch dimension and copy causal mask along it
        if pad_mask is not None:                          # if supplied, 'pad_mask' will contain 0s at pad positions to ignore
            pad_mask = pad_mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask = mask.masked_fill(pad_mask == 0, 0)

        # Project inputs, scale embeddings, apply positional encoding, and apply dropout
        x = self.input_proj(x)
        x *= self.embed_scale # Original Vaswani paper says they scale embedding layers' weights by sqrt(model_dim). Spike No More says it is crucial to avoiding a type of gradient explosion. 
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Send data through the decoder layers and normalize outputs
        x = self.transformer(x, mask=mask)
        x = self.output_norm(x)

        # Project outputs
        x = self.output_proj(x)
        return x
    
    def get_hidden_states(self, x, pad_mask=None):
        """
        Same as forward, but omits the output projection at the last step. Useful when fine-tuning a new classification head at the last layer.
        """
        n_batches, seq_len = x.size()
        mask = self.causal_mask[:seq_len, :seq_len]       # grab causal mask for sequences of this size
        mask = mask.unsqueeze(0).repeat(n_batches, 1, 1)  # add batch dimension and copy causal mask along it
        if pad_mask is not None:                          # if supplied, 'pad_mask' will contain 0s at pad positions to ignore
            pad_mask = pad_mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask = mask.masked_fill(pad_mask == 0, 0)

        # Project inputs, scale embeddings, apply positional encoding, and apply dropout
        x = self.input_proj(x)
        x *= self.embed_scale # Original Vaswani paper says they scale embedding layers' weights by sqrt(model_dim). Spike No More says it is crucial to avoiding a type of gradient explosion. 
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Send data through the decoder layers and normalize outputs
        x = self.transformer(x, mask=mask)
        x = self.output_norm(x)
        return x
    
    def configure_optimizers(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError
#!SECTION
    
# SECTION: A Fine-Tuning Head Built on Top of a Causal Transformer
class FinetuneHead(L.LightningModule):
    def __init__(
            self, 
            pretrained_path, 
            num_classes, 
            num_steps, 
            lr=1e-3, 
            dropout=0.0, 
            attn_dropout=0.0, 
            activation_dropout=0.0
        ):
        '''FinetuneHead. Used to train a classification head on top of a pretrained Transformer. 

        Args:
            pretrained_path (str, Path): path to pretrained model
            num_classes (int): number of classes for the fine-tuning task
            lr (float): learning rate to apply during fine-tuning
            dropout (float): general dropout to apply to the Transformer during fine-tuning (e.g. after attention or embeddings)
            attn_dropout (float): dropout to apply to the attention layers of Transformer during fine-tuning
            activation_dropout (float): dropout to apply to the activations of FFNs of Transformer during fine-tuning
        '''
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        # Load pretrained model. Override lr, num_steps, and dropout proportions to better fit the fine-tuning task
        self.backbone = CausalTransformer.load_from_checkpoint(
            self.hparams.pretrained_path, 
            # dropout=self.hparams.dropout,
            # attn_dropout=self.hparams.attn_dropout,
            # activation_dropout=self.hparams.activation_dropout
        )

        # Deactivate original classification head and create a new one for the task
        # TODO: Play around with activation functions and dropout in the classification head?
        #      Could have a multi-layered head or apply them to the outputs of the pretrained model
        self.backbone.output_proj.weight.requires_grad = False  # output_proj has no bias term
        self.new_output_proj = nn.Linear(self.backbone.hparams.model_dim, self.hparams.num_classes, bias=False)

        # Initialize fine-tuning head using Spike No More
        # LINK: https://arxiv.org/pdf/2312.16903.pdf#page10
        sigma_main = math.sqrt(2/(5*self.backbone.hparams.model_dim))
        nn.init.normal_(self.new_output_proj.weight, mean=0, std=sigma_main)

    def forward(self, x, pad_mask=None):
        # Get the hidden states for each timestep
        x = self.backbone.get_hidden_states(x, pad_mask=pad_mask)

        # Grab the last hidden state and put it through our new classification head
        x = x[:, -1]
        x = self.new_output_proj(x)
        return x
    
    def configure_optimizers(self):
        # Filter params from new classification head vs. pretrained weights
        new_params = self.new_output_proj.parameters()
        pretrained_params = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))

        # Make param groups such that pretrained weights use a 10x smaller learning rate
        param_split = [
            {'params': new_params},
            {'params': pretrained_params, 'lr': self.hparams.lr / 10}
        ]

        # Initialize the optimizer and scheduler
        optimizer = optim.RAdam(
            param_split,
            lr=self.hparams.lr,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=1e-4
        )
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
# !SECTION: End of fine-tuning head