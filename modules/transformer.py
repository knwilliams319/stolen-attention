# SECTION: Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from pos_encoding import PositionalEncoding
from encoder import TransformerEncoder
from lr_scheduler import CosineWarmupScheduler
#!SECTION

# SECTION: A Decoder-Only Transformer Lightning Module for Causal Language Modeling
class CausalTransformer(L.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        lr,
        max_iters,
        warmup,
        max_context_len,
        num_heads=8,
        num_layers=16,
        dropout=0.3,
        attn_dropout=0.1,
        activation_dropout=0.1,
        ffn_dim=4096,
        use_pos_encoding=True,
        use_euclidean_attention=False,
        use_projection_bias=False
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
            use_projection_bias: False (fairseq doesn't include a bias term in its input/output projection layers)
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()
        self._init_layers()

    def _create_model(self):
        # Input projection Layer
        self.input_proj = None  # if None, self._init_layers will set this to nn.Identity
        if self.hparams.input_dim != self.hparams.model_dim:
            self.input_proj = nn.Linear(self.hparams.input_dim, 
                                        self.hparams.model_dim, 
                                        bias=self.hparams.use_projection_bias)

        # Positional encoding for sequences
        if self.hparams.use_pos_encoding:
            self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim, 
                                                          max_len=self.hparams.max_context_len)
        else:
            self.positional_encoding = nn.Identity()

        # Dropout Module for inputs
        self.dropout = nn.Dropout(self.hparams.dropout)

        # Transformer Decoder
        # NOTE: Not a Typo. In a Decoder-only architecture, the Decoder is architecturally equivalent to an Encoder.
        # NOTE: When normalizing before, an extra LayerNorm is used before feeding data to the output_net
        self.transformer = nn.Sequential(
            TransformerEncoder(
                num_layers=self.hparams.num_layers,
                input_dim=self.hparams.model_dim,
                dim_feedforward=self.hparams.ffn_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout,
                attn_dropout=self.hparams.attn_dropout,
                activation_dropout=self.hparams.activation_dropout
            ),
            nn.LayerNorm(self.hparams.model_dim)  # Decoder blocks normalize before their layers, so we need an extra norm here
        )

        # Output classifier
        self.output_proj = nn.Linear(self.hparams.model_dim, 
                                     self.hparams.num_classes, 
                                     bias=self.hparams.use_projection_bias)

    def _init_layers(self):
        # Input projection layer
        if self.input_proj is None:
            self.input_proj = nn.Identity()
        else:
            nn.init.xavier_uniform_(self.input_proj.weight)
            if self.hparams.use_projection_bias:
                nn.init.constant_(self.input_proj.bias, 0.0)
        
        # Output projection layer
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.hparams.use_projection_bias:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input features of shape [Batch, SeqLen, input_dim]
            mask: Mask to apply on the attention outputs (optional)
        """
        # Project inputs, apply positional encoding, and apply dropout
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Send data through the decoder layers
        x = self.transformer(x, mask=mask)

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
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
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