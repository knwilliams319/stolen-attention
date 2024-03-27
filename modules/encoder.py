# SECTION: Necessary imports
import torch.nn as nn

from .attention import DotProductAttention, ManhattanAttention, EuclideanAttention
#!SECTION

# SECTION: A single Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self,
                 input_dim=128, 
                 dim_feedforward=4096,
                 num_heads=8,
                 dropout=0.3,
                 attn_dropout=0.1,
                 activation_dropout=0.1,
                 max_context_len=-1,
                 attention_norm=None,
                 learn_temperatures=False,
                 positional_temperatures=False,
                 use_euclidean_attention=None
                 ):
        """EncoderBlock.

        Args:
            input_dim: Dimensionality of the model
            dim_feedforward: Dimensionality of the hidden layer in the MLP
            num_heads: Number of heads to use in the attention block
            dropout: Dropout probability applied prior to adding residuals
            attn_dropout: Dropout probability passed to self attention module
            activation_dropout: Dropout probability applied after activation of MLP
            max_context_len: Maximum context length expected by this model (needed for EuclideanAttention only)
            use_euclidean_attention: Whether to use EuclideanAttention or normal MultiheadAttention
            learn_temperatures: Whether to learn temperature paramaters to control softmax
            positional_temperatures: Whether to learn position-wise temperatures to control softmax
        """
        super().__init__()

        # Layer normalization for inputs
        self.input_norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)

        # Attention layer
        if use_euclidean_attention is not None:
            # NOTE: This is a deprecated argument for the embed_dim_64/n_heads_8 experiments.
            #       It was created before I implemented ManhattanAttention, so it is a simple switch.
            #       Regardless, its value will be None for newer models; attention_norm should be used instead.
            if use_euclidean_attention == True:
                self.self_attn = DotProductAttention(
                input_dim,
                input_dim,
                num_heads,
                max_context_len,
                dropout=attn_dropout,
                learn_temperatures=learn_temperatures,
                positional_temperatures=positional_temperatures
            )
            else:
                self.self_attn = EuclideanAttention(
                input_dim,
                input_dim,
                num_heads,
                max_context_len,
                dropout=attn_dropout,
                learn_temperatures=learn_temperatures,
                positional_temperatures=positional_temperatures
            )
        elif attention_norm is None:
            self.self_attn = DotProductAttention(
                input_dim,
                input_dim,
                num_heads,
                max_context_len,
                dropout=attn_dropout,
                learn_temperatures=learn_temperatures,
                positional_temperatures=positional_temperatures
            )
        elif attention_norm == 1:
            self.self_attn = ManhattanAttention(
                input_dim,
                input_dim,
                num_heads,
                max_context_len,
                dropout=attn_dropout,
                learn_temperatures=learn_temperatures,
                positional_temperatures=positional_temperatures
            )
        elif attention_norm == 2:
            self.self_attn = EuclideanAttention(
                input_dim,
                input_dim,
                num_heads,
                max_context_len,
                dropout=attn_dropout,
                learn_temperatures=learn_temperatures,
                positional_temperatures=positional_temperatures
            )
        else:
            raise ValueError(f"Attention norm {attention_norm} is not supported!")

        # Dropout modules
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        # Two-layer MLP
        self.up_projection = nn.Linear(input_dim, dim_feedforward)
        self.act = nn.GELU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.down_projection = nn.Linear(dim_feedforward, input_dim)

    def init_modules(self, sigma_main, sigma_proj):
        nn.init.normal_(self.up_projection.weight, mean=0, std=sigma_main)
        nn.init.constant_(self.up_projection.bias, 0)
        nn.init.normal_(self.down_projection.weight, mean=0, std=sigma_proj)
        nn.init.constant_(self.down_projection.bias, 0)
        self.self_attn.init_modules(sigma_main, sigma_proj)

    def forward(self, x, mask=None):
        # TODO: Implement these once I start noticing convergence?
        # NOTE: These links seem to be parallel work on the same concept. I think the B2T Residual (1) has nicer graphics.
        # LINK (1): Bottom-to-Top Residual Connection: https://arxiv.org/pdf/2206.00330v1.pdf
        # LINK (2): ResiDual: https://arxiv.org/pdf/2304.14802.pdf
        # These improve the performance of networks that use Pre-LayerNorm
        
        # Normalize inputs and calculate attention
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x, mask=mask)

        # Add and norm
        x = self.dropout_1(x)
        x += residual
        residual = x
        x = self.ffn_norm(x)

        # MLP plus residual connection
        x = self.up_projection(x)
        x = self.act(x)
        x = self.act_dropout(x)
        x = self.down_projection(x)
        x = self.dropout_2(x)
        x += residual
        return x
#!SECTION
    
# SECTION: A full TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        """TransformerEncoder.

        Args:
            num_layers: Number of EncoderBlocks to use
            **block_args: Arguments to pass to each EncoderBlock
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps
    
    def init_layers(self, sigma_main, sigma_proj):
        for layer in self.layers:
            layer.init_modules(sigma_main, sigma_proj)
#!SECTION
            