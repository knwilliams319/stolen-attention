# SECTION: Necessary imports
import torch.nn as nn

from .attention import MultiheadAttention, EuclideanAttention
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
                 use_euclidean_attention=False,
                 learn_temperatures=False,
                 positional_temperatures=False,
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
        if use_euclidean_attention:
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
            self.self_attn = MultiheadAttention(
                input_dim,
                input_dim,
                num_heads,
                dropout=attn_dropout
            )

        # Dropout modules
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(activation_dropout),
            nn.Linear(dim_feedforward, input_dim)
        )

    def forward(self, x, mask=None):
        # TODO: Implement these once I start noticing convergence
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
        x = self.linear_net(x)
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
#!SECTION