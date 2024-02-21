# SECTION: Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#!SECTION

# SECTION: EuclideanAttention Implementation
class EuclideanAttention(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 num_heads,
                 max_context_len,
                 dropout=0.0,
                 learn_temperatures=False,
                 positional_temperatures=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        assert max_context_len > 0, "Maximum expected context length must be greater than 0."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.learn_temperatures = learn_temperatures
        self.positional_temperatures = positional_temperatures
        self.max_context_len = max_context_len

        self.dropout = nn.Dropout(dropout)

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize temperature params
        if self.learn_temperatures:
            if self.positional_temperatures:
                self.temperatures = nn.Parameter(torch.Tensor(self.max_context_len), requires_grad=True)
            else:
                self.temperatures = nn.Parameter(torch.Tensor(1), requires_grad=True)
            nn.init.uniform_(self.temperatures, 0.95, 1.05) # draw uniformly around 1, which is technically the temperature for F.softmax

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.negative_euclidean(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
    def negative_euclidean(self, q, k, v, mask=None):
        # TODO: Talk to David about the performance boosts from using torch.cdist
        #       Seems like there's room for a custom CUDA kernel that's as fast as torch.cdist but skips the sqrt
        
        # Calculate batched pairwise negative squared distance between embeddings in Q, K
        Q_sq = torch.sum(torch.square(q), axis=3).unsqueeze(3)
        K_sq = torch.sum(torch.square(k), axis=3).unsqueeze(2)
        QK_dot = torch.matmul(q, k.mT)  # .mT returns a view equivalent to k.transpose(-2, -1)
        attn_logits = -(Q_sq - 2*QK_dot + K_sq)  # negative squared euclidean (ramps up to 1.18 it/s after 30s)

        # attn_logits = -torch.cdist(q, k)  # this is negative L2 norm (ramps up to 2.6 it/s after 30s)
        # attn_logits = -torch.square(attn_logits)  # negative squared euclidean (ramps up to 1.36 it/s after 30s)

        if mask is not None:
            attn_logits += mask.unsqueeze(1)  # add head dimension for proper broadcasting

        attention = self.softmax_fn(attn_logits, dim=-1)
        attention = self.dropout(attention)
        values = torch.matmul(attention, v)
        return values, attention
    
    def softmax_fn(self, *args, **kwargs):
        if not self.learn_temperatures:
            return F.softmax(*args, **kwargs)
        else:
            return self._softmax_with_temperatures(*args, **kwargs)
    
    def _softmax_with_temperatures(self, x, dim=-1, eps=1e-3):
        """
        Applies `self.temperature` position-wise to scale `x` prior to softmax over dimension `dim`.
        This is called in place of F.softmax only if temperature parameters are to be learned.
        """
        _, _, seq_len, _ = x.shape  # [bsz, n_heads, seq_len, seq_len]
        temperatures = torch.square(self.temperatures[:seq_len]) + eps  # square temps so they are nonnegative, then apply eps
        #temperatures = torch.clamp(self.temperatures[:seq_len], min=0.5, max=1.5)  # clamping temperatures ensures they are nonnegative
        

        # NOTE: When temperatures are positionally learned, since `temperatures` has shape (`seq_len`,), broadcasting rules ensure 
        #       that columns (positions) of the attention logit matrices `x` receive the same scaling factor. Otherwise, `temperatures` 
        #       has shape (1,) and is applied elementwise. 
        x *= temperatures
        return F.softmax(x, dim=dim)
#!SECTION
    
# SECTION: MultiheadAttention Implementation
# TODO: Add temperatures to classic MultiheadAttention in case we want an ablation
class MultiheadAttention(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 num_heads,
                 dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)

        if mask is not None:
            attn_logits += mask.unsqueeze(1)  # add head dimension for proper broadcasting

        attention = F.softmax(attn_logits, dim=-1)
        attention = self.dropout(attention)
        values = torch.matmul(attention, v)
        return values, attention
#!SECTION
