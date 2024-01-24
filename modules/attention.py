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
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

        # Initialize temperature params
        if self.learn_temperatures:
            if self.positional_temperatures:
                self.temperatures = nn.Parameter(torch.Tensor(self.max_context_len), requires_grad=True)
            else:
                self.temperatures = nn.Parameter(torch.Tensor(1), requires_grad=True)
            nn.init.uniform_(self.temperatures, 0.8, 1.2) # draw uniformly around 1, which is technically the temperature for F.softmax

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
        # Calculate batched pairwise negative squared distance between embeddings in Q, K
        Q_sq = torch.sum(torch.square(q), axis=3).unsqueeze(3)
        K_sq = torch.sum(torch.square(k), axis=3).unsqueeze(2)
        QK_dot = torch.matmul(q, k.mT)  # .mT returns a view equivalent to k.transpose(-2, -1)
        attn_logits = -(Q_sq - 2*QK_dot + K_sq)

        if mask is not None:
            attn_logits += mask.unsqueeze(1)  # add head dimension to mask so addition is properly broadcasted

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
        Computes softmax along dimension `dim` of `input`. Applies `self.temperature` position-wise to scale
        `input` prior to exponentiation. This is called in place of F.softmax only if temperature parameters
        are to be learned. 

        Regardless, even if self.temperature = torch.tensor([1]), this function is no different than F.softmax(input, dim=dim).
        """
        n_batches, n_heads, seq_len, embed_dim = x.shape

        # TODO: how can I make this faster? Having to make a new tensor every time has to be slow...
        # NOTE: we square the temperature params to ensure they are never negative
        # diagonalize the temperatures needed for the input of size sequence_length
        if not self.positional_temperatures:
            eye = torch.eye(seq_len,
                            device='cuda',
                            dtype=self.temperatures.dtype)
            temperatures = eye * (torch.square(self.temperatures) + eps)
        else:
            # embed the first seq_len temperatures into a diagonal matrix that matches dimensionality of x
            temperatures = torch.diag_embed(torch.square(self.temperatures[:seq_len]) + eps)

        # replace -inf from attention mask to smallest representable number. without this, x @ temperatures
        # results in NaNs which make the loss explode when we really just want small numbers that softmax will
        # end up in us ignoring. 
        x = torch.nan_to_num(x)  # TODO: run the debugger on this line. Unlike fairseq, I don't think I need to run this here. 

        x = torch.matmul(x, temperatures)
        x_max = torch.max(x, axis=dim, keepdims=True).values
        exp_x_shifted = torch.exp(x - x_max)
        return exp_x_shifted / torch.sum(exp_x_shifted, axis=dim, keepdims=True)
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

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

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

        # TODO: is this the ordering in which fairseq applies their mask?
        if mask is not None:
            attn_logits += mask.unsqueeze(1)  # add head dimension to mask

        attention = F.softmax(attn_logits, dim=-1)
        attention = self.dropout(attention)
        values = torch.matmul(attention, v)
        return values, attention
#!SECTION