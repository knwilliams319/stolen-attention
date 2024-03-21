# SECTION: Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.spatial as sp
#!SECTION

# SECTION: Base Attention Class
class AttentionMechanism(nn.Module):
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

        #self.q_layernorm = nn.LayerNorm(self.head_dim)
        #self.k_layernorm = nn.LayerNorm(self.head_dim)

        # Stack all weight matrices 1...h together for efficiency
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim, bias=False)  # PaLM doesn't use bias terms here
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        # Learned temperature params
        if self.learn_temperatures:
            if self.positional_temperatures:
                self.temperatures = nn.Parameter(torch.Tensor(self.max_context_len), requires_grad=True)
            else:
                self.temperatures = nn.Parameter(torch.Tensor(1), requires_grad=True)

    def init_modules(self, sigma_main, sigma_proj):
        nn.init.normal_(self.qkv_proj.weight, mean=0, std=sigma_main)
        # nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.normal_(self.o_proj.weight, mean=0, std=sigma_proj)
        nn.init.constant_(self.o_proj.bias, 0)

        if self.learn_temperatures:
            nn.init.uniform_(self.temperatures, 0.95, 1.05) # draw uniformly around 1, which is technically the temperature for F.softmax

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Apply QK-Layernorm
        # LINK: https://arxiv.org/pdf/2309.14322.pdf
        #q = self.q_layernorm(q)
        #k = self.k_layernorm(k)

        # Sniff Q, K, to calculate how many embeddings from each are in the convex hull
        # NOTE: While debugging, this sometimes fails due to deficient Q, K. Some layers must have really strange projections!
        # try:
        #     q_hull = sp.ConvexHull(q[0][0].cpu()) # take first batch of first head
        #     q_vertex_prop = len(q_hull.vertices) / q_hull.npoints  # How many vectors form the convex hull
        #     q_hull_props[layer_idx].append(q_vertex_prop)
        # except sp._qhull.QhullError:
        #     pass
        # try:
        #     k_hull = sp.ConvexHull(k[0][0].cpu()) # take first batch of first head
        #     k_vertex_prop = len(k_hull.vertices) / k_hull.npoints  # How many vectors form the convex hull
        #     k_hull_props[layer_idx].append(k_vertex_prop)
        # except sp._qhull.QhullError:
        #     pass

        # Get attention logits and add attention mask
        attn_logits = self.get_logits(q, k)
        if mask is not None:
            attn_logits += mask.unsqueeze(1)  # add head dimension for proper broadcasting

        # Retrieve attention weights and values
        attention = self.softmax_fn(attn_logits, dim=-1)
        attention = self.dropout(attention)
        values = torch.matmul(attention, v)
    
        # Reshape for output projection
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        # Return outputs
        if return_attention:
            return o, attention
        else:
            return o
        
    def get_logits(self, q, k):
        raise NotImplementedError
    
    def softmax_fn(self, x, dim=-1, temp_eps=1e-3):
        if self.learn_temperatures:
            _, _, seq_len, _ = x.shape  # [bsz, n_heads, seq_len, seq_len]
            temperatures = torch.square(self.temperatures[:seq_len]) + temp_eps  # square temps so they are nonnegative, then apply eps

            # NOTE: When temperatures are positionally learned, since `temperatures` has shape (`seq_len`,), broadcasting rules ensure 
            #       that columns (positions) of the attention logit matrices `x` receive the same scaling factor. Otherwise, `temperatures` 
            #       has shape (1,) and is applied elementwise. 
            x *= temperatures
        return F.softmax(x, dim=dim)
#!SECTION
    
# SECTION: Normal DotProductAttention Implementation
class DotProductAttention(AttentionMechanism):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 num_heads,
                 max_context_len,
                 dropout=0.0,
                 learn_temperatures=False,
                 positional_temperatures=False):
        super().__init__(input_dim, embed_dim, num_heads, max_context_len, dropout=dropout,
                         learn_temperatures=learn_temperatures, positional_temperatures=positional_temperatures)
        
    def get_logits(self, q, k):
        '''
        Get attention logits after which softmax (possibly with temperatures) will be applied.
        Logits will be caluclated via Scaled Dot Product.
        '''
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.mT)  # .mT returns a view equivalent to k.transpose(-2, -1)
        attn_logits = attn_logits / math.sqrt(d_k)
        return attn_logits
#!SECTION

# SECTION: EuclideanAttention Implementation
class EuclideanAttention(AttentionMechanism):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 num_heads,
                 max_context_len,
                 dropout=0.0,
                 learn_temperatures=False,
                 positional_temperatures=False):
        super().__init__(input_dim, embed_dim, num_heads, max_context_len, dropout=dropout,
                         learn_temperatures=learn_temperatures, positional_temperatures=positional_temperatures)
        
    def get_logits(self, q, k):
        '''
        Get attention logits after which softmax (possibly with temperatures) will be applied.
        Logits will be caluclated via Negative Squared Distance.
        '''
        # NOTE: For the same batch size, torch.square(torch.cdist(q, k, norm=2)) is faster than below.
        #       However, it takes more memory, making it slower if we push batch sizes as high as possible. 
        # Calculate batched pairwise negative squared distance between embeddings in Q, K
        Q_sq = torch.sum(torch.square(q), axis=3).unsqueeze(3)
        K_sq = torch.sum(torch.square(k), axis=3).unsqueeze(2)
        QK_dot = torch.matmul(q, k.mT)  # .mT returns a view equivalent to k.transpose(-2, -1)
        attn_logits = -(Q_sq - 2*QK_dot + K_sq)
        return attn_logits
#!SECTION
    
# SECTION: ManhattanAttention Implementation
class ManhattanAttention(AttentionMechanism):
    def __init__(self, 
                 input_dim, 
                 embed_dim, 
                 num_heads,
                 max_context_len,
                 dropout=0.0,
                 learn_temperatures=False,
                 positional_temperatures=False):
        super().__init__(input_dim, embed_dim, num_heads, max_context_len, dropout=dropout,
                         learn_temperatures=learn_temperatures, positional_temperatures=positional_temperatures)
        
    def get_logits(self, q, k):
        '''
        Get attention logits after which softmax (possibly with temperatures) will be applied.
        Logits will be caluclated via Negative Manhattan Distance.
        '''
        # Calculate batched pairwise negative manhattan distance between embeddings in Q, K
        q = q.unsqueeze(3)
        k = k.unsqueeze(2)
        attn_logits = -torch.sum(torch.abs(q-k), dim=-1) # negate distances so smaller ones have larger weights
        return attn_logits
#!SECTION