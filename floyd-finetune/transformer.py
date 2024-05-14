import copy
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import lightning as L

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def scaled_dot_attn(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, seqlen, norm, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.norm = norm
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = scaled_dot_attn(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class DecoderLayerGPT(nn.Module):
    def __init__(self, d_model, heads, seqlen, norm, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model, seqlen, norm, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderGPT(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, seqlen, norm, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayerGPT(d_model, heads, seqlen, norm, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class TransformerGPT(L.LightningModule):
    def __init__(self, vocab_size, d_model, N, heads, dropout, opt):
        super().__init__()
        self.decoder = DecoderGPT(vocab_size, d_model, N, heads, opt.seqlen, opt.norm, dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.opt = opt

        # Causal attention mask which ignores tokens beyond the current position
        causal_mask = torch.tril(torch.ones(opt.seqlen, opt.seqlen))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, trg, trg_mask=None):
        batch_size, seq_len = trg.shape
        mask = self.causal_mask[:seq_len,:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        if trg_mask is not None:
            trg_mask = trg_mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask = mask.masked_fill(trg_mask==0, 0)

        d_output = self.decoder(trg, mask)
        output = self.out(d_output)
        return output
    
    def configure_optimizers(self):
        raise NotImplementedError()
    
    def training_step(self, batch, idx):
        raise NotImplementedError()
    
    def validation_step(self, batch, idx):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt.lr, betas=(0.9, 0.98), eps=1e-9)
    
    def _calculate_loss(self, batch):
        data, labels, mask = batch  # TODO: apply the padding mask for each question?
        data = data
        preds = self(data, trg_mask=mask) # shape = [bsz, context_len, vocab_size]
        labels = labels.long()

        # Try David's custom-built F.cross_entropy that's only over the answer space despite technically allowing the model to output any token
        # NOTE: GPT2Tokenizer has this mapping: A-->317, B-->347, C-->327, D-->360
        batch_size, vocab_size = preds.shape
        denominators = torch.exp(preds[:, 317]) + torch.exp(preds[:, 347]) + torch.exp(preds[:, 327]) + torch.exp(preds[:, 360])
        numerators = torch.exp(preds[torch.arange(batch_size), labels])  # exponentiate score assigned to correct answer choice for each quesiton
        losses = -torch.log(numerators / denominators)
        loss = losses.sum() / batch_size

        # Also find the number of correct answers. The model's chosen answer is the one with the highest logit. 
        answer_mask = torch.ones(vocab_size, dtype=torch.bool, device=preds.device)
        answer_mask[[317, 327, 347, 360]] = False
        preds = preds.masked_fill(answer_mask, -float('inf'))
        choices = preds.argmax(dim=1)
        num_correct = torch.sum(choices == labels)
        return loss, num_correct
    
    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )

        # calculate norms for total update and layers' updates
        total_norm = 0.0
        for _, p in self.named_parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm ** (0.5)
        self.log(
            "grad_norm",
            total_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, num_correct = self._calculate_loss(batch)
        self.log(
            "val_loss",
            loss, 
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False
        )
        self.log(
            "val_accuracy",
            num_correct.float(),  # NOTE: 100% accuracy is 31*16 + 4 = 500 (and without the float conversion, I get a warning)
            on_step=False,
            on_epoch=True,
            reduce_fx="sum"
        )
    
def get_model(opt, vocab_size):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    
    model = TransformerGPT(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt)
       
    if opt.pretrained_path is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.pretrained_path))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model    
# !SECTION: Model Definitons