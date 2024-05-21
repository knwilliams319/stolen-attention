import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

from transformer import TransformerGPT

def OutText(text,opt,screen=True):
    if screen:
        print(text)
    if opt.log_file:
        outFile = open(opt.log_file,"a+")
        outFile.write(text+"\n")
        
def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

def read_indices(filename):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split()
            for t in tokens:
                seq.append(int(t))
    return(seq)

# class Embedder(nn.Module):
#     def __init__(self, vocab_size, d_model):
#         super().__init__()
#         self.d_model = d_model
#         self.embed = nn.Embedding(vocab_size, d_model)
#     def forward(self, x):
#         return self.embed(x.int())

# class PositionalEncoder(nn.Module):
#     def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.dropout = nn.Dropout(dropout)
#         # create constant 'pe' matrix with values dependant on 
#         # pos and i
#         pe = torch.zeros(max_seq_len, d_model)
#         for pos in range(max_seq_len):
#             for i in range(0, d_model, 2):
#                 pe[pos, i] = \
#                 math.sin(pos / (10000 ** ((2 * i)/d_model)))
#                 pe[pos, i + 1] = \
#                 math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x):
#         # make embeddings relatively larger
#         x = x * math.sqrt(self.d_model)
#         #add constant to embedding
#         seq_len = x.size(1)
#         pe = Variable(self.pe[:,:seq_len], requires_grad=False)
#         if x.is_cuda:
#             pe.cuda()
#         x = x + pe
#         return self.dropout(x)
    
# class Norm(nn.Module):
#     def __init__(self, d_model, eps = 1e-6):
#         super().__init__()
    
#         self.size = d_model
        
#         # create two learnable parameters to calibrate normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
        
#         self.eps = eps
    
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#         / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
#         return norm

# def attention(q, k, v, d_k, mask=None, dropout=None):
    
#     scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
#     if mask is not None:
#         mask = mask.unsqueeze(1)
#         scores = scores.masked_fill(mask == 0, -1e9)
    
#     scores = F.softmax(scores, dim=-1)
    
#     if dropout is not None:
#         scores = dropout(scores)
        
#     output = torch.matmul(scores, v)
#     return output

# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads, d_model, seqlen, norm, dropout = 0.1):
#         super().__init__()
        
#         self.d_model = d_model
#         self.d_k = d_model // heads
#         self.h = heads
        
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.sigma = torch.ones([seqlen,seqlen],dtype=torch.float32)
#         self.sigma = self.sigma.cuda()
#         self.norm = norm
        
#         self.dropout = nn.Dropout(dropout)
#         self.out = nn.Linear(d_model, d_model)
    
#     def forward(self, q, k, v, mask=None):
        
#         bs = q.size(0)
        
#         # perform linear operation and split into N heads
#         k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
#         q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
#         v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
#         # transpose to get dimensions bs * N * sl * d_model
#         k = k.transpose(1,2)
#         q = q.transpose(1,2)
#         v = v.transpose(1,2)

#         # calculate attention using function we will define next
#         scores = attention(q, k, v, self.d_k, mask, self.dropout)

#         # concatenate heads and put through final linear layer
#         concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
#         output = self.out(concat)
    
#         return output

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff=2048, dropout = 0.1):
#         super().__init__() 
    
#         # We set d_ff as a default to 2048
#         self.linear_1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)
    
#     def forward(self, x):
#         x = self.dropout(F.relu(self.linear_1(x)))
#         x = self.linear_2(x)
#         return x
    
# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):

#     def __init__(self,
#                  optimizer: torch.optim.Optimizer,
#                  T_max: int,
#                  eta_min: float = 0.,
#                  last_epoch: int = -1,
#                  factor: float = 1.) -> None:
#         # pylint: disable=invalid-name
#         self.T_max = T_max
#         self.eta_min = eta_min
#         self.factor = factor
#         self._last_restart: int = 0
#         self._cycle_counter: int = 0
#         self._cycle_factor: float = 1.
#         self._updated_cycle_len: int = T_max
#         self._initialized: bool = False
#         super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         """Get updated learning rate."""
#         # HACK: We need to check if this is the first time get_lr() was called, since
#         # we want to start with step = 0, but _LRScheduler calls get_lr with
#         # last_epoch + 1 when initialized.
#         if not self._initialized:
#             self._initialized = True
#             return self.base_lrs

#         step = self.last_epoch + 1
#         self._cycle_counter = step - self._last_restart

#         lrs = [
#             (
#                 self.eta_min + ((lr - self.eta_min) / 2) *
#                 (
#                     np.cos(
#                         np.pi *
#                         ((self._cycle_counter) % self._updated_cycle_len) /
#                         self._updated_cycle_len
#                     ) + 1
#                 )
#             ) for lr in self.base_lrs
#         ]

#         if self._cycle_counter % self._updated_cycle_len == 0:
#             # Adjust the cycle length.
#             self._cycle_factor *= self.factor
#             self._cycle_counter = 0
#             self._updated_cycle_len = int(self._cycle_factor * self.T_max)
#             self._last_restart = step

#         return lrs    
    
# class DecoderLayerGPT(nn.Module):
#     def __init__(self, d_model, heads, seqlen, norm, dropout=0.1):
#         super().__init__()
#         self.norm_1 = Norm(d_model)
#         self.norm_2 = Norm(d_model)
        
#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
        
#         self.attn_1 = MultiHeadAttention(heads, d_model, seqlen, norm, dropout=dropout)
#         self.ff = FeedForward(d_model, dropout=dropout)
        
#     def forward(self, x, mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.ff(x2))
#         return x

# class DecoderGPT(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads, seqlen, norm, dropout):
#         super().__init__()
#         self.N = N
#         self.embed = Embedder(vocab_size, d_model)
#         self.pe = PositionalEncoder(d_model, dropout=dropout)
#         self.layers = get_clones(DecoderLayerGPT(d_model, heads, seqlen, norm, dropout), N)
#         self.norm = Norm(d_model)
#     def forward(self, trg, mask):
#         x = self.embed(trg)
#         x = self.pe(x)
#         for i in range(self.N):
#             x = self.layers[i](x, mask)
#         return self.norm(x)
    
# class TransformerGPT(nn.Module):
#     def __init__(self, vocab_size, d_model, N, heads, dropout,opt):
#         super().__init__()
#         self.decoder = DecoderGPT(vocab_size, d_model, N, heads, opt.seqlen, opt.norm, dropout)
#         self.decoder = self.decoder.cuda()
#         self.out = nn.Linear(d_model, vocab_size)
#         self.out = self.out.cuda()
#         self.opt = opt
#     def forward(self, trg, trg_mask):
#         d_output = self.decoder(trg, trg_mask)
#         if self.opt.tied == 0:
#             output = self.out(d_output)
#         else:
#             output = torch.matmul(d_output,self.decoder.embed(self.opt.indices).transpose(0,1))
                    
#         return output
    
def get_modelGPT(opt, vocab_size):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1
    
    # model = TransformerGPT(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model = TransformerGPT.load_from_checkpoint(
            Path(__file__).parent / 'experiments/best-run-b/epoch=4-step=8265.ckpt',
            vocab_size=vocab_size,
            d_model=opt.d_model,
            N=opt.n_layers,
            heads=opt.heads,
            dropout=opt.dropout,
            opt=opt
        )
        # model.load_state_dict(torch.load(Path(__file__).parent / 'experiments/best-run-b/epoch=4-step=8265.ckpt'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model    

def train_model(model, opt):
    
    print("training model...")
    model.train()
    count = 0

    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
    offsets = []
    stride = int(len(opt.train)/bb)
    for i in range(0,len(opt.train),stride):
        offsets.append(i)
    print('stride = ',stride)
    print('offsets = ',offsets)
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()

    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
                    
        for i in range(0,int((stride-aa)/1),aa):
            trg = torch.zeros((bb,aa),dtype=torch.long)
            for j in range(aa):
                for k in range(bb):
                    trg[k,j] = opt.train[offsets[k]+i+j]
            trg = trg.cuda()
                    
            preds = model(trg, mask)
            preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
            ys = trg[:, 1:].contiguous().view(-1)
            
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i*bb + 1) / len(opt.train))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)
        
        test_model(model, opt, epoch)

def train_fast(model, opt):
    
    print("training model...")
    model.train()
    count = 0

    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
    offsets = []
    stride = int(len(opt.train)/bb)
    for i in range(0,len(opt.train),stride):
        offsets.append(i)
    print('train = ',len(opt.train))
    print('stride = ',stride)
    print('offsets = ',offsets)
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    print(mask)
    
    packs = int(stride/aa)
    big_trg = torch.zeros((packs,bb,aa),dtype=torch.int64)
    print('post-alloc')
    for i in range(0,int((stride-aa)/1),aa):
        idx = int(i/aa)
        print(idx)
        for k in range(bb):
            for j in range(aa):
                big_trg[idx,k,j] = opt.train[offsets[k]+i+j]
    print('pre-cuda')
    big_trg = big_trg.cuda()
    print('post-cuda')

    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
                    
        for i in range(0,int((stride-aa)/1),aa):
#            trg = torch.zeros((bb,aa),dtype=torch.long)
#            for j in range(aa):
#                for k in range(bb):
#                    trg[k,j] = opt.train[offsets[k]+i+j]
#            trg = trg.cuda()
            idx = int(i/aa)
            trg = big_trg[idx,:,:]
                    
            preds = model(trg, mask)
            preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
            ys = trg[:, 1:].contiguous().view(-1)
            
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i*bb + 1) / len(opt.train))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)
        
        test_model(model, opt, epoch)
        
def train_fast(model, opt):
    
    print("training model...")
    model.train()
    count = 0

    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
    offsets = []
    stride = int(len(opt.train)/bb)
    for i in range(0,len(opt.train),stride):
        offsets.append(i)
    print('train = ',len(opt.train))
    print('stride = ',stride)
    print('offsets = ',offsets)
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    print(mask)
    
    packs = int(stride/aa)
    big_trg = torch.zeros((packs,bb,aa),dtype=torch.int64)
    print('post-alloc')
    for i in range(0,int((stride-aa)/1),aa):
        idx = int(i/aa)
        if (idx % 1000) == 0:
            print(idx)
        for k in range(bb):
            for j in range(aa):
                big_trg[idx,k,j] = opt.train[offsets[k]+i+j]
    print('pre-cuda')
    big_trg = big_trg.cuda()
    print('post-cuda')

    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
                    
        for i in range(0,int((stride-aa)/1),aa):
#            trg = torch.zeros((bb,aa),dtype=torch.long)
#            for j in range(aa):
#                for k in range(bb):
#                    trg[k,j] = opt.train[offsets[k]+i+j]
#            trg = trg.cuda()
            idx = int(i/aa)
            trg = big_trg[idx,:,:]
                    
            preds = model(trg, mask)
            preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
            ys = trg[:, 1:].contiguous().view(-1)
            
            opt.optimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i*bb + 1) / len(opt.train))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)
        
        test_model(model, opt, epoch)
        
def test_model(model, opt, epoch):
    
    model.eval()
    start = time.time()

    aa = opt.seqlen
    bb = 1
    opt.bb = bb
    
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()

    count = 0

    total_loss = 0
    for i in range(0,len(opt.test)-2*aa,aa):
        count = count + 1
        trg = torch.zeros((1,aa),dtype=torch.long)
        for j in range(aa):
            for k in range(bb):
                trg[k,j] = opt.test[i+j]
        trg = trg.cuda()
        
        preds = model(trg, mask)
        preds = preds[:,:-1,:]
        ys = trg[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
        total_loss += loss.item()
        
    text = ' '
    #OutText(text,opt)
    avg_loss = total_loss/count
    ppl = math.exp(avg_loss)
    text = "%dm: TEST %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss,math.exp(avg_loss))
    #OutText(text,opt)
    text = ' '
    #OutText(text,opt)
        
    model.train()
    opt.bb = opt.batchsize
    
def finetune_a(model,opt,tokenizer):
    print('finetuning....')
    model.train()
    count = 0
    
    data = []
    with open('obqa.train.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))

    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    for text in [' A',' B',' C',' D']:
        tokens = tokenizer(text)
        print(text,' = ',tokens['input_ids'])
    
    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
        correct = 0
        incorrect = 0
        
        for i in range(len(data)):
            
            text = ' ' + data[i]['Answer']
            tokens = tokenizer(text)
            ans = tokens['input_ids'][0]
                        
            text = '[CLS] Fact: %s Stem: %s A: %s B: %s C: %s D: %s [END] %s' % (data[i]['fact'],
                                                                                data[i]['stem'],
                                                                                data[i]['A'],
                                                                                data[i]['B'],
                                                                                data[i]['C'],
                                                                                data[i]['D'],
                                                                                data[i]['Answer'])
            tokens = tokenizer(text)
            pos = len(tokens['input_ids']) - 1
            trg = torch.zeros((bb,aa),dtype=torch.long)
            for j in range(len(tokens['input_ids'])):
                for k in range(bb):
                    trg[k,j] = tokens['input_ids'][j]
            trg = trg.cuda()
                                
            preds = model(trg, mask)
            logits = preds
#            preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
#            ys = trg[:, 1:].contiguous().view(-1)
            
            pos = len(tokens['input_ids']) - 1
            denom = 0.0
            denom = denom + math.exp(logits[0,pos-1,317])
            denom = denom + math.exp(logits[0,pos-1,347])
            denom = denom + math.exp(logits[0,pos-1,327])
            denom = denom + math.exp(logits[0,pos-1,360])
            probA = math.exp(logits[0,pos-1,317]) / denom
            probB = math.exp(logits[0,pos-1,347]) / denom
            probC = math.exp(logits[0,pos-1,327]) / denom
            probD = math.exp(logits[0,pos-1,360]) / denom
            probs = [probA,probB,probC,probD]
            labels = ['A','B','C','D']
            idx = probs.index(max(probs))
            
#            text = '%4d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %s %s' % (i,
#                                                                 probA*100.0,probB*100.0,probC*100.0,probD*100.0,
#                                                                  labels[idx],data[i]['Answer'])
            if labels[idx] == data[i]['Answer']:
                correct = correct + 1
#                text = text + '*'
            else:
                incorrect = incorrect + 1
#            #OutText(text,opt)
            
            prob = torch.exp(logits[0,pos-1,ans])/(torch.exp(logits[0,pos-1,317])+torch.exp(logits[0,pos-1,347])+torch.exp(logits[0,pos-1,327])+torch.exp(logits[0,pos-1,360]))
            loss = -torch.log(prob)
            opt.optimizer.zero_grad()
#            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i*bb + 1) / len(opt.train))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
        testmodel_a(model,opt,tokenizer,epoch)
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)            

def testmodel_a(model,opt,tokenizer,epoch):
    print('validating....')
    model.train()
    count = 0
    
    data = []
    with open('obqa.test.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))

    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    for text in [' A',' B',' C',' D']:
        tokens = tokenizer(text)
        print(text,' = ',tokens['input_ids'])
    
    start = time.time()
    total_loss = 0
    total = 0
    correct = 0
    incorrect = 0

    for i in range(len(data)):

        text = ' ' + data[i]['Answer']
        tokens = tokenizer(text)
        ans = tokens['input_ids'][0]

        text = '[CLS] Fact: %s Stem: %s A: %s B: %s C: %s D: %s [END] %s' % (data[i]['fact'],
                                                                            data[i]['stem'],
                                                                            data[i]['A'],
                                                                            data[i]['B'],
                                                                            data[i]['C'],
                                                                            data[i]['D'],
                                                                            data[i]['Answer'])
        tokens = tokenizer(text)
        pos = len(tokens['input_ids']) - 1
        trg = torch.zeros((bb,aa),dtype=torch.long)
        for j in range(len(tokens['input_ids'])):
            for k in range(bb):
                trg[k,j] = tokens['input_ids'][j]
        trg = trg.cuda()

        preds = model(trg, mask)
        logits = preds
#        preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
#        ys = trg[:, 1:].contiguous().view(-1)
#        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)

        pos = len(tokens['input_ids']) - 1
        denom = 0.0
        denom = denom + math.exp(logits[0,pos-1,317])
        denom = denom + math.exp(logits[0,pos-1,347])
        denom = denom + math.exp(logits[0,pos-1,327])
        denom = denom + math.exp(logits[0,pos-1,360])
        probA = math.exp(logits[0,pos-1,317]) / denom
        probB = math.exp(logits[0,pos-1,347]) / denom
        probC = math.exp(logits[0,pos-1,327]) / denom
        probD = math.exp(logits[0,pos-1,360]) / denom
        probs = [probA,probB,probC,probD]
        labels = ['A','B','C','D']
        idx = probs.index(max(probs))

#            text = '%4d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %s %s' % (i,
#                                                                 probA*100.0,probB*100.0,probC*100.0,probD*100.0,
#                                                                  labels[idx],data[i]['Answer'])
        if labels[idx] == data[i]['Answer']:
            correct = correct + 1
#                text = text + '*'
        else:
            incorrect = incorrect + 1
#            #OutText(text,opt)

        prob = torch.exp(logits[0,pos-1,ans])/(torch.exp(logits[0,pos-1,317])+torch.exp(logits[0,pos-1,347])+torch.exp(logits[0,pos-1,327])+torch.exp(logits[0,pos-1,360]))
        loss = -torch.log(prob)
        total_loss += loss.item()
        total = total + 1
        count = count + 1
            
    text = ' '
    #OutText(text,opt)
    avg_loss = total_loss/count
    ppl = math.exp(avg_loss)
    text = "%dm: TEST %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f ppl = %7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss,math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
    #OutText(text,opt)
    text = ' '
    #OutText(text,opt)
        
    model.train()

def finetune_b(model,opt,tokenizer):
    print('finetuning....')
    model.train()
    count = 0
    
    data = []
    with open('obqa.train.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))
    
    maxlen = -1
    for i in range(len(data)):
        text = '[CLS] Fact: %s Stem: %s A: %s B: %s C: %s D: %s [END] %s' % (data[i]['fact'],
                                                                            data[i]['stem'],
                                                                            data[i]['A'],
                                                                            data[i]['B'],
                                                                            data[i]['C'],
                                                                            data[i]['D'],
                                                                            data[i]['Answer'])
        tokens = tokenizer(text)
        if len(tokens['input_ids']) > maxlen:
            maxlen = len(tokens['input_ids'])
    print('maxlen = ',maxlen)
    
    aa = opt.seqlen    
    bb = opt.batchsize
    opt.bb = bb
    offsets = []
    stride = int(len(data)/bb)
    for i in range(0,len(data),stride):
        offsets.append(i)
    print('stride = ',stride)
    print('offsets = ',offsets)
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    for text in [' A',' B',' C',' D']:
        tokens = tokenizer(text)
        print(text,' = ',tokens['input_ids'])
    
    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
        correct = 0
        incorrect = 0
        
        for i in range(0,stride,1):
            trg = torch.zeros((bb,aa),dtype=torch.long)
            numer_mask = torch.zeros((bb,aa,opt.vocab_size),dtype=torch.long)
            denom_mask = torch.zeros((bb,aa,opt.vocab_size),dtype=torch.long)
            
            for k in range(bb):
                o = offsets[k]
                # get tokenized id of answer
                text = ' ' + data[i+o]['Answer']
                tokens = tokenizer(text)
                ans = tokens['input_ids'][0]
                        
                # create finetuning sequence
                text = '[CLS] Fact: %s Stem: %s A: %s B: %s C: %s D: %s [END] %s' % (data[i+o]['fact'],
                                                                                    data[i+o]['stem'],
                                                                                    data[i+o]['A'],
                                                                                    data[i+o]['B'],
                                                                                    data[i+o]['C'],
                                                                                    data[i+o]['D'],
                                                                                    data[i+o]['Answer'])
                tokens = tokenizer(text)
                pos = len(tokens['input_ids']) - 1
                for j in range(len(tokens['input_ids'])):
                    trg[k,j] = tokens['input_ids'][j]
                numer_mask[k,pos-1,ans] = 1.0
                denom_mask[k,pos-1,317] = 1.0
                denom_mask[k,pos-1,347] = 1.0
                denom_mask[k,pos-1,327] = 1.0
                denom_mask[k,pos-1,360] = 1.0
            trg = trg.cuda()
            numer_mask = numer_mask.cuda()
            denom_mask = denom_mask.cuda()
                                
            preds = model(trg, mask)
            logits = torch.exp(preds)
            denom = torch.sum(torch.sum(logits * denom_mask,dim=2),dim=1)
            numer = torch.sum(torch.sum(logits * numer_mask,dim=2),dim=1)
            probs = numer / denom
            
            incorrect = incorrect + 1
            loss = -torch.log(probs)
            loss = torch.sum(loss,dim=0)/float(bb)
            opt.optimizer.zero_grad()
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i*bb + 1) / len(data))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
        testmodel_b(model,opt,tokenizer,epoch)
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)            

def testmodel_b(model,opt,tokenizer,epoch):
    print('validating....')
    model.eval()
    count = 0
    
    data = []
    with (Path(__file__).parent / 'data/obqa.test.txt').open('rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))

    aa = opt.seqlen    
    bb = 1
    opt.bb = bb
       
    nopeak_mask = np.triu(np.ones((bb,aa,aa),dtype=np.int32),k=1)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    for text in [' A',' B',' C',' D']:
        tokens = tokenizer(text)
        print(text,' = ',tokens['input_ids'])
    
    start = time.time()
    total_loss = 0
    total = 0
    correct = 0
    incorrect = 0

    for i in range(len(data)):

        text = ' ' + data[i]['Answer']
        tokens = tokenizer(text)
        ans = tokens['input_ids'][0]

        text = '[CLS] Fact: %s Stem: %s A: %s B: %s C: %s D: %s [END] %s' % (data[i]['fact'],
                                                                            data[i]['stem'],
                                                                            data[i]['A'],
                                                                            data[i]['B'],
                                                                            data[i]['C'],
                                                                            data[i]['D'],
                                                                            data[i]['Answer'])
        tokens = tokenizer(text)
        pos = len(tokens['input_ids']) - 1
        trg = torch.zeros((bb,aa),dtype=torch.long)
        for j in range(len(tokens['input_ids'])):
            for k in range(bb):
                trg[k,j] = tokens['input_ids'][j]
        trg = trg.cuda()

        preds = model(trg, mask)
        logits = preds
#        preds = preds[:,:-1,:].contiguous().view(-1,preds.size(2))
#        ys = trg[:, 1:].contiguous().view(-1)
#        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)

        denom = 0.0
        denom = denom + math.exp(logits[0,pos-1,317])
        denom = denom + math.exp(logits[0,pos-1,347])
        denom = denom + math.exp(logits[0,pos-1,327])
        denom = denom + math.exp(logits[0,pos-1,360])
        probA = math.exp(logits[0,pos-1,317]) / denom
        probB = math.exp(logits[0,pos-1,347]) / denom
        probC = math.exp(logits[0,pos-1,327]) / denom
        probD = math.exp(logits[0,pos-1,360]) / denom
        probs = [probA,probB,probC,probD]
        labels = ['A','B','C','D']
        idx = probs.index(max(probs))

#            text = '%4d %7.1f%% %7.1f%% %7.1f%% %7.1f%% %s %s' % (i,
#                                                                 probA*100.0,probB*100.0,probC*100.0,probD*100.0,
#                                                                  labels[idx],data[i]['Answer'])
        if labels[idx] == data[i]['Answer']:
            correct = correct + 1
#                text = text + '*'
        else:
            incorrect = incorrect + 1
#            #OutText(text,opt)

        prob = torch.exp(logits[0,pos-1,ans])/(torch.exp(logits[0,pos-1,317])+torch.exp(logits[0,pos-1,347])+torch.exp(logits[0,pos-1,327])+torch.exp(logits[0,pos-1,360]))
        loss = -torch.log(prob)
        total_loss += loss.item()
        total = total + 1
        count = count + 1
            
    text = ' '
    #OutText(text,opt)
    avg_loss = total_loss/count
    ppl = math.exp(avg_loss)
    text = "%dm: TEST %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f ppl = %7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss,math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
    #OutText(text,opt)
    text = ' '
    #OutText(text,opt)
        
    model.train()
    
head = torch.rand((768,4),dtype=float)
head = head.cuda()
    
def finetune_c(model,opt,tokenizer):
    print('finetuning....')
    model.train()
    count = 0
    
    data = []
    with (Path(__file__).parent / 'data/obqa.train.txt').open() as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))
    
    maxlen = -1
    for i in range(len(data)):
        for a in ['A','B','C','D']:
            text = '[CLS] %s %s %s[END]' % (data[i]['fact'],data[i]['stem'],data[i][a])
            tokens = tokenizer(text)
            if len(tokens['input_ids']) > maxlen:
                maxlen = len(tokens['input_ids'])
    print('maxlen = ',maxlen)
    
    aa = opt.seqlen    
    bb = 4
    opt.bb = bb
       
    nopeak_mask = np.ones((bb,aa,aa),dtype=np.int32)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    denom_mask = torch.ones((bb,aa,opt.vocab_size),dtype=torch.long)
    denom_mask = denom_mask.cuda()
    
    for epoch in range(opt.epochs):
        start = time.time()
        total_loss = 0
        total = 0
        correct = 0
        incorrect = 0
        
        for i in range(len(data)):
            trg = torch.zeros((bb,aa),dtype=torch.long)
            numer_mask = torch.zeros((bb,aa,opt.vocab_size),dtype=torch.long)
            lens = torch.zeros((bb),dtype=torch.float)
            for k in range(bb):
                labels = ['A','B','C','D']
                ans = labels.index(data[i]['Answer'])
                text = '[CLS] %s %s %s[END]' % (data[i]['fact'],data[i]['stem'],data[i][labels[k]])
                tokens = tokenizer(text)
                lens[k] = len(tokens['input_ids'])
                for j in range(len(tokens['input_ids'])):
                    trg[k,j] = tokens['input_ids'][j]
                    numer_mask[k,j,trg[k,j]] = 1.0
            trg = trg.cuda()
            numer_mask = numer_mask.cuda()
            lens = lens.cuda()
                                
            preds = model(trg, mask)
            logits = torch.exp(preds)
            denom = torch.sum(torch.sum(logits * denom_mask,dim=2),dim=1)
            numer = torch.sum(torch.sum(logits * numer_mask,dim=2),dim=1)
            probs = numer / denom
            probs = probs / lens
            target = probs[ans] / torch.sum(probs,dim=0)
            
            incorrect = incorrect + 1
            loss = -torch.log(target)
            opt.optimizer.zero_grad()
            loss.backward()
            opt.optimizer.step()
            if opt.SGDR == True:
                opt.sched.step()
            
            total_loss += loss.item()
            total = total + 1
            
            count = count + 1
            if count % int(opt.printevery/bb) == 0:
                count = 0
                p = int(100 * (i + 1) / len(data))
                avg_loss = total_loss/total
                text = "   %dm: epoch %d [%s%s]  %d%%  wps = %7.0f loss = %.3f % 7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, float(aa*bb*total)/float(time.time() - start), avg_loss, math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
                #OutText(text,opt)
            
        if opt.savename is not None:
            print('saving weights...')
            torch.save(model.state_dict(), opt.savename + '/model_weights')   
        testmodel_c(model,opt,tokenizer,epoch)
   
        text = "%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, wps = %7.0f loss = %.03f ppl = %7.1f" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, float(aa*bb*total)/float(time.time()-start), avg_loss,math.exp(avg_loss))
        #OutText(text,opt)            

def testmodel_c(model,opt,tokenizer,epoch):
    print('validating....')
    model.eval()
    count = 0
    
    data = []
    with open('obqa.train.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split('|')
            d = {}
            d['fact'] = tokens[0]
            d['stem'] = tokens[1]
            d['A'] = tokens[2]
            d['B'] = tokens[3]
            d['C'] = tokens[4]
            d['D'] = tokens[5]
            d['Answer'] = tokens[6]   
            data.append(d)
    print('data: %d' % (len(data)))
    
    maxlen = -1
    for i in range(len(data)):
        for a in ['A','B','C','D']:
            text = '[CLS] %s %s %s[END]' % (data[i]['fact'],data[i]['stem'],data[i][a])
            tokens = tokenizer(text)
            if len(tokens['input_ids']) > maxlen:
                maxlen = len(tokens['input_ids'])
    print('maxlen = ',maxlen)
    
    aa = opt.seqlen    
    bb = 4
    opt.bb = bb
       
    nopeak_mask = np.ones((bb,aa,aa),dtype=np.int32)
    mask = Variable(torch.from_numpy(nopeak_mask) == 0)
    mask = mask.cuda()
    
    denom_mask = torch.ones((bb,aa,opt.vocab_size),dtype=torch.long)
    denom_mask = denom_mask.cuda()
    
    start = time.time()
    total_loss = 0
    total = 0
    correct = 0
    incorrect = 0

    for i in range(len(data)):
        trg = torch.zeros((bb,aa),dtype=torch.long)
        numer_mask = torch.zeros((bb,aa,opt.vocab_size),dtype=torch.long)
        lens = torch.zeros((bb),dtype=torch.float)
        for k in range(bb):
            labels = ['A','B','C','D']
            ans = labels.index(data[i]['Answer'])
            text = '[CLS] %s %s %s[END]' % (data[i]['fact'],data[i]['stem'],data[i][labels[k]])
            tokens = tokenizer(text)
            lens[k] = len(tokens['input_ids'])
            for j in range(len(tokens['input_ids'])):
                trg[k,j] = tokens['input_ids'][j]
                numer_mask[k,j,trg[k,j]] = 1.0
        trg = trg.cuda()
        numer_mask = numer_mask.cuda()
        lens = lens.cuda()

        preds = model(trg, mask)
        logits = torch.exp(preds)
        denom = torch.sum(torch.sum(logits * denom_mask,dim=2),dim=1)
        numer = torch.sum(torch.sum(logits * numer_mask,dim=2),dim=1)
        probs = numer / denom
        probs = probs / lens
        target = probs[ans] / torch.sum(probs,dim=0)

        guesses = []
        for b in range(4):
            guesses.append(probs[b].item())
        if ans == guesses.index(max(guesses)):
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        loss = -torch.log(target)
        total_loss += loss.item()
        total = total + 1
        count = count + 1
            
    text = ' '
    #OutText(text,opt)
    avg_loss = total_loss/count
    ppl = math.exp(avg_loss)
    text = "%dm: TEST %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f ppl = %7.1f %7.1f%%" % ((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss,math.exp(avg_loss),100.0*float(correct)/float(correct+incorrect))
    #OutText(text,opt)
    text = ' '
    #OutText(text,opt)
        
    model.train()


def main():
    
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-n_layers', type=int, default=12)
    parser.add_argument('-heads', type=int, default=12)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
#    parser.add_argument('-model', type=str)    
#    parser.add_argument('-style', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
#    parser.add_argument('-resid', type=int, default=1)
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    opt.device = torch.device("cuda:0")
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
#    dir_name = "gaussian_transformer//" + time_name
    # NOTE: I'm commenting out this whole block
    # dir_name = "saved/%s" % (opt.dir_name)
    # if not os.path.exists(dir_name):
    #     os.makedirs(dir_name)
    # source_name = sys.argv[0]
    # dir_name = dir_name + "//"
    # opt.dir_name = dir_name
    # shutil.copy(source_name,dir_name + source_name)
    # opt.log_file = dir_name + "santa.txt"
    
    # text = opt.log_file
    # #OutText(opt.log_file,opt)   
    # #OutText(str(opt),opt)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")    
    # if False:
    #     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    #     opt.train = read_corpus('wiki.train.txt',tokenizer)
    #     opt.valid = read_corpus('wiki.valid.txt',tokenizer)
    #     opt.test = read_corpus('wiki.test.txt',tokenizer)
    #     with open('wiki.train.pkl','wb') as f:
    #         pickle.dump(opt.train,f)
    #         f.close()
    #     with open('wiki.valid.pkl','wb') as f:
    #         pickle.dump(opt.valid,f)
    #         f.close()
    #     with open('wiki.test.pkl','wb') as f:
    #         pickle.dump(opt.test,f)
    #         f.close()
    #     sys.exit(0)
    # else:
    #     with open('wiki.train.pkl','rb') as f:
    #         opt.train = pickle.load(f)
    #         f.close()
    #     with open('wiki.valid.pkl','rb') as f:
    #         opt.valid = pickle.load(f)
    #         f.close()
    #     with open('wiki.test.pkl','rb') as f:
    #         opt.test = pickle.load(f)
    #         f.close()
    
    # obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    opt.indices = opt.indices.cuda()
    
    opt.loadname = 'blah' # anything but None
    model = get_modelGPT(opt,opt.vocab_size)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    # text = 'total params: %d' % (params)
    # #OutText(text,opt)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.savename is not None:
        try:
            os.mkdir(opt.savename)
        except:
            nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0
            
#    train_model(model,opt)
#    train_fast(model,opt)
#    finetune_a(model,opt,tokenizer)
#    finetune_b(model,opt,tokenizer)
#    finetune_c(model,opt,tokenizer)
    testmodel_b(model, opt, tokenizer, 0)
        
if __name__ == "__main__":
    main()     
