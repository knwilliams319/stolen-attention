import torch
from typing import List

class TokenPacker:
    def __init__(self, context_length, pad_token):
        '''
        Creates a Tensor of size (R, `context_length`), where R is equal to `num_tokens` / `context_length`. 
        Meant to be used with a tokenizer. When examples' tokens are sent are sent to the TokenPacker,
        they will be crammed into as many rows of size `context_length` are required to host them. The last
        row will be crammed with `pad_token` if not filled. 
        '''
        self.rows = []
        self.curr_row = torch.zeros(context_length)
        self.curr_idx = 0
        self.context_length = context_length
        self.pad_token = pad_token

    def pack(self, tokens: List[int]) -> None:
        '''
        Packs the tokens into its storage.
        '''
        N = len(tokens)
        if self.curr_idx + N < self.context_length:
            self.curr_row[self.curr_idx:self.curr_idx+N] = torch.tensor(tokens)
            self.curr_idx += N
        else:
            remainder = N # number of tokens that can't fit in self.curr_row
            tokens_consumed = 0 # number of tokens consumed so far
            while remainder > 0:
                num_to_accept = min(self.context_length - self.curr_idx, N - tokens_consumed)
                self.curr_row[self.curr_idx:self.curr_idx+num_to_accept] = torch.tensor(tokens[tokens_consumed:tokens_consumed+num_to_accept])
                if num_to_accept == self.context_length - self.curr_idx: # row is full
                    self.rows.append(self.curr_row)
                    self.curr_idx = 0
                    self.curr_row = torch.zeros(self.context_length)
                else:
                    self.curr_idx += num_to_accept
                tokens_consumed += num_to_accept
                remainder -= num_to_accept
                self.curr_idx = 0

    def to_tensor(self, dtype=torch.float16) -> torch.Tensor:
        '''
        Returns a 2D tensor representing the packed tokens in type dtype. Defaults to float16.
        '''
        self.curr_row[self.curr_idx:] = self.pad_token
        self.rows.append(self.curr_row)
        ret = torch.cat(self.rows).to(dtype)
        self.rows.pop() # we don't want to permanently store this in case more tokens are packed later
        return ret