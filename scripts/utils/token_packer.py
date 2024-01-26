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
        if self.curr_idx + len(tokens) < self.context_length:  # tokens fit in self.curr_row
            self.curr_row[self.curr_idx:self.curr_idx+len(tokens)] = torch.tensor(tokens)  # curr_row is a tensor
            self.curr_idx += len(tokens)
        else:  # tokens don't fit, and may be large enough to span multiple rows
            remainder = len(tokens) # number of tokens that can't fit in self.curr_row
            idx = 0 # index of next token to insert
            while remainder > 0:
                can_fit = min(self.context_length - self.curr_idx, remainder)
                self.curr_row[self.curr_idx:self.curr_idx+can_fit] = torch.tensor(tokens[idx:idx+can_fit])  # fill row
                self.curr_idx += can_fit
                if self.curr_idx == self.context_length: # row is full
                    self.rows.append(self.curr_row)
                    self.curr_idx = 0
                    self.curr_row = torch.zeros(self.context_length)
                idx += can_fit
                remainder -= can_fit

    def to_tensor(self, dtype=torch.float16) -> torch.Tensor:
        '''
        Returns a 2D tensor representing the packed tokens in type dtype. Defaults to float16.
        '''
        self.curr_row[self.curr_idx:] = self.pad_token
        self.rows.append(self.curr_row)
        ret = torch.stack(self.rows).to(dtype)
        self.rows.pop() # we don't want to permanently store this in case more tokens are packed later
        return ret