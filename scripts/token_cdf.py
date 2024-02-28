# Quick script to investigate what vocab size is needed to encode 95% of tokens. 
import torch

# counts = [0] * 32000
# train = torch.load('./data/wikitext-103/unigram.wiki.train.tokens.tokenized.pt')
# for batch in train:
#     for token in batch:
#         counts[token] += 1

# counts = torch.tensor(counts)
# torch.save(counts, 'unigram-counts.pt')

counts = torch.load('./unigram-counts.pt')
counts = torch.cumsum(counts, dim=0)
proportions = counts / counts[-1]
for i, prop in enumerate(reversed(proportions)):
    if prop < 0.95:
        print(32000 - i)
        break
    