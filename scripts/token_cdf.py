# Quick script to calculate a histogram of tokens in the training set
import torch

# Create the bins
counts = [0] * 16000
train = torch.load('./data/wikitext-103/unigram.wiki.train.tokens.tokenized.pt')
for batch in train:
    for token in batch:
        counts[token] += 1

counts = torch.tensor(counts)
torch.save(counts, 'unigram-token-counts.pt')

# Use this part to find out the vocab size needed to capture 95% of tokens
# counts = torch.load('./unigram-token-counts.pt')
# counts = torch.cumsum(counts, dim=0)
# proportions = counts / counts[-1]
# for i, prop in enumerate(reversed(proportions)):
#     if prop < 0.95:
#         print(32000 - i)
#         break
    