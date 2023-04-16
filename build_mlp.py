import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making plots

# read in all the words
words = open('names.txt').read().splitlines()
words[:8]

# build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)

# build the dataset
block_size = 3
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context[:])
        Y.append(ix)
        print(''.join([itos[i] for i in context]), '--->', itos[ix])
        context = context[1:] + [ix]

# convert to tensors
X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
# construct the hidden layer
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
sum(p.nelement() for p in parameters)

# forward pass
emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
# compute the loss manually (multiple tensors created)
# counts = logits.exp()
# prob = counts / counts.sum(dim=1, keepdim=True)
# compute the loss
# loss = -prob[torch.arange(32), Y].log().mean()
# print(loss)
# use F.cross_entropy to compute the loss 
# reasons for using this function:
# 1. fused kernel used internally, simpler math calcs in backprop
# 2. numerical stability
loss = F.cross_entropy(logits, Y)
print(loss)
# test stability of cross entropy
# logits = torch.tensor([-5, -3, 0, 5])
# counts = logits.exp()
# probs = counts / counts.sum()
# print(probs)

