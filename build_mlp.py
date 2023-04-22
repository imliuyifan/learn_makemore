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

# # build the dataset
# block_size = 3
# X, Y = [], []
# for w in words:
#     # print(w)
#     context = [0] * block_size
#     for ch in w + '.':
#         ix = stoi[ch]
#         X.append(context[:])
#         Y.append(ix)
#         # print(''.join([itos[i] for i in context]), '--->', itos[ix])
#         context = context[1:] + [ix]

# # convert to tensors
# X = torch.tensor(X, dtype=torch.long)
# Y = torch.tensor(Y, dtype=torch.long)
# print(X.shape, Y.shape)

# build the dataset

def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context[:])
            Y.append(ix)
            # print(''.join([itos[i] for i in context]), '--->', itos[ix])
            context = context[1:] + [ix]
    # convert to tensors
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
# construct the hidden layer
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
num_param = sum(p.nelement() for p in parameters)
print(f'number of parameters: {num_param}')

# require grad for all parameters
for p in parameters:
    p.requires_grad = True

# try different learning rates
# lre = torch.linspace(-3, 0, 1000)
# lrs = 10 ** lre

# lri = []
# lossi = []

for i in range(10000):
    # mini-batch construct
    ix = torch.randint(0, X.shape[0], (32,))
    # forward pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])
    # print(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update parameters
    # lr = lrs[i]
    lr = 0.1
    for p in parameters:
        p.data += -lr * p.grad
    # track loss
    # lri.append(lre[i].item())
    # lossi.append(loss.item())

emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(loss.item())

# training split, dev/validation split, test split
# 80%, 10%, 10%
