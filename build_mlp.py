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
C = torch.randn((27, 10), generator=g)
# construct the hidden layer
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
num_param = sum(p.nelement() for p in parameters)
print(f'number of parameters: {num_param}')

# require grad for all parameters
for p in parameters:
    p.requires_grad = True

# try different learning rates
lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = []
lossi = []
stepi = []

for i in range(200000):
    # mini-batch construct
    ix = torch.randint(0, Xtr.shape[0], (32,))
    # forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])
    # print(loss.item())
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    # update parameters
    # lr = lrs[i]
    lr = 0.1 if i < 10000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    # track loss
    # lri.append(lre[i].item())
    stepi.append(i)
    lossi.append(loss.log10().item())

print(loss.item())

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

# training split, dev/validation split, test split
# 80%, 10%, 10%

# visualize the embedding
# plt.figure(figsize=(10, 10))
# plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
# for i in range(C.shape[0]):
#     plt.text(C[i, 0].item(), C[i, 1].item(), itos[i], ha='center', va='center', color='white')
# plt.grid('minor')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
block_size = 3

for _ in range(50):
    out = []
    context = [0] * block_size # initialize the all ...
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join([itos[i] for i in out]))
