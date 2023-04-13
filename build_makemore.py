import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# read names.txt file
words = open('names.txt', 'r').read().splitlines()

len(words)

min(len(word) for word in words)
max(len(word) for word in words)

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {ch: i+1 for i, ch in enumerate(chars)}
stoi['.'] = 0
itos = {i: ch for ch, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# get a nice plot
# plt.figure(figsize=(16, 16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
#         plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')

# plt.axis('off')
# plt.show()


g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()

torch.multinomial(p, num_samples=100, replacement=True, generator=g)


p = N[0].float()
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)
idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

# print(itos[idx])

P = (N+1).float()
P /= P.sum(dim=1, keepdim=True)

g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # bigram model
        # p = N[ix].float()
        # p = p / p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


log_likelihood = 0.0
n = 0

for w in ["andrejq"]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{ch1}{ch2}: {prob: .4f} {logprob:.4f}')

print(f"{log_likelihood=}")
# use negative log likelihood
nll = -log_likelihood
print(f"{nll=}")
print(f"{nll/n=}")

# create a training set of bigrams (x,y)
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1, ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)


# create one-hot encoding and then cast to float32
xenc = F.one_hot(xs, num_classes=27).float()
plt.imshow(xenc)
plt.savefig("m1.png")

# random initialize the weight matrix
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g)

logits = xenc @ W # log-counts
counts = logits.exp() # equivalent N
probs = counts / counts.sum(dim=1, keepdim=True)
# btw: the last 2 lines here are together called a `softmax`

nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram
    x = xs[i].item() # input character index
    y = ys[i].item() # label character index
    print('-------')
    print(f'bigram example {i+1}:{itos[x]}{itos[y]} (indexes {x},{y})')
    print('input to the neural net:', x)
    print('output probabilities from the neural net:', probs[i])
    print('label (actual next character):', y)
    p = probs[i, y]
    print('probability assigned by the net to the correct character', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood:', nll.item())
    nlls[i] = nll

print("==========")
print('average negative log likelihood, i.e. loss =', nlls.mean().item())

# -------- !!! OPTIMIZATION (manual test)!!! --------
# random initialize the weight matrix
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network, one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent N
probs = counts / counts.sum(dim=1, keepdim=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean() # loss: negative log likelihood

print(loss.item())

# backward pass
W.grad = None # set to zero the gradient
loss.backward() # compute the gradient

# update the weights
W.data += -0.1 * W.grad

# -------- !!! OPTIMIZATION !!! --------
# create the dataset
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples:', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)


# gradient descent
for k in range(201):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network, one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent N
    probs = counts / counts.sum(dim=1, keepdim=True) # probabilities for next character
    # loss: negative log likelihood + regularization
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() 
    print(f'epoch {k+1}: loss = {loss.item()}')
    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward() # compute the gradient
    # update the weights
    W.data += -50 * W.grad

# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
        # ----------
        # BEFORE:
        p = P[ix]
        # ----------
        # NOW:
        # xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        # logits = xenc @ W # predict log-counts
        # counts = logits.exp() # counts, equivalent N
        # p = counts / counts.sum(dim=1, keepdim=True) # probabilities for next character
        # output sampled result
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

