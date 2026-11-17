"""Reference solution — dl_03_mlp_lm: a Bengio et al. (2003)-style MLP character language model.

Inspired by the 'makemore part 2' lecture in Karpathy's nn-zero-to-hero; written independently.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = "."


def load_words(path="data/tinyshakespeare.txt", min_len=2):
    text = Path(path).read_text(encoding="utf-8").lower()
    return sorted({w for w in re.findall(r"[a-z]+", text) if len(w) >= min_len})


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {BOS: 0, **{c: i + 1 for i, c in enumerate(chars)}}
    return stoi, {i: c for c, i in stoi.items()}


def build_dataset(words, stoi, block_size):
    """Every (context of block_size previous chars, next char) pair; contexts start padded with '.' (index 0)."""
    X, Y = [], []
    for w in words:
        ctx = [0] * block_size
        for ch in list(w) + [BOS]:
            ix = stoi[ch]
            X.append(list(ctx))
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def split_words(words, seed=42, frac=(0.8, 0.1)):
    ws = list(words)
    random.Random(seed).shuffle(ws)
    n1, n2 = int(frac[0] * len(ws)), int((frac[0] + frac[1]) * len(ws))
    return ws[:n1], ws[n1:n2], ws[n2:]


class MLPLM(nn.Module):
    """emb (V x n_embd) -> concat block_size embeddings -> tanh hidden layer -> logits over V."""

    def __init__(self, vocab_size, block_size, n_embd=10, n_hidden=200):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(block_size * n_embd, n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, idx):
        e = self.emb(idx)                       # (B, block, n_embd)
        h = torch.tanh(self.fc1(e.flatten(1)))  # (B, n_hidden)
        return self.fc2(h)                      # (B, V)


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    loss = F.cross_entropy(model(X), Y).item()
    model.train()
    return loss


def train(model, X, Y, steps=1000, lr=0.1, batch_size=32, seed=0, lr_decay_at=None):
    """Mini-batch SGD. Returns the list of per-step training losses."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        if lr_decay_at is not None and step == lr_decay_at:
            for group in opt.param_groups:
                group["lr"] = lr / 10
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def lr_range_test(model, X, Y, lrs, batch_size=32, seed=0):
    """One SGD step per learning rate (in increasing order), on a single model. Returns the loss at each step."""
    g = torch.Generator().manual_seed(seed)
    losses = []
    for lr in lrs:
        ix = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
        losses.append(loss.item())
    return losses


@torch.no_grad()
def sample(model, itos, n=5, generator=None, max_len=30):
    out = []
    for _ in range(n):
        ctx, w = [0] * model.block_size, []
        for _ in range(max_len):
            p = F.softmax(model(torch.tensor([ctx])), dim=1)
            ix = torch.multinomial(p, 1, generator=generator).item()
            if ix == 0:
                break
            w.append(itos[ix])
            ctx = ctx[1:] + [ix]
        out.append("".join(w))
    return out
