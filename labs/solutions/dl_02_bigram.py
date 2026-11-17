"""Reference solution — dl_02_bigram: a character-level bigram language model, counted and learned.

Inspired by the 'makemore' lectures in Karpathy's nn-zero-to-hero (https://github.com/karpathy/nn-zero-to-hero);
written independently and trained on words extracted from TinyShakespeare.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

BOS = "."  # one boundary token marks both the start and the end of a word


def load_words(path="data/tinyshakespeare.txt", min_len=2):
    text = Path(path).read_text(encoding="utf-8").lower()
    return sorted({w for w in re.findall(r"[a-z]+", text) if len(w) >= min_len})


def build_vocab(words):
    chars = sorted(set("".join(words)))
    stoi = {BOS: 0, **{c: i + 1 for i, c in enumerate(chars)}}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def bigram_counts(words, stoi):
    V = len(stoi)
    N = torch.zeros((V, V), dtype=torch.int64)
    for w in words:
        seq = [BOS] + list(w) + [BOS]
        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1
    return N


def counts_to_probs(N, alpha=1.0):
    P = N.double() + alpha
    return P / P.sum(dim=1, keepdim=True)


def nll(words, P, stoi):
    total, n = 0.0, 0
    for w in words:
        seq = [BOS] + list(w) + [BOS]
        for a, b in zip(seq, seq[1:]):
            total -= math.log(P[stoi[a], stoi[b]].item())
            n += 1
    return total / n


def perplexity(avg_nll):
    return math.exp(avg_nll)


def sample_word(P, itos, generator=None, max_len=30):
    out, ix = [], 0
    for _ in range(max_len):
        ix = torch.multinomial(P[ix].float(), num_samples=1, generator=generator).item()
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out)


def make_bigram_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        seq = [BOS] + list(w) + [BOS]
        for a, b in zip(seq, seq[1:]):
            xs.append(stoi[a])
            ys.append(stoi[b])
    return torch.tensor(xs), torch.tensor(ys)


def train_neural_bigram(xs, ys, V, steps=200, lr=50.0, reg=0.0, seed=0):
    """One linear layer on one-hot inputs: logits = onehot(x) @ W. Full-batch gradient descent.
    Returns (W, losses)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((V, V), generator=g, requires_grad=True)
    xenc = F.one_hot(xs, num_classes=V).float()
    losses = []
    for _ in range(steps):
        logits = xenc @ W
        loss = F.cross_entropy(logits, ys) + reg * (W ** 2).mean()
        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
        losses.append(loss.item())
    return W.detach(), losses
