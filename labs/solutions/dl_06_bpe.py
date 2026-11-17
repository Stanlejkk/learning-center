"""Reference solution — dl_06_bpe: a byte-level BPE tokenizer (train / encode / decode).

Inspired by Karpathy's minbpe (https://github.com/karpathy/minbpe); written independently.
Tie-breaking rule used throughout: among equally frequent pairs, pick the one that occurs FIRST in the sequence.
"""
from __future__ import annotations


def get_stats(ids, counts=None):
    """Count adjacent pairs (overlapping). Dict insertion order = first occurrence order."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every non-overlapping occurrence of `pair` (scanning left to right) by the new token `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    def __init__(self):
        self.merges = {}                                   # (int, int) -> int, in training order
        self.vocab = {i: bytes([i]) for i in range(256)}   # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for k in range(vocab_size - 256):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)  # max() returns the first maximal key -> first occurrence wins ties
            idx = 256 + k
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {k + 1}: {pair} -> {idx} ({self.vocab[idx]!r}) had {stats[pair]} occurrences")
        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))  # earliest-learned merge first
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
