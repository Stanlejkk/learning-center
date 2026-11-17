"""Reference solution for labs/info_01_entropy_compression.ipynb (keep in sync with the notebook's TODO cells)."""
from __future__ import annotations

import heapq
import itertools
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np


def normalize(text: str) -> str:
    """Lowercase, keep a-z and single spaces (27-symbol alphabet)."""
    text = re.sub(r"[^a-z]+", " ", text.lower())
    return re.sub(r" +", " ", text).strip()


def entropy(p) -> float:
    """Shannon entropy in bits of a probability vector (zeros allowed)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def unigram_entropy(text: str) -> float:
    c = Counter(text)
    n = len(text)
    return entropy([v / n for v in c.values()])


def conditional_bigram_entropy(text: str) -> float:
    """H(X_{i+1} | X_i) = H(X_i, X_{i+1}) - H(X_i), estimated from consecutive pairs."""
    pairs = Counter(zip(text, text[1:]))
    firsts = Counter(text[:-1])
    m = len(text) - 1
    return entropy([v / m for v in pairs.values()]) - entropy([v / m for v in firsts.values()])


def huffman_code(freqs: dict) -> dict:
    """Binary Huffman code: symbol -> bit string."""
    if len(freqs) == 1:
        return {next(iter(freqs)): "0"}
    cnt = itertools.count()
    heap = [(w, next(cnt), {s: ""}) for s, w in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        w1, _, c1 = heapq.heappop(heap)
        w2, _, c2 = heapq.heappop(heap)
        merged = {s: "0" + b for s, b in c1.items()}
        merged.update({s: "1" + b for s, b in c2.items()})
        heapq.heappush(heap, (w1 + w2, next(cnt), merged))
    return heap[0][2]


def encode(text: str, code: dict) -> str:
    return "".join(code[ch] for ch in text)


def decode(bits: str, code: dict) -> str:
    inv = {b: s for s, b in code.items()}
    out, cur = [], ""
    for b in bits:
        cur += b
        if cur in inv:
            out.append(inv[cur])
            cur = ""
    if cur:
        raise ValueError("trailing bits do not form a codeword")
    return "".join(out)


def avg_code_length(code: dict, freqs: dict) -> float:
    tot = sum(freqs.values())
    return sum(freqs[s] * len(code[s]) for s in freqs) / tot


def zlib_bits_per_char(text: str) -> float:
    return len(zlib.compress(text.encode("ascii"), 9)) * 8 / len(text)


_path = Path("data/english_sample.txt")
if _path.exists():
    TEXT = normalize(_path.read_text(encoding="utf-8"))
    FREQS = Counter(TEXT)
    CODE = huffman_code(FREQS)
    H1 = unigram_entropy(TEXT)
    H2 = conditional_bigram_entropy(TEXT)
