"""Hidden checks for info_01_entropy_compression. Each check_*(ns) raises AssertionError on failure."""
from __future__ import annotations

import heapq
import math
import random
from collections import Counter

SAMPLE = "abracadabra alakazam the quick brown fox jumps over the lazy dog"


def _H(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0)


def _ref_huffman_len(freqs):
    tot = sum(freqs.values())
    heap = [w / tot for w in freqs.values()]
    heapq.heapify(heap)
    L = 0.0
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        L += a + b
        heapq.heappush(heap, a + b)
    return L


def check_entropy(ns):
    H = ns["entropy"]
    assert abs(H([0.5, 0.25, 0.25]) - 1.5) < 1e-9, "entropy([1/2,1/4,1/4]) should be 1.5 bits"
    assert abs(H([1.0])) < 1e-12, "a certain outcome has 0 entropy"
    assert abs(H([1 / 8] * 8) - 3) < 1e-9, "uniform on 8 outcomes is 3 bits"
    assert abs(H([0.5, 0.5, 0.0]) - 1) < 1e-9, "zero-probability outcomes must contribute 0 (0 log 0 = 0)"


def check_unigram_entropy(ns):
    c = Counter(SAMPLE)
    ref = _H([v / len(SAMPLE) for v in c.values()])
    got = ns["unigram_entropy"](SAMPLE)
    assert abs(got - ref) < 1e-6, f"unigram_entropy: expected {ref:.6f}, got {got}"


def check_conditional_bigram_entropy(ns):
    f = ns["conditional_bigram_entropy"]
    assert abs(f("ab" * 50)) < 1e-9, "in 'ababab…' the next char is determined: H(X2|X1) should be 0"
    pairs = Counter(zip(SAMPLE, SAMPLE[1:]))
    firsts = Counter(SAMPLE[:-1])
    m = len(SAMPLE) - 1
    ref = _H([v / m for v in pairs.values()]) - _H([v / m for v in firsts.values()])
    got = f(SAMPLE)
    assert abs(got - ref) < 1e-6, f"conditional_bigram_entropy: expected {ref:.6f}, got {got}"
    assert got <= ns["unigram_entropy"](SAMPLE) + 1e-9, "conditioning cannot increase entropy"


def check_huffman_prefix_free_and_optimal(ns):
    rng = random.Random(0)
    for trial in range(5):
        k = rng.randint(2, 12)
        freqs = {chr(97 + i): rng.randint(1, 50) for i in range(k)}
        code = ns["huffman_code"](freqs)
        assert set(code) == set(freqs), "code must contain exactly the symbols of freqs"
        words = sorted(code.values())
        assert all(set(w) <= {"0", "1"} and w for w in words), "codewords must be non-empty bit strings"
        for a, b in zip(words, words[1:]):
            assert not b.startswith(a), f"not prefix-free: {a!r} is a prefix of {b!r}"
        tot = sum(freqs.values())
        L = sum(freqs[s] * len(code[s]) for s in freqs) / tot
        ref = _ref_huffman_len(freqs)
        assert abs(L - ref) < 1e-9, f"expected length {L:.4f} is not optimal (Huffman gives {ref:.4f})"
        H = _H([v / tot for v in freqs.values()])
        assert H - 1e-9 <= L < H + 1, "Huffman length must satisfy H <= L < H + 1"


def check_encode_decode_roundtrip(ns):
    freqs = Counter(SAMPLE)
    code = ns["huffman_code"](freqs)
    bits = ns["encode"](SAMPLE, code)
    assert set(bits) <= {"0", "1"}, "encode must return a string of '0'/'1'"
    assert len(bits) == sum(len(code[c]) for c in SAMPLE), "encoded length mismatch"
    assert ns["decode"](bits, code) == SAMPLE, "decode(encode(text)) must give back the text"
    L = ns["avg_code_length"](code, freqs)
    assert abs(L - len(bits) / len(SAMPLE)) < 1e-9, "avg_code_length should equal encoded bits / characters"
