"""Hidden checks for dl_02_bigram."""
import math

import torch

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia"]


def check_vocab(ns):
    stoi, itos = ns["build_vocab"](WORDS)
    assert stoi["."] == 0, "the boundary token '.' must have index 0"
    letters = sorted(set("".join(WORDS)))
    assert len(stoi) == len(letters) + 1, f"vocab should be {len(letters) + 1} symbols"
    assert [itos[i] for i in range(1, len(stoi))] == letters, "letters should get indices 1.. in sorted order"


def check_counts(ns):
    stoi, _ = ns["build_vocab"](WORDS)
    N = ns["bigram_counts"](WORDS, stoi)
    assert tuple(N.shape) == (len(stoi), len(stoi)), "counts must be a V x V tensor"
    assert int(N.sum()) == sum(len(w) + 1 for w in WORDS), "a word of length n contributes n+1 bigrams (with both boundaries)"
    assert int(N[stoi["."], stoi["a"]]) == 2, "'.a' (word starts with a) should occur twice"
    assert int(N[stoi["a"], stoi["."]]) == 7, "'a.' (word ends with a) should occur seven times"
    assert int(N[stoi["m"], stoi["m"]]) == 1, "'mm' should occur once"


def check_probs_and_nll(ns):
    stoi, _ = ns["build_vocab"](WORDS)
    N = ns["bigram_counts"](WORDS, stoi)
    P = ns["counts_to_probs"](N, alpha=1.0)
    assert torch.allclose(P.double().sum(1), torch.ones(len(stoi), dtype=torch.float64), atol=1e-6), "rows must sum to 1"
    V = len(stoi)
    want = (N[stoi["e"], stoi["m"]].item() + 1) / (N[stoi["e"]].sum().item() + V)
    assert abs(P[stoi["e"], stoi["m"]].item() - want) < 1e-6, "add-alpha smoothing: (c + alpha) / (row total + alpha * V)"
    val = ns["nll"](["emma"], P, stoi)
    ref = -sum(math.log(P[stoi[a], stoi[b]].item()) for a, b in zip(".emma", "emma.")) / 5
    assert abs(val - ref) < 1e-6, f"nll should be the AVERAGE negative log-likelihood per bigram ({ref:.4f}), got {val}"
    assert abs(ns["perplexity"](2.0) - math.exp(2.0)) < 1e-9, "perplexity = exp(average NLL in nats)"


def check_dataset(ns):
    stoi, _ = ns["build_vocab"](WORDS)
    xs, ys = ns["make_bigram_dataset"](["emma"], stoi)
    assert xs.tolist() == [stoi[c] for c in ".emma"], f"xs for 'emma' should be indices of '.emma', got {xs.tolist()}"
    assert ys.tolist() == [stoi[c] for c in "emma."], f"ys for 'emma' should be indices of 'emma.', got {ys.tolist()}"


def check_neural_matches_counts(ns):
    stoi, _ = ns["build_vocab"](WORDS)
    xs, ys = ns["make_bigram_dataset"](WORDS, stoi)
    V = len(stoi)
    W, losses = ns["train_neural_bigram"](xs, ys, V, steps=300, lr=50.0, reg=0.0, seed=0)
    assert tuple(W.shape) == (V, V), "W must be V x V"
    N = ns["bigram_counts"](WORDS, stoi)
    P = N.double() / N.double().sum(1, keepdim=True).clamp(min=1)
    mle = -torch.log(P[xs, ys]).mean().item()
    assert losses[-1] < losses[0], "training loss should decrease"
    assert losses[-1] < mle + 0.1, f"neural bigram should approach the count-based MLE NLL {mle:.3f}, got {losses[-1]:.3f}"


def check_sampling(ns):
    stoi, itos = ns["build_vocab"](WORDS)
    P = ns["counts_to_probs"](ns["bigram_counts"](WORDS, stoi), alpha=0.0)
    g = torch.Generator().manual_seed(42)
    for _ in range(20):
        w = ns["sample_word"](P, itos, generator=g)
        assert isinstance(w, str) and "." not in w, "sample_word returns a string without the boundary token"
        assert all(c in stoi for c in w), "sampled characters must come from the vocabulary"
