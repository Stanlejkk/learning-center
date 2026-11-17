"""Hidden checks for dl_03_mlp_lm."""
import math

import torch

WORDS = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper", "evelyn", "abigail"]


def check_build_dataset(ns):
    stoi, _ = ns["build_vocab"](WORDS)
    X, Y = ns["build_dataset"](["emma"], stoi, block_size=3)
    assert tuple(X.shape) == (5, 3) and tuple(Y.shape) == (5,), f"'emma' with block 3 -> X (5,3), Y (5,), got {tuple(X.shape)}"
    e, m, a = stoi["e"], stoi["m"], stoi["a"]
    assert X.tolist() == [[0, 0, 0], [0, 0, e], [0, e, m], [e, m, m], [m, m, a]], f"contexts wrong: {X.tolist()}"
    assert Y.tolist() == [e, m, m, a, 0], f"targets wrong: {Y.tolist()}"


def check_split(ns):
    words = [f"w{i}" for i in range(1000)]
    tr, va, te = ns["split_words"](words, seed=1)
    assert (len(tr), len(va), len(te)) == (800, 100, 100), "split should be 80/10/10"
    assert set(tr) | set(va) | set(te) == set(words) and not (set(tr) & set(va)), "splits must partition the words"
    assert tr != words[:800], "shuffle before splitting"


def check_model_shape_and_params(ns):
    torch.manual_seed(0)
    V, block, e, h = 27, 3, 10, 64
    model = ns["MLPLM"](V, block, n_embd=e, n_hidden=h)
    out = model(torch.randint(0, V, (5, block)))
    assert tuple(out.shape) == (5, V), f"logits should be (B, V), got {tuple(out.shape)}"
    n = sum(p.numel() for p in model.parameters())
    want = V * e + block * e * h + h + h * V + V
    assert n == want, f"parameter count {n} != V*e + block*e*h + h + h*V + V = {want}"


def check_training_beats_uniform(ns):
    torch.manual_seed(0)
    stoi, _ = ns["build_vocab"](WORDS)
    X, Y = ns["build_dataset"](WORDS, stoi, block_size=3)
    model = ns["MLPLM"](len(stoi), 3, n_embd=8, n_hidden=64)
    before = ns["evaluate"](model, X, Y)
    losses = ns["train"](model, X, Y, steps=400, lr=0.1, batch_size=32, seed=0)
    after = ns["evaluate"](model, X, Y)
    assert len(losses) == 400, "train returns one loss per step"
    assert after < math.log(len(stoi)) - 0.5 and after < before - 0.5, \
        f"after 400 steps the loss should be well below uniform ln V = {math.log(len(stoi)):.2f}: {before:.2f} -> {after:.2f}"


def check_lr_range_test(ns):
    torch.manual_seed(0)
    stoi, _ = ns["build_vocab"](WORDS)
    X, Y = ns["build_dataset"](WORDS, stoi, block_size=3)
    model = ns["MLPLM"](len(stoi), 3, n_embd=8, n_hidden=64)
    lrs = (10 ** torch.linspace(-3, 1, 50)).tolist()
    losses = ns["lr_range_test"](model, X, Y, lrs)
    assert len(losses) == 50, "one loss per learning rate"
    assert all(isinstance(l, float) for l in losses), "losses should be python floats (use .item())"
    assert min(losses[:40]) < losses[0], "with growing lr the loss should first go down"
