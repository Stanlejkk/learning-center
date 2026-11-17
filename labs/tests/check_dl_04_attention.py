"""Hidden checks for dl_04_attention."""
import torch
import torch.nn.functional as F


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def check_attention_matches_sdpa(ns):
    q, k, v = _rand(2, 3, 5, 8, seed=1), _rand(2, 3, 7, 8, seed=2), _rand(2, 3, 7, 4, seed=3)
    out, w = ns["attention"](q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert tuple(out.shape) == (2, 3, 5, 4), f"output shape {tuple(out.shape)}"
    assert tuple(w.shape) == (2, 3, 5, 7), f"weights shape {tuple(w.shape)}"
    assert torch.allclose(out, ref, atol=1e-8), "output differs from F.scaled_dot_product_attention (did you scale by sqrt(d_k)?)"
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5, dtype=torch.float64)), "attention weights must sum to 1 over keys"


def check_causal_mask(ns):
    m = ns["causal_mask"](4)
    assert m.dtype == torch.bool, "mask should be a bool tensor (True = may attend)"
    assert m.tolist() == [[True, False, False, False], [True, True, False, False],
                          [True, True, True, False], [True, True, True, True]], "causal mask must be lower-triangular"
    q, k, v = _rand(1, 6, 8, seed=4), _rand(1, 6, 8, seed=5), _rand(1, 6, 8, seed=6)
    out, w = ns["attention"](q, k, v, ns["causal_mask"](6))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-8), "masked attention differs from sdpa(is_causal=True) — mask BEFORE softmax with -inf"
    assert torch.all(w[0].triu(1) == 0), "weights above the diagonal must be exactly 0"


def check_multihead_matches_reference(ns):
    torch.manual_seed(0)
    mha = ns["MultiHeadAttention"](16, 4, causal=True).double()
    x = _rand(2, 5, 16, seed=7)
    y = mha(x)
    assert tuple(y.shape) == (2, 5, 16), f"output shape {tuple(y.shape)}"
    n = sum(p.numel() for p in mha.parameters())
    assert n == 4 * 16 * 16 + 4 * 16, f"MHA(16, 4) should have 4d^2 + 4d = 1088 params, got {n}"
    q, k, v = mha.qkv(x).split(16, dim=-1)
    q, k, v = (t.view(2, 5, 4, 4).transpose(1, 2) for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 5, 16)
    ref = mha.proj(ref)
    assert torch.allclose(y, ref, atol=1e-8), "MHA output differs from the reference built from your qkv/proj weights"


def check_causality(ns):
    torch.manual_seed(0)
    mha = ns["MultiHeadAttention"](8, 2, causal=True).double()
    x = _rand(1, 6, 8, seed=8)
    x2 = x.clone()
    x2[:, 4:] += 5.0   # change the future
    y, y2 = mha(x), mha(x2)
    assert torch.allclose(y[:, :4], y2[:, :4], atol=1e-10), "outputs at t<4 changed when only tokens t>=4 changed: information leaks from the future"
    assert not torch.allclose(y[:, 4:], y2[:, 4:]), "outputs at the changed positions should change"
