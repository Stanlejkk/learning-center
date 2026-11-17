"""Hidden checks for dl_07_sampling."""
import torch
import torch.nn.functional as F

NEG = float("-inf")


def check_temperature(ns):
    z = torch.log(torch.tensor([[4.0, 2.0, 1.0, 1.0]]))
    p = F.softmax(ns["apply_temperature"](z, 0.5), -1)
    assert torch.allclose(p, torch.tensor([[16.0, 4.0, 1.0, 1.0]]) / 22, atol=1e-6), "T=1/2 should square the unnormalised probabilities"
    p2 = F.softmax(ns["apply_temperature"](z, 1e6), -1)
    assert torch.allclose(p2, torch.full((1, 4), 0.25), atol=1e-4), "very high T should give a uniform distribution"


def check_top_k(ns):
    z = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0], [0.0, -1.0, 2.0, 1.0, -2.0]])
    out = ns["top_k_filter"](z, 2)
    keep = torch.isfinite(out)
    assert keep.tolist() == [[False, True, False, True, False], [False, False, True, True, False]], "top-k must keep the k largest per row"
    assert torch.equal(out[keep], z[keep]), "kept logits must be unchanged"
    assert torch.isfinite(ns["top_k_filter"](z, 10)).all(), "k >= V keeps everything"


def check_top_p(ns):
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
    z = torch.log(probs)
    for p, n_keep in ((0.45, 1), (0.6, 2), (0.65, 2), (0.75, 3), (0.9, 4), (0.99, 5)):
        out = ns["top_p_filter"](z, p)
        k = int(torch.isfinite(out).sum())
        assert k == n_keep, f"top-p with p={p} on {probs.tolist()[0]} should keep {n_keep} tokens (smallest set with mass >= p), kept {k}"
    shuffled = z[:, [3, 0, 4, 2, 1]]
    out = ns["top_p_filter"](shuffled, 0.6)
    assert torch.isfinite(out).tolist() == [[False, True, False, False, True]], "top-p must work on unsorted logits and keep original positions"
    assert torch.isfinite(ns["top_p_filter"](z, 0.01)).sum() == 1, "always keep at least the top token"


def check_sample_next(ns):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 10, generator=g)
    s = ns["sample_next"](z, top_k=1, generator=g)
    assert s.shape == (64,) and torch.equal(s, z.argmax(-1)), "top_k=1 must equal greedy argmax and return shape (B,)"
    z = torch.log(torch.tensor([[0.6, 0.3, 0.1]])).repeat(20000, 1)
    s = ns["sample_next"](z, top_p=0.85, generator=torch.Generator().manual_seed(1))
    frac = torch.bincount(s, minlength=3).double() / 20000
    assert frac[2] == 0 and abs(frac[0].item() - 2 / 3) < 0.02, f"top_p=0.85 should renormalise to (2/3, 1/3, 0), got {frac.tolist()}"


def check_kv_cache_equivalence(ns):
    torch.manual_seed(0)
    attn = ns["CachedCausalSelfAttention"](32, 4).double().eval()
    x = torch.randn(2, 12, 32, dtype=torch.float64)
    with torch.no_grad():
        full, (k, v) = attn(x)
        assert k.shape == (2, 4, 12, 8), f"cache keys should be (B, h, T, d_head), got {tuple(k.shape)}"
        cache, outs = None, []
        for t in range(12):
            y, cache = attn(x[:, t: t + 1], cache)
            outs.append(y)
        inc = torch.cat(outs, 1)
        assert torch.allclose(full, inc, atol=1e-10), "token-by-token decoding with the KV cache must equal the full causal forward"
        pre, cache = attn(x[:, :7])            # prefill a chunk, then decode a chunk of 5
        rest, _ = attn(x[:, 7:], cache)
        assert torch.allclose(full, torch.cat([pre, rest], 1), atol=1e-10), \
            "chunked prefill + multi-token step must also match (mask new queries against past+new keys)"


def check_kv_cache_bytes(ns):
    f = ns["kv_cache_bytes"]
    assert f(32, 32, 128, 4096, 1, 2) == 2 * 32 * 32 * 128 * 4096 * 2, "bytes = 2 (K,V) * layers * kv_heads * d_head * seq * batch * bytes"
    assert f(32, 8, 128, 8192, 4, 2) == 4 * 2 ** 30, "Llama-3-8B, 4 x 8192 tokens in fp16 should be exactly 4 GiB"
