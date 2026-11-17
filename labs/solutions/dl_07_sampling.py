"""Reference solution — dl_07_sampling: temperature / top-k / top-p sampling and KV-cached incremental decoding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ sampling
def apply_temperature(logits, temperature):
    assert temperature > 0, "use greedy argmax for T = 0"
    return logits / temperature


def top_k_filter(logits, k):
    """Keep the k largest logits per row; set the rest to -inf."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus filter: keep the smallest set of most-likely tokens whose total probability is >= p (always >= 1 token)."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = (cum - probs) >= p          # mass BEFORE this token already reaches p -> not needed
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    """logits: (B, V). Returns sampled ids of shape (B,). Order: temperature -> top-k -> top-p -> softmax -> sample."""
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# ------------------------------------------------------------------ KV cache
class CachedCausalSelfAttention(nn.Module):
    """Causal self-attention that can consume new tokens incrementally.

    forward(x, kv_cache=None) with x: (B, T_new, d) and kv_cache = (k, v) each (B, h, T_past, d_head) or None.
    Returns (y (B, T_new, d), (k_all, v_all)) where k_all/v_all include the new positions.
    New token i sits at absolute position T_past + i and may attend to positions <= T_past + i.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        dh = C // self.h
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.h, dh).transpose(1, 2) for t in (q, k, v))
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        Tk = k.shape[2]
        past = Tk - T
        mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=past)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch=1, bytes_per_el=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * bytes_per_el


@torch.no_grad()
def incremental_outputs(attn, x):
    """Run `attn` one token at a time with the cache; return the stacked outputs (B, T, d)."""
    cache, outs = None, []
    for t in range(x.shape[1]):
        y, cache = attn(x[:, t: t + 1], cache)
        outs.append(y)
    return torch.cat(outs, dim=1)
