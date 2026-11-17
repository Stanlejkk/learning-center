"""Reference solution — dl_04_attention: scaled dot-product attention, causal masking, multi-head attention.

Follows Vaswani et al. 2017 (arXiv:1706.03762); the module layout mirrors nanoGPT's CausalSelfAttention.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    """q: (..., Tq, d), k: (..., Tk, d), v: (..., Tk, dv). mask: bool, broadcastable to (..., Tq, Tk), True = may attend.
    Returns (output (..., Tq, dv), weights (..., Tq, Tk))."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    w = torch.softmax(scores, dim=-1)
    return w @ v, w


def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class MultiHeadAttention(nn.Module):
    """Self-attention with a fused qkv projection.
    Convention: q, k, v = self.qkv(x).split(d_model, dim=-1); heads via .view(B, T, n_heads, d_head).transpose(1, 2)."""

    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.causal = d_model, n_heads, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        h, dh = self.n_heads, C // self.n_heads
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v))  # (B, h, T, dh)
        mask = causal_mask(T, x.device) if self.causal else None
        y, _ = attention(q, k, v, mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
