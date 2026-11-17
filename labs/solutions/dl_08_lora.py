"""Reference solution — dl_08_lora: LoRA (Hu et al. 2021, arXiv:2106.09685) from scratch."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear: y = base(x) + (alpha / r) * x A^T B^T, with A (r x in) random, B (out x r) zero."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)

    def merged_weight(self):
        return self.base.weight + self.scaling * (self.B @ self.A)


def apply_lora(model, target_names=("c_attn",), r=8, alpha=16.0):
    """Freeze every parameter of `model`, then replace each nn.Linear whose attribute name is in `target_names`
    by a LoRALinear wrapper (its A, B are the only trainable parameters). Returns the model."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in target_names:
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_param_formula(d_in, d_out, r):
    return r * (d_in + d_out)
