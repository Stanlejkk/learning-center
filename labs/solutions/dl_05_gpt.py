"""Reference solution — dl_05_gpt: a small character-level GPT trained on TinyShakespeare.

Architecture follows GPT-2 (pre-LN blocks, GELU MLP, learned positions, tied embeddings) as in Karpathy's
nanoGPT / build-nanogpt (https://github.com/karpathy/nanoGPT); written independently for this course.
No training happens at import time — see `if __name__ == "__main__":` at the bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def load_text(path="data/tinyshakespeare.txt"):
    return Path(path).read_text(encoding="utf-8")


def build_char_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, {i: c for c, i in stoi.items()}


def encode(s, stoi):
    return [stoi[c] for c in s]


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def get_batch(data, block_size, batch_size, generator=None, device="cpu"):
    """data: 1-D LongTensor of token ids. Returns x, y of shape (batch_size, block_size), y = x shifted by one."""
    ix = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------------------------------------------ model
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2: scale residual projections by 1/sqrt(2 * n_layer)
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_params(model):
    """Unique trainable parameters (tied weights counted once)."""
    return sum(p.numel() for p in model.parameters())


def gpt_param_formula(cfg):
    V, T, d, L = cfg.vocab_size, cfg.block_size, cfg.n_embd, cfg.n_layer
    return V * d + T * d + L * (12 * d * d + 13 * d) + 2 * d


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size=32, iters=20, device=DEVICE, seed=0):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    losses = [model(*get_batch(data, block_size, batch_size, g, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def train_gpt(model, data, block_size, batch_size=32, max_iters=200, lr=3e-4, device=DEVICE, seed=0, log_every=0):
    """AdamW training loop on random batches from `data`. Returns the list of training losses."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    losses = []
    model.train()
    for it in range(max_iters):
        x, y = get_batch(data, block_size, batch_size, g, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and it % log_every == 0:
            print(f"iter {it}: loss {loss.item():.4f}")
    return losses


if __name__ == "__main__":
    text = load_text()
    stoi, itos = build_char_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    cfg = GPTConfig(vocab_size=len(stoi), block_size=128, n_layer=4, n_head=4, n_embd=256)
    model = GPT(cfg).to(DEVICE)
    print("params:", count_params(model), "formula:", gpt_param_formula(cfg))
    train_gpt(model, train_data, cfg.block_size, batch_size=64, max_iters=2000, lr=1e-3, log_every=200)
    print("val loss:", estimate_loss(model, val_data, cfg.block_size))
    print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=DEVICE), 500)[0].tolist(), itos))
