"""Hidden checks for dl_08_lora. Uses its own tiny model; GPU if available, otherwise CPU."""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c_attn = nn.Linear(d, 3 * d)
        self.c_proj = nn.Linear(3 * d, d)

    def forward(self, x):
        return x + self.c_proj(torch.tanh(self.c_attn(x)))


class _Tiny(nn.Module):
    def __init__(self, V=20, d=32, n=2):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def check_lora_linear_init_and_shapes(ns):
    torch.manual_seed(0)
    base = nn.Linear(16, 24)
    lora = ns["LoRALinear"](base, r=4, alpha=8)
    assert tuple(lora.A.shape) == (4, 16) and tuple(lora.B.shape) == (24, 4), "A should be (r, in), B should be (out, r)"
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x)), "at init (B = 0) the wrapped layer must equal the base layer"
    assert not base.weight.requires_grad and not base.bias.requires_grad, "base weights must be frozen"
    assert lora.A.requires_grad and lora.B.requires_grad, "A and B must be trainable"
    assert lora.A.abs().sum() > 0, "A should be randomly initialised (only B starts at zero)"


def check_lora_forward_and_merge(ns):
    torch.manual_seed(1)
    base = nn.Linear(8, 6).double()
    lora = ns["LoRALinear"](base, r=2, alpha=6)
    with torch.no_grad():
        lora.B.copy_(torch.randn(6, 2, dtype=torch.float64))
    x = torch.randn(3, 8, dtype=torch.float64)
    want = base(x) + 3.0 * x @ lora.A.T @ lora.B.T
    assert torch.allclose(lora(x), want), "forward must be base(x) + (alpha/r) x A^T B^T"
    W = lora.merged_weight()
    assert torch.allclose(F.linear(x, W, base.bias), want), "merged_weight() must equal W0 + (alpha/r) B A"


def check_apply_lora_trainable_count(ns):
    torch.manual_seed(0)
    model = _Tiny()
    total = sum(p.numel() for p in model.parameters())
    model = ns["apply_lora"](model, target_names=("c_attn",), r=4, alpha=8)
    n = ns["count_trainable"](model)
    want = 2 * 4 * (32 + 96)
    assert n == want, f"only the LoRA A/B of the two c_attn layers should be trainable: expected {want}, got {n}"
    assert sum(isinstance(m, ns["LoRALinear"]) for m in model.modules()) == 2, "both c_attn layers should be wrapped"
    assert sum(p.numel() for p in model.parameters()) == total + want, "wrapping must keep the base weights (frozen) and add A, B"
    assert ns["lora_param_formula"](4096, 4096, 8) == 65536, "lora_param_formula(d_in, d_out, r) = r (d_in + d_out)"


def check_finetune_only_adapters(ns):
    torch.manual_seed(0)
    model = _Tiny().to(DEV)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model = ns["apply_lora"](model, target_names=("c_attn",), r=4, alpha=8).to(DEV)
    g = torch.Generator().manual_seed(0)
    seq = torch.randint(0, 20, (8, 17), generator=g).to(DEV)
    x, y = seq[:, :-1], seq[:, 1:]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    first = None
    for _ in range(150):
        loss = F.cross_entropy(model(x).reshape(-1, 20), y.reshape(-1))
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first - 0.5, f"fine-tuning the adapters should reduce the loss: {first:.3f} -> {loss.item():.3f}"
    sd = model.state_dict()
    for k, v in frozen.items():
        k2 = k if k in sd else k.replace("c_attn.", "c_attn.base.")
        assert torch.equal(sd[k2].cpu(), v.cpu()), f"frozen weight {k} changed during fine-tuning"
