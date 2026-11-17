"""Hidden checks for dl_05_gpt. GPU if available, otherwise CPU (all checks are small)."""
import math

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _tiny_cfg(ns, **kw):
    base = dict(vocab_size=65, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    base.update(kw)
    return ns["GPTConfig"](**base)


def check_tokenizer_roundtrip(ns):
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    stoi, itos = ns["build_char_vocab"](text)
    assert len(stoi) == len(set(text)), "one id per distinct character"
    assert list(stoi) == sorted(stoi), "ids should follow sorted character order"
    ids = ns["encode"](text, stoi)
    assert ns["decode"](ids, itos) == text, "decode(encode(s)) must equal s"


def check_get_batch(ns):
    data = torch.arange(1000)
    g = torch.Generator().manual_seed(0)
    x, y = ns["get_batch"](data, 8, 4, generator=g)
    assert tuple(x.shape) == (4, 8) and tuple(y.shape) == (4, 8), "x, y should be (batch, block)"
    assert torch.equal(y.cpu(), x.cpu() + 1), "y must be x shifted one token to the right"
    assert int(x.max()) + 1 < 1000, "batches must stay inside the data"


def check_param_count_formula(ns):
    for kw in (dict(), dict(vocab_size=50257, block_size=1024, n_layer=2, n_head=12, n_embd=768)):
        cfg = _tiny_cfg(ns, **kw)
        model = ns["GPT"](cfg)
        got, formula = ns["count_params"](model), ns["gpt_param_formula"](cfg)
        want = cfg.vocab_size * cfg.n_embd + cfg.block_size * cfg.n_embd + cfg.n_layer * (12 * cfg.n_embd ** 2 + 13 * cfg.n_embd) + 2 * cfg.n_embd
        assert formula == want, f"gpt_param_formula gives {formula}, expected Vd + Td + L(12d^2+13d) + 2d = {want}"
        assert got == want, f"model has {got} params, expected {want} (biases everywhere, 2 LayerNorms/block, final LN, TIED lm_head)"


def check_forward_and_initial_loss(ns):
    torch.manual_seed(0)
    cfg = _tiny_cfg(ns)
    model = ns["GPT"](cfg).to(DEV)
    x = torch.randint(0, 65, (8, 32), device=DEV)
    t = torch.randint(0, 65, (8, 32), device=DEV)   # independent random targets
    logits, loss = model(x, t)
    assert tuple(logits.shape) == (8, 32, 65), f"logits shape {tuple(logits.shape)}"
    assert abs(loss.item() - math.log(65)) < 0.3, f"initial loss {loss.item():.3f} should be close to ln(65) = {math.log(65):.3f} (small init)"
    logits2, loss2 = model(x)
    assert loss2 is None, "loss should be None without targets"


def check_causality(ns):
    torch.manual_seed(0)
    model = ns["GPT"](_tiny_cfg(ns)).to(DEV).eval()
    x = torch.randint(0, 65, (1, 16), device=DEV)
    x2 = x.clone()
    x2[0, 10:] = (x2[0, 10:] + 7) % 65
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x2)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-4), "logits at positions < 10 changed when only later tokens changed"


def check_overfit_one_batch(ns):
    torch.manual_seed(0)
    cfg = _tiny_cfg(ns)
    model = ns["GPT"](cfg).to(DEV)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 65, (40,), generator=g)
    x, y = data[:32][None].to(DEV), data[1:33][None].to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5, f"a working GPT should memorise one 32-token sequence; loss is still {loss.item():.3f}"


def check_generate(ns):
    torch.manual_seed(0)
    cfg = _tiny_cfg(ns)
    model = ns["GPT"](cfg).to(DEV)
    start = torch.zeros((2, 1), dtype=torch.long, device=DEV)
    out = model.generate(start, 50)   # longer than block_size: must crop the context
    assert tuple(out.shape) == (2, 51), f"generate should append max_new_tokens tokens, got {tuple(out.shape)}"
    assert int(out.min()) >= 0 and int(out.max()) < 65, "generated ids out of range"
