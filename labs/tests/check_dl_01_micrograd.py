"""Hidden checks for dl_01_micrograd. Each check_* gets the notebook globals as `ns`."""
import math
import random

import torch


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def check_forward_arithmetic(ns):
    V = ns["Value"]
    a, b = V(2.0), V(-3.0)
    c = a * b + 10
    assert _close(c.data, 4.0), f"2*(-3)+10 should be 4, got {c.data}"
    d = (a - b) / a
    assert _close(d.data, 2.5), f"(2-(-3))/2 should be 2.5, got {d.data}"
    e = 3 * a ** 2 - 1 / b
    assert _close(e.data, 12 + 1 / 3), f"3*a**2 - 1/b should be 12.333..., got {e.data}"
    assert _close(V(0.5).tanh().data, math.tanh(0.5)), "tanh forward wrong"
    assert V(-1.0).relu().data == 0.0 and V(2.0).relu().data == 2.0, "relu forward wrong"
    assert _close(V(1.0).exp().data, math.e), "exp forward wrong"


def check_backward_graph(ns):
    V = ns["Value"]
    a, b, c, d = V(3.0), V(-2.0), V(5.0), V(4.0)
    f = (a * b + c) * d
    f.backward()
    got = [a.grad, b.grad, c.grad, d.grad]
    want = [-8.0, 12.0, 4.0, -1.0]
    assert all(_close(g, w) for g, w in zip(got, want)), f"grads {got} != {want}"


def check_gradient_accumulation(ns):
    V = ns["Value"]
    a, b = V(-4.0), V(1.5)
    f = (a + b) * (b + b * a)   # b feeds three nodes
    f.backward()
    ta, tb = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True), torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    tf = (ta + tb) * (tb + tb * ta)
    tf.backward()
    assert _close(a.grad, ta.grad.item()) and _close(b.grad, tb.grad.item()), \
        f"got ({a.grad}, {b.grad}), expected ({ta.grad.item()}, {tb.grad.item()}) — do you use += in _backward?"


def check_nonlinearity_grads_vs_torch(ns):
    V = ns["Value"]
    x, y = V(0.7), V(-1.3)
    z = (x * y).tanh() + (x ** 3).relu() + (y / 2).exp() - (x - y) ** 2
    z.backward()
    tx = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(-1.3, dtype=torch.float64, requires_grad=True)
    tz = torch.tanh(tx * ty) + torch.relu(tx ** 3) + torch.exp(ty / 2) - (tx - ty) ** 2
    tz.backward()
    assert _close(z.data, tz.item()), f"forward {z.data} vs torch {tz.item()}"
    assert _close(x.grad, tx.grad.item(), 1e-5) and _close(y.grad, ty.grad.item(), 1e-5), \
        f"grads ({x.grad}, {y.grad}) vs torch ({tx.grad.item()}, {ty.grad.item()})"


def check_mlp_matches_torch(ns):
    random.seed(1234)
    model = ns["MLP"](3, [4, 4, 1])
    params = model.parameters()
    assert len(params) == 4 * 4 + 4 * 5 + 1 * 5, f"MLP(3,[4,4,1]) should have 41 parameters, got {len(params)}"
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0]]
    ys = [1.0, -1.0, -1.0]
    preds = [model(x) for x in xs]
    loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), ns["Value"](0.0))
    loss.backward()
    # rebuild the same network in torch: parameters are ordered layer -> neuron -> (w..., b)
    flat = torch.tensor([p.data for p in params], dtype=torch.float64)
    sizes, off, Ws, bs = [3, 4, 4, 1], 0, [], []
    for i in range(3):
        nin, nout = sizes[i], sizes[i + 1]
        blk = flat[off: off + nout * (nin + 1)].view(nout, nin + 1)
        off += nout * (nin + 1)
        Ws.append(blk[:, :nin].clone().requires_grad_(True))
        bs.append(blk[:, nin].clone().requires_grad_(True))
    X, Y = torch.tensor(xs, dtype=torch.float64), torch.tensor(ys, dtype=torch.float64)
    h = X
    for i in range(3):
        h = h @ Ws[i].T + bs[i]
        if i < 2:
            h = torch.tanh(h)
    tl = ((h.squeeze(1) - Y) ** 2).sum()
    tl.backward()
    assert _close(loss.data, tl.item(), 1e-6), f"loss {loss.data} vs torch {tl.item()} (hidden layers tanh, last layer linear?)"
    tgrads = torch.cat([torch.cat([Ws[i].grad, bs[i].grad[:, None]], 1).flatten() for i in range(3)])
    ours = torch.tensor([p.grad for p in params], dtype=torch.float64)
    assert torch.allclose(ours, tgrads, atol=1e-6), "MLP parameter gradients differ from torch"


def check_training_reduces_loss(ns):
    random.seed(0)
    model = ns["MLP"](2, [8, 1])
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [-1.0, 1.0, 1.0, -1.0]   # XOR
    losses = ns["train"](model, xs, ys, steps=60, lr=0.1)
    assert len(losses) == 60, "train should return one loss per step"
    assert losses[-1] < 0.5 * losses[0], f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
