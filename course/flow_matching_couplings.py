import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import ot
from common import glyph_points, FIG

torch.manual_seed(0)
data = torch.tensor(4 * glyph_points("♛", 20000, seed=2) - 2, dtype=torch.float32)


def couple(x0, x1, mode):
    if mode == "OT":
        P = ot.emd(ot.unif(len(x0)), ot.unif(len(x1)), ot.dist(x0.numpy(), x1.numpy()))
        x1 = x1[P.argmax(1)]
    return x0, x1


def train(mode, steps=6000, bs=512):
    net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, x1 = couple(torch.randn(bs, 2), data[torch.randint(len(data), (bs,))], mode)
        t = torch.rand(bs, 1)
        loss = ((net(torch.cat([(1 - t) * x0 + t * x1, t], 1)) - (x1 - x0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net


@torch.no_grad()
def sample(net, x, nfe):
    path = [x]
    for k in range(nfe):
        t = torch.full((len(x), 1), k / nfe)
        x = x + net(torch.cat([x, t], 1)) / nfe
        path.append(x)
    return torch.stack(path)


x0 = torch.randn(2000, 2)
fig, axes = plt.subplots(2, 5, figsize=(17, 7))
for row, mode in enumerate(["independent", "OT"]):
    net = train(mode)
    a, b = couple(torch.randn(256, 2), data[torch.randint(len(data), (256,))], mode)
    ax = axes[row, 0]
    ax.plot(*torch.stack([a, b], 1)[:80].permute(2, 1, 0), c="#7c5bbd", lw=0.5)
    ax.set_title(f"{mode} coupling, training pairs")
    path = sample(net, x0[:300], 64)
    axes[row, 1].plot(*path.permute(2, 0, 1), c="#7c5bbd", lw=0.4, alpha=0.6)
    axes[row, 1].set_title("trajectories, 64 steps")
    for ax, nfe in zip(axes[row, 2:], [1, 4, 64]):
        ax.scatter(*sample(net, x0, nfe)[-1].T, s=1, c="#226db4")
        ax.set_title(f"samples, {nfe} Euler step{'s' * (nfe > 1)}")
for ax in axes.ravel():
    ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", xticks=[], yticks=[])
fig.tight_layout()
fig.savefig(FIG / "flow_matching_couplings.png", dpi=90)
