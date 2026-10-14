import numpy as np
import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from common import conformers, principal_axes, surface_points, frame, save_gif, FIG

torch.manual_seed(0)
mol, X = conformers("c1ccc2c(c1)C3c4ccccc4C2c5ccccc35", hs=False)
X = principal_axes(X[0])
y = torch.tensor(surface_points(mol, X, 1200, scale=0.7), dtype=torch.float32)
x = (2 * torch.randn(1200, 3)).requires_grad_()
loss = SamplesLoss("sinkhorn", p=2, blur=0.3, scaling=0.8)
color = plt.cm.viridis(x[:, 2].detach().argsort().argsort().numpy() / len(x))
lim = y.abs().max().item()


def draw(ax, pts, azim):
    ax.scatter(*pts.T, c=color, s=1.5, depthshade=False)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), box_aspect=(1, 1, 1))
    ax.view_init(15, azim)
    ax.axis("off")


snaps = []
for step in range(31):
    snaps.append(x.detach().numpy().copy())
    (g,) = torch.autograd.grad(loss(x, y), x)
    x.data -= 0.15 * len(x) * g

fig = plt.figure(figsize=(15, 3))
for k, s in enumerate([0, 2, 4, 8, 15, 30]):
    ax = fig.add_subplot(1, 6, k + 1, projection="3d")
    draw(ax, snaps[s], 40)
    ax.set_title(f"step {s}")
fig.tight_layout()
fig.savefig(FIG / "sinkhorn_flow_triptycene.png", dpi=100)

frames = []
for k, s in enumerate(snaps):
    fig = plt.figure(figsize=(3.4, 3.4))
    draw(fig.add_subplot(projection="3d"), s, 40 + 3 * k)
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "sinkhorn_flow_triptycene.gif", fps=12)
