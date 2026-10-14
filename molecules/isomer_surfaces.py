import numpy as np
import matplotlib.pyplot as plt
import ot
from common import conformers, principal_axes, surface_points, frame, save_gif, FIG

n = 1500
clouds, atoms = [], []
for smi in ["C12C3C4C1C5C2C3C45", "C1=CC=CC=CC=C1"]:
    mol, X = conformers(smi, hs=False)
    X = principal_axes(X[0])
    clouds.append(surface_points(mol, X, n, scale=0.32))
    atoms.append(X)
x, y = clouds
P = ot.emd(ot.unif(n), ot.unif(n), ot.dist(x, y))
Tx = n * P @ y
color = plt.cm.tab10(ot.dist(x, atoms[0]).argmin(1))
lim = np.abs(np.r_[x, y]).max()


def draw(ax, pts, azim):
    ax.scatter(*pts.T, c=color, s=3, depthshade=True)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), box_aspect=(1, 1, 1))
    ax.view_init(25, azim)
    ax.axis("off")


fig = plt.figure(figsize=(15, 3))
for k, t in enumerate(np.linspace(0, 1, 6)):
    ax = fig.add_subplot(1, 6, k + 1, projection="3d")
    draw(ax, (1 - t) * x + t * Tx, 30)
    ax.set_title(f"t = {t:.1f}")
fig.tight_layout()
fig.savefig(FIG / "cubane_to_cot.png", dpi=100)

frames = []
for k, t in enumerate(np.linspace(0, 1, 40)):
    s = t * t * (3 - 2 * t)
    fig = plt.figure(figsize=(3.4, 3.4))
    draw(fig.add_subplot(projection="3d"), (1 - s) * x + s * Tx, 30 + 2 * k)
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "cubane_to_cot.gif", fps=15)
