import numpy as np
import matplotlib.pyplot as plt
import ot
from common import glyph_points, frame, save_gif, FIG

n = 2000
x = glyph_points("♞", n, seed=0)
y = glyph_points("♛", n, seed=1)
P = ot.emd(ot.unif(n), ot.unif(n), ot.dist(x, y))
Tx = n * P @ y
color = plt.cm.Spectral(np.argsort(np.argsort(x[:, 0] + 0.6 * x[:, 1])) / n)


def scatter(ax, pts, c, s=2):
    ax.scatter(*pts.T, c=c, s=s, lw=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])


ts = np.linspace(0, 1, 6)
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
for k, t in enumerate(ts):
    scatter(axes[0, k], (1 - t) * x + t * Tx, color)
    keep = rng.random(n) < t
    scatter(axes[1, k], np.where(keep[:, None], y, x), np.where(keep, "#226db4", "#c7372f"))
    axes[0, k].set_title(f"t = {t:.1f}")
axes[0, 0].set_ylabel("displacement")
axes[1, 0].set_ylabel("mixture")
fig.tight_layout()
fig.savefig(FIG / "mccann_chess.png", dpi=110)

frames = []
for t in np.linspace(0, 1, 40):
    s = t * t * (3 - 2 * t)
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    scatter(ax, (1 - s) * x + s * Tx, color, s=3)
    ax.axis("off")
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "mccann_chess.gif", fps=20)
