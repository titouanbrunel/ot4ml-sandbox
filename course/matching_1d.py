import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import ot
from common import FIG

rng = np.random.default_rng(3)
n = 14
x, y = np.sort(rng.random(n)), np.sort(rng.random(n))
fig, axes = plt.subplots(2, 1, figsize=(10, 3.6))
for ax, p in zip(axes, [2, 0.5]):
    P = ot.emd(ot.unif(n), ot.unif(n), np.abs(x[:, None] - y[None]) ** p)
    for i, j in zip(*np.nonzero(P)):
        ax.add_patch(Arc(((x[i] + y[j]) / 2, 0), abs(x[i] - y[j]), abs(x[i] - y[j]), theta2=180, color="#7c5bbd"))
    ax.scatter(x, 0 * x, c="#c7372f", zorder=3)
    ax.scatter(y, 0 * y, c="#226db4", zorder=3)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.03, 0.5), aspect="equal", title=f"$|x - y|^{{{p}}}$")
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "matching_1d.png", dpi=100)
