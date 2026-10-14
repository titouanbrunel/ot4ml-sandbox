import numpy as np
import matplotlib.pyplot as plt
import ot
from common import FIG

rng = np.random.default_rng(0)
x = np.r_[rng.normal([0, 0], 0.3, (60, 2)), rng.normal([0, 3], 0.15, (15, 2))]
y = rng.normal([2.5, 0.3], [0.2, 0.5], (75, 2))
a, b = ot.unif(len(x)), ot.unif(len(y))
M = ot.dist(x, y)
M /= M.max()
plans = {
    "balanced": ot.sinkhorn(a, b, M, 0.01),
    "unbalanced, tau = 1": ot.unbalanced.sinkhorn_unbalanced(a, b, M, 0.01, 1.0),
    "unbalanced, tau = 0.1": ot.unbalanced.sinkhorn_unbalanced(a, b, M, 0.01, 0.1),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, P) in zip(axes, plans.items()):
    for i, j in zip(*np.nonzero(P > P.max() * 0.1)):
        ax.plot(*np.c_[x[i], y[j]], c="#7c5bbd", alpha=P[i, j] / P.max(), lw=0.8)
    ax.scatter(*x.T, s=3000 * P.sum(1), c="#c7372f", zorder=3)
    ax.scatter(*y.T, s=3000 * P.sum(0), c="#226db4", zorder=3)
    ax.set(aspect="equal", title=f"{name}, mass {P.sum():.2f}")
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "unbalanced.png", dpi=100)
