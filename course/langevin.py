import numpy as np
import matplotlib.pyplot as plt
from common import FIG

rng = np.random.default_rng(0)


def grad_V(x):
    return np.c_[4 * x[:, 0] * (x[:, 0] ** 2 - 1), x[:, 1]]


u = np.linspace(-2, 2, 200)
U, W = np.meshgrid(u, u)
V = (U**2 - 1) ** 2 + 0.5 * W**2
h, times = 0.01, [0, 20, 100, 500]
fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
for row, noise in enumerate([0, 1]):
    x = rng.normal([0, 0], 0.15, (1500, 2)) + [0, 1.2]
    for k in range(times[-1] + 1):
        if k in times:
            ax = axes[row, times.index(k)]
            ax.contour(U, W, np.exp(-V), 6, colors="k", linewidths=0.4)
            ax.scatter(*x.T, s=1, c="#c7372f")
            ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", xticks=[], yticks=[], title=f"{'Langevin' if noise else 'gradient descent'}, t = {k * h:.1f}")
        x = x - h * grad_V(x) + noise * np.sqrt(2 * h) * rng.normal(size=x.shape)
fig.tight_layout()
fig.savefig(FIG / "langevin.png", dpi=100)
