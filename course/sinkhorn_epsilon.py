import numpy as np
import matplotlib.pyplot as plt
import ot
from common import FIG

n = 200
t = np.linspace(0, 1, n)


def bumps(*ms):
    h = sum(w * np.exp(-((t - m) ** 2) / (2 * s**2)) for m, s, w in ms) + 1e-6
    return h / h.sum()


a = bumps((0.2, 0.05, 1), (0.55, 0.08, 0.7))
b = bumps((0.35, 0.04, 0.6), (0.65, 0.05, 1), (0.9, 0.03, 0.4))
C = (t[:, None] - t[None]) ** 2
eps = [1e-4, 1e-3, 1e-2, 1e-1]

fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
for ax, e in zip(axes, eps):
    P, log = ot.sinkhorn(a, b, C, e, method="sinkhorn_log", stopThr=1e-12, numItermax=5000, log=True)
    ax.imshow(P ** 0.5, cmap="Purples", origin="lower")
    ax.set_title(f"eps = {e:g}")
    ax.axis("off")
    axes[4].semilogy(10 * np.arange(len(log["err"])), log["err"], label=f"eps = {e:g}")
axes[4].set(xlabel="iteration", ylabel="marginal error", ylim=(1e-12, 1))
axes[4].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "sinkhorn_epsilon.png", dpi=100)
