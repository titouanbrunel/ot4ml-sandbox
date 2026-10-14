import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from common import FIG

n, m = 160, 14
rng = np.random.default_rng(1)
u = (np.arange(n) + 0.5) / n
G = np.stack(np.meshgrid(u, u), -1).reshape(-1, 2)
rho = sum(w * np.exp(-((G - c) ** 2).sum(1) / (2 * s**2)) for c, s, w in [((0.3, 0.3), 0.12, 1), ((0.7, 0.65), 0.15, 0.8), ((0.25, 0.8), 0.07, 0.6)])
rho /= rho.sum()
Y = rng.random((m, 2))
b = np.ones(m) / m


def cells(Y, g):
    return (((G[:, None] - Y[None]) ** 2).sum(-1) - g).argmin(1)


def neg_dual(g):
    D = ((G[:, None] - Y[None]) ** 2).sum(-1) - g
    lab = D.argmin(1)
    mass = np.bincount(lab, rho, m)
    return -(rho @ D[np.arange(len(G)), lab] + b @ g), -(b - mass)


g = minimize(neg_dual, np.zeros(m), jac=True, method="L-BFGS-B").x


def show(ax, lab, Y, title):
    ax.imshow(lab.reshape(n, n), cmap="tab20", origin="lower", extent=(0, 1, 0, 1), alpha=0.55)
    ax.contour(u, u, rho.reshape(n, n), 6, colors="k", linewidths=0.4)
    ax.scatter(*Y.T, c="k", s=15)
    ax.set_title(title)
    ax.axis("off")


fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
lab = cells(Y, 0)
show(axes[0], lab, Y, f"Voronoi, masses {np.bincount(lab, rho, m).min():.3f} to {np.bincount(lab, rho, m).max():.3f}")
lab = cells(Y, g)
show(axes[1], lab, Y, f"Laguerre, masses {np.bincount(lab, rho, m).min():.3f} to {np.bincount(lab, rho, m).max():.3f}")
traj = [Y]
for _ in range(40):
    lab = cells(traj[-1], 0)
    mass = np.bincount(lab, rho, m)
    traj.append(np.stack([np.bincount(lab, rho * G[:, k], m) for k in range(2)], 1) / mass[:, None])
traj = np.array(traj)
show(axes[2], cells(traj[-1], 0), traj[-1], "Lloyd, 40 iterations")
for j in range(m):
    axes[2].plot(*traj[:, j].T, c="k", lw=0.8)
fig.tight_layout()
fig.savefig(FIG / "semidiscrete_lloyd.png", dpi=100)
