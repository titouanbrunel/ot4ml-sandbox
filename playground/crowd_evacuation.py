import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skimage.graph import MCP_Geometric
from common import laguerre, frame, save_gif, FIG

h = 1 / 150
x, y = np.meshgrid(np.arange(0, 1.4, h) + h / 2, np.arange(0, 1, h) + h / 2)
wall = (abs(x - 1) < 0.012) & (abs(y - 0.5) > 0.07) | ((x - 0.88) ** 2 + (y - 0.5) ** 2 < 0.04**2) | (x < 0.012) | (abs(y - 0.5) > 0.488)
dist = MCP_Geometric(np.where(wall, np.inf, 1.0)).find_costs([(i, x.shape[1] - 1) for i in range(x.shape[0])])[0] * h
gy, gx = np.gradient(np.where(wall, dist[~wall].max(), dist), h)
gx, gy = gx / np.hypot(gx, gy).clip(1e-9), gy / np.hypot(gx, gy).clip(1e-9)
G = np.c_[x[~wall], y[~wall]]

N, area, tau = 300, 0.26 / 300, 0.02
rng = np.random.default_rng(0)
Y = np.c_[0.05 + 0.6 * rng.random(N), 0.08 + 0.84 * rng.random(N)]
ids = np.arange(N)
color = plt.cm.Spectral(Y[:, 1])
g = np.full(N, area / np.pi)


def project(Y, g):
    def f(s):
        lab, v = laguerre(G, Y, s * area)
        inside = v < 0
        return -(h * h * v[inside].sum() + area * s.sum() * area) / area**2, h * h * np.bincount(lab[inside], minlength=len(Y)) / area - 1

    g = minimize(f, g / area, jac=True, method="L-BFGS-B", options={"maxiter": 200, "ftol": 0, "gtol": 0.05}).x * area
    lab, v = laguerre(G, Y, g)
    lab = np.where(v < 0, lab, -1)
    k = lab >= 0
    cnt = np.bincount(lab[k], minlength=len(Y))
    Z = np.stack([np.bincount(lab[k], G[k, j], len(Y)) for j in range(2)], 1) / cnt.clip(1)[:, None]
    return np.where(cnt[:, None] > 0, Z, Y), g, lab


def render(lab, ids):
    img = np.ones(x.shape + (3,))
    img[wall] = 0.25
    sub = img[~wall]
    sub[lab >= 0] = color[ids[lab[lab >= 0]], :3]
    img[~wall] = sub
    return img


snaps, frames = [], []
for step in range(400):
    i, j = np.clip((Y[:, 1] / h).astype(int), 0, x.shape[0] - 1), np.clip((Y[:, 0] / h).astype(int), 0, x.shape[1] - 1)
    Y = Y - tau * np.c_[gx[i, j], gy[i, j]]
    keep = Y[:, 0] < 1.3
    Y, g, ids = Y[keep], g[keep], ids[keep]
    if len(Y) == 0:
        break
    Y, g, lab = project(Y, g)
    img = render(lab, ids)
    if step % 2 == 0:
        fig, ax = plt.subplots(figsize=(4.9, 3.5))
        ax.imshow(img, origin="lower")
        ax.axis("off")
        fig.tight_layout(pad=0)
        frames.append(frame(fig))
    snaps.append((step, img))

fig, axes = plt.subplots(1, 5, figsize=(18, 2.9))
for ax, k in zip(axes, np.linspace(0, len(snaps) - 1, 5).astype(int)):
    ax.imshow(snaps[k][1], origin="lower")
    ax.set_title(f"t = {snaps[k][0] * tau:.2f}")
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "crowd_evacuation.png", dpi=90)
save_gif(frames, "crowd_evacuation.gif", fps=20, bounce=False)
