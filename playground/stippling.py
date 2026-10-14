import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import resize
from common import laguerre, semidiscrete, frame, save_gif, FIG

n, m = 384, 3000
rho = (1 - resize(data.camera(), (n, n), anti_aliasing=True)) ** 2
u = (np.arange(n) + 0.5) / n
G = np.stack(np.meshgrid(u, -u), -1).reshape(-1, 2)
rho = rho.ravel() / rho.sum()
b = np.full(m, 1 / m)
Y = np.random.default_rng(0).random((m, 2)) * [1, -1]
traj = [Y]
for it in range(15):
    g = semidiscrete(G, rho, Y, b, tol=0.3, iters=300 if it == 0 else 100)
    lab, _ = laguerre(G, Y, g)
    mass = np.bincount(lab, rho, m)
    Y = np.where(mass[:, None] > 0, np.stack([np.bincount(lab, rho * G[:, k], m) for k in range(2)], 1) / np.maximum(mass, 1e-12)[:, None], Y)
    traj.append(Y)


def dots(ax, Y):
    ax.scatter(*Y.T, s=2.8, c="k", lw=0)
    ax.set(xlim=(0, 1), ylim=(-1, 0), aspect="equal")
    ax.axis("off")


steps = [0, 1, 2, 4, 15]
fig, axes = plt.subplots(1, 6, figsize=(17, 3.1))
for ax, k in zip(axes, steps):
    dots(ax, traj[k])
    ax.set_title(f"iteration {k}")
axes[-1].imshow(data.camera(), cmap="gray")
axes[-1].set_title("image")
axes[-1].axis("off")
fig.tight_layout()
fig.savefig(FIG / "stippling.png", dpi=90)

frames = []
for k in range(len(traj) - 1):
    for t in np.linspace(0, 1, 4 if k < 6 else 1, endpoint=False):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        dots(ax, (1 - t) * traj[k] + t * traj[k + 1])
        fig.tight_layout(pad=0)
        frames.append(frame(fig))
save_gif(frames, "stippling.gif", fps=12)
