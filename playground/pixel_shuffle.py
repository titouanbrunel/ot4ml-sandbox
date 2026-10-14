import numpy as np
import matplotlib.pyplot as plt
import ot
from skimage import data
from skimage.transform import resize
from common import frame, save_gif, FIG

n = 80
src = resize(data.astronaut()[:, 40:472], (n, n), anti_aliasing=True)
tgt = resize(data.chelsea()[:, 60:360], (n, n), anti_aliasing=True)
u = np.arange(n) / n
p = np.stack(np.meshgrid(u, -u), -1).reshape(-1, 2)
c, d = src.reshape(-1, 3), tgt.reshape(-1, 3)
P = ot.emd(ot.unif(n * n), ot.unif(n * n), ot.dist(c, d) + 0.05 * ot.dist(p, p))
q = p[P.argmax(1)]


def draw(ax, t):
    ax.scatter(*((1 - t) * p + t * q).T, c=c, s=6.5, marker="s", lw=0)
    ax.set(xlim=(-0.01, 1), ylim=(-1, 0.01), aspect="equal")
    ax.axis("off")


fig, axes = plt.subplots(1, 7, figsize=(17, 2.8))
for ax, t in zip(axes, np.linspace(0, 1, 6)):
    draw(ax, t)
    ax.set_title(f"t = {t:.1f}")
axes[-1].imshow(tgt)
axes[-1].set_title("target")
axes[-1].axis("off")
fig.tight_layout()
fig.savefig(FIG / "pixel_shuffle.png", dpi=90)

frames = []
for t in np.linspace(0, 1, 40):
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    draw(ax, t * t * (3 - 2 * t))
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "pixel_shuffle.gif", fps=20)
