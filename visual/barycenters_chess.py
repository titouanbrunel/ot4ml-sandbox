import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import ot
from common import glyph, frame, save_gif, FIG

n, reg = 80, 8e-4
A = np.stack([binary_fill_holes(glyph(c, n) > 0.5) for c in "♞♛♜♝"]).astype(float)
A /= A.sum((1, 2), keepdims=True)


def bary(w):
    w = np.asarray(w, float)
    if w.max() > 1 - 1e-9:
        return A[w.argmax()]
    return ot.bregman.convolutional_barycenter2d(A, reg, w, warn=False)


def show(ax, img):
    ax.imshow(img, cmap="Greys", vmin=0, vmax=img.max())
    ax.axis("off")


k = 5
fig, axes = plt.subplots(k, k, figsize=(8, 8))
for i, u in enumerate(np.linspace(0, 1, k)):
    for j, v in enumerate(np.linspace(0, 1, k)):
        show(axes[i, j], bary([(1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v]))
fig.tight_layout()
fig.savefig(FIG / "barycenters_chess.png", dpi=100)

frames = []
corners = np.eye(4)[[0, 1, 3, 2, 0]]
for a, b in zip(corners[:-1], corners[1:]):
    for t in np.linspace(0, 1, 8, endpoint=False):
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        show(ax, bary((1 - t) * a + t * b))
        fig.tight_layout(pad=0)
        frames.append(frame(fig))
save_gif(frames, "barycenters_chess.gif", fps=20, bounce=False)
