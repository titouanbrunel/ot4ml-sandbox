import numpy as np
import matplotlib.pyplot as plt
import ot
from skimage import data
from skimage.transform import rescale
from common import frame, save_gif, FIG

pairs = {"coffee_ihc": (data.coffee(), data.immunohistochemistry())}
rng = np.random.default_rng(0)
ts = np.linspace(0, 1, 5)

for name, (src, tgt) in pairs.items():
    src = rescale(src, 320 / src.shape[1], channel_axis=-1)
    tgt = rescale(tgt, 320 / tgt.shape[1], channel_axis=-1)
    X, Y = src.reshape(-1, 3), tgt.reshape(-1, 3)
    xs, ys = X[rng.choice(len(X), 2000)], Y[rng.choice(len(Y), 2000)]
    T = ot.da.EMDTransport().fit(Xs=xs, Xt=ys)
    TX = T.transform(Xs=X).clip(0, 1)
    show = rng.choice(len(X), 3000)

    fig = plt.figure(figsize=(15, 6))
    for k, t in enumerate(ts):
        Xt = (1 - t) * X + t * TX
        ax = fig.add_subplot(2, 6, k + 1)
        ax.imshow(Xt.reshape(src.shape))
        ax.set_title(f"t = {t:.2f}")
        ax.axis("off")
        ax = fig.add_subplot(2, 6, k + 7, projection="3d")
        ax.scatter(*Xt[show].T, c=Xt[show], s=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xticks=[], yticks=[], zticks=[])
    ax = fig.add_subplot(2, 6, 6)
    ax.imshow(tgt)
    ax.set_title("target palette")
    ax.axis("off")
    ax = fig.add_subplot(2, 6, 12, projection="3d")
    ax.scatter(*ys[:3000].T, c=ys[:3000], s=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xticks=[], yticks=[], zticks=[])
    fig.tight_layout()
    fig.savefig(FIG / f"color_{name}.png", dpi=90)

    frames = []
    for t in np.linspace(0, 1, 16):
        fig, ax = plt.subplots(figsize=(4, 4 * src.shape[0] / src.shape[1]))
        ax.imshow(((1 - t) * X + t * TX).reshape(src.shape))
        ax.axis("off")
        fig.tight_layout(pad=0)
        frames.append(frame(fig))
    save_gif(frames, f"color_{name}.gif", fps=10)
