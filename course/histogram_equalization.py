import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from common import FIG

x = data.moon().astype(float).ravel() / 255
Tx = np.argsort(np.argsort(x, kind="stable")) / (len(x) - 1)
fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), gridspec_kw={"height_ratios": [3, 1]})
for k, t in enumerate(np.linspace(0, 1, 5)):
    xt = (1 - t) * x + t * Tx
    axes[0, k].imshow(xt.reshape(512, 512), cmap="gray", vmin=0, vmax=1)
    axes[0, k].set_title(f"t = {t:.2f}")
    axes[0, k].axis("off")
    axes[1, k].hist(xt, bins=64, range=(0, 1), color="#7c5bbd")
    axes[1, k].set_yticks([])
fig.tight_layout()
fig.savefig(FIG / "histogram_equalization.png", dpi=90)
