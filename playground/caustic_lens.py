import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from geomloss import SamplesLoss
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage import data, color
from skimage.transform import resize
from common import frame, save_gif, FIG

n = 96
img = resize(color.rgb2gray(data.chelsea())[:, 75:375], (n, n), anti_aliasing=True) ** 2
u = (np.arange(n) + 0.5) / n
X = torch.tensor(np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2), dtype=torch.float32)
a = torch.full((n * n,), 1 / n**2)
b = torch.tensor((img + 0.02).ravel() / (img + 0.02).sum(), dtype=torch.float32)
eps = 0.01
F, G = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=0.9, backend="tensorized", potentials=True)(a, X, b, X)
T = torch.softmax(b.log() + (G - torch.cdist(X, X) ** 2 / 2) / eps**2, 1) @ X
h = -F.numpy().reshape(n, n)
T = RegularGridInterpolator((u, u), T.numpy().reshape(n, n, 2), bounds_error=False, fill_value=None)
k = 1536
rays = (np.stack(np.meshgrid(np.arange(k), np.arange(k)), -1).reshape(-1, 2) + np.random.default_rng(0).random((k * k, 2))) / k
Tr = T(rays)


def wall(s, bins=192):
    return gaussian_filter(np.histogram2d(*((1 - s) * rays + s * Tr).T, bins=bins, range=[[0, 1], [0, 1]])[0], 1.0)


vmax = np.percentile(wall(1.0), 99.5)
show = dict(cmap="afmhot", vmin=0, vmax=vmax)
ss = [0.2, 0.4, 0.6, 0.8, 1.0, 1.3]
fig, axes = plt.subplots(1, 7, figsize=(19, 3))
axes[0].imshow(LightSource(315, 35).shade(h, cmap=plt.cm.Blues, vert_exag=400, blend_mode="soft"))
axes[0].set_title("glass height")
for ax, s in zip(axes[1:], ss):
    ax.imshow(wall(s), **show)
    ax.set_title(f"screen at {s:.1f} D")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "caustic_lens.png", dpi=90)

frames = []
for s in np.linspace(0, 1.3, 24):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(wall(s), **show)
    ax.axis("off")
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "caustic_lens.gif", fps=14)
