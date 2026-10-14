import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale
from common import ROOT, frame, save_gif, FIG

URL = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/"
torch.manual_seed(0)
torch.set_grad_enabled(False)


def conv(a, b):
    return [nn.ReflectionPad2d(1), nn.Conv2d(a, b, 3), nn.ReLU()]


def vgg():
    layers = [nn.Conv2d(3, 3, 1)]
    for a, b, k in [(3, 64, 2), (64, 128, 2), (128, 256, 4), (256, 512, 4), (512, 512, 4)]:
        for i in range(k):
            layers += conv(a if i == 0 else b, b)
        layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
    net = nn.Sequential(*layers[:-1])
    net.load_state_dict(torch.hub.load_state_dict_from_url(URL + "vgg_normalised.pth", ROOT / "weights"))
    return net[:31].eval()


def decoder():
    up = [nn.Upsample(scale_factor=2)]
    layers = conv(512, 256) + up + conv(256, 256) * 1 + conv(256, 256) + conv(256, 256) + conv(256, 128) + up
    layers += conv(128, 128) + conv(128, 64) + up + conv(64, 64) + conv(64, 3)[:2]
    net = nn.Sequential(*layers)
    net.load_state_dict(torch.hub.load_state_dict_from_url(URL + "decoder.pth", ROOT / "weights"))
    return net.eval()


def sqrtm(S, inv=False):
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(1e-5)
    return V @ torch.diag(w ** (-0.5 if inv else 0.5)) @ V.T


def gaussian_map(x, y, diagonal=False):
    mx, my = x.mean(0), y.mean(0)
    if diagonal:
        return my + (x - mx) * y.std(0) / x.std(0)
    Sx, Sy = torch.cov(x.T), torch.cov(y.T)
    r, ri = sqrtm(Sx), sqrtm(Sx, inv=True)
    A = ri @ sqrtm(r @ Sy @ r) @ ri
    return my + (x - mx) @ A


def sliced_map(x, y, iters=60):
    z = x.clone()
    n, m, d = len(x), len(y), x.shape[1]
    q = ((torch.arange(n) + 0.5) / n * m).long()
    for _ in range(iters):
        Q = torch.linalg.qr(torch.randn(d, d))[0]
        zp, yp = z @ Q, (y @ Q).sort(0)[0][q]
        order = zp.argsort(0)
        target = torch.empty_like(zp).scatter_(0, order, yp)
        z = z + (target - zp) @ Q.T
    return z


def load(img, width):
    img = rescale(img, width / img.shape[1], channel_axis=-1)
    return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None]


enc, dec = vgg(), decoder()
content, style = load(data.rocket(), 512), load(data.immunohistochemistry(), 512)
fc, fs = enc(content), enc(style)
c, h, w = fc.shape[1:]
x, y = fc[0].reshape(c, -1).T, fs[0].reshape(c, -1).T
maps = {"diagonal Gaussian": gaussian_map(x, y, True), "Gaussian": gaussian_map(x, y), "sliced": sliced_map(x, y)}


def decode(z):
    return dec(z.T.reshape(1, c, h, w)).clamp(0, 1)[0].permute(1, 2, 0).numpy()


ts = np.linspace(0, 1, 5)
fig, axes = plt.subplots(3, 6, figsize=(16, 7.2))
for i, (name, Tx) in enumerate(maps.items()):
    for k, t in enumerate(ts):
        axes[i, k].imshow(decode((1 - t) * x + t * Tx))
        axes[0, k].set_title(f"t = {t:.2f}")
    axes[i, 0].text(-0.05, 0.5, name, transform=axes[i, 0].transAxes, rotation=90, ha="right", va="center")
for ax in axes.ravel():
    ax.axis("off")
axes[1, 5].imshow(style[0].permute(1, 2, 0).numpy())
axes[1, 5].set_title("style")
fig.tight_layout()
fig.savefig(FIG / "style_latent.png", dpi=90)

frames = []
for t in np.linspace(0, 1, 16):
    fig, ax = plt.subplots(figsize=(4, 4 * h / w))
    ax.imshow(decode((1 - t) * x + t * maps["sliced"]))
    ax.axis("off")
    fig.tight_layout(pad=0)
    frames.append(frame(fig))
save_gif(frames, "style_latent.gif", fps=8)
