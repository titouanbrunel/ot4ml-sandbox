from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from skimage.transform import resize

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def glyph(char, n=128, pad=0.12):
    fig = plt.figure(figsize=(4, 4), dpi=128)
    fig.text(0.5, 0.5, char, fontsize=300, ha="center", va="center", family="DejaVu Sans")
    ink = 1 - frame(fig).mean(-1) / 255
    i, j = np.nonzero(ink > 0.5)
    ink = ink[i.min():i.max() + 1, j.min():j.max() + 1]
    m = round(n * (1 - 2 * pad))
    h, w = (np.array(ink.shape) * m / max(ink.shape)).astype(int)
    out = np.zeros((n, n))
    out[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = resize(ink, (h, w), anti_aliasing=True)
    return out.clip(0, 1)


def glyph_points(char, m, seed=0, n=256):
    rng = np.random.default_rng(seed)
    img = glyph(char, n)
    idx = rng.choice(img.size, m, p=img.ravel() / img.sum())
    i, j = np.unravel_index(idx, img.shape)
    return np.c_[j + rng.random(m), n - i - rng.random(m)] / n


def frame(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_gif(frames, name, fps=12, bounce=True):
    frames = list(frames)
    if bounce:
        frames = frames + frames[::-1]
    imageio.mimsave(FIG / name, frames, duration=1000 / fps, loop=0)



def laguerre(G, Y, g):
    from scipy.spatial import cKDTree
    M = g.max()
    d, lab = cKDTree(np.c_[Y, np.sqrt(M - g)]).query(np.c_[G, np.zeros(len(G))], workers=-1)
    return lab, d**2 - M


def semidiscrete(G, rho, Y, b, g=None, tol=0.05, iters=1000):
    from scipy.optimize import minimize
    m = len(Y)

    def f(g):
        lab, v = laguerre(G, Y, g / m)
        return -m * m * (rho @ v + b @ g / m), m * (np.bincount(lab, rho, m) - b)

    g = np.zeros(m) if g is None else m * g
    return minimize(f, g, jac=True, method="L-BFGS-B", options={"maxiter": iters, "ftol": 0, "gtol": tol}).x / m
