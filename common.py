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


def conformers(smiles, n=1, seed=0, hs=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    if not hs:
        mol = Chem.RemoveHs(mol)
    X = np.stack([mol.GetConformer(i).GetPositions() for i in range(len(ids))])
    return mol, X - X.mean(1, keepdims=True)


def principal_axes(X):
    X = X - X.mean(0)
    return X @ np.linalg.svd(X, full_matrices=False)[2].T


def surface_points(mol, X, m, seed=0, scale=1.0):
    from rdkit import Chem
    rng = np.random.default_rng(seed)
    pt = Chem.GetPeriodicTable()
    r = scale * np.array([pt.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    k = rng.choice(len(r), 20 * m, p=r**2 / (r**2).sum())
    u = rng.normal(size=(len(k), 3))
    p = X[k] + r[k, None] * u / np.linalg.norm(u, axis=1, keepdims=True)
    d = np.linalg.norm(p[:, None] - X[None], axis=-1)
    p = p[(d >= r[None] - 1e-6).all(1)]
    return p[rng.choice(len(p), m, replace=False)]
