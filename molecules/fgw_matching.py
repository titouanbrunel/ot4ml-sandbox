import numpy as np
import matplotlib.pyplot as plt
import ot
from rdkit.Chem import AllChem
from common import conformers, FIG

colors = {"C": "#555555", "O": "#d62728"}
mols = []
for smi in ["COc1cc(C=O)ccc1O", "COc1cc(CC=C)ccc1O"]:
    mol, X = conformers(smi, hs=False)
    F = np.array([[a.GetSymbol() == "C", a.GetSymbol() == "O", a.GetIsAromatic()] for a in mol.GetAtoms()], float)
    AllChem.Compute2DCoords(mol)
    mols.append((mol, ot.dist(X[0], X[0], "euclidean"), F, mol.GetConformer().GetPositions()[:, :2]))

(m1, C1, F1, P1), (m2, C2, F2, P2) = mols
p, q = ot.unif(len(F1)), ot.unif(len(F2))
M = ot.dist(F1, F2)
plans = {
    "W on atom features": ot.emd(p, q, M),
    "GW on 3D distances": ot.gromov.gromov_wasserstein(C1, C2, p, q),
    "FGW, alpha = 0.5": ot.gromov.fused_gromov_wasserstein(M, C1 / C1.max(), C2 / C2.max(), p, q, alpha=0.5),
}


def draw(ax, mol, P, c):
    for b in mol.GetBonds():
        ax.plot(*P[[b.GetBeginAtomIdx(), b.GetEndAtomIdx()]].T, c="k", lw=1.5, zorder=1)
    edge = ["#d62728" if a.GetSymbol() == "O" else "k" for a in mol.GetAtoms()]
    ax.scatter(*P.T, c=c, s=160, zorder=2, edgecolors=edge, linewidths=2.5)


P1 = P1 - P1.mean(0) - [4, 0]
P2 = P2 - P2.mean(0) + [4, 0]
c1 = plt.cm.tab20(np.arange(len(p)) / len(p))
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, (name, T) in zip(axes, plans.items()):
    draw(ax, m1, P1, c1)
    draw(ax, m2, P2, c1[T.argmax(0)])
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("vanillin to eugenol: each eugenol atom takes the color of its main match, oxygens circled in red")
fig.tight_layout()
fig.savefig(FIG / "fgw_vanillin_eugenol.png", dpi=100)
