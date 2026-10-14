import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import ot
from common import conformers, FIG

B = 64
rng = np.random.default_rng(0)
mol, X1 = conformers("CC(=CCCC(C)(C=C)O)C", n=B, seed=1)
elem = np.array([a.GetSymbol() for a in mol.GetAtoms()])
sigma = X1.std()
X0 = sigma * rng.normal(size=X1.shape)
X0 -= X0.mean(1, keepdims=True)


def kabsch(a, b):
    U, _, Vt = np.linalg.svd(a.T @ b)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def align(x0, x1, iters=5):
    perm = np.arange(len(x0))
    for _ in range(iters):
        x0r = x0 @ kabsch(x0[perm], x1)
        for e in np.unique(elem):
            idx = np.nonzero(elem == e)[0]
            r, c = linear_sum_assignment(ot.dist(x1[idx], x0r[idx]))
            perm[idx[r]] = idx[c]
    x0 = x0[perm] @ kabsch(x0[perm], x1)
    return x0, ((x0 - x1) ** 2).sum(-1).mean()


def cost(x0, x1):
    return ((x0 - x1) ** 2).sum(-1).mean()


pairs = [[align(X0[i], X1[j]) for j in range(B)] for i in range(B)]
Caligned = np.array([[c for _, c in row] for row in pairs])
sigma_ot = ot.emd(ot.unif(B), ot.unif(B), Caligned).argmax(1)

costs = {
    "independent": [cost(X0[i], X1[i]) for i in range(B)],
    "rotation + permutation": [Caligned[i, i] for i in range(B)],
    "+ minibatch OT": [Caligned[i, sigma_ot[i]] for i in range(B)],
}
starts = {
    "independent": X0[0],
    "rotation + permutation": pairs[0][0][0],
    "+ minibatch OT": pairs[0][sigma_ot[0]][0],
}
ends = {"independent": X1[0], "rotation + permutation": X1[0], "+ minibatch OT": X1[sigma_ot[0]]}

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 4, 1)
for k, v in costs.items():
    ax.hist(v, bins=20, alpha=0.6, label=f"{k}: {np.mean(v):.2f}")
ax.set_xlabel("mean squared displacement per atom (Å²)")
ax.legend(fontsize=8)
ts = np.linspace(0, 1, 20)[:, None, None]
for k, name in enumerate(costs):
    ax = fig.add_subplot(1, 4, k + 2, projection="3d")
    path = (1 - ts) * starts[name] + ts * ends[name]
    for a in range(path.shape[1]):
        ax.plot(*path[:, a].T, c="#c7372f" if elem[a] == "O" else "#555555" if elem[a] == "C" else "#9ab", lw=1)
    ax.scatter(*ends[name].T, c="k", s=6)
    ax.set_title(name)
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "equivariant_coupling_linalool.png", dpi=100)
