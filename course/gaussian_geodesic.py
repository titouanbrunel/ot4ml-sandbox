import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from common import FIG

m0, m1 = np.array([-2.0, 0]), np.array([2.0, 0])
S0 = np.diag([3.0, 0.08])
R = np.array([[0, -1.0], [1.0, 0]])
S1 = R @ np.diag([2.5, 0.12]) @ R.T
r = sqrtm(S0).real
ri = np.linalg.inv(r)
A = ri @ sqrtm(r @ S1 @ r).real @ ri
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.c_[np.cos(theta), np.sin(theta)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, name in zip(axes, ["W2 geodesic", "linear covariance interpolation"]):
    for t in np.linspace(0, 1, 7):
        M = (1 - t) * np.eye(2) + t * A
        S = M @ S0 @ M if name == "W2 geodesic" else (1 - t) * S0 + t * S1
        e = (1 - t) * m0 + t * m1 + circle @ np.real(sqrtm(S)).T
        ax.plot(*e.T, c=plt.cm.coolwarm(t))
    ax.set(aspect="equal", title=name)
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG / "gaussian_geodesic.png", dpi=100)
