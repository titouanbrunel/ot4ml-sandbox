import numpy as np
import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from common import glyph_points, FIG

torch.manual_seed(0)
y = torch.tensor(glyph_points("∂", 800, seed=1), dtype=torch.float32)
x0 = 0.5 + 0.08 * torch.randn(800, 2)
x0[:, 0] -= 0.25
fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
for col, blur in [(0, 0.02), (2, 0.15)]:
    for row, debias in enumerate([False, True]):
        loss = SamplesLoss("sinkhorn", p=2, blur=blur, debias=debias, scaling=0.9)
        x = x0.clone().requires_grad_()
        for step in range(61):
            if step in (10, 60):
                ax = axes[row, col + (step == 60)]
                ax.scatter(*y.T, s=2, c="#226db4", alpha=0.25)
                ax.scatter(*x.detach().T, s=2, c="#c7372f")
                ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
                ax.set_title(f"{'debiased' if debias else 'raw'}, blur {blur}, step {step}")
            (g,) = torch.autograd.grad(loss(x, y), x)
            x.data -= 0.2 * len(x) * g
fig.tight_layout()
fig.savefig(FIG / "sinkhorn_debiasing.png", dpi=100)
