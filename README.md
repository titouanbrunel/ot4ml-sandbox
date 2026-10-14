# Optimal transport, MVA

I came to optimal transport through a research project on flow matching for molecular generation.
This repo is my implementation of the course by Gabriel Peyré and Julie Delon ([OT4ML](https://www.gpeyre.com/ot4ml/)), redoing their examples and extending some of them.

## Images and shapes

<img src="figures/mccann_chess.gif" width="260"> <img src="figures/barycenters_chess.gif" width="260">

![](figures/mccann_chess.png)

Displacement interpolation from a knight to a queen, exact assignment on 2000 points. The bottom row is the mixture $(1-t)\alpha + t\beta$ of the same two measures.

![](figures/barycenters_chess.png)

Entropic barycenters of four silhouettes with bilinear weights, convolutional Sinkhorn on an 80 × 80 grid.

![](figures/color_coffee_ihc.png)

Coffee recolored with the palette of an immunohistochemistry slide. Exact OT between 2000 sampled pixels, extended to the full image, shown at intermediate $t$ with the moving RGB cloud.

![](figures/style_latent.png)

Style transfer as an OT map between the VGG features of the two images, decoded at intermediate $t$ with the AdaIN decoder of Huang and Belongie. Rows: diagonal Gaussian map (this is AdaIN), full Gaussian map $T(x) = m_1 + A(x - m_0)$ with the Bures solution for $A$, sliced OT map in the 512 dimensional feature space.

## Playground

<img src="figures/pixel_shuffle.gif" width="200"> <img src="figures/caustic_lens.gif" width="200"> <img src="figures/stippling.gif" width="200"> <img src="figures/crowd_evacuation.gif" width="200">

![](figures/pixel_shuffle.png)

The astronaut's pixels rearranged into the cat, colors untouched. Exact assignment on RGB plus a small position term, then every pixel flies in a straight line.

![](figures/caustic_lens.png)

A glass slab that draws the cat with sunlight. In the paraxial limit a ray leaving the glass at $x$ hits a screen at distance $sD$ at $(1-s)x + sT(x)$, so the glass height is the Brenier potential of the map from uniform light to the image, up to $|x|^2/2$, and moving the screen is McCann interpolation. Past $D$ the rays cross and caustics appear.

![](figures/stippling.png)

The cameraman drawn with 3000 dots. Each iteration solves the semi discrete problem so that every dot gets the same amount of ink, then moves the dots to the centroids of their Laguerre cells, as in blue noise through OT by de Goes et al.

![](figures/crowd_evacuation.png)

300 people leaving a room with a pillar in front of the door. At each step everyone walks along the geodesic distance to the exit, then the crowd is projected in $W_2$ onto densities at most 1, which gives each person a Laguerre cell capped by a disc (Leclerc, Mérigot, Santambrogio, Stra). The queue at the door comes from the projection alone.

## Course examples

![](figures/matching_1d.png)

Assignment on the line: convex cost gives the monotone matching, concave cost gives nested local exchanges.

![](figures/histogram_equalization.png)

Histogram equalization as the monotone map $F_\beta^{-1} \circ F_\alpha$, interpolated in $t$.

![](figures/gaussian_geodesic.png)

$W_2$ geodesic between Gaussians against linear interpolation of covariances, which swells in the middle.

![](figures/sinkhorn_epsilon.png)

Entropic plans for increasing $\varepsilon$ and the linear convergence of log domain Sinkhorn, slower as $\varepsilon$ decreases.

![](figures/unbalanced.png)

Balanced versus KL unbalanced transport with an outlier cluster. Marker size is the transported mass.

![](figures/sinkhorn_debiasing.png)

Particle descent on raw entropic OT against the Sinkhorn divergence. With a large blur the raw loss collapses onto a curve inside the target.

![](figures/semidiscrete_lloyd.png)

Voronoi cells, Laguerre cells with equal masses found by L-BFGS on the semi discrete dual, and Lloyd iterations.

![](figures/flow_matching_couplings.png)

Flow matching towards a queen with independent and minibatch OT couplings. The OT coupling gives straighter paths and usable samples with a single Euler step.

![](figures/langevin.png)

Gradient descent against Langevin dynamics in a double well. Only the second samples $e^{-V}$.

## Structure

```
common.py     glyph rasterization, Laguerre cells, semi discrete solver, gif helper
visual/       mccann_chess, barycenters_chess, color_transfer, style_transfer_latent
playground/   pixel_shuffle, caustic_lens, stippling, crowd_evacuation
course/       matching_1d, histogram_equalization, gaussian_geodesic, sinkhorn_epsilon,
              unbalanced, sinkhorn_debiasing, semidiscrete_lloyd, flow_matching_couplings, langevin
figures/      outputs
```

```
pip install -r requirements.txt
python -m visual.mccann_chess
```

Scripts run from the repo root and write to `figures/`. The style transfer downloads the VGG encoder and decoder weights from the pytorch-AdaIN releases into `weights/` on first run.
