# Why LEDH+OT Has Gradient Bias But BPF+OT Does Not

## Setup

Parameter of interest: $\theta$ (e.g. `obs_noise_std`).

At each timestep $t$, we have $N$ particles $x_t^i$ and weights $w_t^i$.

## OT Resampling

After computing weights, OT resampling produces new particles:

$$\tilde{x}_t^i = \sum_{j=1}^{N} T_{ij} \cdot x_t^j$$

where $T \in \mathbb{R}^{N \times N}$ is the transport matrix. In matrix form:

$$\tilde{\mathbf{x}}_t = T(\mathbf{x}_t, \mathbf{w}_t) \cdot \mathbf{x}_t$$

**Key point:** $T$ is itself a function of the particles $\mathbf{x}_t$ and weights $\mathbf{w}_t$, because:

1. Particles are centered and scaled:
$$\bar{x} = \frac{1}{N}\sum_i x_t^i, \quad s = \text{std}(\mathbf{x}_t)\sqrt{d}, \quad z_t^i = \frac{x_t^i - \bar{x}}{s}$$

2. Cost matrix computed from scaled particles:
$$C_{ij} = \|z_t^i - z_t^j\|^2$$

3. Sinkhorn iteration on $(C, \log \mathbf{w}_t)$ produces dual potentials $\alpha, \beta$

4. Transport matrix:
$$T_{ij} = \exp\left(\frac{\alpha_i + \beta_j - C_{ij}}{\epsilon}\right) \cdot w_t^j$$

## The Full Derivative

The total derivative of $\tilde{\mathbf{x}}_t$ w.r.t. $\theta$ is:

$$\frac{d\tilde{\mathbf{x}}_t}{d\theta} = \underbrace{\frac{\partial T}{\partial \mathbf{x}_t} \frac{d\mathbf{x}_t}{d\theta} \cdot \mathbf{x}_t}_{\text{Term 1: particle geometry}} + \underbrace{T \cdot \frac{d\mathbf{x}_t}{d\theta}}_{\text{Term 2: direct transport}} + \underbrace{\frac{\partial T}{\partial \mathbf{w}_t} \frac{d\mathbf{w}_t}{d\theta} \cdot \mathbf{x}_t}_{\text{Term 3: weight gradient}}$$

- **Term 1**: How $\theta$ changes particle positions, which changes the cost matrix, which changes $T$, which changes the transported particles.
- **Term 2**: How $\theta$ changes particle positions, directly transported by fixed $T$.
- **Term 3**: How $\theta$ changes weights, which changes $T$, which changes the transported particles.

## BPF vs LEDH

### BPF on Linear Gaussian with $\theta = $ obs_noise_std

BPF particles come from the state transition:

$$x_t^i = F \tilde{x}_{t-1}^i + B \epsilon_t^i$$

Neither $F$, $B$, nor $\epsilon_t^i$ depend on obs_noise_std. Therefore:

$$\frac{d\mathbf{x}_t}{d\theta} = 0$$

This kills **Term 1** and **Term 2** entirely. Only **Term 3** survives:

$$\frac{d\tilde{\mathbf{x}}_t}{d\theta} = \frac{\partial T}{\partial \mathbf{w}_t} \frac{d\mathbf{w}_t}{d\theta} \cdot \mathbf{x}_t$$

The weight gradient $d\mathbf{w}_t/d\theta$ is well-behaved (it flows through $\log p(y_t | x_t^i)$ which depends on $R = \theta^2 D D^\top$). This path gives **ratio 1.08** — nearly correct.

### LEDH on Linear Gaussian with $\theta = $ obs_noise_std

LEDH particles come from the flow:

$$\eta_1^i = \eta_0^i + \sum_{j=1}^{J} \Delta\lambda_j \left[ A_j \eta_1^{i,(j-1)} + b_j \right]$$

where the flow matrices $A_j$ and vectors $b_j$ depend on $\theta$ through $R$ and $R^{-1}$:

$$A_j = -\frac{1}{2} P H^\top (\lambda_j H P H^\top + R)^{-1} H$$

Therefore:

$$\frac{d\mathbf{x}_t}{d\theta} = \frac{d\boldsymbol{\eta}_1}{d\theta} \neq 0$$

Now **all three terms** are active. The extra terms (**Term 1** and **Term 2**) introduce the bias.

## Why Term 1 Is Biased

The OT custom gradient computes $\partial T / \partial \mathbf{x}_t$ approximately:

1. The centering $\bar{x}$ and scaling $s$ were computed with `tf.stop_gradient` (now removed, which fixed BPF). But even without stop_gradient, the Sinkhorn backward is approximate.

2. Sinkhorn iteration is run to convergence in the forward pass, but the backward pass does **not** differentiate through all iterations. Instead, it uses stopped potentials with one extrapolation step:

$$\alpha_{\text{stop}} = \text{stop\_gradient}(\alpha_{\text{converged}})$$
$$\alpha_{\text{new}} = \text{softmin}(\epsilon, C^\top, \log\mathbf{w} + \beta_{\text{stop}}/\epsilon)$$
$$\alpha_{\text{extra}} = \delta \cdot \alpha_{\text{stop}} + (1-\delta) \cdot \alpha_{\text{new}}$$

This is an approximation. The true derivative would require differentiating through all Sinkhorn iterations (implicit differentiation). The approximation is accurate for the weight gradient (**Term 3**) but introduces bias for the particle gradient (**Term 1**) because particle positions affect the cost matrix $C$, which feeds into every Sinkhorn iteration.

3. The cost matrix $C_{ij} = \|z^i - z^j\|^2$ uses the same scaled particles on both sides. The gradient $\partial C / \partial x^i$ has contributions from both the $i$-th row and $i$-th column of $C$. The approximate Sinkhorn backward may not correctly account for this symmetric structure.

## Summary

| | $d\mathbf{x}_t/d\theta$ | Active Terms | OT Gradient | Ratio |
|---|---|---|---|---|
| BPF (LG, obs_noise_std) | $= 0$ | Term 3 only | Accurate | 1.08 |
| LEDH (LG, obs_noise_std) | $\neq 0$ | Terms 1+2+3 | Term 1 biased | 1.33 |

## Proposed Validation

Zero only `dparticles` in the OT custom gradient return (keep `dlog_weights`). This kills Terms 1 and 2 while preserving Term 3. If LEDH ratio moves from 1.33 toward 1.08, the particle-geometry backward (**Term 1**) is confirmed as the bias source.
