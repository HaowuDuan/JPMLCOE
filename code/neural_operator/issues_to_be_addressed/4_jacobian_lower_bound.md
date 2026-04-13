# The J ≥ I Architectural Constraint

**Status:** root cause identified, fix not yet designed
**Date identified:** 2026-04-12
**Confirmed by:** Codex code review + oracle residual test

---

## 1. The problem

The current conditional transport map uses a residual parameterization:

```
T(x; c) = x + d(c) + Σ_m softplus(α_m(c)) T_m(x; c)
```

Each module Jacobian is:

```
J_m(x; c) = softplus(β_m(c)) W_m^T diag(φ'(·)) W_m
```

with `φ' ≥ 0` (WTanh derivative is non-negative) and `softplus(β) > 0`.
So each `J_m ≥ 0` in the PSD sense. The full Jacobian is:

```
J(x; c) = I + Σ_m softplus(α_m(c)) J_m(x; c)
```

Since every term in the sum is PSD, `J ≥ I` in the PSD ordering. In 1D
this means `T'(x) ≥ 1` everywhere — the map can only preserve or expand
local length. **It cannot compress.**

## 2. Why this matters

The oracle 1D transport map for our test cloud (200 equally-spaced
particles on [-5, 5], Gaussian weights with `σ_w = 3`, `μ = 1`) has:

```
J_oracle range: [~0, 2.31]
```

At the edges of the particle interval, the oracle compresses particles
inward (J ≈ 0). At the center, it expands them (J ≈ 2.3). This is the
natural shape of a uniform-to-Gaussian transport: particles at the tails
need to move toward the center.

With `J ≥ I`, the architecture cannot represent J < 1. The best it can
do is `J = 1` at the edges (identity, no compression) and `J > 1` at
the center (some expansion). This is a strict subset of the maps the
oracle needs.

## 3. Evidence

**Oracle residual test** (`test_oracle_residual.py`):
- Oracle MA loss = 1.19e-09 (the loss CAN be zero)
- Oracle cross-check (numerical J vs density-ratio J): mismatch = 0.000011

**Single-cloud overfitting test** (`test_overfit_single_cloud.py`):
- Training is STABLE on one fixed cloud (grad norms 0.7–3.6)
- Loss plateaus at ~0.23 after 200 steps, does not decrease further
- The model has reached the boundary of its representable function class

**The plateau is not caused by:**
- Training distribution variance (eliminated by using one fixed cloud)
- Bandwidth formula bugs (eliminated by using fixed h = 0.25)
- Loss formula errors (A0/A1 sanity tests pass)
- Encoder/trunk wiring bugs (Codex code review found none)
- Gradient computation errors (Codex code review found none)

## 4. What this constraint means physically

A transport map with `J ≥ I` everywhere is a map that can only "spread
things out". It adds the identity plus a non-negative correction. In
the resampling context:

- To move particles from a uniform distribution to a peaked one, you
  need to COMPRESS the tails (bring edge particles inward) and
  EXPAND the center (spread central particles to fill the peak).
- The current architecture can expand the center (J > 1 there) but
  CANNOT compress the tails (J < 1 is impossible). So the tails stay
  at or beyond their original positions.
- This means the architecture fundamentally cannot represent any
  transport from a broader distribution to a narrower one.

## 5. Fix requirements

The fix must satisfy:
1. **J > 0 (PSD)** — the map must be monotone (in 1D) or the gradient
   of a convex potential (in higher d). This is the Brenier requirement.
2. **J can be < I** — the map must be able to compress as well as expand.
3. **Near-identity initialization** — at step zero, the map should
   start close to identity so the loss landscape is smooth early on.
   (This was the original motivation for the residual parameterization.)

These three requirements are in tension. The residual `T = x + S` gives
(1) and (3) but sacrifices (2). Removing the residual gives (1) and (2)
but sacrifices (3).

## 6. Candidate fix directions

### 6a. Trainable base scale

Replace `I` with `λ(c) I` where `λ(c) > 0` is a learned scalar:

```
T(x; c) = λ(c) x + d(c) + Σ_m softplus(α_m(c)) T_m(x; c)
J(x; c) = λ(c) I + Σ_m softplus(α_m(c)) J_m(x; c)
```

Now `J ≥ λ(c) I`, and if `λ(c) < 1` the map can compress. PSD is
guaranteed because `λ(c) > 0` (enforce via softplus on the raw
parameter) and each `J_m ≥ 0`.

Initialize `λ` so that `softplus(raw_λ) ≈ 1` at step zero → near
identity. During training, λ can decrease below 1 to allow compression.

**Pros:** minimal change to existing code (add one Dense layer for λ,
change the residual line). Preserves near-identity init. Preserves PSD.
**Cons:** λ is global (same scale for all spatial locations). A cloud
that needs compression at the edges and expansion at the center would
need the modules to compensate for a global λ < 1 at the center too.

### 6b. Remove the residual, use non-residual PSD map

```
T(x; c) = d(c) + Σ_m softplus(α_m(c)) T_m(x; c)
J(x; c) = Σ_m softplus(α_m(c)) J_m(x; c)
```

Now J is PSD (≥ 0) but can have eigenvalues anywhere in [0, ∞). No
lower bound at I.

**Pros:** removes the constraint entirely. Full PSD map class.
**Cons:** loses near-identity init. At step zero, T(x) is some random
map, not close to x. Training starts from a harder initial condition.
Initialization requires careful design (e.g. Sobolev pretraining from
issue #3, or special weight init to make T ≈ x).

### 6c. Per-module learnable scale (most flexible)

```
J(x; c) = Σ_m γ_m(c) W_m^T D_m W_m
```

where `γ_m(c)` can be any positive scalar (not softplus(α) + 1). Each
module contributes a PSD term with arbitrary positive scale. If the sum
of all terms gives a Jacobian with eigenvalues in (0, ∞), PSD is
preserved.

To get near-identity init: initialize `γ_m` and `W_m` so that
`Σ_m γ_m W_m^T D_m W_m ≈ I` at step zero. This requires the `W_m` to
form an approximate basis and the `γ_m D_m` to sum to the identity
over that basis.

**Pros:** most expressive. No global scale bottleneck.
**Cons:** most complex to initialize. No architectural guarantee of
near-identity at init; depends on careful initialization.

## 7. Recommended next step

Design and implement fix 6a (trainable base scale). Reason:
- Smallest code change (one new Dense layer, one line in `call` and
  `jacobian`)
- Preserves near-identity init
- Preserves PSD
- Addresses the constraint directly (λ < 1 allows compression)
- If λ turns out to be too coarse (global scale insufficient), escalate
  to 6b or 6c

Do NOT implement the fix until the design is cross-checked with Codex.
The fix changes the architecture, which is a structural change, so it
requires explicit user approval.

## 8. Diagnostic that confirms this constraint

Already implemented in the oracle test. Additionally, a 2-line check
at the end of test_overfit_single_cloud.py would confirm directly:

```python
# After training, evaluate J on the training cloud
J_model = model.trunk.jacobian(particles, c)
print(f"min J_model = {tf.reduce_min(J_model).numpy():.4f}")  # should be >= 1
print(f"min J_oracle = {J_grid.min():.4f}")                   # should be << 1
```

This has not been run yet but is predicted to show `min J_model ≥ 1`.
