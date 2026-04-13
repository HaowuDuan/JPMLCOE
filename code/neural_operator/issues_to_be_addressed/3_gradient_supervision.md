# Gradient Supervision and Regularization Options for the Operator

**Status:** filed for later, not the next thing to try
**Last updated:** 2026-04-11

This file collects two related questions that came up while debugging the
Step 11 quality test, both about what kind of training signal we feed the
operator. Neither is the immediate next step. Both are real options if
training stays unstable after the bandwidth fix.

The two questions:

1. **Should we create a set of "true gradients" — i.e. analytic Jacobians of
   known transport maps — and use them as a supervision signal in the
   loss?** This is sometimes called *Sobolev training* in the literature.
2. **What standard regularization techniques exist for controlling the
   smoothness of a network's gradient (its Jacobian) other than the
   local-linearity penalty we already use?**

This document does not propose code changes. It records what we know so we
can pick the right escalation if and when the current training pipeline
proves insufficient.

---

## 1. Why this came up

Right now the training objective for the operator is the squared
Monge-Ampère residual,

```
r(x; c) = log det J(x; c) − ( log p_h(x) − log q_h(T(x; c)) )
L_MA    = E_{x ~ p_h} [ r(x; c)^2 ]
```

evaluated at collocation points sampled from `p_h`. This is **self-supervised**:
no external "ground truth" map is used. The loss tells the network how
*self-consistent* its forward pass is, but not whether it has matched any
known correct answer.

After several debugging rounds the failure modes look like:

- on the first quality run, the operator collapsed to identity on every
  test cloud
- after fixing the training data, the operator started moving particles but
  was wildly inconsistent across clouds
- the loss surface during training is unstable: loss spikes 100×, gradient
  norms hit several hundred, the bandwidth varies by 24× across steps

We do not yet know whether this is fundamentally a *training-stability*
problem (which the bandwidth fix should address) or a *signal-strength*
problem (the MA residual alone is too weak to pin down a good map). The
current ordered debugging plan tries the stability fixes first. This file
is the contingency if that does not get us where we need to be.

---

## 2. What "supervised gradient training" would actually mean

The idea has a name in the literature: **Sobolev training**
(Czarnecki et al. 2017, "Sobolev Training for Neural Networks"). The
construction is:

1. Pick a family of distributions where the optimal transport map and its
   Jacobian are known analytically. For us the obvious choice is
   **Gaussian-to-Gaussian**: between `N(μ_p, Σ_p)` and `N(μ_q, Σ_q)` the
   Brenier map is

   ```
   T(x) = μ_q + A (x − μ_p),
   J(x) = A,            with     A = Σ_p^{-1/2} (Σ_p^{1/2} Σ_q Σ_p^{1/2})^{1/2} Σ_p^{-1/2}.
   ```

   In 1D this collapses to `T(x) = μ_q + (σ_q / σ_p)(x − μ_p)` and
   `J(x) = σ_q / σ_p`. No tuning, no random seeds.

2. Generate many `(p, q)` pairs from this family. For each pair, compute
   the analytic `T_true(x)` and `J_true(x)` at the particle locations.

3. Train the network with a loss that supervises **both** the function
   value and its Jacobian:

   ```
   L_sup = E_{x} [ ‖T_θ(x) − T_true(x)‖^2  +  λ ‖J_θ(x) − J_true(x)‖_F^2 ]
   ```

   The first term is value supervision; the second term is gradient
   supervision. The gradient supervision is what makes it "Sobolev" — you
   are matching the function in a Sobolev norm, not just an L^2 norm.

This gives the network direct, unambiguous information about what the
Jacobian *should* be, instead of asking it to figure out the correct
Jacobian from the implicit constraint of the Monge-Ampère residual.

---

## 3. Why this is more principled than the local-linearity penalty

The two are complementary, not redundant.

| | Local-linearity penalty | Sobolev training |
|---|---|---|
| **Type of signal** | Regularization | Supervision |
| **What it asks of `J`** | "Vary slowly in `x`" | "Match this specific target" |
| **Where the target comes from** | Self-consistency: from `T` itself, via `T(x+δ) − T(x) ≈ Jδ` | External ground truth: an analytic map |
| **Effect on training** | Restricts the function space | Pulls the network toward a specific function |
| **Cost** | One extra forward pass per step | Requires generating an analytic dataset |

The local-linearity penalty controls the **smoothness** of `J(x)` — how
fast it changes from one point to the next. It does not say anything about
whether `J(x)` is **correct**. A wildly wrong but slowly varying `J` can
satisfy the local-linearity penalty perfectly.

Sobolev training controls the **correctness** of `J(x)` against a known
target. It does not say anything about whether `J` varies smoothly between
the target points.

Used together they pin down both properties: Sobolev tells the network
where `J` should be, local-linearity tells it not to oscillate wildly
between target points.

---

## 4. The cleanest entry point: Gaussian-to-Gaussian pretraining

Even without committing to a Sobolev term in the main MA training loop, the
analytic Gaussian-to-Gaussian map is a clean **pretraining** target:

- Generate `K` random `(N(μ_p, σ_p^2), N(μ_q, σ_q^2))` pairs in 1D, with `μ`
  and `log σ` uniform on a wide range.
- For each pair, sample `N` particles from `N(μ_p, σ_p^2)`, compute the
  closed-form `T_true(x_i)` and `J_true(x_i) = σ_q / σ_p`.
- Train the operator with `L_sup` for some number of steps.
- Then switch to the MA loss for fine-tuning on the realistic
  position-correlated synthetic clouds we use for the main training.

This gives the operator a **sane initial map** before it ever sees the
hard self-supervised loss. The cost is one analytic dataset and a
pretraining phase (~few hundred steps). No tuning, no hyperparameters,
no risk of biasing the operator toward a Sinkhorn solution.

If the bandwidth fix does not stabilize MA-only training, this is the
first thing to try as escalation.

---

## 5. Standard regularization techniques other than local-linearity

The local-linearity penalty is one tool. The literature has many. They
split cleanly by which derivative they control.

### First-derivative (Jacobian magnitude / Lipschitz)

These bound `‖J(x)‖` directly. Standard tools.

| Technique | Penalty / constraint | Reference |
|---|---|---|
| Jacobian regularization | `λ ‖J(x)‖_F^2` at training points | Hoffman, Roberts, Sokolic 2019 |
| Gradient penalty (WGAN-GP) | `(‖∇f(x)‖ − 1)^2` to enforce 1-Lipschitz | Gulrajani et al. 2017 |
| Spectral normalization | Constrain each weight matrix to spectral norm ≤ 1 | Miyato et al. 2018 |
| Spectral-norm penalty | Soft version of the above | Yoshida & Miyato 2017 |
| Weight decay | Bounds weight magnitudes, indirect Lipschitz bound | Standard |
| Input noise (Tikhonov-equivalent) | Training with noise penalizes `‖∇f‖^2` | Bishop 1995 |

All of these would make `J` *small*. That is **not** what we want. We
want `J ≈ I`, not `J ≈ 0`. Used directly on the operator they would
fight against the residual parameterization. They are listed here only
for completeness — they are the wrong tool for our specific problem.

If we use one of them, it has to be on the *residual* `S` in
`T(x) = x + S(x)`, not on the full map. Then "small Jacobian" means
"small departure from identity", which is closer to what we want.

### Second-derivative (smoothness of `J`)

These control how fast `J(x)` varies in `x`. This *is* what we want.

| Technique | Penalty | Notes |
|---|---|---|
| Local-linearity penalty (current) | `E[‖T(x+δ) − T(x) − J(x)δ‖^2]` with `δ ~ K_h` | Finite-difference proxy for `‖∇²T‖^2`. Avoids nested autodiff. |
| Direct Hessian penalty | `λ ‖∇²f(x)‖_F^2` | Mathematically cleanest. Requires nested autodiff. Expensive. |
| Curvature regularization | Penalty on Hessian of loss, often via finite-difference proxy | Moosavi-Dezfooli et al. 2019 |
| Sobolev-space penalty | `∫ ‖∇²f‖^2 dx` | Classical PDE / inverse-problem regularizer. Same math. |

The local-linearity penalty we use is the standard cheap finite-difference
proxy for the direct Hessian penalty. It is not a workaround; it is the
right tool for this problem class. It just does not have a single famous
name in deep learning because the people who use it (continuous normalizing
flows, optimal-transport networks, physics-informed networks) call it
different things.

### Architectural smoothness (no penalty term)

These are not "penalties" in the loss but design choices that constrain
the function space.

- **Smooth activations**: tanh, GELU, softplus, RePU(`k≥2`). All `C^1` or
  better. We use weighted tanh.
- **Tied / structured weight matrices**: each module of the form
  `W^T diag(D) W` automatically has a symmetric Jacobian. This is the
  mGradNet-M choice, also a smoothness control of sorts.
- **Residual parameterization**: `T = id + S` with `S` initialized small.
  This is also a form of smoothness control: J starts at I and only
  departs as far as the optimization pushes S.

We already use all three of these. They get us a `C^∞` and PSD-by-construction
J. They do **not** by themselves control how fast J varies in x — that
still needs a regularizer (either local-linearity or one of the others).

---

## 6. Decision criteria

We will revisit this file when training stabilizes. Two questions
determine the right escalation:

1. **Does MA-only training, with stable bandwidth, give the right answer
   on the easy tier of the graded ladder?** If yes, none of this is
   needed. We can finish Step 11 with what we have.
2. **Does MA-only training stabilize at all on the easy tier even with
   the bandwidth fix?** If yes but the moments are wrong, the next
   escalation is the normalized moment loss (already discussed in
   `2_quality_test.md` §7b). If no, the next escalation is
   Gaussian-to-Gaussian Sobolev pretraining.

In other words:

- Bandwidth fix → still wrong moments → moment loss
- Bandwidth fix → still unstable training → Sobolev pretraining
- Bandwidth fix → both wrong moments AND unstable → both, in that order

The order matters because Sobolev pretraining is a structural change
(adds an analytic dataset, a new loss term, a pretraining phase) and we
should only take it on once we have ruled out the cheaper hyperparameter
fixes.

---

## 7. What this is not

This file is not a recommendation to add Sobolev training now. The current
debugging path is:

1. Confirm bandwidth is the instability source via the per-step CSV trace.
2. Apply Codex's fix #1 (source-based bandwidth in `kde.py`).
3. Re-run the graded ladder.
4. Decide based on the new result.

Sobolev training, the moment loss, and any of the other regularizers
listed here only become relevant after step 4 if MA-only training is still
not enough. They are escalations, not the default plan.
