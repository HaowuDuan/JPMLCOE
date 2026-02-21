# Sign Error in Eq (28) of arXiv:2107.04672

**Paper**: Dai & Daum (2021), "Stiffness Mitigation in Stochastic Particle Flow Filters"

This note documents a sign error in the paper's BVP formula (Eq 28) for the
optimal homotopy schedule, and explains why the BVP is numerically intractable
for the paper's own numerical example regardless of sign.

---

## 1. Setup

The paper defines (Remark 3.2, p.9):

$$
M(\beta) = P_0^{-1} + \beta \, H^\top R^{-1} H
$$

where $P_0$ is the prior covariance, $H$ the observation Jacobian, and $R$ the
measurement noise covariance. Define shorthands:

$$
J_{\text{prior}} = P_0^{-1}, \qquad J_{\text{meas}} = H^\top R^{-1} H, \qquad
\frac{\partial M}{\partial \beta} = J_{\text{meas}}.
$$

The nuclear-norm condition number is (Remark 3.2):

$$
\kappa_\nu(M) = \|M\|_\nu \, \|M^{-1}\|_\nu = \operatorname{tr}(M) \, \operatorname{tr}(M^{-1}).
$$

The optimal schedule $\beta^*(\lambda)$ solves the BVP (Theorem 3.1, Eqs 26–27):

$$
\ddot{\beta} = \mu \, \frac{\partial \kappa_\nu}{\partial \beta}, \qquad
\beta(0) = 0, \quad \beta(1) = 1.
$$

---

## 2. Correct Derivative of $\kappa_\nu$

By the product rule:

$$
\frac{\partial \kappa_\nu}{\partial \beta}
= \frac{\partial \operatorname{tr}(M)}{\partial \beta} \, \operatorname{tr}(M^{-1})
+ \operatorname{tr}(M) \, \frac{\partial \operatorname{tr}(M^{-1})}{\partial \beta}.
$$

**First term.** Since $M = J_{\text{prior}} + \beta \, J_{\text{meas}}$:

$$
\frac{\partial \operatorname{tr}(M)}{\partial \beta} = \operatorname{tr}(J_{\text{meas}}).
$$

**Second term.** By Lemma A.1 of the paper ($dA^{-1}/d\theta = -A^{-1}(dA/d\theta)A^{-1}$):

$$
\frac{\partial M^{-1}}{\partial \beta} = -M^{-1} J_{\text{meas}} \, M^{-1},
$$

$$
\frac{\partial \operatorname{tr}(M^{-1})}{\partial \beta}
= \operatorname{tr}\!\bigl(-M^{-1} J_{\text{meas}} \, M^{-1}\bigr)
= -\operatorname{tr}\!\bigl(M^{-1} J_{\text{meas}} \, M^{-1}\bigr).
$$

Combining:

$$
\boxed{
\frac{\partial \kappa_\nu}{\partial \beta}
= \operatorname{tr}(J_{\text{meas}}) \, \operatorname{tr}(M^{-1})
\;-\; \operatorname{tr}(M) \, \operatorname{tr}(M^{-1} J_{\text{meas}} \, M^{-1}).
}
$$

The second term has a **minus** sign.

---

## 3. The Paper's Eq (28) Has a Plus Sign

The paper writes Eq (28) using $\nabla_x \nabla_x^\top \log h$ notation. Translating
to our notation ($J_{\text{meas}} = -\nabla_x \nabla_x^\top \log h$, which is PSD):

$$
\text{Eq (28)}: \quad
\ddot{\beta} = -\mu \bigl[
  \operatorname{tr}(\nabla\nabla\log h) \, \operatorname{tr}(M^{-1})
  + \operatorname{tr}(M) \, \operatorname{tr}(M^{-2} \nabla\nabla\log h)
\bigr].
$$

Substituting $\nabla\nabla\log h = -J_{\text{meas}}$:

$$
\text{Eq (28)} = -\mu \bigl[
  -\operatorname{tr}(J_{\text{meas}}) \, \operatorname{tr}(M^{-1})
  - \operatorname{tr}(M) \, \operatorname{tr}(M^{-2} J_{\text{meas}})
\bigr]
= \mu \bigl[
  \operatorname{tr}(J_{\text{meas}}) \, \operatorname{tr}(M^{-1})
  \;+\; \operatorname{tr}(M) \, \operatorname{tr}(M^{-1} J_{\text{meas}} M^{-1})
\bigr].
$$

This has a **plus** where the correct derivation gives a **minus**.

---

## 4. Finite-Difference Verification

Central finite differences $(f(\beta+\varepsilon) - f(\beta-\varepsilon))/(2\varepsilon)$
with $\varepsilon = 10^{-7}$, using the paper's numerical example (Section 4):

| $\beta$ | Finite diff | Minus sign | Plus sign | FD matches |
|------:|---------------:|--------------:|--------------:|:----------:|
| 0.00 | −559,000.44 | −559,000.43 | +561,729.57 | **minus** |
| 0.01 | −3,762.72 | −3,762.72 | +3,991.69 | **minus** |
| 0.10 | −41.83 | −41.83 | +71.20 | **minus** |
| 0.50 | −0.31 | −0.31 | +9.59 | **minus** |
| 1.00 | +0.59 | +0.59 | +5.55 | **minus** |

Relative error between finite difference and minus-sign formula: < $10^{-8}$ at all points.

---

## 5. Isotropic Sanity Check

For $M = (c + \beta) I$ (isotropic case with $J_{\text{meas}} = I$):

$$
\kappa_\nu = \operatorname{tr}((c+\beta)I) \cdot \operatorname{tr}((c+\beta)^{-1}I)
= n(c+\beta) \cdot \frac{n}{c+\beta} = n^2 = \text{const.}
$$

The condition number is constant (perfect conditioning at all $\beta$), so
$\partial\kappa_\nu/\partial\beta = 0$. Plugging into both formulas:

- **Minus sign**: $\operatorname{tr}(I) \cdot \operatorname{tr}((c+\beta)^{-1}I) - \operatorname{tr}((c+\beta)I) \cdot \operatorname{tr}((c+\beta)^{-2}I) = \frac{n}{c+\beta} - \frac{n}{c+\beta} = 0$ ✓
- **Plus sign**: $\frac{n}{c+\beta} + \frac{n}{c+\beta} = \frac{2n}{c+\beta} \neq 0$ ✗

The plus sign incorrectly predicts a nonzero derivative for a constant function.

---

## 6. BVP Shooting Results

Paper parameters: $P_0 = \text{diag}(1000, 2)$, sensors at $(\pm 3.5, 0)$,
$R = 0.04 I$, $\mu = 0.2$, $\eta_0 = (3, 5)$.

### 6a. Raw $\partial\kappa_\nu/\partial\beta$ — both signs fail

At $\beta = 0$: $|\partial\kappa_\nu/\partial\beta| \approx 559{,}000$. This is
caused by the 500:1 anisotropy in $P_0$ (the $(1,1)$ entry of $M^{-1}$ is 1000
while the $(2,2)$ entry is 2).

With magnitude $O(10^5)$, the BVP $\ddot\beta = \mu \cdot \partial\kappa_\nu/\partial\beta$
is itself extremely stiff:

| Sign | $\partial\kappa_\nu/\partial\beta$ at $\beta=0$ | BVP result |
|:----:|---:|:---|
| Correct (−) | −559,000 | $\ddot\beta \approx -112{,}000$; $\beta$ goes negative for $u_0 < 1.5$, overshoots for $u_0 > 2$. Solver diverges for small $u_0$. |
| Paper (+) | +561,730 | $\ddot\beta > 0$ always ($\beta$ convex); $\beta(1) \geq 1.62$ for all $u_0 \geq 0$. No solution exists. |

### 6b. Paper sign (+): no solution exists

Both terms in the plus-sign formula are strictly positive (traces of PSD matrices
with PD $M$), so $\partial\kappa_\nu/\partial\beta > 0$ for all $\beta \geq 0$.
Therefore $\ddot\beta = \mu \cdot (\text{positive}) > 0$, meaning $\beta$ is
**strictly convex**. A convex function with $\beta(0) = 0$ that must reach
$\beta(1) = 1$ requires an initial velocity $\dot\beta(0)$ small enough that
the convexity doesn't overshoot — but the acceleration is so large (order 200 at
$\beta=0$) that even $\dot\beta(0) = 0.01$ gives $\beta(1) = 1.62$.

Shooting scan:

| $u_0 = \dot\beta(0)$ | $\beta(1)$ |
|---:|---:|
| −1.00 | 1.936 |
| −0.50 | 1.707 |
| 0.01 | 1.624 |
| 0.50 | 1.712 |
| 1.00 | 1.941 |
| 2.00 | 2.637 |

$\beta(1) > 1$ for every $u_0$ tested (including negative). **The BVP has no solution.**

### 6c. Correct sign (−): solvable with log $\kappa$ normalization

The raw correct formula also fails because $|\partial\kappa_\nu/\partial\beta|
\approx 559{,}000$ makes the ODE too stiff for shooting. However, replacing the
objective $\kappa_\nu$ with $\log\kappa_\nu$ in the optimal control problem gives:

$$
\ddot\beta = \mu \, \frac{\partial \log \kappa_\nu}{\partial \beta}
= \mu \, \frac{1}{\kappa_\nu} \frac{\partial \kappa_\nu}{\partial \beta}.
$$

At $\beta = 0$: $\kappa_\nu = 502$, so $\partial\log\kappa_\nu/\partial\beta \approx -1{,}114$ instead of $-559{,}000$. This makes the BVP well-conditioned.

Shooting scan with log $\kappa_\nu$:

| $u_0$ | $\beta(1)$ |
|---:|---:|
| 1.50 | 0.626 |
| 1.70 | 1.009 |
| 2.00 | 1.457 |

Bisection converges to $u_0^* = 1.6947$, giving $\beta(1) = 1.0000000$.

---

## 7. The Resulting Schedule Is Nearly Linear

The optimal schedule from the log $\kappa_\nu$ BVP:

| $\lambda$ | $\beta^*(\lambda)$ | $\beta^* - \lambda$ |
|---:|---:|---:|
| 0.01 | 0.0143 | +0.004 |
| 0.10 | 0.1170 | +0.017 |
| 0.25 | 0.2692 | +0.019 |
| 0.50 | 0.5133 | +0.013 |
| 0.75 | 0.7560 | +0.006 |
| 1.00 | 1.0000 | 0.000 |

Maximum deviation: $\max|\beta^* - \lambda| = 0.0195$.

Compare with the paper's Figure 2(b): $\max|\beta^* - \lambda| \approx 0.14$
(seven times larger). Their $\dot\beta^*(0) \approx 14$ (Figure 2(c)); ours
is 1.69.

**The log $\kappa_\nu$ schedule reduces the condition number modestly:**

| $\lambda$ | $\kappa_\nu$(linear) | $\kappa_\nu$(optimal) | ratio |
|---:|---:|---:|---:|
| 0.01 | 43.3 | 31.6 | 0.73 |
| 0.05 | 11.1 | 9.4 | 0.85 |
| 0.10 | 6.9 | 6.3 | 0.91 |
| 0.50 | 4.0 | 4.0 | 1.00 |

This is not enough to reproduce the paper's Table 1 (tr(P): 1535 → 1028,
a 33% reduction).

---

## 8. Summary

1. **Eq (28) has a sign typo**: the second term should be minus, not plus.
   Verified by finite differences and the isotropic sanity check.

2. **The plus-sign BVP has no solution** for the paper's example: $\ddot\beta > 0$
   everywhere forces $\beta(1) > 1$ for all initial velocities.

3. **The correct minus sign gives** $|\partial\kappa_\nu/\partial\beta| \approx 559{,}000$
   at $\beta = 0$, making the raw BVP numerically intractable.

4. **Our fix**: replace $\kappa_\nu$ with $\log\kappa_\nu$ in the objective,
   giving $\ddot\beta = \mu \cdot (\partial\kappa_\nu/\partial\beta) / \kappa_\nu$.
   This reduces the derivative magnitude from $559{,}000$ to $1{,}114$ and makes
   the BVP solvable.

5. **The resulting schedule is nearly linear** ($\max|\beta^* - \lambda| = 0.02$),
   insufficient to reproduce the paper's reported results.

6. **The paper's Figure 2 and Table 1** show a much more aggressive schedule
   ($\max|\beta^* - \lambda| \approx 0.14$, $\dot\beta^*(0) \approx 14$) with
   significant tr(P) reduction. This cannot be obtained from the nuclear-norm
   condition number as described in Eq (28), with either sign. The authors likely
   used a different implementation than what the equation describes.

---

## Implementation

Our code (`stochastic_edh.py`, `_compute_optimal_schedule`) uses the log $\kappa_\nu$
formulation with the correct minus sign, Radau stiff solver, and grid-scan
bracketing. This is the best we can do given what the paper describes.
