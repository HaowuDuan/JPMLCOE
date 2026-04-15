# HMC Debugging Plan: 2D Stochastic Volatility (LEDH + OT)

## Current state

**Symptom.** HMC chain freezes after one step. With initial `sigma2=1.6` (true=1.0), `step_size=0.02` fixed, `num_leapfrog=5`, `dtype=float32`:

```
[grad check] lp=-194.43, |grad|=2.30
[burn-in 1/30] step_size=0.0200 | sigma2=1.7518 | accept=100%   <- moved away from mode and accepted
[burn-in 2/30] step_size=0.0200 | sigma2=1.7518 | accept=50%    <- rejected
[burn-in 3/30] step_size=0.0200 | sigma2=1.7518 | accept=33%    <- rejected
[burn-in 4/30] step_size=0.0200 | sigma2=1.7518 | accept=25%    <- rejected
[burn-in 5/30] step_size=0.0200 | sigma2=1.7518 | accept=20%    <- rejected
[burn-in 6/30] step_size=0.0200 | sigma2=1.7518 | accept=17%    <- rejected
```

After step 1, every proposal is rejected. Chain is dead.

## What we have already ruled out

- **Gradient correctness.** `tests/hmc/test_gradient_vs_numerical_sv2d.py` shows autodiff vs central FD relative error < 2% for `sigma2 in {0.5, 1.0, 1.5, 2.0}` at T=20, 200 particles, 15 lambda steps. The gradient itself is fine.
- **Implicit Sinkhorn singular matrix.** Was crashing the run with "Input matrix is not invertible" before — fixed by adding `1e-6 * I` ridge regularization to the linear solve in `src/resampling/ot_entropy.py` `_sinkhorn_implicit_vjp`. The original `extra/implicit_sinkhorn_plan.md` called for this, it just never made it into the code.
- **Standalone OT gradient validation.** `tests/hmc/test_ot_gradient_standalone.py` shows the implicit backward matches numerical to ~1e-10 on a single resampling step, in three modes (weights only, particles only, both).
- **Dual averaging chaos.** Earlier runs had step size oscillating 50× between 0.0008 and 0.04 because dual averaging was thrashing. Disabled it in the smoke config (`adaptation_rate: 0.001`). The freeze above is at fixed `step_size=0.02` — not a dual averaging artifact.

## What we do NOT know (this is what the diagnostics need to answer)

1. The actual `delta_H` per leapfrog step. Is it huge? Small but consistently positive?
2. Is the rejection from leapfrog integration error (numerical) or from the proposal landing in low density (geometric)?
3. Is `float32` too noisy for the leapfrog energy conservation?
4. Does the particle filter log-likelihood have noise across particle seeds at `sigma2=1.75`?
5. Does deterministic gradient descent from `sigma2=1.6` actually find the mode? (Loss surface sanity check.)
6. Does a different HMC seed avoid the freeze, or is it systematic?

---

## Diagnostic plan (Codex-designed, in priority order)

Each diagnostic is structured as: **what to do**, **what to measure**, **interpretation tree**.

Cheapest first. Diagnostics 4–6 need no new code; 1–3 need a "replay harness" (see below).

### Diagnostic 6 — Seed-robustness of the freeze (CHEAPEST, RUN FIRST)

- **What to do**: Run 5–10 short burn-ins (5–10 steps each) from the same initial state `sigma2=1.6` with different HMC seeds. First with a fixed PF seed, then with varied PF seeds.
- **What to measure**: First accepted move (where does it land?), rejection streak length, acceptance rate per run.
- **Interpretation tree**:
  - Only one run freezes → it was a bad momentum draw. Single-run anomaly.
  - Most runs freeze with fixed PF seed → systematic bad dynamics; it's a real HMC tuning / numerical issue.
  - Only varying PF seeds changes behavior → particle filter Monte Carlo noise is the lever.
- **Cost**: ~10 min wall clock total (10 runs × ~10 steps × 11 s/step).
- **Code needed**: none — just loop the existing runner with different `seed:` values.

### Diagnostic 5 — Local descent sanity check

- **What to do**: From `sigma2=1.6`, run 20–30 deterministic gradient descent steps on `-logpost` with a fixed PF seed. Repeat with 3 different PF seeds.
- **What to measure**: Sequence of `(sigma2, logpost, grad)`. Does it move toward and find the mode? Do all 3 seeds agree?
- **Interpretation tree**:
  - All 3 seeds move consistently toward the mode and improve `logpost` → loss surface is sane; problem is purely HMC dynamics.
  - Seeds disagree strongly or oscillate → effective objective is noisy or badly conditioned; the issue is upstream of HMC.
- **Cost**: ~15 min total.
- **Code needed**: none — there's a MAP runner (`run_dpf_experiment` with `dpf=map/...`). Use it.

### Diagnostic 4 — Particle filter seed noise at the stuck point

- **What to do**: At `sigma2 in {1.70, 1.75, 1.80}`, run the particle filter 20–50 times with different particle seeds. Record `logpost` and gradient each time.
- **What to measure**: Mean and SD of `logpost` and gradient across seeds. Cross-sigma change in mean.
- **Interpretation tree**:
  - SD is small relative to the cross-sigma2 mean change → MC noise is not driving rejection. Problem is elsewhere.
  - SD comparable to cross-sigma2 change → HMC is seeing a noisy target; chain rejection is statistically expected. Need more particles or variance reduction.
- **Cost**: ~30 min total. Each PF eval is ~5 s.
- **Code needed**: small standalone script that calls `filt.log_marginal_likelihood_tf(obs, seed=...)` in a loop with different seeds, no HMC machinery.

### Diagnostic 1 — Rejected-step replay (REQUIRES REPLAY HARNESS)

- **What to do**: Reproduce burn-in step 2 exactly from the frozen state `sigma2=1.7518` with the same HMC seed, same momentum draw, same PF seed. Instrument the leapfrog integrator to log per-substep `(q, p, U=-logpost, K=p²/2m, H=U+K, grad, |grad|)`.
- **What to measure**: Final `delta_H`, endpoint `q_prop`, acceptance probability `exp(-delta_H)`, and the full `H` trace across leapfrog substeps.
- **Interpretation tree**:
  - `delta_H >> 1` → rejection is integration error dominated. Float precision or step size is the problem.
  - `delta_H ~ 0.1–1` but `U(q_prop) >> U(q_start)` → proposal endpoint is geometrically bad, not badly integrated. The chain is being asked to jump uphill.
  - `delta_H < 0` → replay should accept. If it doesn't, instrumentation has a bug.
- **Cost**: ~5 min once the harness exists.
- **Code needed**: replay harness — see "Replay harness" section below.

### Diagnostic 2 — Same-trajectory step-size sweep

- **What to do**: Using the same starting state, momentum, and PF seed, rerun the trajectory at `step_size in {0.02, 0.01, 0.005}`. First with leapfrog count fixed; then optionally with path length `L*eps` fixed.
- **What to measure**: `delta_H(eps)`, endpoint `q_prop(eps)`, acceptance probability for each.
- **Interpretation tree**:
  - `delta_H` collapses rapidly as `eps` shrinks → leapfrog integration error is the dominant problem. Smaller step is the fix.
  - `delta_H` stays modest but all endpoints land at low density → it's a momentum/geometry issue, not integration. Smaller step won't help much.
- **Cost**: ~15 min once the harness exists.
- **Code needed**: same replay harness as diagnostic 1.

### Diagnostic 3 — float32 vs float64 replay

- **What to do**: Repeat diagnostic 1 in both `dtype=float32` and `dtype=float64`. Same state, momentum, PF seed, code path.
- **What to measure**: Difference in `delta_H`, endpoint `q_prop`, per-step gradients.
- **Interpretation tree**:
  - `float64` materially reduces `delta_H` or flips accept/reject → precision is implicated. Either run float64 or fix the offending op.
  - Traces nearly identical → `float32` is not the primary cause. Don't waste time on a precision rewrite.
- **Cost**: ~10 min once the harness exists. `float64` is ~2× slower.
- **Code needed**: same replay harness as diagnostic 1, plus a `dtype` switch (already in config).

---

## Replay harness — what to build

Diagnostics 1, 2, 3 all need the same capability: **run a single HMC leapfrog trajectory in isolation, with full per-substep instrumentation, given a fixed initial state, initial momentum, particle seed, step_size, and num_leapfrog**.

Sketch:

```python
# code/tests/hmc/replay_hmc_step.py (or similar)

def replay_leapfrog(
    target_log_prob_fn,    # closure: q -> scalar logpost
    q0,                    # initial position (unconstrained sigma2)
    p0,                    # initial momentum (drawn deterministically from a fixed seed)
    step_size,
    num_leapfrog,
    pf_seed,
):
    """Run one leapfrog trajectory and return per-substep diagnostics."""
    trace = []
    q, p = q0, p0
    U0 = -target_log_prob_fn(q)
    K0 = 0.5 * tf.reduce_sum(p ** 2)
    H0 = U0 + K0

    for k in range(num_leapfrog):
        # Half step in p
        with tf.GradientTape() as tape:
            tape.watch(q)
            U = -target_log_prob_fn(q)
        grad_U = tape.gradient(U, q)
        p = p - 0.5 * step_size * grad_U
        # Full step in q
        q = q + step_size * p
        # Half step in p
        with tf.GradientTape() as tape:
            tape.watch(q)
            U_new = -target_log_prob_fn(q)
        grad_U_new = tape.gradient(U_new, q)
        p = p - 0.5 * step_size * grad_U_new

        K = 0.5 * tf.reduce_sum(p ** 2)
        H = U_new + K
        trace.append({
            "k": k, "q": q.numpy().copy(), "p": p.numpy().copy(),
            "U": float(U_new.numpy()), "K": float(K.numpy()),
            "H": float(H.numpy()), "delta_H": float((H - H0).numpy()),
            "grad_norm": float(tf.norm(grad_U_new).numpy()),
        })

    return {"H0": float(H0.numpy()), "trace": trace,
            "q_final": q.numpy(), "p_final": p.numpy()}
```

Wrap this in a small CLI that loads the config, builds the filter and `target_log_prob_fn` exactly the way the runner does, draws `p0` from a fixed seed, and dumps the trace as JSON.

The key is reproducibility: the harness must produce the SAME (q, p, H) trajectory as the actual `tfp.mcmc.HamiltonianMonteCarlo` step would, given the same inputs. Verify this once on a tiny problem before trusting any of the diagnostics.

---

## Suggested running order

1. **Diagnostic 6** (~10 min, no new code) — figure out if the freeze is reproducible across seeds.
2. **Diagnostic 5** (~15 min, no new code) — sanity check that gradient descent works.
3. **Diagnostic 4** (~30 min, small standalone script) — measure target noise from particle filter Monte Carlo.
4. **Build the replay harness** (~1–2 h coding, then quick verification on a toy problem).
5. **Diagnostic 1, 2, 3** in sequence (~30 min total once the harness exists).

After running 1–4 you may already have the answer without needing the harness. If 5 (descent) works, 4 (PF noise) is small, and 6 (seed) shows systematic freeze, then the problem narrows to HMC dynamics → diagnostics 1/2/3 will pinpoint integration vs precision.

## Files involved

- `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2.yaml` — full config
- `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml` — fast smoke (currently 30+30 burn-in/sample, fixed step 0.02)
- `code/src/DF/hmc_runner.py` — HMC runner (where to wire instrumentation if needed)
- `code/src/resampling/ot_entropy.py` — implicit VJP (ridge fix already applied at the linear solve)
- `code/src/filters/particle/ledh_invertible_hmc.py` — filter that gets called
- `code/tests/hmc/test_gradient_vs_numerical_sv2d.py` — passing gradient validation reference
- `code/tests/hmc/test_ot_gradient_standalone.py` — passing standalone OT validation reference

## What NOT to do until diagnostics are run

- Do not change the HMC step size, leapfrog count, dtype, or num_particles "to see if it works." We've been guessing for hours. Measure first.
- Do not switch to NUTS or another sampler. Same dynamics under the hood.
- Do not add more workarounds to `ot_entropy.py`. It's been verified.
