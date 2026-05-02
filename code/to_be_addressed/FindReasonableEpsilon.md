# FindReasonableEpsilon — pre-warmup step-size search

Status: not implemented. Documented for future use.

## Problem this solves

TFP's `DualAveragingStepSizeAdaptation` (DA) cannot prevent step-size runaway when the chain starts in a region where every proposal is accepted. The DA exploratory update is

```
log eps_t = log(shrinkage) - sqrt(t) * error_sum / ((t + t0) * gamma)
```

At 100% accept, `error_sum` grows linearly negative every iteration. `shrinkage_target` only sets the *asymptote* (where DA settles after accept reaches target); it does nothing for this transient. DA grows step until proposals start being rejected — which happens once trajectories are large enough to push the chain into challenging territory. For LEDH+OT pipelines, "challenging territory" coincides with "the OT KKT matrix becomes singular and the backward MatrixSolve crashes."

Concrete example (RB + LEDH+OT, mass matrix [47.26, 82.09], step init 0.001):
```
step 1:  accept=100%, eps=0.0016
step 5:  accept=100%, eps=0.0401
step 10: accept=100%, eps=1.68
step 19: eps=3.31  --> OT MatrixSolve crash
```

## What FindReasonableEpsilon does

From Hoffman & Gelman (2014), Algorithm 4. Run BEFORE dual averaging starts.

```
1. Pick eps = step_size_init (small).
2. Sample momentum p ~ momentum_distribution.
3. Take ONE leapfrog step at (eps, num_leapfrog).
4. Compute log_alpha = log(p_proposal / p_current) = -delta_H.
5. If alpha > 0.5 (accept rate too high): eps *= 2; goto 3.
6. If alpha < 0.5 (accept rate too low):  eps /= 2; goto 3.
7. Stop when alpha crosses 0.5. Return eps.
```

The returned eps is approximately the largest step that gives ~50% accept on a single leapfrog. Hand it to DA as both `initial_step_size` AND `shrinkage_target`. DA then refines from a sane starting point and never has to discover the unstable boundary itself.

## Why this respects the project's hard constraints

- **Adaptive**, not a fixed step.
- **No manual cap**, just a search.
- **Uses standard HMC sampler** (no NUTS/PMMH switch).
- **Works WITH mass matrix**: search runs in the preconditioned space, so the eps it finds is correct for the mass.
- **target_accept_prob unchanged** for the main DA phase.

## Implementation outline (TFP, ~30-50 lines in `hmc_runner.py`)

```python
def find_reasonable_epsilon(
    target_log_prob_fn,
    state,                  # initial unconstrained params
    momentum_distribution,  # honors mass matrix preconditioning
    num_leapfrog_steps,
    seed,
    eps_init=1e-3,
    target_alpha=0.5,
    max_iters=50,
):
    """Hoffman-Gelman reasonable-epsilon search.

    Returns an eps such that one leapfrog at (eps, num_leapfrog) has
    accept-prob ~ target_alpha. Used to seed DA.
    """
    eps = tf.constant(eps_init, dtype=state.dtype)
    log_target = target_log_prob_fn(state)

    def one_proposal(eps_):
        p = momentum_distribution.sample(seed=seed)
        # Single leapfrog trajectory at (eps_, num_leapfrog)
        # Use TFP's leapfrog_integrator directly, NOT the full HMC kernel
        # so we get the proposed state without Metropolis acceptance.
        # ... (see tfp.mcmc.internal.leapfrog_integrator.SimpleLeapfrogIntegrator)
        new_state, new_p, _ = integrator(p, state, log_target)
        log_alpha = (target_log_prob_fn(new_state)
                     - 0.5 * sum_squares(new_p)
                     - log_target
                     + 0.5 * sum_squares(p))
        return float(tf.math.exp(log_alpha).numpy())

    alpha = one_proposal(eps)
    direction = 1 if alpha > target_alpha else -1
    while (direction == 1  and alpha > target_alpha) \
       or (direction == -1 and alpha < target_alpha):
        eps = eps * (2.0 ** direction)
        alpha = one_proposal(eps)
        if iters >= max_iters:
            break
    return eps
```

Wire-up in `run_inference` (after building inner kernel, before DA):

```python
if find_reasonable_epsilon_enabled:
    eps_seed = find_reasonable_epsilon(
        target_log_prob_fn, current_state, momentum_distribution,
        num_leapfrog_steps, seed=tf.constant([seed, 0]),
        eps_init=step_size,
    )
    print(f"  [find_reasonable_epsilon] seed eps = {float(eps_seed.numpy()):.4f}")
    step_size = eps_seed
    # shrinkage_target follows step_size below; no separate change needed
```

Behind a config flag:
```yaml
dpf:
  hmc:
    find_reasonable_epsilon: true   # default false to preserve existing behavior
```

## Practical notes

- **Cost**: ~5–20 leapfrog evaluations as warmup overhead. On CUDA RB (~6s/leapfrog at N=500, T=50, n_lambda=15) that's ~30–120s. Negligible vs hours-long chains.
- **Mass matrix interaction**: the search must use the same `momentum_distribution` as the main HMC. If mass matrix is `M`, momentum is sampled from `N(0, M)`, and the search finds eps that matches M.
- **Target alpha**: 0.5 is the Hoffman-Gelman default. For LEDH+OT, slightly lower (e.g. 0.4) might be safer — gives a smaller starting eps so DA has room to grow if needed.
- **Stochastic target**: PF likelihood is noisy. The search should freeze the PF seed (use the same seed as the chain's first iteration) so the search and the main run see the same surface.
- **Doesn't fix gradient bias.** This addresses step-size runaway only. If the underlying gradient is biased (e.g. sign-flipped from OT backward at near-singular K), no amount of step-size tuning helps.

## Failure modes / things to verify

- If the surface really is flat over orders of magnitude in eps, the search can blow up to huge eps before alpha drops. Cap with `max_iters`.
- If `eps_init` is so small that even one leapfrog gives alpha=1.0 due to PF noise, the search will keep doubling. Use the chain's `step_size` from config as `eps_init`.
- If mass matrix or particle filter changes between this pre-warmup and the main loop, the eps becomes stale. Run the search after all setup is complete.

## When to prefer this over `step_count_smoothing=100` (the bandaid)

| Situation | Bandaid (t0=100) | FindReasonableEpsilon |
|---|---|---|
| Quick experiment, eps_init already in safe ballpark | ✓ | overkill |
| Mass matrix change → unknown new optimal eps | might still crash | ✓ |
| Production chain you want to converge cleanly | unreliable | ✓ |
| Multi-chain comparison where chains start at different inits | per-chain eps differs awkwardly | ✓ each chain finds its own eps |

## References

- Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR 15. Algorithm 4 (FindReasonableEpsilon) and Algorithm 6 (NUTS w/ DA).
- Stan reference manual, HMC algorithm parameters: https://mc-stan.org/docs/reference-manual/hmc.html (Stan implements this internally before DA).
- TFP source: `tensorflow_probability/python/mcmc/dual_averaging_step_size_adaptation.py` (the formulas this doc references).
