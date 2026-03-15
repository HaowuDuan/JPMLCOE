# Auto Mass Vector: MAP Warm-Up + Hessian Estimation

## Motivation
HMC on range-bearing (sigma_range, sigma_bearing) converges to a wrong minimum for one
parameter because the posterior has highly asymmetric curvature between the two dimensions.
A diagonal mass matrix (`PreconditionedHamiltonianMonteCarlo`) compensates, but requires
manual tuning. `mass_vector: auto` removes the guesswork:
1. Short Adam warm-up → approximate MAP point
2. Diagonal Hessian of `-log p(θ|y)` at MAP → mass vector
3. Use mass vector for HMC + start chain from MAP (free warm-start bonus)

## Files to Modify
- `src/DF/hmc_runner.py`
- `src/experiments/run_dpf_experiment.py`
- `configs/dpf/hmc/range_bearing/bpf_ot.yaml`
- `configs/dpf/hmc/range_bearing/bpf_soft.yaml`

## Implementation

### 1. New method `DPFRunner._auto_estimate_mass(dtype, steps, lr)`

Add after `__init__` in `hmc_runner.py`:

```python
def _auto_estimate_mass(self, dtype, steps=50, lr=0.01):
    """Short Adam warm-up → diagonal Hessian at MAP → (mass_list, q_map)."""
    q = tf.Variable(self.param_handler.unconstrained_init, dtype=dtype)
    opt = tf.keras.optimizers.Adam(lr)
    print(f"  [auto mass] {steps}-step warm-up (lr={lr})...")
    for _ in range(steps):
        with tf.GradientTape() as tape:
            nlp = self._negative_log_posterior(q)
        opt.apply_gradients([(tape.gradient(nlp, q), q)])

    q_map = tf.constant(q.numpy(), dtype=dtype)
    constrained = self.param_handler.constrain(q_map)
    print(f"  [auto mass] approx MAP: { {n: round(float(v),4) for n,v in constrained.items()} }")

    # Diagonal Hessian of -log posterior at q_map via nested GradientTape
    with tf.GradientTape() as t2:
        t2.watch(q_map)
        with tf.GradientTape() as t1:
            t1.watch(q_map)
            nlp = self._negative_log_posterior(q_map)
        grad = t1.gradient(nlp, q_map)
    hessian = t2.jacobian(grad, q_map)          # (n_params, n_params)
    mass = tf.linalg.diag_part(hessian)         # diagonal only
    mass = tf.clip_by_value(mass, 0.1, 1000.0)  # guard against negatives / blow-up
    mass_list = [round(float(v), 4) for v in mass.numpy()]
    print(f"  [auto mass] estimated mass_vector={mass_list}")
    return mass_list, q_map
```

### 2. Modify `run_inference` to resolve `mass_vector == 'auto'`

Add `mass_auto_steps=50, mass_auto_lr=0.01` to `run_inference` signature.

Replace the existing `momentum_distribution` build block with:

```python
# Resolve mass vector — may require warm-up if 'auto'
if self.mass_vector == 'auto':
    resolved_mass, current_state = self._auto_estimate_mass(
        dtype, steps=mass_auto_steps, lr=mass_auto_lr
    )
elif self.mass_vector is not None:
    resolved_mass = self.mass_vector
    current_state = self.param_handler.unconstrained_init
else:
    resolved_mass = None
    current_state = self.param_handler.unconstrained_init

# Build momentum distribution (same logic, using resolved_mass)
momentum_distribution = None
if resolved_mass is not None:
    scale_diag = tf.constant([float(m)**0.5 for m in resolved_mass], dtype=dtype)
    momentum_distribution = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(len(resolved_mass), dtype=dtype), scale_diag=scale_diag)
    print(f"  mass_vector={resolved_mass}  (p ~ N(0, diag(m)))")
```

Note: `current_state` is now set here (replacing line 221 `current_state = self.param_handler.unconstrained_init`).

Also store `resolved_mass` in diagnostics via `_finalize` or directly in the result metadata.

### 3. `run_dpf_experiment.py` — pass new params

```python
result = runner.run_inference(
    ...
    mass_auto_steps=int(hmc_cfg.get('mass_auto_steps', 50)),
    mass_auto_lr=float(hmc_cfg.get('mass_auto_lr', 0.01)),
)
```

The `'auto'` string flows through `OmegaConf.to_container` unchanged — no other change needed.

### 4. YAML config update

```yaml
hmc:
  ...
  mass_vector: auto
  # mass_auto_steps: 50    # optional, default 50
  # mass_auto_lr: 0.01     # optional, default 0.01
```

## Verification
1. Run `python src/experiments/run_dpf_experiment.py +experiment=dpf/hmc/range_bearing/bpf_ot`
2. Check log for `[auto mass]` lines — MAP estimate should be near (0.1, 0.1)
3. Estimated mass_vector values should differ between params if curvatures differ
4. Both posterior means should converge near 0.1 in the final summary
5. Compare against `mass_vector: [1.0, 5.0]` run to validate improvement
