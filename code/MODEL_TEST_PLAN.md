# Model Test Plan

One test file: `tests/test_models.py`

## Test 1: Jacobian correctness (finite difference check)

For every model with analytical Jacobians, compare against numerical finite differences.
This is the highest-value test. Hand-derived Jacobians are the #1 source of silent model bugs.

Check `state_jacobian(x)` against finite-diff of `state_transition_mean(x)`.
Check `observation_jacobian(x)` against finite-diff of `observation_mean(x)`.

Models to test:
- kitagawa: nonlinear f AND nonlinear h
- range_bearing: linear f (skip), nonlinear h (range + bearing from position)
- stochastic_volatility: linear f (skip), h is zero (skip — but verify Jacobian is zero)
- acoustic_tracking: linear f (skip), nonlinear h (amplitude from distance)
- acoustic_tracking_full: linear f (skip), nonlinear h (amplitude from distance, 16D state)
- two_sensor_bearing: linear f (skip), nonlinear h (two bearing angles)
- lorenz96: nonlinear f (RK4 Lorenz equations), linear h (skip)
- linear_gaussian: both linear — skip entirely, Jacobian is just F and H by definition

Test at multiple points (not just zero — Jacobians can be correct at x=0 and wrong elsewhere).

## Test 2: Q and R are symmetric positive-definite

One-liner per model. Catches copy-paste errors in noise specification.

## Test 3: Dimension consistency

- `state_transition_mean(x)` output length == `state_dim`
- `observation_mean(x)` output length == `obs_dim`
- `state_jacobian(x)` shape == `(state_dim, state_dim)`
- `observation_jacobian(x)` shape == `(obs_dim, state_dim)`

Not testing Python — testing that the model declaration matches its functions.

## Non-tests (deliberately excluded)

- `sample_state` / `sample_observation`: just f(x) + noise. If f is correct, these are correct.
- Parameter storage: testing that Python stores what you gave it.
- Data generation pipeline: if the experiment runs, it works.
