# Issue 1: Residual Initialization Far From Identity

## Observation

`ConvexPotentialMap` with `residual=True` should produce `T(x) ≈ x` at initialization, since uniform-weighted resampling should be close to a no-op when source equals target.

Current state from `test_residual_near_identity_at_init`:
```
mean |T(x) - x| = 4.83
```

The residual term is large at random init, not small.

## Cause

Each module is multiplied by `softplus(alpha_m)`. With `alpha ~ Uniform(0, 1)`, `softplus(alpha) ~ 0.7-1.3` — not small. Each of the 8 modules contributes a non-trivial amount, accumulated into a large total.

```python
# In ConvexPotentialMap.build():
self.alpha = self.add_weight(
    name='alpha', shape=(self.num_modules,),
    initializer=tf.keras.initializers.RandomUniform(0, 1),  # Should be very negative
    trainable=True,
)
```

## Fix

Initialize `alpha` to a very negative value (e.g. -5 to -10) so `softplus(alpha) ≈ 0` at init. Then `T(x) ≈ x` initially, and training learns the deviation from identity.

```python
# Proposed:
initializer=tf.keras.initializers.Constant(-5.0),
```

Same for `beta` inside each module — should start small so the activation argument is near 0.

Same for `out_bias` — already initialized to zeros, good.

## When to address

Before training. The initialization affects training dynamics significantly:
- Bad init: model needs to "unlearn" the random residual before learning the actual map
- Good init: model starts at identity and only learns the deviation

## Tanh saturation and cloud normalization

`tanh` has a fixed point at zero: `tanh(0) = 0`, `tanh'(0) = 1`. Near
zero the activation acts like the identity and has full gradient. As
the input grows, `tanh` saturates at ±1 and `tanh'(·) → 0`.

Each module computes `tanh(softplus(β) · (W x + b))`. The pre-activation
magnitude is `O(softplus(β) · ‖W‖ · ‖x‖)`. With Glorot-initialized `W`
(entries ~O(1)), `softplus(β)` of order 1, and unnormalized clouds where
`‖x‖` can be 5–10 or more, the pre-activation is large and `tanh`
saturates. When `tanh'(·) ≈ 0`, the module Jacobian
`W^T diag(tanh'(·)) W ≈ 0` — the module contributes nothing to the map
and receives no useful gradient.

**Implication**: if training clouds are not normalized, modules can
silently turn off because the pre-activations push tanh into saturation.
Cloud normalization (subtract mean, divide by std before feeding to the
operator) keeps `‖x_hat‖ ~ O(1)`, keeps pre-activations near zero,
keeps tanh in its linear regime, and keeps all modules active.

This is a second reason (beyond bandwidth stability) that cloud
normalization is important for training. The issue was identified
2026-04-12 but not saved until now.

## Status

The alpha-init part is fixed (alpha_init_bias=-3 or -5 in the
conditional map). The tanh saturation / cloud normalization point is
addressed by normalizing training clouds (see training distribution
fixes in 2_quality_test.md §7d).
