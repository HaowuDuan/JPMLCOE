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

## Status

Open. Will fix as part of training loop implementation (step 7-8 of plan).
