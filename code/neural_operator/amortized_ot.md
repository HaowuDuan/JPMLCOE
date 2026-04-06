# Amortized OT: Supervised Learning Approach (REFERENCE ONLY)

> **Status: Superseded.** This document describes a supervised approach (train on Sinkhorn labels). We are instead implementing the neural operator approach with KDE + physics-informed loss (no supervised learning). Kept as reference.

---

# Original: Neural Network Resampling Implementation Guide

**Target model:** Stochastic volatility, d=2, existing HMC+DPF-OT pipeline in TensorFlow.

**Goal:** Train a neural network to replace Sinkhorn in the resampling step. Single forward pass, differentiable, faster.

---

## 1. Training Data Collection

### What to save

At every Sinkhorn resampling step inside your existing DPF, extract and save:

```python
# Inside your resampling function, after Sinkhorn converges:
training_example = {
    'w_source': w0,          # shape (N,), source weights, on simplex
    'w_target': w1,          # shape (N,), target weights, on simplex
    'positions': X,          # shape (N, 2), particle positions
    'psi_star': psi_star,    # shape (N,), converged Sinkhorn dual variables
    'ess': ess_value,        # scalar, effective sample size (for stratification)
    'epsilon': epsilon,      # scalar, entropic regularization used
}
```

**Why dual variables instead of the transport plan:** The plan π is N×N — too large to store at scale and impossible at N=10⁶. The dual variables ψ* are N-dimensional and fully determine π via the softmax formula.

### How to save

Use TFRecord for efficient I/O:

```python
import tensorflow as tf

writer = tf.io.TFRecordWriter('resampling_data_run001.tfrecord')

def serialize_example(w0, w1, X, psi, ess, eps):
    feature = {
        'w_source': tf.train.Feature(float_list=tf.train.FloatList(value=w0.flatten())),
        'w_target': tf.train.Feature(float_list=tf.train.FloatList(value=w1.flatten())),
        'positions': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
        'psi_star': tf.train.Feature(float_list=tf.train.FloatList(value=psi.flatten())),
        'ess': tf.train.Feature(float_list=tf.train.FloatList(value=[ess])),
        'epsilon': tf.train.Feature(float_list=tf.train.FloatList(value=[eps])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
```

### How much data and from where

**Minimum:** 500 independent filtering trajectories × T time steps per trajectory. For T=100, that is 50,000 training examples. Start here.

**Diversity requirements:**

1. **Vary initial conditions.** Draw x₀ from the stationary distribution N(0, P). Different x₀ lead to different particle cloud geometries.

2. **Vary observation sequences.** Each trajectory gets its own random y_{1:T}. This is automatic if you simulate fresh data each run.

3. **Vary model parameters** (during data collection, not inference). Run the DPF with different values of A, Σ, b drawn from your prior. This gives weight distributions from both well-specified and misspecified models. Each (input, ψ*) pair is still a valid training example regardless of parameter quality.

4. **Stratify by ESS.** The transport problem varies most when weights are degenerate (low ESS). Ensure at least 20% of training data comes from ESS < 0.3 × N. If your filter rarely produces low ESS, artificially create degenerate examples by running with bad parameters or high observation noise.

**Diagnostic:** Plot histograms of ESS across training set. If it's concentrated at ESS ≈ N (nearly uniform weights), the training data is too easy and the network won't learn the hard cases.

### Data from inference runs (parameter learning)

When running HMC for parameter inference, the filter operates with different θ at each HMC iteration. Save resampling data from ALL iterations — early (bad parameters, degenerate weights) and late (good parameters, balanced weights). Every transport plan is correct for its specific input. The inference trajectory gives you free diversity.

---

## 2. Architecture

### Choice: Set Transformer

The input is a set of N particles with weights. Permutation equivariance matters: relabeling particles must relabel the output. A plain MLP treats input dimensions as ordered, so it must learn equivariance from data — wasteful.

Use a **Set Transformer** (Lee et al., 2019). It is a Transformer over set elements with permutation-equivariant self-attention. The architecture:

```
Input: (w_source_i, w_target_i, x_i) per particle → token of dim (2 + d) = 4
  ↓
[Linear projection to hidden dim h]
  ↓
[L layers of self-attention + feedforward]    ← this is the Set Transformer encoder
  ↓
Per-token output: ψ̂_i (one dual variable per particle)
```

Each token represents one particle. The self-attention lets particles exchange information (needed because ψ*_i depends on the global configuration of all particles). The output is per-token, giving ψ̂ ∈ ℝ^N.

### TensorFlow Implementation Skeleton

```python
import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.norm = layers.LayerNormalization()

    def call(self, x):
        attn_out = self.mha(x, x, x)
        return self.norm(x + attn_out)


class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(d_ff, activation='gelu')
        self.dense2 = layers.Dense(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        ff_out = self.dense2(self.dense1(x))
        return self.norm(x + ff_out)


class SetTransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

    def call(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


class ResamplingNetwork(tf.keras.Model):
    def __init__(self, d_model=64, num_heads=4, d_ff=128, num_layers=4, d_input=4):
        super().__init__()
        self.input_proj = layers.Dense(d_model)
        self.blocks = [
            SetTransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        self.output_proj = layers.Dense(1)  # one dual variable per token

    def call(self, tokens):
        """
        tokens: (batch, N, d_input) where d_input = 2 + d
                each token is [w_source_i, w_target_i, x_i1, x_i2]
        returns: (batch, N) predicted dual variables
        """
        x = self.input_proj(tokens)     # (batch, N, d_model)
        for block in self.blocks:
            x = block(x)                # (batch, N, d_model)
        psi = self.output_proj(x)       # (batch, N, 1)
        return tf.squeeze(psi, axis=-1) # (batch, N)
```

### Linear Attention Variant

Replace the standard MultiHeadAttention with linear attention for O(N) cost. In TensorFlow, you implement this manually:

```python
class LinearMultiHeadAttention(layers.Layer):
    """
    Linear attention: replace softmax(QK^T)V with phi(Q)(phi(K)^T V)
    Cost: O(N * d^2) instead of O(N^2 * d)
    """
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.wo = layers.Dense(d_model)

    def feature_map(self, x):
        """elu(x) + 1 feature map (Katharopoulos et al. 2020)"""
        return tf.nn.elu(x) + 1.0

    def call(self, x):
        B, N, _ = tf.shape(x)[0], tf.shape(x)[1], x.shape[-1]
        Q = tf.reshape(self.wq(x), [B, N, self.num_heads, self.d_head])
        K = tf.reshape(self.wk(x), [B, N, self.num_heads, self.d_head])
        V = tf.reshape(self.wv(x), [B, N, self.num_heads, self.d_head])

        Q = self.feature_map(Q)  # (B, N, H, D)
        K = self.feature_map(K)  # (B, N, H, D)

        # Key trick: compute K^T V first → (B, H, D, D), then Q @ (K^T V) → O(N*D^2)
        KV = tf.einsum('bnhd,bnhe->bhde', K, V)     # (B, H, D, D)
        QKV = tf.einsum('bnhd,bhde->bnhe', Q, KV)    # (B, N, H, D)

        # Normalize
        K_sum = tf.einsum('bnhd->bhd', K)             # (B, H, D)
        Z = tf.einsum('bnhd,bhd->bnh', Q, K_sum)      # (B, N, H)
        Z = tf.maximum(Z, 1e-6)
        QKV = QKV / Z[..., tf.newaxis]

        out = tf.reshape(QKV, [B, N, -1])
        return self.wo(out)
```

### Hyperparameters (starting point)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 64 | d=2 state, 4-dim tokens. 64 is more than enough. |
| num_heads | 4 | 16-dim per head, captures different interaction types |
| d_ff | 128 | 2× d_model, standard ratio |
| num_layers | 4 | Start here. Increase if underfitting. |
| N (particles) | 100–500 | For proof of concept. Fixed during training. |
| Batch size | 64 | Standard |
| Learning rate | 1e-4 | Adam, with cosine decay |

For N=100 with standard attention, the N×N attention matrix is 10K entries — trivially fast. Linear attention gives no speedup here. **Use standard attention for the POC. Switch to linear attention only when N > ~5000.**

---

## 3. Training

### Loss function

Primary loss: MSE on dual variables.

```python
loss = tf.reduce_mean(tf.square(psi_pred - psi_star))
```

Optional: marginal constraint regularizer. After reconstructing π from ψ̂, penalize marginal violations:

```python
def marginal_loss(psi_pred, w_source, w_target, positions, epsilon):
    # Compute cost matrix
    C = pairwise_squared_distance(positions, positions)  # (B, N, N)
    # Reconstruct soft plan from dual variables
    log_pi = (psi_pred[:, tf.newaxis, :] - C) / epsilon  # (B, N, N)
    log_pi = log_pi - tf.reduce_logsumexp(log_pi, axis=-1, keepdims=True)
    pi = tf.exp(log_pi) * w_source[:, :, tf.newaxis]
    # Marginal violations
    col_sums = tf.reduce_sum(pi, axis=1)  # should equal w_target
    return tf.reduce_mean(tf.square(col_sums - w_target))

total_loss = mse_loss + 0.1 * marginal_loss(...)
```

**Warning:** The marginal loss involves the N×N cost matrix. At large N, compute it on a subsample or drop it entirely — the MSE loss alone is sufficient if training data is accurate.

### Training loop

```python
model = ResamplingNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

@tf.function
def train_step(tokens, psi_star):
    with tf.GradientTape() as tape:
        psi_pred = model(tokens)
        loss = tf.reduce_mean(tf.square(psi_pred - psi_star))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training
for epoch in range(num_epochs):
    for batch in dataset:
        tokens = tf.concat([
            batch['w_source'][..., tf.newaxis],  # (B, N, 1)
            batch['w_target'][..., tf.newaxis],   # (B, N, 1)
            batch['positions'],                   # (B, N, 2)
        ], axis=-1)  # (B, N, 4)
        loss = train_step(tokens, batch['psi_star'])
```

### Evaluation metrics

1. **ψ error:** `mean |ψ̂ - ψ*| / |ψ*|` — relative error on dual variables.
2. **Warm-start iterations saved:** initialize Sinkhorn with ψ̂ instead of zeros. Count iterations to convergence (residual < tol). Compare against cold-start Sinkhorn.
3. **Filter RMSE:** run the full DPF with neural resampling. Compare state estimation error against DPF with Sinkhorn resampling. This is the metric that matters.
4. **Gradient check:** compute ∂(log-likelihood)/∂θ through the neural resampling. Compare against finite differences. Report relative error.

---

## 4. Integration Into the DPF Pipeline

### Warm-start mode (recommended first)

Don't replace Sinkhorn entirely. Use the network as a warm start:

```python
def neural_warm_start_resampling(w0, w1, X, model, epsilon, sinkhorn_fn, max_correction_steps=3):
    # Step 1: neural network prediction
    tokens = tf.concat([w0[..., None], w1[..., None], X], axis=-1)
    psi_init = model(tokens[tf.newaxis])[0]

    # Step 2: Newton/Sinkhorn correction
    psi_refined = sinkhorn_fn(w0, w1, X, epsilon, init=psi_init, max_iter=max_correction_steps)

    return psi_refined
```

This gives you correct gradients (through the Sinkhorn correction) while reducing iterations from ~30 to ~3.

### Full replacement mode (after validation)

Once the network is accurate enough (ψ error < 1%, filter RMSE matches baseline):

```python
def neural_resampling(w0, w1, X, model, epsilon):
    tokens = tf.concat([w0[..., None], w1[..., None], X], axis=-1)
    psi = model(tokens[tf.newaxis])[0]
    # Reconstruct soft assignment and resample
    C = pairwise_squared_distance(X, X)
    log_pi = (psi[tf.newaxis, :] - C) / epsilon
    pi = tf.nn.softmax(log_pi, axis=-1) * w0[:, tf.newaxis]
    x_new = tf.einsum('ij,jd->id', pi, X)
    return x_new
```

**Note:** In full replacement mode, gradients flow through the neural network, not through Sinkhorn. Gradient correctness depends entirely on network accuracy. Validate thoroughly before switching.

---

## 5. DAgger Bootstrapping

After training the first network:

### Round 2 data collection

```python
# Run DPF with neural warm-start resampling
# The Sinkhorn correction steps give you refined ψ* at the NEW input distribution
# Save these as new training data

new_data = run_dpf_with_neural_warmstart(
    model=trained_model_v1,
    correction_steps=3,
    save_resampling_data=True  # saves (input, ψ*_corrected) pairs
)

# Combine with round 1 data
combined_data = concat(round1_data, new_data)

# Retrain
model_v2 = train(combined_data)
```

### Stopping criterion

Compute validation loss (on held-out trajectories from the current round) after each DAgger round. Stop when validation loss decreases by less than 5% relative to the previous round. Typically 2–3 rounds suffice.

---

## 6. Checklist Before Starting

- [ ] Existing DPF pipeline runs correctly with Sinkhorn (verify on known parameters)
- [ ] Can extract (w0, w1, X, ψ*) from inside the Sinkhorn call
- [ ] TFRecord writer integrated into the resampling step
- [ ] 500+ trajectories worth of training data collected
- [ ] ESS histogram of training data covers the degenerate regime
- [ ] Set Transformer trains and converges on the data (MSE decreasing)
- [ ] Warm-start mode reduces Sinkhorn iterations (measure average and worst case)
- [ ] Filter RMSE with warm-start matches pure Sinkhorn baseline
- [ ] Gradient check passes (neural gradients vs. finite differences, relative error < 5%)
- [ ] DAgger round 2 collected and retrained (if validation loss improved > 5%)

---

## 7. Expected Timeline

| Phase | Task | Duration |
|-------|------|----------|
| Data | Instrument DPF, collect 500 trajectories | 1–2 days |
| Train v1 | Set Transformer, MSE loss, standard attention | 1–2 days |
| Integrate | Warm-start mode in DPF, verify filter accuracy | 1 day |
| Validate | Gradient checks, filter RMSE comparison | 1 day |
| Bootstrap | DAgger rounds 2–3 | 1–2 days |
| Full replacement | Switch from warm-start to direct neural resampling | 1 day |
| **Total** | | **~1.5–2 weeks** |
