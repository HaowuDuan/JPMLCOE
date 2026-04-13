"""Training loop for ConditionalConvexPotentialMap.

Trains the neural operator (encoder + conditional map) to solve Monge-Ampere
on weighted particle clouds.
"""

import tensorflow as tf
import numpy as np
import os
import time

from encoder import WeightedDeepSetsEncoder
from conditional_map import ConditionalConvexPotentialMap
from kde import compute_bandwidth_scalar, sample_from_uniform_kde
from losses import (
    ma_residual_loss,
    local_linearity_penalty,
    eigenvalue_barrier_penalty,
)
from data import sample_random_cloud


class NeuralOperator(tf.keras.Model):
    """Wraps encoder + conditional map into one trainable model."""

    def __init__(self, in_dim, embed_dim, num_modules, context_dim,
                 encoder_hidden=(64, 64), alpha_init_bias=-5.0):
        super().__init__()
        self.encoder = WeightedDeepSetsEncoder(
            in_dim=in_dim, hidden_dims=encoder_hidden, context_dim=context_dim
        )
        self.trunk = ConditionalConvexPotentialMap(
            in_dim=in_dim, embed_dim=embed_dim, num_modules=num_modules,
            context_dim=context_dim, alpha_init_bias=alpha_init_bias,
        )

    def encode(self, particles, weights, h_kde=None):
        return self.encoder(particles, weights, h_kde=h_kde)

    def transport(self, x, c):
        """Apply transport map."""
        return self.trunk(x, c)

    def transport_with_jacobian(self, x, c):
        return self.trunk.forward_and_jacobian(x, c)


def train_step(model, particles, weights, h, query_points, optimizer,
               linearity_weight=0.1, eig_weight=0.0, eig_threshold=0.1,
               seed=None):
    """One training step on a single (particles, weights) example."""
    with tf.GradientTape() as tape:
        c = model.encode(particles, weights, h_kde=h)
        loss_ma, diagnostics = ma_residual_loss(
            model.trunk, particles, weights, h, query_points, c
        )
        if linearity_weight > 0:
            loss_lin = local_linearity_penalty(
                model.trunk, query_points, c, h, n_perturbations=2, seed=seed
            )
        else:
            loss_lin = tf.constant(0.0, dtype=loss_ma.dtype)
        if eig_weight > 0:
            loss_eig, min_eig_mean = eigenvalue_barrier_penalty(
                model.trunk, query_points, c, threshold=eig_threshold
            )
        else:
            loss_eig = tf.constant(0.0, dtype=loss_ma.dtype)
            min_eig_mean = tf.constant(0.0, dtype=loss_ma.dtype)
        loss = loss_ma + linearity_weight * loss_lin + eig_weight * loss_eig

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.zeros_like(v) if g is None else g
             for g, v in zip(grads, model.trainable_variables)]
    grads, grad_norm = tf.clip_by_global_norm(grads, 10.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Pure-diagnostic bandwidth ingredients (no effect on training): the
    # n_eff and weighted/unweighted spreads that go into Silverman's rule.
    n_eff = 1.0 / tf.reduce_sum(weights ** 2)
    mu_w = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered_w = particles - mu_w[None, :]
    cov_w = tf.einsum('n,ni,nj->ij', weights, centered_w, centered_w)
    weighted_std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov_w)))
    mu_u = tf.reduce_mean(particles, axis=0)
    centered_u = particles - mu_u[None, :]
    N_p = tf.cast(tf.shape(particles)[0], particles.dtype)
    cov_u = tf.matmul(centered_u, centered_u, transpose_a=True) / N_p
    unweighted_std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov_u)))

    diagnostics['loss'] = loss
    diagnostics['loss_ma'] = loss_ma
    diagnostics['loss_lin'] = loss_lin
    diagnostics['loss_eig'] = loss_eig
    diagnostics['min_eig_mean'] = min_eig_mean
    diagnostics['grad_norm'] = grad_norm
    diagnostics['h'] = h
    diagnostics['n_eff'] = n_eff
    diagnostics['weighted_std'] = weighted_std
    diagnostics['unweighted_std'] = unweighted_std
    return diagnostics


def _warmup_model_and_optimizer(model, optimizer, n_particles, d, h):
    """Build all Keras variables and optimizer slots before tf.function tracing.

    Also runs one eager gradient check to verify no variable has None gradient.
    """
    particles = tf.zeros((n_particles, d), dtype=tf.float64)
    weights = tf.ones((n_particles,), dtype=tf.float64) / float(n_particles)

    # Build encoder + trunk variables
    c = model.encode(particles, weights, h_kde=h)
    _ = model.transport_with_jacobian(particles, c)

    # Build optimizer slots
    optimizer.build(model.trainable_variables)

    # Eager gradient check
    q = sample_from_uniform_kde(particles, h, 16, tf.constant([0, 0], dtype=tf.int32))
    with tf.GradientTape() as tape:
        c = model.encode(particles, weights, h_kde=h)
        loss, _ = ma_residual_loss(model.trunk, particles, weights, h, q, c)
    grads = tape.gradient(loss, model.trainable_variables)
    missing = [v.name for g, v in zip(grads, model.trainable_variables) if g is None]
    if missing:
        raise RuntimeError(f"Missing gradients for: {missing}")


def _make_compiled_train_step(model, optimizer, linearity_weight, eig_weight,
                               eig_threshold):
    """Build a @tf.function-wrapped training step.

    Closes over model, optimizer, and Python config params so the compiled
    function takes only tensor inputs. Data sampling stays outside.
    """

    @tf.function(reduce_retracing=True)
    def _step(particles, weights, h, query_points, seed):
        with tf.GradientTape() as tape:
            c = model.encode(particles, weights, h_kde=h)
            loss_ma, diag = ma_residual_loss(
                model.trunk, particles, weights, h, query_points, c
            )
            if linearity_weight > 0:
                loss_lin = local_linearity_penalty(
                    model.trunk, query_points, c, h, n_perturbations=2, seed=seed
                )
            else:
                loss_lin = tf.constant(0.0, dtype=loss_ma.dtype)
            if eig_weight > 0:
                loss_eig, min_eig_mean = eigenvalue_barrier_penalty(
                    model.trunk, query_points, c, threshold=eig_threshold
                )
            else:
                loss_eig = tf.constant(0.0, dtype=loss_ma.dtype)
                min_eig_mean = tf.constant(0.0, dtype=loss_ma.dtype)
            loss = loss_ma + linearity_weight * loss_lin + eig_weight * loss_eig

        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, model.trainable_variables)]
        grads, grad_norm = tf.clip_by_global_norm(grads, 10.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Diagnostic bandwidth ingredients
        n_eff = 1.0 / tf.reduce_sum(weights ** 2)
        mu_w = tf.reduce_sum(weights[:, None] * particles, axis=0)
        centered_w = particles - mu_w[None, :]
        cov_w = tf.einsum('n,ni,nj->ij', weights, centered_w, centered_w)
        weighted_std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov_w)))
        mu_u = tf.reduce_mean(particles, axis=0)
        centered_u = particles - mu_u[None, :]
        N_p = tf.cast(tf.shape(particles)[0], particles.dtype)
        cov_u = tf.matmul(centered_u, centered_u, transpose_a=True) / N_p
        unweighted_std = tf.reduce_mean(tf.sqrt(tf.linalg.diag_part(cov_u)))

        return {
            'loss': loss,
            'loss_ma': loss_ma,
            'loss_lin': loss_lin,
            'loss_eig': loss_eig,
            'min_eig_mean': min_eig_mean,
            'grad_norm': grad_norm,
            'h': h,
            'n_eff': n_eff,
            'weighted_std': weighted_std,
            'unweighted_std': unweighted_std,
            'residual_abs_mean': diag['residual_abs_mean'],
            'log_det_J_mean': diag['log_det_J_mean'],
            'log_q_mean': diag['log_q_mean'],
            'log_p_T_mean': diag['log_p_T_mean'],
            'residual_mean': diag['residual_mean'],
        }

    return _step


def bandwidth_scale_schedule(step, num_steps, h_scale_init, h_scale_final):
    """Linear interpolation from h_scale_init at step 0 to h_scale_final at the end.

    Annealing the bandwidth downward provides continuation: start with smoother
    densities (easier PDE, better-conditioned gradients) and end with sharper
    densities closer to the empirical measure.
    """
    if num_steps <= 1:
        return h_scale_final
    frac = float(step) / float(num_steps - 1)
    return h_scale_init + (h_scale_final - h_scale_init) * frac


def train(model, num_steps, n_particles, d, q_points, batch_clouds=1,
          learning_rate=1e-3, linearity_weight=0.1,
          eig_weight=0.0, eig_threshold=0.1,
          h_scale_init=1.0, h_scale_final=1.0,
          log_every=100, base_seed=0,
          trace_csv_path=None,
          bandwidth_policy='resolution',
          use_jit=True):
    """Train the neural operator on synthetic clouds.

    Args:
        model: NeuralOperator
        num_steps: number of optimizer steps
        n_particles: particles per cloud
        d: state dimension
        q_points: number of query points per step
        batch_clouds: number of clouds per step (averaged)
        learning_rate: Adam LR
        linearity_weight: weight on local-linearity penalty
        eig_weight: weight on eigenvalue barrier on J (0 = off)
        eig_threshold: lambda_min threshold below which the barrier kicks in
        h_scale_init: bandwidth scale at step 0 (use >1 for warmup)
        h_scale_final: bandwidth scale at the last step
        log_every: print every N steps
        trace_csv_path: if set, write a per-step CSV trace of training
            diagnostics to this path. Pure logging — does not affect training.
        bandwidth_policy: 'resolution' (default, weight-independent
            spacing-based) or 'silverman' (classical density-estimation
            rule using unweighted source spread).
        use_jit: if True (default), use @tf.function graph-mode compilation
            for the training step. Set False for debugging.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = []

    # Build compiled training step if requested
    compiled_step = None
    if use_jit:
        h_warmup = tf.constant(0.25, dtype=tf.float64)
        _warmup_model_and_optimizer(model, optimizer, n_particles, d, h_warmup)
        compiled_step = _make_compiled_train_step(
            model, optimizer, linearity_weight, eig_weight, eig_threshold
        )

    csv_file = None
    csv_writer = None
    if trace_csv_path is not None:
        import csv
        os.makedirs(os.path.dirname(trace_csv_path) or '.', exist_ok=True)
        csv_file = open(trace_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'step', 'loss', 'loss_ma', 'grad_norm',
            'h', 'h_scale',
            'n_eff', 'weighted_std', 'unweighted_std',
            'residual_abs_mean', 'min_eig_mean', 'loss_eig',
        ])

    t_start = time.perf_counter()
    for step in range(num_steps):
        seed = tf.constant([base_seed + step, 0], dtype=tf.int32)
        cloud_seeds = tf.random.experimental.stateless_split(seed, num=batch_clouds + 1)

        h_scale = bandwidth_scale_schedule(
            step, num_steps, h_scale_init, h_scale_final
        )

        # Average diagnostics over batch_clouds
        step_metrics = {'h_scale': h_scale}
        for b in range(batch_clouds):
            particles, weights = sample_random_cloud(n_particles, d, cloud_seeds[b])
            h = compute_bandwidth_scalar(
                particles, weights,
                policy=bandwidth_policy,
                scale=h_scale,
            )
            query_seed = tf.random.experimental.stateless_split(cloud_seeds[b], num=2)[0]
            query_points = sample_from_uniform_kde(particles, h, q_points, query_seed)

            if compiled_step is not None:
                metrics = compiled_step(
                    particles, weights, h, query_points,
                    cloud_seeds[batch_clouds],
                )
            else:
                metrics = train_step(
                    model, particles, weights, h, query_points, optimizer,
                    linearity_weight=linearity_weight,
                    eig_weight=eig_weight,
                    eig_threshold=eig_threshold,
                    seed=cloud_seeds[batch_clouds],
                )
            for k, v in metrics.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + float(v.numpy()) / batch_clouds

        history.append(step_metrics)

        if csv_writer is not None:
            csv_writer.writerow([
                step,
                step_metrics['loss'],
                step_metrics['loss_ma'],
                step_metrics['grad_norm'],
                step_metrics['h'],
                step_metrics['h_scale'],
                step_metrics['n_eff'],
                step_metrics['weighted_std'],
                step_metrics['unweighted_std'],
                step_metrics['residual_abs_mean'],
                step_metrics['min_eig_mean'],
                step_metrics['loss_eig'],
            ])

        if step % log_every == 0 or step == num_steps - 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"  step {step:5d}/{num_steps} | "
                f"loss={step_metrics['loss']:.4e} | "
                f"ma={step_metrics['loss_ma']:.4e} | "
                f"lin={step_metrics['loss_lin']:.4e} | "
                f"eig={step_metrics['loss_eig']:.4e} | "
                f"min_eig={step_metrics['min_eig_mean']:.3f} | "
                f"h_scale={step_metrics['h_scale']:.3f} | "
                f"h={step_metrics['h']:.3f} | "
                f"|grad|={step_metrics['grad_norm']:.2f} | "
                f"residual_abs={step_metrics['residual_abs_mean']:.4f} | "
                f"{elapsed:.1f}s"
            )

    if csv_file is not None:
        csv_file.close()

    return history
