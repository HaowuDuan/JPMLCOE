"""Local Exact Daum-Huang (LEDH) Invertible Particle Flow Filter. Algorithm 1"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.batched_ekf import batched_ekf_predict, batched_ekf_update
from ...utils.flow_params import compute_flow_params
from ...utils.distributions import compute_flow_weights
from ...utils.ode_solvers import euler_step
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample
from ...resampling.diagnosis import effective_sample_size as ess_tf


class LEDHParticleFlowFilter:
    """
    Local Exact Daum-Huang (LEDH) Invertible Particle Flow Filter - Algorithm 1.

    Uses per-particle local linearization with batched EKF for covariance tracking.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 29,
        regularization: float = 1e-8,
        resample_threshold: float = 0.5,
        resampling_method: Optional[Callable] = None,
        resampling_config: Optional[Dict[str, Any]] = None,
        debug_mode: bool = False,
        **filter_kwargs
    ):
        """
        Args:
            model: StateSpaceModel
            n_particles: Number of particles
            n_lambda_steps: Number of flow integration steps
            regularization: Regularization strength for numerical stability (default: 1e-8)
            resample_threshold: Resample when ESS/N < threshold
            resampling_method: Resampling function (systematic/soft/ot_entropy)
            resampling_config: Dict of additional parameters for resampling
            debug_mode: If True, store detailed diagnostics
        """
        self.model = model
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.regularization = regularization
        self.resample_threshold = resample_threshold
        self.debug_mode = debug_mode

        # Handle resampling method configuration
        if isinstance(resampling_method, str):
            method_map = {
                'systematic': systematic_resample,
                'soft': soft_resample,
                'ot_entropy': ot_entropy_resample,
            }
            self.resampling_method = method_map.get(resampling_method, systematic_resample)
            self.resampling_method_name = resampling_method
        elif resampling_method is not None:
            self.resampling_method = resampling_method
            self.resampling_method_name = getattr(resampling_method, '__name__', 'custom')
        else:
            self.resampling_method = systematic_resample
            self.resampling_method_name = 'systematic'

        # Convert resampling config values to Python scalars
        self.resampling_config = {}
        if resampling_config is not None:
            for key, value in resampling_config.items():
                if isinstance(value, (int, np.integer)):
                    self.resampling_config[key] = int(value)
                elif isinstance(value, (float, np.floating)):
                    self.resampling_config[key] = float(value)
                else:
                    self.resampling_config[key] = value

        # Particles and weights
        self.particles = None
        self.weights = None
        self.particles_prev = None
        self.eta_0 = None
        self.eta_bar_0 = None

        # Per-particle covariances (managed via batched EKF)
        self.particle_covs = None

        # Storage
        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

        # Random seed counter
        self.seed_counter = 0

        # Debug storage
        if self.debug_mode:
            self.debug_info = {
                'timesteps': [],
                'flow_steps': [],
                'particles_before_flow': [],
                'particles_after_flow': [],
                'A_matrices': [],
                'b_vectors': [],
                'H_jacobians': [],
                'jacobian_dets': [],
                'eigenvalues': [],
                'condition_numbers': [],
                'particle_stats': []
            }
        else:
            self.debug_info = None

        self._generate_lambda_steps()

    def _generate_lambda_steps(self):
        """Generate exponentially spaced step sizes as TF tensor."""
        q = 1.2
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        lambda_steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = tf.constant(lambda_steps_np, dtype=tf.float32)

    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_seed: Optional[int] = None):
        """Initialize particles and per-particle filters."""
        if random_seed is not None:
            self.seed_counter = random_seed

        if initial_mean is None:
            # Use model's initial mean directly (not a random sample)
            initial_mean = np.asarray(self.model.mu_0, dtype=np.float32)

        if initial_cov is None:
            if hasattr(self.model, 'Sigma_0'):
                initial_cov = np.asarray(self.model.Sigma_0)
            elif hasattr(self.model, 'stationary_var'):
                initial_cov = np.eye(self.state_dim) * self.model.stationary_var
            else:
                initial_cov = np.eye(self.state_dim)

        # Sample initial particles using TensorFlow
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1
        initial_mean_tf = tf.constant(initial_mean, dtype=tf.float32)
        initial_cov_tf = tf.constant(initial_cov, dtype=tf.float32)

        L = tf.linalg.cholesky(initial_cov_tf)
        z = tf.random.stateless_normal([self.n_particles, self.state_dim], seed=seed, dtype=tf.float32)
        particles_tf = initial_mean_tf + tf.linalg.matmul(z, L, transpose_b=True)

        self.particles = tf.Variable(particles_tf, dtype=tf.float32)
        self.weights = tf.Variable(tf.ones(self.n_particles, dtype=tf.float32) / self.n_particles)

        # Per-particle covariances for batched EKF (TF Variable)
        self.particle_covs = tf.Variable(
            tf.tile(tf.expand_dims(initial_cov_tf, 0), [self.n_particles, 1, 1]),
            dtype=tf.float32
        )

        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    @tf.function
    def _compute_drift_single(self, x: tf.Tensor, A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute drift for a single state vector: dx/dλ = Ax + b."""
        return tf.linalg.matvec(A, x) + b

    def predict(self):
        """Prediction step with batched EKF (all TF ops)."""
        self.particles_prev = tf.Variable(self.particles.value(), dtype=tf.float32)

        # Batched EKF predict - single tf.function call (all TF tensors)
        eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
            self.model, self.particles.value(), self.particle_covs.value()
        )
        self.particle_covs.assign(cov_pred_tf)

        self.eta_bar_0 = tf.Variable(eta_bar_0_tf, dtype=tf.float32)

        # Stochastic prediction - use batch method
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1
        eta_0_tf = self.model.state_transition_batch(self.particles_prev.value(), seed)
        self.eta_0 = tf.Variable(eta_0_tf, dtype=tf.float32)

    def _update_single_particle(self, i: tf.Tensor, eta_bar_current: tf.Tensor,
                               eta_1_current: tf.Tensor, particle_covs_tf: tf.Tensor,
                               eta_bar_0_tf: tf.Tensor, y_tf: tf.Tensor,
                               lambda_val_tf: tf.Tensor, d_lambda_tf: tf.Tensor,
                               R: tf.Tensor, R_inv: tf.Tensor,
                               regularization_tf: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Update a single particle using local flow parameters.

        Returns:
            eta_bar_new: Updated eta_bar for particle i
            eta_1_new: Updated eta_1 for particle i
            log_det_M: Log determinant of Jacobian for particle i
        """
        P_i = particle_covs_tf[i]

        # Compute flow parameters - LEDH uses individual particle as linearization point
        A_i, b_i = compute_flow_params(
            self.model,
            eta_bar_current[i],
            lambda_val_tf,
            y_tf,
            P_i,
            R,
            R_inv,
            eta_bar_0_tf[i],
            self.state_dim,
            regularization_tf
        )

        # Migrate both streams
        eta_bar_i_new = euler_step(eta_bar_current[i], self._compute_drift_single, d_lambda_tf, A_i, b_i)
        eta_1_i_new = euler_step(eta_1_current[i], self._compute_drift_single, d_lambda_tf, A_i, b_i)

        # Track Jacobian determinant
        M_i = tf.eye(self.state_dim, dtype=tf.float32) + d_lambda_tf * A_i
        log_det_M_i = tf.math.log(tf.abs(tf.linalg.det(M_i)))

        return eta_bar_i_new, eta_1_i_new, log_det_M_i

    def update(self, y: tf.Tensor):
        """Update step with per-particle local flow (all TF ops in hot path)."""
        R = self.model.observation_noise_cov

        eta_1 = self.eta_0.value()
        eta_bar = self.eta_bar_0.value()

        lambda_val = tf.constant(0.0, dtype=tf.float32)
        log_theta = tf.zeros(self.n_particles, dtype=tf.float32)

        # Compute R_inv once (shared across all particles)
        R_inv = tf.linalg.inv(R)
        regularization_tf = tf.constant(self.regularization, dtype=tf.float32)

        # Use TF tensors directly (particle_covs is TF Variable, eta_bar_0 is TF Variable)
        particle_covs_tf = self.particle_covs.value()
        eta_bar_0_tf = self.eta_bar_0.value()

        # Flow loop (all TF tensor operations — no per-step tf.constant calls)
        for j in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[j]
            lambda_val = lambda_val + d_lambda

            # eta_bar and eta_1 are already TF tensors (from .value() or map_fn output)
            eta_bar_current = eta_bar
            eta_1_current = eta_1

            # Process all particles using tf.map_fn
            def process_particle(i):
                return self._update_single_particle(
                    i, eta_bar_current, eta_1_current, particle_covs_tf, eta_bar_0_tf,
                    y, lambda_val, d_lambda, R, R_inv, regularization_tf
                )

            results = tf.map_fn(
                process_particle,
                tf.range(self.n_particles),
                fn_output_signature=(
                    tf.TensorSpec([self.state_dim], tf.float32),  # eta_bar_new
                    tf.TensorSpec([self.state_dim], tf.float32),  # eta_1_new
                    tf.TensorSpec([], tf.float32)                   # log_det_M
                ),
                parallel_iterations=10
            )

            eta_bar = results[0]
            eta_1 = results[1]
            log_theta = log_theta + results[2]

        # Normalize Jacobians for numerical stability
        max_log_theta = tf.reduce_max(log_theta)
        log_theta = log_theta - max_log_theta
        theta = tf.exp(log_theta)

        self.particles.assign(eta_1)

        # Compute weights using shared utility (with Jacobians for LEDH)
        weights_new = compute_flow_weights(
            eta_1=eta_1,
            eta_0=self.eta_0.value(),
            particles_prev=self.particles_prev.value(),
            observation=y,
            model=self.model,
            prev_weights=self.weights.value(),
            jacobians=theta,
            clip_range=None  # No clipping - use max-normalization only (matches MATLAB)
        )
        self.weights.assign(weights_new)

        # Store TF tensors — convert to numpy once in filter()
        self.weights_history.append(self.weights.value())

        # Log-likelihood using batch method
        log_likelihood = self.model.log_observation_prob_batch(y, eta_1)

        max_ll = tf.reduce_max(log_likelihood)
        log_lik = max_ll + tf.math.log(tf.reduce_mean(tf.exp(log_likelihood - max_ll)))
        self.log_likelihoods.append(log_lik)

        # Update per-particle covariances via batched EKF (all TF tensors)
        _, cov_updated = batched_ekf_update(
            self.model, self.eta_bar_0.value(), self.particle_covs.value(), y
        )
        self.particle_covs.assign(cov_updated)

        # ESS and resampling
        ess = ess_tf(self.weights.value())
        self.ess_history.append(ess)

        if ess < self.resample_threshold * self.n_particles:
            self._resample()
            self.resampled_at.append(len(self.ess_history) - 1)
            # Count unique particles (numpy needed for np.unique)
            particles_np = self.particles.numpy()
            n_unique = len(np.unique(particles_np, axis=0))
            self.n_unique_particles.append(n_unique)

    def _resample(self):
        """Resample particles and per-particle filters (TF distance computation)."""
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1

        result = self.resampling_method(
            self.particles.value(),
            self.weights.value(),
            seed=seed,
            **self.resampling_config
        )

        # Handle different return types
        if isinstance(result, tuple):
            resampled_particles, new_weights = result
        else:
            resampled_particles = result
            new_weights = tf.ones(self.n_particles, dtype=tf.float32) / tf.cast(self.n_particles, tf.float32)

        # Match resampled particles to original indices using TF distance computation
        # dists[i, j] = ||resampled[i] - original[j]||^2
        dists = tf.reduce_sum(
            (resampled_particles[:, tf.newaxis, :] - self.particles.value()[tf.newaxis, :, :]) ** 2,
            axis=2
        )
        indices = tf.argmin(dists, axis=1)

        # Resample covariances using TF gather
        self.particle_covs.assign(tf.gather(self.particle_covs.value(), indices))

        self.particles.assign(resampled_particles)
        self.weights.assign(new_weights)

    def _estimate_mean_cov(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Estimate weighted mean and covariance (TF ops)."""
        particles = self.particles.value()
        weights = self.weights.value()

        mean = tf.reduce_sum(weights[:, tf.newaxis] * particles, axis=0)
        diff = particles - mean
        cov = tf.reduce_sum(
            weights[:, tf.newaxis, tf.newaxis] *
            tf.einsum('ij,ik->ijk', diff, diff),
            axis=0
        )
        return mean, cov

    def filter(self, observations: np.ndarray,
               initial_mean: Optional[np.ndarray] = None,
               initial_cov: Optional[np.ndarray] = None,
               random_seed: Optional[int] = None) -> FilterResult:
        """Run filter on sequence of observations."""
        self.initialize(initial_mean, initial_cov, random_seed)
        T = len(observations)

        # Pre-convert observations to TF once
        obs_tf = tf.constant(observations, dtype=tf.float32)

        for t in range(T):
            self.predict()
            self.update(obs_tf[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)  # TF tensor
            self.covs.append(cov)    # TF tensor

        resampling_rate = len(self.resampled_at) / T if T > 0 else 0.0

        # Convert accumulated TF tensors to numpy once
        means_np = tf.stack(self.means).numpy()
        covs_np = tf.stack(self.covs).numpy()
        log_liks_tf = tf.stack(self.log_likelihoods) if self.log_likelihoods else None
        ess_np = tf.stack(self.ess_history).numpy()
        weights_np = tf.stack(self.weights_history).numpy()

        return FilterResult(
            means=means_np,
            covs=covs_np,
            log_likelihood=float(tf.reduce_sum(log_liks_tf).numpy()) if log_liks_tf is not None else None,
            log_likelihoods=log_liks_tf.numpy() if log_liks_tf is not None else None,
            ess=ess_np,
            weights_history=weights_np,
            resampled_at=self.resampled_at,
            n_unique=np.array(self.n_unique_particles) if self.n_unique_particles else None,
            metadata={
                'filter_type': 'LEDHParticleFlowFilter',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_method': self.resampling_method_name,
                'resampling_rate': resampling_rate
            }
        )
