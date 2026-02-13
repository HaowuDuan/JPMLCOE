"""Exact Daum-Huang (EDH) Invertible Particle Flow Filter
"""

import tensorflow as tf
import numpy as np
import time
from typing import Tuple, Optional, Callable, Dict, Any
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ...utils.flow_params import compute_flow_params
from ...utils.distributions import compute_flow_weights
from ...utils.ode_solvers import euler_step
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample
from ...resampling.diagnosis import effective_sample_size as ess_tf


class EDHParticleFlowFilter:
    """
    Exact Daum-Huang (EDH) Particle Flow Filter (Invertible Variant).

    Implements Algorithm 2 from the paper with explicit Euler integration
    and the author's recommended exponential step sizes.
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_particles: int = 1000,
        n_lambda_steps: int = 29,  # Author's recommended value
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
            resample_threshold: Resample when ESS/N < threshold
            resampling_method: Resampling function. Options:
                              - systematic_resample (default)
                              - soft_resample
                              - ot_entropy_resample
                              - Any custom callable with signature (particles, weights, seed, **kwargs)
            resampling_config: Dict of additional parameters for resampling method
            debug_mode: If True, store detailed diagnostics
        """
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
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
            # Default to systematic
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

        # Particles and weights (TensorFlow variables)
        self.particles = None
        self.weights = None
        self.particles_prev = None
        self.eta_0 = None

        # Global filter for covariance estimation
        self.global_filter = None
        self.predicted_cov = None  # P_{k|k-1} from EKF/UKF
        self.eta_bar_0 = None      # η̄_0 (mean at λ=0)

        # Cache R_inv if R is constant
        self.R_inv_cache = None
        self.L_cache = None

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

        # Precompute exponentially spaced step sizes (constant across timesteps)
        self._generate_lambda_steps()

    def _create_filter(self, initial_mean: tf.Tensor, initial_cov: tf.Tensor):
        """Create EKF for global covariance guidance."""
        initial_mean_np = initial_mean.numpy() if isinstance(initial_mean, tf.Tensor) else initial_mean
        initial_cov_np = initial_cov.numpy() if isinstance(initial_cov, tf.Tensor) else initial_cov

        return ExtendedKalmanFilter(self.model, mean_0=initial_mean_np, Sigma_0=initial_cov_np,
                                    sample_initial_mean=False)

    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_seed: Optional[int] = None):
        """Initialize particles and global filter."""
        if random_seed is not None:
            self.seed_counter = random_seed

        if initial_mean is None:
            # Use model's initial mean directly (not a random sample)
            initial_mean = np.asarray(self.model.mu_0, dtype=self.np_dtype)

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
        initial_mean_tf = tf.constant(initial_mean, dtype=self.dtype)
        initial_cov_tf = tf.constant(initial_cov, dtype=self.dtype)

        L = tf.linalg.cholesky(initial_cov_tf)
        z = tf.random.stateless_normal([self.n_particles, self.state_dim], seed=seed, dtype=self.dtype)
        particles_tf = initial_mean_tf + tf.linalg.matmul(z, L, transpose_b=True)

        self.particles = tf.Variable(particles_tf, dtype=self.dtype)
        self.weights = tf.Variable(tf.ones(self.n_particles, dtype=self.dtype) / self.n_particles)

        self.global_filter = self._create_filter(initial_mean_tf, initial_cov_tf)

        # Pre-allocate Variables used in predict() to avoid per-timestep allocation
        self.particles_prev = tf.Variable(tf.zeros_like(particles_tf), dtype=self.dtype)
        self.eta_0 = tf.Variable(tf.zeros([self.n_particles, self.state_dim], dtype=self.dtype))

        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def _generate_lambda_steps(self):
        """Precompute exponentially spaced step sizes as TF tensor (called once in __init__)."""
        q = 1.2
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        lambda_steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = tf.constant(lambda_steps_np, dtype=self.dtype)

    @tf.function
    def _compute_drift(self, particles: tf.Tensor, A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute drift for EDH flow: dη/dλ = Aη + b."""
        return tf.linalg.matvec(A, particles) + b

    @tf.function
    def _compute_drift_batch(self, particles: tf.Tensor, A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute drift for batch of particles: dη/dλ = Aη + b (vectorized)."""
        return particles @ tf.transpose(A) + b

    def predict(self):
        """Prediction step (Algorithm 2, lines 4-8)."""
        # Store previous particles for weight calculation
        self.particles_prev.assign(self.particles.value())

        # Line 4: EKF/UKF prediction to get P_{k|k-1}
        # Feed back WEIGHTED ensemble mean to global filter (all TF ops)
        ensemble_mean_prev = tf.reduce_sum(
            self.weights.value()[:, tf.newaxis] * self.particles.value(), axis=0
        )

        self.global_filter.mean.assign(ensemble_mean_prev)
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

        # Store the predicted mean BEFORE particles are propagated
        self.eta_bar_0 = self.global_filter.mean.value()  # TF tensor

        # Lines 5-8: Propagate particles using model's batch method
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1

        eta_0 = self.model.state_transition_batch(self.particles.value(), seed)

        self.particles.assign(eta_0)
        self.eta_0.assign(eta_0)

    def update(self, y: tf.Tensor):
        """Update step (Algorithm 2, lines 9-29)."""
        # Cache R_inv
        R = self.model.observation_noise_cov
        if self.R_inv_cache is None:
            L = tf.linalg.cholesky(R)
            self.R_inv_cache = tf.linalg.inv(tf.transpose(L)) @ tf.linalg.inv(L)
            self.L_cache = L

        # Line 9: Set η̄ = η̄_0 (plain tensor, reassigned each step)
        eta_bar = self.eta_bar_0

        # Line 10: λ = 0
        lambda_val = tf.constant(0.0, dtype=self.dtype)

        # Lines 11-18: Flow loop (all TF tensor operations)
        for j in tf.range(self.n_lambda_steps):
            epsilon_j = self.lambda_steps[j]

            # Line 12: λ = λ + ε_j
            lambda_val = lambda_val + epsilon_j

            # Line 13: Compute A(λ), b(λ)
            # EDH uses ensemble mean as linearization point, no regularization
            A, b = compute_flow_params(
                self.model,
                eta_bar,
                lambda_val,
                y,
                self.predicted_cov,
                R,
                self.R_inv_cache,
                self.eta_bar_0,
                self.state_dim
            )

            # Line 14: Migrate η̄ (epsilon_j is TF scalar, euler_step accepts it)
            eta_bar = euler_step(eta_bar, self._compute_drift, epsilon_j, A, b)

            # Lines 15-17: Migrate particles
            self.particles.assign(euler_step(self.particles.value(), self._compute_drift_batch, epsilon_j, A, b))

        # Lines 19-25: Weight update using shared utility
        eta_1 = self.particles.value()

        # Compute weights using shared utility (already TensorFlow compatible)
        weights_new = compute_flow_weights(
            eta_1=eta_1,
            eta_0=self.eta_0.value(),
            particles_prev=self.particles_prev.value(),
            observation=y,
            model=self.model,
            prev_weights=self.weights.value(),
            jacobians=None,
            clip_range=None  # No clipping - use max-normalization only (matches MATLAB)
        )
        self.weights.assign(weights_new)

        # Store TF tensors — convert to numpy once in filter()
        self.weights_history.append(self.weights.value())

        # Log-likelihood using batch method
        log_likelihood = self.model.log_observation_prob_batch(y, eta_1)

        # ESS and resampling
        ess = ess_tf(self.weights.value())
        self.ess_history.append(ess)

        if ess < self.resample_threshold * self.n_particles:
            self._resample()
            self.resampled_at.append(len(self.ess_history) - 1)
            # Count unique particles (numpy needed for np.unique)
            particles_np = self.particles.numpy()
            unique_particles = len(np.unique(particles_np, axis=0))
            self.n_unique_particles.append(unique_particles)

        # Log-likelihood for model evidence
        max_ll = tf.reduce_max(log_likelihood)
        log_lik = max_ll + tf.math.log(tf.reduce_mean(tf.exp(log_likelihood - max_ll)))
        self.log_likelihoods.append(log_lik)

        # Line 26: Update global filter (EKF needs numpy)
        self.global_filter.update(y.numpy() if isinstance(y, tf.Tensor) else y)

    def _resample(self):
        """Resample particles using configured resampling method."""
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
            new_weights = tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(self.n_particles, self.dtype)

        self.particles.assign(resampled_particles)
        self.weights.assign(new_weights)

    def _estimate_mean_cov(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Estimate weighted mean and covariance from particles (TF ops)."""
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
               random_seed: Optional[int] = None,
               progress_callback: Optional[Callable[[int, int, float], None]] = None) -> FilterResult:
        """Run filter on sequence of observations."""
        self.initialize(initial_mean, initial_cov, random_seed)
        T = len(observations)

        # Pre-convert observations to TF once
        obs_tf = tf.constant(observations, dtype=self.dtype)

        for t in range(T):
            t0 = time.perf_counter()
            self.predict()
            self.update(obs_tf[t])
            mean, cov = self._estimate_mean_cov()
            self.means.append(mean)  # TF tensor
            self.covs.append(cov)    # TF tensor
            if progress_callback is not None:
                progress_callback(t, T, time.perf_counter() - t0)

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
                'filter_type': 'EDHParticleFlowFilter',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_method': self.resampling_method_name,
                'resampling_rate': resampling_rate
            }
        )
