"""Local Exact Daum-Huang (LEDH) Invertible Particle Flow Filter. Algorithm 1"""

import tensorflow as tf
import numpy as np
import time
from typing import Tuple, Optional, Callable, Dict, Any
from ...core.model_base import StateSpaceModel
from ...core.types import FilterResult
from ..kalman.batched_ekf import batched_ekf_predict, batched_ekf_update
from ...utils.flow_params import compute_flow_params_batch
from ...utils.distributions import compute_flow_weights
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
        weight_clip_range: Optional[float] = None,
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
            weight_clip_range: If set, clip log-weights to (-val, val) before normalization.
                              Prevents weight collapse while preserving gradient signal.
                              Typical values: 30-50. None = no clipping (MATLAB behavior).
            debug_mode: If True, store detailed diagnostics
        """
        self.model = model
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.state_dim = model.state_dim
        self.obs_dim = model.obs_dim
        self.n_particles = n_particles
        self.n_lambda_steps = n_lambda_steps
        self.regularization = regularization
        self.resample_threshold = resample_threshold
        self.weight_clip_range = (-weight_clip_range, weight_clip_range) if weight_clip_range is not None else None
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

        # Cache R_inv (constant across timesteps)
        self.R_inv_cache = None

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
        self.lambda_steps = tf.constant(lambda_steps_np, dtype=self.dtype)

    def initialize(self, initial_mean: Optional[np.ndarray] = None,
                   initial_cov: Optional[np.ndarray] = None,
                   random_seed: Optional[int] = None):
        """Initialize particles and per-particle filters."""
        if random_seed is not None:
            self.seed_counter = random_seed

        if initial_mean is None:
            # Use model's initial mean directly (not a random sample)
            initial_mean = np.asarray(self.model.mu_0, dtype=np.float64 if self.dtype == tf.float64 else np.float32)

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

        # Per-particle covariances for batched EKF (TF Variable)
        self.particle_covs = tf.Variable(
            tf.tile(tf.expand_dims(initial_cov_tf, 0), [self.n_particles, 1, 1]),
            dtype=self.dtype
        )

        # Pre-allocate Variables used in predict() to avoid per-timestep allocation
        self.particles_prev = tf.Variable(tf.zeros_like(particles_tf), dtype=self.dtype)
        self.eta_bar_0 = tf.Variable(tf.zeros([self.n_particles, self.state_dim], dtype=self.dtype))
        self.eta_0 = tf.Variable(tf.zeros([self.n_particles, self.state_dim], dtype=self.dtype))

        self.means = []
        self.covs = []
        self.log_likelihoods = []
        self.ess_history = []
        self.weights_history = []
        self.resampled_at = []
        self.n_unique_particles = []

    def predict(self):
        """Prediction step with batched EKF (all TF ops)."""
        self.particles_prev.assign(self.particles.value())

        # Batched EKF predict - single tf.function call (all TF tensors)
        eta_bar_0_tf, cov_pred_tf = batched_ekf_predict(
            self.model, self.particles.value(), self.particle_covs.value()
        )
        self.particle_covs.assign(cov_pred_tf)

        self.eta_bar_0.assign(eta_bar_0_tf)

        # Stochastic prediction - use batch method
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1
        eta_0_tf = self.model.state_transition_batch(self.particles_prev.value(), seed)
        self.eta_0.assign(eta_0_tf)

    def update(self, y: tf.Tensor):
        """Update step with per-particle local flow using batched operations."""
        R = self.model.observation_noise_cov

        eta_1 = self.eta_0.value()
        eta_bar = self.eta_bar_0.value()

        lambda_val = tf.constant(0.0, dtype=self.dtype)
        log_theta = tf.zeros(self.n_particles, dtype=self.dtype)

        # Cache R_inv (constant across timesteps)
        if self.R_inv_cache is None:
            self.R_inv_cache = tf.linalg.inv(R)
        R_inv = self.R_inv_cache
        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)

        # Use TF tensors directly (particle_covs is TF Variable, eta_bar_0 is TF Variable)
        particle_covs_tf = self.particle_covs.value()
        eta_bar_0_tf = self.eta_bar_0.value()

        I_sd = tf.eye(self.state_dim, dtype=self.dtype)

        # Flow loop — batched over all N particles per lambda step
        for j in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[j]
            lambda_val = lambda_val + d_lambda

            # ONE batch call for all N particles (per-particle P_i from batched EKF)
            A_batch, b_batch = compute_flow_params_batch(
                self.model, eta_bar, lambda_val, y, particle_covs_tf,
                R, R_inv, eta_bar_0_tf, self.state_dim, regularization_tf
            )

            # Vectorized Euler step for eta_bar: dx/dλ = A@x + b
            drift_bar = tf.einsum('nij,nj->ni', A_batch, eta_bar) + b_batch
            eta_bar = eta_bar + d_lambda * drift_bar

            # Vectorized Euler step for eta_1 (same A, b)
            drift_1 = tf.einsum('nij,nj->ni', A_batch, eta_1) + b_batch
            eta_1 = eta_1 + d_lambda * drift_1

            # Vectorized log-det of Jacobian: M = I + dλ * A
            M_batch = tf.expand_dims(I_sd, 0) + d_lambda * A_batch  # (N, sd, sd)
            log_det_M = tf.math.log(tf.abs(tf.linalg.det(M_batch)))  # (N,)
            log_theta = log_theta + log_det_M

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
            clip_range=self.weight_clip_range
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
        """Resample particles and per-particle covariances."""
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1

        result = self.resampling_method(
            self.particles.value(),
            self.weights.value(),
            seed=seed,
            **self.resampling_config
        )

        # Get ancestor indices from ResampleResult
        if result.ancestor_indices is not None:
            # Index-based (systematic, soft): direct gather
            indices = result.ancestor_indices
        elif result.transport_matrix is not None:
            # Transport-based (OT): use dominant ancestor per row
            indices = tf.argmax(result.transport_matrix, axis=1)
        else:
            raise ValueError("ResampleResult has neither ancestor_indices nor transport_matrix")

        # Resample covariances using ancestor indices
        self.particle_covs.assign(tf.gather(self.particle_covs.value(), indices))

        self.particles.assign(result.particles)
        self.weights.assign(result.weights)

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

    def log_marginal_likelihood_tf(
        self,
        observations: tf.Tensor,
        seed: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Total log marginal likelihood as a differentiable TF scalar.

        Runs the full LEDH filter (initialize, predict/update loop)
        and sums per-step log-likelihoods. All ops stay in TF graph
        for HMC gradient computation.

        For differentiability, resampling must use a differentiable method
        (ot_entropy or soft). Systematic resampling will break gradients.

        Args:
            observations: (T, obs_dim), dtype matching model
            seed: TF random seed (2,) for initialization

        Returns:
            Scalar tf.Tensor: log p(y_{1:T})
        """
        random_seed = int(seed[0].numpy()) if seed is not None else 42
        self.initialize(random_seed=random_seed)

        T = observations.shape[0]
        total_log_lik = tf.constant(0.0, dtype=self.dtype)

        for t in range(T):
            # Advance time step for time-dependent models (e.g., Kitagawa)
            if hasattr(self.model, 't'):
                self.model.t = t + 1

            self.predict()
            self.update(observations[t])

            # Accumulate the log-likelihood already computed in update()
            total_log_lik = total_log_lik + self.log_likelihoods[-1]

        return total_log_lik

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
                'filter_type': 'LEDHParticleFlowFilter',
                'n_particles': self.n_particles,
                'n_lambda_steps': self.n_lambda_steps,
                'resampling_method': self.resampling_method_name,
                'resampling_rate': resampling_rate
            }
        )
