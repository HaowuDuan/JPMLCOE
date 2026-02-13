"""Local Exact Daum-Huang (LEDH) particle flow filter"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from .flow_base import FlowFilterBase
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ...utils.flow_params import compute_flow_params, compute_flow_params_batch
from ...utils.ode_solvers import euler_step
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample


class LocalExactDaumHuangFlow(FlowFilterBase):
    """
    Local Exact Daum-Huang Flow with per-particle linearization.

    Key modifications from global EDH:
    1. Linearize measurement function at EACH particle location (not just at mean)
    2. Use GLOBAL predictive covariance P_{k|k-1} from a single EKF
    3. Compute A_i and b_i matrices individually for each particle
    
    This is modification (ii) from Section 3 of the paper.

    Flow equation: dx_i/dλ = A_i(λ) @ x_i + b_i(λ)
    where A_i uses local linearization H_i at particle i and global covariance P_{k|k-1}.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 use_feedback: bool = True,
                 regularization: float = 1e-8,
                 resampling_method: Optional[Callable] = None,
                 resampling_config: Optional[Dict[str, Any]] = None,
                 n_threads: Optional[int] = None,
                 debug_mode: bool = False):
        """
        Initialize Local Exact Daum-Huang flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4'
            use_feedback: If True, feed DH mean back to EKF
            regularization: Small value added to diagonal of S matrix for numerical stability
            resampling_method: Resampling method (string or callable)
                'systematic', 'soft', or 'ot_entropy'
            resampling_config: Optional dict with method-specific parameters
                For soft: {'alpha': float}
                For ot_entropy: {'reg': float, 'n_iter': int}
            n_threads: Number of threads for parallelization (None = auto)
            debug_mode: If True, collect detailed diagnostics
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.use_feedback = use_feedback
        self.regularization = regularization
        self.debug_mode = debug_mode

        # Configure resampling method
        if isinstance(resampling_method, str):
            method_map = {
                'systematic': systematic_resample,
                'soft': soft_resample,
                'ot_entropy': ot_entropy_resample,
            }
            self.resampling_method = method_map.get(resampling_method, systematic_resample)
        elif callable(resampling_method):
            self.resampling_method = resampling_method
        else:
            self.resampling_method = systematic_resample

        # Process resampling config (convert to Python scalars for TF compatibility)
        self.resampling_config = {}
        if resampling_config is not None:
            for key, value in resampling_config.items():
                if isinstance(value, (int, np.integer)):
                    self.resampling_config[key] = int(value)
                elif isinstance(value, (float, np.floating)):
                    self.resampling_config[key] = float(value)
                else:
                    self.resampling_config[key] = value

        # Seed counter for stateless random sampling
        self.seed_counter = 0

        # Single GLOBAL EKF for covariance guidance
        self.global_filter = None
        self.predicted_cov = None  # P_{k|k-1} used for ALL particles

        # Cache for optimization
        self.R_inv_cache = None
        self.L_cache = None

        # Store mean at λ=0 for b computation
        self.eta_bar_0 = None
        
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
                'eigenvalues': [],
                'condition_numbers': [],
                'particle_stats': []
            }
        else:
            self.debug_info = None
        
        # Generate exponential lambda schedule (same as ledh_invertible)
        self._generate_lambda_steps()

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles and global EKF for covariance guidance."""
        # Get initial mean and covariance from model
        if hasattr(self.model, 'mu_0') and hasattr(self.model, 'Sigma_0'):
            initial_mean = self.model.mu_0
            initial_cov = self.model.Sigma_0
        else:
            raise ValueError("Model must have mu_0 and Sigma_0 attributes")

        # Convert to TensorFlow for sampling
        initial_mean_tf = tf.constant(initial_mean, dtype=tf.float32)
        initial_cov_tf = tf.constant(initial_cov, dtype=tf.float32)

        # Sample initial particles using TensorFlow
        if random_state is not None:
            seed = [random_state.integers(0, 2**31), random_state.integers(0, 2**31)]
        else:
            seed = [0, 0]
        seed = tf.constant(seed, dtype=tf.int32)

        L = tf.linalg.cholesky(initial_cov_tf)
        z = tf.random.stateless_normal([self.n_particles, self.state_dim], seed=seed)
        particles_tf = initial_mean_tf + tf.linalg.matmul(z, L, transpose_b=True)

        # Store as TensorFlow Variable
        self.particles = tf.Variable(particles_tf, dtype=tf.float32)
        self.weights = tf.Variable(
            tf.ones(self.n_particles, dtype=tf.float32) / tf.cast(self.n_particles, tf.float32),
            dtype=tf.float32
        )

        # Compute empirical mean and covariance (TF ops)
        ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
        diff = self.particles.value() - ensemble_mean
        if self.state_dim == 1:
            initial_cov_emp = tf.reshape(tf.math.reduce_variance(self.particles.value()), [1, 1])
        else:
            initial_cov_emp = tf.matmul(diff, diff, transpose_a=True) / tf.cast(self.n_particles, tf.float32)

        # Initialize global EKF for covariance guidance (constructor needs numpy)
        ensemble_mean_np = ensemble_mean.numpy()
        initial_cov_emp_np = initial_cov_emp.numpy()
        self.global_filter = ExtendedKalmanFilter(
            self.model, mean_0=ensemble_mean_np, Sigma_0=initial_cov_emp_np
        )

        self.global_filter.mean.assign(ensemble_mean)
        self.global_filter.cov.assign(initial_cov_emp)
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

        # Initialize seed counter
        self.seed_counter = 0

    def _generate_lambda_steps(self):
        """
        Generate exponential decay schedule for lambda steps as TF tensor.
        Uses geometric sequence with ratio q=1.2, normalized to sum to 1.
        """
        q = 1.2
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = tf.constant(steps_np, dtype=tf.float32)
        

    @tf.function
    def _compute_drift_single(self, particle: tf.Tensor, lambda_val: tf.Tensor,
                             observation: tf.Tensor, P: tf.Tensor, R: tf.Tensor,
                             R_inv: tf.Tensor, eta_bar_0: tf.Tensor,
                             regularization: tf.Tensor) -> tf.Tensor:
        """
        Compute drift for a single particle.

        Args:
            particle: Current particle, shape (state_dim,)
            lambda_val: Current pseudo-time λ
            observation: Measurement z
            P: GLOBAL predictive covariance
            R: Observation noise covariance
            R_inv: Inverse of R
            eta_bar_0: GLOBAL mean at λ=0
            regularization: Regularization strength

        Returns:
            drift: Particle drift, shape (state_dim,)
        """
        # LEDH uses individual particle as linearization point, with regularization
        A_i, b_i = compute_flow_params(
            self.model, particle, lambda_val, observation,
            P, R, R_inv, eta_bar_0, self.state_dim, regularization
        )
        particle_drift = tf.linalg.matvec(A_i, particle) + b_i

        # Limit drift magnitude to prevent explosive growth
        max_drift_norm = 100.0
        drift_norm = tf.norm(particle_drift)
        scale = tf.minimum(1.0, max_drift_norm / (drift_norm + 1e-10))
        particle_drift = particle_drift * scale

        return particle_drift

    def _flow_step_euler(
        self,
        particles: tf.Tensor,
        y: tf.Tensor,
        lambda_val: tf.Tensor,
        d_lambda: tf.Tensor,
        P: tf.Tensor,
        R: tf.Tensor,
        R_inv: tf.Tensor,
        eta_bar_0: tf.Tensor
    ) -> tf.Tensor:
        """
        Local flow step using Euler integration with batched flow params.

        Each particle uses its own LOCAL linearization H_i and GLOBAL P_{k|k-1}.

        Args:
            particles: Current particles, shape (N, state_dim)
            y: Observation, shape (obs_dim,)
            lambda_val: Current λ
            d_lambda: Step size
            P: GLOBAL predictive covariance (sd, sd) — broadcast inside batch fn
            R: Observation noise covariance
            R_inv: Inverse of R
            eta_bar_0: GLOBAL mean at λ=0

        Returns:
            Updated particles, shape (N, state_dim)
        """
        regularization_tf = tf.constant(self.regularization, dtype=tf.float32)

        # Compute A, b for ALL particles in one batched call
        A_batch, b_batch = compute_flow_params_batch(
            self.model, particles, lambda_val, y, P, R, R_inv,
            eta_bar_0, self.state_dim, regularization_tf
        )

        # Drift: A_i @ x_i + b_i for all particles: (N, sd)
        drift = tf.einsum('nij,nj->ni', A_batch, particles) + b_batch

        # Clip drift magnitude per-particle
        drift_norms = tf.norm(drift, axis=1, keepdims=True)
        scale = tf.minimum(1.0, 100.0 / (drift_norms + 1e-10))
        drift = drift * scale

        # Euler step
        particles_new = particles + drift * d_lambda

        # Apply clipping to prevent divergence
        max_dist = 1000.0
        norms = tf.norm(particles_new, axis=1, keepdims=True)
        scale = tf.minimum(1.0, max_dist / (norms + 1e-10))
        particles_new = particles_new * scale

        return particles_new

    def predict(self):
        """
        Prediction step with mean-only feedback.

        1. Update global filter mean to ensemble mean (if feedback enabled)
        2. Run global EKF prediction to get P_{k|k-1}
        3. Propagate each particle through dynamics with noise

        Note: We do NOT blend empirical covariances back into the global filter.
        The global filter provides covariance guidance only - blending causes
        covariance explosion due to positive feedback.
        """
        # FEEDBACK MECHANISM: Update global filter mean to ensemble mean (TF ops, no numpy)
        if self.use_feedback:
            ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
            self.global_filter.mean.assign(ensemble_mean)

        # Run GLOBAL EKF prediction to get P_{k|k-1}
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

        # Store η̄_0: the DETERMINISTIC predicted mean
        self.eta_bar_0 = self.global_filter.mean.value()  # TF tensor

        # Propagate particles through state transition using model's batch method
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1

        particles_predicted = self.model.state_transition_batch(self.particles.value(), seed)
        self.particles.assign(particles_predicted)

    def update(self, y: tf.Tensor):
        """
        Update step: flow particles from λ=0 to λ=1 using LOCAL linearizations.

        Each particle gets its own H_i but uses GLOBAL P_{k|k-1}.

        Args:
            y: Observation TF tensor, shape (obs_dim,)
        """
        observation = y  # Already TF tensor from flow_base.filter()
        P_tf = self.predicted_cov  # Already TF tensor from predict()
        R_tf = self.model.observation_noise_cov
        eta_bar_0_tf = self.eta_bar_0  # Already TF tensor from predict()

        # Cache R_inv (constant across timesteps)
        if self.R_inv_cache is None:
            self.R_inv_cache = tf.linalg.inv(R_tf)
        R_inv_tf = self.R_inv_cache

        # Use exponential lambda schedule
        particles_flow = self.particles.value()
        lambda_val = tf.constant(0.0, dtype=tf.float32)

        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = particles_flow.numpy().copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.numpy().copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

        # Integrate flow with LOCAL linearizations
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val = lambda_val + d_lambda  # TF tensor accumulation

            # Debug: Capture flow step diagnostics (sample steps)
            if self.debug_mode and i % 10 == 0:
                regularization_tf = tf.constant(self.regularization, dtype=tf.float32)
                A, b = compute_flow_params(
                    self.model, particles_flow[0], lambda_val, observation,
                    P_tf, R_tf, R_inv_tf, eta_bar_0_tf, self.state_dim, regularization_tf
                )
                H = self.model.observation_jacobian(particles_flow[0])

                A_np = A.numpy()
                try:
                    eigvals = np.linalg.eigvals(A_np)
                    cond_A = np.linalg.cond(A_np)
                except:
                    eigvals = np.array([np.nan])
                    cond_A = np.nan

                flow_step_debug = {
                    'step': i,
                    'lambda': float(lambda_val),
                    'epsilon': float(d_lambda),
                    'A_matrix': A_np.copy(),
                    'b_vector': b.numpy().copy(),
                    'H_jacobian': H.numpy().copy(),
                    'eigenvalues': eigvals,
                    'condition_number': cond_A,
                    'particle_mean': tf.reduce_mean(particles_flow, axis=0).numpy(),
                    'particle_std': tf.math.reduce_std(particles_flow, axis=0).numpy()
                }
                timestep_debug['flow_steps'].append(flow_step_debug)

            if self.integration_method == 'euler':
                particles_flow = self._flow_step_euler(
                    particles_flow, observation, lambda_val, d_lambda,
                    P_tf, R_tf, R_inv_tf, eta_bar_0_tf
                )
            elif self.integration_method == 'rk4':
                raise NotImplementedError("RK4 integration not yet implemented for TensorFlow LEDH flow")
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

        # Particles at λ=1 represent posterior
        self.particles.assign(particles_flow)

        # Debug: Store after-flow diagnostics
        if self.debug_mode:
            particles_after = self.particles.numpy()
            timestep_debug['particles_after'] = particles_after.copy()
            timestep_debug['particle_stats_after'] = {
                'mean': np.mean(particles_after, axis=0),
                'cov': np.cov(particles_after.T),
                'min': np.min(particles_after, axis=0),
                'max': np.max(particles_after, axis=0)
            }
            self.debug_info['timesteps'].append(timestep_debug)

        # Update global filter for next prediction cycle (EKF accepts numpy)
        self.global_filter.update(y.numpy() if isinstance(y, tf.Tensor) else y)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        return {
            'final_particles': self.particles.numpy(),
            'predicted_cov': self.predicted_cov.numpy() if isinstance(self.predicted_cov, tf.Tensor) else self.predicted_cov,
            'global_filter_mean': self.global_filter.mean,
            'global_filter_cov': self.global_filter.cov,
            'use_feedback': self.use_feedback,
            'eta_bar_0': self.eta_bar_0.numpy() if isinstance(self.eta_bar_0, tf.Tensor) else self.eta_bar_0
        }
