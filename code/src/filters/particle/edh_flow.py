"""Exact Daum-Huang (EDH) particle flow filter."""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from .flow_base import FlowFilterBase
from ..kalman.extended_kalman import ExtendedKalmanFilter
from ...utils.flow_params import compute_flow_params
from ...utils.ode_solvers import euler_step, rk4_step
from ...resampling import systematic_resample, soft_resample, ot_entropy_resample


class ExactDaumHuangFlow(FlowFilterBase):
    """
    Exact Daum-Huang particle flow filter with feedback mechanism.

    Assumes Gaussian prior/predictive posterior with covariance P.
    Linearizes nonlinear measurement model at mean particle location.

    Flow equation: dη/dλ = A(λ)η + b(λ)
    where A(λ) and b(λ) are computed from Equations (10) and (11).

    All particles have equal weight 1/N throughout.

    Key modification: Implements feedback from particle ensemble to EKF/UKF
    as described in Algorithm 2 of the paper.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 resampling_method: Optional[Callable] = None,
                 resampling_config: Optional[Dict[str, Any]] = None,
                 n_threads: Optional[int] = None,
                 debug_mode: bool = False):
        """
        Initialize Exact Daum-Huang flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4' for ODE integration
            resampling_method: Resampling method (string or callable)
                'systematic', 'soft', or 'ot_entropy'
            resampling_config: Optional dict with method-specific parameters
                For soft: {'alpha': float}
                For ot_entropy: {'reg': float, 'n_iter': int}
            n_threads: Number of threads (None = use CPU count, ignored for now)
            debug_mode: If True, collect detailed diagnostics
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.debug_mode = debug_mode
        self.predicted_cov = None  # P_{k|k-1} from EKF (TF tensor)
        self.eta_bar_0 = None      # η̄_0 (mean at λ=0, TF tensor)
        self.global_filter = None  # EKF for covariance guidance

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

        # Cache R_inv if R is constant
        self.R_inv_cache = None
        self.L_cache = None

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

        # Generate exponential lambda schedule as TF tensor
        self._generate_lambda_steps()

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles and global EKF/UKF for covariance guidance."""
        # Get initial mean and covariance from model
        if hasattr(self.model, 'mu_0') and hasattr(self.model, 'Sigma_0'):
            initial_mean = self.model.mu_0
            initial_cov = self.model.Sigma_0
        else:
            raise ValueError("Model must have mu_0 and Sigma_0 attributes")

        # Convert to TensorFlow for sampling
        initial_mean_tf = tf.constant(initial_mean, dtype=self.dtype)
        initial_cov_tf = tf.constant(initial_cov, dtype=self.dtype)

        # Sample initial particles using TensorFlow
        if random_state is not None:
            seed = [random_state.integers(0, 2**31), random_state.integers(0, 2**31)]
        else:
            seed = [0, 0]
        seed = tf.constant(seed, dtype=tf.int32)

        L = tf.linalg.cholesky(initial_cov_tf)
        z = tf.random.stateless_normal([self.n_particles, self.state_dim], seed=seed, dtype=self.dtype)
        particles_tf = initial_mean_tf + tf.linalg.matmul(z, L, transpose_b=True)

        # Store as TensorFlow Variable
        self.particles = tf.Variable(particles_tf, dtype=self.dtype)
        self.weights = tf.Variable(
            tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(self.n_particles, self.dtype),
            dtype=self.dtype
        )

        # Compute empirical mean and covariance (TF ops)
        ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
        diff = self.particles.value() - ensemble_mean
        if self.state_dim == 1:
            initial_cov_emp = tf.reshape(tf.math.reduce_variance(self.particles.value()), [1, 1])
        else:
            initial_cov_emp = tf.matmul(diff, diff, transpose_a=True) / tf.cast(self.n_particles, self.dtype)

        # Initialize global EKF for covariance guidance (constructor needs numpy)
        ensemble_mean_np = ensemble_mean.numpy()
        initial_cov_emp_np = initial_cov_emp.numpy()
        self.global_filter = ExtendedKalmanFilter(self.model, mean_0=ensemble_mean_np, Sigma_0=initial_cov_emp_np)

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
        self.lambda_steps = tf.constant(steps_np, dtype=self.dtype)

    @tf.function
    def _compute_drift(self, particles: tf.Tensor, A: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        """Compute drift for EDH flow: dη/dλ = Aη + b (vectorized)."""
        return particles @ tf.transpose(A) + b

    def predict(self):
        """
        Prediction step - propagate particles and get covariance from EKF/UKF.

        Uses global EKF/UKF to estimate P_{k|k-1} for flow guidance.
        Implements feedback from particle ensemble to EKF/UKF (Algorithm 2).
        """
        # FEEDBACK: Update global filter mean to ensemble mean (TF ops, no numpy)
        ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
        self.global_filter.mean.assign(ensemble_mean)

        # Run global filter prediction to get P_{k|k-1}
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

        # Store η̄_0: the predicted mean
        self.eta_bar_0 = self.global_filter.mean.value()  # TF tensor

        # Propagate particles through state transition using model's batch method
        seed = tf.constant([self.seed_counter, 0], dtype=tf.int32)
        self.seed_counter += 1

        particles_predicted = self.model.state_transition_batch(self.particles.value(), seed)
        self.particles.assign(particles_predicted)

    def update(self, y: tf.Tensor):
        """
        Update step: flow particles from λ=0 to λ=1.

        All particles maintain equal weight 1/N (no resampling needed for EDH flow).
        After particle flow completes, the ensemble mean is fed back to
        the EKF/UKF before its update step (Algorithm 2, line 18).

        Args:
            y: Observation TF tensor, shape (obs_dim,)
        """
        observation = y  # Already TF tensor from flow_base.filter()
        P_tf = self.predicted_cov  # Already TF tensor from predict()
        R_tf = tf.constant(self.model.observation_noise_cov, dtype=self.dtype)
        eta_bar_0_tf = self.eta_bar_0  # Already TF tensor from predict()

        # Compute R_inv (once per update)
        R_inv_tf = tf.linalg.inv(R_tf)

        # Use exponential lambda schedule
        particles_flow = self.particles.value()
        lambda_val = tf.constant(0.0, dtype=self.dtype)

        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = particles_flow.numpy().copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.numpy().copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

        eta_bar = eta_bar_0_tf

        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val = lambda_val + d_lambda  # TF tensor accumulation

            # Debug: Capture flow step diagnostics (sample steps)
            if self.debug_mode and i % 10 == 0:
                A, b = compute_flow_params(self.model, eta_bar, lambda_val, observation, P_tf, R_tf, R_inv_tf, eta_bar_0_tf, self.state_dim)
                H = self.model.observation_jacobian(eta_bar)

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
                A, b = compute_flow_params(self.model, eta_bar, lambda_val, observation, P_tf, R_tf, R_inv_tf, eta_bar_0_tf, self.state_dim)
                particles_flow = euler_step(particles_flow, self._compute_drift, d_lambda, A, b)
            elif self.integration_method == 'rk4':
                raise NotImplementedError("RK4 integration not yet implemented for TensorFlow EDH flow")
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

            eta_bar = tf.reduce_mean(particles_flow, axis=0)

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
        """Return diagnostic information about flow convergence."""
        return {
            'final_particles': self.particles.numpy(),
            'predicted_cov': self.predicted_cov.numpy() if isinstance(self.predicted_cov, tf.Tensor) else self.predicted_cov,
            'global_filter_mean': self.global_filter.mean,
            'global_filter_cov': self.global_filter.cov
        }
