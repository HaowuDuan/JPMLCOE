"""Tests for Stochastic Volatility models (1D log-space + 2D)."""

import pytest
import numpy as np
import tensorflow as tf

from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data


# =====================================================================
# 1D SV: Log-space mode
# =====================================================================

class TestSVLogSpace:
    """Test the log-space observation transform for 1D SV."""

    @pytest.fixture
    def model_original(self):
        return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5,
                                         log_space=False, dtype=tf.float64)

    @pytest.fixture
    def model_log(self):
        return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5,
                                         log_space=True, dtype=tf.float64)

    def test_state_methods_unchanged(self, model_original, model_log):
        """Log-space mode should not affect state transition methods."""
        x = tf.constant([0.5], dtype=tf.float64)
        tf.debugging.assert_near(
            model_original.state_transition_mean(x),
            model_log.state_transition_mean(x),
        )
        tf.debugging.assert_near(
            model_original.state_transition_cov(x),
            model_log.state_transition_cov(x),
        )
        tf.debugging.assert_near(
            model_original.state_jacobian(x),
            model_log.state_jacobian(x),
        )

    def test_observation_mean_nonzero_in_log_space(self, model_log):
        """In log-space, E[z|x] = log(beta^2) + x + E[log(chi^2_1)] != 0."""
        x = tf.constant([1.0], dtype=tf.float64)
        obs_mean = model_log.observation_mean(x)
        assert obs_mean.numpy()[0] != 0.0, "Log-space obs mean should be nonzero"

    def test_observation_jacobian_nonzero_in_log_space(self, model_log):
        """In log-space, dh/dx = 1 (not 0)."""
        x = tf.constant([1.0], dtype=tf.float64)
        H = model_log.observation_jacobian(x)
        np.testing.assert_allclose(H.numpy(), [[1.0]], atol=1e-10)

    def test_observation_jacobian_zero_in_original(self, model_original):
        """In original mode, dE[y|x]/dx = 0."""
        x = tf.constant([1.0], dtype=tf.float64)
        H = model_original.observation_jacobian(x)
        np.testing.assert_allclose(H.numpy(), [[0.0]], atol=1e-10)

    def test_observation_cov_constant_in_log_space(self, model_log):
        """In log-space, Var[z|x] = pi^2/2 (constant, not state-dependent)."""
        x1 = tf.constant([0.0], dtype=tf.float64)
        x2 = tf.constant([2.0], dtype=tf.float64)
        R1 = model_log.observation_cov(x1).numpy()
        R2 = model_log.observation_cov(x2).numpy()
        np.testing.assert_allclose(R1, R2, atol=1e-10)
        np.testing.assert_allclose(R1, [[np.pi**2 / 2]], rtol=1e-6)

    def test_observation_cov_state_dependent_in_original(self, model_original):
        """In original mode, Var[y|x] = beta^2 * exp(x) (state-dependent)."""
        x1 = tf.constant([0.0], dtype=tf.float64)
        x2 = tf.constant([2.0], dtype=tf.float64)
        R1 = model_original.observation_cov(x1).numpy()[0, 0]
        R2 = model_original.observation_cov(x2).numpy()[0, 0]
        assert R1 != R2

    def test_transform_observations_identity_in_original(self, model_original):
        """Original mode: transform_observations returns unchanged data."""
        obs = np.array([[1.0], [-0.5], [2.0]])
        transformed = model_original.transform_observations(obs)
        np.testing.assert_array_equal(obs, transformed)

    def test_transform_observations_log_squared(self, model_log):
        """Log-space: transform_observations returns log(y^2)."""
        obs = np.array([[2.0], [-3.0], [0.5]])
        transformed = model_log.transform_observations(obs)
        expected = np.log(obs ** 2)
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)

    def test_transform_observations_handles_zero(self, model_log):
        """Log-space: y=0 should not produce -inf (floor applied)."""
        obs = np.array([[0.0], [1.0]])
        transformed = model_log.transform_observations(obs)
        assert np.all(np.isfinite(transformed))

    def test_sampling_unchanged_by_log_space(self, model_original, model_log):
        """sample_observation should produce the same raw y regardless of mode."""
        x = tf.constant([0.5], dtype=tf.float64)
        seed = tf.constant([42, 0], dtype=tf.int32)
        y_orig = model_original.sample_observation(x, seed)
        y_log = model_log.sample_observation(x, seed)
        tf.debugging.assert_near(y_orig, y_log)

    def test_log_observation_prob_finite(self, model_log):
        """log_observation_prob should be finite in log-space mode."""
        x = tf.constant([0.5], dtype=tf.float64)
        y = tf.constant([1.5], dtype=tf.float64)
        ll = model_log.log_observation_prob(y, x)
        assert np.isfinite(ll.numpy())

    def test_batch_methods_log_space(self, model_log):
        """Batch methods should work in log-space mode."""
        particles = tf.constant([[0.5], [1.0], [-0.5]], dtype=tf.float64)
        # observation is already log-transformed for batch methods
        obs = tf.constant([2.0], dtype=tf.float64)

        # These should not error
        ll = model_log.log_observation_prob_batch(obs, particles)
        assert ll.shape == (3,)
        assert np.all(np.isfinite(ll.numpy()))

        H = model_log.observation_jacobian_batch(particles)
        assert H.shape == (3, 1, 1)
        np.testing.assert_allclose(H.numpy(), np.ones((3, 1, 1)), atol=1e-10)

        h = model_log.observation_function_batch(particles)
        assert h.shape == (3, 1)

    def test_ekf_runs_with_log_space(self, model_log):
        """EKF should produce finite results with log-space SV model."""
        from src.filters.kalman.extended_kalman import ExtendedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations_raw = generate_data(model_log, T=50, rng=rng)
        observations = model_log.transform_observations(observations_raw)

        ekf = ExtendedKalmanFilter(model_log, sample_initial_mean=False)
        result = ekf.filter(observations)

        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)

    def test_ukf_runs_with_log_space(self, model_log):
        """UKF should produce finite results with log-space SV model."""
        from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations_raw = generate_data(model_log, T=50, rng=rng)
        observations = model_log.transform_observations(observations_raw)

        ukf = UnscentedKalmanFilter(model_log, sample_initial_mean=False)
        result = ukf.filter(observations)

        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)

    def test_log_marginal_likelihood_differentiable(self, model_log):
        """Gradient of log_marginal_likelihood_tf w.r.t. model params should be finite."""
        from src.filters.kalman.extended_kalman import ExtendedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations_raw = generate_data(model_log, T=20, rng=rng)
        observations = model_log.transform_observations(observations_raw)
        obs_tf = tf.constant(observations, dtype=tf.float64)

        ekf = ExtendedKalmanFilter(model_log, sample_initial_mean=False)

        # Make alpha a tensor that we can differentiate through
        alpha_var = tf.constant(0.91, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(alpha_var)
            model_log.alpha = alpha_var
            ll = ekf.log_marginal_likelihood_tf(obs_tf)

        grad = tape.gradient(ll, alpha_var)
        assert grad is not None, "Gradient should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad.numpy()}"


# =====================================================================
# 2D SV Model
# =====================================================================

class TestSV2D:
    """Test the 2D Stochastic Volatility model."""

    @pytest.fixture
    def model(self):
        return StochasticVolatility2DModel(
            a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0,
            dtype=tf.float64
        )

    def test_dimensions(self, model):
        assert model.state_dim == 2
        assert model.obs_dim == 1

    def test_initial_distribution(self, model):
        mu = model.mu_0
        P = model.Sigma_0
        assert mu.shape == (2,)
        assert P.shape == (2, 2)
        np.testing.assert_allclose(mu.numpy(), [0.0, 0.0])
        # P should be positive definite
        eigvals = np.linalg.eigvalsh(P.numpy())
        assert np.all(eigvals > 0)

    def test_stationary_covariance_satisfies_lyapunov(self, model):
        """P_0 should satisfy P_0 = A P_0 A^T + Sigma."""
        A = model.state_jacobian(tf.zeros([2], dtype=tf.float64)).numpy()
        Sigma = model.process_noise_cov.numpy()
        P0 = model.Sigma_0.numpy()
        lhs = A @ P0 @ A.T + Sigma
        np.testing.assert_allclose(P0, lhs, atol=1e-10)

    def test_stationarity_check_rejects_unstable(self):
        """Should raise ValueError for |eigenvalue| >= 1."""
        with pytest.raises(ValueError, match="not stable"):
            StochasticVolatility2DModel(a1=1.0, a2=0.5)

    def test_state_transition(self, model):
        x = tf.constant([1.0, 0.5], dtype=tf.float64)
        mean = model.state_transition_mean(x)
        assert mean.shape == (2,)
        expected = np.array([0.95 * 1.0, 0.91 * 0.5])
        np.testing.assert_allclose(mean.numpy(), expected, atol=1e-10)

    def test_state_jacobian_is_A(self, model):
        x = tf.constant([1.0, 0.5], dtype=tf.float64)
        F = model.state_jacobian(x)
        A_expected = np.diag([0.95, 0.91])
        np.testing.assert_allclose(F.numpy(), A_expected, atol=1e-10)

    def test_observation_mean(self, model):
        """E[y|x] = b * x_1"""
        x = tf.constant([2.0, 1.0], dtype=tf.float64)
        y_mean = model.observation_mean(x)
        np.testing.assert_allclose(y_mean.numpy(), [2.0], atol=1e-10)

    def test_observation_jacobian(self, model):
        """H = [b, 0] = [1, 0]"""
        x = tf.constant([2.0, 1.0], dtype=tf.float64)
        H = model.observation_jacobian(x)
        np.testing.assert_allclose(H.numpy(), [[1.0, 0.0]], atol=1e-10)

    def test_observation_cov_state_dependent(self, model):
        """Var[y|x] = exp(x_2)"""
        x = tf.constant([0.0, 1.0], dtype=tf.float64)
        R = model.observation_cov(x)
        np.testing.assert_allclose(R.numpy(), [[np.exp(1.0)]], rtol=1e-6)

    def test_log_observation_prob(self, model):
        """log p(y|x) should be a valid log-normal density."""
        x = tf.constant([1.0, 0.5], dtype=tf.float64)
        y = tf.constant([1.5], dtype=tf.float64)
        ll = model.log_observation_prob(y, x)
        assert np.isfinite(ll.numpy())

        # Manual check: N(y; b*x1, exp(x2))
        mean = 1.0 * 1.0
        var = np.exp(0.5)
        expected = -0.5 * (np.log(2 * np.pi * var) + (1.5 - mean)**2 / var)
        np.testing.assert_allclose(ll.numpy(), expected, rtol=1e-6)

    def test_sampling_produces_correct_shapes(self, model):
        seed = tf.constant([42, 0], dtype=tf.int32)
        x0 = model.sample_initial_state(seed)
        assert x0.shape == (2,)

        x1 = model.sample_state_transition(x0, seed)
        assert x1.shape == (2,)

        y = model.sample_observation(x1, seed)
        assert y.shape == (1,)

    def test_generate_data(self, model):
        """generate_data should work with 2D SV model."""
        rng = np.random.default_rng(42)
        initial_state, states, observations = generate_data(model, T=100, rng=rng)
        assert initial_state.shape == (2,)
        assert states.shape == (100, 2)
        assert observations.shape == (100, 1)
        assert np.all(np.isfinite(states))
        assert np.all(np.isfinite(observations))

    def test_batch_methods(self, model):
        """Batch methods should produce correct shapes."""
        seed = tf.constant([42, 0], dtype=tf.int32)
        N = 10

        # Initial state batch
        x0_batch = model.sample_initial_state_batch(N, seed)
        assert x0_batch.shape == (N, 2)

        # State transition batch
        x1_batch = model.state_transition_batch(x0_batch, seed)
        assert x1_batch.shape == (N, 2)

        # Transition mean batch
        mean_batch = model.state_transition_mean_batch(x0_batch)
        assert mean_batch.shape == (N, 2)

        # Log obs prob batch
        obs = tf.constant([1.0], dtype=tf.float64)
        ll_batch = model.log_observation_prob_batch(obs, x0_batch)
        assert ll_batch.shape == (N,)
        assert np.all(np.isfinite(ll_batch.numpy()))

        # Observation Jacobian batch
        H_batch = model.observation_jacobian_batch(x0_batch)
        assert H_batch.shape == (N, 1, 2)

        # State Jacobian batch
        F_batch = model.state_jacobian_batch(x0_batch)
        assert F_batch.shape == (N, 2, 2)

    def test_ekf_runs(self, model):
        """EKF should work with 2D SV model (H_x = [b, 0] != 0)."""
        from src.filters.kalman.extended_kalman import ExtendedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations = generate_data(model, T=50, rng=rng)

        ekf = ExtendedKalmanFilter(model, sample_initial_mean=False)
        result = ekf.filter(observations)

        assert result.means.shape == (50, 2)
        assert result.covs.shape == (50, 2, 2)
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)

    def test_ukf_runs(self, model):
        """UKF should work with 2D SV model."""
        from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations = generate_data(model, T=50, rng=rng)

        ukf = UnscentedKalmanFilter(model, sample_initial_mean=False)
        result = ukf.filter(observations)

        assert result.means.shape == (50, 2)
        assert result.covs.shape == (50, 2, 2)
        assert np.all(np.isfinite(result.means))
        assert np.all(np.isfinite(result.covs))
        assert np.isfinite(result.log_likelihood)

    def test_log_marginal_likelihood_differentiable(self, model):
        """Gradient through log_marginal_likelihood_tf should be finite."""
        from src.filters.kalman.extended_kalman import ExtendedKalmanFilter

        rng = np.random.default_rng(42)
        _, _, observations = generate_data(model, T=20, rng=rng)
        obs_tf = tf.constant(observations, dtype=tf.float64)

        ekf = ExtendedKalmanFilter(model, sample_initial_mean=False)

        b_var = tf.constant(1.0, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(b_var)
            model.b = b_var
            ll = ekf.log_marginal_likelihood_tf(obs_tf)

        grad = tape.gradient(ll, b_var)
        assert grad is not None, "Gradient should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad.numpy()}"

    def test_full_A_matrix(self):
        """Should accept a full (non-diagonal) A matrix."""
        A = [[0.8, 0.1], [0.05, 0.9]]
        model = StochasticVolatility2DModel(A=A, sigma1=0.5, sigma2=1.0,
                                             b=1.0, dtype=tf.float64)
        assert model.state_dim == 2
        # Should be stable
        eigvals = np.linalg.eigvals(np.array(A))
        assert np.all(np.abs(eigvals) < 1)

    def test_full_Sigma_matrix(self):
        """Should accept a full Sigma matrix (extracts diagonal)."""
        Sigma = [[0.25, 0.1], [0.1, 1.0]]
        model = StochasticVolatility2DModel(a1=0.95, a2=0.91, Sigma=Sigma,
                                             b=1.0, dtype=tf.float64)
        # Model extracts diagonal entries: sigma1=sqrt(0.25)=0.5, sigma2=sqrt(1.0)=1.0
        np.testing.assert_allclose(model.sigma1, 0.5, atol=1e-10)
        np.testing.assert_allclose(model.sigma2, 1.0, atol=1e-10)


class TestSV2DGradientFlow:
    """Verify gradients flow through all parameters for HMC compatibility."""

    @pytest.fixture
    def model_and_data(self):
        model = StochasticVolatility2DModel(
            a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0,
            dtype=tf.float64,
        )
        rng = np.random.default_rng(42)
        _, _, observations = generate_data(model, T=20, rng=rng)
        obs_tf = tf.constant(observations, dtype=tf.float64)
        return model, obs_tf

    def _check_gradient(self, model, obs_tf, param_name, init_value):
        """Set param to a watched tensor, compute log_obs_prob_batch, check gradient."""
        particles = model.sample_initial_state_batch(50, tf.constant([42, 0]))

        var = tf.constant(init_value, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(var)
            setattr(model, param_name, var)
            ll = tf.reduce_sum(model.log_observation_prob_batch(obs_tf[0], particles))
        grad = tape.gradient(ll, var)
        # Restore original
        setattr(model, param_name, init_value)
        return grad

    def test_gradient_wrt_b(self, model_and_data):
        model, obs_tf = model_and_data
        grad = self._check_gradient(model, obs_tf, 'b', 1.0)
        assert grad is not None, "Gradient w.r.t. b should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad}"

    def test_gradient_wrt_sigma2(self, model_and_data):
        """sigma2 affects state_transition_batch, not obs directly.
        Check gradient through full propagate-then-observe."""
        model, obs_tf = model_and_data
        var = tf.constant(1.0, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(var)
            model.sigma2 = var
            # Propagate: sigma2 enters through noise
            particles = model.sample_initial_state_batch(50, tf.constant([42, 0]))
            seed2 = tf.constant([99, 0])
            particles = model.state_transition_batch(particles, seed2)
            ll = tf.reduce_sum(model.log_observation_prob_batch(obs_tf[0], particles))
        grad = tape.gradient(ll, var)
        model.sigma2 = 1.0
        assert grad is not None, "Gradient w.r.t. sigma2 should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad}"

    def test_gradient_wrt_a2(self, model_and_data):
        """a2 enters through state_transition_mean_batch."""
        model, obs_tf = model_and_data
        var = tf.constant(0.91, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(var)
            model.a2 = var
            particles = model.sample_initial_state_batch(50, tf.constant([42, 0]))
            seed2 = tf.constant([99, 0])
            particles = model.state_transition_batch(particles, seed2)
            ll = tf.reduce_sum(model.log_observation_prob_batch(obs_tf[0], particles))
        grad = tape.gradient(ll, var)
        model.a2 = 0.91
        assert grad is not None, "Gradient w.r.t. a2 should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad}"

    def test_gradient_wrt_a1(self, model_and_data):
        model, obs_tf = model_and_data
        var = tf.constant(0.95, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(var)
            model.a1 = var
            particles = model.sample_initial_state_batch(50, tf.constant([42, 0]))
            seed2 = tf.constant([99, 0])
            particles = model.state_transition_batch(particles, seed2)
            ll = tf.reduce_sum(model.log_observation_prob_batch(obs_tf[0], particles))
        grad = tape.gradient(ll, var)
        model.a1 = 0.95
        assert grad is not None, "Gradient w.r.t. a1 should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad}"

    def test_gradient_wrt_sigma1(self, model_and_data):
        model, obs_tf = model_and_data
        var = tf.constant(0.5, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(var)
            model.sigma1 = var
            particles = model.sample_initial_state_batch(50, tf.constant([42, 0]))
            seed2 = tf.constant([99, 0])
            particles = model.state_transition_batch(particles, seed2)
            ll = tf.reduce_sum(model.log_observation_prob_batch(obs_tf[0], particles))
        grad = tape.gradient(ll, var)
        model.sigma1 = 0.5
        assert grad is not None, "Gradient w.r.t. sigma1 should not be None"
        assert np.isfinite(grad.numpy()), f"Gradient should be finite, got {grad}"
