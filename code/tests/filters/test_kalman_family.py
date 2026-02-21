"""
Kalman family cross-validation tests.

Verifies that EKF and UKF reduce to KF on linear-Gaussian systems, and that
the Joseph-form covariance update preserves positive-definiteness under
ill-conditioning.

# ============================================================================
# How to run and save results to a file:
#
#   Show output in terminal AND save to file:
#     pytest code/tests/test_kalman_family.py -v 2>&1 | tee code/tests/test_kalman_family_results.txt
#
#   Save to file only (no terminal output):
#     pytest code/tests/test_kalman_family.py -v --tb=short > code/tests/test_kalman_family_results.txt 2>&1
#
#   Save as JUnit XML (for CI or structured parsing):
#     pytest code/tests/test_kalman_family.py --junitxml=code/tests/test_kalman_family_results.xml
#
#   Save as HTML report (requires: pip install pytest-html):
#     pytest code/tests/test_kalman_family.py --html=code/tests/test_kalman_family_results.html
# ============================================================================
"""

import pytest
import numpy as np
import tensorflow as tf

from src.filters.kalman.kalman import KalmanFilter
from src.filters.kalman.extended_kalman import ExtendedKalmanFilter
from src.filters.kalman.unscented_kalman import UnscentedKalmanFilter
from src.models.linear_gaussian import LinearGaussianModel
from src.models.utils import generate_data


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def linear_system():
    """
    Asymmetric F, non-identity H, 3D state, 2D observation.
    Exercises off-diagonal terms in the Kalman gain.
    """
    F = np.array([
        [0.9, 0.2, 0.0],
        [0.0, 0.8, 0.3],
        [0.0, 0.0, 0.7]
    ])
    B = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [0.0, 0.05]
    ])
    H = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.0]
    ])
    D = np.array([
        [0.2, 0.0],
        [0.0, 0.15]
    ])
    mean_0 = np.zeros(3)
    Sigma_0 = np.eye(3)
    return F, B, H, D, mean_0, Sigma_0


@pytest.fixture
def observations(linear_system):
    """Generate 50-step observation sequence from the linear system."""
    F, B, H, D, mean_0, Sigma_0 = linear_system
    model = LinearGaussianModel(
        F, B, H, D, mu_0=mean_0, Sigma_0=Sigma_0, dtype=tf.float64
    )
    rng = np.random.default_rng(42)
    _, _, obs = generate_data(model, T=50, rng=rng)
    return obs


@pytest.fixture
def kf_result(linear_system, observations):
    """Run KF once — shared by EKF and UKF comparison tests."""
    F, B, H, D, mean_0, Sigma_0 = linear_system
    kf = KalmanFilter(
        F, B, H, D, mean_0=mean_0, Sigma_0=Sigma_0, dtype=tf.float64
    )
    return kf.filter(observations)


# ============================================================================
# TEST 1: EKF reduces to KF on linear Gaussian
#
# Why: EKF linearizes with Jacobians. For linear f and h, the Jacobian IS
# the matrix. If this fails, the EKF predict or update has a bug.
# ============================================================================

class TestEKFReducesToKF:
    """EKF should produce identical results to KF on a linear system."""

    def _make_ekf(self, linear_system):
        F, B, H, D, mean_0, Sigma_0 = linear_system
        model = LinearGaussianModel(
            F, B, H, D, mu_0=mean_0, Sigma_0=Sigma_0, dtype=tf.float64
        )
        return ExtendedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0, sample_initial_mean=False
        )

    def test_means_match(self, linear_system, observations, kf_result):
        ekf = self._make_ekf(linear_system)
        ekf_result = ekf.filter(observations)
        np.testing.assert_allclose(
            ekf_result.means, kf_result.means, atol=1e-6,
            err_msg="EKF means diverge from KF on linear system"
        )

    def test_covariances_match(self, linear_system, observations, kf_result):
        ekf = self._make_ekf(linear_system)
        ekf_result = ekf.filter(observations)
        np.testing.assert_allclose(
            ekf_result.covs, kf_result.covs, atol=1e-6,
            err_msg="EKF covariances diverge from KF on linear system"
        )


# ============================================================================
# TEST 2: UKF reduces to KF on linear Gaussian
#
# BACKGROUND — WHY THE UNSCENTED TRANSFORM IS EXACT FOR LINEAR FUNCTIONS
#
# The standard unscented transform (UT) generates 2n+1 sigma points from
# a distribution with mean x̄ and covariance P:
#
#     sp₀ = x̄
#     spᵢ = x̄ + [√((n+λ)P)]ᵢ       for i = 1, ..., n
#     spₙ₊ᵢ = x̄ - [√((n+λ)P)]ᵢ     for i = 1, ..., n
#
# where [√M]ᵢ denotes the i-th column of the Cholesky factor of M, and
# λ = α²(n+κ) - n is a scaling parameter.
#
# The mean and covariance weights are:
#
#     Wᵐ₀ = λ/(n+λ)                  (mean weight for sp₀)
#     Wᶜ₀ = λ/(n+λ) + (1 - α² + β)  (covariance weight for sp₀)
#     Wᵢ  = 1/(2(n+λ))               (both weights for i = 1, ..., 2n)
#
# For a linear function f(x) = Fx, the propagated sigma points are simply
# F·spᵢ. The weighted mean becomes:
#
#     ȳ = Σᵢ Wᵐᵢ · F·spᵢ = F · (Σᵢ Wᵐᵢ · spᵢ) = F · x̄
#
# because the mean weights sum to 1 (Wᵐ₀ + 2n·Wᵢ = 1) and the sigma
# points are symmetric about x̄, so the weighted sum collapses to x̄.
#
# The weighted covariance becomes:
#
#     Σᵢ Wᶜᵢ · (F·spᵢ - F·x̄)(F·spᵢ - F·x̄)ᵀ
#   = F · [Σᵢ Wᶜᵢ · (spᵢ - x̄)(spᵢ - x̄)ᵀ] · Fᵀ
#
# The inner bracketed term equals P because:
#   - The sp₀ contribution is zero (sp₀ = x̄, so the outer product vanishes).
#   - For each pair (i, n+i), the deviations are ±L[:,i] where L = chol((n+λ)P),
#     so the combined contribution is:
#         2·Wᵢ · L[:,i]·L[:,i]ᵀ = 2·[1/(2(n+λ))]·(n+λ)·chol(P)[:,i]·chol(P)[:,i]ᵀ
#                                = chol(P)[:,i] · chol(P)[:,i]ᵀ
#   - Summing over i=1..n gives chol(P)·chol(P)ᵀ = P.
#
# Therefore, the UT propagation of a linear function yields exactly
# ȳ = F·x̄ and P̄ = F·P·Fᵀ, which are the KF predict equations. An
# identical argument holds for the observation step with h(x) = Hx. The
# Kalman gain, innovation covariance, cross-covariance, and updated state
# all reduce to the standard KF formulas. This is true for ANY choice of
# α, β, κ — the parameters only affect the distribution of higher-order
# moment errors, which are zero for linear functions.
#
# NUMERICAL CAVEAT — CATASTROPHIC CANCELLATION WITH SMALL α
#
# Although mathematically exact for any α, small α causes severe numerical
# issues. With the standard choice α = 10⁻³ and n = 3 states:
#
#     λ = α²(n+κ) - n = (10⁻⁶)(3) - 3 = -2.999997
#     n + λ = 3 × 10⁻⁶
#
# This makes the weights extremely large in magnitude:
#
#     Wᵐ₀ = λ/(n+λ) = -2.999997 / (3×10⁻⁶) ≈ -999,999
#     Wᵢ  = 1/(2(n+λ)) = 1/(6×10⁻⁶)         ≈ +166,667
#
# The predicted mean is then computed as:
#
#     ȳ = (-999,999)·f(sp₀) + 166,667·Σⱼ f(spⱼ)
#
# Both terms have magnitude ~10⁶ × ‖state‖. Their difference is O(‖state‖).
# This is textbook catastrophic cancellation: we subtract two numbers that
# agree to 6 significant digits, losing 6 of the ~16 digits available in
# float64. Each predict-update step introduces ~10⁻¹⁰ error in float64,
# which compounds over T steps. In our 50-step test with α=10⁻³, the
# accumulated discrepancy reaches ~3×10⁻⁴, far exceeding the 10⁻⁶
# tolerance needed to confirm mathematical equivalence.
#
# This is the well-known "negative zeroth weight" problem of the standard UT
# (see Julier & Uhlmann 2004, "Unscented Filtering and Nonlinear Estimation",
# Section IV-B). People accept it in practice because for nonlinear systems,
# the higher-order truncation error of the UT (which small α minimizes)
# dominates over the floating-point cancellation error. But for our purpose
# of testing mathematical equivalence on a linear system, we need to
# separate the two concerns.
#
# TEST STRATEGY — TWO COMPLEMENTARY ASSERTIONS
#
# We run two UKF tests:
#
# (a) Default α = 10⁻³ with loose tolerance (atol=5×10⁻⁴):
#     Confirms the implementation is correct "in the large" even with the
#     standard parameterization. The tolerance is set just above the observed
#     cancellation error (~3×10⁻⁴). This is the configuration users will
#     actually run, so it validates that the filter does not diverge or
#     produce qualitatively wrong results.
#
# (b) α = 1 with tight tolerance (atol=10⁻⁶):
#     With α = 1: λ = 1²(3+0) - 3 = 0, so n+λ = 3. The weights become:
#         Wᵐ₀ = 0, Wᶜ₀ = 0 + (1 - 1 + 2) = 2, Wᵢ = 1/6
#     No large magnitudes, no cancellation. This cleanly tests the
#     mathematical identity UT(linear) ≡ KF without numerical noise.
#     If this test fails, the sigma point generation, weight computation,
#     or the predict/update algebra has a genuine bug.
# ============================================================================

class TestUKFReducesToKF:
    """UKF should produce identical results to KF on a linear system."""

    def _make_ukf(self, linear_system, alpha=1e-3, kappa=0.0):
        F, B, H, D, mean_0, Sigma_0 = linear_system
        model = LinearGaussianModel(
            F, B, H, D, mu_0=mean_0, Sigma_0=Sigma_0, dtype=tf.float64
        )
        return UnscentedKalmanFilter(
            model, mean_0=mean_0, Sigma_0=Sigma_0, sample_initial_mean=False,
            alpha=alpha, kappa=kappa
        )

    # --- (a) Default alpha: loose tolerance, tests practical correctness ---

    def test_means_match_default_alpha(self, linear_system, observations, kf_result):
        """With default alpha=1e-3, UKF matches KF within cancellation noise."""
        ukf = self._make_ukf(linear_system, alpha=1e-3)
        ukf_result = ukf.filter(observations)
        np.testing.assert_allclose(
            ukf_result.means, kf_result.means, atol=5e-4,
            err_msg="UKF means (default alpha) diverge beyond cancellation noise"
        )

    def test_covariances_match_default_alpha(self, linear_system, observations, kf_result):
        """With default alpha=1e-3, UKF covariances match KF within cancellation noise."""
        ukf = self._make_ukf(linear_system, alpha=1e-3)
        ukf_result = ukf.filter(observations)
        np.testing.assert_allclose(
            ukf_result.covs, kf_result.covs, atol=5e-4,
            err_msg="UKF covariances (default alpha) diverge beyond cancellation noise"
        )

    # --- (b) Alpha=1: tight tolerance, tests mathematical equivalence ---

    def test_means_match_alpha_one(self, linear_system, observations, kf_result):
        """With alpha=1 (no cancellation), UKF must match KF to near precision."""
        ukf = self._make_ukf(linear_system, alpha=1.0)
        ukf_result = ukf.filter(observations)
        np.testing.assert_allclose(
            ukf_result.means, kf_result.means, atol=1e-6,
            err_msg="UKF means (alpha=1) diverge from KF — sigma point or weight bug"
        )

    def test_covariances_match_alpha_one(self, linear_system, observations, kf_result):
        """With alpha=1 (no cancellation), UKF covariances must match KF to near precision."""
        ukf = self._make_ukf(linear_system, alpha=1.0)
        ukf_result = ukf.filter(observations)
        np.testing.assert_allclose(
            ukf_result.covs, kf_result.covs, atol=1e-6,
            err_msg="UKF covariances (alpha=1) diverge from KF — sigma point or weight bug"
        )


# ============================================================================
# TEST 3: Joseph form preserves positive-definiteness under ill-conditioning
#
# Why: The only real numerical failure mode for KF. The standard covariance
# update formula P = (I - KH)P is algebraically correct but numerically
# dangerous: the matrix (I - KH) can break symmetry due to floating-point
# rounding, and once symmetry is lost, subsequent Cholesky/eigenvalue
# computations fail or produce negative "variances".
#
# The Joseph form rewrites the update as:
#
#     P = (I - KH) P (I - KH)ᵀ + K R Kᵀ
#
# This is algebraically equivalent, but it is a sum of two terms that are
# each symmetric positive semi-definite by construction (ABAᵀ is always
# symmetric for any A, B symmetric). The result is therefore guaranteed
# to be symmetric and at least positive semi-definite, regardless of
# rounding errors in K or H. The cost is roughly double the flops of
# the standard form, but it eliminates the only numerical failure mode
# of the KF.
#
# Setup: high-dimensional state (nx=10), wide process noise scale range
# (1e-3 to 1), partial observations (3 of 10), long sequence (T=500).
# The wide noise scale range creates a covariance matrix with condition
# number ~10⁶ (ratio of largest to smallest diagonal of B·Bᵀ is 10⁶),
# and partial observation means some state components are never directly
# observed, creating a mix of well-constrained and poorly-constrained
# directions in state space. This is a realistic stress test.
# ============================================================================

class TestJosephFormPD:
    """Joseph-form covariance update must keep all eigenvalues positive."""

    def test_pd_preserved_ill_conditioned(self):
        nx = 10
        ny = 3  # partial observations

        rng = np.random.default_rng(123)

        # Random asymmetric F, scaled so spectral radius < 1 (stable)
        F_raw = 0.5 * rng.standard_normal((nx, nx))
        spectral_radius = np.max(np.abs(np.linalg.eigvals(F_raw)))
        F = F_raw / (1.1 * spectral_radius)

        # Wide noise scale range: B diagonal from 1e-3 to 1
        B = np.diag(np.logspace(-3, 0, nx))

        # Partial observation: only observe states 0, 3, 7
        H = np.zeros((ny, nx))
        H[0, 0] = 1.0
        H[1, 3] = 1.0
        H[2, 7] = 1.0

        # Tight observation noise
        D = 0.01 * np.eye(ny)

        kf = KalmanFilter(
            F, B, H, D, use_joseph_form=True, dtype=tf.float64
        )

        T = 500
        observations = rng.standard_normal((T, ny))
        result = kf.filter(observations)

        for t in range(T):
            eigvals = np.linalg.eigvalsh(result.covs[t])
            assert np.all(eigvals > 0), (
                f"Step {t}: min eigenvalue = {eigvals.min():.2e}"
            )


# ============================================================================
# TEST 4: KF reproduces analytical results
#
# The previous tests only cross-validate KF against EKF/UKF (relative
# correctness). This test verifies absolute correctness: the KF output
# must match the closed-form Kalman filter equations computed by hand.
#
# Two sub-tests:
#
# (a) Scalar system — step-by-step analytical recursion
#     For a 1D model x_{k+1} = F·x_k + w, y_k = x_k + v, with
#     w ~ N(0,Q) and v ~ N(0,R), the Kalman recursion is:
#
#       Predict:   μ_pred = F·μ
#                  P_pred = F²·P + Q
#
#       Update:    K = P_pred / (P_pred + R)
#                  μ_upd = μ_pred + K·(y - μ_pred)
#                  P_upd = (1 - K)·P_pred
#
#     These are scalar formulas with no matrix algebra, so we can iterate
#     them in pure Python/NumPy and compare against the KF implementation.
#
# (b) Steady-state covariance converges to DARE solution
#     After many steps of a stable LTI system, the KF covariance converges
#     to the unique stabilizing solution of the Discrete Algebraic Riccati
#     Equation (DARE):
#
#       P_∞ = F·(P_∞ - P_∞·Hᵀ·(H·P_∞·Hᵀ + R)⁻¹·H·P_∞)·Fᵀ + Q
#
#     scipy.linalg.solve_discrete_are computes this. Convergence is
#     geometric with rate equal to the spectral radius of F·(I - K_∞·H),
#     so for a stable system with informative observations, 200 steps is
#     more than enough.
# ============================================================================

class TestKFAnalytical:
    """KF must reproduce known analytical results."""

    def test_scalar_step_by_step(self):
        """KF matches hand-computed scalar Kalman recursion for 20 steps.

        Model: x_{k+1} = 0.9·x_k + w, y_k = x_k + v
               w ~ N(0, 0.01), v ~ N(0, 0.04)
               Prior: x_0 ~ N(0, 1)
        """
        F_val, Q_val, R_val = 0.9, 0.01, 0.04
        mu, P = 0.0, 1.0  # prior

        # Generate observations from a known true state sequence
        rng = np.random.default_rng(99)
        T = 20
        true_states = np.zeros(T + 1)
        observations = np.zeros(T)
        true_states[0] = rng.normal(mu, np.sqrt(P))
        for t in range(T):
            true_states[t + 1] = F_val * true_states[t] + rng.normal(0, np.sqrt(Q_val))
            observations[t] = true_states[t + 1] + rng.normal(0, np.sqrt(R_val))

        # Analytical recursion (scalar formulas)
        mu_analytical = np.zeros(T)
        P_analytical = np.zeros(T)
        for t in range(T):
            # Predict
            mu_pred = F_val * mu
            P_pred = F_val**2 * P + Q_val
            # Update
            K = P_pred / (P_pred + R_val)
            mu = mu_pred + K * (observations[t] - mu_pred)
            P = (1 - K) * P_pred
            mu_analytical[t] = mu
            P_analytical[t] = P

        # KF implementation
        kf = KalmanFilter(
            F=[[F_val]], B=[[np.sqrt(Q_val)]],
            H=[[1.0]], D=[[np.sqrt(R_val)]],
            mean_0=[0.0], Sigma_0=[[1.0]],
            dtype=tf.float64,
        )
        result = kf.filter(observations.reshape(T, 1))

        np.testing.assert_allclose(
            result.means[:, 0], mu_analytical, atol=1e-10,
            err_msg="KF means diverge from scalar analytical recursion"
        )
        np.testing.assert_allclose(
            result.covs[:, 0, 0], P_analytical, atol=1e-10,
            err_msg="KF covariances diverge from scalar analytical recursion"
        )

    def test_steady_state_converges_to_dare(self):
        """After 200 steps, KF covariance matches the DARE solution.

        Uses the same 3D system from the cross-validation tests.
        The DARE solution is the unique positive-definite fixed point of the
        Kalman covariance recursion for a detectable, stabilizable system.
        """
        from scipy.linalg import solve_discrete_are

        F = np.array([
            [0.9, 0.2, 0.0],
            [0.0, 0.8, 0.3],
            [0.0, 0.0, 0.7]
        ])
        B = np.array([
            [0.1, 0.0],
            [0.0, 0.1],
            [0.0, 0.05]
        ])
        H = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0]
        ])
        D = np.array([
            [0.2, 0.0],
            [0.0, 0.15]
        ])

        Q = B @ B.T
        R = D @ D.T

        # DARE: solve for steady-state predicted covariance P_∞
        # scipy convention: P = F P Fᵀ - F P Hᵀ (H P Hᵀ + R)⁻¹ H P Fᵀ + Q
        P_dare = solve_discrete_are(F.T, H.T, Q, R)

        # Run KF for 200 steps with dummy observations
        rng = np.random.default_rng(42)
        T = 200
        observations = rng.standard_normal((T, 2))

        kf = KalmanFilter(
            F, B, H, D,
            mean_0=np.zeros(3), Sigma_0=np.eye(3),
            dtype=tf.float64,
        )
        result = kf.filter(observations)

        # The filter stores the *updated* (posterior) covariance.
        # The DARE gives the *predicted* (prior) covariance P_∞.
        # Updated covariance = (I - K_∞ H) P_∞ in standard form.
        # Alternatively, run one more predict step to get P_pred.
        kf_P_updated = result.covs[-1]

        # Compute expected updated covariance from DARE
        S = H @ P_dare @ H.T + R
        K_dare = P_dare @ H.T @ np.linalg.inv(S)
        P_updated_dare = (np.eye(3) - K_dare @ H) @ P_dare

        np.testing.assert_allclose(
            kf_P_updated, P_updated_dare, rtol=1e-3,
            err_msg="KF covariance did not converge to DARE steady-state"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
