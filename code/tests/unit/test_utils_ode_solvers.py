"""
Tests for src/utils/ode_solvers.py — ODE/SDE integration steps.

# Run and save results:
#   .venv/bin/python -m pytest tests/test_utils_ode_solvers.py -v 2>&1 | tee tests/test_utils_ode_solvers_results.txt
"""

import pytest
import numpy as np
import tensorflow as tf

from src.utils.ode_solvers import euler_step, rk4_step, euler_maruyama_step, integrate_ode


# ============================================================================
# euler_step — T2.1 to T2.3
# ============================================================================

class TestEulerStep:

    def test_scalar_linear_ode(self):
        """T2.1 — Scalar linear ODE x' = -x, Euler step vs exact exp(-dt).

        For x' = a*x with a = -1, the exact solution is x(dt) = x0 * exp(-dt).
        Euler gives x1 = x0 + a*x0*dt = x0*(1 - dt).  For dt=0.01 the
        local truncation error is O(dt^2) ≈ 1e-4, so relative error ~ dt.
        """
        def f(x, a):
            return a * x

        x0 = tf.constant([2.0], dtype=tf.float64)
        dt = 0.01
        a = tf.constant(-1.0, dtype=tf.float64)
        x1 = euler_step(x0, f, dt, a)
        expected = 2.0 * np.exp(-0.01)
        np.testing.assert_allclose(x1.numpy(), [expected], rtol=dt)

    def test_batch_particles(self):
        """T2.2 — Batch particles: (N, d) input → (N, d) output."""
        def drift(x):
            return -x

        particles = tf.ones([50, 2], dtype=tf.float64)
        result = euler_step(particles, drift, 0.01)
        assert result.shape == (50, 2)
        assert np.all(np.isfinite(result.numpy()))

    def test_zero_drift(self):
        """T2.3 — Zero drift: state is unchanged after step."""
        x = tf.constant([3.0, -1.0], dtype=tf.float64)
        result = euler_step(x, lambda x: tf.zeros_like(x), 0.1)
        np.testing.assert_allclose(result.numpy(), x.numpy())


# ============================================================================
# rk4_step — T2.4 to T2.5
# ============================================================================

class TestRK4Step:

    def test_more_accurate_than_euler(self):
        """T2.4 — RK4 is more accurate than Euler for x' = -x with large dt.

        With dt=0.5, exact = exp(-0.5) ≈ 0.6065.
        Euler: x1 = 1*(1-0.5) = 0.5, error ≈ 0.107.
        RK4 local error is O(dt^5) ≈ 3e-5, much smaller.
        """
        def f(x, a):
            return a * x

        dt = 0.5
        x0 = tf.constant([1.0], dtype=tf.float64)
        a = tf.constant(-1.0, dtype=tf.float64)
        exact = np.exp(-0.5)

        euler_result = euler_step(x0, f, dt, a).numpy()[0]
        rk4_result = rk4_step(x0, f, dt, a).numpy()[0]

        euler_err = abs(euler_result - exact)
        rk4_err = abs(rk4_result - exact)
        assert rk4_err < euler_err, (
            f"RK4 error ({rk4_err:.6e}) should be smaller than Euler error ({euler_err:.6e})"
        )

    def test_time_dependent_drift(self):
        """T2.5 — Time-dependent drift: x'(t) = -t.

        Exact solution: x(t) = x(0) - t^2/2.
        At t=1: x(1) = 5.0 - 0.5 = 4.5.
        RK4 is 4th-order accurate, so with dt=1.0 on this quadratic ODE
        the result should be essentially exact (polynomial of degree 2).
        """
        def f_t(x, t):
            return -t * tf.ones_like(x)

        x0 = tf.constant([5.0], dtype=tf.float64)
        x_rk4 = rk4_step(x0, f_t, dt=1.0, t=tf.constant(0.0, dtype=tf.float64))
        expected = 5.0 - 0.5
        np.testing.assert_allclose(x_rk4.numpy(), [expected], rtol=1e-10)


# ============================================================================
# euler_maruyama_step — T2.6 to T2.7
# ============================================================================

class TestEulerMaruyamaStep:

    def test_zero_diffusion_equals_euler(self):
        """T2.6 — Zero diffusion coefficient reduces to deterministic Euler."""
        def f(x, a):
            return a * x

        x0 = tf.constant([2.0], dtype=tf.float64)
        a = tf.constant(-1.0, dtype=tf.float64)
        dt = 0.01

        em_result = euler_maruyama_step(x0, f, dt, a, diffusion_coeff=0.0, seed=None)
        euler_result = euler_step(x0, f, dt, a)
        np.testing.assert_allclose(em_result.numpy(), euler_result.numpy())

    def test_noise_variance_scaling(self):
        """T2.7 — Noise variance scales as diffusion_coeff * dt.

        For zero drift, the update is purely noise: x1 = x0 + sqrt(q*dt) * N(0,1).
        So Var[x1 - x0] = q * dt.  With q=4.0, dt=1.0, Var = 4.0.
        We use N=10000 samples and check the empirical variance.
        """
        def zero_drift(x):
            return tf.zeros_like(x)

        x0 = tf.constant([0.0], dtype=tf.float64)
        q = 4.0
        dt = 1.0
        N = 10000

        deltas = []
        for i in range(N):
            seed = tf.constant([i, 0], dtype=tf.int32)
            x1 = euler_maruyama_step(x0, zero_drift, dt, diffusion_coeff=q, seed=seed)
            deltas.append((x1 - x0).numpy()[0])

        empirical_var = np.var(deltas)
        np.testing.assert_allclose(empirical_var, q * dt, rtol=0.1)


# ============================================================================
# integrate_ode — T2.8 to T2.9
# ============================================================================

class TestIntegrateODE:

    def test_euler_exponential_decay(self):
        """T2.8 — Euler integration of x' = -x from t=0 to t=1.

        Exact: x(1) = exp(-1) ≈ 0.3679.
        With 1000 Euler steps (dt = 1e-3), global error is O(dt) = O(1e-3).
        """
        def f_decay(x):
            return -x

        x0 = tf.constant([1.0], dtype=tf.float64)
        x_final = integrate_ode(x0, f_decay, (0.0, 1.0), 1000, method='euler')
        np.testing.assert_allclose(x_final.numpy(), [np.exp(-1.0)], rtol=1e-2)

    def test_rk4_exponential_decay(self):
        """T2.9 — RK4 integration of x' = -x, tighter tolerance with fewer steps.

        With 100 RK4 steps (dt = 0.01), global error is O(dt^4) = O(1e-8).
        """
        def f_decay(x):
            return -x

        x0 = tf.constant([1.0], dtype=tf.float64)
        x_final = integrate_ode(x0, f_decay, (0.0, 1.0), 100, method='rk4')
        np.testing.assert_allclose(x_final.numpy(), [np.exp(-1.0)], rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
