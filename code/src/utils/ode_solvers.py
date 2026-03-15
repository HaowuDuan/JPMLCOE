"""ODE solvers for particle flow integration - TensorFlow version."""

import tensorflow as tf
from typing import Callable, Any, Optional


@tf.function(jit_compile=True)
def euler_step(x: tf.Tensor, f: Callable, dt: float, *args: Any) -> tf.Tensor:
    """
    Single Euler integration step: x_{n+1} = x_n + f(x_n, *args) * dt

    Args:
        x: Current state of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, *args)
        dt: Time step
        *args: Additional arguments to f

    Returns:
        x_{n+1} of same shape as x
    """
    return x + f(x, *args) * dt


@tf.function(jit_compile=True)
def rk4_step(x: tf.Tensor, f: Callable, dt: float, *args: Any,
             t: Optional[float] = None) -> tf.Tensor:
    """
    Single Runge-Kutta 4th order integration step.

    Args:
        x: Current state of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, *args) or f(x, t, *args) if t is provided
        dt: Time step
        *args: Additional arguments to f
        t: Current time value (optional, for time-dependent drift)

    Returns:
        x_{n+1} of same shape as x
    """
    if t is not None:
        # Time-dependent drift: f(x, t, *args)
        k1 = f(x, t, *args)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, *args)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, *args)
        k4 = f(x + dt * k3, t + dt, *args)
    else:
        # Standard RK4 with fixed args
        k1 = f(x, *args)
        k2 = f(x + 0.5 * dt * k1, *args)
        k3 = f(x + 0.5 * dt * k2, *args)
        k4 = f(x + dt * k3, *args)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@tf.function(jit_compile=True)
def euler_maruyama_step(x: tf.Tensor, f: Callable, dt: float, *args: Any,
                        diffusion_coeff: float = 0.0,
                        seed: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Single Euler-Maruyama integration step for SDEs.

    Integrates: dx = f(x, *args) dt + sqrt(diffusion_coeff) dW

    Args:
        x: Current state of shape (n,) or (batch, n)
        f: Drift function computing f(x, *args)
        dt: Time step
        *args: Additional arguments to f
        diffusion_coeff: Scalar diffusion coefficient q for isotropic noise sqrt(q)*dW.
                        If 0, reduces to deterministic Euler step.
        seed: TensorFlow random seed for stateless sampling

    Returns:
        x_{n+1} of same shape as x
    """
    drift = f(x, *args)

    if diffusion_coeff > 0 and seed is not None:
        noise = tf.random.stateless_normal(tf.shape(x), seed=seed, dtype=x.dtype)
        noise = noise * tf.sqrt(tf.cast(dt, x.dtype)) * tf.sqrt(tf.cast(diffusion_coeff, x.dtype))
        return x + drift * dt + noise
    else:
        return x + drift * dt


@tf.function(jit_compile=True)
def integrate_ode(x0: tf.Tensor, f: Callable, t_span: tuple, n_steps: int,
                 method: str = 'rk4', *args: Any) -> tf.Tensor:
    """
    Integrate ODE from t_start to t_end using TensorFlow.

    Args:
        x0: Initial condition of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, *args)
        t_span: (t_start, t_end)
        n_steps: Number of integration steps
        method: 'euler' or 'rk4'
        *args: Additional arguments to f

    Returns:
        Final state x(t_end) of same shape as x0
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps

    x = x0
    for _ in tf.range(n_steps):
        if method == 'euler':
            x = euler_step(x, f, dt, *args)
        elif method == 'rk4':
            x = rk4_step(x, f, dt, *args)
        else:
            raise ValueError(f"Unknown method: {method}")

    return x
