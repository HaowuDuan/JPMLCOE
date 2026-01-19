"""ODE solvers for particle flow integration."""

import numpy as np
from typing import Callable, Any


def euler_step(x: np.ndarray, f: Callable, dt: float, *args: Any) -> np.ndarray:
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


def rk4_step(x: np.ndarray, f: Callable, dt: float, *args: Any) -> np.ndarray:
    """
    Single Runge-Kutta 4th order integration step.

    Args:
        x: Current state of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, *args)
        dt: Time step
        *args: Additional arguments to f

    Returns:
        x_{n+1} of same shape as x
    """
    k1 = f(x, *args)
    k2 = f(x + 0.5 * dt * k1, *args)
    k3 = f(x + 0.5 * dt * k2, *args)
    k4 = f(x + dt * k3, *args)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_ode(x0: np.ndarray, f: Callable, t_span: tuple, n_steps: int,
                 method: str = 'rk4', *args: Any) -> np.ndarray:
    """
    Integrate ODE from t_start to t_end.

    Args:
        x0: Initial condition of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, t, *args)
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
    for _ in range(n_steps):
        if method == 'euler':
            x = euler_step(x, f, dt, *args)
        elif method == 'rk4':
            x = rk4_step(x, f, dt, *args)
        else:
            raise ValueError(f"Unknown method: {method}")

    return x
