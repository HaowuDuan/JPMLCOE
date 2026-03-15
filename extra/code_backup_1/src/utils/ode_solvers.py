"""ODE solvers for particle flow integration."""

import numpy as np
from typing import Callable, Any, Optional


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


def rk4_step(x: np.ndarray, f: Callable, dt: float, *args: Any,
             t: Optional[float] = None, t_arg_index: Optional[int] = None) -> np.ndarray:
    """
    Single Runge-Kutta 4th order integration step.

    Supports time-dependent drift functions where the time parameter needs to
    advance at each intermediate stage (k1, k2, k3, k4).

    Args:
        x: Current state of shape (n,) or (batch, n)
        f: Function computing dx/dt = f(x, *args)
           If t_arg_index is specified, args[t_arg_index] is the time parameter
           that will be advanced at each RK4 stage.
        dt: Time step
        *args: Additional arguments to f
        t: Current time value (alternative to t_arg_index for time-dependent drift).
           If provided, f should accept time as first positional arg after x: f(x, t, *args)
        t_arg_index: Index in args where the time parameter is located.
           If provided, that argument will be advanced by dt/2 for k2/k3 and dt for k4.

    Returns:
        x_{n+1} of same shape as x

    Examples:
        # Time-independent drift (standard RK4):
        rk4_step(x, f, dt, A, b)

        # Time-dependent drift with t as separate argument:
        rk4_step(x, lambda x, t, A, b: drift(x, A, b, t), dt, A, b, t=lambda_val)

        # Time-dependent drift with t in args:
        rk4_step(x, f, dt, lambda_val, observation, t_arg_index=0)
    """
    if t is not None:
        # Time passed as separate argument - f signature is f(x, t, *args)
        k1 = f(x, t, *args)
        k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt, *args)
        k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt, *args)
        k4 = f(x + dt * k3, t + dt, *args)
    elif t_arg_index is not None:
        # Time is in args at specified index
        args_list = list(args)
        t_val = args_list[t_arg_index]

        k1 = f(x, *args)

        args_list[t_arg_index] = t_val + 0.5 * dt
        k2 = f(x + 0.5 * dt * k1, *tuple(args_list))

        k3 = f(x + 0.5 * dt * k2, *tuple(args_list))

        args_list[t_arg_index] = t_val + dt
        k4 = f(x + dt * k3, *tuple(args_list))
    else:
        # Standard RK4 with fixed args
        k1 = f(x, *args)
        k2 = f(x + 0.5 * dt * k1, *args)
        k3 = f(x + 0.5 * dt * k2, *args)
        k4 = f(x + dt * k3, *args)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def euler_maruyama_step(x: np.ndarray, f: Callable, dt: float, *args: Any,
                        diffusion_coeff: float = 0.0,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
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
        rng: NumPy random generator for reproducibility. If None, uses np.random.

    Returns:
        x_{n+1} of same shape as x
    """
    drift = f(x, *args)

    if diffusion_coeff > 0:
        if rng is not None:
            noise = rng.normal(0, np.sqrt(dt), size=x.shape)
        else:
            noise = np.random.normal(0, np.sqrt(dt), size=x.shape)
        diffusion = noise * np.sqrt(diffusion_coeff)
    else:
        diffusion = 0.0

    return x + drift * dt + diffusion


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
