"""Utilities for data generation and model testing."""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional
from ..core.model_base import StateSpaceModel


def _make_seed(rng: np.random.Generator) -> tf.Tensor:
    """Create a TF stateless random seed from a numpy rng."""
    return tf.constant(rng.integers(0, 2**31, size=2), dtype=tf.int32)


def generate_data(
    model: StateSpaceModel,
    T: int = 500,
    rng: Optional[np.random.Generator] = None,
    use_true_process_noise: bool = True,
    use_fixed_initial_state: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a state-space model.

    Uses the standard filtering convention:
        x_0 ~ p(x_0)                                   (initial state)
        x_t = f(x_{t-1}) + process_noise,  t = 1..T    (state transitions)
        y_t = h(x_t) + obs_noise,          t = 1..T    (observations)

    All filters in this codebase follow predict-then-update, so observations
    must correspond to states after the first transition (not the initial state).

    Args:
        model: State-space model instance (TensorFlow-based)
        T: Number of time steps (produces T transitions and T observations)
        rng: Optional numpy random generator for reproducibility
        use_true_process_noise: If True and model has V_true, use V_true for
            data generation. Important for paper reproduction where filters
            use larger process noise (V_filter) than true generation (V_true).
        use_fixed_initial_state: If True, use model.mu_0 as fixed initial state.
            If False (default), sample from initial distribution.

    Returns:
        Tuple of (initial_state, true_states, observations) where:
        - initial_state: Array of shape (state_dim,) — x_0
        - true_states: Array of shape (T, state_dim) — [x_1, ..., x_T]
        - observations: Array of shape (T, obs_dim) — [y_1, ..., y_T]
    """
    if rng is None:
        rng = np.random.default_rng()

    # If model has V_true, temporarily switch Q for data generation
    original_Q = None
    if hasattr(model, 'Q'):
        original_Q = model.Q
        if use_true_process_noise and hasattr(model, 'V_true'):
            model.Q = model.V_true

    true_states = np.zeros((T, model.state_dim))
    observations = np.zeros((T, model.obs_dim))

    # Get initial state x_0 (fixed or sampled)
    if use_fixed_initial_state:
        if not hasattr(model, 'mu_0'):
            raise ValueError("Model must have mu_0 attribute for fixed initial state")
        initial_state = np.asarray(model.mu_0, dtype=np.float64)
    else:
        initial_state = np.asarray(
            model.sample_initial_state(_make_seed(rng)), dtype=np.float64
        )

    # Generate x_1, ..., x_T and y_1, ..., y_T
    current_state = tf.constant(initial_state, dtype=tf.float32)

    for t in range(T):
        current_state = model.sample_state_transition(current_state, _make_seed(rng))
        true_states[t] = np.asarray(current_state)
        observations[t] = np.asarray(
            model.sample_observation(current_state, _make_seed(rng))
        )

    # Restore original Q
    if original_Q is not None and hasattr(model, 'V_true'):
        model.Q = original_Q

    return initial_state, true_states, observations
