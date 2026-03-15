"""Utilities for data generation and model testing."""

import numpy as np
from typing import Tuple, Optional
from ..core.model_base import StateSpaceModel


def generate_data(
    model: StateSpaceModel,
    T: int = 500,
    rng: Optional[np.random.Generator] = None,
    observe_initial: bool = True,
    use_true_process_noise: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a state-space model.

    Args:
        model: State-space model instance
        T: Number of time steps
        rng: Optional numpy random generator for reproducibility
        observe_initial: If True (default), observations[0] is at initial state.
                        If False, observations[0] is after first transition.
                        Use False for predict-then-update filters (e.g., kernel flow).
        use_true_process_noise: If True and model has V_true, use V_true for data generation.
                               This is important for paper reproduction where filters use
                               larger process noise (V_filter) than the true data generation (V_true).

    Returns:
        Tuple of (true_states, observations) where:
        - true_states: Array of shape (T, state_dim)
        - observations: Array of shape (T, obs_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    # If model has V_true, temporarily switch Q for data generation
    # Only do this if model has Q attribute (some models like StochasticVolatilityModel don't)
    original_Q = None
    if hasattr(model, 'Q'):
        original_Q = model.Q
        if use_true_process_noise and hasattr(model, 'V_true'):
            model.Q = model.V_true

    true_states = np.zeros((T, model.state_dim))
    observations = np.zeros((T, model.obs_dim))

    if observe_initial:
        # Original behavior: observations[0] at initial state
        true_states[0] = model.sample_initial_state(rng)
        observations[0] = model.sample_observation(true_states[0], rng)

        for t in range(1, T):
            true_states[t] = model.sample_state_transition(true_states[t-1], rng)
            observations[t] = model.sample_observation(true_states[t], rng)
    else:
        # New behavior: observations[0] after first transition
        # For predict-then-update filters
        current_state = model.sample_initial_state(rng)

        for t in range(T):
            current_state = model.sample_state_transition(current_state, rng)
            true_states[t] = current_state
            observations[t] = model.sample_observation(current_state, rng)

    # Restore original Q
    if original_Q is not None and hasattr(model, 'V_true'):
        model.Q = original_Q

    return true_states, observations
