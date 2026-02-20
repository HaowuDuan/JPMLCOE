"""Named constants for particle flow filters.

Uses frozen dataclasses to group related configuration values that were
previously hardcoded as magic numbers throughout the filter implementations.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FlowScheduleConfig:
    """Configuration for the exponential lambda schedule (Li & Coates 2017)."""
    geometric_ratio: float = 1.2  # Paper's recommended ratio for exponential schedule


@dataclass(frozen=True)
class DriftClipConfig:
    """Configuration for drift and particle clipping in LEDH flow."""
    max_drift_norm: float = 100.0      # Maximum drift magnitude per flow step
    max_particle_norm: float = 1000.0  # Maximum particle distance from origin
    epsilon: float = 1e-10             # Avoid division by zero in norm clipping


@dataclass(frozen=True)
class BVPShootingConfig:
    """Configuration for BVP shooting solver in stiffness-mitigating schedule."""
    n_ode_steps: int = 500         # Euler steps per shooting integration
    n_bisection: int = 40          # Bisection iterations for initial velocity
    bracket_lo: float = 0.1        # Initial lower bracket
    bracket_hi: float = 20.0       # Initial upper bracket
    bracket_hi_max: float = 50.0   # Extended upper bracket if initial fails
    bracket_lo_min: float = 0.01   # Extended lower bracket if initial fails
