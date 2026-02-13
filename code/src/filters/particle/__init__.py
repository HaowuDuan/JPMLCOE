"""Particle filter implementations."""

from .particle_base import ParticleFilterBase
from .bootstrap_pf_tf import ParticleFilterTF
from .edh_flow import ExactDaumHuangFlow
from .ledh_flow import LocalExactDaumHuangFlow
from .edh_invertible import EDHParticleFlowFilter
from .ledh_invertible import LEDHParticleFlowFilter
from .kernel_flow import KernelMappingPF
from .stochastic_edh import StochasticEDHFlow

# TensorFlow Probability particle filter (optional)
try:
    from .particle_tfp import ParticleFilterTFP
    _TFP_AVAILABLE = True
except ImportError:
    _TFP_AVAILABLE = False

__all__ = [
    'ParticleFilterBase',
    'ParticleFilterTF',
    'ExactDaumHuangFlow',
    'LocalExactDaumHuangFlow',
    'EDHParticleFlowFilter',
    'LEDHParticleFlowFilter',
    'KernelMappingPF',
    'StochasticEDHFlow',
]

if _TFP_AVAILABLE:
    __all__.append('ParticleFilterTFP')
