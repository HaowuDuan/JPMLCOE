"""Particle filter implementations."""

from .particle_base import ParticleFilterBase
from .bootstrap_pf import ParticleFilter
from .edh_flow import ExactDaumHuangFlow
from .ledh_flow import LocalExactDaumHuangFlow
from .kernel_local_flow import KernelLocalFlow
from .edh_invertible import EDHParticleFlowFilter
from .ledh_invertible import LEDHParticleFlowFilter
from .kernel_flow import KernelMappingPF
from .stochastic_edh import StochasticEDHFlow

# TensorFlow-based filters (optional)
try:
    from .bootstrap_pf_tf import ParticleFilterTF
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# TensorFlow Probability particle filter (optional)
try:
    from .particle_tfp import ParticleFilterTFP
    _TFP_AVAILABLE = True
except ImportError:
    _TFP_AVAILABLE = False

__all__ = [
    'ParticleFilterBase',
    'ParticleFilter',
    'ExactDaumHuangFlow',
    'LocalExactDaumHuangFlow',
    'KernelLocalFlow',
    'EDHParticleFlowFilter',
    'LEDHParticleFlowFilter',
    'KernelMappingPF',
    'StochasticEDHFlow',
]

if _TF_AVAILABLE:
    __all__.append('ParticleFilterTF')

if _TFP_AVAILABLE:
    __all__.append('ParticleFilterTFP')
