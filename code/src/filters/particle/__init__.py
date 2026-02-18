"""Particle filter implementations."""

from .particle_base import ParticleFilterBase
from .bootstrap_pf_tf import ParticleFilterTF
from .edh_flow import ExactDaumHuangFlow
from .edh_flow_global import ExactDaumHuangFlowglobal
from .ledh_flow import LocalExactDaumHuangFlow
from .edh_invertible import EDHParticleFlowFilter
from .ledh_invertible import LEDHParticleFlowFilter
from .ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from .kernel_flow import KernelMappingPF
from .stochastic_edh import StochasticEDHFlow
from .stochastic_edh_paper import StochasticEDHFlowPaper
from .sde_local_correction import SDELocalCorrection

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
    'ExactDaumHuangFlowglobal',
    'LocalExactDaumHuangFlow',
    'EDHParticleFlowFilter',
    'LEDHParticleFlowFilter',
    'LEDHParticleFlowFilterHMC',
    'KernelMappingPF',
    'StochasticEDHFlow',
    'StochasticEDHFlowPaper',
    'SDELocalCorrection',
]

if _TFP_AVAILABLE:
    __all__.append('ParticleFilterTFP')
