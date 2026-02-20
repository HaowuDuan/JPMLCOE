"""Shared resampling method resolution for particle filters."""

import numpy as np
from ..resampling import systematic_resample, soft_resample, ot_entropy_resample

_METHOD_MAP = {
    'systematic': systematic_resample,
    'soft': soft_resample,
    'ot_entropy': ot_entropy_resample,
}


def resolve_resampling(resampling_method, resampling_config):
    """
    Resolve resampling method string/callable and sanitize config scalars.

    Returns:
        (method_fn, method_name, config_dict)
    """
    if isinstance(resampling_method, str):
        method_fn = _METHOD_MAP.get(resampling_method, systematic_resample)
        method_name = resampling_method
    elif resampling_method is not None:
        method_fn = resampling_method
        method_name = getattr(resampling_method, '__name__', 'custom')
    else:
        method_fn = systematic_resample
        method_name = 'systematic'

    config = {}
    if resampling_config is not None:
        for key, value in resampling_config.items():
            if isinstance(value, (int, np.integer)):
                config[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                config[key] = float(value)
            else:
                config[key] = value

    return method_fn, method_name, config
