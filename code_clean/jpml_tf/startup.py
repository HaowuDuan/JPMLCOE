"""Process-level TensorFlow configuration.

Call ``configure_tf()`` once per process, at the top of every entry
point (CLI, test, notebook), BEFORE importing any computational
submodule of ``jpml_tf`` or creating any TensorFlow tensor. It locks
the global numeric mode to ``float64``, configures device visibility,
and optionally enables deterministic ops.

The function is idempotent: repeated calls reapply the requested
settings without raising.
"""

from __future__ import annotations

import os


_CONFIGURED = False


def configure_tf(
    *,
    force_cpu: bool = False,
    allow_gpu_growth: bool = True,
    deterministic_ops: bool = False,
    legacy_keras: bool = True,
) -> None:
    """Lock global TF numeric and device policy to rebuild defaults.

    Effects, in order:
    1. Set ``TF_USE_LEGACY_KERAS`` to ``"1"`` if ``legacy_keras`` else ``"0"``.
       TensorFlow inspects this environment variable on its first import;
       changing it after TF is loaded has no further effect.
    2. Import TensorFlow.
    3. Hide all GPUs from TF if ``force_cpu`` is True; else, if
       ``allow_gpu_growth`` is True, request incremental memory allocation
       on every visible GPU.
    4. Enable deterministic ops if ``deterministic_ops`` is True (slower).
    5. Set the keras default float dtype to ``"float64"``.
    6. Set the keras mixed-precision global policy to ``"float64"``.
    """
    global _CONFIGURED

    os.environ["TF_USE_LEGACY_KERAS"] = "1" if legacy_keras else "0"

    import tensorflow as tf

    if force_cpu:
        tf.config.set_visible_devices([], "GPU")
    elif allow_gpu_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before any device is
                # initialized. If we are too late, skip silently —
                # the device is already configured.
                pass

    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()

    tf.keras.backend.set_floatx("float64")
    tf.keras.mixed_precision.set_global_policy("float64")

    _CONFIGURED = True


def is_configured() -> bool:
    """Return whether ``configure_tf()`` has been called in this process."""
    return _CONFIGURED
