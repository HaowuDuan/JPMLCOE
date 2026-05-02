"""Phase 01 startup tests.

Verify that ``jpml_tf.startup.configure_tf()`` locks the global TF
dtype policy to float64 and is idempotent.
"""

import os


def test_configure_tf_sets_float64() -> None:
    from jpml_tf.startup import configure_tf

    configure_tf()

    import tensorflow as tf

    assert tf.keras.backend.floatx() == "float64"
    assert tf.keras.mixed_precision.global_policy().compute_dtype == "float64"


def test_configure_tf_is_idempotent() -> None:
    from jpml_tf.startup import configure_tf, is_configured

    configure_tf()
    configure_tf()
    assert is_configured()

    import tensorflow as tf

    assert tf.keras.backend.floatx() == "float64"


def test_configure_tf_sets_legacy_keras_env_var() -> None:
    from jpml_tf.startup import configure_tf

    configure_tf(legacy_keras=True)
    assert os.environ.get("TF_USE_LEGACY_KERAS") == "1"
