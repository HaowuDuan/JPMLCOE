"""Phase 01 import tests.

Three import-side-effect checks against the package root:
1. ``import jpml_tf`` succeeds.
2. ``jpml_tf.__version__`` exposes a non-empty string.
3. Importing ``jpml_tf`` does NOT load TensorFlow as a side effect.

The third check is load-bearing. Phase 02+ will set the global TF dtype
policy to float64 in ``jpml_tf.startup.configure_tf()``. That only works
if no TF code runs before ``configure_tf()`` is called. If anyone ever
adds an ``import tensorflow`` to the package root, this test fails on
the next pytest run.
"""

import sys


def test_jpml_tf_imports() -> None:
    import jpml_tf  # noqa: F401


def test_jpml_tf_exposes_version_string() -> None:
    import jpml_tf

    assert isinstance(jpml_tf.__version__, str)
    assert len(jpml_tf.__version__) > 0


def test_jpml_tf_does_not_import_tensorflow() -> None:
    # Drop any TF modules another test might have loaded so this assertion
    # measures only the side effects of importing jpml_tf itself.
    for mod in [m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow.")]:
        del sys.modules[mod]

    import jpml_tf  # noqa: F401

    tf_loaded = [m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow.")]
    assert tf_loaded == [], (
        f"Importing jpml_tf must not load TensorFlow, but found: {tf_loaded[:5]}"
    )
