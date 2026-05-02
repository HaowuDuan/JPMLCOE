"""jpml_tf — TensorFlow rebuild of JPMLCOE.

The package root is intentionally empty: importing ``jpml_tf`` must not
trigger TensorFlow imports, scenario loading, or array creation. Every
entry point (CLI, test, notebook) calls ``jpml_tf.startup.configure_tf()``
explicitly before any computational module is imported.
"""

__version__ = "0.1.0"
