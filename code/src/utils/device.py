"""TensorFlow device detection and configuration."""

import logging

logger = logging.getLogger(__name__)

_device_spec: str | None = None


def setup_tensorflow_device(device: str = "auto", memory_growth: bool = True) -> str:
    """Configure TensorFlow to use the best available device.

    Args:
        device: "auto" (CUDA > MPS > CPU), "gpu", or "cpu".
        memory_growth: Enable GPU memory growth to avoid OOM.

    Returns:
        Device string, e.g. '/GPU:0' or '/CPU:0'.
    """
    import tensorflow as tf

    global _device_spec

    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        _device_spec = "/CPU:0"
        logger.info("TensorFlow using CPU (forced)")
        return _device_spec

    if _device_spec is not None:
        logger.info(f"Using cached device: {_device_spec}")
        return _device_spec

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        _device_spec = "/CPU:0"
        logger.info("TensorFlow using CPU (no GPU found)")
        return _device_spec

    # Enable memory growth
    if memory_growth:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Could not set GPU memory growth: {e}")

    # Detect GPU type for logging
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        name = details.get("device_name", "")
        gpu_type = "MPS" if "Apple" in name else "CUDA"
    except Exception:
        gpu_type = "GPU"

    _device_spec = "/GPU:0"
    logger.info(f"TensorFlow using {gpu_type}: {_device_spec}")
    return _device_spec
