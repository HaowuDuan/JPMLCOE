"""
TensorFlow device detection and configuration.

Checks for CUDA (NVIDIA GPU), MPS (Apple Silicon/Metal), and falls back to CPU.
Configures GPU memory growth and logs the selected device.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cached result after first call
_device_info: Optional[Tuple[str, str]] = None


def setup_tensorflow_device(
    device: str = "auto",
    memory_growth: bool = True,
    log_level: int = logging.INFO,
    prefer_gpu: bool = True,  # Deprecated: use device="auto" or "gpu" instead
) -> str:
    """
    Configure TensorFlow to use the specified or best available device.
    
    Args:
        device: One of:
            - "auto": Use best available (CUDA > MPS > CPU)
            - "cpu": Force CPU only
            - "gpu": Prefer GPU when available (same as auto for GPU choice)
        memory_growth: If True, enable GPU memory growth to avoid OOM.
        log_level: Logging level for device info (default: INFO).
        prefer_gpu: Deprecated, use device= instead. Kept for backward compatibility.
    
    Returns:
        Device spec string, e.g. '/GPU:0', '/CPU:0'
    
    Example:
        >>> device = setup_tensorflow_device(device="cpu")
        >>> # Or from config: setup_tensorflow_device(device=cfg.device)
    """
    import tensorflow as tf
    
    global _device_info
    
    # Backward compatibility: prefer_gpu=False overrides device
    if not prefer_gpu and device == "auto":
        device = "cpu"
    
    # device=="cpu" always forces CPU (don't use cache)
    if device == "cpu":
        force_cpu()
        return _device_info[0]
    
    if _device_info is not None:
        device_spec, device_type = _device_info
        logger.log(log_level, f"Using cached device: {device_spec} ({device_type})")
        return device_spec
    
    prefer_gpu = device in ("auto", "gpu")
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    device_spec = '/CPU:0'
    device_type = 'CPU'
    
    if prefer_gpu and gpus:
        # Configure GPU memory growth
        if memory_growth:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Could not set GPU memory growth: {e}")
        
        # Distinguish CUDA vs Metal (MPS)
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            device_name = details.get('device_name', 'Unknown')
            if 'Apple' in device_name or 'Metal' in str(details).lower():
                device_type = 'MPS (Apple Silicon)'
            else:
                device_type = 'CUDA (NVIDIA GPU)'
        except (AttributeError, TypeError, RuntimeError):
            # get_device_details may not exist in older TF
            device_type = 'GPU (CUDA or Metal)'
        
        # Use canonical device spec for tf.device(): /GPU:0, /CPU:0
        device_spec = '/GPU:0' if 'GPU' in gpus[0].name else gpus[0].name
        logger.log(
            log_level,
            f"TensorFlow using {device_type}: {device_spec}"
        )
    else:
        if not prefer_gpu:
            logger.log(log_level, "TensorFlow using CPU (prefer_gpu=False)")
        elif not gpus:
            logger.log(
                log_level,
                f"TensorFlow using CPU (no GPU found; {len(cpus)} CPU(s) available)"
            )
        else:
            logger.log(log_level, f"TensorFlow using CPU: {device_spec}")
    
    _device_info = (device_spec, device_type)
    return device_spec


def get_device_info() -> Tuple[str, str]:
    """
    Get the current device info (spec and type).
    Call setup_tensorflow_device() first to initialize.
    
    Returns:
        (device_spec, device_type) e.g. ('/GPU:0', 'CUDA (NVIDIA GPU)')
    """
    global _device_info
    if _device_info is None:
        setup_tensorflow_device()
    return _device_info


def force_cpu() -> None:
    """
    Force TensorFlow to use CPU only (e.g. for debugging).
    Call before any TensorFlow operations.
    """
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    global _device_info
    _device_info = ('/CPU:0', 'CPU (forced)')


def reset_device_cache() -> None:
    """Reset the cached device info (e.g. for testing)."""
    global _device_info
    _device_info = None
