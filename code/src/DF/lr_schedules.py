"""Learning rate schedules for MAP optimization.

WarmupConstantCosineDecay: 3-phase schedule
  1. Linear warmup (default 10%)
  2. Constant plateau (default 70%)
  3. Cosine decay to alpha * peak_lr (default 20%)
"""

import math
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="DF")
class WarmupConstantCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warmup → constant → cosine decay learning rate schedule.

    Args:
        peak_lr: Maximum learning rate (reached after warmup).
        total_steps: Total number of optimizer steps.
        warmup_fraction: Fraction of steps for linear warmup (default 0.10).
        constant_fraction: Fraction of steps at constant peak_lr (default 0.70).
        alpha: Final LR as a fraction of peak_lr (default 0.01).
        name: Schedule name for TF ops.

    The decay fraction is 1 - warmup_fraction - constant_fraction (default 0.20).

    For a 300-step run with peak_lr=0.01:
        Steps   0–29:  warmup    0 → 0.01
        Steps  30–239: constant  0.01
        Steps 240–299: cosine    0.01 → 0.0001
    """

    def __init__(
        self,
        peak_lr,
        total_steps,
        warmup_fraction=0.10,
        constant_fraction=0.70,
        alpha=0.01,
        name="WarmupConstantCosineDecay",
    ):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_fraction = float(warmup_fraction)
        self.constant_fraction = float(constant_fraction)
        self.alpha = float(alpha)
        self._name = name

        self.warmup_steps = max(1, int(round(self.total_steps * self.warmup_fraction)))
        self.constant_steps = max(1, int(round(self.total_steps * self.constant_fraction)))
        self.decay_steps = max(1, self.total_steps - self.warmup_steps - self.constant_steps)

    def __call__(self, step):
        with tf.name_scope(self._name):
            step = tf.cast(step, tf.float32)
            peak = tf.constant(self.peak_lr, dtype=tf.float32)
            alpha = tf.constant(self.alpha, dtype=tf.float32)

            warmup_end = tf.constant(self.warmup_steps, dtype=tf.float32)
            decay_start = tf.constant(self.warmup_steps + self.constant_steps, dtype=tf.float32)
            decay_len = tf.constant(max(1, self.decay_steps - 1), dtype=tf.float32)

            # Phase 1: linear warmup
            warmup_lr = peak * step / tf.maximum(warmup_end, 1.0)

            # Phase 3: cosine decay
            progress = tf.clip_by_value((step - decay_start) / decay_len, 0.0, 1.0)
            cosine_lr = peak * (alpha + (1.0 - alpha) * 0.5 * (1.0 + tf.cos(math.pi * progress)))

            return tf.where(
                step < warmup_end, warmup_lr,
                tf.where(step < decay_start, peak, cosine_lr)
            )

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_fraction": self.warmup_fraction,
            "constant_fraction": self.constant_fraction,
            "alpha": self.alpha,
            "name": self._name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
