"""ConvexPotentialMap: a neural network parameterizing T = grad(phi) for convex phi.

Architecture port of mGradNet-M from GradNetOT
(https://github.com/cShreyas/GradNetOT) from PyTorch to TF.

Each module computes:
    T_m(x) = W^T phi(softplus(beta) * (W @ x + b))
where the tied W/W^T structure guarantees the Jacobian is symmetric PSD,
which means T_m is the gradient of a convex potential.

The full map is:
    T(x) = bias + sum_m softplus(alpha_m) * T_m(x)
With residual parameterization, T(x) = x + bias + sum_m softplus(alpha_m) * T_m(x).

This is exactly the structure required by Brenier's theorem for an optimal
transport map under quadratic cost: T must be the gradient of a convex function.

Activations:
- WTanh: combines tanh and tanhshrink with positive coefficients (monotone)
- WSoftmax: combines softmax and softmin with positive coefficients (monotone)
"""

import tensorflow as tf
from typing import Optional


# ============================================================================
# Activations
# ============================================================================

class WTanh(tf.keras.layers.Layer):
    """Monotone WTanh: f(x) = softplus(a)*tanh(x) + softplus(b)*tanhshrink(x).

    tanhshrink(x) = x - tanh(x).
    Both terms have non-negative scalar derivative when monotone=True:
        f'(x) = softplus(a)*sech^2(x) + softplus(b)*tanh^2(x) >= 0
    """

    def __init__(self, dim: int, monotone: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.monotone = monotone

    def build(self, input_shape):
        self.a = self.add_weight(
            name='a', shape=(self.dim,),
            initializer=tf.keras.initializers.RandomUniform(0, 1),
            trainable=True,
        )
        self.b = self.add_weight(
            name='b', shape=(self.dim,),
            initializer=tf.keras.initializers.RandomUniform(0, 1),
            trainable=True,
        )

    def call(self, x):
        a = tf.nn.softplus(self.a) if self.monotone else self.a
        b = tf.nn.softplus(self.b) if self.monotone else self.b
        # tanhshrink(x) = x - tanh(x)
        tanh_x = tf.nn.tanh(x)
        tanhshrink_x = x - tanh_x
        return a * tanh_x + b * tanhshrink_x

    def derivative(self, x):
        """Element-wise derivative f'(x). Used for analytic Jacobians."""
        a = tf.nn.softplus(self.a) if self.monotone else self.a
        b = tf.nn.softplus(self.b) if self.monotone else self.b
        # d/dx [tanh(x)] = sech^2(x) = 1 - tanh^2(x)
        # d/dx [tanhshrink(x)] = 1 - sech^2(x) = tanh^2(x)
        tanh_x = tf.nn.tanh(x)
        sech2 = 1.0 - tanh_x * tanh_x
        return a * sech2 + b * tanh_x * tanh_x


class WSoftmax(tf.keras.layers.Layer):
    """Monotone WSoftmax: f(x) = softplus(a)*softmax(x) - softplus(b)*softmin(x).

    softmin(x) = softmax(-x).

    Note: this activation couples coordinates (Jacobian is not diagonal).
    Use WTanh for simpler analytic Jacobian.
    """

    def __init__(self, monotone: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.monotone = monotone

    def build(self, input_shape):
        self.a = self.add_weight(
            name='a', shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
        )
        self.b = self.add_weight(
            name='b', shape=(),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
        )

    def call(self, x):
        a = tf.nn.softplus(self.a) if self.monotone else self.a
        b = tf.nn.softplus(self.b) if self.monotone else self.b
        return a * tf.nn.softmax(x, axis=-1) - b * tf.nn.softmax(-x, axis=-1)


# ============================================================================
# Single module: T_m(x) = W^T phi(softplus(beta) * (W x + b))
# ============================================================================

class ConvexPotentialModule(tf.keras.layers.Layer):
    """Single building block of ConvexPotentialMap with tied W/W^T structure.

    Forward:
        z = W @ x + b
        z = phi(softplus(beta) * z)
        out = W^T @ z

    Jacobian (when phi has elementwise positive derivative):
        J = W^T @ diag(phi'(softplus(beta) * (W x + b)) * softplus(beta)) @ W
    This is symmetric PSD because it has the form A^T D A with D > 0 (diagonal).
    A symmetric PSD Jacobian means T_m is the gradient of a convex potential.
    """

    def __init__(self, in_dim: int, embed_dim: int,
                 activation_cls=WTanh, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.activation_cls = activation_cls

    def build(self, input_shape):
        # Tied weight matrix: shape (embed_dim, in_dim)
        self.W = self.add_weight(
            name='W', shape=(self.embed_dim, self.in_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.b = self.add_weight(
            name='b', shape=(self.embed_dim,),
            initializer='zeros',
            trainable=True,
        )
        self.beta = self.add_weight(
            name='beta', shape=(),
            initializer=tf.keras.initializers.RandomUniform(0, 1),
            trainable=True,
        )
        # Activation has its own learnable parameters
        self.act = self.activation_cls(self.embed_dim) \
            if self.activation_cls is WTanh else self.activation_cls()

    def call(self, x):
        # x: (batch, in_dim)
        beta_pos = tf.nn.softplus(self.beta)
        # z = W @ x + b: (batch, embed_dim)
        z = tf.linalg.matvec(self.W, x) if x.shape.rank == 1 else x @ tf.transpose(self.W) + self.b
        # Apply activation with scaled input
        z = self.act(z * beta_pos)
        # Project back: (batch, in_dim)
        out = z @ self.W
        return out

    def jacobian(self, x):
        """Analytic Jacobian J(x) = W^T diag(phi'(scaled_z) * beta_pos) W.

        x: (batch, in_dim)
        Returns: (batch, in_dim, in_dim)

        Only works for elementwise activations (e.g., WTanh).
        """
        if not isinstance(self.act, WTanh):
            raise NotImplementedError(
                "Analytic Jacobian only implemented for WTanh activation. "
                "Use tape.batch_jacobian for WSoftmax."
            )
        beta_pos = tf.nn.softplus(self.beta)
        # Pre-activation: scaled_z = beta_pos * (W x + b)
        z = x @ tf.transpose(self.W) + self.b  # (batch, embed_dim)
        scaled_z = z * beta_pos
        # Element-wise derivative: (batch, embed_dim)
        d = self.act.derivative(scaled_z) * beta_pos
        # J = W^T @ diag(d) @ W. Vectorized:
        # For each batch element: W^T (d * W) where * is broadcast
        # W shape: (embed_dim, in_dim)
        # d shape: (batch, embed_dim)
        # d_W shape: (batch, embed_dim, in_dim) by broadcasting
        d_W = tf.expand_dims(d, -1) * tf.expand_dims(self.W, 0)  # (batch, embed, in)
        # J = W^T @ d_W: (in, embed) @ (batch, embed, in) -> (batch, in, in)
        J = tf.einsum('ei,bej->bij', self.W, d_W)
        return J


# ============================================================================
# Full ConvexPotentialMap
# ============================================================================

class ConvexPotentialMap(tf.keras.Model):
    """Neural network parameterizing T = grad(phi) for a convex potential phi.

    T(x) = bias + sum_m softplus(alpha_m) * T_m(x)

    With residual parameterization: T(x) = x + bias + sum_m softplus(alpha_m) * T_m(x).

    The Jacobian J(x) = nabla T(x) is symmetric PSD by construction (since each
    T_m has a W^T D W Jacobian and the sum of PSD matrices with positive
    coefficients is PSD).

    By Brenier's theorem, this is exactly the structural form of an optimal
    transport map under quadratic cost.
    """

    def __init__(self, in_dim: int, embed_dim: int, num_modules: int,
                 activation_cls=WTanh, residual: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_modules = num_modules
        self.residual = residual
        self.modules_list = [
            ConvexPotentialModule(in_dim, embed_dim, activation_cls=activation_cls)
            for _ in range(num_modules)
        ]

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha', shape=(self.num_modules,),
            initializer=tf.keras.initializers.RandomUniform(0, 1),
            trainable=True,
        )
        self.out_bias = self.add_weight(
            name='out_bias', shape=(self.in_dim,),
            initializer='zeros',
            trainable=True,
        )

    def call(self, x):
        alpha_pos = tf.nn.softplus(self.alpha)
        z = tf.zeros_like(x)
        for i, mod in enumerate(self.modules_list):
            z = z + alpha_pos[i] * mod(x)
        z = z + self.out_bias
        if self.residual:
            return x + z
        return z

    def jacobian(self, x):
        """Analytic Jacobian of T(x).

        J = (I if residual else 0) + sum_m softplus(alpha_m) * J_m(x)
        Returns: (batch, in_dim, in_dim)
        """
        alpha_pos = tf.nn.softplus(self.alpha)
        batch_size = tf.shape(x)[0]
        J = tf.zeros((batch_size, self.in_dim, self.in_dim), dtype=x.dtype)
        for i, mod in enumerate(self.modules_list):
            J = J + alpha_pos[i] * mod.jacobian(x)
        if self.residual:
            I = tf.eye(self.in_dim, dtype=x.dtype, batch_shape=[batch_size])
            J = I + J
        return J

    def forward_and_jacobian(self, x):
        """Returns (T(x), J(x)) in one call."""
        return self(x), self.jacobian(x)
