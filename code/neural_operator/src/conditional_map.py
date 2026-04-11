"""Conditional ConvexPotentialMap.

A neural operator that takes a context vector c (from the encoder) and
parameterizes a transport map T(x; c).

Module structure (per module m):
    u_m(x, c) = W_m @ x + U_m @ c + b_m(c)
    z_m(x, c) = phi_m(softplus(beta_m(c)) * u_m(x, c))
    T_m(x, c) = W_m^T @ z_m(x, c)

Full map (residual):
    T(x; c) = x + d(c) + sum_m softplus(alpha_m(c)) * T_m(x, c)

Convexity in x is preserved because:
- W_m is fixed (NOT context-dependent)
- The W_m^T diag(.) W_m structure stays intact for fixed c
- Context only modulates the diagonal positive scales (alpha_m, beta_m) and
  the bias/shift inside the activation

Jacobian (analytic):
    J_T(x; c) = I + sum_m softplus(alpha_m(c)) * softplus(beta_m(c)) *
                W_m^T diag(phi'(scaled_arg)) W_m
"""

import tensorflow as tf

from models import WTanh


class ConditionalGNMModule(tf.keras.layers.Layer):
    """Single conditional module.

    Forward:
        u = W @ x + U @ c + b(c)
        z = phi(softplus(beta(c)) * u)
        out = W^T @ z

    Where:
    - W is a fixed learned weight (NOT context-dependent)
    - U is a fixed learned weight that mixes context into the pre-activation
    - b(c), beta(c) are produced from c via small linear layers
    """

    def __init__(self, in_dim: int, embed_dim: int, context_dim: int,
                 activation_cls=WTanh, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.activation_cls = activation_cls

    def build(self, input_shape):
        # Tied weight matrix (NOT context-dependent)
        self.W = self.add_weight(
            name='W', shape=(self.embed_dim, self.in_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Context-to-embed projection (additive shift inside activation)
        self.U = self.add_weight(
            name='U', shape=(self.embed_dim, self.context_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        # Context-to-bias projection
        self.b_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name='b_proj')
        # Context-to-beta projection (scalar; softplus applied later)
        self.beta_proj = tf.keras.layers.Dense(1, use_bias=True, name='beta_proj')
        # Activation has its own learnable parameters
        self.act = self.activation_cls(self.embed_dim) \
            if self.activation_cls is WTanh else self.activation_cls()

    def _compute_pre_act(self, x, c):
        """Compute pre-activation u and the positive beta from context.

        x: (batch, in_dim)
        c: (context_dim,)

        Returns:
            u: (batch, embed_dim) pre-activation
            beta_pos: scalar positive beta
        """
        # W @ x: (batch, embed_dim)
        Wx = x @ tf.transpose(self.W)
        # U @ c: (embed_dim,) — broadcast to (1, embed_dim)
        Uc = tf.linalg.matvec(self.U, c)[None, :]
        # b(c): (1, embed_dim)
        b_c = self.b_proj(c[None, :])
        u = Wx + Uc + b_c
        # beta(c): scalar
        beta_raw = self.beta_proj(c[None, :])[0, 0]
        beta_pos = tf.nn.softplus(beta_raw)
        return u, beta_pos

    def call(self, x, c):
        """Forward pass.

        x: (batch, in_dim)
        c: (context_dim,)

        Returns: (batch, in_dim)
        """
        u, beta_pos = self._compute_pre_act(x, c)
        z = self.act(u * beta_pos)
        out = z @ self.W
        return out

    def jacobian(self, x, c):
        """Analytic Jacobian J(x; c) = beta_pos * W^T diag(phi'(scaled_u)) W.

        x: (batch, in_dim)
        c: (context_dim,)

        Returns: (batch, in_dim, in_dim)
        """
        if not isinstance(self.act, WTanh):
            raise NotImplementedError(
                "Analytic Jacobian only implemented for WTanh activation."
            )
        u, beta_pos = self._compute_pre_act(x, c)
        scaled_u = u * beta_pos
        # Element-wise activation derivative: (batch, embed_dim)
        d = self.act.derivative(scaled_u) * beta_pos
        # J = W^T diag(d) W. Vectorized as einsum.
        d_W = tf.expand_dims(d, -1) * tf.expand_dims(self.W, 0)  # (batch, embed, in)
        J = tf.einsum('ei,bej->bij', self.W, d_W)
        return J


class ConditionalConvexPotentialMap(tf.keras.Model):
    """Conditional T(x; c) parameterized by context c from the encoder.

    T(x; c) = x + d(c) + sum_m softplus(alpha_m(c)) * T_m(x, c)

    The output bias d(c) and the per-module gates alpha_m(c) are produced
    from c via linear projections. The W_m weights inside each module are
    fixed (not context-dependent), preserving PSD Jacobian.
    """

    def __init__(self, in_dim: int, embed_dim: int, num_modules: int,
                 context_dim: int, activation_cls=WTanh,
                 alpha_init_bias: float = -5.0, **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_modules = num_modules
        self.context_dim = context_dim
        self.alpha_init_bias = alpha_init_bias

        self.modules_list = [
            ConditionalGNMModule(
                in_dim, embed_dim, context_dim, activation_cls=activation_cls
            )
            for _ in range(num_modules)
        ]

    def build(self, input_shape):
        # Context-to-alpha projection (one scalar per module)
        # Initialize with negative bias so softplus(alpha) ~ 0 at init,
        # giving T(x; c) ~ x (residual near identity).
        self.alpha_proj = tf.keras.layers.Dense(
            self.num_modules,
            use_bias=True,
            bias_initializer=tf.keras.initializers.Constant(self.alpha_init_bias),
            name='alpha_proj',
        )
        # Context-to-output-bias projection
        self.out_bias_proj = tf.keras.layers.Dense(
            self.in_dim,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='out_bias_proj',
        )

    def _alpha_pos(self, c):
        """Per-module positive gate from context."""
        # alpha_proj input: (1, context_dim) -> (1, num_modules)
        alpha_raw = self.alpha_proj(c[None, :])[0]
        return tf.nn.softplus(alpha_raw)

    def call(self, x, c):
        """Forward pass.

        x: (batch, in_dim)
        c: (context_dim,)

        Returns: (batch, in_dim)
        """
        alpha_pos = self._alpha_pos(c)
        out_bias = self.out_bias_proj(c[None, :])[0]  # (in_dim,)

        z = tf.zeros_like(x)
        for i, mod in enumerate(self.modules_list):
            z = z + alpha_pos[i] * mod(x, c)
        z = z + out_bias[None, :]
        return x + z  # residual

    def jacobian(self, x, c):
        """Analytic Jacobian J(x; c) = I + sum_m alpha_m(c) * J_m(x, c).

        x: (batch, in_dim)
        c: (context_dim,)

        Returns: (batch, in_dim, in_dim)
        """
        alpha_pos = self._alpha_pos(c)
        batch_size = tf.shape(x)[0]
        I = tf.eye(self.in_dim, dtype=x.dtype, batch_shape=[batch_size])
        J = I
        for i, mod in enumerate(self.modules_list):
            J = J + alpha_pos[i] * mod.jacobian(x, c)
        return J

    def forward_and_jacobian(self, x, c):
        """Returns (T(x; c), J(x; c)) in one call."""
        return self(x, c), self.jacobian(x, c)
