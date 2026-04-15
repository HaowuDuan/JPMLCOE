#!/usr/bin/env bash
set -euo pipefail

# Remote-only runner for office. Reminder before use from local machine:
#   make push
# so configs/acoustic_tracking_ledh_flow_f64.yaml exists under ~/JPML/code on office.

printf '============================================================
'
printf 'Acoustic LEDH flow diagnostic run
'
printf 'Date: %s
' "$(date)"
printf 'Host: %s
' "$(hostname)"
printf 'User: %s
' "$(whoami)"
printf '============================================================
'

cd "$HOME/JPML/code"

LOG="acoustic_flow_diagnostic_run_$(date +%Y%m%d_%H%M%S).log"
printf 'Writing log to %s/%s
' "$(pwd)" "$LOG"

NVIDIA_PATH="$(.venv/bin/python -c "import nvidia; print(list(nvidia.__path__)[0])")"
export LD_LIBRARY_PATH="$NVIDIA_PATH/cuda_runtime/lib:$NVIDIA_PATH/cublas/lib:$NVIDIA_PATH/cudnn/lib:$NVIDIA_PATH/cufft/lib:$NVIDIA_PATH/curand/lib:$NVIDIA_PATH/cusolver/lib:$NVIDIA_PATH/cusparse/lib:$NVIDIA_PATH/nccl/lib:$NVIDIA_PATH/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_PATH/cuda_nvcc"
export TF_USE_LEGACY_KERAS=1
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR=/tmp/mpl
mkdir -p "$MPLCONFIGDIR"

OUT_DIR="outputs/acoustic_tracking/acoustic_tracking_ledh_flow_f64"
if [[ -d "$OUT_DIR" ]]; then
  BAK_DIR="${OUT_DIR}.bak.$(date +%s)"
  printf 'Existing %s found; moving to %s
' "$OUT_DIR" "$BAK_DIR" | tee -a "$LOG"
  mv "$OUT_DIR" "$BAK_DIR"
fi

cat > /tmp/ledh_flow_instrumented.py <<'PYEOF'
"""Local Exact Daum-Huang (LEDH) particle flow filter"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any

from src.filters.particle.flow_base import FlowFilterBase
from src.filters.kalman.filter_factory import create_kalman_filter
from src.utils.flow_params import compute_flow_params, compute_flow_params_batch
from src.utils.linalg import safe_inv, to_numpy
from src.utils.ode_solvers import euler_step
from src.utils.constants import FlowScheduleConfig, DriftClipConfig
from src.utils.distributions import sample_particles_cholesky
from src.utils.resampling_config import resolve_resampling


class LocalExactDaumHuangFlow(FlowFilterBase):
    cond_S_history = []
    H_spectral_norm_history = []
    H_sigma_min_history = []
    A_fro_history = []
    b_norm_history = []
    ess_over_n_history = []
    particle_cov_trace_history = []

    """
    Local Exact Daum-Huang Flow with per-particle linearization.

    Key modifications from global EDH:
    1. Linearize measurement function at EACH particle location (not just at mean)
    2. Use GLOBAL predictive covariance P_{k|k-1} from a single EKF
    3. Compute A_i and b_i matrices individually for each particle
    
    This is modification (ii) from Section 3 of the paper.

    Flow equation: dx_i/dλ = A_i(λ) @ x_i + b_i(λ)
    where A_i uses local linearization H_i at particle i and global covariance P_{k|k-1}.
    """

    def __init__(self, model, n_particles: int = 1000,
                 n_lambda_steps: int = 100,
                 integration_method: str = 'euler',
                 use_feedback: bool = True,
                 regularization: float = 1e-8,
                 resampling_method: Optional[Callable] = None,
                 resampling_config: Optional[Dict[str, Any]] = None,
                 n_threads: Optional[int] = None,
                 debug_mode: bool = False,
                 flow_config: FlowScheduleConfig = FlowScheduleConfig(),
                 clip_config: DriftClipConfig = DriftClipConfig(),
                 filter_type: str = 'ekf'):
        """
        Initialize Local Exact Daum-Huang flow filter.

        Args:
            model: StateSpaceModel instance
            n_particles: Number of particles
            n_lambda_steps: Number of discretization steps for λ ∈ [0,1]
            integration_method: 'euler' or 'rk4'
            use_feedback: If True, feed DH mean back to EKF
            regularization: Small value added to diagonal of S matrix for numerical stability
            resampling_method: Resampling method (string or callable)
                'systematic', 'soft', or 'ot_entropy'
            resampling_config: Optional dict with method-specific parameters
                For soft: {'alpha': float}
                For ot_entropy: {'reg': float, 'n_iter': int}
            n_threads: Number of threads for parallelization (None = auto)
            debug_mode: If True, collect detailed diagnostics
            filter_type: 'ekf' or 'ukf' for the global covariance guidance filter
        """
        super().__init__(model, n_particles, n_lambda_steps, integration_method, n_threads)
        self.filter_type = filter_type
        self.flow_config = flow_config
        self.clip_config = clip_config
        self.dtype = getattr(model, 'dtype', tf.float64)
        self.np_dtype = np.float64 if self.dtype == tf.float64 else np.float32
        self.use_feedback = use_feedback
        self.regularization = regularization
        self.debug_mode = debug_mode

        # Resolve resampling method and config
        self.resampling_method, self.resampling_method_name, self.resampling_config = (
            resolve_resampling(resampling_method, resampling_config)
        )

        # Single GLOBAL EKF for covariance guidance
        self.global_filter = None
        self.predicted_cov = None  # P_{k|k-1} used for ALL particles

        # Cache for optimization
        self.R_inv_cache = None
        self.L_cache = None

        # Store mean at λ=0 for b computation
        self.eta_bar_0 = None
        
        # Debug storage
        if self.debug_mode:
            self.debug_info = {
                'timesteps': [],
                'flow_steps': [],
                'particles_before_flow': [],
                'particles_after_flow': [],
                'A_matrices': [],
                'b_vectors': [],
                'H_jacobians': [],
                'eigenvalues': [],
                'condition_numbers': [],
                'particle_stats': []
            }
        else:
            self.debug_info = None
        
        # Generate exponential lambda schedule (same as ledh_invertible)
        self._generate_lambda_steps()

    def initialize(self, random_state: Optional[np.random.Generator] = None):
        """Initialize particles and global EKF for covariance guidance."""
        # Get initial mean and covariance from model
        if hasattr(self.model, 'mu_0') and hasattr(self.model, 'Sigma_0'):
            initial_mean = self.model.mu_0
            initial_cov = self.model.Sigma_0
        else:
            raise ValueError("Model must have mu_0 and Sigma_0 attributes")

        # Convert to TensorFlow for sampling
        initial_mean_tf = tf.constant(initial_mean, dtype=self.dtype)
        initial_cov_tf = tf.constant(initial_cov, dtype=self.dtype)

        # Initialize RNG key
        if random_state is not None:
            seed_val = random_state.integers(0, 2**31)
            self.rng_key = tf.constant([seed_val, 0], dtype=tf.int32)
        else:
            self.rng_key = tf.constant([42, 0], dtype=tf.int32)

        # Sample initial particles using TensorFlow
        seed = self._next_seed()

        particles_tf = sample_particles_cholesky(
            initial_mean_tf, initial_cov_tf, self.n_particles, self.state_dim, seed, self.dtype
        )

        # Store as TensorFlow Variable
        self.particles = tf.Variable(particles_tf, dtype=self.dtype)
        self.weights = tf.Variable(
            tf.ones(self.n_particles, dtype=self.dtype) / tf.cast(self.n_particles, self.dtype),
            dtype=self.dtype
        )

        # Compute empirical mean and covariance (TF ops)
        ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
        diff = self.particles.value() - ensemble_mean
        if self.state_dim == 1:
            initial_cov_emp = tf.reshape(tf.math.reduce_variance(self.particles.value()), [1, 1])
        else:
            initial_cov_emp = tf.matmul(diff, diff, transpose_a=True) / tf.cast(self.n_particles, self.dtype)

        # Initialize global EKF for covariance guidance (constructor needs numpy)
        ensemble_mean_np = ensemble_mean.numpy()
        initial_cov_emp_np = initial_cov_emp.numpy()
        self.global_filter = create_kalman_filter(
            self.filter_type, self.model,
            mean_0=ensemble_mean_np, Sigma_0=initial_cov_emp_np
        )

        self.global_filter.mean.assign(ensemble_mean)
        self.global_filter.cov.assign(initial_cov_emp)
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

    def _generate_lambda_steps(self):
        """
        Generate exponential decay schedule for lambda steps as TF tensor.
        Uses geometric sequence with ratio q=1.2, normalized to sum to 1.
        """
        q = self.flow_config.geometric_ratio
        epsilon_1 = (1 - q) / (1 - q**self.n_lambda_steps)
        steps_np = epsilon_1 * q**np.arange(self.n_lambda_steps)
        self.lambda_steps = tf.constant(steps_np, dtype=self.dtype)
        

    def _flow_step_euler(
        self,
        particles: tf.Tensor,
        y: tf.Tensor,
        lambda_val: tf.Tensor,
        d_lambda: tf.Tensor,
        P: tf.Tensor,
        R: tf.Tensor,
        R_inv: tf.Tensor,
        eta_bar_0: tf.Tensor
    ) -> tf.Tensor:
        """
        Local flow step using Euler integration with batched flow params.

        Each particle uses its own LOCAL linearization H_i and GLOBAL P_{k|k-1}.

        Args:
            particles: Current particles, shape (N, state_dim)
            y: Observation, shape (obs_dim,)
            lambda_val: Current λ
            d_lambda: Step size
            P: GLOBAL predictive covariance (sd, sd) — broadcast inside batch fn
            R: Observation noise covariance
            R_inv: Inverse of R
            eta_bar_0: GLOBAL mean at λ=0

        Returns:
            Updated particles, shape (N, state_dim)
        """
        regularization_tf = tf.constant(self.regularization, dtype=self.dtype)

        # Compute A, b for ALL particles in one batched call
        A_batch, b_batch = compute_flow_params_batch(
            self.model, particles, lambda_val, y, P, R, R_inv,
            eta_bar_0, self.state_dim, regularization_tf
        )

        # Diagnostics for the exact matrix inverted in compute_flow_params_batch.
        H_batch = self.model.observation_jacobian_batch(particles)
        P_b = tf.expand_dims(P, 0) if len(P.shape) == 2 else P
        if regularization_tf > 0.0:
            trace_P = tf.linalg.trace(P_b)
            state_dim_f = tf.cast(self.state_dim, P_b.dtype)
            reg_strength = regularization_tf * (trace_P / state_dim_f)
            P_b = P_b + reg_strength[..., tf.newaxis, tf.newaxis] * tf.eye(self.state_dim, dtype=P_b.dtype)
        S_batch = lambda_val * tf.matmul(tf.matmul(H_batch, P_b), H_batch, transpose_b=True) + (R if R.shape.rank == 3 else tf.expand_dims(R, 0))
        S_svals = tf.linalg.svd(S_batch, compute_uv=False)
        H_svals = tf.linalg.svd(H_batch, compute_uv=False)
        self.__class__.cond_S_history.append((S_svals[:, 0] / S_svals[:, -1]).numpy())
        self.__class__.H_spectral_norm_history.append(H_svals[:, 0].numpy())
        self.__class__.H_sigma_min_history.append(H_svals[:, -1].numpy())
        self.__class__.A_fro_history.append(tf.norm(A_batch, axis=[1, 2]).numpy())
        self.__class__.b_norm_history.append(tf.norm(b_batch, axis=1).numpy())

        # Drift: A_i @ x_i + b_i for all particles: (N, sd)
        drift = tf.einsum('nij,nj->ni', A_batch, particles) + b_batch

        # Clip drift magnitude per-particle
        drift_norms = tf.norm(drift, axis=1, keepdims=True)
        scale = tf.minimum(tf.constant(1.0, dtype=drift_norms.dtype), self.clip_config.max_drift_norm / (drift_norms + self.clip_config.epsilon))
        drift = drift * scale

        # Euler step
        particles_new = particles + drift * d_lambda

        # Apply clipping to prevent divergence
        norms = tf.norm(particles_new, axis=1, keepdims=True)
        scale = tf.minimum(tf.constant(1.0, dtype=norms.dtype), self.clip_config.max_particle_norm / (norms + self.clip_config.epsilon))
        particles_new = particles_new * scale

        return particles_new

    def predict(self, t=None):
        """
        Prediction step with mean-only feedback.

        1. Update global filter mean to ensemble mean (if feedback enabled)
        2. Run global EKF prediction to get P_{k|k-1}
        3. Propagate each particle through dynamics with noise

        Note: We do NOT blend empirical covariances back into the global filter.
        The global filter provides covariance guidance only - blending causes
        covariance explosion due to positive feedback.
        """
        if t is not None and hasattr(self.model, 't'):
            self.model.t = t

        # FEEDBACK MECHANISM: Update global filter mean to ensemble mean (TF ops, no numpy)
        if self.use_feedback:
            ensemble_mean = tf.reduce_mean(self.particles.value(), axis=0)
            self.global_filter.mean.assign(ensemble_mean)

        # Run GLOBAL EKF prediction to get P_{k|k-1}
        self.global_filter.predict()
        self.predicted_cov = self.global_filter.cov.value()  # TF tensor

        # Store η̄_0: the DETERMINISTIC predicted mean
        self.eta_bar_0 = self.global_filter.mean.value()  # TF tensor

        # Propagate particles through state transition using model's batch method
        seed = self._next_seed()

        particles_predicted = self.model.state_transition_batch(self.particles.value(), seed, t=t)
        self.particles.assign(particles_predicted)

    def update(self, y: tf.Tensor):
        """
        Update step: flow particles from λ=0 to λ=1 using LOCAL linearizations.

        Each particle gets its own H_i but uses GLOBAL P_{k|k-1}.

        Args:
            y: Observation TF tensor, shape (obs_dim,)
        """
        observation = y  # Already TF tensor from flow_base.filter()
        P_tf = self.predicted_cov  # Already TF tensor from predict()
        R_tf = self.model.observation_noise_cov
        eta_bar_0_tf = self.eta_bar_0  # Already TF tensor from predict()

        # Cache R_inv (constant across timesteps)
        if self.R_inv_cache is None:
            self.R_inv_cache = safe_inv(R_tf)
        R_inv_tf = self.R_inv_cache

        # Use exponential lambda schedule
        particles_flow = self.particles.value()
        lambda_val = tf.constant(0.0, dtype=self.dtype)

        # Debug: Store particles before flow
        if self.debug_mode:
            particles_before = particles_flow.numpy().copy()
            timestep_debug = {
                'timestep': len(self.means),
                'observation': y.numpy().copy(),
                'particles_before': particles_before,
                'flow_steps': []
            }

        # Integrate flow with LOCAL linearizations
        for i in range(self.n_lambda_steps):
            d_lambda = self.lambda_steps[i]
            lambda_val = lambda_val + d_lambda  # TF tensor accumulation

            # Debug: Capture flow step diagnostics (sample steps)
            if self.debug_mode and i % 10 == 0:
                regularization_tf = tf.constant(self.regularization, dtype=self.dtype)
                A, b = compute_flow_params(
                    self.model, particles_flow[0], lambda_val, observation,
                    P_tf, R_tf, R_inv_tf, eta_bar_0_tf, self.state_dim, regularization_tf
                )
                H = self.model.observation_jacobian(particles_flow[0])

                A_np = A.numpy()
                try:
                    eigvals = np.linalg.eigvals(A_np)
                    cond_A = np.linalg.cond(A_np)
                except:
                    eigvals = np.array([np.nan])
                    cond_A = np.nan

                flow_step_debug = {
                    'step': i,
                    'lambda': float(lambda_val),
                    'epsilon': float(d_lambda),
                    'A_matrix': A_np.copy(),
                    'b_vector': b.numpy().copy(),
                    'H_jacobian': H.numpy().copy(),
                    'eigenvalues': eigvals,
                    'condition_number': cond_A,
                    'particle_mean': tf.reduce_mean(particles_flow, axis=0).numpy(),
                    'particle_std': tf.math.reduce_std(particles_flow, axis=0).numpy()
                }
                timestep_debug['flow_steps'].append(flow_step_debug)

            if self.integration_method == 'euler':
                particles_flow = self._flow_step_euler(
                    particles_flow, observation, lambda_val, d_lambda,
                    P_tf, R_tf, R_inv_tf, eta_bar_0_tf
                )
            elif self.integration_method == 'rk4':
                raise NotImplementedError("RK4 integration not yet implemented for TensorFlow LEDH flow")
            else:
                raise ValueError(f"Unknown integration method: {self.integration_method}")

        # Particles at λ=1 represent posterior
        self.particles.assign(particles_flow)
        _, particle_cov = self._estimate_mean_cov()
        self.__class__.ess_over_n_history.append(1.0)
        self.__class__.particle_cov_trace_history.append(float(tf.linalg.trace(particle_cov).numpy()))

        # Debug: Store after-flow diagnostics
        if self.debug_mode:
            particles_after = self.particles.numpy()
            timestep_debug['particles_after'] = particles_after.copy()
            timestep_debug['particle_stats_after'] = {
                'mean': np.mean(particles_after, axis=0),
                'cov': np.cov(particles_after.T),
                'min': np.min(particles_after, axis=0),
                'max': np.max(particles_after, axis=0)
            }
            self.debug_info['timesteps'].append(timestep_debug)

        # Update global filter for next prediction cycle (EKF accepts numpy)
        self.global_filter.update(to_numpy(y))

    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        return {
            'final_particles': self.particles.numpy(),
            'predicted_cov': to_numpy(self.predicted_cov),
            'global_filter_mean': self.global_filter.mean,
            'global_filter_cov': self.global_filter.cov,
            'use_feedback': self.use_feedback,
            'eta_bar_0': to_numpy(self.eta_bar_0)
        }
PYEOF

cat > /tmp/acoustic_diagnostic.py <<'PYEOF'
#!/usr/bin/env python3
"""Run acoustic LEDH-flow conditioning diagnostics using /tmp instrumentation."""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
CODE_DIR = Path.home() / "JPML" / "code"
sys.path.insert(0, str(CODE_DIR))

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from src.models.utils import generate_data

spec = importlib.util.spec_from_file_location("ledh_flow_instrumented", "/tmp/ledh_flow_instrumented.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

base_cfg = OmegaConf.load(CODE_DIR / "configs/config.yaml")
exp_cfg = OmegaConf.load(CODE_DIR / "configs/experiment/acoustic_tracking/acoustic_tracking_ledh_flow.yaml")
model_cfg = OmegaConf.load(CODE_DIR / "configs/model/acoustic_tracking_full.yaml")
filter_cfg = OmegaConf.load(CODE_DIR / "configs/filter/ledh_flow.yaml")
dtype = tf.float64 if str(base_cfg.get("dtype", "float64")) == "float64" else tf.float32

model = hydra.utils.instantiate(model_cfg, dtype=dtype)
initial_state, states, observations = generate_data(model, T=int(exp_cfg.T), rng=np.random.default_rng(int(exp_cfg.seed)))
filter_obj = mod.LocalExactDaumHuangFlow(
    model,
    n_particles=int(filter_cfg.n_particles),
    n_lambda_steps=int(filter_cfg.n_lambda_steps),
    integration_method=str(filter_cfg.integration_method),
    use_feedback=bool(filter_cfg.use_feedback),
    regularization=float(filter_cfg.regularization),
    n_threads=None if filter_cfg.n_threads is None else int(filter_cfg.n_threads),
)
result = filter_obj.filter(observations, random_state=np.random.default_rng(int(exp_cfg.seed)))

def flat(name):
    vals = getattr(mod.LocalExactDaumHuangFlow, name)
    return np.concatenate([np.ravel(v) for v in vals]) if vals else np.array([])

out = {
    "cond_S": flat("cond_S_history"),
    "H_spectral_norm": flat("H_spectral_norm_history"),
    "H_sigma_min": flat("H_sigma_min_history"),
    "A_fro": flat("A_fro_history"),
    "b_norm": flat("b_norm_history"),
    "ess_over_n": np.asarray(mod.LocalExactDaumHuangFlow.ess_over_n_history),
    "particle_cov_trace": np.asarray(mod.LocalExactDaumHuangFlow.particle_cov_trace_history),
    "log_likelihoods": result.log_likelihoods if result.log_likelihoods is not None else np.array([]),
}
np.savez("/tmp/acoustic_ledh_flow_diagnostics.npz", **out)

cond = out["cond_S"]
hmin = out["H_sigma_min"]
a_fro = out["A_fro"]
ess = out["ess_over_n"]
print("cond(S): min median p90 p99 max gt1e4 gt1e6 gt1e8")
print(np.min(cond), np.median(cond), np.percentile(cond, 90), np.percentile(cond, 99), np.max(cond), np.sum(cond > 1e4), np.sum(cond > 1e6), np.sum(cond > 1e8))
print("sigma_min(H): min median max")
print(np.min(hmin), np.median(hmin), np.max(hmin))
print("||A||_F: min median max")
print(np.min(a_fro), np.median(a_fro), np.max(a_fro))
print("ESS/N: min median mean")
print(np.min(ess), np.median(ess), np.mean(ess))
print("cond histogram log bins")
bins = np.logspace(np.log10(max(np.min(cond), 1e-12)), np.log10(np.max(cond)), 11)
hist, edges = np.histogram(cond, bins=bins)
for lo, hi, count in zip(edges[:-1], edges[1:], hist):
    print(f"{lo:.3e} {hi:.3e} {int(count)}")
PYEOF

printf '
[Experiment A] float64 acoustic LEDH flow
' | tee -a "$LOG"
set +e
.venv/bin/python -m src.experiments.run_experiment experiment=acoustic_tracking/acoustic_tracking_ledh_flow_f64 2>&1 | tee -a "$LOG"
EXP_A_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$EXP_A_STATUS" -ne 0 ]]; then
  printf '[Experiment A] ERROR: exited with status %s; continuing to diagnostic.
' "$EXP_A_STATUS" | tee -a "$LOG"
else
  printf '[Experiment A] completed successfully.
' | tee -a "$LOG"
fi

printf '
[Experiment B] instrumented acoustic LEDH flow diagnostic
' | tee -a "$LOG"
set +e
.venv/bin/python /tmp/acoustic_diagnostic.py 2>&1 | tee -a "$LOG"
EXP_B_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$EXP_B_STATUS" -ne 0 ]]; then
  printf '[Experiment B] ERROR: exited with status %s.
' "$EXP_B_STATUS" | tee -a "$LOG"
else
  printf '[Experiment B] completed successfully.
' | tee -a "$LOG"
fi

printf '
[Summary] float64 rerun metrics
' | tee -a "$LOG"
SUMMARY_JSON="$OUT_DIR/summary.json"
if [[ -f "$SUMMARY_JSON" ]]; then
  if command -v jq >/dev/null 2>&1; then
    jq -r '"RMSE: \(.rmse)
log_likelihood: \(.log_likelihood)
wall_time_seconds: \(.metadata.performance.wall_time_seconds)"' "$SUMMARY_JSON" | tee -a "$LOG"
  else
    .venv/bin/python - <<'PYSUM' | tee -a "$LOG"
import json
from pathlib import Path
p = Path('outputs/acoustic_tracking/acoustic_tracking_ledh_flow_f64/summary.json')
d = json.loads(p.read_text())
print(f"RMSE: {d.get('rmse')}")
print(f"log_likelihood: {d.get('log_likelihood')}")
print(f"wall_time_seconds: {d.get('metadata', {}).get('performance', {}).get('wall_time_seconds')}")
PYSUM
  fi
else
  printf 'No summary.json found at %s
' "$SUMMARY_JSON" | tee -a "$LOG"
fi

printf '
Diagnostic summary was printed above by /tmp/acoustic_diagnostic.py and captured in %s
' "$LOG" | tee -a "$LOG"
printf 'Diagnostic NPZ: /tmp/acoustic_ledh_flow_diagnostics.npz
' | tee -a "$LOG"
printf 'Final statuses: Experiment A=%s, Experiment B=%s
' "$EXP_A_STATUS" "$EXP_B_STATUS" | tee -a "$LOG"
