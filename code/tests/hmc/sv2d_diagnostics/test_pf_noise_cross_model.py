"""Cross-model PF noise characterization (Codex 4-metric design).

For each of the four models we have HMC results on (linear gaussian, range
bearing, 1D stochastic volatility, 2D stochastic volatility), profile the
LEDH+OT particle filter over a parameter grid with multiple PF seeds, and
compute four metrics that characterize the surface:

  1. Roughness          — rms(second-difference(ll)) / span(ll), per seed.
                          Low = smooth, high = jagged.
  2. Sign consistency   — at grid points where the seed-averaged slope is
                          non-trivial (|slope| > 0.2 * max|slope|), fraction
                          of (seed, theta) pairs whose gradient sign matches
                          the reference. High = stable direction.
  3. Shape correlation  — mean pairwise correlation of centered loglik curves
                          across seeds. High = same shape, just shifted.
  4. Mode variability   — argmax over grid per seed; mean, SD, # distinct modes.
                          Wide spread = HMC chases seed-specific posteriors.

Each model uses the same filter config its production HMC run used. The
parameter grid spans between init and true values with 25 percent padding.

Pure characterization, no assertions. Saves raw + summary to JSON.

Run:
  cd code && python -m pytest tests/hmc/sv2d_diagnostics/test_pf_noise_cross_model.py -v -s
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from src.models.linear_gaussian import LinearGaussianModel
from src.models.range_bearing import RangeBearingModel
from src.models.stochastic_volatility import StochasticVolatilityModel
from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.differentiable_model import DifferentiableModel
from _gradient_test_utils import save_result, reset_results


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DTYPE = tf.float64    # was float32 — switched to test the precision hypothesis
N_PARTICLES = 500
N_GRID = 8
N_SEEDS = 6
PADDING_FRAC = 0.25
DATA_SEED = 42
ONLY_MODELS = {"stochastic_volatility_2d"}   # set to None to run all four


def _build_lg():
    return LinearGaussianModel(
        F=np.array([[0.9]], dtype=np.float32),
        B=np.array([[1.0]], dtype=np.float32),
        H=np.array([[1.0]], dtype=np.float32),
        D=np.array([[1.0]], dtype=np.float32),
        obs_noise_std=1.0,
        dtype=DTYPE,
    )


def _build_rb():
    return RangeBearingModel(sigma_range=0.1, sigma_bearing=0.1, dtype=DTYPE)


def _build_sv1d():
    return StochasticVolatilityModel(
        alpha=0.91, sigma=1.0, beta=0.5, log_space=True, dtype=DTYPE,
    )


def _build_sv2d():
    return StochasticVolatility2DModel(
        a1=0.95, a2=0.91, sigma1=0.5, sigma2=1.0, b=1.0, dtype=DTYPE,
    )


MODEL_CONFIGS = [
    {
        "name": "linear_gaussian",
        "param": "obs_noise_std",
        "true_value": 1.0,
        "init_value": 2.0,
        "T": 50,
        "n_lambda_steps": 15,
        "build": _build_lg,
        "ess_hmc": 192,
    },
    {
        "name": "range_bearing",
        "param": "sigma_range",
        "true_value": 0.1,
        "init_value": 0.3,
        "T": 50,
        "n_lambda_steps": 15,
        "build": _build_rb,
        "ess_hmc": 25,
    },
    {
        "name": "stochastic_volatility_1d",
        "param": "alpha",
        "true_value": 0.91,
        "init_value": 0.8,
        "T": 100,
        "n_lambda_steps": 29,
        "build": _build_sv1d,
        "ess_hmc": 3.6,
        "param_clip": (0.001, 0.999),
    },
    {
        "name": "stochastic_volatility_2d",
        "param": "sigma2",
        "true_value": 1.0,
        "init_value": 1.6,
        "T": 100,
        "n_lambda_steps": 29,
        "build": _build_sv2d,
        "ess_hmc": None,
    },
]


# -----------------------------------------------------------------------------
# Filter eval
# -----------------------------------------------------------------------------
def _build_filter(diff_model, n_lambda_steps):
    return LEDHParticleFlowFilterHMC(
        model=diff_model,
        n_particles=N_PARTICLES,
        n_lambda_steps=n_lambda_steps,
        resampling_method="ot_entropy",
        resampling_config={"epsilon": 0.5},
        weight_clip_range=50.0,
        stop_gradient_resampling=False,
        eager_mode=False,
        always_resample=True,
    )


def _generate_obs(model_builder, T):
    base = model_builder()
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(base, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def build_model_and_filter(cfg):
    """Build (diff_model, filt) ONCE per model so the @tf.function trace is reused."""
    base_model = cfg["build"]()
    diff_model = DifferentiableModel(base_model, [cfg["param"]])
    filt = _build_filter(diff_model, cfg["n_lambda_steps"])
    return diff_model, filt


def eval_at(cfg, diff_model, filt, theta_val, pf_seed_int, observations):
    param_var = tf.constant(theta_val, dtype=DTYPE)
    pf_seed = tf.constant([pf_seed_int, 0], dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(param_var)
        diff_model.update_parameters({cfg["param"]: param_var})
        ll = filt.log_marginal_likelihood_tf(observations, seed=pf_seed)
    grad = tape.gradient(ll, param_var)

    return float(ll.numpy()), float(grad.numpy() if grad is not None else float("nan"))


# -----------------------------------------------------------------------------
# Grid construction
# -----------------------------------------------------------------------------
def make_grid(cfg, n=N_GRID):
    lo, hi = sorted([cfg["init_value"], cfg["true_value"]])
    pad = (hi - lo) * PADDING_FRAC
    grid_lo, grid_hi = lo - pad, hi + pad
    if "param_clip" in cfg:
        cl, ch = cfg["param_clip"]
        grid_lo = max(grid_lo, cl)
        grid_hi = min(grid_hi, ch)
    return np.linspace(grid_lo, grid_hi, n)


# -----------------------------------------------------------------------------
# Metrics (Codex 4-metric design)
# -----------------------------------------------------------------------------
def metric_roughness(ll_curves):
    """rms(second_diff) / span, median across seeds."""
    rough = []
    for ll in ll_curves:
        second = np.diff(ll, n=2)
        span = max(ll.max() - ll.min(), 1e-12)
        rough.append(float(np.sqrt(np.mean(second ** 2)) / span))
    return float(np.median(rough)), rough


def metric_sign_consistency(ll_curves, grad_curves, theta_grid, threshold=0.2):
    """Fraction of (seed, theta) where sign(grad_seed) matches sign of
    seed-averaged FD slope, restricted to informative grid points."""
    mean_ll = ll_curves.mean(axis=0)
    # Central finite difference of mean_ll
    ref_slope = np.gradient(mean_ll, theta_grid)
    max_abs = np.max(np.abs(ref_slope)) + 1e-12
    informative = np.abs(ref_slope) > threshold * max_abs
    if informative.sum() == 0:
        return float("nan"), informative.sum().item()

    matches = 0
    total = 0
    for seed_idx in range(grad_curves.shape[0]):
        for t_idx in range(grad_curves.shape[1]):
            if not informative[t_idx]:
                continue
            if np.sign(grad_curves[seed_idx, t_idx]) == np.sign(ref_slope[t_idx]):
                matches += 1
            total += 1
    return (matches / total if total > 0 else float("nan")), int(informative.sum())


def metric_shape_correlation(ll_curves):
    """Mean pairwise correlation of centered ll curves across seeds."""
    centered = ll_curves - ll_curves.mean(axis=1, keepdims=True)
    n = centered.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            v_i = centered[i]
            v_j = centered[j]
            num = np.sum(v_i * v_j)
            den = np.sqrt(np.sum(v_i ** 2) * np.sum(v_j ** 2)) + 1e-12
            corrs.append(float(num / den))
    return float(np.mean(corrs)), corrs


def metric_normalized_seed_sd(ll_curves):
    """mean over theta of SD across seeds, divided by span of seed-averaged ll."""
    sd_per_theta = ll_curves.std(axis=0, ddof=1)
    mean_ll = ll_curves.mean(axis=0)
    span = max(mean_ll.max() - mean_ll.min(), 1e-12)
    return float(sd_per_theta.mean() / span)


def metric_mode_variability(ll_curves, theta_grid):
    modes_idx = ll_curves.argmax(axis=1)
    mode_thetas = theta_grid[modes_idx]
    return {
        "mean": float(mode_thetas.mean()),
        "sd": float(mode_thetas.std(ddof=1) if len(mode_thetas) > 1 else 0.0),
        "n_distinct": int(len(set(modes_idx.tolist()))),
        "mode_indices": modes_idx.tolist(),
    }


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
class TestPFNoiseCrossModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)

    def test_compare_models(self):
        all_rows = []

        configs = MODEL_CONFIGS
        if ONLY_MODELS is not None:
            configs = [c for c in configs if c["name"] in ONLY_MODELS]

        for cfg in configs:
            print(f"\n=== {cfg['name']}  param={cfg['param']}  "
                  f"init={cfg['init_value']}  true={cfg['true_value']}  "
                  f"T={cfg['T']}  n_lambda={cfg['n_lambda_steps']} ===")
            obs = _generate_obs(cfg["build"], cfg["T"])
            theta_grid = make_grid(cfg)
            print(f"  grid: {[f'{t:.4f}' for t in theta_grid]}")

            # Build the model + filter ONCE so the tf.function trace is reused
            # across all (seed, theta) calls.
            diff_model, filt = build_model_and_filter(cfg)

            ll_curves = np.full((N_SEEDS, N_GRID), np.nan)
            grad_curves = np.full((N_SEEDS, N_GRID), np.nan)

            t0 = time.perf_counter()
            for s, pf_seed in enumerate(range(N_SEEDS)):
                for k, theta in enumerate(theta_grid):
                    ll, grad = eval_at(cfg, diff_model, filt, float(theta), pf_seed, obs)
                    ll_curves[s, k] = ll
                    grad_curves[s, k] = grad
                print(
                    f"  seed={pf_seed}  ll[mid]={ll_curves[s, N_GRID//2]:+.4f}  "
                    f"grad[mid]={grad_curves[s, N_GRID//2]:+.4f}"
                )
            elapsed = time.perf_counter() - t0
            print(f"  elapsed: {elapsed:.1f}s")

            # Metrics
            roughness_med, roughness_per_seed = metric_roughness(ll_curves)
            sign_frac, n_informative = metric_sign_consistency(
                ll_curves, grad_curves, theta_grid
            )
            shape_corr, _ = metric_shape_correlation(ll_curves)
            seed_sd_norm = metric_normalized_seed_sd(ll_curves)
            mode_info = metric_mode_variability(ll_curves, theta_grid)

            row = {
                "model": cfg["name"],
                "param": cfg["param"],
                "init_value": cfg["init_value"],
                "true_value": cfg["true_value"],
                "T": cfg["T"],
                "n_particles": N_PARTICLES,
                "n_lambda_steps": cfg["n_lambda_steps"],
                "n_seeds": N_SEEDS,
                "n_grid": N_GRID,
                "theta_grid": theta_grid.tolist(),
                "ll_curves": ll_curves.tolist(),
                "grad_curves": grad_curves.tolist(),
                "metrics": {
                    "roughness_median": roughness_med,
                    "roughness_per_seed": roughness_per_seed,
                    "sign_consistency_fraction": sign_frac,
                    "n_informative_grid_points": n_informative,
                    "shape_correlation_mean": shape_corr,
                    "normalized_seed_sd": seed_sd_norm,
                    "mode_variability": mode_info,
                },
                "hmc_ess": cfg["ess_hmc"],
                "elapsed_seconds": elapsed,
            }
            all_rows.append(row)

            print(
                f"  metrics:  rough={roughness_med:.4f}  "
                f"sign={sign_frac:.2f} (over {n_informative} pts)  "
                f"shape_corr={shape_corr:.3f}  seed_sd_norm={seed_sd_norm:.3f}  "
                f"mode SD={mode_info['sd']:.4f} (n_distinct={mode_info['n_distinct']})"
            )

        # Comparison table
        print("\n" + "=" * 110)
        print(f"{'model':<24}{'rough':>10}{'sign_cons':>12}{'shape_corr':>13}"
              f"{'seed_sd_norm':>15}{'mode_sd':>11}{'hmc_ess':>11}")
        print("-" * 110)
        for r in all_rows:
            m = r["metrics"]
            ess_str = f"{r['hmc_ess']}" if r['hmc_ess'] is not None else "frozen"
            print(
                f"{r['model']:<24}"
                f"{m['roughness_median']:>10.4f}"
                f"{m['sign_consistency_fraction']:>12.2f}"
                f"{m['shape_correlation_mean']:>13.3f}"
                f"{m['normalized_seed_sd']:>15.3f}"
                f"{m['mode_variability']['sd']:>11.4f}"
                f"{ess_str:>11}"
            )
        print("=" * 110)

        save_result(__file__, {
            "diagnostic": "pf_noise_cross_model_4metric",
            "config": {
                "dtype": "float32",
                "n_particles": N_PARTICLES,
                "n_grid": N_GRID,
                "n_seeds": N_SEEDS,
                "padding_frac": PADDING_FRAC,
                "data_seed": DATA_SEED,
            },
            "rows": all_rows,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
