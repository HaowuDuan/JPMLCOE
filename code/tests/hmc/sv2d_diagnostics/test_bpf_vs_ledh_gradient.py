"""BPF+OT vs LEDH+OT gradient comparison on SV2D.

The hypothesis: LEDH+OT delivers a near-zero or sign-flipping gradient at
sigma2=1.6..2.0, while BPF+OT delivers a clearly negative gradient at the
same points. If true, this explains both the LEDH MAP failure (final_grad_norm
~0.05, stuck at init=2.0) and the LEDH HMC freeze, and isolates the bug to
the LEDH flow gradient pipeline (NOT to OT, which is shared between the two
filters).

Both filters share:
  - Same SV2D model
  - Same observations (data seed 42, T=100)
  - Same N (1000), same dtype (float32, the production failure mode)
  - Same OT entropy resampling, eps=0.5
  - Same trainable parameter sigma2

Differs:
  - BPF: bootstrap proposal, no flow
  - LEDH: 29-step LEDH flow, then OT resampling

Two passes (both characterization, no assertions):
  - Pass 1: fixed PF seed = 42 across the full grid (matches HMC's deterministic surrogate).
            Records log_likelihood and autodiff gradient.
  - Pass 2: 5 PF seeds at sigma2 in {1.6, 1.8, 2.0} (the region where LEDH MAP gets stuck).
            Records mean grad, SD, sign consistency.

Decisive comparison:
  - BPF at sigma2 in {1.6, 1.8, 2.0}: should show consistently negative gradient
    with |mean| > SD across seeds. (downhill toward truth)
  - LEDH at the same points: if gradient is near zero or sign-flips across seeds,
    the LEDH gradient pipeline is bugged.

Run:
  cd code && python -m pytest tests/hmc/sv2d_diagnostics/test_bpf_vs_ledh_gradient.py -v -s
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.models.utils import generate_data
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.filters.particle.bootstrap_pf_hmc import BootstrapPFHMC
from src.DF.differentiable_model import DifferentiableModel
from _gradient_test_utils import save_result, reset_results


# -----------------------------------------------------------------------------
# Configuration  — both filters identical except for the proposal mechanism
# -----------------------------------------------------------------------------
DTYPE = tf.float32
T = 100
DATA_SEED = 42
N_PARTICLES = 1000
N_LAMBDA_STEPS = 29
EPSILON_OT = 0.5

MODEL_KWARGS = dict(a1=0.95, a2=0.91, sigma1=0.5, b=1.0)
TRUE_SIGMA2 = 1.0

GRID = [1.0, 1.4, 1.6, 1.8, 2.0]
PASS2_GRID = [1.6, 1.8, 2.0]
PASS2_SEEDS = [0, 1, 2, 3, 4]
FIXED_SEED = 42


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------
def build_observations():
    true_model = StochasticVolatility2DModel(sigma2=TRUE_SIGMA2, dtype=DTYPE, **MODEL_KWARGS)
    rng = np.random.default_rng(DATA_SEED)
    _, _, obs = generate_data(true_model, T=T, rng=rng)
    return tf.constant(obs, dtype=DTYPE)


def build_filter(filter_kind: str, sigma2_init: float):
    base_model = StochasticVolatility2DModel(sigma2=sigma2_init, dtype=DTYPE, **MODEL_KWARGS)
    diff_model = DifferentiableModel(base_model, ["sigma2"])

    if filter_kind == "bpf":
        filt = BootstrapPFHMC(
            model=diff_model,
            n_particles=N_PARTICLES,
            resampling_method="ot_entropy",
            resampling_config={"epsilon": EPSILON_OT},
            stop_gradient_resampling=False,
            eager_mode=False,
            always_resample=True,
        )
    elif filter_kind == "ledh":
        filt = LEDHParticleFlowFilterHMC(
            model=diff_model,
            n_particles=N_PARTICLES,
            n_lambda_steps=N_LAMBDA_STEPS,
            resampling_method="ot_entropy",
            resampling_config={"epsilon": EPSILON_OT},
            weight_clip_range=50.0,
            stop_gradient_resampling=False,
            eager_mode=False,
            always_resample=True,
        )
    else:
        raise ValueError(filter_kind)
    return diff_model, filt


def eval_at(filter_kind: str, sigma2_val: float, pf_seed_int: int, observations):
    diff_model, filt = build_filter(filter_kind, sigma2_val)
    sigma2_var = tf.constant(sigma2_val, dtype=DTYPE)
    pf_seed = tf.constant([pf_seed_int, 0], dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(sigma2_var)
        diff_model.update_parameters({"sigma2": sigma2_var})
        ll = filt.log_marginal_likelihood_tf(observations, seed=pf_seed)
    grad = tape.gradient(ll, sigma2_var)
    return float(ll.numpy()), float(grad.numpy() if grad is not None else float("nan"))


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------
class TestBPFvsLEDHGradient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)
        cls.observations = build_observations()

    def test_bpf_vs_ledh_gradient(self):
        # ----- Pass 1: fixed PF seed across the full grid -----
        pass1 = {"bpf": [], "ledh": []}
        for filter_kind in ("bpf", "ledh"):
            print(f"\n=== Pass 1: {filter_kind.upper()}  PF_seed={FIXED_SEED}  grid={GRID} ===")
            t0 = time.perf_counter()
            for sigma2 in GRID:
                ll, grad = eval_at(filter_kind, sigma2, FIXED_SEED, self.observations)
                pass1[filter_kind].append({"sigma2": sigma2, "ll": ll, "grad": grad})
                print(f"  sigma2={sigma2:.2f}  ll={ll:+.4f}  grad={grad:+.4e}")
            print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

        # ----- Pass 2: multiple PF seeds at the stuck region -----
        pass2 = {"bpf": {}, "ledh": {}}
        for filter_kind in ("bpf", "ledh"):
            print(f"\n=== Pass 2: {filter_kind.upper()}  seeds={PASS2_SEEDS}  grid={PASS2_GRID} ===")
            t0 = time.perf_counter()
            for sigma2 in PASS2_GRID:
                lls, grads = [], []
                for pf_seed in PASS2_SEEDS:
                    ll, grad = eval_at(filter_kind, sigma2, pf_seed, self.observations)
                    lls.append(ll)
                    grads.append(grad)
                grad_arr = np.array(grads)
                grad_mean = float(grad_arr.mean())
                grad_sd = float(grad_arr.std(ddof=1))
                ref_sign = np.sign(grad_mean) if grad_mean != 0 else 0
                sign_agreement = float(np.mean(np.sign(grad_arr) == ref_sign))
                pass2[filter_kind][sigma2] = {
                    "lls": lls, "grads": grads,
                    "grad_mean": grad_mean, "grad_sd": grad_sd,
                    "sign_agreement_with_mean": sign_agreement,
                    "snr": abs(grad_mean) / grad_sd if grad_sd > 0 else float("inf"),
                }
                print(
                    f"  sigma2={sigma2:.2f}  grad mean={grad_mean:+.4e}  "
                    f"sd={grad_sd:.4e}  SNR={abs(grad_mean)/max(grad_sd,1e-12):.2f}  "
                    f"sign_agree={sign_agreement:.2f}  grads={[f'{g:+.3e}' for g in grads]}"
                )
            print(f"  elapsed: {time.perf_counter() - t0:.1f}s")

        # ----- Comparison table for pass 1 -----
        print("\n" + "=" * 92)
        print(f"Pass 1 — fixed PF seed = {FIXED_SEED}")
        print(f"{'sigma2':<10}{'BPF ll':>14}{'BPF grad':>14}{'LEDH ll':>14}{'LEDH grad':>14}")
        print("-" * 92)
        for i, sigma2 in enumerate(GRID):
            b = pass1["bpf"][i]
            l = pass1["ledh"][i]
            print(
                f"{sigma2:<10.2f}{b['ll']:>+14.4f}{b['grad']:>+14.4e}"
                f"{l['ll']:>+14.4f}{l['grad']:>+14.4e}"
            )
        print("=" * 92)

        # ----- Verdict -----
        bpf_signs_pass2 = [pass2["bpf"][s]["grad_mean"] for s in PASS2_GRID]
        bpf_snrs = [pass2["bpf"][s]["snr"] for s in PASS2_GRID]
        ledh_signs_pass2 = [pass2["ledh"][s]["grad_mean"] for s in PASS2_GRID]
        ledh_snrs = [pass2["ledh"][s]["snr"] for s in PASS2_GRID]

        bpf_clean = all(g < 0 for g in bpf_signs_pass2) and all(s > 1.0 for s in bpf_snrs)
        ledh_broken = (
            (not all(g < 0 for g in ledh_signs_pass2))
            or any(s < 1.0 for s in ledh_snrs)
        )
        verdict = (
            "ledh_gradient_bug_confirmed" if bpf_clean and ledh_broken else
            "no_clear_pattern" if not bpf_clean else
            "ledh_gradient_actually_ok"
        )
        print(f"\n=== verdict: {verdict} ===")
        print(f"  BPF  pass2 means: {[f'{g:+.3e}' for g in bpf_signs_pass2]}")
        print(f"  LEDH pass2 means: {[f'{g:+.3e}' for g in ledh_signs_pass2]}")
        print(f"  BPF  pass2 SNRs:  {[f'{s:.2f}' for s in bpf_snrs]}")
        print(f"  LEDH pass2 SNRs:  {[f'{s:.2f}' for s in ledh_snrs]}")

        save_result(__file__, {
            "diagnostic": "bpf_vs_ledh_gradient_sv2d",
            "config": {
                "dtype": "float32",
                "T": T,
                "n_particles": N_PARTICLES,
                "n_lambda_steps_ledh": N_LAMBDA_STEPS,
                "epsilon_ot": EPSILON_OT,
                "data_seed": DATA_SEED,
                "fixed_pf_seed": FIXED_SEED,
                "pass2_seeds": PASS2_SEEDS,
                "grid": GRID,
                "pass2_grid": PASS2_GRID,
            },
            "verdict": verdict,
            "pass1": pass1,
            "pass2": {k: {str(s): v for s, v in d.items()} for k, d in pass2.items()},
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
