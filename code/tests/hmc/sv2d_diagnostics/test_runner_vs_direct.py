"""Diagnostic: what does the production runner do differently from a direct
TFP kernel call that causes the SV2D HMC chain to freeze?

Yesterday's run via DPFRunner.run_inference froze after step 1 (rejection
cascade). Today's direct HamiltonianMonteCarlo call with the SAME seed (42)
accepts cleanly. Something between "build kernel + step it" and
"DPFRunner.run_inference" is breaking the chain.

Three conditions, all at seed=42, all identical otherwise:
  A. Direct kernel        — tfp.mcmc.HamiltonianMonteCarlo, plain.
  B. Wrapped kernel       — wrapped in DualAveragingStepSizeAdaptation,
                            num_adaptation_steps=0 (matches the runner exactly).
  C. Full runner pipeline — DPFRunner.run_inference, the actual code that froze.

Comparison logic:
  A clean, B clean, C frozen   -> bug is in DPFRunner machinery (not the wrapper).
  A clean, B frozen, C frozen  -> bug is in DualAveragingStepSizeAdaptation.
  A clean, B clean,  C clean   -> yesterday's freeze was a transient anomaly.

No assertions. Save full per-step trace + verdict to JSON.

Run:
  cd code && python -m pytest tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py -v -s
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from sv2d_diagnostics._setup import (
    build_all, build_observations,
    INIT_SIGMA2, DTYPE, T, N_PARTICLES, N_LAMBDA_STEPS,
    EPSILON_OT, WEIGHT_CLIP, MODEL_KWARGS,
)
from src.models.stochastic_volatility_2d import StochasticVolatility2DModel
from src.filters.particle.ledh_invertible_hmc import LEDHParticleFlowFilterHMC
from src.DF.types import ParameterSpec
from src.DF.hmc_runner import DPFRunner
from _gradient_test_utils import save_result, reset_results


# -----------------------------------------------------------------------------
# Configuration (all conditions identical)
# -----------------------------------------------------------------------------
HMC_SEED = 42
NUM_BURNIN = 10
STEP_SIZE = 0.02
NUM_LEAPFROG = 5
TARGET_ACCEPT = 0.65


# -----------------------------------------------------------------------------
# Conditions A and B — direct kernel control
# -----------------------------------------------------------------------------
def _build_inner_kernel(target_log_prob_fn):
    return tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG,
    )


def _read(kr, wrapped):
    if wrapped:
        return (
            bool(kr.inner_results.is_accepted.numpy()),
            float(kr.inner_results.accepted_results.target_log_prob.numpy()),
        )
    return (
        bool(kr.is_accepted.numpy()),
        float(kr.accepted_results.target_log_prob.numpy()),
    )


def run_chain_direct(target_log_prob_fn, q0, wrapped: bool):
    inner = _build_inner_kernel(target_log_prob_fn)
    if wrapped:
        kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=inner,
            num_adaptation_steps=0,
            target_accept_prob=TARGET_ACCEPT,
        )
    else:
        kernel = inner

    tf.random.set_seed(HMC_SEED)
    current = q0
    kr = kernel.bootstrap_results(current)

    records = []
    for step in range(NUM_BURNIN):
        t0 = time.perf_counter()
        new, kr = kernel.one_step(current, kr)
        dt = time.perf_counter() - t0
        accepted, lp = _read(kr, wrapped)
        sigma2 = float(tf.nn.softplus(new[0]).numpy())
        records.append({
            "step": step + 1,
            "accepted": accepted,
            "sigma2": sigma2,
            "log_prob": lp,
            "wall_time_s": dt,
        })
        current = new
    return records


# -----------------------------------------------------------------------------
# Condition C — full DPFRunner pipeline
# -----------------------------------------------------------------------------
def run_chain_via_runner():
    """Use DPFRunner.run_inference exactly as the production code does."""
    base_model = StochasticVolatility2DModel(
        sigma2=INIT_SIGMA2, dtype=DTYPE, **MODEL_KWARGS,
    )
    spec = ParameterSpec(
        name="sigma2",
        init_value=INIT_SIGMA2,
        constraint="positive",
        prior=tfp.distributions.LogNormal(
            loc=tf.constant(0.0, dtype=DTYPE),
            scale=tf.constant(0.5, dtype=DTYPE),
        ),
    )
    runner = DPFRunner(
        base_model=base_model,
        filter_class=LEDHParticleFlowFilterHMC,
        filter_kwargs=dict(
            n_particles=N_PARTICLES,
            n_lambda_steps=N_LAMBDA_STEPS,
            resampling_method="ot_entropy",
            resampling_config={"epsilon": EPSILON_OT},
            weight_clip_range=WEIGHT_CLIP,
            stop_gradient_resampling=False,
            eager_mode=False,
            always_resample=True,
        ),
        param_specs={"sigma2": spec},
        sampler="hmc",
    )
    obs_np = build_observations().numpy()
    result = runner.run_inference(
        observations=obs_np,
        num_samples=0,
        num_burnin=NUM_BURNIN,
        step_size=STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG,
        adaptation_rate=0.001,    # ~0 adaptation steps
        target_accept_prob=TARGET_ACCEPT,
        seed=HMC_SEED,
        grad_clip_norm=100.0,
    )
    diag = result.diagnostics
    is_accepted = diag.get("is_accepted_per_step", []) or diag.get("is_accepted", [])
    step_sizes = diag.get("step_size_per_step", []) or diag.get("step_size", [])
    records = []
    for i, acc in enumerate(is_accepted):
        rec = {"step": i + 1, "accepted": bool(acc)}
        if i < len(step_sizes):
            rec["step_size_at_step"] = float(step_sizes[i])
        records.append(rec)
    return records, diag


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def summarize(records, label):
    n = len(records)
    n_acc = sum(1 for r in records if r["accepted"])
    accepts = [r["accepted"] for r in records]
    longest_streak = 0
    cur = 0
    for a in accepts:
        if not a:
            cur += 1
            longest_streak = max(longest_streak, cur)
        else:
            cur = 0
    freeze = (
        n >= 6 and accepts[0] and not any(accepts[1:6])
    )
    final_sigma2 = records[-1].get("sigma2", None) if records else None
    return {
        "label": label,
        "n_steps": n,
        "n_accepts": n_acc,
        "accept_rate": n_acc / n if n > 0 else 0.0,
        "longest_rejection_streak": longest_streak,
        "final_sigma2": final_sigma2,
        "freeze_after_first_accept": freeze,
    }


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestRunnerVsDirect(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)

    def _print_records(self, records, label):
        print(f"\n--- {label} ---")
        for r in records:
            extra = (
                f"  sigma2={r['sigma2']:.4f}  lp={r['log_prob']:+.4f}"
                if "sigma2" in r else ""
            )
            print(f"  step {r['step']:2d}  accepted={r['accepted']}{extra}")

    def test_A_direct_kernel(self):
        bundle = build_all(initial_sigma2=INIT_SIGMA2)
        records = run_chain_direct(
            bundle["target_log_prob_fn"], bundle["init_unconstrained"], wrapped=False
        )
        self._print_records(records, "A: direct kernel")
        summary = summarize(records, "A_direct_kernel")
        save_result(__file__, {
            "diagnostic": "runner_vs_direct_A_direct",
            "summary": summary,
            "records": records,
        })

    def test_B_wrapped_kernel(self):
        bundle = build_all(initial_sigma2=INIT_SIGMA2)
        records = run_chain_direct(
            bundle["target_log_prob_fn"], bundle["init_unconstrained"], wrapped=True
        )
        self._print_records(records, "B: wrapped kernel (DualAveraging num_adaptation_steps=0)")
        summary = summarize(records, "B_wrapped_kernel")
        save_result(__file__, {
            "diagnostic": "runner_vs_direct_B_wrapped",
            "summary": summary,
            "records": records,
        })

    def test_C_full_runner(self):
        records, diag = run_chain_via_runner()
        self._print_records(records, "C: full DPFRunner.run_inference")
        summary = summarize(records, "C_full_runner")
        save_result(__file__, {
            "diagnostic": "runner_vs_direct_C_full_runner",
            "summary": summary,
            "records": records,
            "diagnostics_keys_seen": list(diag.keys()),
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
