"""Diagnostic 4: particle filter Monte Carlo noise at the stuck point.

Question: at sigma2 ~ 1.75 (where yesterday's chain froze), is the particle
filter log-likelihood noisy enough across particle seeds to explain HMC
rejection? If the SD across PF seeds is comparable to the change in mean
log-likelihood across nearby sigma2 values, the chain is seeing a noisy target
and rejection is statistically expected.

Method: at sigma2 in {1.70, 1.75, 1.80}, run the filter NUM_REPLICAS times
with different particle seeds. Record log_lik and gradient. Compute mean and
SD per sigma2. Compare SD to cross-sigma2 mean delta.

Pure characterization, no assertions.

Run:
  cd code && python -m pytest tests/hmc/sv2d_diagnostics/test_pf_noise.py -v -s
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf

from sv2d_diagnostics._setup import (
    build_observations, build_filter, DTYPE, T, N_PARTICLES, N_LAMBDA_STEPS,
)
from _gradient_test_utils import save_result, reset_results


SIGMA2_VALUES = [1.70, 1.75, 1.80]
NUM_REPLICAS = 20
PF_SEEDS = list(range(NUM_REPLICAS))


def eval_at(sigma2_val, pf_seed_int, observations):
    """Build filter at sigma2_val, evaluate log_lik and grad with the given PF seed."""
    diff_model, filt = build_filter(initial_sigma2=sigma2_val)
    sigma2_var = tf.constant(sigma2_val, dtype=DTYPE)
    pf_seed = tf.constant([pf_seed_int, 0], dtype=tf.int32)

    with tf.GradientTape() as tape:
        tape.watch(sigma2_var)
        diff_model.update_parameters({"sigma2": sigma2_var})
        ll = filt.log_marginal_likelihood_tf(observations, seed=pf_seed)
    grad = tape.gradient(ll, sigma2_var)

    return float(ll.numpy()), float(grad.numpy() if grad is not None else float("nan"))


class TestPFNoise(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)
        cls.observations = build_observations()

    def test_pf_noise_sweep(self):
        results_per_sigma = {}

        for sigma2_val in SIGMA2_VALUES:
            print(f"\n--- sigma2 = {sigma2_val} ---")
            lls, grads = [], []
            for pf_seed in PF_SEEDS:
                ll, grad = eval_at(sigma2_val, pf_seed, self.observations)
                lls.append(ll)
                grads.append(grad)
                print(f"  pf_seed={pf_seed:2d}  ll={ll:+.4f}  grad={grad:+.4f}")

            ll_mean = float(np.mean(lls))
            ll_std = float(np.std(lls, ddof=1))
            grad_mean = float(np.mean(grads))
            grad_std = float(np.std(grads, ddof=1))
            print(f"  mean ll={ll_mean:+.4f} ± {ll_std:.4f}    "
                  f"mean grad={grad_mean:+.4f} ± {grad_std:.4f}")

            results_per_sigma[sigma2_val] = {
                "lls": lls, "grads": grads,
                "ll_mean": ll_mean, "ll_std": ll_std,
                "grad_mean": grad_mean, "grad_std": grad_std,
            }

        # Compare cross-sigma2 mean delta to within-sigma2 SD.
        sigmas_sorted = sorted(results_per_sigma.keys())
        comparisons = []
        for i in range(len(sigmas_sorted) - 1):
            s_lo, s_hi = sigmas_sorted[i], sigmas_sorted[i + 1]
            ll_delta = results_per_sigma[s_hi]["ll_mean"] - results_per_sigma[s_lo]["ll_mean"]
            ll_pooled_std = 0.5 * (
                results_per_sigma[s_lo]["ll_std"] + results_per_sigma[s_hi]["ll_std"]
            )
            snr = abs(ll_delta) / ll_pooled_std if ll_pooled_std > 0 else float("inf")
            comparisons.append({
                "from_sigma2": s_lo,
                "to_sigma2": s_hi,
                "ll_mean_delta": ll_delta,
                "ll_pooled_std": ll_pooled_std,
                "snr": snr,
            })
            print(f"  delta sigma2 {s_lo}->{s_hi}: ll change {ll_delta:+.4f}, "
                  f"pooled SD {ll_pooled_std:.4f}, SNR {snr:.2f}")

        verdict = (
            "noise_dominates" if all(c["snr"] < 1.0 for c in comparisons) else
            "signal_dominates" if all(c["snr"] > 3.0 for c in comparisons) else
            "marginal"
        )
        print(f"\n=== verdict: {verdict} ===")

        save_result(__file__, {
            "diagnostic": "4_pf_noise_at_stuck_point",
            "config": {
                "dtype": "float32",
                "n_particles": N_PARTICLES,
                "n_lambda_steps": N_LAMBDA_STEPS,
                "T": T,
                "sigma2_values": SIGMA2_VALUES,
                "num_replicas": NUM_REPLICAS,
                "pf_seeds": PF_SEEDS,
            },
            "verdict": verdict,
            "per_sigma2": {
                str(k): v for k, v in results_per_sigma.items()
            },
            "comparisons": comparisons,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
