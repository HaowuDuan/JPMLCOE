"""Diagnostics 1 and 2: leapfrog replay and step-size sweep.

Replay one HMC leapfrog trajectory in isolation, with full per-substep
instrumentation, given a fixed initial state, momentum, particle filter seed,
step size, and num_leapfrog. Used to answer:

  Diagnostic 1 — Rejected-step replay
    Is the rejection from huge integration error (delta_H >> 1) or from a
    geometrically bad endpoint (delta_H ~ 0.1-1, U_new much worse than U_start)?

  Diagnostic 2 — Same-trajectory step-size sweep
    Hold momentum and PF seed fixed, sweep step_size in {0.02, 0.01, 0.005}.
    If delta_H collapses with smaller step, integration error dominates. If not,
    the geometry is the issue.

This file builds the replay harness once and runs both diagnostics. No
assertions, characterization only.

Run:
  cd code && python -m pytest tests/hmc/sv2d_diagnostics/test_hmc_leapfrog_replay.py -v -s
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf

from sv2d_diagnostics._setup import (
    build_all, INIT_SIGMA2, DTYPE, T, N_PARTICLES, N_LAMBDA_STEPS,
)
from _gradient_test_utils import save_result, reset_results


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NUM_LEAPFROG = 5
STEP_SIZES = [0.02, 0.01, 0.005]
MOMENTUM_SEED = 42  # deterministic momentum draw


# -----------------------------------------------------------------------------
# Replay harness
# -----------------------------------------------------------------------------
def _value_and_grad(target_log_prob_fn, q):
    with tf.GradientTape() as tape:
        tape.watch(q)
        lp = target_log_prob_fn(q)
    grad = tape.gradient(lp, q)
    return lp, grad


def replay_leapfrog(target_log_prob_fn, q0, p0, step_size, num_leapfrog):
    """Run num_leapfrog leapfrog steps and return per-substep diagnostics.

    Standard kinetic energy K(p) = 0.5 * ||p||^2 (unit mass).
    Hamiltonian H = U + K where U = -log_prob.
    """
    q = tf.identity(q0)
    p = tf.identity(p0)

    lp_start, _ = _value_and_grad(target_log_prob_fn, q)
    U0 = -float(lp_start.numpy())
    K0 = float(0.5 * tf.reduce_sum(p ** 2).numpy())
    H0 = U0 + K0

    trace = []
    for k in range(num_leapfrog):
        # Half step in p
        lp, grad_lp = _value_and_grad(target_log_prob_fn, q)
        # grad of U = -grad of log_prob
        p = p - 0.5 * step_size * (-grad_lp)
        # Full step in q
        q = q + step_size * p
        # Half step in p
        lp_new, grad_lp_new = _value_and_grad(target_log_prob_fn, q)
        p = p - 0.5 * step_size * (-grad_lp_new)

        U = -float(lp_new.numpy())
        K = float(0.5 * tf.reduce_sum(p ** 2).numpy())
        H = U + K
        trace.append({
            "k": k + 1,
            "q_unconstrained": float(q.numpy()[0]),
            "sigma2": float(tf.nn.softplus(q[0]).numpy()),
            "p": float(p.numpy()[0]),
            "U": U,
            "K": K,
            "H": H,
            "delta_H": H - H0,
            "grad_log_prob": float(grad_lp_new.numpy()[0]),
            "grad_norm": float(tf.norm(grad_lp_new).numpy()),
        })

    delta_H_total = trace[-1]["delta_H"]
    accept_prob = float(min(1.0, tf.exp(-delta_H_total).numpy()))

    return {
        "H0": H0,
        "U0": U0,
        "K0": K0,
        "trace": trace,
        "delta_H_final": delta_H_total,
        "U_final": trace[-1]["U"],
        "accept_prob": accept_prob,
    }


def draw_momentum(q0, seed_int):
    """Deterministic momentum draw from a stateless seed. Unit mass, std N(0,1)."""
    return tf.random.stateless_normal(
        shape=q0.shape, seed=[seed_int, 0], dtype=q0.dtype,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestHMCLeapfrogReplay(unittest.TestCase):
    """Diagnostics 1 + 2: leapfrog replay and step-size sweep."""

    @classmethod
    def setUpClass(cls):
        reset_results(__file__)
        cls.bundle = build_all(initial_sigma2=INIT_SIGMA2)

    def _replay_at_step_size(self, step_size, q0, p0):
        return replay_leapfrog(
            target_log_prob_fn=self.bundle["target_log_prob_fn"],
            q0=q0, p0=p0,
            step_size=step_size,
            num_leapfrog=NUM_LEAPFROG,
        )

    def test_replay_and_step_size_sweep(self):
        """Diagnostics 1 and 2 together: same q0, same p0, sweep step_size."""
        q0 = self.bundle["init_unconstrained"]
        p0 = draw_momentum(q0, MOMENTUM_SEED)
        print(f"\nq0 (unconstrained) = {float(q0.numpy()[0]):+.6f}, "
              f"sigma2 = {float(tf.nn.softplus(q0[0]).numpy()):.4f}")
        print(f"p0 = {float(p0.numpy()[0]):+.6f}  (seed={MOMENTUM_SEED})")

        sweep_results = []
        for step_size in STEP_SIZES:
            print(f"\n--- step_size = {step_size} ---")
            res = self._replay_at_step_size(step_size, q0, p0)

            for substep in res["trace"]:
                print(
                    f"  k={substep['k']}  "
                    f"sigma2={substep['sigma2']:.4f}  "
                    f"U={substep['U']:+.4f}  K={substep['K']:.4f}  "
                    f"H={substep['H']:+.4f}  dH={substep['delta_H']:+.4f}  "
                    f"|grad|={substep['grad_norm']:.3f}"
                )
            print(
                f"  H0={res['H0']:+.4f} -> H_final={res['trace'][-1]['H']:+.4f}  "
                f"delta_H_final={res['delta_H_final']:+.4f}  "
                f"accept_prob={res['accept_prob']:.4f}"
            )

            sweep_results.append({
                "step_size": step_size,
                "num_leapfrog": NUM_LEAPFROG,
                "H0": res["H0"],
                "U0": res["U0"],
                "K0": res["K0"],
                "delta_H_final": res["delta_H_final"],
                "U_final": res["U_final"],
                "accept_prob": res["accept_prob"],
                "sigma2_final": res["trace"][-1]["sigma2"],
                "trace": res["trace"],
            })

        # Interpretation: how does delta_H scale with step_size?
        # If integration error is the issue, delta_H should fall ~quadratically.
        ratios = []
        for i in range(len(sweep_results) - 1):
            big = abs(sweep_results[i]["delta_H_final"])
            small = abs(sweep_results[i + 1]["delta_H_final"])
            ratio = big / small if small > 1e-10 else float("inf")
            ratios.append({
                "from_step_size": sweep_results[i]["step_size"],
                "to_step_size": sweep_results[i + 1]["step_size"],
                "abs_dH_ratio": ratio,
            })
            print(
                f"  step {sweep_results[i]['step_size']} -> {sweep_results[i+1]['step_size']}: "
                f"|dH| {big:.4f} -> {small:.4f}  (ratio {ratio:.2f})"
            )

        # Verdict
        first_dH = abs(sweep_results[0]["delta_H_final"])
        verdict = (
            "integration_error_dominant"
            if all(r["abs_dH_ratio"] > 2.0 for r in ratios) and first_dH > 1.0
            else
            "geometry_dominant"
            if first_dH < 1.0 and sweep_results[0]["U_final"] > sweep_results[0]["U0"] + 5
            else
            "ambiguous"
        )
        print(f"\n=== verdict: {verdict} ===")

        save_result(__file__, {
            "diagnostic": "1_2_leapfrog_replay_and_step_sweep",
            "config": {
                "dtype": "float32",
                "n_particles": N_PARTICLES,
                "n_lambda_steps": N_LAMBDA_STEPS,
                "T": T,
                "init_sigma2": INIT_SIGMA2,
                "step_sizes": STEP_SIZES,
                "num_leapfrog": NUM_LEAPFROG,
                "momentum_seed": MOMENTUM_SEED,
                "pf_seed": "fixed [42, 0]",
            },
            "q0_unconstrained": float(q0.numpy()[0]),
            "p0": float(p0.numpy()[0]),
            "verdict": verdict,
            "step_ratios": ratios,
            "sweep": sweep_results,
        })


if __name__ == "__main__":
    unittest.main(verbosity=2)
