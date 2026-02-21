# Kalman Family Test Plan

One test file: `tests/test_kalman_family.py`

## Test 1: EKF reduces to KF on linear Gaussian

- Create a linear Gaussian system (asymmetric F, non-identity H)
- Run KF on generated data
- Run EKF on the same data (with the same model wrapped as nonlinear)
- Assert: means match exactly (`np.allclose`)
- Assert: covariances match exactly (`np.allclose`)

Why: EKF linearizes with Jacobians. For linear f and h, the Jacobian IS the matrix.
If this fails, the EKF predict or update has a bug.

## Test 2: UKF reduces to KF on linear Gaussian

- Same setup as Test 1
- Run UKF on the same data
- Assert: means match KF exactly (`np.allclose`)
- Assert: covariances match KF exactly (`np.allclose`)

Why: The unscented transform is exact for linear functions.
If this fails, sigma point computation or weights are wrong.

## Test 3: Joseph form preserves positive-definiteness under ill-conditioning

- Ill-conditioned system: high-dimensional state (nx=10), wide noise scale range,
  partial observations, long sequence (T=500)
- Run KF with `use_joseph_form=True`
- Assert: all covariances have strictly positive eigenvalues at every step

Why: The only real numerical failure mode for KF. Naive (I-KH)P
loses symmetry/PD over time. Joseph form prevents this.

## Non-tests (deliberately excluded)

- Shape checks: if it runs, shapes are correct. Python crashes on shape mismatch.
- Init/reset: testing that Python stores what you gave it.
- History storage: testing that list.append works.
- Formula re-derivation: already verified analytically by hand.
