# Diagnosis & Action Plan: DPF/HMC Kalman Family Integration

---

## 1. Root cause: cubic_sensor filter experiment had no model config

**Symptom:** Filter experiments (`run_experiment.py`) couldn't find cubic_sensor, but HMC DPF experiments (`run_dpf_experiment.py`) worked fine with it.

**Why:** The two runners use completely different config architectures:

| | Filter experiments | DPF experiments |
|---|---|---|
| Base config | `config.yaml` | `config_dpf.yaml` |
| Model config | Required file in `configs/model/` | Inline inside each DPF yaml |
| Composition | Hydra defaults list (`- override /model: cubic_sensor`) | Direct inline YAML (`model: _target_: ...`) |

DPF configs are self-contained. Filter configs require a separate `configs/model/cubic_sensor.yaml`. That file didn't exist until now — **fix applied**: `configs/model/cubic_sensor.yaml` created.

---

## 2. Critical blocker: KF and UKF lack `log_marginal_likelihood_tf`

**Symptom:** Cannot use KalmanFilter or UnscentedKalmanFilter with the HMC runner.

**Why:** `src/DF/hmc_runner.py:87` calls:
```python
log_likelihood = self.filter_obj.log_marginal_likelihood_tf(observations)
```

Status of each Kalman filter:

| Filter | `log_marginal_likelihood_tf` | HMC-compatible? |
|--------|------------------------------|-----------------|
| `KalmanFilter` | ❌ Missing | ❌ No (deeper issue — see §4) |
| `ExtendedKalmanFilter` | ✅ Present (lines 190–252) | ✅ Yes |
| `UnscentedKalmanFilter` | ❌ Missing | ✅ After adding method |

---

## 3. KF DPF incompatibility (structural)

Even after adding `log_marginal_likelihood_tf` to KalmanFilter, HMC parameter inference on KF would not work. The HMC runner modifies model noise params via `DifferentiableModel` wrapper (e.g. `obs_noise_std`). EKF/UKF read noise dynamically from the model (`model.observation_cov(x)`), so changes propagate. KF pre-computes `self.R = D @ D^T` at construction and never re-reads it — so parameter updates would be invisible to it.

**Resolution:** Skip KalmanFilter for DPF/HMC. Add EKF and UKF only.

---

## 4. Files modified / created

### Already done
- `configs/model/cubic_sensor.yaml` ✅
- `configs/filter/kalman_cubic_sensor.yaml` ✅ (block YAML for Hydra interpolation)
- `configs/experiment/cubic_sensor/*.yaml` (6 configs) ✅
- `src/experiments/run_experiment.py` — `.copy()` bug fixed (TF tensor, not numpy) ✅

### Added in this session
- `src/filters/kalman/unscented_kalman.py` — `log_marginal_likelihood_tf` added ✅
- `configs/dpf/hmc/cubic_sensor/ekf.yaml` ✅
- `configs/dpf/hmc/cubic_sensor/ukf.yaml` ✅
- `configs/dpf/hmc/linear_gaussian/ekf.yaml` ✅
- `configs/dpf/hmc/linear_gaussian/ukf.yaml` ✅

---

## 5. How to run

```bash
# From JPMLCOE/code/

# Filter experiments (run_experiment.py)
python -m src.experiments.run_experiment experiment=cubic_sensor/cubic_sensor_ekf
python -m src.experiments.run_experiment experiment=cubic_sensor/cubic_sensor_ukf
python -m src.experiments.run_experiment experiment=cubic_sensor/cubic_sensor_pf_sys
python -m src.experiments.run_experiment experiment=cubic_sensor/cubic_sensor_kf

# DPF / HMC parameter inference (run_dpf_experiment.py)
python -m src.experiments.run_dpf_experiment dpf=hmc/cubic_sensor/ekf
python -m src.experiments.run_dpf_experiment dpf=hmc/cubic_sensor/ukf
python -m src.experiments.run_dpf_experiment dpf=hmc/linear_gaussian/ekf
python -m src.experiments.run_dpf_experiment dpf=hmc/linear_gaussian/ukf
```
