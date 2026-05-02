# Diagnosis for JAX + JIT Rebuild

This pass catalogs non-uniform design patterns, design smells, and structural problems in the current TensorFlow/Hydra codebase. It is descriptive and oriented toward a JAX rebuild.

## Numeric-Type Discipline

### Mixed dtype defaults across config, model constructors, and runners

Representative sites: `code/configs/config.yaml:13`, `code/configs/config_dpf.yaml:9`, `code/src/models/linear_gaussian.py:47`, `code/src/experiments/run_experiment.py:205`, `code/src/experiments/run_dpf_experiment.py:137`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:6`.

Today, the main Hydra configs default to `float64`, the `LinearGaussianModel` constructor defaults to `tf.float32`, `run_experiment` defaults missing config dtype to `float64`, and `run_dpf_experiment` defaults missing config dtype to `float32`. Individual DPF configs also override dtype at the experiment level; one SV2D smoke config records that `float32` damaged the gradient surface. This makes numerical behavior depend on which entry point and config layer created the same conceptual model.

Under JAX, this will affect `jit` cache keys, device behavior, and gradient checks. A function compiled for `float32` is a different compiled program than the same function compiled for `float64`; implicit promotion is also more explicit in JAX when `jax_enable_x64` is off. Mixed dtype choices will create recompilation and can silently change posterior geometry or Sinkhorn stability.

Uniform replacement pattern: make dtype a single explicit experiment-level static choice, convert it once into a JAX dtype at the boundary, and pass it into model/filter constructors as static config. Internal kernels should derive all constants from array dtypes (`jnp.asarray(value, dtype=x.dtype)`) and should not have independent Python defaults that conflict with config.

### Python/NumPy/SciPy numeric work is embedded in model construction

Representative sites: `code/src/models/linear_gaussian.py:70`, `code/src/models/linear_gaussian.py:78`, `code/src/models/linear_gaussian.py:117`, `code/src/models/linear_gaussian.py:118`, `code/src/models/linear_gaussian.py:123`, `code/src/models/linear_gaussian.py:125`.

Today, the linear Gaussian model stores both TensorFlow and NumPy dtype choices, converts constructor inputs to `tf.constant`, and computes a default stationary covariance through NumPy/SciPy plus `.numpy()` from a TensorFlow tensor. This is workable during eager object construction, but the model boundary is not cleanly separated into static preprocessing versus differentiable array computation.

For JAX, SciPy/NumPy work inside model construction is harmless only if construction is strictly outside `jit`, outside `grad`, and outside vectorized runs. If model initialization becomes part of a compiled factory or if initial covariance depends on learnable parameters, host-side NumPy/SciPy calls and `.numpy()` equivalents cannot be traced.

Uniform replacement pattern: split model config normalization from model evaluation. Precompute static matrices and optional stationary covariances outside compiled kernels, or implement the required solve with `jax.scipy` and return it as part of a pure model pytree. Compiled code should receive arrays, not objects that compute hidden NumPy state.

### Prior distributions carry independent dtype and are cast around

Representative sites: `code/src/DF/parameter_handler.py:149`, `code/src/DF/parameter_handler.py:157`, `code/src/DF/parameter_handler.py:159`, `code/src/DF/parameter_handler.py:160`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:35`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:39`.

Today, priors are instantiated from Hydra as TensorFlow Probability distribution objects and may have a dtype different from the model or sampler dtype. The handler compensates by casting parameter values to the prior dtype and casting log-probabilities back.

Under JAX, parameter transforms and priors need to be part of the same traced log-density. A distribution object with hidden dtype, object methods, and runtime casts will either be outside JIT or cause non-uniform compiled programs. Mixed prior/model dtype also changes gradients around bijectors.

Uniform replacement pattern: represent each parameter spec as a pure pytree containing dtype-normalized scalar arrays and a small enum for transform and prior family. Implement `constrain`, `unconstrain`, `log_abs_det_jacobian`, and `log_prior` as pure JAX functions over ordered parameter vectors or named pytrees.

## Parameter Handling

### Learnable parameters are mutable model attributes

Representative sites: `code/src/DF/differentiable_model.py:12`, `code/src/DF/differentiable_model.py:46`, `code/src/DF/differentiable_model.py:57`, `code/src/DF/differentiable_model.py:60`, `code/src/DF/hmc_runner.py:86`, `code/src/DF/hmc_runner.py:90`, `code/src/filters/particle/ledh_invertible_hmc.py:192`, `code/src/filters/particle/ledh_invertible_hmc.py:194`, `code/src/filters/particle/ledh_invertible_hmc.py:422`, `code/src/filters/particle/ledh_invertible_hmc.py:435`.

Today, HMC and MAP preserve gradients by assigning constrained tensor values onto model attributes with `setattr`, then restoring Python-side values after compiled calls. The HMC filter also mutates model attributes inside a traced function so the graph reads symbolic tensors.

Under JAX, mutating object attributes from inside a `jit` or `grad` region is not a valid program model. It also makes `vmap` unsafe because parallel parameter values would race through one shared object. Even outside JAX, this design creates stale-trace risk when compiled closures capture model methods and attribute names.

Uniform replacement pattern: make model parameters explicit data. Use a model definition pytree for static structure and a parameter pytree for dynamic values. Every model function should take `(params, state, input, key, static_model)` and return arrays. Filters and samplers should never write parameters into a model object.

### Trainable versus frozen parameters are inferred from YAML coupling

Representative sites: `code/src/experiments/run_dpf_experiment.py:72`, `code/src/experiments/run_dpf_experiment.py:84`, `code/src/experiments/run_dpf_experiment.py:87`, `code/src/experiments/run_dpf_experiment.py:90`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:8`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:14`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:35`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:56`.

Today, a trainable parameter must exist as a model config key, a model attribute, a `dpf.trainable_params` entry, and optionally a `data.true_params` entry. `_build_param_specs` pulls the initial value from `cfg.model[name]` unless an override exists. Frozen parameters are whatever remains in the model config.

For JAX, implicit coupling makes it hard to build one typed parameter pytree and a stable JIT signature. It also makes parameter subsets hard to `vmap` across experiments because the meaning of the same model config changes when `trainable_params` changes.

Uniform replacement pattern: define a single parameter schema per model with fields marked as `trainable`, `fixed`, or `data_true`. Build three explicit pytrees: `theta_init`, `theta_fixed`, and `theta_true` for simulation. The sampler receives only `theta_trainable_unconstrained`; the likelihood receives a merged parameter pytree produced by a pure function.

### Samplers and filters duplicate parameter plumbing

Representative sites: `code/src/DF/hmc_runner.py:79`, `code/src/DF/hmc_runner.py:87`, `code/src/DF/hmc_runner.py:99`, `code/src/DF/hmc_runner.py:336`, `code/src/DF/hmc_runner.py:340`, `code/src/DF/hmc_runner.py:346`, `code/src/filters/particle/bootstrap_pf_hmc.py:231`, `code/src/filters/particle/bootstrap_pf_hmc.py:245`, `code/src/filters/particle/ledh_invertible_hmc.py:407`, `code/src/filters/particle/ledh_invertible_hmc.py:417`.

Today, DPFRunner constrains parameters and mutates the model before calling a filter. HMC-specialized filters then rediscover trainable names, stack current parameter values, mutate the model inside compiled closures, and restore attributes after calls. Parameter passing is therefore split across runner, wrapper, and filter.

In JAX, this pattern would fragment the JIT boundary and make gradients depend on hidden side effects. It also prevents clean `vmap` over parameter particles or chains because parameters are not explicit function arguments all the way down.

Uniform replacement pattern: create one likelihood function with signature like `loglik(theta, data, key, static_cfg) -> scalar, aux`. The sampler owns unconstrained-space transforms; the filter owns only filtering state transitions. Parameter values should flow through normal function arguments from sampler to model to filter.

## Control Flow and Mutation

### Filters maintain mutable TensorFlow Variables and Python history lists

Representative sites: `code/src/filters/particle/particle_base.py:38`, `code/src/filters/particle/particle_base.py:43`, `code/src/filters/particle/ledh_invertible.py:83`, `code/src/filters/particle/ledh_invertible.py:97`, `code/src/filters/particle/ledh_invertible.py:156`, `code/src/filters/particle/ledh_invertible.py:160`, `code/src/filters/particle/ledh_invertible.py:192`, `code/src/filters/particle/ledh_invertible.py:205`, `code/src/filters/kalman/extended_kalman.py:72`, `code/src/filters/kalman/extended_kalman.py:83`, `code/src/filters/kalman/extended_kalman.py:258`.

Today, experiment-path filters are objects with mutable particles, weights, covariance variables, RNG state, and append-only Python diagnostic lists. This makes the filter object both an algorithm and a result accumulator.

JAX `jit`, `grad`, `vmap`, and `scan` require state to be explicit values. Hidden object mutation cannot be traced, cannot be parallelized safely, and cannot be replayed deterministically from inputs alone. Python lists accumulated during filtering also force host execution or dynamic shapes.

Uniform replacement pattern: model every filter as pure transitions over a typed carry: `carry_t -> carry_{t+1}, output_t`. Use `lax.scan` for time, fixed-shape arrays for diagnostics, and a separate host-side postprocessor for saving or plotting results.

### Python loops, list accumulation, and host conversions are used in hot algorithm paths

Representative sites: `code/src/filters/kalman/extended_kalman.py:213`, `code/src/filters/kalman/extended_kalman.py:219`, `code/src/filters/kalman/extended_kalman.py:299`, `code/src/filters/kalman/extended_kalman.py:302`, `code/src/filters/kalman/extended_kalman.py:313`, `code/src/filters/particle/ledh_invertible.py:404`, `code/src/filters/particle/ledh_invertible.py:409`, `code/src/filters/particle/ledh_invertible.py:417`, `code/src/DF/hmc_runner.py:242`, `code/src/DF/hmc_runner.py:263`.

Today, some differentiable likelihood paths use Python `for range(T)` over `observations.shape[0]`, while experiment paths append tensors or NumPy arrays to lists and convert at the end. The HMC runner steps TFP kernels in Python and stores samples with `.numpy().copy()`.

For JAX, Python loops over dynamic `T` are either unrolled at trace time or fail when `T` is traced. Appending to lists is not JIT-compatible. Host conversions inside a sampling loop prevent compiling the loop and also prevent `vmap` over chains.

Uniform replacement pattern: use `lax.scan` for time-recursive filters and sampler trajectories. Allocate diagnostics as scan outputs with fixed shapes. Keep host conversions at the outermost boundary after compiled functions return.

### Tensor-dependent branch points create non-smooth or non-JIT-safe paths

Representative sites: `code/src/filters/particle/bootstrap_pf_hmc.py:160`, `code/src/filters/particle/bootstrap_pf_hmc.py:179`, `code/src/filters/particle/bootstrap_pf_hmc.py:180`, `code/src/filters/particle/ledh_invertible_hmc.py:301`, `code/src/filters/particle/ledh_invertible_hmc.py:326`, `code/src/filters/particle/ledh_invertible_hmc.py:327`, `code/src/filters/particle/bootstrap_pf_hmc.py:317`, `code/src/filters/particle/bootstrap_pf_hmc.py:318`, `code/src/filters/kalman/extended_kalman.py:174`, `code/src/filters/kalman/extended_kalman.py:175`.

Today, compiled HMC filters use `tf.cond` for conditional resampling, while eager paths branch with Python `if ess < ...`. Comments already note discontinuity and HMC step-size collapse. EKF update also conditionally skips an update based on a tensor norm.

In JAX, Python `if` on a tracer is invalid. `lax.cond` is valid but still creates branch-dependent gradient structure. For HMC and gradient-based optimization, discontinuous resampling switches are a modeling issue, not only a compiler issue.

Uniform replacement pattern: choose a JAX-safe resampling policy per algorithm. For differentiable likelihoods, prefer always-on smooth resampling or a continuous relaxation with fixed computation. For non-differentiable particle filters, use explicit `lax.cond` outside gradient-based likelihoods and document that the likelihood is not smooth.

### Compiled closures capture object state and static assumptions

Representative sites: `code/src/filters/particle/ledh_invertible_hmc.py:41`, `code/src/filters/particle/ledh_invertible_hmc.py:44`, `code/src/filters/particle/ledh_invertible_hmc.py:134`, `code/src/filters/particle/ledh_invertible_hmc.py:146`, `code/src/filters/particle/ledh_invertible_hmc.py:150`, `code/src/filters/particle/ledh_invertible_hmc.py:184`, `code/src/filters/particle/bootstrap_pf_hmc.py:91`, `code/src/filters/particle/bootstrap_pf_hmc.py:102`, `code/src/filters/particle/bootstrap_pf_hmc.py:114`.

Today, HMC-specialized filters build compiled functions that capture model objects, particle counts, resampling choices, method maps, and trainable parameter names by closure. The file itself warns that parameter-dependent model methods can become stale after parameter updates.

JAX compilation also separates static and dynamic values, but captured Python objects are treated as static constants. If model behavior depends on mutable object fields, recompilation or stale compiled behavior is likely. This is especially harmful for `vmap` over models or parameter subsets.

Uniform replacement pattern: make all static choices explicit static arguments or dataclass fields in a frozen config, and make all dynamic values explicit arrays. Avoid closures over mutable objects. Compile pure functions whose static inputs are hashable configs and whose dynamic inputs are pytrees.

## Effectful Code Inside or Near Hot Paths

### Randomness is split across object state, global state, and ad hoc seeds

Representative sites: `code/src/filters/particle/ledh_invertible.py:105`, `code/src/filters/particle/ledh_invertible.py:113`, `code/src/filters/particle/ledh_invertible.py:116`, `code/src/filters/particle/bootstrap_pf_tf.py:234`, `code/src/DF/hmc_runner.py:93`, `code/src/DF/hmc_runner.py:215`, `code/src/DF/hmc_runner.py:489`, `code/src/DF/hmc_runner.py:528`, `code/src/DF/hmc_runner.py:562`, `code/src/experiments/run_experiment.py:203`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:51`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:55`.

Today, some filters carry mutable stateless RNG keys, some call `tf.random.uniform` to create seeds, DPFRunner fixes the PF likelihood seed to `[42, 0]`, and samplers call `tf.random.set_seed`. Config files also carry multiple seeds for sampler and generated data.

In JAX, RNG is explicit. Hidden key mutation, global seeding, and fixed inner likelihood seeds will either be impossible under `jit` or will make reproducibility and vectorization unclear. `vmap` over chains requires a key per chain and predictable splitting.

Uniform replacement pattern: thread `jax.random.PRNGKey` explicitly through data generation, filter transitions, resampling, and sampler transitions. Store seed policy in config as data, then derive keys in one place. Avoid global random state and avoid hard-coded likelihood seeds inside target log-probability functions.

### Logging, callbacks, timing, and `.numpy()` are interleaved with computation

Representative sites: `code/src/experiments/run_experiment.py:100`, `code/src/experiments/run_experiment.py:106`, `code/src/experiments/run_experiment.py:237`, `code/src/DF/hmc_runner.py:112`, `code/src/DF/hmc_runner.py:115`, `code/src/DF/hmc_runner.py:121`, `code/src/DF/hmc_runner.py:122`, `code/src/DF/hmc_runner.py:228`, `code/src/DF/hmc_runner.py:250`, `code/src/DF/hmc_runner.py:391`, `code/src/filters/particle/ledh_invertible_hmc.py:597`, `code/src/filters/particle/ledh_invertible_hmc.py:601`.

Today, diagnostic prints, `tf.print`, Python callbacks, timing, and `.numpy()` conversions appear inside gradient evaluation and sampling loops. Some are debug-only, but they live in the same runner paths as production inference.

Under JAX, side effects inside `jit` are restricted and callbacks are special-purpose tools with synchronization cost. `.numpy()`-style host materialization breaks compilation and forces device synchronization. Timing inside compiled code measures host behavior poorly.

Uniform replacement pattern: compiled functions should return metrics as auxiliary arrays or scalar summaries. Host code can print, time, call callbacks, and write logs after compiled steps. For long samplers, use a pure scan that returns a trace buffer, then summarize it outside JIT.

### File I/O and result serialization are coupled to experiment runners and tests

Representative sites: `code/src/experiments/run_experiment.py:294`, `code/src/experiments/run_experiment.py:300`, `code/src/experiments/run_experiment.py:329`, `code/src/experiments/run_dpf_experiment.py:379`, `code/src/experiments/run_dpf_experiment.py:405`, `code/src/experiments/run_dpf_experiment.py:411`, `code/tests/hmc/_gradient_test_utils.py:106`, `code/tests/hmc/_gradient_test_utils.py:112`, `code/tests/hmc/_gradient_test_utils.py:126`, `code/tests/jit/_jit_test_utils.py:82`, `code/tests/jit/_jit_test_utils.py:97`.

Today, running an experiment or many tests writes `.npy`, `.json`, `.csv`, and plot files as part of normal execution paths. Test helpers append result JSON records and reset files.

For JAX, file I/O itself is a host concern and is harmless outside compiled kernels. The structural issue is that compute code, diagnostics, and persistence are not sharply separated, which makes it unclear what should be JIT-compiled and what is a host effect.

Uniform replacement pattern: split execution into pure compute functions, host orchestration, and persistence adapters. Tests should assert on returned values; optional artifact writing should be gated by explicit flags or separate diagnostic scripts.

## Abstraction Leaks and Inconsistent Interfaces

### The base filter interface does not cover the differentiable likelihood protocol

Representative sites: `code/src/core/filter_base.py:20`, `code/src/core/filter_base.py:26`, `code/src/core/filter_base.py:36`, `code/src/DF/hmc_runner.py:92`, `code/src/DF/hmc_runner.py:94`, `code/src/filters/particle/bootstrap_pf_hmc.py:205`, `code/src/filters/particle/ledh_invertible_hmc.py:362`, `code/src/filters/kalman/extended_kalman.py:190`.

Today, `Filter` defines `reset`, `predict`, `update`, and `filter`, all NumPy-facing. DPFRunner assumes filters also implement `log_marginal_likelihood_tf`, but that method is not part of the base protocol and has different implementation styles across EKF, BPF-HMC, and LEDH-HMC.

In JAX, the distinction between user-facing filtering and differentiable log-likelihood needs to be explicit. Otherwise the rebuild will repeat the current split between stateful experiment filters and special HMC filters.

Uniform replacement pattern: define separate protocols for `filter_apply` and `loglik_apply`. Both should be pure functions over pytrees and keys. A host wrapper can adapt them to result dataclasses, but sampler code should depend only on the pure log-likelihood protocol.

### Experiment dispatch depends on class names and signature inspection

Representative sites: `code/src/experiments/run_experiment.py:44`, `code/src/experiments/run_experiment.py:47`, `code/src/experiments/run_experiment.py:64`, `code/src/experiments/run_experiment.py:85`, `code/src/experiments/run_experiment.py:112`, `code/src/experiments/run_experiment.py:116`, `code/src/experiments/run_experiment.py:120`, `code/src/experiments/run_experiment.py:126`.

Today, the experiment runner branches on `cfg.filter._target_.split('.')[-1]`, checks whether the filter name contains `Kalman`, and uses `inspect.signature` to decide whether a progress callback can be passed. Different filters receive different random arguments (`random_state`, `random_seed`, or none).

This is mostly a Python orchestration issue, not a direct JAX tracing issue. It will still hurt the JAX rebuild because the public filter API is not uniform enough for a single compiled runner or benchmark harness.

Uniform replacement pattern: make filter construction and application signatures uniform. Config should select an algorithm enum and static options; all filters should expose the same apply function shape. Optional features such as progress reporting should live only in host wrappers, not in the algorithm interface.

### Resampler interfaces expose different semantics under one callable shape

Representative sites: `code/src/resampling/systematic.py:6`, `code/src/resampling/systematic.py:53`, `code/src/resampling/soft.py:5`, `code/src/resampling/soft.py:6`, `code/src/resampling/ot_entropy.py:470`, `code/src/resampling/ot_entropy.py:495`, `code/src/resampling/ot_entropy.py:636`, `code/src/utils/resampling_config.py:20`, `code/src/utils/resampling_config.py:27`.

Today, all resamplers return `ResampleResult`, but semantics differ: systematic and soft return ancestor indices, OT returns a transport matrix, OT accepts a seed only for API consistency, and soft requires an `alpha` argument with no default. Unknown resampling strings silently fall back to systematic.

In JAX, these differences matter for JIT static branches and output pytrees. A function whose output sometimes has ancestor indices and sometimes a transport matrix is awkward to scan or vmap unless the shape contract is normalized. Silent fallback also hides config errors.

Uniform replacement pattern: define a typed resampling policy with explicit modes. Return a uniform pytree with fixed fields and sentinel arrays of fixed shape, or specialize compiled filters by resampling mode as a static choice. Invalid config values should fail at config validation, not silently choose systematic.

## Config Layering

### DPF configs duplicate full model and filter blocks instead of composing shared groups

Representative sites: `code/configs/config.yaml:3`, `code/configs/config.yaml:4`, `code/configs/config.yaml:5`, `code/configs/config.yaml:7`, `code/configs/config_dpf.yaml:3`, `code/configs/config_dpf.yaml:5`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:8`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:16`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:32`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:53`.

Today, normal experiments compose `model`, `filter`, and optional `experiment` groups, while DPF configs are selected as whole global configs that inline `model`, `filter`, `dpf`, and `data`. The config inventory found 372 YAML files under `code/configs/`, with substantial scenario and sampler duplication.

For JAX, config duplication increases the chance that two runs meant to compare algorithms differ in unrelated static choices. It also makes it harder to derive a canonical static config object for JIT cache keys.

Uniform replacement pattern: separate model, filter, resampler, sampler, data, and output config groups for all run types. Compose them consistently, then validate into one typed static config dataclass before constructing JAX functions.

### User knobs and internal algorithm switches are mixed in run configs

Representative sites: `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:20`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:24`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:25`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:26`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:30`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:18`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:24`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:25`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:44`.

Today, configs expose statistical choices (`n_particles`, `epsilon`, priors), performance/debug switches (`eager_mode`), gradient surgery (`stop_gradient_resampling`), alternate HMC-only resampling settings, and sampler adaptation internals in the same layer. Some configs include comments documenting one-off diagnostic decisions.

Under JAX, these are different kinds of choices: some are model semantics, some are static compilation choices, and some are host debugging controls. Mixing them makes it hard to decide what should be static, what should be swept, and what should be excluded from compiled function signatures.

Uniform replacement pattern: use typed config sections such as `model`, `filter`, `resampling`, `sampler`, `numerics`, `debug`, and `output`. Mark JIT-static fields explicitly. Keep diagnostic-only settings in separate diagnostic configs or host flags.

### Dtype and seed policy are repeated across config levels

Representative sites: `code/configs/config.yaml:12`, `code/configs/config.yaml:13`, `code/configs/config_dpf.yaml:9`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:51`, `code/configs/dpf/hmc/linear_gaussian/ledh_ot.yaml:55`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:6`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:46`, `code/configs/dpf/hmc/stochastic_volatility_2d/ledh_ot_sigma2_smoke.yaml:51`.

Today, root configs define dtype and seed defaults, while individual DPF files repeat sampler seeds and data seeds and sometimes override dtype. The runner also has hard-coded fallback dtype and seed behavior.

For JAX, repeated seed and dtype policy will create accidental recompilation and unclear reproducibility. Seed splitting needs to be a first-class part of the computation, not a set of repeated scalar fields.

Uniform replacement pattern: centralize `numerics.dtype` and `rng.master_seed`, derive named keys in the runner, and pass keys explicitly. Use per-run overrides only through a validated hierarchy with one source of truth.

## Test Design

### Tests mix assertions, diagnostics, saved artifacts, and scripts

Representative sites: `code/tests/hmc/_gradient_test_utils.py:106`, `code/tests/hmc/_gradient_test_utils.py:130`, `code/tests/hmc/_gradient_test_utils.py:196`, `code/tests/hmc/_gradient_test_utils.py:204`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:20`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:205`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:227`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:258`, `code/tests/filters/test_kalman_family.py:510`.

Today, some tests assert numerical properties, some only save diagnostic records, some use `unittest.TestCase` under pytest, and several include script-style `__main__` runners. The map also shows uneven use of `save_result` and `reset_results` across test files.

For a JAX rebuild, this makes it hard to know which tests are gates and which are investigative notebooks in test form. JIT behavior also needs tests that distinguish compile success, numerical correctness, deterministic replay, and gradient quality.

Uniform replacement pattern: split tests into gate tests and diagnostic scripts. Gate tests should be side-effect-free by default and assert on returned arrays. Diagnostic artifact writing should require an explicit flag or separate command.

### Tests rely heavily on host conversions and fixed global constants

Representative sites: `code/tests/hmc/test_gradient_vs_numerical_lg.py:35`, `code/tests/hmc/test_gradient_vs_numerical_lg.py:39`, `code/tests/hmc/test_gradient_vs_numerical_lg.py:65`, `code/tests/hmc/test_gradient_vs_numerical_lg.py:96`, `code/tests/hmc/test_gradient_vs_numerical_lg.py:140`, `code/tests/jit/_jit_test_utils.py:22`, `code/tests/jit/_jit_test_utils.py:33`, `code/tests/jit/_jit_test_utils.py:46`, `code/tests/jit/_jit_test_utils.py:60`.

Today, tests hard-code dtype, particle counts, seeds, and finite-difference radii at module scope. JIT helpers force host synchronization via `.numpy()` to trigger compilation and timing. That is appropriate for TF/XLA probing, but it blurs compile testing, performance testing, and numerical testing.

In JAX, explicit synchronization is still needed for timing, but it should be isolated. Module-level constants are fine for fixed tests, but broader JAX validation will need parametrized dtype, device, precision, and PRNG modes.

Uniform replacement pattern: build small pure fixtures returning `(config, params, data, key)`, then write separate tests for shape/dtype, deterministic replay, compile under `jit`, gradient checks, and benchmark timing. Keep `block_until_ready()` timing helpers out of correctness assertions.

### Test import paths are adjusted inside test modules

Representative sites: `code/tests/hmc/test_gradient_vs_numerical_lg.py:11`, `code/tests/hmc/test_gradient_vs_numerical_lg.py:15`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:26`, `code/tests/hmc/sv2d_diagnostics/test_runner_vs_direct.py:31`.

Today, some tests modify `sys.path` locally to make `src` imports resolve. This is an execution-environment smell rather than a numerical problem.

For JAX, this is harmless to `jit` directly, but it will complicate a rebuild if tests are expected to run from multiple working directories or as a packaged module. It can also hide import-order differences between tests and production scripts.

Uniform replacement pattern: define one package/import layout and run tests from that environment. Put path setup in test runner configuration, not inside individual test modules.

## Model and Data Boundaries

### Models combine simulation, deterministic filtering methods, and log-probability methods

Representative sites: `code/src/core/model_base.py:30`, `code/src/core/model_base.py:73`, `code/src/core/model_base.py:110`, `code/src/core/model_base.py:142`, `code/src/core/model_base.py:160`, `code/src/core/model_base.py:179`, `code/src/models/linear_gaussian.py:163`, `code/src/models/linear_gaussian.py:193`, `code/src/models/linear_gaussian.py:209`.

Today, one model object provides sampling, transition means, covariances, Jacobians, observation functions, batch defaults, and parameter-dependent properties. Some models override batch behavior; the base class falls back to Python loops or `tf.map_fn`.

For JAX, this is too much object surface for a clean pytree model. Sampling needs keys, deterministic functions need pure arrays, and Jacobians can often come from autodiff rather than hand-coded methods. A large object protocol also makes `vmap` over models or particles harder.

Uniform replacement pattern: define a model as a collection of pure functions and static metadata: `transition(params, x, key, t)`, `observation(params, x, key, t)`, `log_obs(params, y, x, t)`, and optional analytic linearizations. Batch behavior should come from `vmap`, with hand-specialized versions only where necessary.

### Default batch methods are Python-loop based

Representative sites: `code/src/core/model_base.py:160`, `code/src/core/model_base.py:162`, `code/src/core/model_base.py:163`, `code/src/core/model_base.py:165`, `code/src/core/model_base.py:167`, `code/src/core/model_base.py:169`, `code/src/core/model_base.py:171`, `code/src/core/model_base.py:173`, `code/src/core/model_base.py:179`, `code/src/core/model_base.py:181`.

Today, the base model provides convenience batch methods using Python list comprehensions and static `particles.shape[0]`. Some concrete models likely override these, but the abstract fallback is not JIT-friendly.

In JAX, particle batching should be expressed with `vmap`, not Python loops over particles. Python loops over traced particle counts will fail or unroll, and static shape dependence will make recompilation sensitive to particle count.

Uniform replacement pattern: remove Python-loop batch fallbacks from compiled pathways. Use `vmap` as the default batching rule and specialize only when a model needs a custom vectorized implementation.

## Open Questions for the User

- Should the JAX rebuild prioritize differentiable particle-filter likelihoods for HMC/MAP, non-differentiable but fast particle filtering, or both as separate algorithm families?
- Should conditional resampling be preserved for non-gradient PF runs, or should all JAX gradient-facing filters use an always-resample or smooth-resampling policy?
- Is `float64` intended to be the default numerical mode for all JAX runs, or only for sensitive SV2D/OT-gradient configurations?
- Which saved test artifacts are required deliverables, and which are historical diagnostics that can move out of pytest?
- Should the DPF config family remain as whole-run scenario configs, or should it be normalized to compose the same `model`, `filter`, `resampling`, and `sampler` groups as regular experiments?
- Are hand-coded Jacobians part of the desired JAX design, or should the rebuild rely primarily on `jax.jacfwd`/`jax.jacrev` with optional analytic overrides?
