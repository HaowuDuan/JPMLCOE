# Phase 01 — Foundation, Config, Results Sink

**Objective**: scaffold the package, establish the startup discipline (TF-visible-devices, dtype policy), define the Pydantic-validated scenario schema, and implement the uniform JSON result sink. All subsequent phases depend on this.

## Deliverable files

```
code_new/
  jpml_tf/
    __init__.py                  # NO side effects; docstring only
    startup.py                   # configure global float64 policy, limit_gpu_memory_growth()
    types.py                     # FilterResult, SampleResult, etc. (NamedTuples/dataclasses)
    scenario/
      __init__.py
      schema.py                  # Pydantic v2 schema: Scenario, ModelSpec, FilterSpec, SamplerSpec
      loader.py                  # YAML + JSON loaders both -> Scenario
    results/
      __init__.py
      sink.py                    # save_result, reset_results; append JSON with stable keys
      schema.py                  # TestCaseResult pydantic model
  tests/
    conftest.py                  # shared fixtures
    test_01_scaffold.py          # scenario parse; result sink round-trip
    fixtures/
      scenario_hard.yaml         # SV2D + LEDH + 29 steps + multiplicative noise
      scenario_minimal.yaml      # LG + BPF + 1 step
```

## Startup discipline

`jpml_tf/startup.py` exposes:

```python
def configure_tf(
    *,
    force_cpu: bool = False,
    allow_gpu_growth: bool = True,
    deterministic_ops: bool = False,
    legacy_keras: bool = True,
) -> None:
    """Call once at the top of every entry point BEFORE importing any jpml_tf submodule."""
```

This replaces the scattered `os.environ` calls in current `tests/*.py`. All entry points (pytest, scripts, bash) call `configure_tf()` first.

`configure_tf()` is also where the rebuild locks the global numeric mode to `float64` before any computational module import. There is no scenario-level dtype switch in the compiled path.

## Scenario schema (Pydantic v2)

`scenario/schema.py` defines:

```python
class ParameterSpec(BaseModel):
    name: str
    constraint: Literal["positive", "bounded", "unconstrained"]
    bounds: tuple[float, float] | None = None
    trainable: bool
    prior: PriorSpec
    initial_value: float

class ModelSpec(BaseModel):
    family: Literal["linear_gaussian", "stochastic_volatility_1d",
                    "stochastic_volatility_2d", "range_bearing", "custom"]
    algorithm: str | None = None        # registry key; defaults to `family`
    parameters: list[ParameterSpec]
    model_constants: dict[str, Any] = {}  # opaque non-parameter config only
    noise_structure: Literal["additive", "multiplicative"] = "additive"

class FilterSpec(BaseModel):
    family: Literal["kalman", "particle", "flow_particle"]
    algorithm: str                       # registry key: "bootstrap", "ledh_ot", ...
    n_particles: int | None = None
    n_lambda_steps: int | None = None
    resampling: "ResamplingSpec"
    # ... other fields with defaults

class SamplerSpec(BaseModel):
    family: Literal["hmc", "nuts", "map"]
    algorithm: str
    # ... common fields

class Scenario(BaseModel):
    model: ModelSpec
    filter: FilterSpec
    sampler: SamplerSpec
    data: DataSpec
    diagnostics: DiagnosticsSpec
```

**Key design**: `family` is a closed enum (stable interface); `algorithm` is an open string (registry key). Adding a new algorithm does not change the Scenario schema. Parameter trainability is explicit on each `ParameterSpec`; it is never inferred from which YAML container a parameter happened to land in.

## Result sink contract

Every test writes to `tests/results/<test_file_stem>.json` with:

```python
class TestCaseResult(BaseModel):
    case_name: str
    schema_version: int = 1
    timestamp: datetime
    scenario_digest: str                 # SHA256 of the scenario YAML
    metrics: dict[str, float]            # scalars only
    artifacts: dict[str, str]            # relative paths to .npy/.npz/.png
    pass_fail: Literal["pass", "fail", "characterization"] | None
    notes: str | None
```

`save_result(test_file, case)` appends; `reset_results(test_file)` wipes for a session. Artifacts go in `tests/results/artifacts/<test_file_stem>/<case_name>/`.

**No inline numpy arrays in JSON.** Anything >20 floats goes as an external file reference.

## Gate tests for this phase

`tests/test_01_scaffold.py`:

1. `test_startup_configures_tf_once`: calling `configure_tf(force_cpu=True)` twice is idempotent; `tf.config.list_physical_devices("GPU")` returns `[]` after.
2. `test_scenario_hard_fixture_parses`: `Scenario.parse_yaml("scenario_hard.yaml")` succeeds; `scenario.filter.algorithm == "ledh_ot"`; `scenario.model.noise_structure == "multiplicative"`; `scenario.model.parameters[0].constraint == "positive"`; `scenario.model.parameters[0].trainable is True`.
3. `test_scenario_json_yaml_equivalent`: parsing same scenario from YAML and from JSON produces bit-identical Pydantic models.
4. `test_result_sink_append`: three `save_result` calls append to the same file; schema validates each time.
5. `test_result_sink_artifact_path`: artifact referenced by relative path exists and is under `tests/results/artifacts/`.

## Pass criteria

All 5 gate tests pass. No `tf.function` retracing warnings from startup. Scenario YAML parses without special-casing. Result sink never writes more than 256KB to JSON per test (arrays go external).

## Risks

- Pydantic v2 syntax vs v1 (user may have v1 installed). Mitigation: pin v2 in `requirements.txt`.
- `tf.config.set_visible_devices([], 'GPU')` must be called BEFORE any TF op. If another module imports TF first, test fails. Mitigation: `configure_tf` is the ONLY place TF is imported in `__init__`.

## Estimated effort

2 days. Most of the time is Pydantic schema design + hard-fixture YAML crafting.
