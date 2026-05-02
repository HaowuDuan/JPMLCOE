# Phase 08: Migration, Legacy Quarantine, and Final Gate Suite

## 1. Goal

Move the old TensorFlow code tree under `legacy/`, make the new `jpml_tf/` tree from Phases 01 through 07 the only active implementation root, migrate existing YAML scenarios into the new Pydantic scenario schema, port the relevant HMC and filter tests onto the uniform JSON result sink, and gate the final state by requiring all earlier phase gates to pass with no imports from `legacy/` into `jpml_tf/`.

## 2. What Gets Built

Files and modules owned by this phase:

- `legacy/` - the existing `code/` tree moved verbatim to a read-only historical location. No new implementation work lands here.
- `jpml_tf/scenario/migrate.py` - one-shot scenario migration helpers from old Hydra-style YAMLs to the Phase 01 whole-run schema.
- `jpml_tf/scenario/aliases.py` - target-string-to-registry-key alias maps for models, filters, samplers, and resamplers.
- `jpml_tf/testing/audit.py` - import-audit helpers and result-sink usage audits for migrated tests.
- `scripts/migrate_legacy_configs.py` - host-side entry point that migrates one file or a whole config subtree and writes new scenario YAML or JSON.
- `scripts/move_code_to_legacy.py` - host-side repo layout migration script or one-time command wrapper that moves `code/` to `legacy/` and verifies the new root layout.
- `tests/test_08_migration.py` - the single gate test file for this phase.
- Ported numerical tests under the new tree, especially the prior `tests/hmc/` and `tests/filters/` families, updated so every numerical case uses the Phase 01 result sink and external-artifact rules.

Migration result containers and host-side utilities:

```python
# jpml_tf/scenario/migrate.py
class MigrationRecord(BaseModel):
    source_path: str
    output_path: str
    scenario_digest: str
    warnings: list[str] = []
    dropped_fields: list[str] = []

class MigrationSummary(BaseModel):
    migrated: list[MigrationRecord]
    failed: list[str]

def migrate_legacy_scenario(
    source_path: Path,
    *,
    output_root: Path,
) -> MigrationRecord: ...

def migrate_legacy_tree(
    source_root: Path,
    *,
    output_root: Path,
) -> MigrationSummary: ...

# jpml_tf/testing/audit.py
class ImportAuditResult(NamedTuple):
    legacy_imports: list[str]
    forbidden_paths: list[str]

class ResultSinkAuditResult(NamedTuple):
    tests_missing_sink: list[str]
    tests_missing_save_call: list[str]

def audit_no_legacy_imports(package_root: Path) -> ImportAuditResult: ...
def audit_result_sink_usage(test_root: Path) -> ResultSinkAuditResult: ...
```

The migration script applies explicit mapping rules from the old config layout to the new scenario schema:

1. `model._target_` becomes `model.family` plus `model.algorithm` through `aliases.py`. Every migrated parameter becomes a `ParameterSpec` entry with `initial_value` taken from the old model block and an explicit `trainable` flag. Only opaque non-parameter structural values move into `model_constants`.
2. `filter._target_` becomes `filter.family` plus `filter.algorithm`. Structural fields such as `n_particles`, `n_lambda_steps`, and filter-specific params are copied into the new filter section. `resampling_method`, `resampling_config`, `resample_threshold`, `always_resample`, and `stop_gradient_resampling` collapse into the Phase 04 resampling-policy object.
3. `dpf.sampler` plus the matching `dpf.hmc`, `dpf.map`, or `dpf.mh` block becomes the new sampler section. Concrete strings like `custom_hmc` remain algorithm data, not new interface enums.
4. `dpf.trainable_params` is eliminated as a separate top-level mechanism. Its contents are merged into the explicit parameter schema under `model.parameters[*].trainable`, and any parameter absent from that legacy block becomes an explicit `trainable=False` entry rather than falling into a separate parameter bucket.
5. `data.T`, `data.seed`, and `data.true_params` move directly into the new data section. Any old-derived or generated arrays still save as external artifacts under the result sink.
6. Top-level Hydra fields such as `defaults`, `hydra`, and `tf_log_level`, plus runtime-only fields such as `eager_mode`, are dropped from the scenario payload or mapped to host startup defaults. They do not appear in the new compiled-path schema.
7. Any old config that requests `float32` is rewritten to global `float64` with a migration warning. Per-experiment dtype toggling is not preserved.

Representative alias examples:

```text
src.models.linear_gaussian.LinearGaussianModel
  -> family="linear_gaussian", algorithm="default"

src.filters.particle.bootstrap_pf_hmc.BootstrapPFHMC
  -> family="particle", algorithm="bootstrap_particle"

src.filters.particle.ledh_invertible.LEDHParticleFlowFilter
  -> family="flow_particle", algorithm="ledh_ot" or "ledh_default" depending on migrated resampling config

dpf.sampler: hmc
  -> family="mcmc", algorithm="tfp_hmc"

dpf.sampler: custom_hmc
  -> family="mcmc", algorithm="tfp_hmc_clipped" or other explicitly registered experimental key

dpf.sampler: map
  -> family="optimizer", algorithm="adam_map"
```

This phase does not create compatibility imports from `legacy/` into `jpml_tf/`. The whole point of the move is to quarantine the old tree, not to make the new implementation depend on it.

## 3. What Gets Tested and Acceptance Criteria

Gate test file: `tests/test_08_migration.py`

- `test_move_code_tree_to_legacy_and_activate_jpml_tf_root(result_sink)` asserts the old implementation tree exists under `legacy/`, asserts the new package root is `jpml_tf/`, and asserts there is no active implementation import path still rooted in the old `code/` tree.
- `test_migrate_representative_hmc_and_map_configs_to_new_schema(result_sink)` migrates at least one old HMC config and one old MAP config such as `legacy/configs/dpf_smoke/hmc/linear_gaussian/bpf_sys.yaml` and `legacy/configs/dpf_smoke/map/linear_gaussian/bpf_ot.yaml`, then asserts the resulting scenario validates through the Phase 01 schema and preserves the intended model/filter/sampler/resampling choices.
- `test_bulk_config_migration_validates_generated_scenarios(result_sink)` runs the one-shot migration script over representative legacy config subtrees, asserts every generated scenario validates, and records warnings for dropped or normalized fields such as `float32` or Hydra-only runtime settings.
- `test_ported_filter_and_hmc_tests_use_uniform_result_sink(result_sink)` audits the migrated numerical tests under the new root and asserts every relevant test case accepts or uses the Phase 01 result-sink fixture rather than ad hoc JSON, text, or plot writes.
- `test_no_jpml_tf_imports_from_legacy_modules(result_sink)` runs `audit_no_legacy_imports` over `jpml_tf/` and the migrated test tree, then asserts zero imports from `legacy/` and zero forbidden path references in active implementation modules.
- `test_all_phase_01_to_07_gate_tests_pass_after_migration(result_sink)` executes the gate suite from Phases 01 through 07 under the migrated repo layout and asserts zero failures. The migration phase is not complete until the prior hard gates still pass end to end.

Gate-pass condition:

- `pytest tests/test_08_migration.py` passes.
- Every test saves a JSON result through the Phase 01 `result_sink` fixture.
- `legacy/` exists and contains the old implementation tree.
- `jpml_tf/` is the only active implementation root.
- Migrated scenarios validate through the new whole-run schema without special-case loaders.
- The migrated HMC and filter tests use the uniform JSON result sink and external-artifact rules.
- There are no imports from `legacy/` into `jpml_tf/`.
- All gate tests from Phases 01 through 07 still pass after migration.

No compatibility-shim alternative is accepted for this phase. If the new code still imports the old tree, the rebuild has not actually retired the old architecture.

## 4. What the Reader Needs to Understand

### Key Concepts

Migration is a host-side tree and data rewrite, not a new numerical abstraction. The numerical work is already complete by the end of Phase 07. Phase 08 makes that work the actual repo default and seals off the previous implementation so the codebase no longer has two competing centers of gravity.

The old Hydra-style configs mixed runtime knobs, trainable-parameter declarations, model constants, sampler settings, and framework workarounds. The new schema separates those concerns explicitly. The migration script is therefore part data transformation and part policy enforcement: it removes structure the rebuild intentionally does not keep.

Tests are part of the migration surface. Moving configs without moving the corresponding HMC and filter tests would preserve hidden assumptions in the old tree. The rebuild only finishes when the active tests run against `jpml_tf/`, write through the new result sink, and stop reaching into legacy helpers.

### Invariants Established

- The old implementation tree is preserved under `legacy/` as reference only.
- The active implementation root is `jpml_tf/`.
- New scenario files use the Phase 01 whole-run schema and no longer depend on Hydra composition.
- Trainable versus frozen parameters are encoded in explicit parameter schema entries, not separate `dpf.trainable_params` plumbing at runtime.
- Ported numerical tests save results through the uniform sink and external-artifact discipline.
- No active implementation imports `legacy/`.
- The earlier gate suite remains green after the repo move.

### Tricky Bits and Rationale

Target-string aliasing must be explicit. Old `_target_` strings encode both family and algorithm, sometimes with HMC-specific or resampling-specific variants embedded in class names. `aliases.py` keeps that mapping centralized so the migration script does not grow string heuristics in multiple places.

The old configs sometimes duplicate resampling choices in multiple places, for example `resampling_method`, `hmc_resampling_method`, and `stop_gradient_resampling`. The migration step collapses those onto one policy object per active objective path. If a config encoded genuinely different forward and gradient resampling behavior, that difference must become explicit in the migrated scenario rather than hiding in parallel fields.

Dropping `float32` is intentional. Some legacy MAP configs defaulted to `float32`, but the rebuilt plan locked global `float64` for critical paths. The migration must warn and normalize rather than preserve a configuration freedom the new architecture explicitly forbids.

Ported-test audits should be structural, not only manual. AST or simple source-text audits are acceptable here because the question is whether tests call `save_result` and use the shared fixture, not whether one particular numerical metric looks good. The numerical metrics are already covered by the earlier gate files.

### Alternatives Considered

Keeping the old code in place and adding thin forwarding wrappers is rejected because it leaves the import graph and ownership boundaries ambiguous.

Preserving Hydra defaults and runtime-only fields in the new schema is rejected because it would reintroduce the configuration layering the rebuild is meant to remove.

Porting configs first and tests later is rejected because it would leave the new tree without its corresponding regression surface during the most fragile repo-layout change.

Allowing active imports from `legacy/` is rejected because that turns the old tree into a hidden dependency instead of a historical reference.

### Locked Design Decisions Realized

- The new code root is `jpml_tf/`, not the old `code/` tree.
- The old implementation is quarantined under `legacy/` instead of partially deleted or partially kept alive.
- Whole-run scenario configs remain the only active on-disk configuration format.
- The uniform JSON result sink from Phase 01 becomes mandatory for the migrated HMC and filter tests as well.
- The hard gates from Phases 01 through 07 remain authoritative after migration. Broad repo cleanup is not allowed to weaken them.

### JIT Boundary Decisions

Migration scripts, tree moves, import audits, scenario rewriting, and test-result audits are entirely host-side. They do not sit inside `tf.function`, and they do not introduce new compiled kernels.

The only compiled code exercised in this phase is the already-established numerical code from prior phases when `tests/test_08_migration.py` reruns the earlier gate suite. Phase 08 adds no new compiled numerical boundary and no new permission for file I/O inside kernels.

## 5. Dependencies

This phase depends on:

- Phase 01 for the scenario schema, YAML/JSON validation, and uniform result sink.
- Phase 02 for explicit parameter-schema structure, which the migration script targets when translating old `trainable_params`.
- Phase 03 through Phase 07 for the active `jpml_tf/` implementation and the gate tests that must remain green after the move.

Later dependencies:

- None inside the fixed rebuild plan. This is the terminal phase. Any future research feature work should start from the migrated `jpml_tf/` tree and treat `legacy/` as reference-only.
