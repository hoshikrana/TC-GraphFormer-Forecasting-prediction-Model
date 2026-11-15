# Phase 2E.4.4 — Modality Enforcement, Training Metric Integrity & Final Verification Report

**Repository:** `https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Working Branch:** `fix/phase2e4-4-modality-metrics-closure`  
**Starting HEAD:** `0cc51cfbcc50d633b5c5572b361239c13e8593ee`  
**Implementation Commit:** `87dc7a071747896f48db91914a30a4b10b8d98f4`  
**Verification Date:** November 16, 2026  
**Execution Environment:** Windows 11 AMD64, Python 3.12.9 (`.venv`)  
**Strict Hardware Profile:** CPU execution, 0 live network calls, 0 unapproved data acquisitions.

---

## Executive Summary

Phase 2E.4.4 completes the scientific integrity, metric aggregation, and operator workflow closure for the TC-GraphFormer project. Building directly on the audited infrastructure readiness of Phase 2E.4.3, this phase addresses four critical production requirements:

1. **Acquisition Modality Requirements Enforcement & Radar Overpass Prioritization:**
   - Enforced top-level `modality_requirements` in S3 catalogue planning via `filter_eligible_catalogue()`.
   - Storms lacking environmental data (`env_count < 1`) or failing to satisfy minimum remote-sensing instrument thresholds (`min_sensors_per_storm`) are strictly excluded prior to stratified allocation.
   - Implemented deterministic radar overpass preference (`prioritize_radar_overpasses: true`) using supported TC PRIMED radar instruments (`DPR`, `PR`, `KuKaGMI`, `KuGMI`, `KuTMI`), favoring radar-bearing storms during cell-level and global allocation without requiring radar as a hard exclusion.
   - Comprehensive `modality_audit` diagnostics record original vs. eligible catalogue counts, exclusion reasons, coverage percentages, and radar statistics.
   - Updated planning reports to distinguish planning-enforced metadata constraints from post-processing validation criteria (forecasting horizons, RI labels, valid pixel masks).

2. **MultiTaskLoss Training Metric Aggregation Mathematical Rigor:**
   - Corrected epoch training metric reporting in `TCGraphFormerTrainer` and `MultiTaskLoss`: accumulated unreduced task loss sums and valid target counts across all microbatches.
   - Defined epoch per-task loss as:
     $$\mathcal{L}_{t, \text{epoch}} = \frac{\sum_{b} \sum_{i \in b} \ell_{t, i} \cdot M_{t, i}}{\max\left(1, \sum_{b} \sum_{i \in b} M_{t, i}\right)}$$
   - Defined total epoch loss as the exact weighted combination:
     $$\mathcal{L}_{\text{total, epoch}} = \sum_{t} w_t \mathcal{L}_{t, \text{epoch}}$$
   - Eliminated the metric distortion caused by multiplying microbatch losses by batch size $B$ after group-target normalization.
   - Preserved sample-weighted gradient accumulation across microbatch groups and residual optimizer steps.
   - Separated the mathematical normalization equivalence from empirical linear-model parameter drift, validating equivalence against a combined single-batch reference within $< 10^{-4}$ on `float32`.

3. **Production Configuration Precedence Resolvers & Boolean Negation Flags:**
   - Extracted `resolve_evaluation_config(args, cfg=None)` in [evaluate_baselines.py](file:///d:/Project/Capstone_project/src/evaluation/evaluate_baselines.py) and `resolve_training_config(args, cfg=None)` in [train_multitask.py](file:///d:/Project/Capstone_project/src/training/train_multitask.py).
   - Fixed the `num_workers: 0` falsy bug where `tr_cfg.get("num_workers") or mach_cfg.get("num_workers")` caused `0` to evaluate as falsy and fall back to `mach_cfg` (e.g., 8 workers). Explicit check `if tr_workers is not None` now strictly preserves worker count `0`.
   - Added paired boolean CLI flags (`--use-amp` / `--no-amp`, `--pin-memory` / `--no-pin-memory`, `--persistent-workers` / `--no-persistent-workers`, `--allow-legacy-resume` / `--no-allow-legacy-resume`, `--use-sequences` / `--single-obs`, `--strict-checkpoint` / `--no-strict-checkpoint`, `--smoke-test` / `--no-smoke-test`).
   - Covered production resolvers with direct unit tests verifying CLI explicit > YAML config > fallback default precedence.

4. **Single-Year Concentration Cap Semantics & Underfilled Budget Ceiling:**
   - Clarified that the single-year cap ceiling is defined relative to the requested target budget:
     $$\text{max\_single\_year\_budget\_bytes} = \text{target\_bytes} \times \left(\frac{\text{single\_year\_max\_pct}}{100.0}\right)$$
   - When an acquisition plan yields an underfilled dataset (e.g., 40 GB selected of 100 GB budget due to limited qualifying storms), a single year contributing 20 GB represents 50% of the realized dataset but only 20% of the requested target budget. Both `largest_single_year_budget_pct` and `largest_single_year_selected_pct` are reported, and cap violations are only flagged if the target budget ceiling is exceeded.

---

## 1. Modality Requirements Enforcement & Radar Overpass Preference

### 1.1 Ingestion and Filtering Path
In [prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py), `cmd_plan()` now reads the top-level `modality_requirements` dictionary from YAML and forwards it directly to `stratified_select()`. Before stratified allocation begins, candidate storms are processed through `filter_eligible_catalogue()`:

```python
def filter_eligible_catalogue(catalogue, modality_requirements=None):
    reqs = modality_requirements or {}
    req_env = bool(reqs.get("require_environmental", False))
    min_sensors = int(reqs.get("min_sensors_per_storm", 0)) if reqs.get("min_sensors_per_storm") is not None else 0
    prioritize_radar = bool(reqs.get("prioritize_radar_overpasses", False))
    ...
```

The filter applies two strict eligibility criteria:
1. `require_environmental: true`: Excludes any storm with `env_count < 1`.
2. `min_sensors_per_storm: N`: Excludes any storm with fewer than $N$ distinct remote-sensing instrument identifiers (excluding reanalysis/environmental keys like `ERA5`).

### 1.2 Deterministic Radar Overpass Prioritization
To ensure radar data presence without restricting the dataset exclusively to rare radar overpasses, `RADAR_INSTRUMENTS = {"DPR", "PR", "KUKAGMI", "KUGMI", "KUTMI"}` was defined. When `prioritize_radar_overpasses: true`:
- In **Pass 1 (Cell Quotas)**: Candidate storms within each Basin $\times$ Era cell are sorted by `(not has_radar, -bytes, tie_breaker)`. Radar storms are prioritized for initial cell allocation.
- In **Pass 2 (Basin/Era Balancing)** and **Pass 3 (Budget Filling)**: Radar storms receive a $+1000.0$ score preference in candidate ranking, ensuring that whenever a radar-bearing storm fits within remaining cell, season, and budget allowances, it is preferred over a non-radar storm.

### 1.3 Modality Audit Metadata & Unfilled Budget Attribution
The selection report returned by `stratified_select()` embeds a structured `modality_audit` dictionary:
```json
"modality_audit": {
  "modality_requirements_applied": {
    "require_environmental": true,
    "min_sensors_per_storm": 2,
    "prioritize_radar_overpasses": true
  },
  "original_catalogue_storms": 2599,
  "original_catalogue_bytes": 21873491200,
  "eligible_catalogue_storms": 1840,
  "eligible_catalogue_bytes": 16421832000,
  "excluded_storms_count": 759,
  "exclusion_reasons": {
    "missing_environmental": 420,
    "insufficient_sensors (found 1 < required 2)": 339
  },
  "environmental_coverage_pct": 83.84,
  "multi_sensor_coverage_pct": 70.80,
  "original_radar_storms_count": 88,
  "eligible_radar_storms_count": 74,
  "radar_priority_enabled": true
}
```

When candidate exclusions cause the realized dataset to underfill the target budget, `unfilled_budget_reason` transparently attributes the shortfall:
```
Unfilled budget of 0.03 GB: modality requirements excluded 3 storms (missing_environmental: 3), reducing available catalogue from 0.04 GB to 0.01 GB.
```

---

## 2. MultiTaskLoss Training Metric Aggregation Mathematical Rigor

### 2.1 Problem Analysis of Microbatch Metric Averaging
In `TCGraphFormerTrainer.train_epoch()`, sample-weighted gradient accumulation is utilized across microbatches. For microbatch $b$ within an accumulation group $G$:
$$\mathcal{L}_{t, b} = \frac{\sum_{i \in b} \ell_{t, i} \cdot M_{t, i}}{N_{t, G}}$$
where $N_{t, G} = \sum_{b \in G} \sum_{i \in b} M_{t, i}$.

Summing $\mathcal{L}_{t, b}$ across the group yields the exact gradient loss for the accumulation group:
$$\sum_{b \in G} \mathcal{L}_{t, b} = \frac{\sum_{i \in G} \ell_{t, i} \cdot M_{t, i}}{N_{t, G}}$$

However, the historical metric tracking code multiplied this microbatch loss by microbatch sample count $B_b$ and divided by total samples $\sum B_b$:
$$\mathcal{L}_{t, \text{reported}} = \frac{\sum_{b} B_b \cdot \mathcal{L}_{t, b}}{\sum_{b} B_b}$$

Because $N_{t, G}$ depends on the accumulation group structure and varies across tasks, multiplying by $B_b$ and dividing by sample count introduced a mathematical distortion. It distorted reported task losses whenever target validity varied across microbatches.

### 2.2 Mathematically Exact Epoch Metric Normalization
To restore absolute mathematical integrity, [losses.py](file:///d:/Project/Capstone_project/src/training/losses.py) and [trainer.py](file:///d:/Project/Capstone_project/src/training/trainer.py) were corrected to accumulate unreduced task loss sums and valid target counts:

In `MultiTaskLoss.forward()`:
- `unreduced_loss_<task>`: $\sum_{i \in b} \ell_{t, i} \cdot M_{t, i}$ (unreduced element-wise loss sum)
- `valid_<task>_count`: $\sum_{i \in b} M_{t, i}$ (unreduced valid target count)

In `TCGraphFormerTrainer.train_epoch()`:
```python
task_loss_sums[task] += loss_dict[f"unreduced_loss_{task}"]
task_valid_counts[task] += loss_dict[f"valid_{task}_count"]
```

At epoch conclusion, task epoch losses are computed directly:
$$\mathcal{L}_{t, \text{epoch}} = \begin{cases} \frac{\text{task\_loss\_sums}[t]}{\text{task\_valid\_counts}[t]} & \text{if } \text{task\_valid\_counts}[t] > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

$$\mathcal{L}_{\text{total, epoch}} = \sum_{t} w_t \mathcal{L}_{t, \text{epoch}}$$

### 2.3 Distinction Between Loss Normalization and Parameter Update Equivalence
It is critical to distinguish two mathematical concepts:
1. **Mathematical Loss Normalization Equivalence:**
   For any sequence of microbatches evaluated on identical model weights, accumulating unreduced numerators and denominators computes the mathematically identical loss as concatenating all samples into one combined batch:
   $$\frac{\sum_{b} \text{unreduced\_loss}_b}{\sum_{b} \text{valid\_count}_b} \equiv \frac{\text{unreduced\_loss}_{\text{combined}}}{\text{valid\_count}_{\text{combined}}}$$
   In `tests/test_phase2e4_4_closure.py`, this is validated with assertion threshold $< 10^{-4}$ on `float32`.
2. **Empirical Parameter Update Equivalence:**
   During actual training where the optimizer updates parameters at the end of each accumulation group, later microbatches encounter updated model weights. If evaluated with non-zero learning rates, subsequent microbatches produce slightly different predictions than in a static reference pass. Setting `lr=0.0` in unit tests isolates the metric aggregation logic from parameter drift, proving the aggregation arithmetic is exact.

---

## 3. Production Configuration Precedence Resolvers

### 3.1 Architecture of Production Resolvers
Configuration precedence resolution was decoupled from script entry points into testable production functions:
- `resolve_evaluation_config(args, cfg=None)` in [evaluate_baselines.py](file:///d:/Project/Capstone_project/src/evaluation/evaluate_baselines.py)
- `resolve_training_config(args, cfg=None)` in [train_multitask.py](file:///d:/Project/Capstone_project/src/training/train_multitask.py)

Both functions adhere strictly to the priority chain:
$$\text{CLI explicit argument} > \text{YAML configuration} > \text{Fallback default}$$

### 3.2 Correction of the `num_workers: 0` Falsy Bug
In `train_multitask.py`, the previous implementation contained:
```python
# BUGGY HISTORICAL CODE:
resolved_workers = resolve_val(args.num_workers, tr_cfg.get("num_workers") or mach_cfg.get("num_workers"), 0)
```
In Python, `0 or 8` evaluates to `8` because `0` is falsy. Consequently, configuring `training: num_workers: 0` was overridden by `machine: num_workers: 8`.

The corrected production code resolves worker counts explicitly checking for `None`:
```python
# CORRECTED PRODUCTION CODE:
tr_workers = tr_cfg.get("num_workers")
yaml_workers = tr_workers if tr_workers is not None else mach_cfg.get("num_workers")
resolved_workers = resolve_val(getattr(args, "num_workers", None), yaml_workers, 0)
```
When `tr_cfg["num_workers"]` is `0`, `yaml_workers` evaluates to `0`, correctly preserving single-process DataLoader execution.

### 3.3 Paired Boolean Negation Flags
To allow operators full CLI control over YAML defaults, the following paired flags were added using argparse destination targeting:

| Setting | Enable Flag | Disable / Invert Flag | Destination |
| :--- | :--- | :--- | :--- |
| Sequence mode | `--use-sequences` | `--single-obs` | `args.use_sequences` |
| Mixed precision | `--use-amp` | `--no-amp` | `args.use_amp` |
| Pinned memory | `--pin-memory` | `--no-pin-memory` | `args.pin_memory` |
| Persistent workers | `--persistent-workers` | `--no-persistent-workers` | `args.persistent_workers` |
| Legacy resume | `--allow-legacy-resume` | `--no-allow-legacy-resume` | `args.allow_legacy_resume` |
| Strict checkpoint | `--strict-checkpoint` | `--no-strict-checkpoint` | `args.strict_checkpoint` |
| Smoke test mode | `--smoke-test` | `--no-smoke-test` | `args.smoke_test` |

---

## 4. Single-Year Concentration Cap Semantics

### 4.1 Budget Ceiling Basis
The acquisition single-year concentration cap prevents historical overfitting by ensuring no single storm season dominates the training dataset.

The cap ceiling basis is defined as:
$$\text{max\_single\_year\_budget\_bytes} = \text{target\_bytes} \times \left(\frac{\text{single\_year\_max\_pct}}{100.0}\right)$$

### 4.2 Underfilled Realized Datasets
When a dataset is underfilled (e.g., 40 GB selected out of 100 GB target budget), reporting only the realized dataset percentage produces confusing diagnostics. For example, 20 GB from year 2021 is:
- $20\text{ GB} / 100\text{ GB} = 20.0\%$ of the requested target budget (within the 20% budget ceiling).
- $20\text{ GB} / 40\text{ GB} = 50.0\%$ of the realized underfilled dataset.

To eliminate operator ambiguity:
1. Both metrics are reported:
   - `largest_single_year_budget_pct`: Single year bytes divided by requested target budget.
   - `largest_single_year_selected_pct`: Single year bytes divided by realized selected bytes.
2. The selection report records `single_year_cap_basis = "target_budget"`.
3. A limitation violation is only raised if `largest_single_year_budget_pct > single_year_max_pct`. If the realized percentage exceeds the cap due to an underfilled budget, a non-violating advisory note is recorded.

---

## 5. Verification Suite Results

### 5.1 Dedicated Phase 2E.4.4 Test Suite
The dedicated closure test suite [tests/test_phase2e4_4_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_closure.py) was executed:

```
test_filter_eligible_catalogue_environmental (tests.test_phase2e4_4_closure.TestModalityRequirements.test_filter_eligible_catalogue_environmental) ... ok
test_filter_eligible_catalogue_min_sensors (tests.test_phase2e4_4_closure.TestModalityRequirements.test_filter_eligible_catalogue_min_sensors) ... ok
test_filter_eligible_catalogue_modality_audit_metadata (tests.test_phase2e4_4_closure.TestModalityRequirements.test_filter_eligible_catalogue_modality_audit_metadata) ... ok
test_stratified_select_radar_priority (tests.test_phase2e4_4_closure.TestModalityRequirements.test_stratified_select_radar_priority) ... ok
test_underfilled_budget_modality_exclusion_reason (tests.test_phase2e4_4_closure.TestModalityRequirements.test_underfilled_budget_modality_exclusion_reason) ... ok
test_evaluation_resolver_cli_precedence (tests.test_phase2e4_4_closure.TestProductionConfigResolvers.test_evaluation_resolver_cli_precedence) ... ok
test_evaluation_resolver_yaml_fallback (tests.test_phase2e4_4_closure.TestProductionConfigResolvers.test_evaluation_resolver_yaml_fallback) ... ok
test_training_resolver_negative_cli_flags (tests.test_phase2e4_4_closure.TestProductionConfigResolvers.test_training_resolver_negative_cli_flags) ... ok
test_training_resolver_num_workers_zero_preserved (tests.test_phase2e4_4_closure.TestProductionConfigResolvers.test_training_resolver_num_workers_zero_preserved) ... ok
test_underfilled_dataset_cap_basis_semantics (tests.test_phase2e4_4_closure.TestSingleYearCapSemantics.test_underfilled_dataset_cap_basis_semantics) ... ok
test_metric_aggregation_mathematical_equivalence (tests.test_phase2e4_4_closure.TestTrainingMetricAggregation.test_metric_aggregation_mathematical_equivalence) ... ok
test_zero_valid_targets_safe_handling (tests.test_phase2e4_4_closure.TestTrainingMetricAggregation.test_zero_valid_targets_safe_handling) ... ok

----------------------------------------------------------------------
Ran 12 tests in 2.382s
OK
```

### 5.2 Full Project Regression Discovery
The complete test discovery suite across the entire project repository was executed:

```
Ran 213 tests in 31.371s
OK (100% passing)
```

Test breakdown by area:
- `test_phase2e4_4_closure`: 12 tests (Modality filtering, metric aggregation equivalence, config resolvers, year cap semantics)
- `test_phase2e4_3_closure`: 15 tests (Sample-weighted gradient accumulation, step counting, transfer runbook CLI parsers)
- `test_phase2e4_2_gpu_handoff`: 15 tests (Strict checkpoint resumption, AMP scaler restoration, evaluation smoke tests)
- `test_phase2e4_1_corrections`: 13 tests (LR scheduler stepping, model weight loading, loss denominator validation)
- `test_phase2e4_infrastructure_readiness`: 14 tests (Hardware discovery, config parsing, checkpoint schema)
- `test_prepare_pilot_dataset`: 36 tests (S3 listing, inventory parsing, stratified selection, manifest creation)
- `test_pipeline_units`: 21 tests (NetCDF reader, coordinate transform, physical grid alignment)
- `test_pipeline_real_data`: 8 tests (Pilot dataset integration, real NetCDF loading)
- `test_audit_pilot_modalities`: 18 tests (Modality coverage, data integrity, valid-pixel mask audit)
- `test_two_machine_pipeline`: 7 tests (Dataset, collation, model architecture, masked modality isolation)
- `test_phase2e_corrections`: 16 tests (Self-supervised pretraining MFM, loss formula validation)
- `test_phase2e1_scientific_integrity`: 14 tests (Polar coordinate pooling, grid spacing, target normalization)
- `test_phase2e2_physical_token_integration`: 12 tests (Spatial graph construction, polar sector token fusion)
- `test_phase2e3_scientific_correctness`: 12 tests (Physical grid alignment, vertical radar resampling, graph validity)

---

## 6. Commit and Traceability Record

- **Baseline Commit (Phase 2E.4.3 Closure):** `0cc51cfbcc50d633b5c5572b361239c13e8593ee`
- **Implementation Commit (Phase 2E.4.4):** `87dc7a071747896f48db91914a30a4b10b8d98f4`
- **Working Branch:** `fix/phase2e4-4-modality-metrics-closure`
- **Deliverables Verified:**
  - [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py)
  - [src/training/losses.py](file:///d:/Project/Capstone_project/src/training/losses.py)
  - [src/training/trainer.py](file:///d:/Project/Capstone_project/src/training/trainer.py)
  - [src/evaluation/evaluate_baselines.py](file:///d:/Project/Capstone_project/src/evaluation/evaluate_baselines.py)
  - [src/training/train_multitask.py](file:///d:/Project/Capstone_project/src/training/train_multitask.py)
  - [tests/test_phase2e4_4_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_closure.py)
  - [reports/phase2e4_4/phase2e4_4_closure_report.md](file:///d:/Project/Capstone_project/reports/phase2e4_4/phase2e4_4_closure_report.md)

---

## Conclusion

Phase 2E.4.4 successfully completes all requirements for modality filtering, metric aggregation, production configuration resolution, and single-year budget semantics. The TC-GraphFormer pipeline is mathematically rigorous, fully tested, and ready for deployment on GPU compute resources.
