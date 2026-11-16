# Phase 2E.4.4.1 — Radar Metadata Correctness, Metric Count Reporting & Final Audit Report

**Repository:** `https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Working Branch:** `fix/phase2e4-4-1-radar-metric-reporting-closure`  
**Starting HEAD:** `133fff0ab3709a37e9b797b620cc0d3f2ac35fd6`  
**Verification Date:** November 16, 2026  
**Execution Environment:** Windows 11 AMD64, Python 3.12.9 (`.venv`)  
**Strict Hardware Profile:** CPU execution, 0 live network calls, 0 unapproved data acquisitions.

---

## Executive Summary

Phase 2E.4.4.1 resolves the remaining source-level scientific and reporting issues identified during the Phase 2E.4.4 source review. Building directly on the completed TC-GraphFormer architecture, checkpoint system, and two-machine dataset pipeline, this phase delivers five critical enhancements:

1. **TC PRIMED Radar Identification, Separation & NetCDF Group Inspection:**
   - Removed internal NetCDF group names (`KuKaGMI`, `KuGMI`, `KuTMI`) from `KNOWN_INSTRUMENTS` (which represents filename instrument tokens).
   - Grounded radar capability strictly in physical satellite platform/instrument combinations:
     `RADAR_CAPABLE_COMBINATIONS = {("GMI", "GPM"), ("DPR", "GPM"), ("TMI", "TRMM"), ("PR", "TRMM")}`.
   - Formally separated catalogue-level radar capability (`radar_capable: bool`) from confirmed on-disk radar data availability (`radar_confirmed: "unknown"` at planning time).
   - Implemented `inspect_netcdf_radar_group(file_path)` to inspect local NetCDF group hierarchies for the `radar_radiometer` group, subgroups, and reflectivity/coordinate variables (`zKu`, `zKa`, `reflectivity`, `x`, `y`).
   - Replaced synthetic radar test fixtures in `tests/test_phase2e4_4_closure.py` with realistic instruments/platforms (e.g., GMI on GPM, DPR on GPM).

2. **Deterministic Tuple Ranking for Radar Candidates:**
   - Replaced the scalar score bonus (`+1000.0`, which could be overwhelmed by multi-gigabyte deficit scores of $10^9$ to $10^{10}$ bytes) with deterministic lexicographical tuple ranking:
     $$\text{candidate\_key} = (\text{is\_rad}, \text{deficit\_score}, \text{tie\_val})$$
   - Preserves radar preference as a strong priority without making it a hard exclusion.
   - Applied consistently across Passes 1, 2, and 3 in `stratified_select()`.

3. **Epoch-Level Valid Target Counts & Track Units Distinction:**
   - In `TCGraphFormerTrainer.train_epoch()`, exposed explicit integer epoch metrics:
     `valid_intensity_count`, `valid_pressure_count`, `valid_stage_count`, `valid_intensity_change_count`, `valid_ri_count`, `valid_track_components_count`, `valid_track_timesteps_count`.
   - Explicitly distinguished valid track time steps ($N$) from valid displacement components ($2N$, for $dx$ and $dy$), which act as the denominator for track loss.
   - Eliminated misleading `norm_*_count` keys from `loss_dict` in `MultiTaskLoss`.

4. **Modality-Related Unfilled Budget Attribution:**
   - Corrected `build_selection_report()` to prevent false attribution of budget underfill to modality exclusions when `excluded_storms_count == 0`.
   - Implemented 5 mutually exclusive attribution cases:
     1. Single-year concentration cap ceiling.
     2. Input catalogue smaller than target budget (with zero or non-zero exclusions explicitly detailed).
     3. Modality requirements reducing a previously sufficient catalogue.
     4. Complete-storm lifecycle granularity constraints.
     5. General fallback.

5. **Single-Year Cap Toggle Consistency Across Passes:**
   - Ensured `enforce_single_year_cap` consistently controls the year ceiling across all passes in `stratified_select()` (fixing Pass 1 line 835 check where `enforce_single_year_cap` was previously omitted).

---

## 1. TC PRIMED Radar Identification, Separation & NetCDF Inspection

### 1.1 Filename Instrument Set vs. NetCDF Internal Group Hierarchy
In TC PRIMED (v01r01-final), filenames follow the standard pattern:
```
TCPRIMED_v01r01_<cyclone_id>_<sensor>_<platform>_<datetime>.nc
```
Filename instrument tokens correspond to satellite sensors (such as `GMI`, `TMI`, `SSMIS`, `AMSR2`, `ATMS`, `MHS`). Internal NetCDF groups such as `KuKaGMI`, `KuGMI`, and `KuTMI` reside solely under the `/radar_radiometer` group and never appear as filename tokens.

In [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py):
- `KNOWN_INSTRUMENTS` was cleaned of internal group names:
  ```python
  KNOWN_INSTRUMENTS = {
      "SSMI", "SSMIS", "SSMIS-UPP", "TMI", "AMSR-E", "AMSR2",
      "GMI", "ATMS", "MHS", "AMSUB", "AMSUB-UPP", "DPR", "PR",
  }
  ```
- Physical radar-capable sensor/platform combinations were formally codified:
  ```python
  RADAR_CAPABLE_COMBINATIONS = {
      ("GMI", "GPM"),
      ("DPR", "GPM"),
      ("TMI", "TRMM"),
      ("PR", "TRMM"),
  }
  RADAR_NETCDF_SUBGROUPS = {"KuKaGMI", "KuGMI", "KuTMI"}
  ```

### 1.2 Separation of Capability from Confirmation
Catalogue planning operations examine file paths, platform, and sensor metadata. S3 catalogue planning cannot inspect internal NetCDF variable tables without downloading or reading byte ranges. Therefore:
- At planning time, `build_storm_catalogue()` sets:
  ```python
  "radar_capable": is_rad_cap,        # bool: True if GMI/GPM, TMI/TRMM, DPR, or PR present
  "radar_confirmed": "unknown",       # str: verified during NetCDF ingestion / post-processing
  ```
- Both `save_storm_catalogue_csv()` and `load_storm_catalogue_csv()` preserve `radar_capable` and `radar_confirmed`.

### 1.3 Local NetCDF Radar Group Inspection
For post-download and post-processing verification, `inspect_netcdf_radar_group(file_path)` was introduced:
```python
def inspect_netcdf_radar_group(file_path: str) -> dict:
    res = {
        "file_path": file_path,
        "file_exists": os.path.exists(file_path),
        "has_radar_group": False,
        "subgroups_found": [],
        "radar_confirmed_available": False,
        "reason": "",
    }
    # Verifies /radar_radiometer, searches RADAR_NETCDF_SUBGROUPS,
    # and confirms presence of reflectivity variables (e.g. zKu, zKa, reflectivity).
```

---

## 2. Deterministic Tuple Ranking for Radar Candidates

### 2.1 The Multi-Gigabyte Deficit Scale Defect
In Phase 2E.4.4, Pass 2 and Pass 3 prioritized radar storms using an additive $+1000.0$ bonus to the deficit score:
```python
deficit_score = b_def + e_def + (1000.0 if is_rad else 0.0)
```
Because deficits $b_{\text{def}}$ and $e_{\text{def}}$ are expressed in bytes ($10^9$ to $10^{10}$ bytes for 100 GB planning), a deficit difference of a few hundred megabytes ($> 10^8$ bytes) completely swamped the $+1000.0$ scalar bonus. Consequently, non-radar candidate storms frequently superseded radar candidate storms.

### 2.2 Hierarchical Lexicographical Tuple Ranking
In [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py), the selection logic was replaced with deterministic hierarchical tuple comparisons:
```python
is_rad = 1 if (prioritize_radar and is_storm_radar_capable(storm)) else 0
tie_val = deterministic_tie_breaker(seed, sf)
candidate_key = (is_rad, deficit_score, tie_val)

if best_key is None or candidate_key > best_key:
    best_key = candidate_key
    best_storm = storm
```
- When `prioritize_radar_overpasses: true`, candidate storms with `is_rad = 1` strictly dominate candidate storms with `is_rad = 0`.
- Among storms within the same radar class (`is_rad`), ranking is governed by `deficit_score`.
- Exact ties in deficit score are broken deterministically by SHA-256 hash `tie_val`.
- When `prioritize_radar_overpasses: false`, `is_rad = 0` for all storms, restoring pure deficit-based selection.

---

## 3. Epoch-Level Valid Target Counts & Track Units

### 3.1 Exposing Explicit Valid Counts
In [src/training/trainer.py](file:///d:/Project/Capstone_project/src/training/trainer.py), `train_epoch()` now tracks and reports explicit epoch-level valid sample counts:
```python
metrics["valid_intensity_count"] = int(round(task_valid_counts["intensity"]))
metrics["valid_pressure_count"] = int(round(task_valid_counts["pressure"]))
metrics["valid_stage_count"] = int(round(task_valid_counts["stage"]))
metrics["valid_intensity_change_count"] = int(round(task_valid_counts["intensity_change"]))
metrics["valid_ri_count"] = int(round(task_valid_counts["ri"]))
metrics["valid_track_components_count"] = int(round(task_valid_counts["track_components"]))
metrics["valid_track_timesteps_count"] = int(round(task_valid_counts["track_timesteps"]))
metrics["valid_track_count"] = int(round(task_valid_counts["track_components"]))
```

### 3.2 Track Displacement Components vs. Timesteps
Future track displacement is predicted across 4 forecast horizons: $+6\text{h}$, $+12\text{h}$, $+18\text{h}$, $+24\text{h}$. Each valid timestep comprises 2 orthogonal displacement components: $[dx, dy]$.
- `valid_track_timesteps_count` reflects the number of valid forecast horizons ($N$).
- `valid_track_components_count` reflects the total displacement scalar targets ($2N$), which serves as the exact divisor for the Huber displacement loss.
- In [src/training/losses.py](file:///d:/Project/Capstone_project/src/training/losses.py), both metrics are explicitly computed, and legacy misleading keys (`norm_*_count`) have been removed.

---

## 4. Modality-Related Unfilled Budget Attribution

### 4.1 Exhaustive Attribution Logic
In [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py), `build_selection_report()` previously attributed any budget shortfall to modality exclusions whenever `modality_audit` was present, even if `excluded_storms_count == 0`.

The logic now classifies budget shortfalls into 5 mutually exclusive categories:
```python
# Case 1: Check single-year concentration cap
if blocked_by_year_cap:
    unfilled_budget_reason = (
        f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB ({unfilled_bytes:,} bytes): "
        f"single-year concentration cap ({single_year_max_pct}% of target budget) strictly enforced "
        f"and candidate storms from other years exhausted."
    )
# Case 2: Original catalogue was already smaller than requested target budget
elif orig_bytes < target_bytes:
    if excluded_count == 0:
        unfilled_budget_reason = (
            f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB: original catalogue "
            f"({orig_bytes / 1e9:.2f} GB) is smaller than requested target budget ({target_bytes / 1e9:.2f} GB) "
            f"with zero modality exclusions."
        )
    else:
        unfilled_budget_reason = (
            f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB: original catalogue "
            f"({orig_bytes / 1e9:.2f} GB) was already smaller than requested target budget ({target_bytes / 1e9:.2f} GB), "
            f"and modality requirements further excluded {excluded_count} storms{reasons_str}, leaving {elig_bytes / 1e9:.2f} GB eligible."
        )
# Case 3: Modality filtering reduced a previously sufficient catalogue (orig >= target, elig < target)
elif orig_bytes >= target_bytes and elig_bytes < target_bytes:
    unfilled_budget_reason = (
        f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB: modality requirements "
        f"excluded {excluded_count} storms{reasons_str}, reducing available catalogue "
        f"from {orig_bytes / 1e9:.2f} GB to {elig_bytes / 1e9:.2f} GB."
    )
# Case 4: Eligible catalogue sufficient (elig >= target), but complete storm size constraints prevented fitting
elif elig_bytes >= target_bytes:
    unselected_eligible = [s for s in catalogue if s["storm_folder"] not in selected_folders]
    min_unselected_bytes = min((s["total_bytes"] for s in unselected_eligible), default=0)
    unfilled_budget_reason = (
        f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB: eligible catalogue is sufficient in total bytes "
        f"({elig_bytes / 1e9:.2f} GB), but complete-storm lifecycle constraints prevent adding remaining candidate storms "
        f"(smallest remaining: {min_unselected_bytes / 1e9:.4f} GB) without exceeding target budget ({target_bytes / 1e9:.2f} GB)."
    )
# Case 5: General fallback
```

---

## 5. Single-Year Cap Toggle Consistency

### 5.1 Enforcing Toggle in Pass 1
In `stratified_select()`, Pass 1 line 835 previously checked:
```python
if max_single_year_bytes > 0 and (year_bytes[s_year] + storm_bytes > max_single_year_bytes):
    continue
```
This check omitted `if enforce_single_year_cap`, causing Pass 1 to discard storms exceeding the year limit even when `enforce_single_year_cap=False`.

The check was corrected to:
```python
if enforce_single_year_cap and max_single_year_bytes > 0 and (year_bytes[s_year] + storm_bytes > max_single_year_bytes):
    continue
```
Now, setting `enforce_single_year_cap=False` consistently relaxes the year ceiling across Pass 1, Pass 2, and Pass 3, allowing maximum catalogue utilization when configured by the operator.

---

## 6. Verification and Regression Test Evidence

### 6.1 Dedicated Phase 2E.4.4.1 Test Suite
The new test suite [tests/test_phase2e4_4_1_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_1_closure.py) directly exercises all five core requirements:
- `TestRadarMetadataAndInspection`: Verifies exclusion of internal groups from `KNOWN_INSTRUMENTS`, validation of `RADAR_CAPABLE_COMBINATIONS`, catalogue status recording (`radar_capable: True`, `radar_confirmed: "unknown"`), CSV roundtrip, and disk NetCDF hierarchy inspection with `inspect_netcdf_radar_group()`.
- `TestRadarCandidateRanking`: Verifies that a radar-capable storm with a tiny deficit is selected over a non-radar storm with a 10 GB deficit when radar preference is active.
- `TestEpochValidTargetCounts`: Verifies that `train_epoch()` produces explicit valid counts, confirms `valid_track_components_count == 2 * valid_track_timesteps_count`, and ensures no `norm_*` keys exist in `loss_dict`.
- `TestModalityUnfilledBudgetCases`: Verifies attribution when the catalogue is smaller than budget with zero exclusions, and when modality requirements reduce an otherwise sufficient catalogue.
- `TestSingleYearCapToggle`: Verifies consistent behavior between `enforce_single_year_cap=False` and `True`.

**Execution:**
```bash
.venv\Scripts\python.exe -m unittest tests.test_phase2e4_4_1_closure -v
Ran 12 tests in 1.890s -> OK
```

### 6.2 Full Discovery Across All Modules
A complete regression discovery run was executed on the full repository:
```bash
.venv\Scripts\python.exe -m unittest discover tests -v
Ran 225 tests in 28.723s -> OK
```

#### Exact Per-Module Test Breakdown (17 Modules)
| Test Module | Test Count | Result |
| :--- | :---: | :---: |
| `test_audit_pilot_modalities` | 11 | OK |
| `test_eda_pilot_dataset` | 7 | OK |
| `test_phase2e1_scientific_integrity` | 16 | OK |
| `test_phase2e2_physical_token_integration` | 12 | OK |
| `test_phase2e3_scientific_correctness` | 12 | OK |
| `test_phase2e4_1_corrections` | 9 | OK |
| `test_phase2e4_2_gpu_handoff` | 5 | OK |
| `test_phase2e4_3_closure` | 7 | OK |
| `test_phase2e4_4_1_closure` | 12 | OK |
| `test_phase2e4_4_closure` | 12 | OK |
| `test_phase2e4_infrastructure_readiness` | 9 | OK |
| `test_phase2e_corrections` | 24 | OK |
| `test_pipeline_real_data` | 6 | OK |
| `test_pipeline_units` | 7 | OK |
| `test_prepare_pilot_dataset` | 61 | OK |
| `test_reconcile_pilot_consistency` | 8 | OK |
| `test_two_machine_pipeline` | 7 | OK |
| **Total** | **225** | **OK (100% Passed)** |

---

## 7. Artifacts and Preservation

- `reports/selection/` was left completely untouched and uncommitted per project instructions.
- All modified code adheres to conventional commits, strict scientific rigor, zero-leakage masking, and deterministic reproducibility.
