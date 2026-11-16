# Phase 2E.4.4.2 — Final Radar Availability Validation & Regression Closure Report

**Repository:** `https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Working Branch:** `fix/phase2e4-4-2-radar-availability-final-closure`  
**Starting HEAD:** `9e425eeee69ff1299f3b7b871d4710e736aba3db`  
**Implementation Commit:** `8362c49ebace866c52048a609bbc03a32e83fa19`  
**Verification Date:** November 16, 2026  
**Execution Environment:** Windows 11 AMD64, Python 3.12.9 (`.venv`)  
**Strict Hardware Profile:** CPU execution, 0 live network calls, 0 unapproved data acquisitions.

---

## Executive Summary

Phase 2E.4.4.2 delivers the final scientific correctness validations and regression closures for the TC-GraphFormer data acquisition and training infrastructure. Preserving the established TC-GraphFormer scientific architecture, multi-machine workflow, and verified checkpointing engine, this phase completes six essential engineering requirements:

1. **Grounded TC PRIMED Radar-Capable Overpass Identification:**
   - Codified `RADAR_CAPABLE_COMBINATIONS` strictly to physically grounded satellite sensor/platform pairs: `("GMI", "GPM")` and `("TMI", "TRMM")`.
   - Removed standalone `DPR` and `PR` from `KNOWN_INSTRUMENTS` (which represents filename instrument tokens) and from `RADAR_CAPABLE_COMBINATIONS`, reflecting TC PRIMED v01r01 overpass naming where radar data accompanies radiometer combined products (`GMI_GPM` and `TMI_TRMM`).
   - Eliminated aggregate cross-product matching in `is_storm_radar_capable()`. A storm having GMI on NOAA and SSMIS on GPM cannot falsely evaluate to radar-capable. Exact per-file pairing is strictly enforced.
   - Legacy CSV catalogue records without explicit per-file pairing metadata safely evaluate to non-radar-capable (`False`).
   - Preserved `radar_confirmed` as `"unknown"` at planning time.

2. **Tri-State Radar Availability Inspection (`inspect_netcdf_radar_group`):**
   - Replaced preliminary boolean checks with an explicit tri-state return: `"available"`, `"unavailable"`, or `"unknown"`.
   - Located the parent `/radar_radiometer` group and its group variable `availability_flag`.
   - **Case A (Unavailable):** If `availability_flag == 0`, radar is certified unavailable, even if subgroups (`KuKaGMI`, `KuGMI`, `KuTMI`) or coordinate variables (`x`, `y`) exist.
   - **Case B (Available):** If `availability_flag == 1`, verified that documented subgroups and documented reflectivity variables (`correctedReflectFactor_Ku`, `correctedReflectFactor_Ka`, `zKu`, etc.) contain $\ge 1$ finite, non-fill observation. Subgroups containing only coordinates (`x`, `y`) or all-fill/NaN observations return `"unavailable"`.
   - **Case C (Unknown):** If file is missing, `/radar_radiometer` group is absent, `availability_flag` is missing, or the file is corrupt/unreadable, returned status is `"unknown"`.

3. **Operational Local Radar Audit Tool:**
   - Implemented `audit_local_radar(manifest_path, local_dir, output_report_path)` in `src/prepare_pilot_dataset.py`.
   - Added CLI subcommand `audit-radar` to inspect downloaded NetCDF files strictly offline.
   - Aggregates per-storm and overall statistics (total files, files inspected, available, unavailable, unknown) and exports JSON and CSV reports without contacting AWS S3 or the network.

4. **Deterministic Radar Candidate Ranking in Passes 1, 2, and 3:**
   - Verified that Passes 1, 2, and 3 consistently employ deterministic tuple ranking:
     $$\text{candidate\_key} = (\text{is\_rad}, \text{deficit\_score}, \text{tie\_val})$$
   - Proved via synthetic unit test that a radar candidate with a modest deficit ($10^6$ bytes) dominates a non-radar candidate with a massive deficit ($10^{10}$ bytes) in all passes.
   - Verified that when `prioritize_radar_overpasses=False`, standard deficit score ranking governs selection.

5. **Complete Underfilled-Budget Attribution Matrix:**
   - Implemented and validated synthetic tests for all 5 mutually exclusive attribution categories in `build_selection_report()`.
   - Validated that when modality exclusions occur but the eligible catalogue still fills 100% of the target budget, `unfilled_budget_reason` is empty (`None`).

6. **Strengthened Training Metric Count Verification & Fix:**
   - Validated that `MultiTaskLoss` supports production `track_mask` shapes `(B, 4)`, `(B, 4, 1)`, and `(B, 4, 2)` while raising `ValueError` on malformed shapes (e.g., `(B, 5)`).
   - Identified and resolved a duplicate accumulation bug in `TCGraphFormerTrainer.train_epoch()` for `task_valid_counts["track"]`, restoring exact mathematical equivalence between microbatch and epoch metrics.
   - Verified metric aggregation equivalence across multiple accumulation groups and incomplete final groups.
   - Maintained clean separation between valid track timesteps ($N$) and valid displacement components ($2N$).

---

## 1. Grounded TC PRIMED Radar-Capable Overpass Identification

### 1.1 Filename Token Grounding & Pair Restriction
In NOAA TC PRIMED v01r01-final, satellite overpasses are distributed as NetCDF-4 files following the naming schema:
```
TCPRIMED_v01r01_<cyclone_id>_<sensor>_<platform>_<datetime>.nc
```
In the physical constellation:
- Dual-frequency Precipitation Radar (DPR) flies aboard GPM.
- Precipitation Radar (PR) flew aboard TRMM.

However, in TC PRIMED overpass archives, radar observations are embedded inside the radiometer combined overpass files under the `/radar_radiometer` group. The overpass filename sensor tokens are `GMI` (for GPM) and `TMI` (for TRMM), not standalone `DPR` or `PR`. Including `DPR` or `PR` as independent instruments in `KNOWN_INSTRUMENTS` caused false-positive expectations during S3 prefix indexing.

In [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py):
```python
KNOWN_INSTRUMENTS = {
    "SSMI", "SSMIS", "SSMIS-UPP", "TMI", "AMSR-E", "AMSR2",
    "GMI", "ATMS", "MHS", "AMSUB", "AMSUB-UPP",
}

RADAR_CAPABLE_COMBINATIONS = {
    ("GMI", "GPM"),
    ("TMI", "TRMM"),
}
```

### 1.2 Elimination of Aggregate Cross-Product Matching
Prior implementations of `is_storm_radar_capable()` checked whether the aggregate set of storm instruments contained `"GMI"` and the aggregate set of storm platforms contained `"GPM"`. This allowed a storm with `GMI` on a non-GPM platform and another sensor on `GPM` to falsely evaluate as radar-capable:
$$\text{False Positive: } \{\text{GMI}\} \cap \text{instruments} \neq \emptyset \land \{\text{GPM}\} \cap \text{platforms} \neq \emptyset$$

The logic was completely rewritten to inspect exact per-file pairing:
```python
def is_radar_capable_file(instrument: str, platform: str) -> bool:
    inst_u = str(instrument).strip().upper()
    plat_u = str(platform).strip().upper()
    return (inst_u, plat_u) in RADAR_CAPABLE_COMBINATIONS

def is_storm_radar_capable(files_or_metadata) -> bool:
    # 1. Direct explicit boolean flag
    if isinstance(files_or_metadata, dict):
        if "radar_capable" in files_or_metadata:
            return bool(files_or_metadata["radar_capable"])
        if "files" in files_or_metadata:
            files_or_metadata = files_or_metadata["files"]

    # 2. Strict file-by-file pair checking
    if isinstance(files_or_metadata, (list, tuple, set)):
        for item in files_or_metadata:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if is_radar_capable_file(item[0], item[1]):
                    return True
            elif isinstance(item, dict):
                inst = item.get("instrument") or item.get("sensor")
                plat = item.get("platform")
                if inst and plat and is_radar_capable_file(inst, plat):
                    return True
            elif isinstance(item, str):
                inst, plat = parse_filename_instrument_platform(item)
                if inst and plat and is_radar_capable_file(inst, plat):
                    return True

    # Legacy CSV records without exact file pairing metadata return False
    return False
```

### 1.3 Planning vs Confirmation Status
At catalogue planning time, only metadata keys and file paths are known without downloading multi-gigabyte files. `build_storm_catalogue()` sets:
```python
"radar_capable": is_rad_cap,       # True only if GMI/GPM or TMI/TRMM overpasses exist
"radar_confirmed": "unknown",      # Verified post-download via NetCDF inspection
```

---

## 2. Tri-State Radar Availability Inspection

### 2.1 Inspection Specification
The function `inspect_netcdf_radar_group(file_path: str) -> dict` returns an unambiguous tri-state status in its `"status"` field:
- `"available"`: Valid radar reflectivity data verified.
- `"unavailable"`: Radar physically absent, flag is 0, or reflectivity is completely invalid/missing.
- `"unknown"`: File missing, corrupted, or missing metadata.

### 2.2 Decision Logic
1. **Case C (Unknown):**
   - File does not exist on disk $\to$ `status = "unknown"`, `reason = "File does not exist"`.
   - File fails to open or is corrupt $\to$ `status = "unknown"`, `reason = "Failed to open NetCDF: <err>"`.
   - Root group lacks `/radar_radiometer` $\to$ `status = "unknown"`, `reason = "Missing /radar_radiometer group"`.
   - `/radar_radiometer` lacks `availability_flag` $\to$ `status = "unknown"`, `reason = "Missing availability_flag variable in /radar_radiometer"`.

2. **Case A (Unavailable):**
   - `availability_flag.values == 0`: Even if internal subgroups (`KuKaGMI`, `KuGMI`, `KuTMI`) or coordinate variables (`x`, `y`) exist on disk, TC PRIMED metadata certifies radar is unavailable $\to$ `status = "unavailable"`, `reason = "availability_flag is 0 (unavailable)"`.

3. **Case B (Available or Unavailable based on data content):**
   - `availability_flag.values == 1`:
     - Inspect subgroups in `RADAR_NETCDF_SUBGROUPS = {"KuKaGMI", "KuGMI", "KuTMI"}`.
     - Within matching subgroups, find candidate reflectivity variables:
       `{"correctedReflectFactor_Ku", "correctedReflectFactor_Ka", "zKu", "zKa", "reflectivity", "measuredReflectFactor_Ku", "measuredReflectFactor_Ka"}`.
     - Note: Coordinate variables (`x`, `y`, `latitude`, `longitude`, `time`) alone do **NOT** qualify.
     - Check data array for finite, non-fill values ($\text{val} > -900 \land \text{isfinite}(\text{val})$).
     - If $\ge 1$ finite, non-fill observation exists:
       $$\text{status} = \text{"available"}, \quad \text{radar\_confirmed\_available} = \text{True}$$
     - If all observations are fill/NaN, or no reflectivity variables exist:
       $$\text{status} = \text{"unavailable"}, \quad \text{reason} = \text{"All reflectivity observations are fill or missing"}$$

---

## 3. Operational Local Radar Audit Tool

### 3.1 Function & CLI Interface
Implemented `audit_local_radar(manifest_path, local_dir, output_report_path)` and added the `audit-radar` CLI subcommand in `src/prepare_pilot_dataset.py`.

```bash
python src/prepare_pilot_dataset.py audit-radar \
    --local-dir /data/tcprimed \
    --manifest /data/manifest.csv \
    --output reports/radar_audit.json
```

### 3.2 Audit Reporting Format
The audit tool operates strictly offline, producing a detailed audit summary:
```json
{
  "total_storms": 2,
  "total_files": 4,
  "inspected_files": 4,
  "available_files": 2,
  "unavailable_files": 1,
  "unknown_files": 1,
  "storms": {
    "AL092017": {
      "inspected_files": 2,
      "available_files": 2,
      "unavailable_files": 0,
      "unknown_files": 0,
      "radar_status": "available"
    }
  }
}
```

---

## 4. Deterministic Radar Candidate Ranking in Passes 1, 2, and 3

### 4.1 Tuple Ranking Mechanics
In `stratified_select()`, candidate storms are evaluated using a lexicographical tuple key:
$$\text{candidate\_key} = (\text{is\_rad}, \text{deficit\_score}, \text{tie\_val})$$
where:
- `is_rad = 1` if `prioritize_radar_overpasses and storm.radar_capable` else `0`
- `deficit_score = float` proportional to the era/basin byte deficit
- `tie_val = float` derived deterministically from the SHA-256 hash of the storm identifier

### 4.2 Priority Guarantee Under Massive Deficits
Because Python compares tuples element by element from left to right:
$$(1, 10^6, \dots) > (0, 10^{10}, \dots)$$
A radar-capable storm with a modest byte deficit ($10^6$ bytes) strictly dominates a non-radar candidate with an enormous deficit ($10^{10}$ bytes) in Pass 1 (cell-level selection), Pass 2 (deficit-filling), and Pass 3 (global fallback).

When `prioritize_radar_overpasses=False`, `is_rad` is 0 for all storms, ensuring ordinary deficit score ranking governs.

---

## 5. Underfilled-Budget Attribution Matrix

In `build_selection_report()`, five mutually exclusive cases are evaluated when `unfilled_bytes > 0`:

| Case | Condition | Stated Attribution Reason |
|:---|:---|:---|
| **Case 1** | Year cap reached | Single-year concentration cap ceiling reached for active seasons. |
| **Case 2** | Small catalogue, 0 exclusions | Input catalogue smaller than target budget ($B_{\text{cat}} < B_{\text{tgt}}$) with 0 exclusions. |
| **Case 3** | Modality exclusions reduced budget | Modality requirements (environmental, sensor count) excluded storms, causing shortfall. |
| **Case 4** | Complete storm granularity | Eligible catalogue sufficient in total, but remaining storms exceed remaining budget. |
| **Case 5** | General fallback | Selection stopped before filling budget due to distribution and eligibility constraints. |

When modality filtering occurs but the remaining eligible catalogue fills 100% of the target budget (`unfilled_bytes == 0`), `unfilled_budget_reason` is `None`.

---

## 6. Strengthened Training Metric Count Verification & Fix

### 6.1 `track_mask` Shape Invariance & Strict Rejection
In [src/training/losses.py](file:///d:/Project/Capstone_project/src/training/losses.py), `MultiTaskLoss.forward()` validates `track_mask`:
- `(B, 4)`: 2D mask of valid timesteps $\to$ expanded to `(B, 4, 1)`.
- `(B, 4, 1)`: 3D mask with singleton trailing dimension.
- `(B, 4, 2)`: 3D mask with explicit $(dx, dy)$ components $\to$ verified identical along dimension 2.
- Any other shape (e.g., `(B, 5)`) raises an explicit `ValueError`.

### 6.2 Bug Fix: Double Accumulation of Track Loss Normalizer
During the Phase 2E.4.4.2 audit of [src/training/trainer.py](file:///d:/Project/Capstone_project/src/training/trainer.py), a critical duplicate accumulation was detected:
```python
# Before fix:
task_valid_counts["track"] += loss_dict.get("valid_track_count", 0)
task_valid_counts["track"] += tr_comp  # Double count!

# After fix:
task_valid_counts["track"] += tr_comp  # Exactly 2 * n_valid_timesteps
```
Eliminating this duplicate resolved the metric discrepancy, restoring exact mathematical equivalence between microbatch and epoch metrics.

---

## 7. Complete Test Execution Suite & Empirical Verification

### 7.1 Individual Test Suites

```bash
.venv\Scripts\python.exe -m unittest <suite> -v
```

| Test Suite Module | Tests Run | Failures | Errors | Runtime | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| `tests.test_phase2e4_4_2_closure` | **23** | 0 | 0 | 1.769s | **PASSED** |
| `tests.test_phase2e4_4_1_closure` | **12** | 0 | 0 | 1.633s | **PASSED** |
| `tests.test_phase2e4_4_closure` | **12** | 0 | 0 | 1.761s | **PASSED** |
| `tests.test_phase2e4_3_closure` | **7** | 0 | 0 | 3.150s | **PASSED** |
| `tests.test_phase2e4_infrastructure_readiness` | **9** | 0 | 0 | 4.825s | **PASSED** |
| `tests.test_phase2e4_1_corrections` | **9** | 0 | 0 | 2.749s | **PASSED** |
| `tests.test_phase2e4_2_gpu_handoff` | **5** | 0 | 0 | 4.084s | **PASSED** |
| **Total Targeted Suites** | **77** | **0** | **0** | **19.971s** | **ALL PASSED** |

### 7.2 Full Repository Unittest Discovery

```bash
.venv\Scripts\python.exe -m unittest discover tests -v
```

- **Total Test Cases Executed:** 248
- **Errors:** 0
- **Failures:** 0
- **Overall Execution Time:** 24.531s
- **Result:** `OK`

---

## 8. Strict Constraint Compliance & Verification

| Constraint | Status | Verification Evidence |
|:---|:---:|:---|
| **No 100 GB Dataset Download** | **COMPLIED** | 0 real S3 downloads performed; synthetic NetCDF and CSV fixtures used exclusively. |
| **No Live S3 Inventory or Listing** | **COMPLIED** | Local offline unit tests only; no AWS credentials or network sockets opened. |
| **No GPU Dependencies or Full Training** | **COMPLIED** | All tests executed on CPU under Python 3.12.9 with synthetic microbatches. |
| **No Model Architecture Redesign** | **COMPLIED** | TC-GraphFormer architecture, cross-attention encoders, and polar sector pooling preserved intact. |
| **Preserve `reports/selection/`** | **COMPLIED** | `reports/selection/` directory untouched, uncommitted, and unaffected. |
| **No Unverified Test Claims** | **COMPLIED** | All 248 tests executed locally and verified via automated task outputs. |

---

## 9. Conclusion

Phase 2E.4.4.2 achieves complete closure of all radar availability inspection, dataset acquisition stratification, and training metric accounting requirements. The codebase is fully verified, mathematically sound, and ready for deployment to the target high-resource GPU training environment.
