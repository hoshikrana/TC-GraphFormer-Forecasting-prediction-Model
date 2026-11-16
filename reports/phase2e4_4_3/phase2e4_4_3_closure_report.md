# Phase 2E.4.4.3 — Final Audit Semantics, Selection Logic & Training Metric Closure Report

**Repository:** `https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Working Branch:** `fix/phase2e4-4-3-final-audit-and-metric-closure`  
**Starting HEAD:** `8362c49ebace866c52048a609bbc03a32e83fa19`  
**Implementation Commit:** `acfbc919e342cfc526303f7e10a4e210c11ffc66`  
**Verification Date:** November 16, 2026  
**Execution Environment:** Windows 11 AMD64, Python 3.12.9 (`.venv`)  
**Strict Hardware Profile:** CPU execution, 0 live network calls, 0 unapproved data acquisitions.  
**CI/Status Check Note:** All test suites executed and verified locally; no remote GitHub Actions workflows were triggered or evaluated during this phase.

---

## Executive Summary

Phase 2E.4.4.3 closes the remaining audit semantics, selection logic reachability, and training metric consistency gaps in TC-GraphFormer's data acquisition and training engine. The physical architecture, cross-attention encoders, polar sector pooling, checkpoint engine, and multi-machine pipeline remain strictly preserved.

This phase implements five essential engineering corrections:

1. **Local Radar Audit Applicability & Isolation:**
   - Isolated non-radar-capable files (e.g. SSMIS, ATMS, MHS, AMSR2) and environmental files as `not_applicable`, recording them in a dedicated `not_applicable_count` rather than allowing them to contaminate `unknown_count`.
   - Preserved the three physical radar confirmation states (`available`, `unavailable`, `unknown`) strictly for applicable radar-capable candidates (`GMI/GPM` and `TMI/TRMM`).
   - Added robust filename parsing fallback when manifest sensor/platform fields are absent, and reported explicit uncertainty when applicability cannot be established.
   - Enforced strict internal count consistency: $\text{total\_files} = \text{available} + \text{unavailable} + \text{unknown} + \text{not\_applicable}$.

2. **Unified Audit CLI Interface:**
   - Added `--local-dir` as an explicit first-class CLI argument alias alongside `--data-dir` and `--destination` in the `audit-radar` subcommand.
   - Verified offline execution of:
     ```bash
     python src/prepare_pilot_dataset.py audit-radar --local-dir <local_path> --manifest <manifest_path> --output <report_path>
     ```

3. **Resolution of Redundant Selection Stages:**
   - Discovered that Pass 3 in `stratified_select()` was an exact copy of Pass 2's candidate eligibility checks and was 100% unreachable.
   - Safely removed the redundant Pass 3 code block without altering effective selection semantics.
   - Formally documented the two actual, effective selection stages:
     - **Stage 1 (Pass 1 - Cell-Budget Stratification):** Intra-cell deterministic ranking by `(0 if radar_capable else 1, -total_bytes, tie_breaker)`.
     - **Stage 2 (Pass 2 - Deficit-Balanced Global Filling):** Cross-catalogue greedy deficit filling maximizing `candidate_key = (is_rad, deficit_score, tie_val)`.

4. **Reachable Underfilled-Budget Attribution Matrix:**
   - Identified that Case 5 (general fallback) was mathematically unreachable because Cases 2, 3, and 4 form a complete partition of the eligible/original catalogue byte space under valid inputs ($\text{elig\_bytes} \le \text{orig\_bytes}$).
   - Restructured the attribution matrix into four genuinely reachable categories (with Case 2 partitioned into Case 2a for zero exclusions and Case 2b for non-zero exclusions):
     - Case 1: Single-year concentration cap ceiling.
     - Case 2a: Original catalogue smaller than budget with 0 exclusions.
     - Case 2b: Original catalogue smaller than budget with non-zero exclusions.
     - Case 3: Modality exclusions reducing previously sufficient catalogue.
     - Case 4: Complete storm lifecycle granularity constraint.
   - Updated test assertions to check exact distinct attribution messages rather than generic strings.

5. **Track-Mask Shape Contract & Symmetric Verification:**
   - In `MultiTaskLoss` and `TCGraphFormerTrainer`, enforced identical validity masks across X and Y components when using 3D masks `(B, 4, 2)`.
   - Rejection of asymmetric X/Y component masks with an explicit `ValueError`.
   - Production shape `(B, 4)` and singleton shape `(B, 4, 1)` fully supported.
   - End-to-end `train_epoch()` accumulation verified on production `(B, 4)` track masks across multiple accumulation groups with an incomplete final residual group.

---

## 1. Local Radar Audit Applicability & Isolation

### 1.1 Problem Diagnosis
Previously, `audit_local_radar()` evaluated every file in the input directory or manifest using `inspect_netcdf_radar_group()`. For non-radar satellite overpasses (e.g., SSMIS on DMSP platforms, ATMS on NOAA platforms) and environmental files, `/radar_radiometer` is absent by design. These files were recorded as `"unknown"`, falsely inflating the unknown radar count to include valid, expected non-radar overpasses.

### 1.2 Implemented Applicability Classification
In [src/prepare_pilot_dataset.py](file:///d:/Project/Capstone_project/src/prepare_pilot_dataset.py), `audit_local_radar()` now applies strict hierarchical applicability classification:

```python
is_env = is_environmental_file(fname) or is_environmental_file(file_path)
is_rad_cap = is_radar_capable_file(inst, plat) if (inst and plat) else False
known_insts_u = {k.upper() for k in KNOWN_INSTRUMENTS}
known_plats_u = {p.upper() for p in KNOWN_PLATFORMS}

if is_env:
    applicability = "not_applicable"
    status = "not_applicable"
    reason = "Environmental file (radar not applicable)"
elif is_rad_cap:
    applicability = "applicable"
    inspect_res = inspect_netcdf_radar_group(file_path)
    status = inspect_res.get("status", "unknown")
    reason = inspect_res.get("reason", "")
else:
    inst_u = str(inst).strip().upper() if inst else ""
    plat_u = str(plat).strip().upper() if plat else ""
    if inst_u in known_insts_u or plat_u in known_plats_u:
        applicability = "not_applicable"
        status = "not_applicable"
        reason = f"Sensor {inst or 'unknown'} on platform {plat or 'unknown'} is not radar-capable"
    else:
        applicability = "unknown"
        status = "unknown"
        reason = f"Uncertain sensor/platform applicability ({inst}/{plat}); cannot establish radar applicability"
```

### 1.3 Count Consistency Guarantee
The resulting report aggregates:
- `total_files`: total files audited.
- `total_available`: candidate files with confirmed radar observations.
- `total_unavailable`: candidate files where radar is certified unavailable.
- `total_unknown`: candidate files where radar status could not be resolved (e.g. corrupt or missing file) or unresolvable sensor/platform.
- `total_not_applicable` / `not_applicable_count`: non-radar overpasses and environmental files.

Internal consistency holds globally and per-storm:
$$\text{total\_files} = \text{total\_available} + \text{total\_unavailable} + \text{total\_unknown} + \text{total\_not\_applicable}$$

---

## 2. Audit CLI Interface

### 2.1 Argument Support
In `src/prepare_pilot_dataset.py`, the `audit-radar` subcommand accepts:
- `--local-dir`: Path to local directory with NetCDF files.
- `--manifest`: Path to manifest CSV.
- `--output`: Output path for JSON or CSV report.
- `--data-dir` and `--destination`: Preserved as backward-compatible aliases for `--local-dir`.

### 2.2 Tested CLI Invocation
```bash
python src/prepare_pilot_dataset.py audit-radar \
    --local-dir /data/tcprimed \
    --manifest /data/manifest.csv \
    --output reports/radar_audit.json
```
The command runs completely offline with zero network or AWS calls.

---

## 3. Effective Two-Stage Selection & Reachability

### 3.1 Unreachability Analysis of Legacy Pass 3
Inspection of `stratified_select()` revealed:
- **Pass 1** fills era $\times$ basin cells up to cell budgets.
- **Pass 2** loops `while selected_bytes < target_bytes:`, evaluating every unselected storm where:
  - `selected_bytes + storm_bytes <= target_bytes`, and
  - If `enforce_single_year_cap`, single-year cap is respected.
  Pass 2 terminates when `best_storm is None`.
- **Pass 3** re-evaluated the exact same candidate conditions. Because Pass 2 already exhausted all storms satisfying those conditions, Pass 3 found zero candidates and could never select any storm.

### 3.2 Two Clean Effective Selection Stages
The dead code was removed, leaving two clear and documented stages:

1. **Stage 1 (Pass 1 - Cell-Budget Allocation):**
   - Candidate storms partitioned into `(era, basin)` cells.
   - Candidates sorted by:
     $$\text{sort\_key} = (0 \text{ if radar\_capable else } 1, -\text{total\_bytes}, \text{tie\_val})$$
   - Greedily added up to each cell's budget, respecting total target budget and single-year concentration cap.

2. **Stage 2 (Pass 2 - Deficit-Balanced Global Filling):**
   - Evaluates all remaining unselected candidates.
   - Greedily selects candidate maximizing:
     $$\text{candidate\_key} = (\text{is\_rad}, \text{deficit\_score}, \text{tie\_val})$$
     where $\text{is\_rad} = 1$ if `prioritize_radar and radar_capable` else $0$, and $\text{deficit\_score} = \text{basin\_deficit} + \text{era\_deficit}$.
   - Halts when total budget is filled or no candidate fits.

---

## 4. Underfilled-Budget Attribution Matrix

### 4.1 Partition of the State Space
For any valid dataset selection where $\text{unfilled\_bytes} > 0$:
Since filtering only discards storms, $\text{elig\_bytes} \le \text{orig\_bytes}$ is an invariant.
The relationship between $\text{orig\_bytes}$, $\text{elig\_bytes}$, and $\text{target\_bytes}$ partitions into:
1. $\text{orig\_bytes} < \text{target\_bytes}$ (Case 2: Insufficient Catalogue)
   - $2a$: $\text{excluded\_count} == 0$ (zero modality exclusions)
   - $2b$: $\text{excluded\_count} > 0$ (modality requirements further reduced catalogue)
2. $\text{orig\_bytes} \ge \text{target\_bytes} \land \text{elig\_bytes} < \text{target\_bytes}$ (Case 3: Modality Shortfall)
3. $\text{elig\_bytes} \ge \text{target\_bytes}$ (Case 4: Complete Storm Granularity)
4. $\text{blocked\_by\_year\_cap} == \text{True}$ (Case 1: Single-Year Concentration Cap Ceiling)

Because Cases 2, 3, and 4 cover the entire real line where $\text{elig\_bytes} \le \text{orig\_bytes}$, Case 5 (general fallback) was mathematically unreachable under valid inputs. The report and test suite now document and verify the four genuine reachable categories.

---

## 5. Track-Mask Semantics & Metric Verification

### 5.1 Symmetry Validation for Paired Displacements
In TC PRIMED and TC-GraphFormer, track forecasting predicts 2D displacement vectors $(dx, dy)$ over 4 future horizons (e.g. 12h, 24h, 36h, 48h). A forecast horizon is either physically valid in both coordinates or invalid in both.

When target masks are provided in 3D format `(B, 4, 2)`:
```python
if not torch.equal(track_mask[:, :, 0], track_mask[:, :, 1]):
    raise ValueError(
        "Asymmetric track_mask: X and Y components must have identical validity masks for paired displacement targets."
    )
valid_tr_timesteps = float(track_mask[:, :, 0].sum().item())
valid_tr_components = valid_tr_timesteps * 2.0
```
- Asymmetric masks are rejected with `ValueError`.
- Valid timesteps count ($N$) is an integer (never fractional).
- Valid components count ($2N$) strictly equals $2 \times \text{valid timesteps}$.
- Production shape `(B, 4)` and `(B, 4, 1)` are seamlessly supported.

### 5.2 Microbatch Accumulation Invariance
In `TCGraphFormerTrainer.train_epoch()`, multi-task accumulation with production `(B, 4)` masks across 5 batches with `accumulate_grad_batches=2` (two complete groups and one incomplete residual group) was empirically verified:
- Valid track components count: exactly 72 (9 samples $\times$ 4 timesteps $\times$ 2 coordinates).
- Valid track timesteps count: exactly 36 (9 samples $\times$ 4 timesteps).
- Reported epoch loss matches microbatch sum to $< 10^{-4}$ precision.
- Optimizer steps executed: exactly 3.

---

## 6. Test Execution Suite & Empirical Verification

### 6.1 Targeted Test Suites

```bash
.venv\Scripts\python.exe -m unittest <suite> -v
```

| Suite | Tests Run | Failures | Errors | Runtime | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| [tests/test_phase2e4_4_3_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_3_closure.py) | **17** | 0 | 0 | 2.060s | **PASSED** |
| [tests/test_phase2e4_4_2_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_2_closure.py) | **23** | 0 | 0 | 1.972s | **PASSED** |
| [tests/test_phase2e4_4_1_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_1_closure.py) | **12** | 0 | 0 | 1.633s | **PASSED** |
| [tests/test_phase2e4_4_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_4_closure.py) | **12** | 0 | 0 | 1.761s | **PASSED** |
| [tests/test_phase2e4_3_closure.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_3_closure.py) | **7** | 0 | 0 | 3.150s | **PASSED** |
| [tests/test_phase2e4_infrastructure_readiness.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_infrastructure_readiness.py) | **9** | 0 | 0 | 4.825s | **PASSED** |
| [tests/test_phase2e4_1_corrections.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_1_corrections.py) | **9** | 0 | 0 | 2.749s | **PASSED** |
| [tests/test_phase2e4_2_gpu_handoff.py](file:///d:/Project/Capstone_project/tests/test_phase2e4_2_gpu_handoff.py) | **5** | 0 | 0 | 4.084s | **PASSED** |
| **Total Targeted Suites** | **94** | **0** | **0** | **22.234s** | **ALL PASSED** |

### 6.2 Full Repository Unittest Discovery

```bash
.venv\Scripts\python.exe -m unittest discover tests -v
```

- **Total Test Cases Executed:** 265
- **Errors:** 0
- **Failures:** 0
- **Overall Execution Time:** 37.352s
- **Result:** `OK`

---

## 7. Strict Constraint Compliance & Verification

| Constraint | Status | Verification Evidence |
|:---|:---:|:---|
| **No 100 GB Dataset Download** | **COMPLIED** | 0 real S3 downloads performed; local synthetic NetCDF files used. |
| **No Live S3 Inventory or Listing** | **COMPLIED** | Local offline tests only; no AWS network calls or credentials accessed. |
| **No GPU Dependencies or Full Training** | **COMPLIED** | All tests executed on CPU under Python 3.12.9 with synthetic microbatches. |
| **No Model Architecture Redesign** | **COMPLIED** | TC-GraphFormer encoders, cross-attention fusion, and polar sector pooling preserved intact. |
| **Preserve `reports/selection/`** | **COMPLIED** | `reports/selection/` directory untouched, uncommitted, and unaffected. |
| **No Unverified Test Claims** | **COMPLIED** | All 265 tests executed locally and verified via automated task outputs. |
| **Accurate CI Reporting** | **COMPLIED** | Stated clearly that all checks ran locally; no remote GitHub workflow ran. |

---

## 8. Conclusion

Phase 2E.4.4.3 completes the final infrastructure-correction milestones. The codebase enforces sound radar applicability semantics, provides a consistent offline CLI audit tool, removes dead selection stages, ensures mathematically reachable budget attribution reporting, and guarantees symmetric displacement loss masking with verified gradient accumulation.
