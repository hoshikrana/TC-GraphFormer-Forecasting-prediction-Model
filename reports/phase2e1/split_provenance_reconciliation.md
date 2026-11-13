# TC PRIMED Split Provenance & Sequence Padding Reconciliation Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Storm Split Distribution Reconciliation

In Phase 2D and 2E documentation, an apparent contradiction emerged regarding the storm-level split counts across the 80 pilot storms:
- **Phase 2E Implementation Report Statement:** Stated in prose: *"56 TRAIN, 16 VAL, 8 TEST storms"*.
- **Canonical Manifest Artifact:** `reports/pipeline/storm_split_manifest.csv` and `sample_index.csv`.

### Forensic Audit of Real Split Manifest
An inspection of the canonical source file `reports/pipeline/storm_split_manifest.csv` reveals the ground truth:

```python
import pandas as pd
df = pd.read_csv("reports/pipeline/storm_split_manifest.csv")
print(df["partition"].value_counts())
# TRAIN: 56 storms
# VAL:   12 storms
# TEST:  12 storms
# Total: 80 storms
```

| Partition | Storm Count | Fraction of Storms | Overpass Samples | Fraction of Samples | Basins Represented |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TRAIN** | **56** | $70.0\%$ | **1,742** | $67.0\%$ | AL, EP, WP, CP, IO, SH |
| **VAL** | **12** | $15.0\%$ | **612** | $23.5\%$ | AL, EP, WP, CP, IO, SH |
| **TEST** | **12** | $15.0\%$ | **245** | $9.5\%$ | AL, EP, WP, CP, IO, SH |
| **TOTAL** | **80** | $100.0\%$ | **2,599** | $100.0\%$ | Global 6 Basins |

### Root Cause
The manifest generated during Phase 2A and maintained throughout Phase 2B, 2C, 2D, and 2E was generated using a target split ratio of **70% / 15% / 15%** ($56 / 12 / 12$).
The statement *"56 / 16 / 8"* in Phase 2E prose was an errant typographic error during documentation drafting, whereas all data pipelines, loaders, and tests executed on the canonical **56 / 12 / 12** split.

**Reconciliation Ruling:** The official, canonical storm split is **56 TRAIN / 12 VAL / 12 TEST** (1,742 / 612 / 245 samples). Unit test `test_10_storm_split_and_sequence_padding_reconciliation` asserts these exact figures.

---

## 2. Sequence Padding Alignment Reconciliation

In causal sequence modeling, padding occurs when historical observation sequences have variable lengths ($L_i \le T_{\max}$). A conflict existed between code behavior and documentation:
- **Documentation Claim:** Phase 2E reported *"front padding with causal masking"*.
- **Actual Code Behavior (`src/data/collate.py:collate_sequence_batch`):**
  ```python
  # Code implementation:
  for b, seq in enumerate(batch):
      L = len(seq)
      for t in range(L):
          ir_seq[b, t] = seq[t]["ir"]
          modality_masks_seq[b, t] = seq[t]["modality_masks"]
      # Padded positions are at t = L .. T_max - 1
  ```

### Forensic Analysis
In `collate_sequence_batch`:
- For a sequence of length $L$, the real observations are placed at chronological timesteps $t = 0, \dots, L-1$.
- Unused timesteps from $t = L, \dots, T_{\max}-1$ are preallocated with zeros.
- The sequence padding mask marks:
  $$\text{seq\_pad\_mask}[b, t] = \begin{cases} \text{False} & \text{if } t < L \text{ (valid observation)} \\ \text{True} & \text{if } t \ge L \text{ (padded timestep)} \end{cases}$$
- This is **END PADDING** (or right padding), not front padding.
- In `CausalTemporalTransformer`, attention is computed with:
  1. A causal lower-triangular mask ($M_{ij} = -\infty$ for $j > i$) preventing attention into future timesteps.
  2. A key padding mask ($K_{ij} = -\infty$ for $j \ge L$) preventing attention into padded end-timesteps.

**Reconciliation Ruling:** `collate_sequence_batch` uses **end-padding** combined with causal attention masking. The erroneous "front padding" terminology in Phase 2E reports is formally superseded by this report.
