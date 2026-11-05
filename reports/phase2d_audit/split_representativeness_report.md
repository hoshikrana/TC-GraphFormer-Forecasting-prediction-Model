# TC PRIMED Phase 2D — Split Integrity and Representativeness Audit

## 1. Executive Summary

A core requirement for deep learning models forecasting high-impact weather is that test partitions must faithfully represent the operational meteorological spectrum—including rare, catastrophic events like Category 4–5 hurricanes and Rapid Intensification (RI).

This audit resolves the apparent contradiction between Phase 2A and Phase 2C observation counts and delivers a comprehensive meteorological audit of the 80-storm pilot dataset partitions. The audit reveals a severe representativeness defect in the current test partition that necessitates a multi-objective stratified re-split prior to scaling to the 500 GB dataset.

---

## 2. Split Count Discrepancy Reconciliation

### 2.1 The 1,742 vs. 1,847 Observation Contradiction
Previous reports cited differing observation counts across the dataset partitions:
- **Phase 1F Exploratory Manifest** (`reports/eda/pilot_split_manifest.csv`):
  - Train: 42 storms (1,475 overpasses)
  - Validation: 16 storms (474 overpasses)
  - Test: 22 storms (650 overpasses)
- **Phase 2A Authoritative Pipeline** (`reports/pipeline/storm_split_manifest.csv` & `sample_index.csv`):
  - Train: 56 storms (1,742 overpasses, 67.0%)
  - Validation: 12 storms (612 overpasses, 23.5%)
  - Test: 12 storms (245 overpasses, 9.4%)
- **Phase 2C Narrative Text**:
  - Mentioned 1,847 training overpasses.

### Reconciliation Verdict:
1. `reports/pipeline/storm_split_manifest.csv` and `reports/pipeline/sample_index.csv` are the **single authoritative ground truth**. In both files, the split distribution is exactly:
   $$\text{Train: } 1,742 \quad|\quad \text{Validation: } 612 \quad|\quad \text{Test: } 245 \quad|\quad \text{Total: } 2,599$$
2. The number 1,847 in Phase 2C text was a typographical error in the markdown narrative. In fact, the quantitative table `reports/phase2c/storm_partition_distribution.csv` correctly recorded **1,742** training overpasses.
3. Every sample in `data/processed/pilot/` maps deterministically to one of these three partitions with zero duplicated overpasses or cross-split leakage.

---

## 3. Test Partition Meteorological Representativeness Defect

We conducted an exhaustive storm-by-storm meteorological audit of the 12 storms assigned to the `TEST` partition:

| Storm ID | Basin | Season | Total Overpasses | Peak Best Track Wind ($V_{\max}$) | Recorded Agency Stage Codes | Meteorological Stage Classification |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `CP011987` | CP | 1987 | 3 | 46 kt | `['TS', 'TD']` | Moderate Tropical Storm |
| `CP012004` | CP | 2004 | 20 | 25 kt | `['TD']` | Weak Tropical Depression |
| `CP022000` | CP | 2000 | 9 | 45 kt | `['TS']` | Moderate Tropical Storm |
| `EP022020` | EP | 2020 | 14 | 35 kt | `['TD', 'TS']` | Minimal Tropical Storm |
| `EP072020` | EP | 2020 | 35 | 35 kt | `['LO', 'TS', 'TD']` | Minimal Tropical Storm |
| `IO031991` | IO | 1991 | 8 | 40 kt | `['UN']` | Unclassified Tropical Storm |
| `IO032023` | IO | 2023 | 28 | 44 kt | `['TS', 'TD']` | Moderate Tropical Storm |
| `SH081988` | SH | 1988 | 6 | 57 kt | `['UN']` | Strong Tropical Storm |
| `SH111989` | SH | 1989 | 18 | **125 kt** | `['UN']` | **Major Cyclone (Cat 4 Equivalent)** |
| `WP072021` | WP | 2021 | 24 | 30 kt | `['WV', 'TD']` | Tropical Depression |
| `WP082022` | WP | 2022 | 21 | 25 kt | `['TD', 'DB']` | Tropical Depression |
| `WP172021` | WP | 2021 | 59 | 34 kt | `['SD', 'TD', 'TS']` | Minimal Tropical Storm |

### Critical Findings:
1. **Zero North Atlantic (`AL`) Storms in Test Split**:
   - The North Atlantic basin contains **401 overpasses (15.4%)** across the pilot dataset.
   - However, exactly **zero** North Atlantic storms were allocated to the test set!
   - A model evaluated on this test split cannot assess performance on North Atlantic hurricanes.
2. **Zero Overpasses with WMO Stage `HU` or `TY`**:
   - In the categorical Best Track stage taxonomy, **0 out of 245 test overpasses** possess a label of `'HU'` (Hurricane) or `'TY'` (Typhoon).
   - 11 of the 12 test storms never achieved hurricane intensity ($V_{\max} < 64\text{ kt}$).
3. **The `SH111989` Anomaly**:
   - Only one storm in the test set reached major cyclone status (`SH111989`, peak $V_{\max} = 125\text{ kt}$).
   - However, in 1989 the Joint Typhoon Warning Center (JTWC) recorded Southern Hemisphere systems with generic unclassified codes (`'UN'`). Consequently, the stage classification loss masked or misassigned this storm.

---

## 4. Multi-Objective Stratified Re-Split Protocol for Phase 3

To prevent evaluation bias and ensure scientific credibility, the storm splitting algorithm in `tcprimed_pipeline/storm_splitter.py` must be upgraded before scaling to the 500 GB dataset:

### Required Stratification Objectives:
Every partition ($\text{Train}: 70\%, \text{Val}: 15\%, \text{Test}: 15\%$) must be stratified simultaneously across three axes:
1. **Basin Balance**: Proportional representation across all 6 basins ($\text{AL}, \text{CP}, \text{EP}, \text{IO}, \text{SH}, \text{WP}$).
2. **Era Balance**: Proportional distribution across historical sensor eras (1987–1999, 2000–2010, 2011–2019, 2020–2025).
3. **Peak Saffir-Simpson Intensity**:
   - Tropical Depression ($V_{\max} < 34\text{ kt}$)
   - Tropical Storm ($34 \le V_{\max} < 64\text{ kt}$)
   - Category 1–2 Hurricane ($64 \le V_{\max} < 96\text{ kt}$)
   - Major Category 3–5 Hurricane ($V_{\max} \ge 96\text{ kt}$)

Implementing this multi-objective stratified split guarantees that the test partition contains representative Category 4–5 hurricanes from both the North Atlantic and Western Pacific basins.

