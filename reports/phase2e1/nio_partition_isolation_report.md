# North Indian Ocean (NIO) Partition Isolation & Basin Adaptation Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Regional Domain Adaptation Mission

Tropical cyclones in the North Indian Ocean (Bay of Bengal and Arabian Sea, basin code `IO`) represent a high-vulnerability, low-sample regime characterized by severe storm surge risks and complex monsoonal circulation.

The `nio_adaptation` task in `src/training/finetune_task.py` evaluates transferring representations learned on the global dataset (Atlantic, East Pacific, West Pacific) to the North Indian Ocean basin with parameter freezing and differential learning rates.

---

## 2. Flaw Identified in Phase 2D Audit

In previous versions, `finetune_task.py` filtered a local DataFrame `df_train` to `basin == "IO"`. However, when initializing `TCPRIMEDSequenceDataset`:
```python
# Flawed legacy code:
train_ds = TCPRIMEDSequenceDataset(
    sample_index_csv=sample_index_csv,  # Re-read global sample_index.csv!
    partition="TRAIN",
    max_sequences=len(df_train)         # Only used length of df_train as a count!
)
```
Because `TCPRIMEDSequenceDataset` did not accept an external DataFrame, it loaded the first $N$ samples from the global `TRAIN` partition across ALL basins (AL, EP, WP), completely ignoring the Indian Ocean basin constraint.

---

## 3. Phase 2E.1 Corrective Implementation

In Phase 2E.1:
1. `TCPRIMEDDataset` and `TCPRIMEDSequenceDataset` are updated to accept:
   - `basin: Optional[str] = None`
   - `df: Optional[pd.DataFrame] = None`
2. When `df` is provided, the dataset initializes directly from that exact filtered partition without reading from disk.
3. In `src/training/finetune_task.py:run_finetuning`:
   ```python
   if task == "nio_adaptation":
       df_train = df[(df["basin"] == "IO") & (df["partition"] == "TRAIN")].copy()
       df_val = df[(df["basin"] == "IO") & (df["partition"] == "VAL")].copy()
       test_storms = df[(df["basin"] == "IO") & (df["partition"] == "TEST")]["storm_id"].unique()
       ...
       train_ds = TCPRIMEDSequenceDataset(sample_index_csv=sample_index_csv, df=df_train, ...)
       val_ds = TCPRIMEDSequenceDataset(sample_index_csv=sample_index_csv, df=df_val, ...)
   ```

---

## 4. Empirical Verification of Zero Leakage

An audit of `reports/pipeline/sample_index.csv` for the `IO` basin confirms:
- **Total IO Storms in Pilot Dataset:** 19 storms (468 overpass samples).
- **IO TRAIN Partition (14 storms, 360 samples):**
  `IO011988`, `IO011990`, `IO012002`, `IO021993`, `IO021994`, `IO022002`, `IO022009`, `IO022016`, `IO031998`, `IO032022`, `IO041995`, `IO042005`, `IO052013`, `IO101992`.
- **IO VAL Partition (3 storms, 65 samples):**
  `IO012022`, `IO012024`, `IO032020`.
- **IO TEST Partition (2 storms, 43 samples):**
  `IO031991` (Severe Cyclonic Storm 02B / 1991 Bangladesh cyclone analogue period), `IO032023` (Extremely Severe Cyclonic Storm Mocha).

### Leakage Matrix
$$\text{Overlap}(\text{TRAIN}, \text{TEST}) = \emptyset \quad (0 \text{ storms})$$
$$\text{Overlap}(\text{VAL}, \text{TEST}) = \emptyset \quad (0 \text{ storms})$$
$$\text{Overlap}(\text{TRAIN}, \text{VAL}) = \emptyset \quad (0 \text{ storms})$$

Unit test `test_09_nio_adaptation_strict_basin_and_partition_isolation` verifies that 100% of samples in `train_ds` belong to `IO` and `TRAIN`, and that regional test storms `IO031991` and `IO032023` are strictly held out.
