# TC-GraphFormer — Machine B Pilot GPU Training, Final Corrections & 100 GB Readiness Report

- **Timestamp (UTC):** `2026-11-23T06:00:00Z`
- **Workspace:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Branch / Commit:** `fix/phase2e4-4-3-final-audit-and-metric-closure` (`24a16e190652368ad0f4b3f29876d579c525cbaf`)
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, Driver `581.42`, PyTorch `2.9.1+cu128`), Intel Core i9-14900K (24C/32T), 128 GB RAM

---

## 1. Priority 1: Evaluation Integrity & Leakage Audit

### 1.1 Storm-Level Partition Isolation (`IO031991`, `IO032023`, and all 12 `TEST` Storms)
- **Disjoint Splits:** Verified in `reports/pipeline/storm_split_manifest.csv` (`80` total storms: `56` `TRAIN`, `12` `VAL`, `12` `TEST`) and `reports/pipeline/sample_index.csv` (`2,599` overpasses: `1,742` `TRAIN`, `612` `VAL`, `245` `TEST`). Pairwise storm intersections across `TRAIN`, `VAL`, and `TEST` are strictly empty ($\emptyset$).
- **`IO031991` & `IO032023` Verification:**
  - `IO031991` (`8` samples, max wind `138.9 kt`) and `IO032023` (`28` samples, max wind `35.0 kt`) belong exclusively to `partition == "TEST"`.
  - In `src/training/finetune_task.py` (`--task nio_adaptation`), North Indian Ocean adaptation explicitly filters:
    - `df_train = df[(df["basin"] == "IO") & (df["partition"] == "TRAIN")]` (`468` samples across `10` `TRAIN` storms)
    - `df_val = df[(df["basin"] == "IO") & (df["partition"] == "VAL")]` (`203` samples across `2` `VAL` storms: `IO011997`, `IO042007`)
  - Neither `IO031991` nor `IO032023` (nor any other `TEST` storm) ever entered pretraining, multi-task training, `intensity_change` fine-tuning, `nio_adaptation` fine-tuning, or normalization statistics (`reports/pipeline/normalization_stats.json`, computed exclusively on `TRAIN`).

### 1.2 Process Contamination Identified & Repaired
1. **Checkpoint Overwrite & Post-Hoc Selection Repair:**
   - *Issue Found:* In the initial pilot run, `src/training/finetune_task.py` called `trainer.save_checkpoint(..., is_best=is_best)`, which overwrote `checkpoints/best_checkpoint.pt` during single-task fine-tuning. Furthermore, the manual recovery (`task-583`) compared `TEST` MAE across saved epochs `[1, 3, 7, 8, 10]` to restore `multitask_epoch_007.pt`.
   - *Fix Applied:* Modified `src/training/finetune_task.py` so `trainer.save_checkpoint(..., is_best=False)` never overwrites `checkpoints/best_checkpoint.pt`, and instead writes task-specific best checkpoints to `checkpoints/finetuned_{task}_best.pt`.
   - *Strict `VAL`-Only Checkpoint Selection:* Retrained 10-epoch multi-task sequence training from `checkpoints/pretrained_backbone.pt`; `TCGraphFormerTrainer` automatically selected **Epoch 6 (`val_loss = 25.4371`)** strictly on `VAL` (`612` samples) as `checkpoints/best_checkpoint.pt`. All held-out `TEST` evaluations below are scored once against this strictly `VAL`-selected checkpoint.
2. **Dataset Partition Logging Fix:**
   - *Issue Found:* `TCPRIMEDDataset` defaulted `self.partition = "TRAIN"` whenever an in-memory DataFrame (`df=df_val`) was passed without an explicit `partition=` argument, causing misleading console logs (`Initialized TCPRIMEDDataset for partition: TRAIN` on `VAL` subsets).
   - *Fix Applied:* `src/data/dataset.py` now infers the true partition label from `df["partition"].unique()` when `partition` is omitted.

---

## 2. Priority 2: Diagnosis & Measured Improvements on Weak Forecasting Tasks

All diagnoses, physical priors, loss weights, and probability thresholds were derived and tuned strictly on `TRAIN` and `VAL`.

### 2.1 Rapid Intensification (`RI`, $\Delta V_{24\text{h}} \ge 30\text{ kt}$)
- **Diagnosis:**
  - Extreme class imbalance (`TRAIN`: `34 / 1,173` valid `24h` samples = `2.90%`; `VAL`: `15 / 428` = `3.50%`; `TEST`: `1 / 113` = `0.88%`).
  - Previously, `pos_weight = 33.53` in `BCEWithLogitsLoss` shifted raw logits by $+\ln(33.53) \approx +3.51$, inflating mean predicted probabilities (`0.1490` on `VAL`, `0.2683` on `TEST`) and degrading Brier calibration (`0.0617` on `VAL`, `0.2261` on `TEST`).
- **Evidence-Based Fix:**
  - Initialized `ri_head` output bias to the `TRAIN` prior log-odds (`-3.5124`) and applied a prior log-odds shift (`prior_shift = 0.85 * ln(pos_weight)`) inside `loss_ri` (`src/training/losses.py`) so `ri_logits` directly output calibrated posterior probabilities while preserving `33.53x` positive-class gradient weighting.
- **Measured `VAL` & Held-Out `TEST` Evidence:**

| Split | Metric | Previous Pilot | Improved (`VAL`-Selected Epoch 6) | Climatology Baseline | Improvement / Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`VAL` (`n=428`, 15 pos)** | **PR-AUC** | `0.0639` | **`0.3223`** | `0.0326` | **`+404%` (`9.89x` over Climatology)** |
| **`VAL` (`n=428`, 15 pos)** | **Brier Score** | `0.0617` | **`0.0293`** | `0.0339` | **`-52.5%` error (Beats Climatology by `-13.6%`)** |
| **`VAL` (`n=428`, 15 pos)** | **F1 (`VAL`-tuned `th=0.09`)** | `0.1463` | **`0.4762`** (`P=0.370, R=0.667`) | `0.0000` | **Detects `10 / 15` `VAL` RI events (`+225%` F1)** |
| **`TEST` (`n=113`, 1 pos)** | **Brier Score** | `0.2261` | **`0.0096`** | `0.0091` | **`23.5x` reduction in Brier calibration error** |
| **`TEST` (`n=113`, 1 pos)** | **Mean Pred Prob vs Base Rate** | `0.2683` vs `0.0088` | **`0.0089` vs `0.0088`** | `0.0290` | **Well-calibrated posterior probability** |

---

### 2.2 Multi-Horizon Track Displacement (`+6h, +12h, +18h, +24h` in km)
- **Diagnosis:**
  - Targets `dx_dt_6h..24h` and `dy_dt_6h..24h` are in **kilometers (km)**, with `TRAIN` 24h mean displacement of `~375 km`.
  - Previously, `track_head` was zero-initialized without observation-time storm translation velocity (`storm_speed_mps`, `storm_heading_deg`) or dynamic range scaling, so predictions remained stuck near `[-35, +45] km` (essentially zero-motion persistence).
- **Evidence-Based Fix:**
  - Extracted causal observation-time `kinematic_state` (`u_kmh, v_kmh, speed, sin/cos heading, sin/cos lat, hemisphere, dist_to_land`) in `TCPRIMEDDataset`.
  - Added a zero-parameter deterministic Fourier kinematic context encoder in `TCGraphFormer` (preserving exact `333` `state_dict` keys) and combined a `TRAIN`-fitted damped kinematic translation prior with `20.0x`-scaled learned synoptic graph residuals in `MultiTaskHead`.
- **Measured `VAL` & Held-Out `TEST` Track MAE (km):**

| Horizon | `VAL` Persistence | `VAL` Climatology | `VAL` Velocity Baseline | **`VAL` TC-GraphFormer (Prev $\rightarrow$ New)** | `TEST` Persistence | `TEST` Climatology | **`TEST` TC-GraphFormer (Prev $\rightarrow$ New)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`+6h`** | `120.31 km` | `104.59 km` | `16.66 km` | `106.03` $\rightarrow$ **`17.94 km`** (`-83.1%`) | `117.53 km` | `97.66 km` | `101.29` $\rightarrow$ **`16.24 km`** (`-84.0%`) |
| **`+12h`** | `231.39 km` | `197.78 km` | `51.39 km` | `203.22` $\rightarrow$ **`51.05 km`** (`-74.9%`, beats Vel) | `239.55 km` | `203.21 km` | `207.98` $\rightarrow$ **`42.31 km`** (`-79.7%`) |
| **`+18h`** | `345.73 km` | `293.89 km` | `106.24 km` | `305.51` $\rightarrow$ **`105.46 km`** (`-65.5%`, beats Vel) | `335.89 km` | `280.46 km` | `290.96` $\rightarrow$ **`89.87 km`** (`-69.1%`) |
| **`+24h`** | `419.04 km` | `345.21 km` | `164.56 km` | `369.50` $\rightarrow$ **`162.50 km`** (`-56.0%`, beats Vel) | `454.32 km` | `377.23 km` | `395.36` $\rightarrow$ **`154.04 km`** (`-61.0%`) |

---

### 2.3 Intense Cyclones (`>= 64 kt`, `>= 96 kt`), Multi-Horizon $\Delta V$, and Southern Hemisphere (`SH`) Wind Error Analysis
- **Diagnosis:**
  - `TRAIN` wind mean is `38.19 kt`, with only `9.9%` of samples $\ge 64\text{ kt}$ (`172 / 1,742`) and `4.1%` $\ge 96\text{ kt}$ (`72 / 1,742`).
  - Zero-initialized `intensity_head` with unweighted Huber `beta=5.0` compressed predictions into `[25, 51] kt`, causing severe under-prediction on `VAL` Major Hurricanes (`>= 96 kt`, $n=22$, true mean `120.6 kt`, previous predicted mean `39.4 kt`, `MAE = 81.22 kt`).
  - In `TEST`, `SH` has `24` samples across two storms (`SH071989` and `SH111989`), of which `10 / 24` (`41.7%`) are Hurricane/Major Hurricane overpasses up to `125.0 kt`, whereas `TRAIN` has **`0` Southern Hemisphere samples above `100 kt`** (and early-1989 `F08` SSM/I swaths lack radar/GPROF/IR coverage).
- **Evidence-Based Fix:**
  - Initialized `intensity_head` (`38.19 kt`), `pressure_head` (`997.25 hPa`), and `dv_head` (`[0.23, 0.58, 1.08, 1.60] kt`) to `TRAIN` priors with `12.0x` (`wind/pressure`) and `5.0x` (`dv`) dynamic output scaling.
  - Applied regime-weighted Huber loss (`beta=8.0`, $w_{\text{regime}} = 1 + 2.5 \cdot \text{clamp}((v_{\text{true}} - 50)/50, 0, 2)^2$) in `src/training/losses.py` to amplify high-intensity cyclone gradients without overfitting noise.
- **Measured `VAL` & Held-Out `TEST` Wind & $\Delta V$ Evidence:**
  - **`VAL` Major Hurricanes (`>= 96 kt`, $n=22$, True Mean `120.6 kt`):** Predicted mean jumped from `39.4 kt` $\rightarrow$ **`94.9 kt`** (max `112.6 kt`); **MAE dropped from `81.22 kt` $\rightarrow$ `25.76 kt` (`-68.3%` error reduction)**.
  - **`VAL` Hurricanes (`64–95 kt`, $n=43$, True Mean `78.1 kt`):** **MAE dropped from `46.81 kt` $\rightarrow$ `37.78 kt` (`-19.3%`)**.
  - **`VAL` Central Pacific (`CP`, $n=86$, 30 hurricane overpasses):** **Wind MAE dropped from `29.28 kt` $\rightarrow$ `12.55 kt` (`-57.1%`)**.
  - **`VAL` Overall Wind ($n=612$):** **MAE dropped from `15.66 kt` $\rightarrow$ `13.67 kt`** (`RMSE = 18.34 kt`), beating Climatology (`15.73 kt`) and Linear Regression (`15.53 kt`).
  - **`VAL` Multi-Horizon $\Delta V$ (`+6h / +12h / +18h / +24h`):** Improved from `4.12 / 7.60 / 10.59 / 13.42 kt` $\rightarrow$ **`4.05 / 7.23 / 9.67 / 12.12 kt`**, beating Persistence (`4.21 / 7.91 / 11.02 / 14.05 kt`) across all 4 horizons (`+1.93 kt` gain at `+24h`).
  - **Held-Out `TEST` Overall Wind ($n=245$):** **MAE improved from `7.47 kt` $\rightarrow$ `6.96 kt`** (`RMSE = 15.63 kt`), beating Climatology (`12.31 kt` by `-43.5%`) and Linear Regression (`12.75 kt` by `-45.4%`).
  - **Held-Out `TEST` Basin Wind MAE:**
    - `CP` ($n=5$): `5.15` $\rightarrow$ **`2.64 kt`** (`-48.7%`)
    - `WP` ($n=88$): `4.00` $\rightarrow$ **`3.22 kt`** (`-19.5%`)
    - `EP` ($n=59$): **`5.08 kt`**
    - `IO` ($n=69$): `7.19` $\rightarrow$ **`6.47 kt`** (`-10.0%`)
    - `SH` ($n=24$): `35.64` $\rightarrow$ **`33.50 kt`** (`-6.0%`; for Category 1–2 `SH` hurricanes `64–95 kt`, predicted mean is **`94.7 kt` vs true `76.8 kt`, `MAE = 24.73 kt`**, confirming the model now predicts hurricane-force winds in `SH`, while the remaining error on 6 Category 3–4 `SH111989` overpasses requires the 100 GB dataset's `26.25%` `SH` allocation).

---

## 3. Priority 3: Bounded Caching Benchmark & 100 GB Acquisition Readiness

### 3.1 Bounded LRU + Reusable Disk Cache Benchmark (`src/data/dataset.py`)
- Replaced unbounded global dict caching with:
  1. **Bounded In-Memory LRU Cache (`_GLOBAL_SAMPLE_CACHE`):** `OrderedDict` strictly capped at `max_cache_entries = 2048` overpasses (`~1.2 GB` tensor payload), evicting oldest samples automatically (`FIFO/LRU`).
  2. **Reusable On-Disk Preprocessing Cache (`data/cache/resampled_64x64`):** Atomic `.pt` storage of `64x64` cKDTree-resampled tensors + `kinematic_state` (`~510 KB/sample`; `~1.3 GB` for pilot, `~17–25 GB` for 100 GB dataset).
  3. **Per-Group Swath Coordinate & `cKDTree` Reuse:** Caches `(lat, lon)` extraction and `cKDTree.query` indices per NetCDF group (`GPROF`, `environmental_synoptic`, `passive_microwave`), accelerating initial cold resampling by `~3.5x`.
- **Measured Throughput & Memory (`2,599` pilot samples):**
  - **Max LRU Size:** Strictly held at **`2,048 / 2,048` entries** (`551` LRU evictions verified during single pass over `2,599` samples).
  - **Peak Process RSS Memory:** **`2,235.7 MB` (`~2.18 GB` out of `128 GB` system RAM)**.
  - **Warm Cache Throughput:** **`2,599` samples loaded in `21.65s` (`120.0 samples/sec`)**, reducing 10-epoch sequence training time to **`~27 seconds/epoch`** (`4.5 minutes` total for 10 epochs).

### 3.2 Verified 100 GB Storm-Stratified Acquisition Plan (`reports/acquisition_100gb/tcprimed_100gb_manifest.csv`)
- Generated using `src/prepare_pilot_dataset.py plan --config configs/acquisition_100gb.yaml --inventory-csv reports/acquisition_100gb/s3_inventory_100gb.csv`:
  - **Total Budget / Selected Volume:** `100.00 GB` budget $\rightarrow$ **`99.9978 GB` (`100.00%` utilization)** across **`13,898` files** (`6,949` microwave/radar swaths + `6,949` paired ERA5 environmental files) from **`109` complete storms** spanning **`25` seasons (`1998–2022`)**.
  - **Radar Priority Enforced:** Selected **`78` TRMM (`TMI/PR`)** and **`34` GPM (`GMI/DPR`)** radar-rich storms from `2,593` eligible radar storms.
  - **Basin Balance:** `WP: 30.25%` (`33` storms), `SH: 26.25%` (`28` storms — directly addressing `SH` intense cyclones), `AL: 17.65%` (`19` storms), `EP: 14.13%` (`16` storms), `IO: 6.86%` (`8` storms), `CP: 4.86%` (`5` storms).

---

## 4. Saved Checkpoints & Reports Inventory

- `checkpoints/pretrained_backbone.pt` — 3-epoch MFM self-supervised backbone (`val_loss = 0.3703`)
- `checkpoints/best_checkpoint.pt` — Strictly `VAL`-selected multi-task checkpoint (**Epoch 6**, `val_loss = 25.4371`)
- `checkpoints/finetuned_intensity_change_best.pt` — Best `intensity_change` fine-tuned checkpoint (`val_loss = 5.8247`)
- `checkpoints/finetuned_nio_adaptation_best.pt` — Best `nio_adaptation` fine-tuned checkpoint (`val_loss = 23.7080`)
- `reports/phase2e4/baseline_comparison.csv` — Leak-free `VAL` & `TEST` baseline comparison table
- `reports/phase2e4/task_metrics_by_horizon.csv` — Multi-horizon track and $\Delta V$ metrics (`+6h, +12h, +18h, +24h`)
- `reports/phase2e4/basin_evaluation.csv` — Per-basin evaluation breakdown (`AL, CP, EP, IO, SH, WP`)
- `reports/phase2e4/evaluation_summary.json` — Complete JSON evaluation summary
- `reports/phase2e4/held_out_forecast_examples.json` — Held-out `TEST` forecast vs. ground-truth examples
- `reports/acquisition_100gb/tcprimed_100gb_manifest.csv` — Deterministic 100 GB S3 acquisition manifest (`13,898` files, `109` storms)
