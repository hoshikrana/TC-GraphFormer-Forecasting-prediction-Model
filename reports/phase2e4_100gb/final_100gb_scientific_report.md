# TC-GraphFormer — 100 GB Production Dataset Acquisition, Training & Final Scientific Evaluation Report

- **Timestamp (UTC):** `2026-11-24T01:45:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified` (`https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`)
- **Git Branch:** `feat/phase2e4-pilot-corrections-and-100gb-readiness` (branched from baseline `24a16e190652368ad0f4b3f29876d579c525cbaf`)
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, Driver `581.42`, PyTorch `2.9.1+cu128`), Intel Core i9-14900K (24C/32T), `128 GB` RAM

---

## 1. Dataset Manifest, Acquisition & Split Provenance

### 1.1 Deterministic 100 GB Acquisition Manifest
- **Manifest File:** `reports/acquisition_100gb/tcprimed_100gb_manifest.csv`
- **Manifest SHA-256:** `956fe84ce8d96c3280d43d3eb11388c1390735bbaf3e5912e7918d83295b382f`
- **Total Files Downloaded & Verified:** **`13,898 / 13,898` NetCDF files** (`13,789` satellite overpasses + `109` storm-lifecycle ERA5 environmental files; `0` missing, `0` incomplete, `0` failed)
- **Exact Raw Dataset Size:** **`99,997,769,720` bytes** (`99.997770 GB` decimal / `93.130180 GiB`, `100.00%` of the 100 GB budget)
- **Processed Container Footprint (`data/processed/tcprimed_100gb/`):** **`109` per-storm NetCDF-4/HDF5 containers** (`31.46 GiB`, `2.96x` compression ratio)
- **Total Indexed Overpass Samples:** **`13,789` observations** across **`109` complete storms** (`25` distinct seasons, `1997–2025`, all `6` global basins)

### 1.2 Storm-Level Partitioning & Prior Pilot Isolation (`reports/pipeline_100gb/storm_split_manifest.csv`)
- **Disjoint Storm-Level Splits:** $\text{TRAIN} \cap \text{VAL} = \emptyset$, $\text{TRAIN} \cap \text{TEST} = \emptyset$, $\text{VAL} \cap \text{TEST} = \emptyset$.
- **Prior Pilot Storm Isolation:** All `5` overlapping Pilot `TRAIN` storms (`CP022025, CP042006, IO031998, IO032022, WP032000`) are retained in `TRAIN`; the `1` overlapping Pilot `TEST` storm (`WP072021`) is isolated in `VAL`; **`0` prior pilot storms appear in `FINAL_TEST` (`TEST`)**.

| Basin | `TRAIN` Storms (`Samples`) | `VAL` Storms (`Samples`) | `FINAL_TEST` Storms (`Samples`) | Total Storms (`Samples`) | Manifest Byte Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`AL` (North Atlantic)** | `12` (`2,649`) | `3` (`644`) | `3` (`176`) | `18` (`3,469`) | `24.93%` |
| **`CP` (Central Pacific)** | `3` (`228`) | `1` (`95`) | `1` (`12`) | `5` (`335`) | `2.35%` |
| **`EP` (Eastern Pacific)** | `16` (`2,147`) | `3` (`420`) | `3` (`241`) | `22` (`2,808`) | `19.97%` |
| **`IO` (North Indian Ocean)** | `18` (`1,471`) | `4` (`207`) | `4` (`386`) | `26` (`2,064`) | `14.85%` |
| **`SH` (Southern Hemisphere)** | `5` (`1,049`) | `1` (`26`) | `1` (`28`) | `7` (`1,103`) | `7.96%` |
| **`WP` (Western Pacific)** | `21` (`3,001`) | `5` (`423`) | `5` (`586`) | `31` (`4,010`) | `29.94%` |
| **Total** | **`75` (`10,545`)** | **`17` (`1,815`)** | **`17` (`1,429`)** | **`109` (`13,789`)** | **`100.00%`** |

### 1.3 Candidate vs. Confirmed Radar & Modality Availability (`reports/pipeline_100gb/modality_availability.csv`)
- **Candidate Radar-Capable Coverage (Manifest S3 Metadata):** `78` `TMI/TRMM` storms (`1,306` files) and `34` `GMI/GPM` storms (`397` files), with `3` overlap storms (`EP072014, EP102014, WP112014`) observed by both satellites ($\text{TRMM} \cup \text{GPM} = 109 / 109$ unique candidate radar-capable storms, `1,703` candidate overpass files).
- **Confirmed Valid Observations in Processed Containers (`13,789` overpasses):**
  - **Passive Microwave (`has_pmw`):** **`13,789 / 13,789` (`100.00%`)** across `109 / 109` storms
  - **GPROF Precipitation (`has_gprof`):** **`13,761 / 13,789` (`99.80%`)** across `109 / 109` storms
  - **Causal ERA5 Synoptic Environment (`has_causal_env`, $t_{\text{env}} \le t_{\text{obs}}$):** **`13,644 / 13,789` (`98.95%`)** across `109 / 109` storms
  - **Infrared (`has_ir`):** **`12,725 / 13,789` (`92.28%`)** across `108 / 109` storms
  - **Confirmed 3D Radar Reflectivity (`has_radar`):** **`1,665 / 13,789` (`12.07%`)** across **`109 / 109` storms** (`97.77%` of the `1,703` candidate `TMI/GMI` overpasses contain confirmed finite Ku-band radar profiles)

---

## 2. Caching Throughput, Issue-Time Safety & Training Runtime

- **Forecast Issue-Time Safety (`v3_strict_causal_kin_cappi16_grid64`):**
  - Eliminated IBTrACS centered-difference `storm_speed_mps` / `storm_heading_deg` from model inputs.
  - `kinematic_state` computes translation velocity `(causal_u_kmh, causal_v_kmh)` exclusively via backward finite differences across historical fixes at $t_{\text{prev}} \le t_{\text{obs}}$ (`2.0h <= t_obs - t_prev <= 24.0h`).
- **Cache & DataLoader Benchmark (`13,789` samples):**
  - **Parallel Cold Cache Build (`12` CPU workers):** `13,789` samples resampled and written atomically to `data/cache/resampled_64x64/` in `3,052.23s` (`4.5 samples/sec`).
  - **Warm `DataLoader` Throughput (`batch_size=32`):** **`13,789` samples in `119.63s` (`115.3 samples/sec`)**.
  - **Bounded In-Memory LRU & RAM:** Strictly capped at **`2,048 / 2,048` entries**, peak process RSS **`2,122.3 MB` (`~2.07 GB`)**.
- **Training Configuration & Runtime (`configs/gpu_node_100gb.yaml`):**
  - **Self-Supervised MFM Pretraining (`TRAIN` only, `10,545` samples):** `3` epochs, `batch_size=32`, `lr=3e-4`, `mask_ratio=0.30`. MFM MSE loss converged from `0.59527` (Epoch 1) $\rightarrow$ `0.35587` (Epoch 2) $\rightarrow$ **`0.28097` (Epoch 3)** across `618,585` valid tokens (`184,049` masked). Saved to `checkpoints/tcprimed_100gb/pretrained_backbone.pt`.
  - **Supervised Chronological Sequence Multi-Task Training (`10,545` `TRAIN` sequences, `1,815` `VAL` sequences):** `12` epochs, `max_history_len=4`, `max_history_hours=72.0`, `batch_size=32`, `accumulate_grad_batches=2` (effective batch size `64`), `lr=3e-4`, `weight_decay=1e-4`, `precision=fp16` (`AMP`), `ri_pos_weight=18.10` (`TRAIN`-only `9,016 neg / 498 pos`), `seed=42`.
  - **Measured Epoch Runtime:** **`444.1s – 449.8s` (`~7.45 minutes`) per epoch** (`~23.6` 4-step sequences/sec = `~94.4` overpass observations/sec).
  - **Strict `VAL`-Only Checkpoint Selection:** `TCGraphFormerTrainer` selected **Epoch 5 (`val_loss = 16.7881`)** strictly on `VAL` as `checkpoints/tcprimed_100gb/best_checkpoint.pt`.
  - **Isolated Task Fine-Tuning (`VAL`-monitored):**
    - `checkpoints/tcprimed_100gb/finetuned_intensity_change_best.pt`: `val_loss` improved from `5.1948` $\rightarrow$ **`5.1353`** (Epoch 3).
    - `checkpoints/tcprimed_100gb/finetuned_nio_adaptation_best.pt`: `val_loss = 18.5643` (Epoch 1, strictly holding out NIO `TEST` storms `IO011998, IO012008, IO021999, IO042019`).

---

## 3. Final Scientific Evaluation Results (`VAL` and Pristine `FINAL_TEST`)

All results below are evaluated on the frozen `VAL`-selected checkpoint (`checkpoints/tcprimed_100gb/best_checkpoint.pt`, Epoch 5).

### 3.1 Current-State Estimation Tasks (Separated from Future Forecasting)

| Partition | Task | Model / Baseline | Valid $N$ | MAE | RMSE | Bias | Units / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`VAL`** | **Current Intensity ($V_{\max}$)** | Climatology (Train Mean) | `1,815` | `24.71` | `32.28` | `+0.86` | `kt` |
| **`VAL`** | **Current Intensity ($V_{\max}$)** | Linear Regression | `1,815` | `24.80` | `32.16` | `+2.39` | `kt` |
| **`VAL`** | **Current Intensity ($V_{\max}$)** | **TC-GraphFormer** | `1,815` | **`7.84`** | **`10.52`** | `+1.36` | `kt` (**`-68.3%` vs Climatology**) |
| **`VAL`** | **Central Pressure ($P_{\min}$)** | Climatology (Train Mean) | `1,604` | `18.92` | `26.10` | `+0.47` | `hPa` |
| **`VAL`** | **Central Pressure ($P_{\min}$)** | Linear Pressure-Wind Fit | `1,604` | `5.45` | `7.23` | `+0.67` | `hPa` (uses ground-truth $V_{\max}$) |
| **`VAL`** | **Central Pressure ($P_{\min}$)** | **TC-GraphFormer** | `1,604` | **`5.99`** | **`8.29`** | `+1.54` | `hPa` (**`-68.3%` vs Climatology**) |
| **`VAL`** | **Development Stage (5-class)** | Majority Class (`TS`) | `1,549` | `78.05` | — | — | Error % (`Accuracy: 21.95%`) |
| **`VAL`** | **Development Stage (5-class)** | **TC-GraphFormer** | `1,549` | **`34.86`** | — | — | Error % (**`Accuracy: 65.14%`, `+43.19%` gain**) |
| **`FINAL_TEST`** | **Current Intensity ($V_{\max}$)** | Climatology (Train Mean) | `1,429` | `28.30` | `32.75` | `+7.08` | `kt` |
| **`FINAL_TEST`** | **Current Intensity ($V_{\max}$)** | Linear Regression | `1,429` | `29.08` | `33.40` | `+7.10` | `kt` |
| **`FINAL_TEST`** | **Current Intensity ($V_{\max}$)** | **TC-GraphFormer** | `1,429` | **`9.93`** | **`13.57`** | `+4.31` | `kt` (**`-64.9%` vs Climatology**) |
| **`FINAL_TEST`** | **Central Pressure ($P_{\min}$)** | Climatology (Train Mean) | `1,173` | `20.01` | `23.37` | `-4.01` | `hPa` |
| **`FINAL_TEST`** | **Central Pressure ($P_{\min}$)** | Linear Pressure-Wind Fit | `1,173` | `6.20` | `7.15` | `+0.54` | `hPa` (uses ground-truth $V_{\max}$) |
| **`FINAL_TEST`** | **Central Pressure ($P_{\min}$)** | **TC-GraphFormer** | `1,173` | **`5.64`** | **`8.07`** | `-1.20` | `hPa` (**Beats Climatology & Linear P-W Fit**) |
| **`FINAL_TEST`** | **Development Stage (5-class)** | Majority Class (`TS`) | `1,170` | `73.08` | — | — | Error % (`Accuracy: 26.92%`) |
| **`FINAL_TEST`** | **Development Stage (5-class)** | **TC-GraphFormer** | `1,170` | **`42.39`** | — | — | Error % (**`Accuracy: 57.61%`, `+30.69%` gain**) |

---

### 3.2 Future Multi-Horizon Intensity Change Forecasting ($\Delta V$, `+6h, +12h, +18h, +24h` in kt)

| Partition | Horizon | Valid $N$ | Persistence MAE (`RMSE`) | Climatology MAE (`RMSE`) | **TC-GraphFormer MAE (`RMSE`)** | **Skill vs. Persistence** | **Skill vs. Climatology** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VAL`** | **`+6h`** | `1,722` | `3.49` (`5.67`) | `3.50` (`5.67`) | **`3.22` (`4.75`)** | **`+0.27 kt` (`-16.2%` RMSE)** | **`+0.28 kt`** |
| **`VAL`** | **`+12h`** | `1,671` | `6.80` (`10.39`) | `6.81` (`10.39`) | **`5.96` (`8.38`)** | **`+0.84 kt` (`-19.3%` RMSE)** | **`+0.85 kt`** |
| **`VAL`** | **`+18h`** | `1,627` | `9.76` (`14.30`) | `9.77` (`14.30`) | **`8.27` (`11.28`)** | **`+1.49 kt` (`-21.1%` RMSE)** | **`+1.50 kt`** |
| **`VAL`** | **`+24h`** | `1,576` | `12.38` (`17.85`) | `12.39` (`17.83`) | **`10.18` (`13.64`)** | **`+2.20 kt` (`-23.6%` RMSE)** | **`+2.21 kt`** |
| **`FINAL_TEST`** | **`+6h`** | `1,352` | `3.58` (`5.63`) | `3.59` (`5.63`) | **`3.02` (`4.27`)** | **`+0.56 kt` (`-24.2%` RMSE)** | **`+0.57 kt`** |
| **`FINAL_TEST`** | **`+12h`** | `1,312` | `7.00` (`10.56`) | `7.01` (`10.56`) | **`5.62` (`8.02`)** | **`+1.38 kt` (`-24.1%` RMSE)** | **`+1.39 kt`** |
| **`FINAL_TEST`** | **`+18h`** | `1,266` | `10.17` (`14.94`) | `10.19` (`14.94`) | **`7.96` (`11.17`)** | **`+2.21 kt` (`-25.2%` RMSE)** | **`+2.23 kt`** |
| **`FINAL_TEST`** | **`+24h`** | `1,227` | `13.32` (`19.19`) | `13.36` (`19.18`) | **`10.02` (`14.13`)** | **`+3.30 kt` (`-26.4%` RMSE)** | **`+3.34 kt`** |

---

### 3.3 Rapid Intensification Forecasting (`RI`, $\Delta V_{24\text{h}} \ge 30\text{ kt}$)

- **`VAL`-Calibrated Decision Threshold:** `th = 0.25` (selected strictly on `VAL` and applied frozen to `FINAL_TEST`).

| Partition | Model | Valid $N$ | Positive / Negative Events | PR-AUC | Brier Score | Precision (`@0.25`) | Recall (`@0.25`) | F1 (`@0.25`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VAL`** | Training Climatology (`0.052`) | `1,576` | `92` pos / `1,484` neg (`5.84%`) | `0.0436` | `0.0550` | `0.0000` | `0.0000` | `0.0000` |
| **`VAL`** | **TC-GraphFormer** | `1,576` | `92` pos / `1,484` neg (`5.84%`) | **`0.5017`** (`11.5x`) | **`0.0386`** (`-29.8%`) | **`0.4907`** | **`0.5761`** (`53/92`) | **`0.5300`** |
| **`FINAL_TEST`** | Training Climatology (`0.052`) | `1,227` | `87` pos / `1,140` neg (`7.09%`) | `0.0956` | `0.0662` | `0.0000` | `0.0000` | `0.0000` |
| **`FINAL_TEST`** | **TC-GraphFormer** | `1,227` | `87` pos / `1,140` neg (`7.09%`) | **`0.4152`** (`4.34x`) | **`0.0512`** (`-22.7%`) | **`0.4132`** | **`0.5747`** (`50/87`) | **`0.4808`** |

---

### 3.4 Multi-Horizon Track Displacement Forecasting (`+6h, +12h, +18h, +24h` in km)

Compared against **Zero-Motion Persistence**, **Training Climatology**, and **Strictly Causal Observation-Time Velocity Extrapolation** ($t_{\text{prev}} \le t_{\text{obs}}$):

| Partition | Horizon | Valid $N$ | Persistence MAE (`RMSE`) | Climatology MAE (`RMSE`) | Causal Velocity MAE (`RMSE`) | **TC-GraphFormer MAE (`RMSE`)** | **Skill vs. Persistence** | **Skill vs. Causal Velocity** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VAL`** | **`+6h`** | `290` | `121.18` (`141.92`) | `106.26` (`133.95`) | `34.70` (`44.77`) | **`35.72` (`44.97`)** | `+85.46 km` | `-1.02 km` |
| **`VAL`** | **`+12h`** | `287` | `241.77` (`284.65`) | `212.07` (`269.32`) | `84.18` (`105.54`) | **`85.89` (`106.36`)** | `+155.88 km` | `-1.71 km` |
| **`VAL`** | **`+18h`** | `276` | `358.74` (`420.42`) | `314.54` (`396.33`) | `144.98` (`175.06`) | **`147.15` (`178.18`)** | `+211.59 km` | `-2.17 km` |
| **`VAL`** | **`+24h`** | `270` | `477.87` (`564.38`) | `418.76` (`533.75`) | `213.15` (`257.63`) | **`213.65` (`260.55`)** | `+264.22 km` | `-0.50 km` |
| **`FINAL_TEST`** | **`+6h`** | `238` | `110.62` (`128.32`) | `97.19` (`115.47`) | `33.89` (`45.13`) | **`33.99` (`44.99`)** | **`+76.63 km`** | `-0.10 km` (**`+0.14 km` RMSE**) |
| **`FINAL_TEST`** | **`+12h`** | `231` | `220.40` (`257.17`) | `192.26` (`231.80`) | `79.45` (`109.43`) | **`78.57` (`108.39`)** | **`+141.83 km`** | **`+0.88 km` (`+1.04 km` RMSE)** |
| **`FINAL_TEST`** | **`+18h`** | `225` | `322.63` (`365.45`) | `280.10` (`325.56`) | `132.28` (`168.72`) | **`130.90` (`166.12`)** | **`+191.73 km`** | **`+1.38 km` (`+2.60 km` RMSE)** |
| **`FINAL_TEST`** | **`+24h`** | `219` | `430.20` (`489.04`) | `371.03` (`436.04`) | `194.06` (`245.82`) | **`187.73` (`238.78`)** | **`+242.47 km`** | **`+6.33 km` (`+7.04 km` RMSE)** |

---

### 3.5 Strong-Cyclone Regime & Per-Basin Performance

#### A. Wind Speed Error by Cyclone Intensity Regime

| Partition | Intensity Regime | Valid $N$ | True Mean (kt) | Predicted Mean (kt) | **MAE (kt)** | **RMSE (kt)** | **Bias (kt)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`VAL`** | Tropical Depression / Storm (`< 64 kt`) | `1,340` | `38.54` | `40.44` | **`7.51`** | `10.21` | `+1.90` |
| **`VAL`** | Category 1–2 Hurricane (`64–95 kt`) | `230` | `74.63` | `75.45` | **`9.83`** | `12.76` | `+0.81` |
| **`VAL`** | **Major Hurricane ($\ge 96\text{ kt}$)** | `245` | `123.33` | `122.25` | **`7.78`** | **`9.85`** | **`-1.08`** |
| **`VAL`** | All Hurricanes ($\ge 64\text{ kt}$) | `475` | `99.75` | `99.59` | **`8.77`** | `11.35` | `-0.16` |
| **`FINAL_TEST`** | Tropical Depression / Storm (`< 64 kt`) | `1,032` | `31.15` | `34.59` | **`9.40`** | `12.98` | `+3.44` |
| **`FINAL_TEST`** | Category 1–2 Hurricane (`64–95 kt`) | `233` | `78.09` | `86.95` | **`14.21`** | `18.10` | `+8.86` |
| **`FINAL_TEST`** | **Major Hurricane ($\ge 96\text{ kt}$)** | `164` | `114.25` | `117.54` | **`7.16`** | **`8.88`** | **`+3.29`** |
| **`FINAL_TEST`** | All Hurricanes ($\ge 64\text{ kt}$) | `397` | `93.03` | `99.59` | **`11.30`** | `14.99` | `+6.56` |

#### B. Per-Basin Wind Speed MAE (`reports/phase2e4_100gb/basin_evaluation.csv`)

| Basin | `VAL` Samples | `VAL` Mean Wind (kt) | **`VAL` Wind MAE (kt)** | `FINAL_TEST` Samples | `FINAL_TEST` Mean Wind (kt) | **`FINAL_TEST` Wind MAE (kt)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AL` (North Atlantic)** | `644` | `54.05` | **`8.56 kt`** | `176` | `55.05` | **`11.01 kt`** |
| **`CP` (Central Pacific)** | `95` | `31.34` | **`6.35 kt`** | `12` | `26.25` | **`1.41 kt`** |
| **`EP` (Eastern Pacific)** | `420` | `51.21` | **`6.20 kt`** | `241` | `54.88` | **`7.35 kt`** |
| **`IO` (North Indian Ocean)** | `207` | `34.65` | **`7.50 kt`** | `386` | `58.17` | **`8.66 kt`** |
| **`SH` (Southern Hemisphere)** | `26` | `42.38` | **`23.67 kt`** | `28` | `30.00` | **`4.98 kt`** |
| **`WP` (Western Pacific)** | `423` | `74.39` | **`7.91 kt`** | `586` | `38.49` | **`11.90 kt`** |

---

## 4. Final-Test Limitations & Uncertainty

1. **Southern Hemisphere (`SH`) and Central Pacific (`CP`) Storm Count in `VAL` and `FINAL_TEST`:**
   - Because the 100 GB stratification allocated `8.0%` to `SH` (`7` storms) and `2.0%` to `CP` (`5` storms), `VAL` and `FINAL_TEST` each contain `1` `SH` storm (`26` `VAL` and `28` `FINAL_TEST` samples) and `1` `CP` storm (`95` `VAL` and `12` `FINAL_TEST` samples). While `FINAL_TEST` `SH` MAE improved to `4.98 kt` (`CP`: `1.41 kt`), single-storm regional sample sizes carry higher variance than `WP` (`5` `TEST` storms, `586` samples) or `IO` (`4` `TEST` storms, `386` samples).
2. **Future Track Target Availability (`~16.6%` of overpasses):**
   - `MetadataExtractor.query_future_track_coordinates` strictly requires an exact sequential Best Track timestamp within $\pm 30\text{ min}$ of $t_{\text{obs}} + h$ without synthetic interpolation. Consequently, `+24h` track targets are evaluated on the `270` (`VAL`) and `219` (`FINAL_TEST`) overpasses occurring within $\pm 30\text{ min}$ of 6-hourly synoptic Best Track fixes.

---

## 5. Exact Commands to Reproduce the 100 GB Experiment

```powershell
Set-Location "D:\Enjula uchoi\TC-GraphFormer-verified"
$PY = "D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\Enjula uchoi\TC-GraphFormer-verified"

# 1. Download verified 100 GB manifest (13,898 files, 99,997,769,720 bytes across 109 storms)
& $PY -u src/prepare_pilot_dataset.py download `
  --manifest reports/acquisition_100gb/tcprimed_100gb_manifest.csv `
  --destination data/raw/tcprimed_100gb `
  --workers 16

# 2. Audit downloaded raw files against manifest
& $PY -u src/prepare_pilot_dataset.py audit `
  --manifest reports/acquisition_100gb/tcprimed_100gb_manifest.csv `
  --destination data/raw/tcprimed_100gb `
  --reports-dir reports/acquisition_100gb

# 3. Execute preprocessing pipeline (builds 109 containers in data/processed/tcprimed_100gb and leak-free splits)
& $PY -u -m tcprimed_pipeline run `
  --config configs/pipeline_100gb.yaml `
  --resume

# 4. Self-supervised Masked Feature Modeling (MFM) pretraining on TRAIN (10,545 samples)
& $PY -u src/training/pretrain.py `
  --train-index reports/pipeline_100gb/sample_index.csv `
  --checkpoint-dir checkpoints/tcprimed_100gb `
  --output-report reports/phase2e4_100gb/pretraining_summary.json `
  --gradient-report reports/phase2e4_100gb/pretraining_gradient_report.json `
  --epochs 3 `
  --batch-size 32 `
  --lr 3e-4

# 5. Supervised chronological storm-sequence multi-task training (VAL-only checkpoint selection)
& $PY -u src/training/train_multitask.py `
  --config configs/gpu_node_100gb.yaml `
  --train-index reports/pipeline_100gb/sample_index.csv `
  --pretrained-checkpoint checkpoints/tcprimed_100gb/pretrained_backbone.pt `
  --checkpoint-dir checkpoints/tcprimed_100gb `
  --summary-report reports/phase2e4_100gb/multitask_training_summary.json `
  --use-sequences `
  --max-history-len 4 `
  --epochs 12 `
  --batch-size 32 `
  --accumulate-grad-batches 2 `
  --lr 3e-4 `
  --device cuda `
  --precision fp16 `
  --use-amp `
  --seed 42

# 6. Optional task-specific fine-tuning (isolated to finetuned_{task}_best.pt)
& $PY -u src/training/finetune_task.py `
  --checkpoint checkpoints/tcprimed_100gb/best_checkpoint.pt `
  --sample-index reports/pipeline_100gb/sample_index.csv `
  --checkpoint-dir checkpoints/tcprimed_100gb `
  --task intensity_change `
  --epochs 3 `
  --batch-size 32 `
  --output-report reports/phase2e4_100gb/finetuning_intensity_change_summary.json

# 7. Frozen FINAL_TEST evaluation against Persistence, Climatology, Linear, and Causal Velocity baselines
& $PY -u src/evaluation/evaluate_baselines.py `
  --config configs/gpu_node_100gb.yaml `
  --sample-index reports/pipeline_100gb/sample_index.csv `
  --checkpoint checkpoints/tcprimed_100gb/best_checkpoint.pt `
  --output-dir reports/phase2e4_100gb `
  --use-sequences `
  --max-history-len 4 `
  --batch-size 32 `
  --device cuda
```
