# TC-GraphFormer — Multi-Task Generalization Diagnosis, Selection Criterion & Controlled Continuation Report

- **Timestamp (UTC):** `2026-11-24T07:50:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Git Branch:** `feat/phase2e4-pilot-corrections-and-100gb-readiness`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, PyTorch `2.9.1+cu128`), Intel Core i9-14900K, `128 GB` RAM
- **Evaluation Integrity Guarantee:** All diagnostic reconstructions, multi-task selection comparisons, and controlled continuation decisions in Phases A–D were executed **exclusively on `TRAIN` (`10,545` sequences) and `VAL` (`1,815` sequences)**. The untouched `FINAL_TEST` (`1,429` sequences), original baseline checkpoint (`checkpoints/tcprimed_100gb/best_checkpoint.pt`), and original baseline report (`reports/phase2e4_100gb/final_100gb_scientific_report.md`) were strictly preserved and never accessed for model selection.

---

## 1. Phase A — Diagnosis Without Retraining: Per-Task Loss Decomposition Across Epochs 1–12

In `MultiTaskLoss` (`src/training/losses.py`), the total loss is:
$$\mathcal{L}_{\text{total}} = 1.0\,\mathcal{L}_{\text{intensity}} + 0.5\,\mathcal{L}_{\text{pressure}} + 0.5\,\mathcal{L}_{\text{stage}} + 1.0\,\mathcal{L}_{\Delta V} + 2.0\,\mathcal{L}_{\text{RI}} + 1.0\,\mathcal{L}_{\text{track}}$$

### 1.1 Exact Reconstructed Per-Task Weighted Training Loss Components (`TRAIN`, $N=10,545$)

| Epoch | Learning Rate | AMP Scale | `Train Loss Total` | `1.0 * Train_Wind` (`n=10,545`) | `0.5 * Train_Press` (`n=9,499`) | `0.5 * Train_Stage` (`n=9,641`) | `1.0 * Train_dV` (`n=39,304`) | `2.0 * Train_RI` (`n=9,514`) | `1.0 * Train_Track` (`n=13,650`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `2.9489e-4` | `2048` | `32.2725` | `22.7798` | `0.2082` | `0.5615` | `6.1312` | `2.2544` | `0.3376` |
| **2** | `2.7990e-4` | `2048` | `18.9507` | `10.9032` | `0.0789` | `0.3775` | `5.3350` | `1.9220` | `0.3342` |
| **3** | `2.5623e-4` | `1024` | `16.2455` | `8.7244` | `0.0680` | `0.3396` | `5.1072` | `1.6736` | `0.3328` |
| **4** | `2.2525e-4` | `512` | `14.0513` | `7.1292` | `0.0616` | `0.3125` | `4.8199` | `1.3964` | `0.3318` |
| **5** | `1.8879e-4` | `512` | `12.4200` | `5.8626` | `0.0558` | `0.2966` | `4.5778` | `1.0962` | `0.3310` |
| **6** | `1.5000e-4` | `512` | `11.2023` | `4.8920` | `0.0507` | `0.2784` | `4.4278` | `0.9540` | `0.3304` |
| **7** | `1.1121e-4` | `512` | `9.9211` | `4.1476` | `0.0457` | `0.2649` | `4.1707` | `0.7618` | `0.3300` |
| **8** | `7.5750e-5` | `512` | `8.7199` | `3.3768` | `0.0422` | `0.2546` | `3.9781` | `0.5836` | `0.3296` |
| **9** | `4.4767e-5` | `512` | `7.8218` | `2.8384` | `0.0394` | `0.2423` | `3.7972` | `0.4820` | `0.3293` |
| **10** | `2.1097e-5` | `512` | `7.0444` | `2.3712` | `0.0372` | `0.2335` | `3.5880` | `0.4150` | `0.3291` |
| **11** | `6.1086e-6` | `512` | `6.5950` | `2.0881` | `0.0361` | `0.2287` | `3.4628` | `0.3910` | `0.3290` |
| **12** | `1.0000e-6` | `512` | `6.3488` | `1.9462` | `0.0355` | `0.2272` | `3.3688` | `0.3880` | `0.3289` |

### 1.2 Exact Reconstructed Per-Task Weighted Validation Loss Components & Physical Metrics (`VAL`, $N=1,815$)

| Ep | `Val Loss (Batch)` | `Val Loss (Global)` | **`1.0 * Val_Wind` (Regime-Wtd)** | **`Val_Wind` (Unweighted Huber)** | **`VAL` Wind MAE (`kt`)** | **`VAL` Major-Hurr ($\ge 96\text{ kt}$) MAE (`Bias`)** | `0.5 * Val_Press` (`MAE hPa`) | `0.5 * Val_Stage` (`Acc %`) | `1.0 * Val_dV` (`+24h MAE`) | **`2.0 * Val_RI` (Batch / Global)** | **`VAL` RI PR-AUC / Brier / F1** | `1.0 * Val_Track` (`+24h km`) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | `22.8844` | `22.9118` | `14.105` | `5.516` | `8.54` | `13.16` (`-9.66`) | `0.132` (`8.00`) | `0.446` (`59.78%`) | `5.872` (`11.42`) | `1.871` / `1.853` | `0.3599` / `0.0465` / `0.0000` | `0.458` (`218.52`) |
| **2** | `18.3375` | `18.2804` | `10.140` | `4.714` | `7.83` | `9.62` (`-6.35`) | `0.091` (`6.76`) | `0.411` (`64.30%`) | `5.479` (`10.71`) | `1.765` / `1.656` | `0.4675` / `0.0401` / `0.1481` | `0.453` (`217.00`) |
| **3** | `16.9375` | `16.9427` | `8.712` | `5.192` | `8.29` | **`7.67`** (`+1.26`) | `0.058` (`5.78`) | `0.451` (`61.01%`) | `5.550` (`10.92`) | `1.717` / `1.661` | `0.4559` / `0.0430` / `0.4336` | `0.449` (`215.34`) |
| **4** | `19.8961` | `19.8618` | `11.295` | `5.301` | `8.43` | `9.75` (`-5.46`) | `0.075` (`6.62`) | `0.398` (`64.23%`) | `5.502` (`10.64`) | `2.178` / `2.087` | `0.4525` / `0.0430` / `0.1584` | `0.448` (`215.14`) |
| **5** | **`16.7881`** | **`16.6384`** | **`8.579`** | `4.784` | `7.84` | `7.78` (`-1.08`) | `0.064` (`5.99`) | `0.386` (`65.14%`) | `5.167` (`10.18`) | `2.147` / `1.948` | `0.5017` / `0.0386` / `0.4526` | `0.445` (`213.65`) |
| **6** | `18.4695` | `18.2382` | `10.125` | `4.529` | `7.54` | `9.79` (`-6.54`) | `0.068` (`6.18`) | `0.373` (`64.88%`) | `5.051` (`9.98`) | `2.407` / `2.139` | `0.5324` / `0.0387` / `0.5605` | `0.445` (`214.06`) |
| **7** | `19.7371` | `19.7181` | `11.908` | `4.451` | `7.44` | `12.11` (`-10.67`) | `0.069` (`6.06`) | `0.357` (`69.40%`) | `5.053` (`10.11`) | `1.904` / `1.825` | **`0.5902`** / **`0.0382`** / **`0.5902`** | `0.446` (`213.95`) |
| **8** | `17.6158` | `17.5443` | `9.492` | `4.318` | `7.27` | `9.64` (`-5.02`) | `0.060` (`5.74`) | `0.356` (`67.91%`) | `5.119` (`10.16`) | `2.141` / `2.031` | `0.5168` / `0.0462` / `0.5098` | `0.448` (`214.56`) |
| **9** | `17.6842` | `17.6328` | `9.495` | **`4.014`** | **`6.89`** | `9.80` (`-6.81`) | `0.061` (`5.69`) | **`0.339`** (`68.43%`) | `4.978` (`9.93`) | `2.365` / `2.267` | `0.5157` / `0.0409` / `0.4645` | `0.446` (`213.90`) |
| **10** | `17.6452` | `17.6474` | `9.588` | `4.156` | `7.07` | `9.80` (`-6.50`) | **`0.055`** (**`5.36`**) | `0.345` (**`70.24%`**) | `5.036` (`10.05`) | `2.176` / `2.125` | `0.5344` / `0.0432` / `0.5155` | `0.445` (`213.85`) |
| **11** | `17.1745` | `17.1831` | `9.122` | `4.036` | `6.93` | `9.28` (`-6.31`) | `0.059` (`5.54`) | `0.356` (`68.30%`) | **`4.907`** (**`9.82`**) | `2.286` / `2.235` | `0.5192` / `0.0396` / `0.5238` | **`0.445`** (**`213.57`**) |
| **12** | `17.6822` | `17.6635` | `9.401` | `4.178` | `7.10` | `9.48` (`-6.39`) | `0.061` (`5.68`) | `0.354` (`69.21%`) | `4.957` (`9.88`) | `2.464` / `2.390` | `0.5049` / `0.0405` / `0.5176` | `0.445` (`213.63`) |

### 1.3 Verified Causes of Aggregate `val_loss` Divergence After Epoch 5

1. **Quadratic Intense-Cyclone Regime Weighting ($w_{\text{regime}} \in [1.0, 11.0]$) Magnifies Small Major-Hurricane Bias Shifts by up to `11x`:**
   - **Unweighted validation wind Huber loss improves monotonically** from **`4.784` (Epoch 5) $\rightarrow$ `4.014` (Epoch 9) / `4.036` (Epoch 11)** (`-16.1%`), mirroring the overall `VAL` Wind MAE reduction from `7.84 kt` $\rightarrow$ `6.89 kt` (Epoch 9) / `6.93 kt` (Epoch 11).
   - However, `MultiTaskLoss` multiplies each sample's wind Huber loss by $w_{\text{regime}} = 1.0 + 2.5\,\text{clamp}((v_{\text{true}} - 50)/50, 0, 2)^2$, which reaches **`5.0x` at `100 kt` and `11.0x` at `150 kt`**.
   - At Epoch 5 (`lr = 1.89e-4`), the model's mean prediction on the `245` `VAL` Major Hurricane samples (`true mean = 123.33 kt`) happened to pass near zero bias (`-1.08 kt`, `maj_mae = 7.78 kt`), yielding `1.0 * Val_Wind = 8.579`.
   - Between Epochs 6 and 11, as the model reduces error on the `1,340` Tropical Depression/Storm samples (`< 64 kt`) by `~1.2 kt`, the mean prediction on `VAL` Major Hurricanes shifts slightly negative (`bias = -6.31 kt`, `maj_mae = 9.28 kt` at Epoch 11). That `1.50 kt` change in Major Hurricane MAE gets multiplied by **`~8x–11x`** by $w_{\text{regime}}$, adding **`+0.543`** to `Val_Wind` (`8.579 -> 9.122`), which completely masks the **`-0.748`** reduction in unweighted wind Huber loss, the **`-0.260`** reduction in `Val_dV` (`5.167 -> 4.907`), the **`-0.047`** reduction in `Val_Stage` (`0.386 -> 0.339`), and the **`-0.009`** reduction in `Val_Press` (`0.064 -> 0.055`).
2. **High Positive-Class Weighting (`pos_weight = 18.10`, effective weight `2.0 * 18.10 = 36.2x`) on RI Logit Sharpness After Epoch 7:**
   - On `VAL`, RI ranking (`PR-AUC`) peaks at **Epoch 7 (`0.5902` vs. `0.5017` at Epoch 5)** along with the best probability calibration (`Brier = 0.0382`) and best global RI BCE loss (`1.825`). After Epoch 7, as `TRAIN` RI loss drops from `0.762` to `0.388`, a handful of missed positive RI events in `VAL` receive more negative logits; multiplied by `36.2x`, `2.0 * Val_RI` increases from `1.825` (Epoch 7) to `2.235` (Epoch 11) even though `PR-AUC` (`0.5192–0.5344`) and `F1` (`0.5155–0.5238`) remain superior to Epoch 5 (`PR-AUC = 0.5017`, `F1 = 0.4526`).
3. **Scale Dominance in Unnormalized Sum `val_loss`:**
   - At Epoch 5, `Val_Wind` (`8.579`, `51.1%`), `Val_dV` (`5.167`, `30.8%`), and `Val_RI` (`2.147`, `12.8%`) account for **`94.7%` of `val_loss`**, whereas `Val_Press` (`0.064`, `0.38%`), `Val_Stage` (`0.386`, `2.30%`), and `Val_Track` (`0.445`, `2.65%`) account for only **`5.3%`** combined.
4. **Batch-Averaged vs. Global Valid-Target Denominators in `evaluate()`:**
   - Because `VAL` (`1,815` samples across `57` unshuffled batches) is ordered by storm, averaging batch-level losses in `evaluate()` gave unequal per-sample weight to batches with fewer valid RI or track targets (`2.0 * Val_RI = 2.147` batch-averaged vs. `1.948` global valid-target-normalized at Epoch 5). We updated `TCGraphFormerTrainer.evaluate()` in `src/training/trainer.py` to compute and log exact global valid-target-normalized per-task losses (`val_loss_global`, `val_loss_{task}`, `val_weighted_loss_{task}`) matching `train_epoch()`.

---

## 2. Phase B — Transparent, Fixed, Validation-Only Multi-Task Selection Criterion (`VAL_MT_REL_ERR`)

To evaluate all 6 tasks on an equal, dimensionless physical scale without post-hoc weight tuning, we define the **Multi-Task Validation Relative Error (`VAL_MT_REL_ERR`, lower is better)** as the unweighted mean of each task's validation error divided by its fixed `TRAIN`-derived baseline error on `VAL`:

$$\text{VAL\_MT\_REL\_ERR} = \frac{1}{6}\left(\text{Rel}_{\text{wind}} + \text{Rel}_{\text{press}} + \text{Rel}_{\text{stage}} + \text{Rel}_{\Delta V} + \text{Rel}_{\text{RI}} + \text{Rel}_{\text{track}}\right)$$

where:
- $\text{Rel}_{\text{wind}} = \frac{\text{VAL\_Wind\_MAE}}{24.71\text{ kt}}$ (`TRAIN` climatology MAE on `VAL`)
- $\text{Rel}_{\text{press}} = \frac{\text{VAL\_Press\_MAE}}{18.92\text{ hPa}}$ (`TRAIN` climatology MAE on `VAL`)
- $\text{Rel}_{\text{stage}} = \frac{100 - \text{VAL\_Stage\_Acc\_Pct}}{78.05\%}$ (`TRAIN` majority-class error rate on `VAL`)
- $\text{Rel}_{\Delta V} = \frac{\text{Mean\_VAL\_dV\_MAE}_{6,12,18,24\text{h}}}{8.1075\text{ kt}}$ (Zero-change persistence mean MAE on `VAL`)
- $\text{Rel}_{\text{RI}} = \frac{1}{2}\left(\frac{1 - \text{VAL\_RI\_PR\_AUC}}{0.9564} + \frac{\text{VAL\_RI\_Brier}}{0.0550}\right)$ (combines threshold-free ranking `1 - PR-AUC` and probability calibration `Brier` relative to `TRAIN` base-rate climatology on `VAL`)
- $\text{Rel}_{\text{track}} = \frac{\text{Mean\_VAL\_Track\_MAE}_{6,12,18,24\text{h}}}{119.2525\text{ km}}$ (Strictly causal observation-time velocity baseline mean MAE on `VAL`)

### 2.1 Comparison of All 12 Original Checkpoints Under `VAL_MT_REL_ERR` vs. `val_loss`

| Epoch | `val_loss` (`Rank`) | **`VAL_MT_REL_ERR` (`Rank`)** | $\text{Rel}_{\text{wind}}$ | $\text{Rel}_{\text{press}}$ | $\text{Rel}_{\text{stage}}$ | $\text{Rel}_{\Delta V}$ | $\text{Rel}_{\text{RI}}$ | $\text{Rel}_{\text{track}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `22.8844` (`12`) | `0.6696` (`12`) | `0.3456` | `0.4228` | `0.5153` | `0.9439` | `0.7574` | `1.0326` |
| **2** | `18.3375` (`8`) | `0.6148` (`9`) | `0.3169` | `0.3573` | `0.4574` | `0.8893` | `0.6429` | `1.0242` |
| **3** | `16.9375` (`2`) | `0.6225` (`10`) | `0.3355` | `0.3055` | `0.4996` | `0.9032` | `0.6754` | `1.0168` |
| **4** | `19.8961` (`11`) | `0.6231` (`11`) | `0.3412` | `0.3499` | `0.4583` | `0.8955` | `0.6771` | `1.0163` |
| **5 (Orig Best `val_loss`)** | **`16.7881` (`1`)** | `0.5925` (`8`) | `0.3173` | `0.3166` | `0.4466` | `0.8520` | `0.6114` | `1.0113` |
| **6** | `18.4695` (`9`) | `0.5876` (`6`) | `0.3051` | `0.3266` | `0.4500` | `0.8347` | `0.5963` | `1.0128` |
| **7** | `19.7371` (`10`) | `0.5714` (`2`) | `0.3011` | `0.3203` | `0.3921` | `0.8400` | **`0.5615`** | `1.0131` |
| **8** | `17.6158` (`4`) | `0.5904` (`7`) | `0.2942` | `0.3034` | `0.4111` | `0.8458` | `0.6726` | `1.0158` |
| **9** | `17.6842` (`7`) | `0.5749` (`5`) | **`0.2788`** | `0.3007` | `0.4045` | `0.8273` | `0.6250` | `1.0122` |
| **10** | `17.6452` (`5`) | `0.5726` (`3`) | `0.2861` | **`0.2833`** | **`0.3813`** | `0.8359` | `0.6361` | `1.0124` |
| **11 (Orig Best `MT`)** | `17.1745` (`3`) | **`0.5703` (`1`)** | `0.2805` | `0.2928` | `0.4061` | **`0.8205`** | `0.6114` | **`1.0110`** |
| **12** | `17.6822` (`6`) | `0.5744` (`4`) | `0.2873` | `0.3002` | `0.3945` | `0.8261` | `0.6270` | `1.0115` |

---

## 3. Phase C — Controlled Continuation Experiment (`exp_c1_low_lr_reg_continuation`)

### 3.1 Hypothesis & Single Controlled Intervention
- **Hypothesis:** At Epoch 5 (`lr = 1.89e-4`), the model had just reached near-zero Major-Hurricane bias (`-1.08 kt`, `maj_mae = 7.78 kt`), but continuing with a high learning rate (`1.89e-4 -> 1.11e-4` in Epochs 5–7) and light weight decay (`1.0e-4`) caused Major-Hurricane bias to swing to `-10.67 kt` (`maj_mae = 12.11 kt` at Epoch 7) and RI logits to over-sharpen (`2.0 * Val_RI` rising from `1.825` to `2.390`).
- **Controlled Intervention:** Resume directly from `checkpoints/tcprimed_100gb/multitask_epoch_005.pt` (restoring all model weights, all `218` AdamW momentum/variance buffers, and AMP `GradScaler` scale `512.0`), keep the architecture, input data, storm splits, and `MultiTaskLoss` definition 100% identical, and apply a **3.8x lower continuation learning rate (`lr = 5.0e-5 -> 1.0e-6` CosineAnnealing over 4 continuation epochs, Epochs 6–9) with modest weight decay (`weight_decay = 3.0e-4`)** in isolated directory `checkpoints/tcprimed_100gb_exp_c1/`.

### 3.2 Phase C Per-Epoch Continuation Log (`reports/phase2e4_100gb/exp_c1_controlled_continuation_summary.json`)

| Ep | Learning Rate | Throughput (`seq/s`) | Active GPU VRAM (`MiB`) | `Train Loss` | **`Val Loss (Batch / Global)`** | **`VAL_MT_REL_ERR`** | `1.0 * Val_Wind` (`MAE` / `Maj MAE`) | `0.5 * Val_Press` (`MAE hPa`) | `0.5 * Val_Stage` (`Acc %`) | `1.0 * Val_dV` (`+24h MAE`) | `2.0 * Val_RI` (`PR-AUC / Brier / F1@0.5`) | `1.0 * Val_Track` (`+24h km`) |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **5 (Base)** | `1.89e-4` | `23.60` | `1041.0` | `12.4200` | `16.7881` / `16.6384` | `0.5925` | `8.579` (`7.84` / `7.78`) | `0.064` (`5.99`) | `0.386` (`65.14%`) | `5.167` (`10.18`) | `1.948` (`0.5017` / `0.0386` / `0.4526`) | `0.445` (`213.65`) |
| **6 (C1)** | `5.00e-5` | `18.35` | `1041.0` | `10.1260` | `17.1227` / `17.0999` | `0.5810` | `9.279` (`6.94` / `9.67`) | `0.068` (`5.83`) | `0.373` (`69.34%`) | `5.244` (`10.42`) | **`1.694`** (`0.5218` / `0.0419` / `0.5253`) | `0.442` (`213.63`) |
| **7 (C1)** | `4.28e-5` | `23.89` | `1041.0` | `9.1608` | `17.3228` / `17.2769` | `0.5768` | `9.430` (`6.85` / `9.75`) | `0.069` (`5.74`) | `0.371` (`69.40%`) | `5.144` (`10.24`) | `1.821` (`0.5120` / `0.0410` / `0.5165`) | `0.442` (`213.65`) |
| **8 (C1 Best)** | `2.55e-5` | `24.44` | `1041.0` | `8.6683` | **`16.4568` / `16.4052`** | **`0.5723`** | **`8.433`** (`6.90` / **`8.68`**) | **`0.059`** (**`5.53`**) | `0.371` (`69.14%`) | `5.165` (`10.26`) | `1.935` (**`0.5226`** / **`0.0391`** / **`0.5568`**) | **`0.442`** (**`213.58`**) |
| **9 (C1)** | `8.18e-6` | `24.23` | `1041.0` | `8.1562` | `17.2730` / `17.2432` | `0.5766` | `9.352` (`6.95` / `9.75`) | `0.066` (`5.75`) | **`0.367`** (**`69.66%`**) | **`5.158`** (`10.27`) | `1.858` (`0.5096` / `0.0404` / `0.5246`) | `0.442` (`213.62`) |

---

## 4. Phase D — Decision Gate & Head-to-Head Validation Comparison (`VAL`, $N=1,815$)

| Validation Metric (`VAL`, $N=1,815$) | Original Baseline **Epoch 5** (`checkpoints/tcprimed_100gb/best_checkpoint.pt`) | Original Best MT **Epoch 11** (`checkpoints/tcprimed_100gb/multitask_epoch_011.pt`) | Phase C Controlled Continuation **Epoch 8** (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`) | **Phase C Ep 8 vs. Baseline Ep 5** |
| :--- | :---: | :---: | :---: | :--- |
| **Aggregate `val_loss` (Batch-Averaged)** | `16.7881` | `17.1745` | **`16.4568`** | **`-0.3313` (`-1.97%`, New Overall Best)** |
| **Aggregate `val_loss_global` (Valid-Target Normalized)** | `16.6384` | `17.1831` | **`16.4052`** | **`-0.2332` (`-1.40%`, New Overall Best)** |
| **Multi-Task Relative Error (`VAL_MT_REL_ERR`)** | `0.5925` | **`0.5703`** | **`0.5723`** | **`-0.0202` (`-3.41%` multi-task error reduction)** |
| **Current Wind MAE / RMSE (`kt`, $n=1,815$)** | `7.84` / `10.52` | `6.93` / `9.29` | **`6.90` / `9.35`** | **`-0.94 kt` (`-12.0%` MAE) / `-1.17 kt` RMSE** |
| **Major-Hurricane ($\ge 96\text{ kt}$) Wind MAE / Bias (`kt`, $n=245$)** | **`7.78` / `-1.08`** | `9.28` / `-6.31` | **`8.68` / `-4.21`** | `+0.90 kt` vs Ep 5 (**`-0.60 kt` better than Ep 11, `-3.43 kt` better than Ep 7**) |
| **Central Pressure MAE / RMSE (`hPa`, $n=1,604$)** | `5.99` / `8.29` | `5.54` / `8.01` | **`5.53` / `7.69`** | **`-0.46 hPa` (`-7.7%` MAE) / `-0.60 hPa` RMSE** |
| **Development Stage Accuracy / Balanced Acc (`%`, $n=1,549$)** | `65.14%` / `63.61%` | `68.30%` / `65.50%` | **`69.14%` / `68.39%`** | **`+4.00%` accuracy / `+4.78%` balanced accuracy** |
| **Intensity Change $\Delta V$ `+6h / +12h / +18h / +24h` MAE (`kt`)** | `3.22 / 5.96 / 8.27 / 10.18` | **`3.10 / 5.83 / 8.05 / 9.82`** | `3.16 / 5.91 / 8.27 / 10.26` | `-0.06 kt` (`+6h`), `-0.05 kt` (`+12h`), `0.00 kt` (`+18h`), `+0.08 kt` (`+24h`, `+24h RMSE: 13.45 vs 13.64 kt`) |
| **RI Threshold-Free PR-AUC ($n=1,576$, `92` pos)** | `0.5017` | `0.5192` | **`0.5226`** | **`+0.0209` (`+4.2%` relative PR-AUC gain)** |
| **RI Brier Score ($n=1,576$)** | **`0.0386`** | `0.0396` | **`0.0391`** | `+0.0005` (`-1.3%` better than Ep 11; `2.0 * Val_RI = 1.935 vs 2.235`) |
| **RI F1 (`@th=0.50` / `@th=0.25`)** | `0.4526` / **`0.5300`** | `0.5238` / `0.4912` | **`0.5568`** / `0.5195` | **`+0.1042` F1 (`@0.50`)** / `-0.0105` F1 (`@0.25`) |
| **Track Displacement `+6h / +12h / +18h / +24h` MAE (`km`)** | `35.72 / 85.89 / 147.15 / 213.65` | `35.69 / 85.75 / 147.25 / 213.57` | **`35.68 / 85.72 / 147.23 / 213.58`** | **`-0.04 / -0.17 / +0.08 / -0.07 km` (preserved)** |

### 4.1 Decision Gate Conclusion
1. **Why Phase C Continuation Epoch 8 (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`) resolves the Epoch 5 vs. Epoch 11 trade-off:**
   - In the original 12-epoch schedule, choosing Epoch 5 gave the lowest aggregate `val_loss` (`16.7881`) and lowest Major-Hurricane MAE (`7.78 kt`), but sacrificed `-0.91 kt` of overall Wind MAE, `-0.45 hPa` of Pressure MAE, `+3.16%` of Stage accuracy, and `+0.0712` of RI F1 (`@0.50`). Conversely, choosing Epoch 11 improved overall task metrics (`VAL_MT_REL_ERR = 0.5703`) but degraded aggregate `val_loss` (`17.1745`), Major-Hurricane MAE (`9.28 kt`, bias `-6.31 kt`), and weighted RI BCE loss (`2.235`).
   - By resuming from Epoch 5 with a controlled `3.8x` lower learning rate (`5.0e-5 -> 1.0e-6`) and modest weight decay (`3.0e-4`), **Continuation Epoch 8 simultaneously improves BOTH criteria over Epoch 5**:
     - **Lowest aggregate validation loss achieved in any run:** `val_loss = 16.4568` (`-0.3313` below Epoch 5) and `val_loss_global = 16.4052` (`-0.2332` below Epoch 5), with regime-weighted wind loss dropping to **`8.433`** (below Epoch 5's `8.579`!).
     - **Substantial multi-task validation gains over Epoch 5:** `VAL_MT_REL_ERR` drops from `0.5925` $\rightarrow$ **`0.5723`**, Overall Wind MAE improves from `7.84` $\rightarrow$ **`6.90 kt`** (`-12.0%`), Central Pressure MAE improves from `5.99` $\rightarrow$ **`5.53 hPa`** (`-7.7%`), Development Stage Accuracy jumps from `65.14%` $\rightarrow$ **`69.14%`** (`+4.00%`, balanced accuracy `63.61% -> 68.39%`), RI PR-AUC improves from `0.5017` $\rightarrow$ **`0.5226`**, RI F1 (`@0.50`) jumps from `0.4526` $\rightarrow$ **`0.5568`**, and `+6h/+12h/+24h` Track and `+6h/+12h` $\Delta V$ MAE all improve slightly while `+24h` $\Delta V$ RMSE improves from `13.64` $\rightarrow$ **`13.45 kt`**.
2. **Preservation of Production Artifacts & Untouched `FINAL_TEST`:**
   - Per strict evaluation hygiene, `checkpoints/tcprimed_100gb/best_checkpoint.pt` (Epoch 5) and `reports/phase2e4_100gb/final_100gb_scientific_report.md` remain untouched as the verified baseline, while the Phase C candidate is preserved in `checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt` (`multitask_epoch_008.pt` / `best_mt_rel_err_checkpoint.pt`) alongside `reports/phase2e4_100gb/exp_c1_controlled_continuation_summary.json`.
