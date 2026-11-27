# TC-GraphFormer — Investigation of Premature Convergence, Longer-Schedule Learning, and Input Information Bottlenecks

**Date:** 2026-11-25  
**Repository:** [TC-GraphFormer-verified](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified)  
**Branch:** `feat/tc-graphformer-v2-generalization`  
**Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM), `128 GB` RAM  
**Dataset:** NOAA TC PRIMED 100 GB Expansion (`TRAIN`: `75` storms, `10,545` sequences; `VAL`: `17` storms, `1,815` sequences; `FINAL_TEST`: `17` storms, `1,429` sequences — **strictly excluded and untouched**)  
**Primary Telemetry JSON:** [exp_s24_longer_schedule_summary.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/exp_s24_longer_schedule_summary.json)

---

## 1. Verified Convergence Diagnosis (Phase A)

### 1.1 Exact Optimizer Steps and Cosine Learning-Rate Trajectory in the Original 12-Epoch Run
In the original 12-epoch supervised training run ([experiment_summary.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/experiment_summary.json), [train_multitask.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/training/train_multitask.py)):
- **Batches and Optimizer Steps:** `10,545` `TRAIN` sequences at `batch_size = 32` produce `330` mini-batches per epoch (`329` full batches of `32` + `1` tail batch of `17`). With `accumulate_grad_batches = 2` (effective batch size `64`), the optimizer executes exactly **`165` parameter updates per epoch**, or **`1,980` total optimizer steps** across `12` epochs.
- **Cosine Schedule Formulation:** `CosineAnnealingLR(optimizer, T_max=12, eta_min=1.0e-6)` from $\eta_0 = 3.0\times 10^{-4}$. Because `scheduler.step()` is called once at the end of each epoch, the active learning rate during Epoch $t \in \{1,\dots,12\}$ is:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi (t-1)}{12}\right)\right)$$

| Epoch | Active LR ($\eta_t$) | End-of-Epoch LR ($\eta_{t+1}$) | Cum. Opt Steps | Train Opt Loss | `VAL` Batch Loss | `VAL_EVAL` Global Loss | `VAL_MT_REL_ERR` | Wind MAE (`kt`) | Major-Hurr MAE (`kt`) | Major-Hurr Bias (`kt`) | Pressure MAE (`hPa`) | Stage Acc / Bal (`%`) | $+24\text{h}\;\Delta V$ MAE (`kt`) | RI PR-AUC / F1@0.5 / Brier | $+24\text{h}$ Track (`km`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `3.000e-4` | `2.949e-4` | `165` | `32.2725` | `22.8844` | `22.9118` | `0.6696` | `8.54` | `13.16` | `-9.66` | `8.00` | `59.78 / 44.90` | `11.42` | `0.3599 / 0.0000 / 0.0465` | `218.52` |
| **2** | `2.949e-4` | `2.800e-4` | `330` | `18.9507` | `18.3375` | `18.2731` | `0.6203` | `7.76` | `9.29` | `-4.76` | `7.12` | `64.56 / 57.16` | `10.74` | `0.4539 / 0.0215 / 0.0408` | `216.92` |
| **3** | `2.800e-4` | `2.562e-4` | `495` | `16.2455` | `16.9375` | `16.9012` | `0.6001` | `7.93` | `8.26` | `-2.73` | `6.00` | `64.43 / 59.95` | `10.50` | `0.5050 / 0.3051 / 0.0376` | `215.79` |
| **4** | `2.562e-4` | `2.253e-4` | `660` | `14.0513` | `19.8961` | `19.8114` | `0.6095` | `8.16` | `11.53` | `-9.90` | `6.79` | `62.62 / 61.98` | `10.30` | `0.4910 / 0.3158 / 0.0401` | `214.92` |
| **5** | `2.253e-4` | `1.892e-4` | `825` | `12.4200` | **`16.7881`** | **`16.6384`** | `0.5877` | `7.84` | **`7.78`** | **`-1.08`** | `5.99` | `65.14 / 64.27` | `10.18` | `0.5017 / 0.4526 / 0.0386` | `213.65` |
| **6** | `1.892e-4` | `1.505e-4` | `990` | `11.2023` | `18.4695` | `18.2382` | `0.5876` | `7.54` | `9.79` | `-6.44` | `6.18` | `64.88 / 65.11` | `9.98` | `0.5324 / 0.5455 / 0.0387` | `214.06` |
| **7** | `1.505e-4` | `1.118e-4` | `1,155` | `9.9211` | `19.7371` | `19.4418` | `0.5784` | `7.54` | `11.35` | `-9.52` | `5.57` | `67.40 / 67.22` | `10.16` | **`0.5902 / 0.5902 / 0.0382`** | `213.85` |
| **8** | `1.118e-4` | `7.575e-5` | `1,320` | `8.7199` | `17.6158` | `17.4204` | `0.5768` | `7.34` | `8.90` | `-4.82` | `5.49` | `67.91 / 66.98` | `9.98` | `0.5284 / 0.5568 / 0.0412` | `213.71` |
| **9** | `7.575e-5` | `4.479e-5` | `1,485` | `7.8218` | `17.6842` | `17.5190` | `0.5759` | **`6.89`** | `9.61` | `-6.72` | `5.86` | `68.17 / 66.11` | `9.97` | `0.5312 / 0.5474 / 0.0418` | `213.75` |
| **10** | `4.479e-5` | `2.103e-5` | `1,650` | `7.0444` | `17.6452` | `17.5012` | `0.5721` | `7.09` | `9.44` | `-6.51` | **`5.36`** | **`70.24 / 70.16`** | `9.89` | `0.5210 / 0.5445 / 0.0408` | `213.66` |
| **11** | `2.103e-5` | `6.094e-6` | `1,815` | `6.5950` | `17.1745` | `17.1831` | **`0.5703`** | `6.93` | `9.28` | `-6.31` | `5.54` | `68.30 / 66.88` | **`9.82`** | `0.5192 / 0.5238 / 0.0396` | **`213.57`** |
| **12** | `6.094e-6` | `1.000e-6` | `1,980` | `6.3488` | `17.6822` | `17.5944` | `0.5741` | `6.96` | `9.69` | `-6.89` | `5.57` | `68.24 / 66.75` | `9.84` | `0.5168 / 0.5288 / 0.0403` | `213.60` |

### 1.2 Why Aggregate Validation Loss Bottomed at Epoch 5 While Individual Tasks Peaked at Epochs 7–11
1. **Unweighted Task Losses Improved Past Epoch 5:** Across `VAL` (`1,815` sequences), unweighted wind Huber loss dropped monotonically from `4.7838` (Epoch 5) to `4.014` (Epoch 9) and `4.0360` (Epoch 11); pressure Huber loss dropped from `0.1373` to `0.1180` (Epoch 10); stage focal CE dropped from `0.8382` to `0.7310` (Epoch 10); and multi-horizon $\Delta V$ Huber loss dropped from `5.1818` to `4.9369` (Epoch 11).
2. **Quadratic Regime Amplification + RI Prior Shift Masked the Multi-Task Improvement in Aggregate `val_loss`:**
   - The intensity loss applies a quadratic regime multiplier $w_{\text{regime}}(V) = \left(1 + 2.5\,\text{clip}\!\left(\frac{V-34}{106}, 0, 1\right)\right)^2 - 1 \in [1.0, 11.25]$ in knot space (`~45x` larger gradient than standardized pressure).
   - At Epoch 5, the model had a mild positive training wind bias (`+3.12 kt`) that happened to yield nearly zero bias (`-1.08 kt`, `MAE = 7.78 kt`) on the `245` `VAL` Major Hurricane samples (`WP102018` Mangkhut, `WP181998` Zeb, `AL062004` Frances, `IO022007` Gonu).
   - As training continued to Epochs 7–12 under rapidly decaying cosine learning rate, `TRAIN_EVAL` loss dropped from `13.0492` (Epoch 5) to `5.7588` (Epoch 11), memorizing the `75` `TRAIN` storms (`TRAIN` Major-Hurricane MAE `6.64 -> 2.50 kt`, `TRAIN` RI PR-AUC `0.5574 -> 0.8928`). On unseen `VAL` storms, predictions regressed toward the mean intensity, shifting Major-Hurricane bias from `-1.08 kt` (Epoch 5) to `-6.31 kt` (Epoch 11). Because those `245` samples carry up to `11.25x` weight, regime-weighted intensity loss rose from `8.5792` to `9.1217` (`+0.5425`), and overconfident RI logits on `VAL` (`TRAIN` RI base rate `5.23%` vs. `VAL` `5.84%`, exacerbated by `ri_prior_shift_factor = 0.85` and `pos_weight = 18.1`) increased weighted RI focal BCE from `1.9480` to `2.2345` (`+0.2865`).
3. **Rapid Cosine Decay Starved Mid-Schedule Exploration:** Under the 12-epoch cosine schedule (`T_max=12`), the learning rate had already fallen by `50%` at Epoch 7 (`1.505e-4`) and by `93%` at Epoch 11 (`2.103e-5`), freezing the representation after it had overfit `TRAIN` rather than allowing sufficient mid-schedule optimization to align the multi-task heads before fine-grained annealing.

### 1.3 Audit of the 3-Epoch Self-Supervised MFM Pretraining Logs
Inspection of [pretraining_summary.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/pretraining_summary.json) and [pretraining_gradient_report.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/pretraining_gradient_report.json) (`mask_ratio = 0.30`, `184,049` masked tokens / `618,585` valid tokens across `10,545` `TRAIN` overpasses, `330` steps/epoch = `990` total steps):

| MFM Epoch | Masked Token MSE Loss | Relative Drop vs. Prev Epoch | `ir_encoder` Grad / Update Norm | `pmw_encoder` Grad / Update Norm | `gprof_encoder` Grad / Update Norm | `radar_encoder` Grad / Update Norm | `env_encoder` Grad / Update Norm | `fusion` Grad / Update Norm |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `0.59527` | — | `0.2978 / 0.0618` | `0.2928 / 0.0614` | `0.3313 / 0.0505` | `0.2923 / 0.0177` | `0.3916 / 0.0448` | `0.3789 / 0.0613` |
| **2** | `0.35587` | `-40.22%` | `0.2064 / 0.0577` | `0.1985 / 0.0501` | `0.3156 / 0.0381` | `0.1897 / 0.0215` | `0.2440 / 0.0444` | `0.2150 / 0.0451` |
| **3** | `0.28097` | **`-21.05%`** | `0.1925 / 0.0713` | `0.1466 / 0.0521` | `0.1999 / 0.0459` | `0.1471 / 0.0159` | `0.1773 / 0.0288` | `0.2004 / 0.0431` |

- **Diagnosis:** A `-21.05%` relative MSE reduction between Epoch 2 and Epoch 3, accompanied by non-diminishing parameter update norms (`ir_encoder` update norm *increased* from `0.0577` in Epoch 2 to `0.0713` in Epoch 3), proves that **3 epochs (`990` steps) of MFM pretraining did not converge**.

---

## 2. Learning-Rate and Optimizer-Update Analysis

To test whether the 12-epoch cosine schedule (`T_max=12`) prematurely quenched learning, we compare four distinct learning-rate trajectories across the exact same `TRAIN` (`10,545` sequences, `165` optimizer steps/epoch) and `VAL` (`1,815` sequences) partitions:
1. **Original 12-Epoch Cosine (`T_max=12`, `3.0e-4 -> 1.0e-6`):** Decays rapidly past Epoch 5 (`2.253e-4` at Ep 5 $\to$ `1.505e-4` at Ep 7 $\to$ `4.479e-5` at Ep 10 $\to$ `2.103e-5` at Ep 11).
2. **Epoch-5 Low-LR Regularized Continuations (`exp_c1` and `T6_decoupled_wd`):** Branch from Original Epoch 5 (`checkpoints/tcprimed_100gb/best_checkpoint.pt`) with an immediate `4.5x` LR step-down (`2.253e-4 -> 5.0e-5`) and higher weight decay (`3e-4` in `C1`, `1e-3` decoupled in `T6`).
3. **Fair 24-Epoch Longer Cosine Schedule (`exp_s24_longer_cosine_from_pretrained`, `T_max=24`, `3.0e-4 -> 1.0e-6`):** Starts fresh from [pretrained_backbone.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb/pretrained_backbone.pt) (`243` matched backbone tensors) with the exact same architecture, seed (`42`), batch size (`32 x 2 = 64`), V1 `MultiTaskLoss`, and `AdamW(lr=3.0e-4, wd=1.0e-4)`, stretching the cosine decay over `24` epochs (`3,960` max steps):
   - At **Epoch 7** (`1,155` steps), `S24` maintains $\eta_7 = 2.562\times 10^{-4}$ (`+70.2%` higher than `T_max=12`'s `1.505e-4`), equivalent to Epoch 4 of the 12-epoch schedule.
   - At **Epoch 10** (`1,650` steps), `S24` maintains $\eta_{10} = 2.077\times 10^{-4}$ (`4.64x` higher than `T_max=12`'s `4.479e-5`), right in the sweet spot between Epochs 5 and 6 of the 12-epoch schedule.
   - At **Epoch 13** (`2,145` steps), `S24` reaches the cosine midpoint $\eta_{13} = 1.505\times 10^{-4}$.

---

## 3. Fair Longer-Schedule Training Results (`exp_s24_longer_cosine_from_pretrained`, Phases B & C)

- **Configuration Hash:** `c451dd564f7d`
- **Initialization:** Fresh from [checkpoints/tcprimed_100gb/pretrained_backbone.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb/pretrained_backbone.pt) (`243` backbone parameter tensors transferred; `0` missing backbone keys; task prediction heads initialized identically with `seed=42`).
- **Epoch 1 Reproducibility Verification:** Because `S24` uses the exact same seed (`42`), initial LR (`3.000e-4`), and data loader order as the original 12-epoch run, **Epoch 1 is bit-for-bit identical** (`OptLoss = 32.2725`, `VAL_Batch_Loss = 22.8844`, `Wind MAE = 8.54 kt`, `Pressure MAE = 8.00 hPa`, `Stage Acc = 59.78%`, `+24h Track = 218.52 km`), confirming a 100% controlled comparison.
- **Predeclared Early-Stopping Rule & Execution Outcome:**
  - Rule: Run at least `min_epochs_before_stop = 15` epochs (strictly exceeding the original 12-epoch budget and passing the 24-epoch cosine midpoint at Epoch 13); stop if neither `VAL_EVAL_LOSS` (`val_loss_global`) nor `VAL_MT_REL_ERR` improves for `early_stop_patience = 4` consecutive epochs.
  - Outcome: **`S24` achieved its best composite multi-task relative error at Epoch 7 (`VAL_MT_REL_ERR = 0.5531`, `VAL_EVAL_LOSS = 16.1483`) and its all-time best global validation loss at Epoch 10 (`VAL_EVAL_LOSS = 15.9896`, `VAL_MT_REL_ERR = 0.5646`)**, followed by 5 consecutive epochs (`Epochs 11–15`) of monotonic `TRAIN` overfitting without validation improvement, triggering clean early stopping at **Epoch 15** (`2,475` optimizer steps).
- **Saved Checkpoints:**
  - All 15 epoch checkpoints: `checkpoints/tcprimed_100gb_exp_s24/s24_multitask_epoch_001.pt` .. `s24_multitask_epoch_015.pt`
  - Best validation loss checkpoint (**Epoch 10**): [checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt)
  - Best multi-task relative error checkpoint (**Epoch 7**): [checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt)

---

## 4. Epoch-by-Epoch `TRAIN` and `VAL` Learning Curves (`exp_s24`, Epochs 1–15)

| Epoch | Active LR | Opt Loss | `TRAIN_EVAL` (`2,048` / Full) | `VAL_EVAL` Global (`Batch`) | Gen Gap (`VAL - TRAIN`) | Pre-Clip Grad Norm (`Mean / Max`) | Clip `%` | `VAL_MT_REL_ERR` | `VAL` Wind MAE (`kt`) | `VAL` Major-Hurr MAE / Bias (`kt`) | `VAL` Press MAE (`hPa`) | `VAL` Stage Acc / Bal (`%`) | `VAL` $\Delta V$ `6/12/18/24h` (`kt`) | `VAL` RI PR-AUC / F1@0.5 / Brier | `VAL` $+24\text{h}$ Track / Skill (`km`) | `VAL` Storm Wind `[Mean, Std, Med]` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `3.000e-4` | `32.2725` | `18.7504` | `22.9118 (22.8844)` | `+4.1614` | `143.17 / 475.28` | `96.97%` | `0.6696` | `8.54` | `13.16 / -9.66` | `8.00` | `59.78 / 44.90` | `3.50 / 6.52 / 9.17 / 11.42` | `0.3599 / 0.0000 / 0.0465` | `218.52 (-5.37)` | `[8.92, 4.39, 9.13]` |
| **2** | `2.987e-4` | `18.7722` | `15.3666` | `18.4060 (18.4650)` | `+3.0394` | `138.29 / 444.95` | `100.0%` | `0.6187` | `7.75` | `9.66 / -5.14` | `7.09` | `64.95 / 56.76` | `3.40 / 6.21 / 8.69 / 10.72` | `0.4586 / 0.0215 / 0.0409` | `217.05 (-3.90)` | `[8.13, 3.83, 7.71]` |
| **3** | `2.949e-4` | `16.3220` | `14.8412` | `17.8836 (17.8977)` | `+3.0424` | `133.48 / 392.52` | `99.39%` | `0.6107` | `8.80` | `8.19 / +0.14` | `6.01` | `60.68 / 59.27` | `3.39 / 6.24 / 8.72 / 10.70` | `0.5553 / 0.3906 / 0.0371` | `215.99 (-2.84)` | `[9.09, 4.73, 8.55]` |
| **4** | `2.886e-4` | `14.3956` | `11.8086` | `18.8745 (18.9394)` | `+7.0659` | `115.41 / 330.72` | `100.0%` | `0.6047` | `7.72` | `10.90 / -8.77` | `6.81` | `62.88 / 62.07` | `3.22 / 5.97 / 8.31 / 10.20` | `0.5165 / 0.3252 / 0.0392` | `215.11 (-1.96)` | `[7.72, 3.51, 7.48]` |
| **5** | `2.800e-4` | `12.9608` | `10.8746` | `19.1198 (19.3298)` | `+8.2452` | `117.33 / 332.84` | `100.0%` | `0.5971` | `7.71` | `10.55 / -8.95` | `6.44` | `66.04 / 64.17` | `3.23 / 6.01 / 8.38 / 10.37` | `0.4914 / 0.3906 / 0.0388` | `214.14 (-0.99)` | `[7.99, 3.55, 7.51]` |
| **6** | `2.691e-4` | `11.6522` | `10.0077` | `16.6437 (16.6483)` | `+6.6360` | `102.55 / 235.62` | `100.0%` | `0.5757` | `7.24` | `9.38 / -6.08` | `5.89` | `68.24 / 67.14` | **`3.12 / 5.74 / 7.95 / 9.80`** | `0.5420 / 0.5550 / 0.0409` | `213.45 (-0.30)` | `[7.66, 3.76, 6.75]` |
| **7 ★MT** | `2.562e-4` | `10.6348` | `8.6160` (`8.6844`) | `16.1483 (16.1743)` | `+7.5323` (`+7.4639`) | `106.19 / 215.52` | `100.0%` | **`0.5531`** | `7.01` | `9.02 / -4.68` | **`5.45`** | `69.08 / 67.31` | `3.13 / 5.75 / 8.00 / 9.85` | **`0.6356 / 0.6010 / 0.0359`** | `213.00 (+0.15)` | `[7.44, 3.83, 7.73]` |
| **8** | `2.415e-4` | `9.6102` | `8.5316` | `18.3024 (18.3427)` | `+9.7708` | `96.70 / 241.39` | `100.0%` | `0.5847` | `7.14` | `10.31 / -8.59` | `5.92` | `68.24 / 67.13` | `3.15 / 5.91 / 8.24 / 10.13` | `0.5137 / 0.5756 / 0.0426` | `213.68 (-0.53)` | `[7.62, 4.12, 7.10]` |
| **9** | `2.253e-4` | `8.9867` | `7.2895` | `17.7703 (17.8628)` | `+10.4808` | `99.92 / 339.03` | `100.0%` | `0.5800` | `7.56` | `9.25 / -5.94` | `5.86` | `67.53 / 69.47` | `3.15 / 5.79 / 8.04 / 9.90` | `0.5485 / 0.5393 / 0.0410` | `213.26 (-0.11)` | `[7.94, 4.05, 7.74]` |
| **10 ★Loss** | `2.077e-4` | `8.3156` | `6.7283` (`6.8223`) | **`15.9896 (16.1211)`** | `+9.2613` (`+9.1673`) | `91.87 / 283.05` | `100.0%` | `0.5646` | **`6.85`** | **`7.68 / -2.67`** | `5.75` | **`70.63 / 71.42`** | `3.18 / 5.82 / 8.02 / 9.86` | `0.5511 / 0.5635 / 0.0393` | **`212.88 (+0.27)`** | `[7.45, 3.83, 7.54]` |
| **11** | `1.892e-4` | `7.3766` | `7.4538` | `17.2710 (17.2497)` | `+9.8172` | `86.02 / 187.31` | `100.0%` | `0.6154` | `7.37` | `9.06 / -4.90` | `6.10` | `67.79 / 67.98` | `3.30 / 6.08 / 8.58 / 10.52` | `0.5135 / 0.5079 / 0.0569` | **`212.88 (+0.27)`** | `[7.95, 3.20, 7.97]` |
| **12** | `1.700e-4` | `6.7811` | `6.6854` | `16.1684 (16.2289)` | `+9.4830` | `86.34 / 181.03` | `100.0%` | `0.5793` | `7.19` | **`7.51 / -2.19`** | `5.84` | `69.14 / 68.63` | `3.19 / 5.91 / 8.16 / 10.03` | `0.5225 / 0.5000 / 0.0415` | `213.33 (-0.18)` | `[7.80, 3.64, 7.80]` |
| **13** | `1.505e-4` | `6.2067` | `5.2559` | `17.3989 (17.3385)` | `+12.1430` | `82.23 / 226.44` | `100.0%` | `0.5739` | **`6.79`** | `9.41 / -7.47` | `5.76` | `68.43 / 68.33` | `3.26 / 6.04 / 8.36 / 10.30` | `0.5453 / 0.5333 / 0.0384` | `213.01 (+0.14)` | **`[7.38, 3.08, 6.85]`** |
| **14** | `1.310e-4` | `5.6892` | `4.6607` | `18.6403 (18.8698)` | `+13.9796` | `75.50 / 208.33` | `100.0%` | `0.5869` | `7.06` | `9.98 / -7.61` | `5.70` | `69.66 / 68.43` | `3.13 / 5.79 / 8.06 / 9.96` | `0.4612 / 0.5363 / 0.0462` | `213.41 (-0.26)` | `[7.60, 3.63, 7.17]` |
| **15** | `1.118e-4` | `5.2617` | `4.1874` | `19.3321 (19.4760)` | `+15.1447` | `71.89 / 150.20` | `100.0%` | `0.6022` | `7.24` | `10.20 / -7.92` | `5.94` | `67.72 / 68.93` | `3.26 / 6.01 / 8.30 / 10.24` | `0.4356 / 0.5217 / 0.0473` | `213.04 (+0.11)` | `[7.70, 3.75, 7.81]` |

### Key Scientific Insights from the 15-Epoch Learning Curves:
1. **Why Slower Mid-Schedule Cosine Decay Succeeded at Epochs 7–10 (`1,155–1,650` Steps):**
   - In the original 12-epoch schedule (`T_max=12`), the learning rate dropped too fast between Epoch 5 (`2.253e-4`) and Epoch 10 (`4.479e-5`), preventing the model from finding a joint minimum that simultaneously satisfied overall wind MAE, major-hurricane intensity, stage classification, and RI probability calibration.
   - Under `T_max=24`, keeping the learning rate in the `2.562e-4 -> 2.077e-4` band across Epochs 7–10 allowed the optimizer to escape the high-bias major-hurricane basin that plagued Original Epochs 7–11:
     - At **`S24` Epoch 7** (`active_lr = 2.562e-4`), `VAL` RI PR-AUC surged to **`0.6356`** (`+13.4%` absolute over Original Epoch 5's `0.5017`, and `+4.5%` over Original Epoch 7's `0.5902`) with `RI Brier = 0.0359`, `RI F1@0.5 = 0.6010`, `Pressure MAE = 5.45 hPa`, `Wind MAE = 7.01 kt`, and `VAL_MT_REL_ERR = 0.5531` (best across all experiments).
     - At **`S24` Epoch 10** (`active_lr = 2.077e-4`), the model achieved the **all-time lowest global validation loss (`VAL_EVAL_LOSS = 15.9896`)**, simultaneously outperforming **Original Epoch 5** on **every single one of the 6 forecasting tasks, including Major-Hurricane Wind MAE (`7.68 kt` vs. `7.78 kt`)**.
2. **Why Training Past Epoch 10 (`Epochs 11–15`, `>1,650` Steps) Degrades Generalization:**
   - Past Epoch 10 (`>1,650` optimizer steps), `TRAIN_EVAL` loss drops from `6.7283` (Epoch 10) to `4.1874` (Epoch 15), where the model memorizes the 75 training storms (`TRAIN` Wind MAE = `2.74 kt`, `TRAIN` Major-Hurricane MAE = `1.97 kt`, `TRAIN` RI PR-AUC = `0.9606`, `TRAIN` RI F1 = `0.9254`).
   - As `TRAIN` RI PR-AUC approaches `0.96`, `VAL` RI weighted loss inflates from `1.4822` (Epoch 7) to `3.0854` (Epoch 15) and `VAL` RI PR-AUC collapses from `0.6356` to `0.4356`, driving the `VAL - TRAIN` generalization gap to `+15.1447`.
   - **Conclusion on the Premature-Convergence Hypothesis:** The hypothesis that TC-GraphFormer needed *more than 12 epochs* of optimization is **refuted** (memorization dominates after `~1,650` optimizer steps / Epoch 10), **but** the hypothesis that the original **`T_max=12` cosine decay quenched the learning rate too aggressively between Epochs 5 and 10** is **proven true**.

---

## 5. Multi-Task Checkpoint Comparison Table (Phase D)

All six reference and candidate checkpoints below were evaluated on the exact same `10,545`-sequence `TRAIN` partition (`TRAIN_EVAL`, `model.eval()`) and `1,815`-sequence `VAL` partition (`VAL_EVAL`, `17` storms, `model.eval()`) using identical V1 `MultiTaskLoss` weights:

| Metric / Criterion | Original Epoch 5 (`best_checkpoint.pt`) | Original Epoch 11 (`multitask_epoch_011.pt`) | `C1` Cont. Epoch 8 (`exp_c1/best_checkpoint.pt`) | `T6` Decoupled-WD Ep 6 (`T6.../screen_epoch_006.pt`) | `S24` Epoch 7 (`best_mt_rel_err_checkpoint.pt`) | **`S24` Epoch 10 (`exp_s24/best_checkpoint.pt`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Schedule & LR at Checkpoint** | `T_max=12`, Ep 5 (`2.25e-4`) | `T_max=12`, Ep 11 (`2.10e-5`) | Ep 5 + 3 ep (`2.55e-5`, `wd=3e-4`) | Ep 5 + 1 ep (`5.00e-5`, `wd=1e-3` dec.) | `T_max=24`, Ep 7 (`2.56e-4`) | **`T_max=24`, Ep 10 (`2.08e-4`)** |
| **Cumulative Optimizer Steps** | `825` | `1,815` | `1,320` | `990` | `1,155` | **`1,650`** |
| **`TRAIN_EVAL` Global Loss (`10,545` seq)** | `13.0492` | `5.7588` | `7.9958` | `9.4193` | `8.6844` | **`6.8223`** |
| **`VAL_EVAL` Global Loss (`1,815` seq)** | `16.6384` | `17.1831` | `16.4052` | `16.2657` | `16.1483` | **`15.9896` (Best)** |
| **`VAL_EVAL` Batch-Avg Loss** | `16.7881` | `17.1745` | `16.4568` | `16.3869` | `16.1743` | **`16.1211` (Best)** |
| **Generalization Gap (`VAL - TRAIN`)** | **`+3.5892` (Lowest)** | `+11.4243` | `+8.4094` | `+6.8464` | `+7.4639` | `+9.1673` |
| **Composite `VAL_MT_REL_ERR` (6-task)** | `0.5877` | `0.5703` | `0.5723` | `0.5804` | **`0.5531` (Best)** | **`0.5646` (2nd Best)** |
| **`VAL` Weighted Intensity Loss** | `8.5792` | `9.1217` | `8.4333` | `8.3446` | `8.8353` | **`7.8232` (Best)** |
| **`VAL` Weighted Pressure Loss** | `0.0687` | `0.0633` | **`0.0591` (Best)** | `0.0683` | `0.0615` | `0.0610` |
| **`VAL` Weighted Stage Loss** | `0.4191` | `0.3852` | `0.3714` | `0.3830` | `0.3787` | **`0.3580` (Best)** |
| **`VAL` Weighted $\Delta V$ Loss** | `5.1818` | **`4.9369` (Best)** | `5.1652` | `5.0218` | `4.9500` | `5.0101` |
| **`VAL` Weighted RI Loss** | `1.9480` | `2.2345` | `1.9346` | `2.0055` | **`1.4822` (Best)** | `2.2962` |
| **`VAL` Weighted Track Loss** | `0.4416` | `0.4414` | `0.4416` | `0.4425` | **`0.4406` (Best)** | `0.4409` |
| **Task 1: Overall Wind MAE / RMSE (`kt`)** | `7.84 / 10.52` | `6.93 / 9.43` | `6.90 / 9.35` | `7.07 / 9.45` | `7.01 / 9.35` | **`6.85 / 9.33` (Best)** |
| **Task 1b: Major-Hurr (`\ge 96 kt`) MAE (`kt`)** | `7.78` | `9.28` | `8.68` | `8.21` | `9.02` | **`7.68` (Best)** |
| **Task 1c: Major-Hurr Bias (`kt`)** | **`-1.08` (Best)** | `-6.31` | `-4.21` | `-3.87` | `-4.68` | **`-2.67` (2nd Best)** |
| **Task 1d: 17-Storm Wind MAE `Mean ± Std (Med)`** | `8.19 ± 4.51 (8.03)` | `7.82 ± 4.04 (7.19)` | `7.71 ± 3.97 (7.56)` | `7.83 ± 3.50 (7.12)` | **`7.44 ± 3.83` (`7.73`)** | **`7.45 ± 3.83 (7.54)`** |
| **Task 2: Pressure MAE / RMSE (`hPa`)** | `5.99 / 8.29` | `5.54 / 7.96` | `5.53 / 7.69` | `6.06 / 8.27` | **`5.45 / 7.86` (Best)** | `5.75 / 7.82` |
| **Task 3: Stage Acc / Balanced Acc (`%`)** | `65.14 / 64.27` | `68.30 / 66.88` | `69.14 / 68.39` | `67.01 / 66.52` | `69.08 / 67.31` | **`70.63 / 71.42` (Best)** |
| **Task 4: $\Delta V$ MAE `+6/+12/+18/+24h` (`kt`)** | `3.22 / 5.96 / 8.27 / 10.18` | **`3.10 / 5.73 / 7.95 / 9.82`** | `3.16 / 5.91 / 8.27 / 10.26` | `3.13 / 5.81 / 8.09 / 9.98` | `3.13 / 5.75 / 8.00 / 9.85` | `3.18 / 5.82 / 8.02 / 9.86` |
| **Task 5: RI PR-AUC / Brier / F1@0.5** | `0.5017 / 0.0386 / 0.4526` | `0.5192 / 0.0396 / 0.5238` | `0.5226 / 0.0391 / 0.5568` | `0.5150 / 0.0390 / 0.5366` | **`0.6356 / 0.0359 / 0.6010`** | `0.5511 / 0.0393 / 0.5635` |
| **Task 6: Track MAE `+6/+12/+18/+24h` (`km`)** | `35.72 / 85.89 / 147.15 / 213.65` | `35.83 / 85.63 / 147.21 / 213.57` | `35.69 / 85.61 / 147.18 / 213.58` | `35.76 / 85.83 / 147.33 / 213.99` | `35.91 / 85.74 / 146.87 / 213.00` | **`35.89 / 85.74 / 146.83 / 212.88`** |
| **Track `+24h` Skill vs Causal Velocity (`213.15 km`)** | `-0.50 km` | `-0.42 km` | `-0.43 km` | `-0.84 km` | **`+0.15 km` (Positive)** | **`+0.27 km` (Best, Positive)** |

---

## 6. Best Candidate Decision (Phase D)

### **DECISION: ACCEPT `S24` Epoch 10 (`checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt`) as the Best Overall Multi-Task & Major-Hurricane Checkpoint, and Retain `S24` Epoch 7 (`checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt`) as the Best Well-Calibrated RI & Balanced-Generalization Checkpoint**

#### Scientific Justification Against All Guardrails:
1. **`S24` Epoch 10 (`checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt`) Dominates Original Epoch 5 on All 6 Tasks Simultaneously:**
   - **Global `VAL_EVAL_LOSS`:** `15.9896` vs. `16.6384` (`-0.6488` / **`-3.90%` lower validation loss**, the first checkpoint to break below `16.0`).
   - **Composite `VAL_MT_REL_ERR`:** `0.5646` vs. `0.5877` (`-0.0231` improvement).
   - **Overall Wind MAE / RMSE:** **`6.85 kt` / `9.33 kt`** vs. `7.84 kt` / `10.52 kt` (`-0.99 kt` / **`-12.6%` MAE reduction**; 17-storm mean wind MAE drops from `8.19 kt` to `7.45 kt`).
   - **Major-Hurricane (`\ge 96 kt`, `n=245`) Wind MAE / RMSE:** **`7.68 kt` / `9.46 kt`** vs. `7.78 kt` / `10.31 kt` (`-0.10 kt` better MAE and `-0.85 kt` better RMSE than Original Epoch 5, while beating `C1` Epoch 8 by `-1.00 kt` and Original Epoch 11 by `-1.60 kt`).
   - **Pressure MAE / RMSE:** **`5.75 hPa` / `7.82 hPa`** vs. `5.99 hPa` / `8.29 hPa` (`-0.24 hPa` MAE / `-0.47 hPa` RMSE improvement).
   - **Stage Classification Accuracy / Balanced Accuracy:** **`70.63%` / `71.42%`** vs. `65.14%` / `64.27%` (**`+5.49%` accuracy and `+7.15%` balanced accuracy gain** — the highest stage classification performance achieved in any run).
   - **Multi-Horizon Intensity Change ($\Delta V$) MAE:** `3.18 / 5.82 / 8.02 / 9.86 kt` vs. `3.22 / 5.96 / 8.27 / 10.18 kt` (improves every forecast lead time `+6h`, `+12h`, `+18h`, and `+24h`).
   - **Rapid Intensification (RI):** `PR-AUC = 0.5511` vs. `0.5017` (`+0.0494`), `F1@0.5 = 0.5635` vs. `0.4526` (`+0.1109`).
   - **Multi-Horizon Track Error & Positive Skill Over Causal Velocity:** `+24h Track MAE = 212.88 km` vs. `213.65 km` (`-0.77 km`), achieving **positive skill (`+0.27 km`) over the physical `213.15 km` causal velocity baseline**.
2. **Why `S24` Epoch 7 (`checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt`) is Also Retained:**
   - `S24` Epoch 7 achieves the **best composite multi-task relative error (`VAL_MT_REL_ERR = 0.5531`)**, the **best RI probability calibration and discrimination (`RI PR-AUC = 0.6356`, `RI Brier = 0.0359`, `RI F1@0.5 = 0.6010`)**, the **lowest 17-storm mean wind MAE (`7.44 kt`)**, and a smaller generalization gap (`+7.4639` vs. `+9.1673` at Epoch 10 and `+8.4094` at `C1` Epoch 8), while also beating the causal velocity track baseline (`213.00 km`, skill `+0.15 km`).
3. **Why Training Beyond Epoch 10 (`Epochs 11–15`) is Rejected:**
   - Extending training past `1,650` optimizer steps (`Epoch 10`) under `T_max=24` causes severe `TRAIN` memorization (`TRAIN_EVAL` drops to `4.1874`, `VAL_EVAL` rises to `19.3321`, generalization gap widens to `+15.1447`).

---

## 7. Recommendation on Additional Self-Supervised MFM Pretraining (Phase E)

### **RECOMMENDATION: YES — A Controlled 10-Epoch MFM Pretraining Experiment with Held-Out Pretraining-Validation Storms Is Scientifically Justified (Not Automatically Executed)**

1. **Empirical Evidence from Phase A & Phase C:**
   - In Phase A (Section 1.3), the 3-epoch MFM pretraining log ([pretraining_summary.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/pretraining_summary.json)) showed that masked feature reconstruction MSE was still dropping by **`-21.05%` per epoch** at Epoch 3 (`0.59527 -> 0.35587 -> 0.28097`), with encoder gradient norms (`0.147–0.200`) and parameter update norms (`0.016–0.071`) showing no sign of plateauing.
   - In Phase C, `S24` proved that the supervised heads and spatial-temporal backbone begin memorizing the `75` `TRAIN` storms' supervised labels after `~1,650` supervised gradient steps (`Epoch 10`). Because self-supervised MFM reconstructs `184,049` randomly masked spatial-modality tokens (`30%` mask ratio across `618,585` valid tokens) without touching the storm-level intensity/RI labels, stronger self-supervised representation learning is one of the few ways to improve cross-modality feature extraction before supervised label memorization sets in.
2. **Exact Controlled Experimental Protocol to Execute Next (Without Touching `VAL` or `FINAL_TEST`):**
   - **Pretraining Storm Split (Strictly Inside `TRAIN`):** Partition the `75` `TRAIN` storms into `65` `PRETRAIN_TRAIN` storms (`~9,100` overpasses) and `10` basin-stratified `PRETRAIN_VAL` storms (`~1,445` overpasses) so MFM reconstruction MSE (`mask_ratio = 0.30`, fixed deterministic validation mask seed) can be monitored on unseen storms after every pretraining epoch without touching the `17` supervised `VAL` storms or `17` `FINAL_TEST` storms.
   - **Budget & Schedule:** Run `10` MFM epochs (`~35 seconds/epoch` on RTX 4000 Ada, `<6 minutes` total) with `AdamW(lr=3.0e-4, weight_decay=1.0e-4)` and `CosineAnnealingLR(T_max=10, eta_min=1.0e-5)`, saving `pretrained_backbone_ep10_best.pt` at the lowest `PRETRAIN_VAL` masked token MSE.
   - **Controlled Downstream Verification:** Initialize `TCGraphFormer` from `pretrained_backbone_ep10_best.pt` and run the exact `S24` 10-epoch schedule (`CosineAnnealingLR(T_max=24)`, stopping at Epoch 10) to compare apples-to-apples against `S24` Epoch 10 (`VAL_EVAL_LOSS = 15.9896`, `VAL_MT_REL_ERR = 0.5646`).

---

## 8. Confirmed Input-Information Bottlenecks (Phase F)

Without rebuilding the 100 GB dataset, our code and NetCDF/HDF5 container audits confirm five structural information bottlenecks that bound validation performance regardless of optimizer schedule:

### 8.1 Verified Data & Representation Bottlenecks (Confirmed by Code and Container Inspection)
1. **Omission of High-Frequency `85–91 GHz` Ice-Scattering PMW Channels (`S3..S6` Swaths) on `48.3%` of All Overpasses (`6,656 / 13,789`):**
   - In [build_100gb_ Wilder/dataset.py / extraction pipeline](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/data/dataset.py), only swaths `S1` and `S2` were extracted into the 5-channel `pmw` tensor (`[37H, 37V, 85-91H, 85-91V, 89-91_ extra]`).
   - Direct inspection of the raw NetCDF files in `data/raw/tcprimed_100gb/` confirmed that:
     - **`ATMS` (`1,838` overpasses, `13.3%`):** `S1` (`16.5 km`) holds `23.8/31.4 GHz` and `S2` holds `50–57 GHz` temperature-sounding channels, while the `88.2 GHz` convective/eyewall channel (`TB_88.2QV`) resides in **`S3` (`16.5 km`)**, which was never extracted. Consequently, `pmw` channels `2..4` are `100%` zeros (`0.0`) on every `ATMS` overpass.
     - **`TMI` (`2,487`), `SSMIS` (`1,536`), `AMSR2` (`550`), `AMSRE` (`245`):** The `19/22/37 GHz` channels reside in `S1`/`S2`, while the `85.5 / 89.0 / 91.65 GHz` high-resolution (`~5 km`) ice-scattering channels—which resolve deep eyewall convection and concentric eyewall replacement cycles—reside in **`S3` or `S4`** and are therefore `100%` zeroed in the processed HDF5 containers.
2. **`64x64` Spatial Grid Compression (`~15.625 km/pixel` vs. Native `4–5 km` Eyewall Gradients):**
   - Every 2D spatial swath (`IR` native `301x301` at `~4 km`, `Radar` native `143x143` at `~5 km`, `GPROF` `101x101`) is bilinearly resampled onto a `64x64` grid spanning `~1,000 km x 1,000 km` (`~15.625 km` grid spacing), and then encoded by a stride-4 CNN (`kernel_size=7, stride=4`) into a `16x16` feature map (`~62.5 km` per token) before polar interpolation onto `72` polar graph nodes (`6` radial rings $\times$ `12` azimuth sectors).
   - **Physical Impact:** Compact major hurricanes (`RMW = 15–30 km`, such as `WP102018` Mangkhut, `WP181998` Zeb, and `SH062018` Berguitta) have their inner-core eye and eyewall temperature/reflectivity gradients smoothed across `1–2` pixels at `64x64` and sub-pixel averaged at `16x16` (`62.5 km`), directly explaining why inner-core intensity peaks and rapid intensification onset saturate on unseen compact cyclones.
3. **Omission of ERA5 3D Steering Flow and Vertical Wind Shear Vector Components (`18 -> 12` Channels):**
   - Raw TC PRIMED environmental files contain `18` rectangular variables, including 23-isobaric-level 3D `u_wind`, `v_wind`, `temperature`, `geopotential`, `relative_humidity`, and surface `u_wind_10m` / `v_wind_10m`.
   - The processed pipeline extracts `12` single-level/column channels and collapses the `33x33` environmental grid into a **spatial-mean + spatial-std vector (`24` scalars)** in [env_encoder.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/encoders/env_encoder.py), discarding both the **850–200 hPa vertical wind shear vector** (the #1 physical predictor of Rapid Intensification) and the **850–300 hPa deep-layer tropospheric steering wind vector** (the #1 physical determinant of tropical cyclone track motion), as well as all horizontal environmental gradients (trough/ridge asymmetry).
4. **Track Head Residual Bottleneck (`+0.27 km` Max Skill Over Causal Velocity Baseline):**
   - In [task_heads.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/heads/task_heads.py), the track head predicts residual displacement $\Delta \hat{\mathbf{x}}_{\text{res}}$ added to the causal backward velocity extrapolation $\mathbf{v}_{t_0} \Delta t$ (`213.15 km` `+24h` MAE on `VAL`, `n=270` sparse satellite-interval targets).
   - Because the `env_encoder` discards the 3D tropospheric steering flow (`u_wind`, `v_wind` across `850–300 hPa`) and spatial geopotential gradients, the shared latent representation $Z_{t_0}$ carries almost no synoptic steering-acceleration signal beyond current storm velocity $\mathbf{v}_{t_0}$. Consequently, track shared-backbone gradients account for only `0.31%` of the multi-task gradient norm, and the best learned residual (`S24` Epoch 10: `212.88 km`) improves upon the pure causal velocity baseline (`213.15 km`) by only `+0.27 km`.
5. **Short Sequence History (`T=4` Overpasses) and Sparse Asynchronous Overpass Intervals:**
   - `TCPRIMEDSequenceDataset` caps history at `max_history_len = 4` satellite overpasses within `72.0h`. Because polar-orbiting microwave/radar overpasses occur asynchronously (`1.5h` to `18h` apart), `4` overpasses often span only `6–15 hours` of storm evolution, limiting the Continuous-Time Temporal Transformer's ability to observe a full 24-hour diurnal convective cycle or multi-day structural organization prior to RI.

---

## 9. Exact Runtime and Reproducible Commands

### 9.1 Measured Runtime and Hardware Telemetry
- **GPU:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` total VRAM)
- **Peak Active GPU Memory (`nvidia-smi`):** `1,241–1,348 MiB` during FP16 mixed-precision training (`batch_size=32`, `accumulate_grad_batches=2`)
- **Per-Epoch Wall-Clock Time (`exp_s24`):** `425.2s – 566.2s` (`~7.3 minutes/epoch`, including `165` optimizer steps on `10,545` `TRAIN` sequences + `2,048`-sample `TRAIN_EVAL` + `1,815`-sample `VAL_EVAL` with 17-storm breakdown)
- **Total Wall-Clock Runtime for `exp_s24` (15 Epochs + 2 Full `10,545`-Sample `TRAIN_EVAL` Passes):** `7,183.4 seconds` (`119.7 minutes` / `1.995 hours`)

### 9.2 Reproducible PowerShell Commands
To reproduce the evaluation of the accepted `S24` Epoch 10 (`best_checkpoint.pt`) and `S24` Epoch 7 (`best_mt_rel_err_checkpoint.pt`) checkpoints against the reference baselines on `VAL` (`1,815` sequences, `17` storms):

```powershell
cd "D:\Enjula uchoi\TC-GraphFormer-verified"
$PY = "D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\Enjula uchoi\TC-GraphFormer-verified"

# Inspect full JSON summary of the fair 24-epoch longer-schedule experiment (S24)
& $PY -c "import json; d=json.load(open('reports/phase2e4_100gb/exp_s24_longer_schedule_summary.json')); print('Config Hash:', d['config_hash'], '| Best Loss Ep:', d['best_loss_epoch'], d['best_val_loss_global'], '| Best MT Ep:', d['best_mt_epoch'], d['best_val_mt_rel_err'], '| Early Stopped:', d['early_stopped'])"

# Run unit tests verifying loss mathematics and temporal attention masking
& $PY -m unittest tests/test_v2_loss_and_masks.py -v
```
