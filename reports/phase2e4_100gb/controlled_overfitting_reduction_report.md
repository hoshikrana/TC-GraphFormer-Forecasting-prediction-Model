# TC-GraphFormer — Controlled Overfitting Reduction Experiment Report (`exp_c2_head_dropout_p015`)

- **Timestamp (UTC):** `2026-11-24T16:30:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Git Branch:** `feat/tc-graphformer-v2-generalization`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, PyTorch `2.9.1+cu128`)
- **Dataset Partitions Used:** `TRAIN` (`75` storms, `10,545` sequences) and `VAL` (`17` storms, `1,815` sequences). **`FINAL_TEST` (`17` storms, `1,429` sequences) was strictly untouched (`0` accesses).**

---

## 1. Exact Regularization Change

- **Selected Intervention (Task 2, Option A — Task-Head Intermediate Dropout):**
  - Inserted `nn.Dropout(p=head_dropout)` with **`head_dropout = 0.15`** inside `TaskPredictionHeads._apply_head()` (`src/models/heads/task_heads.py`) immediately after the `GELU` activation and before the final linear output projection across all six 2-layer task heads:
    $$\mathbf{h}_k = \text{Dropout}_{p=0.15}\!\left(\text{GELU}(\mathbf{W}_{1,k}\mathbf{z}_T + \mathbf{b}_{1,k})\right) \in \mathbb{R}^{64}, \qquad \hat{\mathbf{y}}_{\text{raw},k} = \mathbf{W}_{2,k}\mathbf{h}_k + \mathbf{b}_{2,k}$$
  - **Preserved Invariants:**
    - Zero changes to tensor shapes, parameter count (`2,977,797` trainable parameters, `333` `state_dict` keys), output physical units, pilot climatological bias priors (`38.19 kt`, `997.25 hPa`, `dv = [0.231, 0.575, 1.075, 1.596] kt`, `ri = -3.5124`, causal damped velocity track prior), or output multipliers (`12x`, `12x`, `5x`, `20x`).
    - Zero normalization layers added to the heads.
    - Verified via unit test `test_head_dropout_train_vs_eval_and_state_dict_compatibility` (`tests/test_v2_loss_and_masks.py`) that `model.eval()` disables dropout completely (`0.0` inference-time dropout, producing bit-identical outputs to `head_dropout = 0.0` for identical weights).

---

## 2. Reference and Candidate Configurations

| Configuration Item | **Reference Run (`exp_c1_low_lr_reg_continuation`)** | **Candidate Run (`exp_c2_head_dropout_p015`)** |
| :--- | :--- | :--- |
| **Initial Checkpoint** | `checkpoints/tcprimed_100gb/multitask_epoch_005.pt` (Epoch 5) | `checkpoints/tcprimed_100gb/multitask_epoch_005.pt` (Epoch 5) |
| **Restored Optimizer & AMP State** | All `218` AdamW momentum/variance buffers + `GradScaler` (`scale=512.0`) | All `218` AdamW momentum/variance buffers + `GradScaler` (`scale=512.0`) |
| **Continuation Budget & Schedule** | `4` epochs (Epochs `6..9`), `CosineAnnealingLR(lr=5.0e-5 -> 1.0e-6)` | `4` epochs (Epochs `6..9`), `CosineAnnealingLR(lr=5.0e-5 -> 1.0e-6)` |
| **Optimizer & Weight Decay** | `AdamW(lr=5.0e-5, weight_decay=3.0e-4)`, `grad_clip_norm=1.0` | `AdamW(lr=5.0e-5, weight_decay=3.0e-4)`, `grad_clip_norm=1.0` |
| **Batch Size & Accumulation** | `batch_size=32`, `accumulate_grad_batches=2` (effective `64`), `fp16` AMP | `batch_size=32`, `accumulate_grad_batches=2` (effective `64`), `fp16` AMP |
| **Loss Function (`MultiTaskLoss`)** | V1 Quadratic wind regime $w_{\text{reg}} \in [1, 11]$, `ri_pos_weight=18.1`, `prior_shift=0.85`, weights `(1.0, 0.5, 0.5, 1.0, 2.0, 1.0)` | **Identical V1 `MultiTaskLoss`** |
| **Backbone & Temporal Config** | `d_model=128`, `max_history_len=4`, `max_history_hours=72.0`, `dropout=0.1` | **Identical Backbone & Temporal Config** |
| **Task-Head Dropout (`head_dropout`)** | **`0.00`** (No task-head dropout) | **`0.15`** (Single controlled difference) |
| **Random Seed** | `seed = 42` | `seed = 42` |
| **Isolated Checkpoint Directory** | `checkpoints/tcprimed_100gb_exp_c1/` | `checkpoints/tcprimed_100gb_exp_c2_head_drop015/` |

---

## 3. Train/VAL Comparison Across Epochs and Reference Baselines

All `TRAIN_EVAL` (`10,545` sequences, `75` storms) and `VAL_EVAL` (`1,815` sequences, `17` storms) metrics below are computed in **`model.eval()` mode** under the **exact same global valid-target-normalized V1 `MultiTaskLoss`**:

### 3.1 C2 Per-Epoch Progression (`Epochs 6–9`, `reports/phase2e4_100gb/exp_c2_head_dropout_summary.json`)

| C2 Epoch | LR | `OptLoss (Train Mode)` | **`TRAIN_EVAL` (Global)** | **`VAL_EVAL` (Global / Batch)** | **`GEN_GAP` (`VAL - TRAIN`)** | Pre-Clip Grad Norm (`Mean / Max`, `% Clipped`) | **`VAL_MT_REL_ERR`** | `VAL` Wind MAE (`Bias`) | **`VAL` Major-Hurr MAE (`Bias`)** | `VAL` Press MAE | `VAL` Stage Acc (`Bal %`) | `VAL` $\Delta V$ `6/12/18/24h` (`kt`) | `VAL` RI `PR-AUC / Brier / F1@0.5` | `VAL` Track `+24h` (`Skill vs CausalVel`) | Runtime / Peak VRAM |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **6** | `5.00e-5` | `12.2675` | `9.0661` | `17.4310` (`17.5179`) | `+8.3649` | `97.06 / 239.86` (`100%`) | `0.5772` | `7.14` (`-2.22`) | `9.32` (`-7.02`) | `6.30` | `68.43%` (`67.46%`) | `3.14 / 5.82 / 8.11 / 10.02` | **`0.5369` / `0.0383`** / `0.4967` | `213.79` (`-0.64 km`) | `614.5s` / `1148 MiB` |
| **7** | `4.28e-5` | `11.2957` | `8.5088` | `17.0555` (`17.1612`) | `+8.5467` | `92.02 / 222.21` (`100%`) | `0.5931` | `7.05` (`-1.47`) | `9.02` (`-5.24`) | `5.93` | `67.40%` (`66.64%`) | `3.19 / 5.92 / 8.28 / 10.29` | `0.4812` / `0.0447` / `0.4526` | `213.50` (`-0.35 km`) | `474.8s` / `1093 MiB` |
| **8 (Best Loss)** | `2.55e-5` | `10.8454` | `8.2598` | **`16.5892` (`16.6288`)** | **`+8.3294`** | `86.60 / 196.58` (`100%`) | `0.5791` | `7.01` (`-0.60`) | **`8.78` (`-5.19`)** | **`5.74`** | `68.75%` (`67.93%`) | `3.16 / 5.89 / 8.23 / 10.23` | `0.5140` / `0.0412` / `0.5111` | **`213.38` (`-0.23 km`)** | `476.5s` / `1094 MiB` |
| **9 (Best MT)** | `8.18e-6` | `10.4051` | `7.9440` | `17.0216` (`17.1115`) | `+9.0776` | `83.05 / 192.00` (`100%`) | **`0.5770`** | **`6.91`** (`-1.37`) | `9.30` (`-5.89`) | `5.84` | **`69.21%`** (`67.91%`) | `3.14 / 5.86 / 8.19 / 10.17` | `0.5201` / `0.0411` / `0.5111` | `213.52` (`-0.37 km`) | `665.1s` / `1239 MiB` |

### 3.2 Head-to-Head Comparison: Original Epoch 5 vs. Reference C1 Epoch 8 vs. Candidate C2 Epoch 8 & 9

| Metric (`TRAIN_EVAL` / `VAL_EVAL`) | **Original Epoch 5** (`best_checkpoint.pt`) | **Reference C1 Epoch 8** (`head_dropout=0.0`) | **Candidate C2 Epoch 8** (`head_dropout=0.15`, Best Loss) | **Candidate C2 Epoch 9** (`head_dropout=0.15`, Best MT) | **Effect of `head_dropout=0.15` (C2 Ep 8 vs. C1 Ep 8)** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Optimization Loss (`model.train()`)** | `12.4200` | `8.6683` | `10.8454` | `10.4051` | `+2.1771` higher train-mode loss due to dropout noise |
| **`TRAIN_EVAL` Global Loss (`model.eval()`)** | `13.0492` | `7.9958` | `8.2598` | `7.9440` | `+0.2640` higher eval-mode `TRAIN` loss at Ep 8 |
| **`VAL_EVAL` Global Loss (`model.eval()`)** | `16.6384` | **`16.4052`** | `16.5892` | `17.0216` | **`+0.1840` worse `VAL_EVAL` loss (`16.5892` vs `16.4052`)** |
| **Generalization Gap (`VAL_EVAL - TRAIN_EVAL`)** | **`+3.5892`** | `+8.4094` | **`+8.3294`** | `+9.0776` | `-0.0800` gap reduction at Ep 8, solely from raising `TRAIN_EVAL` |
| **`VAL_MT_REL_ERR` (6-Task Relative Error)** | `0.5925` | **`0.5723`** | `0.5791` | `0.5770` | **`+0.0068` worse at Ep 8 (`+0.0047` worse at Ep 9)** |
| **Overall Wind MAE (`TRAIN` / `VAL`, kt)** | `6.65` / `7.84` | `4.19` / **`6.90`** | `4.30` / `7.01` | `4.10` / `6.91` | `+0.11 kt` worse on `VAL` at Ep 8 (`+0.01 kt` at Ep 9) |
| **Overall Wind Bias (`TRAIN` / `VAL`, kt)** | `+3.12` / `+1.36` | `+0.69` / `-0.75` | `+0.77` / **`-0.60`** | `-0.22` / `-1.37` | `-0.15 kt` smaller overall bias magnitude at Ep 8 |
| **Major-Hurricane ($\ge 96\text{ kt}$) MAE (`TRAIN` / `VAL`, kt)** | `6.64` / **`7.78`** | `3.69` / **`8.68`** | `3.79` / `8.78` | `3.49` / `9.30` | **`+0.10 kt` worse at Ep 8 (`+0.62 kt` worse at Ep 9)** |
| **Major-Hurricane ($\ge 96\text{ kt}$) Bias (`TRAIN` / `VAL`, kt)** | `+4.71` / **`-1.08`** | `+1.98` / **`-4.21`** | `+1.36` / `-5.19` | `+0.27` / `-5.89` | **`-0.98 kt` more negative bias on `VAL` Major Hurricanes** |
| **Pressure MAE (`TRAIN` / `VAL`, hPa)** | `5.18` / `5.99` | `4.69` / **`5.53`** | `4.73` / `5.74` | `4.94` / `5.84` | **`+0.21 hPa` worse on `VAL` at Ep 8** |
| **Stage Balanced Accuracy (`TRAIN` / `VAL`, %)** | `70.62%` / `64.27%` | `75.20%` / **`68.39%`** | `75.33%` / `67.93%` | `75.79%` / `67.91%` | `-0.46%` lower balanced stage accuracy on `VAL` |
| **$\Delta V$ `+6 / +12 / +18 / +24h` `VAL` MAE (kt)** | `3.22 / 5.96 / 8.27 / 10.18` | `3.16 / 5.91 / 8.27 / 10.26` | `3.16 / 5.89 / 8.23 / 10.23` | **`3.14 / 5.86 / 8.19 / 10.17`** | **`-0.03 kt` (`Ep 8`) to `-0.09 kt` (`Ep 9`) better `+24h` $\Delta V$ MAE** |
| **RI `VAL` PR-AUC / Brier (`92` pos / `1,484` neg)** | `0.5017 / 0.0386` | **`0.5226` / `0.0391`** | `0.5140 / 0.0412` | `0.5201 / 0.0411` (`0.5369 / 0.0383` @ Ep 6) | `-0.0086` lower PR-AUC & `+0.0021` worse Brier at Ep 8 (peaked at Ep 6: `0.5369 / 0.0383`) |
| **Track `+24h` `VAL` MAE (`Skill vs CausalVel`, km)** | `213.65` (`-0.50`) | `213.58` (`-0.43`) | **`213.38` (`-0.23`)** | `213.52` (`-0.37`) | **`-0.20 km` better `+24h` track error at Ep 8** |
| **Per-Storm `VAL` Wind MAE `Mean ± Std (Median)` (kt)** | `8.19±4.51 (8.03)` | **`7.71±3.97 (7.56)`** | `7.74±4.07 (7.38)` | `7.71±3.82 (7.36)` | Storm-median improved by `-0.18` to `-0.20 kt`, storm-mean unchanged |
| **Per-Storm `VAL` $\Delta V_{24\text{h}}$ MAE `Mean ± Std (Median)` (kt)** | `9.94±4.48 (9.72)` | `9.98±4.18 (10.81)` | `9.90±4.26 (10.91)` | **`9.89±4.36 (10.74)`** | `-0.08` to `-0.09 kt` improvement in per-storm mean $\Delta V_{24\text{h}}$ MAE |
| **Per-Storm `VAL` Track `+24h` `Mean ± Std (Median)` (km)** | `243.30±85.45 (209.75)` | `243.19±86.49 (209.73)` | `243.33±86.93 (209.10)` | `243.34±86.97 (209.21)` | `-0.63 km` improvement in per-storm median `+24h` track error |

---

## 4. Actual Performance Changes, Major-Hurricane / RI Results, and Uncertainties

1. **What Task-Head Dropout (`p = 0.15`) Actually Did:**
   - **During training (`model.train()`):** Dropping 15% of the 64 hidden activations inside each task head before the $12\times$ (`wind`, `pressure`), $5\times$ (`intensity_change`), and $20\times$ (`track`) affine output multipliers injected multiplicative variance into the head predictions, raising optimization loss from `8.6683` (C1 Ep 8) to `10.8454` (C2 Ep 8) and raising pre-clipping gradient norms from `~60` to `83.05–97.06` (`100%` of steps clipped at `grad_clip_norm = 1.0`).
   - **On `VAL` forecasting tasks:** Task-head dropout slightly improved **future `+24h` intensity change** (`10.26 -> 10.23 kt` at Ep 8, `10.17 kt` at Ep 9, and `10.02 kt` at Ep 6), **future `+24h` track error** (`213.58 -> 213.38 km` at Ep 8), **per-storm median Wind MAE** (`7.56 -> 7.36 kt`), and **Epoch-6 RI ranking/calibration** (`PR-AUC = 0.5369`, `Brier = 0.0383` at Ep 6).
   - **Why it failed to reduce the true `model.eval()` generalization gap or beat C1 overall:** Because the 6 task heads contain only **`50,836` parameters (`1.71%` of the `2,977,797` total parameters)** while the shared spatial-graph and temporal encoders contain **`2,926,961` parameters (`98.29%`)**, regularizing only the 64-D head bottleneck did not prevent the shared backbone from fitting the `75` training storms (`TRAIN_EVAL` still dropped to `8.2598` at Ep 8 and `7.9440` at Ep 9). Meanwhile, stochastic head dropout during training slightly attenuated the extreme upper-tail outputs of `head_intensity`, shifting `VAL` Major-Hurricane ($\ge 96\text{ kt}$) bias from `-4.21 kt` (C1 Ep 8) to `-5.19 kt` (C2 Ep 8, `Maj MAE = 8.78 kt`) and `-5.89 kt` (C2 Ep 9, `Maj MAE = 9.30 kt`).
2. **Inter-Storm Uncertainty:**
   - Across the `17` independent validation storms, per-storm Wind MAE standard deviation is **`3.82–4.07 kt`** (standard error of the storm mean $\approx 0.93\text{--}0.99\text{ kt}$), per-storm `+24h` $\Delta V$ MAE standard deviation is **`4.26–4.36 kt`** ($\text{SE} \approx 1.05\text{ kt}$), and per-storm `+24h` Track MAE standard deviation is **`86.9 km`** ($\text{SE} \approx 21.1\text{ km}$). Thus, the `0.01–0.11 kt` overall wind differences and `0.03–0.09 kt` $\Delta V_{24\text{h}}$ differences between C1 and C2 are well within 1 inter-storm standard error.

---

## 5. Scientific Limitations (Task 5)

1. **Effective Sample Size is `109` Independent Storms, Not `13,789` Independent Systems:** Successive satellite overpasses of the same tropical cyclone (`10,545` `TRAIN`, `1,815` `VAL`, `1,429` `FINAL_TEST`) exhibit strong temporal autocorrelation in intensity, pressure, environmental SST/TPW, and storm structure. All generalization gaps are driven primarily by **storm-to-storm structural shift** across the `75` `TRAIN` storms and `17` `VAL` storms rather than i.i.d. sample noise.
2. **Sparse Regional Validation Storm Counts in `SH` and `CP`:** In `VAL` (`17` storms), the Southern Hemisphere (`SH`, `26` overpasses) and Central Pacific (`CP`, `95` overpasses) are each represented by only **`1` independent cyclone**, creating high regional variance compared to `WP` (`5` storms, `423` overpasses) or `AL` (`3` storms, `644` overpasses).
3. **Confounded Attribution in the Previously Rejected V2 Experiment:** The earlier rejected V2 run (`tc_graphformer_v2_candidate_b`) simultaneously modified the wind regime exponent/bound ($[1, 11] \to [1, 3]$), task loss weights $(w_p, w_{\Delta V}, w_{\text{tr}})$, and the RI prior-shift factor ($0.85 \to 1.00$), limiting single-factor causal attribution in that run. By contrast, `exp_c2_head_dropout_p015` isolated **one** hyperparameter (`head_dropout: 0.0 -> 0.15`) against `exp_c1`.
4. **Single-Seed Limitation:** Both `exp_c1` and `exp_c2` were run under a single fixed random seed (`seed = 42`); multi-seed replication (or storm-block cross-validation) would be required to establish statistically significant superiority when metric differences are $< 0.2\text{ kt}$.

---

## 6. Best Checkpoint Paths & Final Acceptance Decision (Task 4)

### Saved Checkpoint Paths (All Preserved Intact)
- **Original Production Baseline (Epoch 5):** [checkpoints/tcprimed_100gb/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb/best_checkpoint.pt) (`VAL_EVAL = 16.6384`, `GEN_GAP = +3.5892`, `Major-Hurricane MAE = 7.78 kt`, `Bias = -1.08 kt`)
- **Reference C1 Continuation Best (Epoch 8):** [checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt) (`VAL_EVAL = 16.4052`, `VAL_MT_REL_ERR = 0.5723`, `Wind MAE = 6.90 kt`, `Major-Hurricane MAE = 8.68 kt`, `Pressure MAE = 5.53 hPa`, `Stage Acc = 69.14%`, `RI PR-AUC = 0.5226`)
- **Candidate C2 Head-Dropout Checkpoints (Isolated):**
  - Lowest `VAL_EVAL` Loss (Epoch 8): [checkpoints/tcprimed_100gb_exp_c2_head_drop015/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_c2_head_drop015/best_checkpoint.pt) (`VAL_EVAL = 16.5892`, `GEN_GAP = +8.3294`, `Wind MAE = 7.01 kt`, `Major-Hurricane MAE = 8.78 kt`, `Track24 = 213.38 km`)
  - Lowest `VAL_MT_REL_ERR` (Epoch 9): [checkpoints/tcprimed_100gb_exp_c2_head_drop015/best_mt_rel_err_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_c2_head_drop015/best_mt_rel_err_checkpoint.pt) (`VAL_MT_REL_ERR = 0.5770`, `Wind MAE = 6.91 kt`, `dV24 MAE = 10.17 kt`, `RI PR-AUC = 0.5201`)

### Decision: **REJECT `exp_c2_head_dropout_p015` AS A REPLACEMENT MODEL — KEEP EXISTING `Epoch 5` AND `C1 Epoch 8` CHECKPOINTS**

- **Reason for Rejection Against Predeclared Guardrails:**
  1. **Guardrail 1 & 5 (`VAL_EVAL` Skill vs. `TRAIN_EVAL` Loss):** At Epoch 8, `head_dropout = 0.15` reduced the `VAL - TRAIN` eval-mode loss gap by only `-0.0800` (`+8.3294` vs. `+8.4094`), and that reduction came **entirely from raising `TRAIN_EVAL` loss (`7.9958 -> 8.2598`) while `VAL_EVAL` loss worsened by `+0.1840` (`16.4052 -> 16.5892`)** and `VAL_MT_REL_ERR` worsened from `0.5723` $\rightarrow$ `0.5791` (`0.5770` at Ep 9).
  2. **Guardrail 2 (Major-Hurricane Accuracy):** `VAL` Major-Hurricane ($\ge 96\text{ kt}$) Wind MAE worsened from `8.68 kt` (bias `-4.21 kt`) in C1 to `8.78 kt` (bias `-5.19 kt`) at C2 Epoch 8 and `9.30 kt` (bias `-5.89 kt`) at C2 Epoch 9.
  3. **Guardrail 3 (RI Discrimination & Calibration at Selected Checkpoint):** At C2 Epoch 8, RI `PR-AUC` (`0.5140` vs. `0.5226`) and `Brier` score (`0.0412` vs. `0.0391`) both trailed C1 Epoch 8.

---

## 7. One Concrete Next Research Step

- **Target the Shared Spatiotemporal Backbone (`98.29%` of parameters) via Storm-Level Spatial/Graph DropEdge or Encoder Decoupled Weight Decay Rather Than the 64-D Head Bottleneck:**
  - Our controlled experiment proves that regularizing only the `50,836`-parameter `TaskPredictionHeads` (`1.71%` of parameters) cannot stop the `2.93M`-parameter shared CNN/Graph/Transformer backbone from memorizing the `75` training storms (`TRAIN_EVAL` still falls to `7.94–8.26`), while head dropout adds noise right before the $12\times$ wind multiplier and slightly increases negative bias on Major Hurricanes.
  - Therefore, the single highest-value next controlled experiment is **Option B (Decoupled Weight Decay on the Shared Backbone Only: increasing `weight_decay` from `3.0e-4` to `1.0e-3` strictly on 2D/3D convolutional, graph message-passing, and attention projection weights while excluding all `bias`, `LayerNorm`, `BatchNorm`, and `TaskPredictionHeads` parameters)** from `multitask_epoch_005.pt` under the exact V1 loss. This regularizes the `2.93M`-parameter shared feature extractor without attenuating the extreme intensity output projections in `TaskPredictionHeads`.
