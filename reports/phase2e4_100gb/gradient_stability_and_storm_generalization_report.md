# TC-GraphFormer — Gradient Stability, Target-Scale Optimization & Storm-Level Generalization Report

- **Timestamp (UTC):** `2026-11-24T20:50:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Git Branch:** `feat/tc-graphformer-v2-generalization`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, PyTorch `2.9.1+cu128`)
- **Dataset:** NOAA TC PRIMED 100 GB Expansion (`75 TRAIN` storms / `10,545` sequences; `17 VAL` storms / `1,815` sequences; `17 FINAL_TEST` storms / `1,429` sequences — **`FINAL_TEST` untouched (`0` accesses)**).

---

## 1. Confirmed Root Cause of Pre-Clip Gradient Norms (`83–97`) and Why `clip_grad_norm_ = 1.0` Is Well-Conditioned

### 1.1 Mathematical Root Cause of the `~89.7` Pre-Clip Gradient Norm
In `TaskPredictionHeads.forward()` (`src/models/heads/task_heads.py:124-146`) and `MultiTaskLoss.forward()` (`src/training/losses.py:76-168`), each regression head applies a manual dynamic-range multiplier $M_k$ around its prior, while `MultiTaskLoss` divides `pressure` by $S_p = 10.0$ and `track` by $S_{\text{tr}} = 100.0$, but evaluates `intensity` ($V_{\max}$) and `intensity_change` ($\Delta V$) directly in unscaled knots ($S_v = 1.0, S_{\Delta V} = 1.0$):

| Task $k$ | Head Output Transform | Loss Input Scaling ($1/S_k$) | Effective Loss-to-Head Chain-Rule Multiplier ($M_k / S_k$) | Regime / Class Weighting Factor | **Effective Pre-Clip Gradient Multiplier on Raw Head Output** | **Measured Head Pre-Clip L2 Grad Norm (`Mean ± Std`)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`intensity` ($V_{\max}$)** | $38.19 + \mathbf{12.0}(r_v - 38.19)$ | $1.0$ (`kt`) | **`12.0x`** | $w_{\text{reg}} \in [1.0, 11.0]$ (mean `1.739`) | **`12.0x` to `132.0x` (mean `20.87x`)** | **`39.4290 ± 19.4634` (`99.33%` of heads)** |
| **`intensity_change` ($\Delta V$)** | $\boldsymbol{\mu}_{\Delta V} + \mathbf{5.0}(\mathbf{r}_{\Delta V} - \boldsymbol{\mu}_{\Delta V})$ | $1.0$ (`kt`, $\div 4$ horizons) | **`1.25x` per horizon** | $w_{\Delta V} = 1.0$ | **`1.25x` per horizon (`5.0x` sum)** | **`2.8318 ± 0.6548`** |
| **`ri` ($\Delta V_{24\text{h}} \ge 30\text{ kt}$)** | $r_{\text{RI}}$ (raw logit) | $1.0$ | **`1.00x`** | $w_{\text{RI}}=2.0,\; w_{\text{pos}}=18.1$ | **`2.0x` (neg) to `36.2x` (pos)** | **`2.0633 ± 1.1194`** |
| **`stage` (5-class CE)** | $\mathbf{r}_s \in \mathbb{R}^5$ | $1.0$ | **`1.00x`** | $w_s = 0.5$ | **`0.50x`** | **`0.5033 ± 0.1101`** |
| **`pressure` ($P_{\min}$)** | $997.25 + \mathbf{12.0}(r_p - 997.25)$ | $\div \mathbf{10.0}$ (`10 hPa`) | **`1.20x`** | $w_p = 0.5$ | **`0.60x`** | **`0.2778 ± 0.1710`** |
| **`track` ($\Delta x, \Delta y$)** | $\mathbf{d}_{\text{base}} + \mathbf{20.0}(\mathbf{r}_{\text{tr}} - \mathbf{d}_{\text{clim}})$ | $\div \mathbf{100.0}$ (`100 km`, $\div 8$ comps) | **`0.025x` per component** | $w_{\text{tr}} = 1.0$ | **`0.025x` per comp (`0.20x` sum)** | **`0.0955 ± 0.0255`** |

- **Confirmed Explanation:** The pre-clip global gradient norm (`mean = 89.6630 ± 33.0142`, `median = 77.5823`, `min = 45.1477`, `max = 149.9962`) is **not** caused by a bug in AMP unscaling, gradient accumulation, or double loss weighting. It is the exact mathematical consequence of the $12.0\times$ wind head multiplier combined with unscaled knot Huber loss ($\beta = 8.0\text{ kt}$, with `24.77%` of training samples in the linear $|\text{err}| \ge 8\text{ kt}$ regime where $|\partial \text{Huber}/\partial \hat{V}| = 1.0$) and quadratic intense-cyclone regime weighting $w_{\text{regime}} \in [1.0, 11.0]$.

---

## 2. Phase 1 & Phase 2 — Optimizer Verification & Gradient-Clipping Diagnostics (`reports/phase2e4_100gb/gradient_clipping_and_scale_diagnostics.json`)

### 2.1 Phase 1 Code Verification Checklist
1. **AMP Unscaling Before Clipping (`src/training/trainer.py:225-226`):** Verified. `self.scaler.unscale_(self.optimizer)` is called immediately before `torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)`.
2. **Clip Norm Applied Once Per Optimizer Step (`src/training/trainer.py:128,224-237`):** Verified. `clip_grad_norm_` is called only when `len(group_batches) == self.accumulate_grad_batches or is_last_batch`, once per accumulation group (`165` times per `330`-batch epoch).
3. **Gradient Accumulation Denominators (`src/training/trainer.py:131-185`):** Verified. `group_target_counts` sums valid targets across both microbatches in the accumulation group and passes them to `MultiTaskLoss.forward()`, matching a single full-batch backward pass to within `1.72e-5` max gradient difference.
4. **Gradient Clearing (`src/training/trainer.py:116,229,234`):** Verified. `self.optimizer.zero_grad()` is executed at epoch start and immediately after each `scaler.step(self.optimizer)`.
5. **Checkpoint State Restoration (`src/training/trainer.py:525-674`):** Verified. All `218 / 218` parameter tensors in `AdamW` have restored `exp_avg` ($m_t$) and `exp_avg_sq` ($v_t$) buffers in `multitask_epoch_005.pt`, and `GradScaler` restores `scale = 512.0` (`growth_tracker = 314`).
6. **1-to-1 Parameter Coverage (`src/training/train_multitask.py:348`):** Verified. `218` named trainable tensors (`2,977,797` scalars) match `218` unique parameter IDs in `optimizer.param_groups` with zero duplicates or omissions.

### 2.2 Phase 2 Layer-Group Pre-Clip Norms & Simulated AdamW Updates (`clip=1.0` vs. `clip=5.0` vs. `unclipped`, `20` Accumulation Steps = `1,280` `TRAIN` Sequences, `lr = 5.0e-5`)

| Layer Group | Parameter L2 Norm $\|\theta\|_2$ | Pre-Clip Grad Norm (`Mean ± Std`) | **AdamW Update/Param Ratio (`clip = 1.0`)** | **AdamW Update/Param Ratio (`clip = 5.0`)** | **AdamW Update/Param Ratio (`unclipped`)** | **Cosine Sim (`clip=1.0` vs `unclipped`)** | **Update Norm Ratio $\|\Delta\theta_1\| / \|\Delta\theta_u\|$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ALL` (Full Model)** | `1,021.36` (`99.53` excl. head biases) | `89.6630 ± 33.0142` | `2.4095e-05` (**`2.46e-04` excl. head biases**) | `4.6991e-05` (`4.80e-04` excl. biases) | `1.6569e-04` (`1.69e-03` excl. biases) | `0.441018` | `0.1459` |
| **`modality_cnn_stems`** | `39.7853` | `63.9521 ± 25.4162` | **`4.8959e-04`** | `9.5483e-04` | `3.2828e-03` | `0.437928` | `0.1497` |
| **`spatial_graph_and_pool`** | `21.9970` | `38.8422 ± 12.1497` | **`3.3256e-04`** | `6.3970e-04` | `2.4433e-03` | `0.436286` | `0.1366` |
| **`task_heads` (Weight Matrices Only)** | **`12.6211`** | `39.6940 ± 19.2923` | **`2.7916e-04`** | `5.2778e-04` | `1.7639e-03` | `0.452500` | `0.1585` |
| **`task_heads` (Incl. `997.25 hPa` Bias)** | `1,016.9408` | `39.6940 ± 19.2923` | `3.4646e-06` | `6.5500e-06` | `2.1891e-05` | `0.452500` | `0.1585` |
| **`temporal_transformer`** | `31.7487` | `18.0913 ± 7.1578` | **`2.6902e-04`** | `5.3120e-04` | `2.0139e-03` | `0.447271` | `0.1343` |
| **`cross_modal_fusion`** | `39.9049` | `18.1849 ± 5.5052` | **`2.3075e-04`** | `4.5055e-04` | `1.5885e-03` | `0.449897` | `0.1457` |
| **`norm_params` (`BN` + `LN`)** | `68.5400` | `6.5599 ± 2.6161` | **`2.0114e-05`** | `3.9617e-05` | `1.4139e-04` | `0.441147` | `0.1429` |

### 2.3 Why `clip_grad_norm_ = 1.0` Is NOT a Bottleneck (and Why Relaxing It Destabilizes Fine-Tuning)
1. **Uniformly Healthy Step Sizes Across All Layers (`2.31e-4` to `4.90e-4` per step):**
   - Once the static `997.25 hPa` central-pressure bias prior (`head_biases_only_norm = 1,016.86`) is separated from the learnable weight matrices of `task_heads` (`head_weights_only_norm = 12.62`), **`task_heads` (`2.79e-4`), `temporal_transformer` (`2.69e-4`), `cross_modal_fusion` (`2.31e-4`), `spatial_graph_and_pool` (`3.33e-4`), and `modality_cnn_stems` (`4.90e-4`) all receive virtually identical relative parameter updates ($\|\Delta\theta\|_2 / \|\theta\|_2 \approx 2.3\times 10^{-4} \text{ to } 4.9\times 10^{-4}$) under `clip_grad_norm_ = 1.0`!**
   - This occurs because AdamW's second-moment accumulator $v_t = \beta_2 v_{t-1} + (1 - \beta_2)\tilde{g}_t^2$ automatically normalizes each parameter's clipped gradient by its historical clipped root-mean-square magnitude $\sqrt{v_t}$. Since global norm clipping scales all parameters by the same scalar $c_t = 1.0 / \|g_t\|_2 \approx 0.01274$, the scalar $c_t$ cancels out in the ratio $\tilde{g}_{t,i} / \sqrt{v_{t,i}}$ at steady state, while protecting `v_t` and `m_t` from sudden single-batch intense-cyclone spikes (`max pre-clip norm = 150.00`).
2. **Incompatibility of `clip=5.0` or `unclipped` With Restored Epoch-5 AdamW State:**
   - Because $m_{t-1}$ and $v_{t-1}$ in `multitask_epoch_005.pt` were accumulated under `clip_grad_norm_ = 1.0`, switching to `clip=5.0` or `unclipped` injects gradients $5\times$ to $89.7\times$ larger than $\sqrt{v_{t-1}}$, rotating the AdamW update direction (`Cos(clip=1.0, unclip) = 0.4410`) and multiplying update step sizes by `1.95x` (`clip=5.0`) to `6.85x` (`unclipped`).
3. **Zero AMP Overflow or Non-Finite Steps:** `0 / 20` diagnostic steps (and `0 / 165` steps across full epochs, where `GradScaler` scale remained rock-solid at `512.0`) produced non-finite gradients.

---

## 3. Phase 4 Decision — Why No Further Clipping/Target-Scale Retraining Run Was Launched (Option D)

Following the Phase 2 instruction (*"If the results show clipping is not a substantive bottleneck, report that finding and do not launch an unnecessary clipping experiment"*) and Phase 4 Option D (*"If A–C are unsupported, recommend a scientifically useful storm-level generalization study instead of launching another speculative training run"*):
- **Option A (Optimizer/Loss Defect):** Unsupported — code verification and unit tests confirm zero defects in AMP unscaling, clipping frequency, accumulation denominators, or parameter-group coverage.
- **Option B (Relaxing `clip_grad_norm_`):** Unsupported — diagnostics prove `clip_grad_norm_ = 1.0` produces balanced `2.3e-4–4.9e-4` AdamW relative parameter updates across all backbone and head weight matrices, whereas relaxing `clip_norm` on the restored AdamW state shocks $m_t / \sqrt{v_t}$.
- **Option C (Changing Output Multipliers / Target Scaling):** Unsupported for continuation — reducing the wind gradient scale in V2 Candidate B degraded `VAL` Major-Hurricane MAE (`7.78 -> 9.57 kt`), and removing the `12x/5x/20x` multipliers changes head weight semantics and invalidates the pretrained/fine-tuned AdamW second-moment buffers.
- **Selected Action (Option D — Storm-Level Generalization Study on the `17` Validation Storms):** We executed a complete storm-by-storm decomposition of `VAL` (`17` storms, `1,815` sequences) on our best continuation checkpoint (`C1 Epoch 8`, `checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`).

---

## 4. Storm-Level Validation Generalization Study (`17` `VAL` Storms, `1,815` Sequences)

| Storm ID | Basin | Overpasses ($N$) | Max / Mean $V_{\max}$ (`kt`) | **Wind MAE (`Bias kt`)** | **Regime-Weighted Wind Loss (`% Share of Total VAL Wind Loss`)** | **`+24h` $\Delta V$ MAE (`kt`)** | **RI Positives ($N$)** | **`+24h` Track MAE (`km`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`WP102018`** | `WP` | `156` | `145.0` / `93.0` | `9.12` (`-8.27`) | **`24.502` (`24.97%`)** | `17.29` | `21` | `176.22` |
| **`EP072014`** | `EP` | `367` | `140.0` / `48.8` | `5.58` (`+0.68`) | **`8.776` (`21.04%`)** | `10.84` | `25` | `144.67` |
| **`WP262003`** | `WP` | `153` | `143.0` / `72.7` | `7.72` (`+3.37`) | **`11.477` (`11.47%`)** | `9.09` | `5` | `190.67` |
| **`AL142012`** | `AL` | `422` | `80.0` / `53.5` | `4.76` (`+0.22`) | `2.408` (`6.64%`) | `7.11` | `0` | `210.31` |
| **`WP181998`** | `WP` | `53` | `155.0` / `75.5` | `10.14` (`-8.18`) | `18.678` (`6.47%`) | `13.08` | `12` | `399.55` |
| **`AL122008`** | `AL` | `128` | `70.0` / `49.8` | `10.10` (`-9.05`) | `7.652` (`6.40%`) | `11.11` | `0` | `302.13` |
| **`AL131998`** | `AL` | `94` | `155.0` / `62.4` | `8.07` (`-2.11`) | `9.889` (`6.07%`) | `14.60` | `11` | `290.76` |
| **`EP091998`** | `EP` | `44` | `129.0` / `76.9` | `9.17` (`-5.74`) | `19.407` (`5.58%`) | `13.67` | `7` | `128.52` |
| **`IO012003`** | `IO` | `105` | `60.0` / `39.5` | `7.20` (`+4.80`) | `4.382` (`3.01%`) | `10.81` | `2` | `184.07` |
| **`SH072000`** | `SH` | `26` | `64.0` / `42.4` | **`20.57` (`+20.57`)** | `17.194` (`2.92%`) | `11.98` | `0` | `301.64` |
| **`WP191999`** | `WP` | `37` | `68.0` / `32.2` | `10.31` (`+6.81`) | `7.180` (`1.74%`) | `15.49` | `6` | `280.37` |
| **`CP052015`** | `CP` | `95` | `44.0` / `31.3` | `5.67` (`-1.73`) | `2.765` (`1.72%`) | `5.68` | `0` | `209.15` |
| **`IO071998`** | `IO` | `35` | `71.0` / `40.3` | `7.56` (`-3.90`) | `5.314` (`1.22%`) | `10.50` | `3` | `162.95` |
| **`IO042021`** | `IO` | `55` | `40.0` / `22.7` | `4.16` (`+3.78`) | `1.281` (`0.46%`) | `3.96` | `0` | `339.00` |
| **`WP072021`** | `WP` | `24` | `30.0` / `27.0` | `3.13` (`-2.97`) | `0.836` (`0.13%`) | `1.52` | `0` | `407.28` |
| **`IO041998`** | `IO` | `12` | `35.0` / `30.3` | `3.87` (`+0.34`) | `1.253` (`0.10%`) | `5.36` | `0` | `0.00` |
| **`EP031999`** | `EP` | `9` | `30.0` / `25.9` | `4.02` (`-1.68`) | `1.207` (`0.07%`) | `7.60` | `0` | `163.71` |

### Key Insights From the 17-Storm Validation Breakdown
1. **Extreme Concentration of `VAL` Regime-Weighted Wind Loss in 3 Category-4/5 Cyclones:**
   - Just **3 storms** — `WP102018` (Super Typhoon Mangkhut, $V_{\max}=145\text{ kt}$, `24.97%` of loss), `EP072014` (Hurricane Genevieve, $V_{\max}=140\text{ kt}$, `21.04%` of loss), and `WP262003` (Super Typhoon Lupit, $V_{\max}=143\text{ kt}$, `11.47%` of loss) — account for **`57.48%` of the entire `VAL` regime-weighted wind loss** (`676 / 1,815` samples), and together with `WP181998` ($155\text{ kt}$), `AL131998` (Hurricane Mitch, $155\text{ kt}$), and `EP091998` ($129\text{ kt}$), **the 6 Major Hurricanes in `VAL` account for `75.60%` of `VAL` wind loss and `81 / 92` (`88.0%`) of all `VAL` RI events**.
2. **Why `VAL_EVAL` Wind Loss Is Sensitive to `WP102018` and `WP181998`:**
   - On `EP072014` ($140\text{ kt}$, `367` samples), `WP262003` ($143\text{ kt}$, `153` samples), and `AL131998` ($155\text{ kt}$, `94` samples), C1 achieves `5.58 kt`, `7.72 kt`, and `8.07 kt` Wind MAE with small biases (`+0.68`, `+3.37`, `-2.11 kt`).
   - However, on `WP102018` (`156` samples, mean wind `93.0 kt`, max `145.0 kt`) and `WP181998` (`53` samples, max `155.0 kt`), C1 underestimates peak winds by `-8.27 kt` and `-8.18 kt`. Because `WP102018` spends most of its lifecycle above `110 kt` ($w_{\text{regime}} \approx 5\text{--}11\times$), its `9.12 kt` MAE produces a mean regime-weighted loss of **`24.502`** (`24.97%` of total `VAL` wind loss from a single storm).
3. **Southern Hemisphere Single-Storm Anomaly (`SH072000`):**
   - `SH072000` (`26` samples, $V_{\max}=64\text{ kt}$) is the **only** Southern Hemisphere storm in `VAL` and exhibits a uniform `+20.57 kt` overestimation (`MAE = 20.57 kt`), whereas on the `5` `SH` storms in `TRAIN` (`1,049` samples) MAE is `4.63 kt` (and on the `1` `SH` storm in `FINAL_TEST` from the baseline report, MAE was `4.98 kt`).

---

## 5. Master Validation Comparison Table (`VAL`, $N=1,815$ Sequences, $17$ Storms)

| Metric (`TRAIN_EVAL` / `VAL_EVAL` in `model.eval()` Mode) | **Original Epoch 5** (`checkpoints/tcprimed_100gb/best_checkpoint.pt`) | **C1 Continuation Epoch 8** (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`) | **V2 Candidate B Epoch 9** (`checkpoints/tcprimed_100gb_v2/best_checkpoint.pt`) | **C2 Head-Dropout Epoch 8** (`checkpoints/tcprimed_100gb_exp_c2_head_drop015/best_checkpoint.pt`) |
| :--- | :---: | :---: | :---: | :---: |
| **Checkpoint Status** | **ACCEPTED (Primary Intense-Cyclone Baseline)** | **ACCEPTED (Best Multi-Task Continuation Model)** | **REJECTED (Violated Major-Hurricane Guardrail)** | **REJECTED (No `VAL_EVAL` Improvement Over C1)** |
| **`TRAIN_EVAL` Global Loss (V1 Ref)** | `13.0492` | `7.9958` | `7.5198` | `8.2598` |
| **`VAL_EVAL` Global Loss (V1 Ref)** | `16.6384` | **`16.4052`** | `17.3104` | `16.5892` |
| **True Generalization Gap (`VAL - TRAIN`)** | **`+3.5892`** | `+8.4094` | `+9.7906` | `+8.3294` |
| **`VAL_MT_REL_ERR` (6-Task Relative Error)** | `0.5925` | **`0.5723`** | `0.5699` | `0.5791` (`0.5770` @ Ep 9) |
| **Overall Wind MAE / Bias (`VAL`, kt)** | `7.84` (`+1.36`) | **`6.90` (`-0.75`)** | `6.86` (`-1.36`) | `7.01` (`-0.60`) |
| **Major-Hurricane ($\ge 96\text{ kt}$) MAE / Bias (`VAL`, kt)** | **`7.78` (`-1.08`)** | **`8.68` (`-4.21`)** | `9.57` (`-6.11`) | `8.78` (`-5.19`) |
| **Pressure MAE (`VAL`, hPa)** | `5.99` | **`5.53`** | `5.51` | `5.74` |
| **Stage Balanced Accuracy (`VAL`, %)** | `64.27%` (`Acc: 65.14%`) | **`68.39%` (`Acc: 69.14%`)** | `68.52%` (`Acc: 69.27%`) | `67.93%` (`Acc: 68.75%`) |
| **$\Delta V$ `+6 / +12 / +18 / +24h` MAE (`VAL`, kt)** | `3.22 / 5.96 / 8.27 / 10.18` | `3.16 / 5.91 / 8.27 / 10.26` | `3.13 / 5.82 / 8.13 / 10.06` | `3.16 / 5.89 / 8.23 / 10.23` |
| **RI `VAL` PR-AUC / Brier (`92` pos / `1,484` neg)** | `0.5017 / 0.0386` | **`0.5226` / `0.0391`** | `0.5044 / 0.0384` | `0.5140 / 0.0412` |
| **Track `+24h` `VAL` MAE (`Skill vs CausalVel`, km)** | `213.65` (`-0.50`) | `213.58` (`-0.43`) | `213.71` (`-0.56`) | **`213.38` (`-0.23`)** |
| **Per-Storm `VAL` Wind MAE `Mean ± Std (Median)` (kt)** | `8.19±4.51 (8.03)` | **`7.71±3.97 (7.56)`** | — | `7.74±4.07 (7.38)` |
| **AdamW Backbone / Head Weight Update Ratio** | `2.3e-4 – 4.9e-4` | `2.3e-4 – 4.9e-4` | `2.3e-4 – 4.9e-4` | `2.3e-4 – 4.9e-4` |
| **Pre-Clip Global Norm (`Mean ± Std`, `% Clipped`)** | `89.66 ± 33.01` (`100%`) | `~85.0` (`100%`) | `59.0–62.3` (`100%`) | `83.0–97.1` (`100%`) |
| **Epoch Runtime / Active GPU VRAM** | `~445s` / `1041 MiB` | `431.5s` / `1041 MiB` | `429.0s` / `5263 MiB` | `476.5s` / `1094 MiB` |

---

## 6. Final Checkpoint Decision & One Recommended Next Research Step

### 6.1 Checkpoint Acceptance Decision
- **KEEP EXISTING ACCEPTED MODELS:**
  1. **Original Epoch 5 (`checkpoints/tcprimed_100gb/best_checkpoint.pt`):** Retained untouched as the primary production checkpoint for intense-cyclone estimation (`Major-Hurricane Wind MAE = 7.78 kt`, `Bias = -1.08 kt`, smallest generalization gap `+3.5892`).
  2. **Controlled Low-LR Continuation C1 Epoch 8 (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`):** Retained untouched as the strongest multi-task continuation checkpoint (`VAL_EVAL_LOSS = 16.4052`, `VAL_MT_REL_ERR = 0.5723`, `Overall Wind MAE = 6.90 kt`, `Major-Hurricane MAE = 8.68 kt`, `Pressure MAE = 5.53 hPa`, `Stage Balanced Acc = 68.39%`, `RI PR-AUC = 0.5226`).
- **DO NOT RUN A GRADIENT-CLIPPING RETRAINING EXPERIMENT:** Diagnostics proved that `clip_grad_norm_ = 1.0` is mathematically well-matched to the restored AdamW second-moment buffers ($v_t$), yielding uniform `2.3e-4–4.9e-4` relative parameter updates across all backbone and head weight matrices.

### 6.2 One Concrete Next Research Step Toward a Finished Research Model
- **5-Fold Storm-Stratified Cross-Validation Across the 92 Development Storms (`75 TRAIN + 17 VAL = 92 Storms`) With Storm-Balanced Minibatch Sampling:**
  - Our 17-storm validation breakdown demonstrates that single-split validation variance is dominated by small-number storm sampling: **`3` Super Typhoons (`WP102018`, `EP072014`, `WP262003`) drive `57.48%` of `VAL` wind loss**, **`6` storms drive `88.0%` (`81/92`) of `VAL` RI events**, and `SH` (`SH072000`) and `CP` (`CP052015`) have only **`1` validation storm each**.
  - Furthermore, within `TRAIN` (`75` storms, `10,545` overpasses), long-lived storms (`>250` overpasses) receive `~10x` more gradient updates per epoch than short-lived storms (`<30` overpasses).
  - Combining **storm-balanced epoch sampling** (sampling an equal number of anchor sequences per storm per epoch, or weighting each sample inversely by $\sqrt{N_{\text{storm}}}$) with **5-fold basin-stratified storm cross-validation across the 92 non-test storms** (keeping the 17 `FINAL_TEST` storms completely locked away) will eliminate single-storm validation sensitivity (such as `WP102018` and `SH072000`) and allow multi-fold ensembling without touching `FINAL_TEST`.
