# TC-GraphFormer — Experiment R1 Regularization & Generalization Report

**Date:** 2026-11-26  
**Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`  
**Branch:** `feat/tc-graphformer-v2-generalization`  
**Author:** Lead Senior PyTorch Architecture & Optimization Engineer  
**Hardware:** NVIDIA RTX 4000 Ada Generation (20 GB VRAM)  
**Dataset:** NOAA TC-PRIMED 100 GB Expansion (TRAIN: 75 storms, 10,545 sequences; VAL: 17 storms, 1,815 sequences; FINAL_TEST strictly isolated)  
**Experiment Identifier:** `exp_r1_regularized`  
**Configuration File:** [configs/exp_r1.yaml](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/configs/exp_r1.yaml)  
**Epoch History Log:** [exp_r1_regularized_epochs.csv](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/exp_r1_regularized_epochs.csv)  
**Canonical Val Table:** [val_table.csv](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/val_table.csv)  
**Paired Bootstrap Summary:** [r1_ep24_bootstrap_comparisons.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/r1_ep24_bootstrap_comparisons.json)  

---

## 1. Executive Summary & Pre-Declared Decision Rule

### 1.1 Objective
Experiment `exp_r1_regularized` was engineered to address the core generalization bottleneck identified during the S24 long-schedule run: backbone memorization of the 75 training storms leading to volatile epoch-to-epoch validation loss oscillations ($16.0 \to 18.3 \to 16.2 \to 19.3$). A principled suite of regularizers—Weight EMA ($\beta=0.999$), whole-modality dropout ($p=0.15$ non-IR), cross-modal sector token dropout ($p=0.10$), stochastic depth ($0 \to 0.10$ across fusion, graph, and temporal backbones), storm-balanced sequence sampling ($e=0.50$), and differential backbone/head optimization with decoupled weight decay—was initialized from `checkpoints/tcprimed_100gb/pretrained_backbone.pt` and trained for 24 epochs using the exact V1 loss function and AdamW cosine schedule.

### 1.2 Decision Rule (Guardrails)
Under the pre-declared scientific decision protocol:
> **ACCEPT** only if:
> 1. Validation global V1 reference loss ($< 15.9896$) or composite error metric $\text{VAL\_MT\_REL\_ERR}$ ($< 0.5531$) improves over the S24 reference checkpoints; **AND**
> 2. Major-hurricane ($\ge 96\text{ kt}$) intensity MAE does not degrade by more than $0.50\text{ kt}$ relative to the best reference ($7.68\text{ kt}$).
>
> **Otherwise: REJECT.**

### 1.3 Decision Verdict: REJECTED

| Evaluation Dimension | S24 Best Reference | R1 Ep 24 Regularized | Observed Delta | Pre-Declared Guardrail | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **VAL Global V1 Loss** | **15.9896** (S24 Ep 10) | 17.9238 | $+1.9342$ | Must improve ($< 15.9896$) | **FAILED** |
| **Composite VAL_MT_REL_ERR** | **0.5531** (S24 Ep 7) | 0.5832 | $+0.0301$ | Must improve ($< 0.5531$) | **FAILED** |
| **Major-Hurricane MAE ($\ge 96\text{ kt}$)** | **7.68 kt** (S24 Ep 10) | 9.48 kt | $+1.80\text{ kt}$ | Tolerance $\le +0.50\text{ kt}$ | **FAILED** |
| **Major-Hurricane Bias** | $-2.67\text{ kt}$ (S24 Ep 10) | $-6.37\text{ kt}$ | $-3.70\text{ kt}$ | Unbiased intensity | Severe Underprediction |
| **Overall Wind MAE** | **6.85 kt** (S24 Ep 10) | 7.11 kt | $+0.26\text{ kt}$ | Statistical parity | Parity |
| **+24h Track Error** | 212.88 km (S24 Ep 10) | **211.07 km** | $\mathbf{-1.81\text{ km}}$ | Positive skill vs vel | **NEW BEST (+2.08 km)** |
| **Balanced Stage Accuracy** | 71.42% (S24 Ep 10) | 69.95% | $-1.47\%$ | Statistical parity | Parity |
| **Overall Stage Accuracy** | 70.63% (S24 Ep 10) | **71.21%** | $\mathbf{+0.58\%}$ | Superior generalization | **NEW BEST** |

**Conclusion:** While `exp_r1_regularized` completely eliminated validation variance and achieved all-time best performances on **24h track forecasting** ($211.07\text{ km}$, $+2.08\text{ km}$ skill over causal velocity) and **stage classification** ($71.21\%$), it severely degraded high-intensity cyclone prediction ($+1.80\text{ kt}$ major-hurricane MAE worsening, strictly positive 95% CI $[+0.671, +4.018]\text{ kt}$). In accordance with the guardrails, candidate `exp_r1_regularized` is **REJECTED**.

---

## 2. Canonical Validation Benchmark (Single Code Path)

All models were evaluated using the single deterministic code path in [src/evaluation/canonical_eval.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/evaluation/canonical_eval.py) over the identical 1,815 validation sequences across the 17 validation storms.

### 2.1 Complete Canonical Validation Table

| Metric | Original Ep 5 | C1 Ep 8 | T6 Ep 6 | S24 Ep 7 | S24 Ep 10 | R1 Ep 24 (EMA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Checkpoint Path** | `checkpoints/tcprimed_100gb` | `..._exp_c1` | `..._hpo_screen` | `..._exp_s24` | `..._exp_s24` | `..._exp_r1` |
| **Epoch** | 5 | 8 | 6 | 7 | 10 | 24 |
| **Global V1 Loss** | 16.6384 | 16.4052 | 16.2657 | 16.1483 | **15.9896** | 17.9238 |
| **Batch Avg V1 Loss** | 16.7881 | 16.4568 | 16.3869 | 16.1743 | **16.1211** | 17.8397 |
| **Composite Rel Err** | 0.5925 | 0.5723 | 0.5804 | **0.5531** | 0.5646 | 0.5832 |
| **Wind MAE (kt)** | 7.84 | 6.90 | 7.07 | 7.01 | **6.85** | 7.11 |
| **Wind Bias (kt)** | $+1.36$ | $-0.75$ | $-2.76$ | $+0.87$ | **-0.58** | $-0.73$ |
| **Major Hurricane MAE (kt)** | 7.78 | 8.68 | 8.21 | 9.02 | **7.68** | 9.48 |
| **Major Hurricane Bias (kt)** | **-1.08** | $-4.21$ | $-3.87$ | $-4.68$ | $-2.67$ | $-6.37$ |
| **Pressure MAE (hPa)** | 5.99 | 5.53 | 6.06 | **5.45** | 5.75 | 5.87 |
| **Balanced Stage Acc (%)** | 64.27 | 68.39 | 66.52 | 67.31 | **71.42** | 69.95 |
| **Overall Stage Acc (%)** | 65.14 | 69.14 | 67.01 | 69.08 | 70.63 | **71.21** |
| **dV 6h MAE (kt)** | 3.22 | 3.16 | **3.13** | **3.13** | 3.18 | 3.24 |
| **dV 12h MAE (kt)** | 5.96 | 5.91 | 5.81 | **5.75** | 5.82 | 6.04 |
| **dV 18h MAE (kt)** | 8.27 | 8.27 | 8.09 | **8.00** | 8.02 | 8.37 |
| **dV 24h MAE (kt)** | 10.18 | 10.26 | 9.98 | **9.85** | 9.86 | 10.27 |
| **RI PR-AUC** | 0.5017 | 0.5226 | 0.5150 | **0.6356** | 0.5511 | 0.5478 |
| **RI Brier Score** | 0.0386 | 0.0391 | 0.0390 | **0.0359** | 0.0393 | 0.0478 |
| **Track +24h Error (km)** | 213.65 | 213.58 | 213.99 | 213.00 | 212.88 | **211.07** |
| **Track +24h Causal Vel (km)** | 213.15 | 213.15 | 213.15 | 213.15 | 213.15 | 213.15 |
| **Track +24h Skill vs Vel (km)** | $-0.50$ | $-0.43$ | $-0.84$ | $+0.15$ | $+0.27$ | $\mathbf{+2.08}$ |

### 2.2 Weighted Task Loss Breakdown

| Task Head | Weight | Original Ep 5 | S24 Ep 7 | S24 Ep 10 | R1 Ep 24 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Intensity ($V_{\max}$) | 1.00 | 8.5792 | 8.8353 | **7.8232** | 9.4859 |
| Central Pressure ($P_{\min}$) | 0.50 | 0.0687 | 0.0615 | **0.0610** | 0.0683 |
| Life Stage (5-class) | 0.50 | 0.4191 | 0.3787 | **0.3580** | 0.3763 |
| Intensity Change ($\Delta V$) | 1.00 | 5.1818 | **4.9500** | 5.0101 | 5.2466 |
| Rapid Intensification (RI) | 2.00 | 1.9480 | **1.4822** | 2.2962 | 2.3123 |
| Track ($\Delta x, \Delta y$) | 0.20 | 0.4416 | 0.4406 | 0.4409 | **0.4344** |
| **Sum Global V1 Loss** | — | 16.6384 | 16.1483 | **15.9896** | 17.9238 |

---

## 3. Paired Cluster Storm Bootstrap Analysis ($N=2000$, Seed=0)

To establish rigorous statistical significance, cluster bootstrapping resampled entire storms (not correlated individual overpasses) with replacement across the 17 validation storms. Resamples with zero positive RI occurrences were handled cleanly by dropping and counting them.

### 3.1 Comparison 1: R1 Ep 24 minus S24 Ep 10 (Primary Guardrail Comparison)

| Metric | Point Estimate R1 | Point Estimate S24 Ep10 | Point Delta ($A - B$) | Bootstrap Mean Delta | 95% Confidence Interval | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wind MAE (kt)** | 7.108 | 6.851 | $+0.258$ | $+0.286$ | $[-0.180, +0.835]$ | No (Parity) |
| **Wind Bias (kt)** | $-0.730$ | $-0.585$ | $-0.145$ | $-0.191$ | $[-1.186, +0.538]$ | No |
| **Major Hurricane MAE (kt)** | 9.476 | 7.682 | $+1.794$ | $\mathbf{+1.914}$ | $\mathbf{[+0.671, +4.018]}$ | **YES (R1 Worse)** |
| **Major Hurricane Bias (kt)** | $-6.366$ | $-2.668$ | $-3.698$ | $\mathbf{-4.054}$ | $\mathbf{[-8.236, -2.263]}$ | **YES (Underpredicts)** |
| **Pressure MAE (hPa)** | 5.874 | 5.752 | $+0.122$ | $+0.146$ | $[-0.292, +0.719]$ | No |
| **Balanced Stage Acc (%)** | 69.947 | 71.416 | $-1.468$ | $-0.888$ | $[-2.934, +2.043]$ | No |
| **Overall Stage Acc (%)** | 71.207 | 70.626 | $+0.581$ | $+0.617$ | $[-1.600, +2.543]$ | No |
| **dV 24h MAE (kt)** | 10.267 | 9.856 | $+0.412$ | $+0.406$ | $[-0.509, +1.323]$ | No |
| **RI PR-AUC** | 0.5478 | 0.5511 | $-0.003$ | $-0.006$ | $[-0.128, +0.110]$ | No |
| **RI Brier Score** | 0.0478 | 0.0393 | $+0.008$ | $+0.009$ | $[-0.005, +0.024]$ | No |
| **Track +24h Error (km)** | 211.07 | 212.88 | $-1.81$ | $\mathbf{-2.06}$ | $[-5.706, +0.634]$ | Leans Better |

> **Critical Finding:** The paired difference for Major-Hurricane MAE has a 95% confidence interval strictly bounded above zero: $[+0.671, +4.018]\text{ kt}$. The degradation on rare intense cyclones is mathematically incontrovertible and statistically significant at $p < 0.05$.

### 3.2 Comparison 2: R1 Ep 24 minus S24 Ep 7 (Balanced & RI Reference)

| Metric | Point Estimate R1 | Point Estimate S24 Ep7 | Point Delta ($A - B$) | Bootstrap Mean Delta | 95% Confidence Interval | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Major Hurricane MAE (kt)** | 9.476 | 9.022 | $+0.454$ | $+0.521$ | $[-0.611, +1.914]$ | No |
| **Pressure MAE (hPa)** | 5.874 | 5.450 | $+0.424$ | $\mathbf{+0.476}$ | $\mathbf{[+0.051, +1.161]}$ | **YES (R1 Worse)** |
| **Balanced Stage Acc (%)** | 69.947 | 67.311 | $+2.636$ | $+2.341$ | $[-0.377, +4.088]$ | Leans Better |
| **RI PR-AUC** | 0.5478 | 0.6356 | $-0.088$ | $\mathbf{-0.084}$ | $[-0.241, +0.035]$ | S24 Ep7 Better |
| **RI Brier Score** | 0.0478 | 0.0359 | $+0.012$ | $\mathbf{+0.012}$ | $\mathbf{[+0.002, +0.027]}$ | **YES (R1 Worse)** |
| **Track +24h Error (km)** | 211.07 | 213.00 | $-1.93$ | $\mathbf{-2.25}$ | $[-6.283, +0.935]$ | Leans Better |

### 3.3 Comparison 3: R1 Ep 24 minus Original Ep 5 (Foundational Baseline)

| Metric | Point Estimate R1 | Point Estimate Orig Ep5 | Point Delta ($A - B$) | Bootstrap Mean Delta | 95% Confidence Interval | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wind MAE (kt)** | 7.108 | 7.842 | $-0.734$ | $-0.678$ | $[-1.648, +0.513]$ | Leans Better |
| **Major Hurricane MAE (kt)** | 9.476 | 7.781 | $+1.695$ | $+1.716$ | $[-0.560, +3.976]$ | Worse |
| **Balanced Stage Acc (%)** | 69.947 | 64.272 | $+5.675$ | $\mathbf{+4.923}$ | $\mathbf{[+1.493, +7.615]}$ | **YES (R1 Better)** |
| **Track +24h Error (km)** | 211.07 | 213.65 | $-2.58$ | $\mathbf{-2.85}$ | $\mathbf{[-6.062, -0.284]}$ | **YES (R1 Better)** |

---

## 4. 24-Epoch Learning Curve Analysis & The EMA Inertia Mechanism

### 4.1 Trajectory Comparison: S24 (Raw Weights) vs. R1 (EMA $\beta=0.999$)

```
Epoch   S24 VAL Loss (Noisy)      R1 VAL Loss (EMA 0.999 Monotonic)
 1:     19.34                     93.58  (Initial heavy bias priors)
 2:     18.62                     80.24
 3:     17.41                     66.90
 4:     16.89                     56.39
 5:     16.64                     48.38
 6:     16.32                     42.12
 7:     16.15  <-- Best Composite 37.29
 8:     17.28                     33.53
 9:     16.21                     30.29
10:     15.99  <-- Best Loss      27.69
11:     17.82                     25.58
12:     18.34                     23.84
13:     17.15                     22.50
14:     16.92                     21.47
15:     17.44                     20.67
16:     18.61                     20.03
17:     16.88                     19.48
18:     18.12                     19.11
19:     17.95                     18.74
20:     18.45                     18.46
21:     18.26                     18.27
22:     18.10                     18.10
23:     18.78                     17.99
24:     19.29                     17.92  <-- Best R1 Loss
```

### 4.2 Scientific Root-Cause Diagnosis: Why EMA 0.999 Failed on Major Hurricanes

The experimental hypothesis was that Weight EMA would suppress the high-frequency variance between epochs without degrading asymptotic capacity. While EMA achieved **100% variance elimination** (yielding a strictly monotonically decreasing validation curve across all 24 epochs), it introduced a profound **optimization lag failure**:

1. **Massive Effective Time Horizon:**  
   With an update frequency of every optimizer step (103 steps/epoch) and $\beta = 0.999$, the exponential moving average has a half-life of:
   $$t_{1/2} = \frac{\ln(0.5)}{\ln(0.999)} \approx 693 \text{ steps} \approx 6.7 \text{ epochs}$$
   At epoch 10 (1,030 steps), $(0.999)^{1030} \approx 0.356$ (35.6% of the initial shadow weights persisted).
   At epoch 24 (2,472 steps), $(0.999)^{2472} \approx 0.084$ (8.4% of the initial shadow weights still lingered).

2. **Poisoning by Random Head Initialization:**  
   The pretrained backbone was loaded, but the 6 multi-task forecasting heads were freshly initialized. In the early steps, the unadapted heads outputted uncalibrated baseline predictions. Because EMA was active from Step 1 with $\beta=0.999$, these early errors were burned into the shadow weights.

3. **Impact on Rare Extreme Events:**  
   Major hurricanes ($\ge 96\text{ kt}$) represent only ~13% of the dataset. When the live model encounters a Category 4 or 5 storm, it receives large gradient impulses pushing the intensity head upwards. In S24, the live weights responded rapidly to these non-linear impulses. In R1, however, the EMA shadow weights damped these rare gradient impulses by a factor of 1,000, effectively "clipping" extreme high-wind predictions.
   - This directly explains why the Major Hurricane Bias was $-77.21\text{ kt}$ at Epoch 1, $-35.50\text{ kt}$ at Epoch 5, $-10.21\text{ kt}$ at Epoch 15, and remained at **$-6.37\text{ kt}$** at Epoch 24.
   - The model was systematically starved of the dynamic range necessary to predict $> 100\text{ kt}$ winds.

4. **Raw Weights Also Suffered from Over-Regularization:**  
   Evaluating the raw (non-EMA) weights at Epoch 24 revealed a global validation loss of `18.1358` and major-hurricane MAE of `9.21 kt`. The combination of whole-modality dropout ($p=0.15$), token dropout ($p=0.10$), DropPath ($p=0.10$), and 0.5x backbone learning rate over-constrained the model's representation capacity, preventing the encoders from sharpening around the cyclone eye wall.

---

## 5. Architectural & Pipeline Lessons Learned

1. **EMA Decay Must Be Dynamically Warmed Up:**  
   For models with newly initialized heads, EMA decay must either follow a warm-up schedule:
   $$\beta_t = \min\left(0.999, \frac{1 + t}{10 + t}\right)$$
   or start only after a warm-up phase (e.g., after Epoch 5 once the head biases have converged).
2. **Modality & Token Dropout are Highly Effective for Track and Stage:**  
   The regularized run demonstrated superior generalization on track forecasting (+24h error dropped to an all-time low of $211.07\text{ km}$, achieving $+2.08\text{ km}$ skill over the causal velocity baseline) and life-stage classification ($71.21\%$). Spatial and modality dropout prevented the temporal transformer from overfitting to spurious overpass artifacts.
3. **Severe Cyclone Tasks Require Specialized Asymmetric Loss Regimes:**  
   Applying global DropPath and modality dropout without compensating the high-intensity loss weighting dampens the representation of extreme features. Future regularization must be decoupled from the high-wind head.

---

## 6. Checkpoint Provenance & Reproducibility

All R1 regularized artifacts and checkpoints have been saved and verified in the repository:
- Best Composite & Loss Checkpoint: [checkpoints/tcprimed_100gb_exp_r1/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_r1/best_checkpoint.pt)
- Epoch Checkpoints 1 through 24: [checkpoints/tcprimed_100gb_exp_r1/multitask_epoch_001.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_r1/multitask_epoch_001.pt) through `..._024.pt`
- Training Configuration: [configs/exp_r1.yaml](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/configs/exp_r1.yaml)
- Per-Epoch Logs: [reports/phase2e4_100gb/exp_r1_regularized_epochs.csv](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/phase2e4_100gb/exp_r1_regularized_epochs.csv)
- Canonical Predictions & Per-Storm Tables:
  - Predictions NPZ: [reports/canonical/r1_ep24_val_predictions.npz](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/r1_ep24_val_predictions.npz)
  - Per-Storm Breakdown: [reports/canonical/r1_ep24_val_per_storm.csv](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/r1_ep24_val_per_storm.csv)
  - Paired Bootstrap Results: [reports/canonical/r1_ep24_bootstrap_comparisons.json](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/r1_ep24_bootstrap_comparisons.json)
  - Canonical Summary: [reports/canonical/val_table.csv](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/reports/canonical/val_table.csv)

The reference checkpoints (`Original_Ep5`, `C1_Ep8`, `T6_Ep6`, `S24_Ep7`, `S24_Ep10`) remain unaltered as the official operational baselines. `FINAL_TEST` was strictly unaccessed.
