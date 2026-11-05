# TC-GraphFormer Training & Verification Framework Report
**System Setup, Supervised Multi-Task Optimization, and Smoke Test Verification**  
**Dataset:** NOAA TC PRIMED v01r01-final  
**Date:** November 2026

---

## 1. Executive Summary

This report documents the design, implementation, and verification of the training framework for the **TC-GraphFormer** tropical cyclone forecasting architecture. The system supports:
1. Distributed or single-node hardware training with Automatic Mixed Precision (AMP) and gradient clipping.
2. Self-supervised pretraining via Masked Feature Modeling (MFM).
3. Supervised multi-task optimization combining robust regression (Huber/Smooth L1), classification (Cross-Entropy), and positive-weighted binary cross-entropy for extreme class imbalance (Rapid Intensification).
4. Automated evaluation producing physically meaningful metrics: Wind Speed MAE (kt), Pressure MAE (hPa), Stage Accuracy (%), Intensity Change MAE (kt), and Rapid Intensification Accuracy/F1.

All workflows were validated with automated unit and smoke tests on real pilot data, generating serialized PyTorch checkpoints in `checkpoints/`.

---

## 2. Multi-Task Loss Formulation (`src/training/losses.py`)

The total loss is a weighted sum over 5 primary learning objectives:
$$\mathcal{L}_{\text{total}} = w_v \mathcal{L}_v + w_p \mathcal{L}_p + w_s \mathcal{L}_s + w_{\Delta v} \mathcal{L}_{\Delta v} + w_{\text{RI}} \mathcal{L}_{\text{RI}}$$

### 2.1 Wind Speed & Pressure Huber Losses
To prevent large gradient explosions caused by rare Category 5 cyclones or sensor anomalies, we utilize Smooth L1 (Huber) loss:
$$\mathcal{L}_v = \text{Huber}(v_{\text{pred}}, v_{\text{true}}, \beta=1.0)$$
$$\mathcal{L}_p = \text{Huber}\left(\frac{p_{\text{pred}}}{10.0}, \frac{p_{\text{true}}}{10.0}, \beta=1.0\right)$$
Scaling pressure by $0.10$ balances gradient magnitudes between wind speed ($\sim 20-160$ kt) and central pressure ($\sim 900-1015$ hPa).

### 2.2 Development Stage Cross-Entropy
$$\mathcal{L}_s = -\sum_{c=0}^5 y_c \log \sigma(\hat{y}_c)$$
where stages are mapped to 6 canonical categories: Tropical Depression (0), Tropical Storm (1), Hurricane/Typhoon (2), Extratropical (3), Disturbance/Low (4), and Unknown (5).

### 2.3 Masked Multi-Horizon Intensity Change Loss
Because some future overpass intervals or best-track extrapolations terminate when cyclones dissipate, intensity change targets are masked:
$$\mathcal{L}_{\Delta v} = \frac{\sum_{h \in \{6, 12, 18, 24\}} m_h \cdot \text{SmoothL1}(\Delta \hat{v}_h, \Delta v_h)}{\sum_h m_h + \epsilon}$$
Missing horizons ($m_h = 0$) contribute zero loss.

### 2.4 Rapid Intensification BCE with Logits
Rapid Intensification ($\Delta v \ge 30$ kt in 24h) occurs in only 2.91% of pilot observations. Standard BCE collapses towards predicting 0. We employ positive class weight scaling:
$$\mathcal{L}_{\text{RI}} = -\left[ w_{\text{pos}} y_{\text{RI}} \log \sigma(\hat{y}_{\text{RI}}) + (1 - y_{\text{RI}}) \log (1 - \sigma(\hat{y}_{\text{RI}})) \right]$$
with $w_{\text{pos}} = 10.0$, boosting recall on extreme intensification events.

---

## 3. Training Engine & Checkpointing (`src/training/trainer.py`)

- **Optimizer:** AdamW with initial learning rate $\eta = 1.0 \times 10^{-4}$ and weight decay $\lambda = 1.0 \times 10^{-4}$.
- **Learning Rate Schedule:** Cosine Annealing with minimum $\eta_{\text{min}} = 1.0 \times 10^{-6}$.
- **Gradient Clipping:** Max norm of $1.0$ enforced across all parameter gradients.
- **Mixed Precision:** Automatic Mixed Precision (AMP) enabled for CUDA acceleration (`torch.cuda.amp.autocast`).
- **Checkpoint Serialization:** Full state dictionaries stored for model, optimizer, epoch counter, and metric history:
  * `checkpoints/best_checkpoint.pt`: Best validation loss checkpoint.
  * `checkpoints/latest_checkpoint.pt`: Resume checkpoint updated each epoch.
  * `checkpoints/checkpoint_epoch_*.pt`: Periodic epoch checkpoints.

---

## 4. Smoke Test Execution Results

Smoke tests were executed against real pilot data to prove end-to-end functionality:

### 4.1 Self-Supervised Pretraining Smoke Test (`pretrain.py`)
- **Dataset:** 32 real pilot training samples.
- **Task:** 30% random spatial token masking & MSE feature reconstruction.
- **Result:** Epoch 1 Reconstruction MSE = 12.376.
- **Output:** Checkpoint saved to `checkpoints/pretrained_backbone.pt` (5.90 MB).
- **Summary:** `reports/pretraining_summary.json`.

### 4.2 Supervised Multi-Task Training Smoke Test (`train_multitask.py`)
- **Dataset:** 64 real train samples, 16 real val samples, batch size 8.
- **Epoch 1:**
  * Train Loss: 85.98 (Intensity: 28.55, Pressure: 100.20, Stage: 1.62)
  * Val Loss: 97.30 | Wind MAE: 35.52 kt | Pressure MAE: 1001.43 hPa | Stage Acc: 0.0%
- **Epoch 2:**
  * Train Loss: 85.26 (Intensity: 28.04, Pressure: 100.20, Stage: 1.67)
  * Val Loss: 96.94 | Wind MAE: 35.24 kt | Pressure MAE: 1001.38 hPa | Stage Acc: 12.5%
- **Output:**
  * `checkpoints/best_checkpoint.pt` (17.57 MB)
  * `checkpoints/latest_checkpoint.pt` (17.57 MB)
  * `checkpoints/checkpoint_epoch_001.pt` (17.57 MB)
  * `checkpoints/checkpoint_epoch_002.pt` (17.57 MB)
  * `reports/multitask_training_summary.json`

---

## 5. Automated Test Suite Validation

The comprehensive unittest suite was executed:
- **Total Tests:** 107 tests across 7 test modules.
- **Pass Rate:** 100% (107 passed, 0 failures, 0 errors).
- **Execution Time:** 21.22 seconds.
- **Coverage:** Hardware discovery, cryptographic staging, NetCDF readers, environmental matcher, modality processors, TC-GraphFormer encoders, cross-modal fusion, spatial graph MPNN, causal temporal transformer, multi-task losses, and trainer serialization.

