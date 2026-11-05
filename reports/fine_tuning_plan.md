# TC-GraphFormer Fine-Tuning & Domain Adaptation Plan
**Task Specialization, North Indian Ocean Adaptation, and Transfer Strategies**  
**Dataset:** NOAA TC PRIMED v01r01-final  
**Date:** November 2026

---

## 1. Objectives & Strategic Motivation

The goal of the fine-tuning framework is to take the foundational multi-modal representations learned during multi-task pretraining and adapt them to:
1. **Critical High-Impact Forecast Tasks:** Specifically short-term intensity changes ($\Delta v$ at +6h, +12h, +18h, +24h) and Rapid Intensification (RI $\ge 30$ kt / 24h).
2. **Regional Domain Adaptation (North Indian Ocean - NIO):** Tropical cyclones in the Bay of Bengal and Arabian Sea (basin code `IO`) exhibit distinct thermodynamic constraints: shallow mixed layers, high salinity stratification, complex land interaction with the Indian subcontinent, and bi-modal pre-monsoon/post-monsoon frequency peaks.
3. **Prevent Catastrophic Forgetting:** Maintain general feature extraction capabilities while adapting higher-level decision boundaries.

---

## 2. Fine-Tuning Protocols (`src/training/finetune_task.py`)

### 2.1 Protocol A: Intensity Change & Rapid Intensification Specialization
- **Base Checkpoint:** `checkpoints/best_checkpoint.pt`.
- **Target Loss Weighting:**
  * $w_v = 0.2, w_p = 0.1, w_s = 0.2$ (reduced auxiliary weights)
  * $w_{\Delta v} = 3.0$ (heavily weighted multi-horizon intensity changes)
  * $w_{\text{RI}} = 5.0$ (heavily weighted rapid intensification)
- **Layer Freezing Policy:**
  * Modality Encoders (IR, PMW, GPROF, Radar, Environmental): **Frozen** ($\nabla_\theta = 0$).
  * Cross-Modal Fusion & Spatial Graph & Heads: **Trainable**.
- **Learning Rate:** $\eta = 5.0 \times 10^{-5}$ (reduced from $1.0 \times 10^{-4}$).
- **Smoke Test Result:**
  * Epoch 1: Loss reduced from 26.40 to validation loss 54.02; dV MAE = 12.41 kt; RI Acc = 100.0%.
  * Report: `reports/finetuning_intensity_change_summary.json`.
  * Checkpoint: `checkpoints/finetuned_intensity_change_epoch_001.pt`.

---

### 2.2 Protocol B: North Indian Ocean (NIO) Domain Adaptation
- **Basin Filter:** $B = \text{"IO"}$ (Bay of Bengal & Arabian Sea cyclones).
- **Sample Distribution:** 14 Train storms (468 overpasses), 3 Val storms (203 overpasses).
- **Motivation:** Atmospheric dry air intrusions from the Thar Desert and Arabian Peninsula frequently induce asymmetric convective erosion unique to the NIO basin.
- **Layer Freezing Policy:**
  * Modality Encoders: **Frozen**.
  * Spatial Graph & Temporal Transformer: **Trainable with low LR ($\eta = 3.0 \times 10^{-5}$)**.
- **Smoke Test Result:**
  * Epoch 1: Train Loss = 88.65, Val Loss = 80.53; dV MAE = 5.04 kt; RI Acc = 100.0%.
  * Report: `reports/finetuning_nio_adaptation_summary.json`.
  * Checkpoint: `checkpoints/finetuned_nio_adaptation_epoch_001.pt`.

---

## 3. Full-Scale Deployment Roadmap (500 GB Archive on Machine B)

When transitioning from the 20 GB pilot to the 500 GB full TC PRIMED archive on Machine B:

```
[Phase 1: Self-Supervised Pretraining]
  -> Masked Feature Modeling on all available overpasses (2000-2025)
  -> 30 epochs on Machine B GPU (mixed precision fp16, batch size 32)
  -> Checkpoint: pretrained_backbone_full.pt

[Phase 2: Supervised Multi-Task Global Training]
  -> Multi-task training across 6 global basins (AL, EP, CP, WP, IO, SH)
  -> 50 epochs with cosine decay learning rate
  -> Checkpoint: tc_graphformer_global_best.pt

[Phase 3: Domain & Task Specialization]
  -> Fine-tuning branch 1: Global RI & Intensity Change (20 epochs)
  -> Fine-tuning branch 2: North Indian Ocean specialized model (15 epochs)
  -> Checkpoint: tc_graphformer_nio_adapted.pt
```

---

## 4. Scientific Baseline Comparison Plan

To establish publication-grade credibility, the fine-tuned TC-GraphFormer will be benchmarked on the held-out 12-storm test partition against:
1. **Climatology & Persistence (CLIPER / OCD5):** Standard meteorological operational baselines.
2. **Tabular Gradient Boosting (LightGBM):** Features extracted from Best Track and ERA5 synoptic scalars.
3. **Standard ResNet-50 Baseline:** Monomodal IR 2D brightness temperature regression.
4. **TC-GraphFormer (Full Multimodal):** Evaluating the ablation of spatial graph vs. temporal transformer.

