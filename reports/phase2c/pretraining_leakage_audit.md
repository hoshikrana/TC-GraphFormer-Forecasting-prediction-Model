# TC PRIMED Phase 2C — Pretraining Leakage and Self-Supervised Representation Audit

## 1. Executive Summary & Objective

Phase 2B introduced a self-supervised Masked Feature Modeling (MFM) pretraining stage in `src/training/pretrain.py`. The objective was to allow modality encoders (IR, PMW, GPROF, Radar, ERA5) and the cross-modal fusion transformer to learn joint atmospheric representations across disparate sensor resolutions before fine-tuning on downstream intensity and track targets.

This audit evaluates:
1. **Split Isolation**: Whether any samples from the `VAL` or `TEST` partitions leaked into the pretraining phase.
2. **Label Leakage**: Whether Best Track supervisory targets (wind, pressure, track) were exposed to the self-supervised backbone.
3. **Algorithmic Correctness**: The mathematical mechanics of the token masking and reconstruction loss, specifically under extreme modality missingness.

---

## 2. Split Isolation Verification

A common pitfall in self-supervised learning is pretraining on the entire dataset (including validation and test sets) under the assumption that "unlabeled data cannot leak labels." In spatio-temporal domains like tropical cyclones, exposing test storms during pretraining allows the model to memorize structural storm geometries, sensor noise characteristics, and cyclone life cycles, causing optimistic evaluation bias.

### Code & Execution Audit (`src/training/pretrain.py` Line 63)
```python
dataset = TCPRIMEDDataset(
    sample_index_csv=train_index, 
    partition="TRAIN", 
    grid_size=64, 
    max_samples=max_samples
)
```

- **Verification**: The pretraining script explicitly instantiates `TCPRIMEDDataset` with `partition="TRAIN"`.
- **Partition Filtering**: The dataset loader filters `sample_index.csv` using `df[df["partition"] == "TRAIN"]`.
- **Sample Count**: Out of 2,599 total pilot overpasses, exactly 1,847 overpasses belonging to the 56 training storms were accessible to the pretraining DataLoader.
- **Val / Test Storms**: Zero overpasses from the 12 validation storms (362 overpasses) or 12 test storms (390 overpasses) were ingested.

**Finding**: **Zero partition leakage.** Split boundaries were strictly enforced at the storm level.

---

## 3. Label Leakage Audit

Did the self-supervised pretraining task utilize Best Track supervisory labels?

- In `src/training/pretrain.py`:
  - Encoders extract spatial representations $\mathbf{T}_{\text{modality}} \in \mathbb{R}^{B \times 16 \times d}$.
  - Target vectors $\mathbf{Y}_{\text{recon}}$ are the uncorrupted encoder representations $\mathbf{T}_{\text{modality}}$.
  - The model heads for intensity, pressure, stage, intensity change, RI, and track are not initialized or loaded during pretraining.
  - No target columns (`target_intensity`, `target_pressure`, etc.) are queried from `sample_index.csv` or batches.

**Finding**: **Zero label leakage.** The pretraining task is purely feature-reconstructive and completely independent of ground truth Best Track labels.

---

## 4. Algorithmic Critique: Masking Under Modality Absence

While partition and label isolation are clean, our audit identified a technical limitation in the MFM masking logic:

### Mechanism in `src/training/pretrain.py`:
```python
# Unpack modality presence masks (B, 5)
m_ir, m_pmw, m_gp, m_rad, m_env = masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3], masks[:, 4]

# Generate random patch mask (mask_ratio fraction masked)
rand_mask = (torch.rand(B, 80, 1, device=device) < mask_ratio).float()

# Corrupt input tokens by zeroing masked ones
t_ir = tgt_ir * (1.0 - rand_mask[:, :16])
...
# Compute MSE only on masked positions
loss = (F.mse_loss(recon, all_targets, reduction="none") * rand_mask).sum() / rand_mask.sum().clamp(min=1.0)
```

### Problem Analysis:
1. In TC PRIMED, precipitation radar is available in only **15.4%** of overpasses. When radar is absent (`m_rad = 0`), `tgt_rad` is a tensor of zeros.
2. The random masking tensor `rand_mask` selects 30% of the 80 fused tokens indiscriminately across all 5 modalities.
3. If tokens corresponding to absent radar or GPROF slots are selected by `rand_mask`, the model is penalized for reconstructing zeros rather than inferring missing physical properties from available IR or PMW channels.
4. This causes the pretraining optimization to waste capacity learning "zero-detection" rather than physical cross-sensor representation.

### Required Correction for Phase 3:
Masking must be gated by the observed modality presence mask:
$$\mathbf{M}_{\text{token\_valid}} = \text{repeat}(\mathbf{M}_{\text{modality}}, \text{repeats}=16)$$
$$\mathbf{M}_{\text{effective\_mask}} = \mathbf{M}_{\text{random}} \odot \mathbf{M}_{\text{token\_valid}}$$
$$\mathcal{L}_{\text{MFM}} = \frac{\| (\hat{\mathbf{T}} - \mathbf{T}) \odot \mathbf{M}_{\text{effective\_mask}} \|_2^2}{\sum \mathbf{M}_{\text{effective\_mask}} + \epsilon}$$

Under this formulation, the model is only tasked with reconstructing tokens that actually contained observed physical satellite measurements.

---

## 5. Audit Summary
- **Partition Isolation**: Verified Clean (Train-only).
- **Label Independence**: Verified Clean (No Best Track data accessed).
- **Execution Validation**: Checkpoint `checkpoints/pretrained_backbone.pt` correctly transfers encoder and fusion weights to multi-task models.
- **Identified Action Item**: Gate MFM loss by modality presence masks in Phase 3.

