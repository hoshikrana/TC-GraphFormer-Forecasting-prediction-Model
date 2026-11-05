# TC PRIMED Phase 2C — Unresolved Issues and Scientific Risk Register

## 1. Overview & Issue Taxonomy

This document catalogs all technical discrepancies, scientific limitations, hardware bottlenecks, and algorithmic risks uncovered during the Phase 2C Scientific Integrity Audit. Each issue is categorized by severity, accompanied by root cause analysis, evidence, and specific corrective actions required before scaling to the 500 GB (2000–2025) dataset.

```
+-------------------------------------------------------------------------------------------------------+
| SEVERITY LEVEL | COUNT | DEFINITION                                                                   |
+----------------+-------+------------------------------------------------------------------------------+
| CRITICAL       |   3   | Fatal blocker; prevents execution or invalidates scientific conclusions.     |
| HIGH           |   4   | Significant scientific or performance defect; must be resolved before Phase 3|
| MEDIUM         |   3   | Engineering inefficiency or metadata gap; straightforward correction.        |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Critical Blockers (Must Be Resolved Prior to Dataset Download)

### Issue C-1: Workstation Local Disk Space Exhaustion
- **Category**: Hardware / Infrastructure Blocker
- **Severity**: **CRITICAL**
- **Evidence**:
  - Drive `C:` Free Space: **38.86 GB** (90.2% utilized)
  - Drive `D:` Free Space: **31.33 GB** (59.9% utilized)
  - Total Workstation Capacity Available: **70.19 GB**
  - Planned Raw Dataset Target: **500.00 GB**
  - Total Projected Storage Footprint (Raw + Processed HDF5 + Shards + Run Checkpoints + 20 GB Headroom): **608.60 GB**
- **Deficit**: **538.41 GB shortage**
- **Impact**: Any attempt to trigger raw downloads on internal drives will cause fatal OS filesystem exhaustion, application crashes, and SQLite/HDF5 corruption.
- **Mandatory Corrective Action**: Procurement and mounting of a dedicated **1 TB or 2 TB high-speed external NVMe SSD** (USB 3.2 Gen 2 / USB4) before initiating Phase 3 downloads.

---

### Issue C-2: Disproven Remote GPU Execution Claims
- **Category**: Computational Execution Reality
- **Severity**: **CRITICAL**
- **Evidence**:
  - `checkpoints/best_checkpoint.pt`: Saved `device: "cpu"`
  - `checkpoints/finetuned_intensity_change_epoch_001.pt`: Saved `device: "cpu"`
  - `reports/phase2c/training_execution_audit.csv`: Audited all 4 runs; 100% ran locally on CPU.
  - Previous Phase 2B notes claimed "GPU training across two machines via AnyDesk".
- **Root Cause**: AnyDesk is a GUI desktop streaming application. It does not network CUDA devices or remote interpreters. Python was executed locally on Machine A's AMD64 CPU for 1–2 epochs as a software sanity check.
- **Impact**: The model has never been trained to convergence on GPU compute. Multi-task loss remains high (val loss 80.53).
- **Mandatory Corrective Action**: Establish a real headless compute pipeline (SSH + Git + Conda/Venv + PyTorch CUDA 12.x on Machine B, with automated data staging via `rsync`).

---

### Issue C-3: Complete Partition Imbalance in Test Split (Zero Hurricane Overpasses)
- **Category**: Experimental Design & Scientific Validity
- **Severity**: **CRITICAL**
- **Evidence** (`reports/phase2c/storm_partition_distribution.csv`):
  - In `TRAIN` (56 storms, 1,847 overpasses): 459 Hurricane (HU/TY) overpasses (24.9%).
  - In `VAL` (12 storms, 362 overpasses): 140 Hurricane overpasses (38.7%).
  - In `TEST` (12 storms, 390 overpasses): **0 Hurricane (HU/TY) overpasses (0.00%)** and **0 Extratropical (EX) overpasses (0.00%)**.
- **Root Cause**: Phase 1F generated random 70/15/15 storm-level splits stratified only by basin and historical period, without stratifying by **cyclone peak intensity category**. The 12 randomly selected test storms happened to be Tropical Depressions and Tropical Storms that never achieved hurricane intensity.
- **Impact**: The test set cannot evaluate the model's ability to forecast Category 1–5 hurricanes or detect Rapid Intensification at major intensity levels.
- **Mandatory Corrective Action**: Re-partition the dataset using multi-objective stratified sampling across Basin $\times$ Era $\times$ Peak Saffir-Simpson Category.

---

## 3. High-Priority Scientific Defects

### Issue H-1: Severe Unadjusted Model Initialization Bias Underperforming Baselines
- **Category**: Model Convergence / Baseline Benchmark
- **Severity**: **HIGH**
- **Evidence** (`reports/phase2c/baseline_comparison.csv`):
  - Wind Speed MAE: Baseline Persistence = **15.51 kt** vs. TC-GraphFormer = **39.14 kt**
  - Pressure MAE: Climatology = **5.14 hPa** vs. TC-GraphFormer = **995.89 hPa**
- **Root Cause**: Linear heads `head_intensity` and `head_pressure` were initialized with standard zero-mean normal weights without output bias offsets matching physical atmospheric scales ($V \approx 50\text{ kt}, P \approx 1000\text{ hPa}$). After only 2 CPU epochs, the network predicted near zero for pressure, resulting in ~1000 hPa error.
- **Corrective Action**:
  1. Initialize output layer bias of `head_pressure` to $1000.0$ and `head_intensity` to $50.0$.
  2. Normalize regression targets during training using z-score normalization: $y_{\text{norm}} = (y - \mu) / \sigma$.
  3. Train for 30–50 epochs on GPU until loss stabilizes.

---

### Issue H-2: Masked Feature Pretraining Penalizes Absent Sensor Modalities
- **Category**: Self-Supervised Learning Algorithm
- **Severity**: **HIGH**
- **Evidence** (`src/training/pretrain.py` Line 110–123):
  - `rand_mask = (torch.rand(B, 80, 1) < 0.3).float()`
  - Radar is absent in 84.6% of overpasses, represented by zero tensors.
  - When radar tokens are masked, the loss computes MSE against zero targets.
- **Root Cause**: Random masking was applied across all 80 multimodal slots without masking out unobserved modality slots.
- **Impact**: Pretraining wastes network capacity teaching the model to predict zero vectors for missing sensors rather than learning cross-modal atmospheric features.
- **Corrective Action**: Gate random patch masking by the boolean modality presence mask:
  $$\mathbf{M}_{\text{effective}} = \mathbf{M}_{\text{rand}} \odot \text{repeat}(\mathbf{M}_{\text{modality}}, 16)$$

---

### Issue H-3: Track Target Sparsity Due to Synoptic Time Matching
- **Category**: Target Engineering & Label Availability
- **Severity**: **HIGH**
- **Evidence** (`reports/phase2c/target_definition_audit.csv`):
  - Track displacement target $+6\text{h}$ is valid for only **402 out of 2,599 overpasses** (**84.5% missingness**).
- **Root Cause**: Overpasses occur continuously at arbitrary orbital times (e.g., 04:22 UTC), whereas Best Track fixes are primarily recorded at 6-hourly synoptic intervals (00, 06, 12, 18 UTC). The pipeline used a strict 30-minute matching window without temporal interpolation.
- **Impact**: Track forecasting cannot be trained effectively with only 15.5% label coverage.
- **Corrective Action**: Implement piecewise cubic Hermite spline interpolation on Best Track $(lat, lon)$ coordinates to generate continuous ground-truth positions for arbitrary $+6\text{h}, +12\text{h}, +24\text{h}$ horizons.

---

### Issue H-4: Extreme Class Imbalance in Rapid Intensification (RI)
- **Category**: Imbalanced Classification
- **Severity**: **HIGH**
- **Evidence** (`reports/phase2c/ri_class_imbalance_experiments.csv`):
  - Positive RI prevalence in pilot: **2.91%** (52 positive events out of 1,784 valid pairs).
  - Standard BCE loss: Precision = 0.00%, Recall = **0.00%**, F1 = **0.0000** (model predicts all negative).
  - Class-Ratio Weighted BCE ($w=33.3$): Recall = **73.08%**, PR-AUC = **0.1873**.
- **Impact**: Without loss reweighting or focal loss, the model fails to detect any Rapid Intensification events.
- **Corrective Action**: Deploy Class-Ratio Weighted BCE ($w \ge 15.0$) or Focal Loss ($\alpha=0.25, \gamma=2.0$) as the standard operational loss for `head_ri`.

---

## 4. Medium-Priority Engineering Items

### Issue M-1: Staging Manifest Omission of Partition and Sample Metadata
- **Category**: Metadata Tracking
- **Severity**: **MEDIUM**
- **Evidence**: `staging/staging_manifest.json` tracks 82 files with valid SHA-256 hashes, but sets `split: "unknown"` and `num_samples: 0`.
- **Corrective Action**: Update `src/infrastructure/staging.py` to join with `reports/pipeline/sample_index.csv`.

### Issue M-2: Modern Sensor Data Volume Expansion (+27.5% per Overpass)
- **Category**: Storage Estimation
- **Severity**: **MEDIUM**
- **Evidence**: Modern 2020+ overpasses average 612.9 KB vs. 480.7 KB for 1987–1999 overpasses due to high-channel radiometers (GMI, AMSR2) and DPR radar.
- **Corrective Action**: Apply dynamic per-era storage weights in the Phase 3 budget calculator rather than a single uniform average.

### Issue M-3: Static Geometric Graph Edges Lacking Environmental Wind Shear
- **Category**: Model Architecture
- **Severity**: **MEDIUM**
- **Evidence**: `src/models/graph/spatial_graph.py` computes edge attributes purely from static $(i, j)$ grid geometry.
- **Corrective Action**: Condition edge features on the 850–200 hPa ERA5 vertical wind shear magnitude and relative direction.

---

## 5. Summary Matrix & Action Plan

| Issue ID | Severity | Description | Resolution Phase | Verification Metric |
| :--- | :--- | :--- | :--- | :--- |
| **C-1** | CRITICAL | Local Disk Space Exhaustion (<71 GB free) | Pre-Phase 3 | Mount >= 1 TB External NVMe SSD |
| **C-2** | CRITICAL | Disproven GPU execution claims | Pre-Phase 3 | Automated SSH + PyTorch CUDA execution |
| **C-3** | CRITICAL | Test set has 0 Hurricane overpasses | Pre-Phase 3 | Peak-category stratified re-split |
| **H-1** | HIGH | Target normalization / initialization bias | Phase 3 Training | Target z-scoring & bias offset init |
| **H-2** | HIGH | MFM pretraining masking absent sensors | Phase 3 Pretrain | Modality-gated loss masking |
| **H-3** | HIGH | Track target sparsity (84.5% missing) | Pre-Phase 3 | Hermite spline label interpolation |
| **H-4** | HIGH | RI class imbalance (2.91% positive) | Phase 3 Training | Class-Ratio Weighted BCE / Focal Loss |
| **M-1** | MEDIUM | Staging manifest split metadata | Pre-Phase 3 | Join manifest with sample index |
| **M-2** | MEDIUM | Modern era +27.5% storage inflation | Pre-Phase 3 | Era-weighted storage formula |
| **M-3** | MEDIUM | Graph lacks dynamic wind shear vectors | Phase 3 Model | Shear-relative edge attributes |

**Conclusion**: All 10 issues are clearly diagnosed with established mathematical or engineering remedies. Once Critical items C-1, C-2, and C-3 are enacted, the system is fully prepared for Phase 3 dataset scaling.

