# TC PRIMED Masked Feature Modeling (MFM) & Transfer Provenance Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Elimination of Pretraining Input Leakage

In self-supervised Masked Feature Modeling (MFM), the objective is to force modality encoders and cross-modal attention layers to infer missing atmospheric structures (e.g. predicting hidden microwave eyewall convection from surrounding infrared and environmental moisture fields).

### The Flaw Identified in Phase 2D Audit
In previous implementations, masking was applied **after** the convolutional modality encoders:
```python
# Flawed legacy approach:
tokens = encoder(unmasked_full_image)  # Encoder already observed full image!
corrupted_tokens = tokens * (1.0 - mask) # Zeroing token after convolution
```
Because convolutional receptive fields span large spatial regions, the encoder had already observed all pixels. The network could trivially memorize features, defeating the self-supervised pretext task.

### Phase 2E.1 Solution: Raw Input Polar Sector Masking & Invariance
In `src/training/pretrain.py:run_pretraining`:
1. **Target Generation (`torch.no_grad()`):**
   Unmasked sensor arrays pass through the encoders with gradient tracking disabled:
   ```python
   with torch.no_grad():
       t_ir_tgt, v_ir_tgt = model.ir_encoder(ir, m_ir, pixel_mask=pix_ir, return_validity=True)
       t_pmw_tgt, v_pmw_tgt = model.pmw_encoder(pmw, m_pmw, channel_mask=ch_pmw, pixel_mask=pix_pmw, return_validity=True)
       ...
       all_targets = torch.cat([t_ir_tgt, t_pmw_tgt, ...], dim=1).detach()  # (B, 80, d_model)
   ```
2. **Valid-Token Selection via Native Spatial Coverage:**
   A spatial sector token is valid if and only if the modality is present ($m = 1$) **and** the spatial sector has strictly positive valid pixel coverage ($> 0$ valid pixels inside that sector at native coordinate resolution):
   ```python
   valid_tok_mask = torch.cat([v_ir_tgt, v_pmw_tgt, v_gp_tgt, v_rad_tgt, v_env_tgt], dim=1).unsqueeze(-1)  # (B, 80, 1)
   all_targets = all_targets * valid_tok_mask  # Suppress any target leakage from unobserved sectors
   ```
   Unobserved sectors (due to narrow satellite swaths or missing channels) have `valid_tok_mask == 0.0` and are **strictly excluded** from random masking selection and loss evaluation.
3. **Physical Polar Sector Masking:**
   When a spatial token $p \in \{0, \dots, 15\}$ of modality $m$ is selected for masking, the **raw input pixels** and the **pixel validity mask** of that spatial polar sector are zeroed out in the student input:
   ```python
   ir_stud[b, :, reg_masks[p] > 0] = 0.0
   pix_ir_stud[b, :, reg_masks[p] > 0] = 0.0
   ```
4. **Student Forward Pass:**
   The student encoders operate strictly on the masked tensors (`ir_stud`, `pmw_stud`, etc.).
   **The student encoder cannot observe masked regions through any receptive field.**
5. **Cross-Modal Transformer Prediction:**
   Cross-modal fusion combines visible tokens across modalities. In MFM, queries corresponding to masked observed tokens attend across other modalities and sectors. Truly unobserved sectors remain padded to prevent attention leakage.
6. **Masked Loss Computation:**
   Loss is evaluated strictly at masked valid tokens with exact feature element normalization:
   $$\mathcal{L}_{\text{MFM}} = \frac{\sum_{b, k} \|\hat{\mathbf{y}}_{b, k} - \mathbf{y}_{b, k}\|^2 \cdot \mathbf{M}_{b, k}^{\text{recon}}}{\sum_{b, k} \mathbf{M}_{b, k}^{\text{recon}} \cdot d_{\text{model}} + \epsilon}$$

---

## 2. Cell-Centered Reference Grid Spacing ($15.625\text{ km}$)

The storm-centered reference grid spanning $[-500, +500]\text{ km}$ at $64 \times 64$ resolution now uses exact cell-centered coordinates:
$$\Delta x = \frac{2 \cdot 500.0}{64} = 15.625\text{ km}, \quad \Delta y = \frac{2 \cdot 500.0}{64} = 15.625\text{ km}$$
$$x_j = -500.0 + (j + 0.5) \cdot 15.625\text{ km} \in [-492.1875, +492.1875]\text{ km}$$
$$y_i = 500.0 - (i + 0.5) \cdot 15.625\text{ km} \in [+492.1875, -492.1875]\text{ km}$$

This resolves the previous boundary endpoint artifact where `linspace(-500, 500, 64)` produced $(500 - (-500))/63 = 15.873\text{ km}$. Outer cell boundaries now span $[-500.0, +500.0]\text{ km}$ exactly.

---

## 3. Pretraining vs. Transfer Provenance

A key requirement of Phase 2E.1 is establishing unambiguous architectural provenance between self-supervised pretraining and downstream supervised fine-tuning.

### Pretraining Scope (Single Overpasses)
Pretraining operates on individual multi-modal overpass observations to learn invariant cross-sensor representations of cyclone physical structure.
- **Components Trained:**
  1. `IREncoder` (stem, ConvBlocks, polar pooling, token projection)
  2. `PMWEncoder` (stem, ConvBlocks, channel gating, polar pooling, token projection)
  3. `GPROFEncoder` (stem, ConvBlocks, channel gating, polar pooling, token projection)
  4. `RadarEncoder` (3D convolutions, trilinear pooling, polar pooling, token projection)
  5. `EnvEncoder` (stem, ConvBlocks, channel gating, polar pooling, token projection)
  6. `CrossModalFusion` (multi-head cross-attention across 80 tokens)
  7. `MaskedFeatureReconstructionHead` (projection MLP back to target feature space)

### Verified Gradient Flow
In unit test `test_08_mfm_pretraining_workflow_and_all_modality_gradients`, executing `run_pretraining()` across all 5 modalities verified non-zero gradient norms across all backbone components:
- `ir_encoder`: $0.031459 > 0$
- `pmw_encoder`: $0.032429 > 0$
- `gprof_encoder`: $0.027050 > 0$
- `radar_encoder`: $0.026716 > 0$
- `env_encoder`: $0.030069 > 0$
- `fusion`: $0.115706 > 0$
- `recon_head`: $0.232565 > 0$

### Transfer Scope (Supervised Multi-Task & Sequence Modeling)
When fine-tuning on forecasting tasks (`run_finetuning`) or supervised training (`train_multitask`):
- **Transferred Backbone Weights:**
  - All 5 Modality Encoders (`ir_encoder`, `pmw_encoder`, `gprof_encoder`, `radar_encoder`, `env_encoder`).
  - Cross-Modal Fusion Transformer (`fusion`).
  - Successfully loaded via `load_pretrained_backbone(strict=False)` with 254 parameter tensors transferred.
- **Freshly Initialized Task Structures:**
  - `SpatialGraphEncoder`: Initialized fresh. (MPNN learns task-specific spatial propagation).
  - `CausalTemporalTransformer`: Initialized fresh. (Temporal attention learns multi-timestep historical sequencing, which does not exist in single-overpass pretraining).
  - `TaskPredictionHeads`: Initialized fresh. (Linear heads for intensity, pressure, stage classification, intensity change, RI, and track).

This separation is scientifically sound and eliminates any false claim that temporal or graph parameters were trained during single-overpass pretraining.
