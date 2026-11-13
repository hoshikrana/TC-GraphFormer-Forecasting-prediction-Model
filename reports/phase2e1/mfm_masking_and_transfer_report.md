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

### Phase 2E.1 Solution: Raw Input Spatial Patch Masking
In `src/training/pretrain.py:run_pretraining`:
1. **Target Generation (`torch.no_grad()`):**
   Unmasked sensor arrays pass through the encoders with gradient tracking disabled:
   ```python
   with torch.no_grad():
       t_ir_tgt = model.ir_encoder(ir, m_ir, pixel_mask=pix_ir).detach()
       t_pmw_tgt = model.pmw_encoder(pmw, m_pmw, channel_mask=ch_pmw, pixel_mask=pix_pmw).detach()
       ...
       all_targets = torch.cat([t_ir_tgt, t_pmw_tgt, ...], dim=1)  # (B, 80, d_model)
   ```
2. **Raw Input Patch Masking:**
   When a spatial token $p \in \{0, \dots, 15\}$ of modality $m$ is selected for masking, the **raw input pixels** and the **pixel validity mask** of that spatial region are zeroed out in the student input:
   ```python
   # Patch p maps to grid block [h_s:h_e, w_s:w_e]
   ir_stud[b, :, h_s:h_e, w_s:w_e] = 0.0
   pix_ir_stud[b, :, h_s:h_e, w_s:w_e] = 0.0
   ```
3. **Student Forward Pass:**
   The student encoders operate strictly on the masked tensors (`ir_stud`, `pmw_stud`, etc.).
   **The student encoder cannot observe masked regions through any receptive field.**
4. **Reconstruction Head:**
   Cross-modal fusion combines visible tokens across modalities. The reconstruction head predicts the target token representations at the masked positions.
5. **Masked Loss Computation:**
   Loss is evaluated strictly at masked valid tokens:
   $$\mathcal{L}_{\text{MFM}} = \frac{\sum_{b, k} \|\hat{\mathbf{y}}_{b, k} - \mathbf{y}_{b, k}\|^2 \cdot \mathbf{M}_{b, k}^{\text{recon}}}{\sum_{b, k} \mathbf{M}_{b, k}^{\text{recon}} \cdot d_{\text{model}} + \epsilon}$$

---

## 2. Pretraining vs. Transfer Provenance

A key requirement of Phase 2E.1 is establishing unambiguous architectural provenance between self-supervised pretraining and downstream supervised fine-tuning.

### Pretraining Scope (Single Overpasses)
Pretraining operates on individual multi-modal overpass observations to learn invariant cross-sensor representations of cyclone physical structure.
- **Components Trained:**
  1. `IREncoder` (stem, ConvBlocks, token projection)
  2. `PMWEncoder` (stem, ConvBlocks, channel gating)
  3. `GPROFEncoder` (stem, ConvBlocks, channel gating)
  4. `RadarEncoder` (3D convolutions, trilinear pooling)
  5. `EnvEncoder` (stem, ConvBlocks, channel gating)
  6. `CrossModalFusion` (multi-head cross-attention across 80 tokens)
  7. `MaskedFeatureReconstructionHead` (projection MLP back to target feature space)

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
