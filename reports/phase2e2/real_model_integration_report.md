# TC-GraphFormer — Phase 2E: Pilot Integration Test Report

**Execution Date:** 2026-11-14  
**Dataset:** NOAA TC PRIMED v01r01-final (Pilot ~20 GB, 2,599 observations)  
**Verification Target:** Complete end-to-end data-to-model integration pipeline with 100% pilot observation audit and active radar learning

---

## 1. Summary of End-to-End Test Execution

The pilot integration test verified the entire causal forecasting pipeline on real satellite observations:

```text
Standardized NetCDF Containers (2,599 observations audited)
  -> Full All-Sample Data Audit (Zero non-finite tensors, zero processing errors)
  -> TCPRIMEDSequenceDataset (Real storm history AL022021 with GMI DPR radar, T=4 observations)
  -> collate_sequence_batch (Batched [B, T, C, H, W] with pixel masks, voxel masks, channel masks, delta-t)
  -> 5 Modality Encoders (IR, PMW, GPROF, Radar, Environmental Synoptic)
  -> Polar Sector Pooling (4 radial rings x 4 azimuthal sectors = 16 geometric tokens)
  -> Mask-Aware Cross-Modal Attention Fusion (80 multimodal tokens with attention padding masks)
  -> Storm-Relative Spatial Graph MPNN (16 geometric nodes, 8 edge dimensions)
  -> Continuous-Time Causal Temporal Transformer (Irregular delta-t hours)
  -> Anchor Horizon Representation Gathering (T_anchor)
  -> Multi-Task Prediction Heads (Wind, Pressure, Stage, dV, RI, Track)
  -> Masked MultiTaskLoss (Zero contribution from missing targets)
  -> Backpropagation & Parameter Gradient Updates (All 10 modules including radar_encoder)
  -> Checkpoint Serialization & Integrity Verification
```

---

## 2. Comprehensive All-Sample Data Audit (2,599 Observations)

Every observation in `sample_index.csv` ($N = 2599$) was dereferenced and verified through `TCPRIMEDDataset`:

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Total Pilot Observations Audited** | `2599` / `2599` | **100% Dereferenced** |
| **Audit Execution Time** | `40.8 s` (`63.7` samp/s) | Clean Execution |
| **Audit Database Artifact** | [`pilot_all_observations_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2e2/pilot_all_observation_audit.csv ) | Serialized |
| **Invalid / Non-Finite Tensors** | `0` | **Zero Defects** |
| **Shape Mismatches** | `0` | **Zero Defects** |
| **Container Processing Errors** | `0` | **Zero Defects** |

### Modality Availability Across 2,599 Observations

| Modality | Available Count | Availability Percentage |
| :--- | :--- | :--- |
| **Infrared (IRWIN)** | `2262` | `87.0%` |
| **Passive Microwave (PMW)** | `2247` | `86.5%` |
| **GPROF Precipitation** | `2595` | `99.8%` |
| **Radar Radiometer (DPR/PR)** | `75` | `2.9%` |
| **Environmental Reanalysis (ERA5)** | `2521` | `97.0%` |

### Coordinate Provenance Breakdown

| **Radar** | none: 2524, radar_beam_geodetic_km: 74, nominal_fallback: 1 |
| **Passive_microwave** | tangent_plane_geodetic_km: 2247, none: 352 |


---

## 3. Sensor and Modality Loading Results

| Instrument | Sample ID | Storm ID | PMW Channels Present | Radar Present | Env Present | PMW Tensor Sum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GMI** | SMP_AL022021_GMI_GPM_20210614123541 | AL022021 | [1.0, 1.0, 1.0, 1.0] | Yes | Yes | 8349.8 |
| **TMI** | SMP_AL022001_TMI_TRMM_20010711231736 | AL022001 | [1.0, 1.0, 1.0, 0.0] | No | Yes | 3632.08 |
| **AMSR2** | SMP_AL022021_AMSR2_GCOMW1_20210614072518 | AL022021 | [1.0, 1.0, 0.0, 0.0] | No | Yes | 2905.86 |
| **SSMI** | SMP_AL011988_SSMI_F08_19880806102620 | AL011988 | [1.0, 1.0, 1.0, 1.0] | No | Yes | 7994.38 |
| **SSMIS** | SMP_AL022021_SSMIS_F16_20210613214311 | AL022021 | [1.0, 1.0, 1.0, 0.0] | No | Yes | 5976.08 |
| **AMSUB** | SMP_AL022001_AMSUB_NOAA16_20010711162751 | AL022001 | [0.0, 0.0, 0.0, 1.0] | No | Yes | 2911.74 |
| **MHS** | SMP_AL022021_MHS_METOPB_20210614013814 | AL022021 | [0.0, 0.0, 0.0, 1.0] | No | Yes | 2429.27 |

- **Radar Radiometer Loading:** Verified on `SMP_AL022021_GMI_GPM_20210614123541`. Ku reflectivity profile `(88, 373, 49)` interpolated to `(1, 16, 32, 32)` with 255 valid voxels preserved and max normalized dBZ of `0.529`.
- **Radar-Absent Verification:** Verified on no-radar overpasses; radar tensor is strictly zero and `modality_masks[3] == 0.0`.

---

## 4. Real Storm Historical Sequence with Radar

- **Storm:** `AL022021`
- **Sequence Length:** `T = 4`
- **Timestamps and Instruments:**
  - Step 0: `SMP_AL022021_SSMIS_F18_20210614093137` (SSMIS) at `2021-06-14 09:31:37 UTC` (Delta-t:  0.00h) — No radar
  - Step 1: `SMP_AL022021_SSMIS_F16_20210614100848` (SSMIS) at `2021-06-14 10:08:48 UTC` (Delta-t:  0.62h) — No radar
  - Step 2: `SMP_AL022021_SSMIS_F17_20210614122616` (SSMIS) at `2021-06-14 12:26:16 UTC` (Delta-t:  2.91h) — No radar
  - Step 3: `SMP_AL022021_GMI_GPM_20210614123541` (GMI) at `2021-06-14 12:35:41 UTC` (Delta-t:  3.07h) — RADAR PRESENT (255 valid voxels)

---

## 5. Multi-Task Target and Mask Values for Anchor Observation

- **Anchor Sample:** `SMP_AL022021_GMI_GPM_20210614123541`
- **Current Intensity:** `30.0 kt` (Mask: `1.0`)
- **Central Pressure:** `1005.0 hPa` (Mask: `1.0`)
- **Development Stage:** Class `0` (Mask: `1.0`, Raw: `TD`)
- **Intensity Changes (+6h, +12h, +18h, +24h):** `[[5.0, 10.0, 20.0, 24.0]] kt` (Masks: `[[1.0, 1.0, 1.0, 1.0]]`)
- **Rapid Intensification (>=30kt/24h):** `0.0` (Mask: `1.0`)
- **Track Displacements (East/North km):**
  - +6h: `[0.0, 0.0] km` (Mask: `0.0`)
  - +12h: `[0.0, 0.0] km` (Mask: `0.0`)
  - +18h: `[0.0, 0.0] km` (Mask: `0.0`)
  - +24h: `[0.0, 0.0] km` (Mask: `0.0`)

---

## 6. Model Parameter Gradient Norms

All 10 modules, including `radar_encoder`, received valid autograd signals without gradient saturation or vanishing:

| Architecture Component | L2 Gradient Norm | Training Status | Active Verification |
| :--- | :--- | :--- | :--- |
| `ir_encoder` | `3.340783` | Updated | > 0.0 Verified |
| `pmw_encoder` | `4.535266` | Updated | > 0.0 Verified |
| `gprof_encoder` | `2.879015` | Updated | > 0.0 Verified |
| `radar_encoder` | `2.954607` | Updated | **> 0.0 Confirmed (Active Radar Learning)** |
| `env_encoder` | `3.176214` | Updated | > 0.0 Verified |
| `cross_modal_fusion` | `3.896943` | Updated | > 0.0 Verified |
| `spatial_aggregator` | `14.098198` | Updated | > 0.0 Verified |
| `spatial_graph_mpnn` | `2.298577` | Updated | > 0.0 Verified |
| `temporal_transformer` | `5.952555` | Updated | > 0.0 Verified |
| `task_prediction_heads` | `7.878052` | Updated | > 0.0 Verified |

---

## 7. Verification Status

- [x] All 2,599 pilot observations dereferenced and verified (zero non-finite tensors, zero processing errors).
- [x] Full audit database generated and saved to `reports/phase2e/pilot_all_observations_audit.csv`.
- [x] Radar Ku measurements extracted and normalized into 3D volumes with precipitation echo masks.
- [x] Deterministic PMW channel mapping enforced per sensor family.
- [x] Historical sequence strictly causal ($t_k \le t_{anchor}$) with radar-present observation verified.
- [x] Pixel and voxel validity masks explicitly propagated to model forward pass.
- [x] Complete backward pass generates non-zero parameter updates across all components, including `radar_encoder`.
- [x] Checkpoint serialization and schema verified.
