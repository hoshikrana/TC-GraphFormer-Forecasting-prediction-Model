# TC PRIMED & TC-GraphFormer Remaining Known Issues & Limitations
**Phase:** 2E — Targeted Core Data-to-Model Integration Repair  
**Document:** Scientific & Operational Limitation Register  
**Status:** Open for Resolution during Phase 3 Dataset Scaling  

---

## 1. Observational Sparsity in 3D Precipitation Radar

### Limitation Description
In the 20 GB pilot dataset (2,599 overpass files across 80 storms), active spaceborne precipitation radar measurements (`radar_radiometer/correctedReflectFactor_Ku`) are present in only **76 overpasses (2.9%)**.

### Physical Cause
Spaceborne precipitation radars (the Precipitation Radar on TRMM and Dual-frequency Precipitation Radar on GPM Core Observatory) are active instruments with narrow orbital swaths ($245\text{ km}$ across track). In contrast, passive microwave radiometers have swaths spanning $800 - 1700\text{ km}$, and geostationary IR covers the entire storm disk. Consequently, coincident radar overpasses of the storm center are rare physical occurrences.

### Current Mitigation & Status in Phase 2E
- The dataset loader detects valid radar observations dynamically, setting `has_radar = True` strictly when non-zero, non-fill Ku reflectivities exist.
- When absent, the radar tensor is zero-masked with `radar_pixel_mask = 0.0`.
- The cross-modal attention module dynamically attends to active modalities via `modality_masks`.
- In Phase 3, the full 2000–2025 dataset will include all available GPM and TRMM overpasses, yielding an estimated $\approx 2,500$ radar overpasses.

---

## 2. Passive Microwave (PMW) Constellation Heterogeneity

### Limitation Description
The 9 satellite sensor families in TC PRIMED fall into two distinct physical design classes:
1. **Conical Scanners (GMI, TMI, SSMI, SSMIS, AMSR2, AMSRE):** Possess low-frequency imaging channels ($10.65\text{ GHz}, 19.35\text{ GHz}, 37.0\text{ GHz}$) with dual polarization (V and H) that penetrate clouds to sense surface wind and liquid hydrometeors.
2. **Cross-Track Sounders (MHS, AMSUB, ATMS):** Primarily atmospheric temperature and humidity sounders lacking low-frequency channels; they observe high-frequency sounding channels ($88.2 - 190.3\text{ GHz}$).

### Current Mitigation & Status in Phase 2E
- The Phase 2E loader projects all sensors onto a canonical 4-frequency physical grid (`~19V, ~19H, ~37V, ~85V`) with an accompanying binary vector `pmw_channel_mask (4,)`.
- For MHS, AMSUB, and ATMS, channels 0–2 are set to 0.0, and `pmw_channel_mask[3] = 1.0`.
- This prevents sounders from corrupting low-frequency channels with false cold temperatures. However, single-channel sounder observations lack the low-frequency surface emission information available in conical scanners.

---

## 3. ERA5 Spatial Resolution vs. Eyewall Scale

### Limitation Description
The environmental synoptic reanalysis fields (ECMWF ERA5) are sampled at $0.25^\circ \times 0.25^\circ$ spatial resolution ($\approx 25 - 28\text{ km}$ at tropical latitudes).

### Physical Consequence
While $25\text{ km}$ resolution is well-suited for capturing large-scale synoptic steering flow ($u_{10m}, v_{10m}$), environmental vertical wind shear, and broad sea surface temperature (SST) gradients, it cannot resolve the intense inner core pressure gradient or eyewall dynamics ($< 10\text{ km}$).

### Current Mitigation
- TC-GraphFormer fuses ERA5 strictly as a large-scale background environmental context modality.
- Inner core vortex dynamics and eyewall structure are extracted from the high-resolution satellite imagery (IR at $4\text{ km}$ and PMW at $10-15\text{ km}$).

---

## 4. Best-Track Temporal Interpolation

### Limitation Description
Operational best-track records (from NHC and JTWC) provide intensity and center fixes at standard synoptic 6-hour intervals (00:00, 06:00, 12:00, 18:00 UTC). Low-Earth-orbit satellite overpasses occur at arbitrary intermediate minutes (e.g. 14:37 UTC).

### Implementation in TC PRIMED
TC PRIMED linearly interpolates the best-track maximum sustained wind and central pressure to the exact satellite overpass timestamp. For storms undergoing rapid intensification (RI) or sudden eyewall replacement, linear interpolation over a 6-hour interval may introduce minor temporal smoothing.

---

## 5. Compute & Storage Scaling for Phase 3 (500 GB Dataset)

### Operational Consideration
The Phase 2E pipeline and model architecture are validated on the 20 GB pilot dataset (2,599 overpasses). Expanding to the 2000–2025 curated dataset will involve:
- Approximately 50,000 overpasses ($\approx 500\text{ GB}$ raw NetCDF storage).
- Single-epoch training on CPU is not viable at 50,000 samples.
- **Requirement for Machine B:** The two-machine staging architecture (`scripts/stage_two_machine_data.py`) must be used to transfer preprocessed `.tar.gz` chunks to the remote GPU node (Machine B) for multi-epoch GPU training under CUDA/cuDNN.
