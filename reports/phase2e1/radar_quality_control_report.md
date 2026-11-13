# TC PRIMED Radar Volumetric Quality Control Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Atmospheric Radar Structure in NOAA TC PRIMED

NOAA TC PRIMED v01r01 includes spaceborne precipitation radar data from two primary instruments:
1. **TRMM Precipitation Radar (PR):** 13.8 GHz (Ku-band), operating 1997–2015.
2. **GPM Dual-Frequency Precipitation Radar (DPR):** 13.6 GHz (Ku-band) and 35.5 GHz (Ka-band), operating 2014–present.

In the standardized NetCDF-4 containers, radar volumetric observations are stored under the group `radar_radiometer/` with variable `correctedReflectFactor_Ku`:
- **Dimensions:** `(range_gates: 88, nscans: ~350, nray: 49)`.
- **Vertical Coverage:** 88 range gates spanning $0\text{ km}$ to $\sim 21.8\text{ km}$ altitude, with gate resolution $\Delta z \approx 250\text{ m}$.
- **Horizontal Swath Width:** 49 cross-track rays spanning approximately $245\text{ km}$ (GPM Ku-swath width).

---

## 2. Quality Control & Range Thresholds

Radar reflectivity factor $Z$ in decibels relative to $1\text{ mm}^6/\text{m}^3$ (dBZ) exhibits wide dynamic range:
- **Atmospheric Noise / Clear Air:** $-30\text{ dBZ}$ to $+10\text{ dBZ}$.
- **Minimum Detectable Signal (MDS):** $\approx +12\text{ dBZ}$ for GPM DPR Ku-band.
- **Stratiform Rain / Melting Layer:** $+20\text{ dBZ}$ to $+40\text{ dBZ}$ (bright-band signatures).
- **Convective Eyewall Cores:** $+45\text{ dBZ}$ to $+55\text{ dBZ}$.
- **Severe Hail / Graupel Columns:** Up to $+60\text{ dBZ}$.

### Audit of Pilot Radar Profiles
In the local 20 GB pilot dataset:
- Exactly 76 NetCDF files contain GMI/DPR overpasses with `has_radar == True`.
- All 76 files have valid Ku reflectivity profiles with `availability_flag == 1`.
- Peak convective reflectivities in the pilot reach **$55.06\text{ dBZ}$** (observed in Hurricane Bill `AL032009`).
- Missing gates and out-of-swath rays are indicated by sentinel values $\le -9000.0$ or NaNs.

### Phase 2E.1 Quality Control Criteria
In `src/data/dataset.py:_load_radar_volume`:
1. **Physical Range Gate Filtering:**
   $$\text{valid} = (\neg \text{isnan}(\text{Ku})) \land (\text{Ku} > -9000.0) \land (\text{Ku} \ge -20.0\text{ dBZ}) \land (\text{Ku} \le 100.0\text{ dBZ})$$
2. **Precipitation Echo Classification:**
   Values with $\text{Ku} \ge 12.0\text{ dBZ}$ represent genuine precipitation echoes exceeding the instrument detection threshold.
3. **Physical Normalization:**
   Reflectivity is normalized onto $[0.0, 1.0]$ spanning $[-20.0\text{ dBZ}, +80.0\text{ dBZ}]$:
   $$\text{Ku}_{\text{norm}} = \text{clip}\left(\frac{\text{Ku} - (-20.0)}{100.0}, \, 0.0, \, 1.0\right)$$
   Where:
   - $-20\text{ dBZ} \to 0.00$
   - $+12\text{ dBZ} \to 0.32$ (detection threshold)
   - $+40\text{ dBZ} \to 0.60$ (moderate rain)
   - $+60\text{ dBZ} \to 0.80$ (intense convective eyewall)
   - $+80\text{ dBZ} \to 1.00$ (upper physical boundary)

---

## 3. 3D Volumetric Interpolation & Voxel Masking

To integrate the irregular volumetric radar swath with the storm-centered architecture:
1. Spatial dimensions $(nscans, nray)$ are interpolated to $(H/2, W/2) = (32, 32)$ using 2D/3D trilinear interpolation.
2. Vertical range gates are resampled from 88 gates to $D=16$ uniform vertical atmospheric layers ($0$ to $20\text{ km}$ altitude, $\Delta z \approx 1.25\text{ km}$).
3. The binary voxel validity mask (`radar_voxel_mask`) is downsampled using **nearest-neighbor interpolation** to prevent partial interpolation weights from fabricating false echoes in empty voxels.
4. Cleaned volume tensor enforces:
   $$\mathbf{X}_{\text{radar}} = \mathbf{X}_{\text{resized}} \odot \mathbf{M}_{\text{voxel\_resized}}$$
5. Downstream `RadarEncoder` applies 3D convolutions with voxel-mask gating, guaranteeing that unobserved radar regions contribute zero activation to multi-modal cross-attention.
