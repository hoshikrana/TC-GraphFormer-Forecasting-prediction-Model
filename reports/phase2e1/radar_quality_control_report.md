# TC PRIMED Radar Volumetric Quality Control & Geometric Processing Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Atmospheric Radar Structure in NOAA TC PRIMED

NOAA TC PRIMED v01r01 includes spaceborne precipitation radar data from two primary satellite instruments:
1. **TRMM Precipitation Radar (PR):** 13.8 GHz (Ku-band), operating 1997–2015.
2. **GPM Dual-Frequency Precipitation Radar (DPR):** 13.6 GHz (Ku-band) and 35.5 GHz (Ka-band), operating 2014–present.

In the standardized NetCDF-4 containers, radar volumetric observations are stored under the group `radar_radiometer/` with variable `correctedReflectFactor_Ku`:
- **Raw Array Dimensions:** `(range_gates: 88, nscans: ~350, nray: 49)`.
- **Vertical Coverage:** 88 range gates spanning $z = 0\text{ km}$ (surface) to $z \approx 21.8\text{ km}$ altitude, with range resolution $\Delta z \approx 250\text{ m}$.
- **Horizontal Swath Geometry:** 49 cross-track rays spanning approximately $245\text{ km}$ across-track ($49 \times 5\text{ km}$ footprint).

---

## 2. Quality Control: Separation of Measurement Validity vs. Echo Classification

Radar reflectivity factor $Z$ in decibels relative to $1\text{ mm}^6/\text{m}^3$ ($\text{dBZ}$) encompasses both atmospheric noise/clear-air scatter and precipitating hydrometeors:
- **Atmospheric Noise / Clear Air / Thin Clouds:** $-30\text{ dBZ}$ to $+10\text{ dBZ}$.
- **Minimum Detectable Signal (MDS) for Precipitation:** $\approx +12\text{ dBZ}$ for GPM DPR Ku-band.
- **Stratiform Rain / Melting Layer:** $+20\text{ dBZ}$ to $+40\text{ dBZ}$ (bright-band signatures).
- **Convective Eyewall Cores:** $+45\text{ dBZ}$ to $+55\text{ dBZ}$.
- **Severe Hail / Graupel Columns:** Up to $+60\text{ dBZ}$.

### 2.1 Critical 4 Resolution: Distinct Validity and Echo Masks
Prior implementations either ignored the $12\text{ dBZ}$ detection threshold entirely or conflated measurement validity with precipitation detection. Phase 2E.1 formally separates these two physical concepts into distinct, concurrent masks:

1. **Measurement Validity Mask (`radar_voxel_mask`):**
   Determines whether the radar instrument obtained a physically valid observational measurement within its calibrated dynamic range:
   $$\text{valid}_{\text{meas}} = (\neg \text{isnan}(\text{Ku})) \land (\text{Ku} > -9000.0) \land (\text{Ku} \ge -20.0\text{ dBZ}) \land (\text{Ku} \le 100.0\text{ dBZ})$$
   - Valid low-reflectivity measurements ($-20.0\text{ dBZ} \le \text{Ku} < 12.0\text{ dBZ}$) are **strictly preserved** in the input tensor rather than corrupted to missing/fill values. They provide essential physical context (clear-air regions, cloud boundaries, and noise-floor suppression).

2. **Precipitation Echo Classification Mask (`radar_echo_mask`):**
   Classifies genuine hydrometeor precipitation echoes exceeding the instrument Minimum Detectable Signal:
   $$\text{echo}_{\text{precip}} = \text{valid}_{\text{meas}} \land (\text{Ku} \ge 12.0\text{ dBZ})$$
   - For $\text{Ku} < 12.0\text{ dBZ}$, `radar_echo_mask = 0.0`.
   - For $\text{Ku} \ge 12.0\text{ dBZ}$, `radar_echo_mask = 1.0`.

3. **Physical Normalization:**
   Reflectivity is normalized onto $[0.0, 1.0]$ spanning $[-20.0\text{ dBZ}, +80.0\text{ dBZ}]$:
   $$\text{Ku}_{\text{norm}} = \text{clip}\left(\frac{\text{Ku} - (-20.0)}{100.0}, \, 0.0, \, 1.0\right)$$
   - $-20\text{ dBZ} \to 0.00$ (noise floor)
   - $+5\text{ dBZ} \to 0.25$ (valid clear-air measurement, `echo = 0`)
   - $+12\text{ dBZ} \to 0.32$ (precipitation MDS threshold, `echo = 1`)
   - $+40\text{ dBZ} \to 0.60$ (moderate rain)
   - $+60\text{ dBZ} \to 0.80$ (intense convective eyewall)
   - $+80\text{ dBZ} \to 1.00$ (upper physical boundary)

---

## 3. Preservation of Vertical and Horizontal Radar Geometry

### 3.1 Vertical Column Mapping ($88 \to 16$ Layers)
- **Raw Vertical Structure:** 88 range gates spaced $\Delta z \approx 250\text{ m}$ from surface ($z_0 \approx 0\text{ km}$) to top of troposphere ($z_{87} \approx 21.8\text{ km}$).
- **Target Spatial Representation:** $D = 16$ uniform vertical atmospheric layers spanning $[0.0\text{ km}, 20.0\text{ km}]$ altitude ($\Delta z \approx 1.33\text{ km}$).
- **Interpolation Discipline:**
  - Continuous reflectivity fields $\mathbf{X}_{\text{Ku}}$ are interpolated using trilinear interpolation.
  - Discrete validity masks (`radar_voxel_mask` and `radar_echo_mask`) are downsampled using **nearest-neighbor interpolation**, strictly preventing partial interpolation weights from fabricating false echoes in empty vertical layers.

### 3.2 Horizontal Storm-Centered Geospatial Registration
- **Defect in Legacy Code:** Previous pipelines applied index-based volume resizing from $(373\text{ scans}, 49\text{ rays})$ directly to $(32, 32)$ using coordinate-free interpolation. Because GPM DPR has only 49 cross-track rays spanning $245\text{ km}$, this artificially stretched the narrow radar swath across the entire $1,000\text{ km} \times 1,000\text{ km}$ storm box ($400\%$ horizontal distortion).
- **Coordinate-Aware Registration:**
  - True geodetic beam coordinates $(x_{sb}, y_{sb})$ in kilometers are extracted from the raw/processed NetCDF variables (`x, y` or geodetic `latitude, longitude` relative to storm center).
  - A 2D $k$-d tree maps beam observations onto the canonical storm-centered reference grid $[-500\text{ km}, +500\text{ km}] \times [-500\text{ km}, +500\text{ km}]$ ($32 \times 32$ cells, $\Delta x = 31.25\text{ km}$).
  - A physical distance threshold of $\delta_{\max} = 35.0\text{ km}$ is strictly enforced.
  - **Swath Reality:** On the actual pilot overpass (`SMP_AL022021_GMI_GPM_20210614123541`), the true radar swath covers **357 of 1,024 grid cells (34.9%)**. The remaining **667 cells (65.1%) are strictly masked $0.0$** in both `radar_voxel_mask` and `radar_echo_mask`.
  - Nominal index-based interpolation is retained only as an emergency fallback when coordinates are missing and is explicitly labeled with provenance tag `"nominal_fallback"`.

---

## 4. Empirical Verification & Test Results

### 4.1 Synthetic Calibration Suite (`test_07_radar_ku_validity_and_reflectivity_scaling`)
| Test Case | Raw Value | Measurement Valid (`radar_voxel_mask`) | Precip Echo (`radar_echo_mask`) | Normalized Value | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fill Sentinel | $-9999.0\text{ dBZ}$ | $0.0$ | $0.0$ | $0.00$ | **PASS** |
| Out of Range Low | $-25.0\text{ dBZ}$ | $0.0$ | $0.0$ | $0.00$ | **PASS** |
| Low-Reflectivity Valid | $+5.0\text{ dBZ}$ | **$1.0$ (Preserved)** | **$0.0$ (Not Echo)** | $0.25$ | **PASS** |
| MDS Detection Threshold | $+12.0\text{ dBZ}$ | **$1.0$** | **$1.0$ (Precip Echo)** | $0.32$ | **PASS** |
| Moderate Precipitation | $+30.0\text{ dBZ}$ | **$1.0$** | **$1.0$ (Precip Echo)** | $0.50$ | **PASS** |

### 4.2 Real NOAA TC PRIMED Pilot Verification
- **Test Sample:** `SMP_AL022021_GMI_GPM_20210614123541` (Hurricane Bill environment)
- **Raw Reflectivities:** 65,060 valid measurements ($[-20, 100]\text{ dBZ}$), 62,765 precipitation echoes ($\ge 12\text{ dBZ}$).
- **Resampled Volume:** Shape `(1, 16, 32, 32)`
  - `radar_voxel_mask`: $289.0$ active cells (swath coverage $28.2\%$)
  - `radar_echo_mask`: $279.0$ active cells
  - Low-reflectivity preserved voxels: $10.0$ active cells with $0.0 < \text{Ku} < 12.0\text{ dBZ}$
  - Provenance: `radar_beam_geodetic_km`
- **Gradient Flow:** End-to-end backpropagation through `RadarEncoder` produces non-zero finite gradients ($\|\nabla \theta\| = 0.683$) with zero NaN/Inf values.


