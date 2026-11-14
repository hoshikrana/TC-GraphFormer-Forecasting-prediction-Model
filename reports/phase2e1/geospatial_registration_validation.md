# TC PRIMED Geospatial Registration & Swath Alignment Validation Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Associated Test:** [test_phase2e1_scientific_integrity.py](file:///d:/Project/Capstone_project/tests/test_phase2e1_scientific_integrity.py)

---

## 1. Storm-Centered Canonical Reference Grid

In tropical cyclone satellite remote sensing, observations originate from diverse satellite platforms in low-Earth orbit (swaths with varying cross-track widths and scan geometries) and geostationary orbit (rectilinear planetary projections).

To provide physically consistent multi-modal alignment across instruments, Phase 2E.1 establishes a **standardized storm-centered Cartesian reference grid**:

$$\mathcal{G}_{\text{ref}} = \left\{ (x_i, y_j) \in [-500\text{ km}, +500\text{ km}] \times [+500\text{ km}, -500\text{ km}] \right\}$$

- **Spatial Domain:** $1,000\text{ km} \times 1,000\text{ km}$ ($[-500\text{ km}, +500\text{ km}]$ relative to vortex center).
- **Grid Dimensions:** $64 \times 64$ spatial cells.
- **Pixel Resolution:** $\Delta x = \Delta y = \frac{1000\text{ km}}{64} = 15.625\text{ km}$.
- **Center Alignment:** The origin $(x=0, y=0)$ coincides with the cyclone center $(lat_0, lon_0)$ reported in the best-track / TC PRIMED target variables.
- **Orientation:** $X$ is positive East (zonal displacement), $Y$ is positive North (meridional displacement).

---

## 2. Source Coordinate Systems in TC PRIMED NetCDF Files

A comprehensive audit of raw pilot NetCDF files (`data/raw/pilot/`) and standardized processed containers (`data/processed/pilot/`) revealed two coordinate regimes:

### 2.1 Universal Transverse Mercator (UTM) Metric Coordinates
- **Source Variables:** `x` and `y` in kilometers (`units: km`).
- **Found in:** Raw NetCDF files for `infrared`, `passive_microwave/S1`, `passive_microwave/S2`, and `radar_radiometer/KuGMI`.
- **Calculation:** Derived by NOAA TC PRIMED using GeographicLib Transverse Mercator projection centered on the interpolated storm track center.
- **Processing in Phase 2E.1:** When `x` and `y` are present, they are utilized directly as true storm-relative metric coordinates. `source = "utm_storm_centered_km"`.

### 2.2 Geodetic Coordinates with Tangent-Plane Projection
- **Source Variables:** `latitude` and `longitude` in degrees.
- **Found in:** Processed pilot containers (`passive_microwave/S1`, `passive_microwave/S2`).
- **Calculation:** Projected onto the local storm-centered tangent plane via equirectangular metric transformation:
  $$\Delta x = (\lambda - \lambda_0) \cdot \cos(\phi_0) \cdot 111.32\text{ km/deg}$$
  $$\Delta y = (\phi - \phi_0) \cdot 110.57\text{ km/deg}$$
  Where $(\phi_0, \lambda_0)$ is the storm center latitude and longitude, with dateline wrapping handled via $((\Delta \lambda + 180^\circ) \pmod{360^\circ}) - 180^\circ$.
- **Processing in Phase 2E.1:** `source = "tangent_plane_geodetic_km"`.

### 2.3 Nominal Fallback Regime & Provenance Tracking
- **Found in:** Older containers or synthetic test cases where coordinate variables were not copied during packaging.
- **Behavior:** Falls back safely to bilinear/nearest interpolation over the nominal $[-500, +500]\text{ km}$ domain, explicitly tagging metadata with `source = "nominal_fallback"`.
- **Integrity Rule:** Nominal fallback is never described as verified geospatial alignment; it is tracked explicitly in sample metadata under `geospatial_provenance`.

---

## 3. KDTree Swath Resampling & Swath Edge Masking

A critical defect identified in Phase 2D was **artificial pixel fabrication**: when a narrow satellite swath (e.g., $400\text{ km}$ wide) was interpolated to the full $64 \times 64$ grid using unconstrained bilinear interpolation, regions outside the swath were filled with interpolated/mean values and marked as valid pixels.

### Phase 2E.1 Solution
In `src/data/dataset.py:TCPRIMEDDataset._resample_swath_with_coords`:
1. Valid swath points are identified strictly where:
   $$\text{valid} = (\neg \text{isnan}(v)) \land (v > -9000.0) \land (\neg \text{isnan}(x)) \land (\neg \text{isnan}(y)) \land (|x| \le 1000\text{ km}) \land (|y| \le 1000\text{ km})$$
2. A fast $k$-d tree (`scipy.spatial.cKDTree`) is constructed over valid swath coordinates.
3. Reference grid points $(x_{\text{ref}}, y_{\text{ref}}) \in \mathbb{R}^{4096 \times 2}$ are queried with a **distance upper bound** $\delta_{\max} = 35.0\text{ km}$ (matching typical passive microwave footprint spacing):
   $$\text{dist}_k, \, \text{idx}_k = \text{tree.query}(p_k, \, \text{distance\_upper\_bound} = 35.0\text{ km})$$
4. If $\text{dist}_k \le 35.0\text{ km}$: The grid point receives the nearest valid observation value, and $\text{pixel\_mask}_k = 1.0$.
5. If $\text{dist}_k > 35.0\text{ km}$: The grid point lies outside the satellite swath. Its value is set to $0.0$, and its $\text{pixel\_mask}_k = 0.0$.

---

## 4. Critical 2 Resolution: Complete Geospatial Registration & Radar Beam Mapping

### 4.1 Root Cause of Critical 2
A significant gap previously separated the dataset coordinate loader from the container writer:
1. `StorageWriter` previously saved coordinates only for PMW swaths, omitting coordinates for Infrared, GPROF, Radar, and ERA5.
2. The ERA5 synoptic slice extracted 1D `latitude` ($121,$) and `longitude` ($121,$) axes, while `_resample_swath_with_coords` strictly expected 2D arrays of the same shape as the data, causing ERA5 to trigger `nominal_fallback`.
3. 3D volumetric radar profiles (Ku band) used naive trilinear interpolation across scan, beam, and range dimensions, stretching a $\sim 245\text{ km}$ wide GPM DPR swath across the entire $1,000\text{ km} \times 1,000\text{ km}$ domain and hallucinating radar echo in unobserved sectors.

### 4.2 Comprehensive Corrections Applied
1. **Container Coordinate Preservation (`tcprimed_pipeline/storage_writer.py`):**
   - **Infrared:** Saves `latitude`, `longitude` (`degrees_north`, `degrees_east`) and `x`, `y` (`km`) onto `("ir_y", "ir_x")`.
   - **GPROF:** Saves `latitude`, `longitude` and `x`, `y` onto `("gp_scan", "gp_pixel")`.
   - **Radar:** Saves beam geolocation `latitude`, `longitude`, `x`, `y` onto `("radar_scan", "radar_beam")`, and vertical range gate metadata `binHeight` onto `("radar_range", "radar_scan", "radar_beam")`.
   - **Environmental:** Saves 1D coordinate axes `latitude` on `("env_ny",)` and `longitude` on `("env_nx",)`.
   - **Modality Processor:** Expanded candidate radar group scanning to `["KuKaGMI", "KuGMI", "KuTMI"]` ensuring full TRMM DPR compatibility.

2. **1D Coordinate Axis Meshing (`src/data/dataset.py:_resample_swath_with_coords`):**
   - Detects 1D latitude and longitude axes matching 2D field dimensions $(n_y, n_x)$.
   - Constructs exact 2D geographic meshgrid using `np.meshgrid(lat_raw, lon_raw, indexing="ij")`.
   - Maps 2D coordinates to storm-centered tangent plane via equirectangular projection, yielding `source = "tangent_plane_geodetic_km"`.

3. **Volumetric Radar Beam Geospatial Registration (`src/data/dataset.py:_load_radar_volume`):**
   - Extracts 2D horizontal beam coordinates from `(x, y)` in km or geodetic `(latitude, longitude)` relative to vortex center.
   - Resamples vertical range gates to `target_depth = 16` levels using 1D/3D trilinear gate interpolation.
   - Evaluates a 2D $k$-d tree over valid horizontal beam positions against the target grid $[-500, +500]\text{ km}$ ($32 \times 32$ spatial cells).
   - Enforces `distance_upper_bound = max_dist_km` ($35.0\text{ km}$). Target cells outside the radar swath are strictly masked $0.0$ across all 16 depth layers, preventing spatial distortion.
   - Real pilot DPR test verified that $\sim 27.5\%$ of cells inside the actual swath are populated with real vertical echo, while $72.5\%$ outside the swath remain strictly unobserved.
   - Correctly marks `source = "radar_beam_geodetic_km"`.

4. **Provenance Tracking (`src/data/dataset.py`, `src/data/collate.py`):**
   - Returned sample dictionary includes `item["geospatial_provenance"]` tracking exact coordinate origin for each modality (`"tangent_plane_geodetic_km"`, `"utm_storm_centered_km"`, `"radar_beam_geodetic_km"`, or `"nominal_fallback"`).
   - Fallback is never disguised as verified alignment.

### 4.3 Verification Evidence
- Unit test `test_12_geospatial_coordinate_preservation_and_registration` validates:
  1. `StorageWriter` writes coordinates across all 4 modalities and verifies NetCDF structure.
  2. 1D ERA5 axes resample to $(1, 64, 64)$ with `source == "tangent_plane_geodetic_km"`.
  3. Radar beam coordinates resample to $(1, 16, 32, 32)$ with `source == "radar_beam_geodetic_km"`.
  4. Radar cells outside the swath ($|x| > 200\text{ km}$) are strictly zeroed and masked $0.0$.
  5. Missing coordinates fall back cleanly to `"nominal_fallback"`.
- All **143 unit and integration tests** in the repository pass cleanly (100% OK).
- Full pilot integration test (`scripts/run_pilot_integration_test.py`) executed end-to-end with verified finite gradients across all encoders.

### Verification Evidence
- Unit test `test_04_coordinate_aware_swath_resampling_and_distance_masking` validates that a $200\text{ km} \times 200\text{ km}$ swath placed at the center produces valid pixels ($1.0$) in the core, while grid corners ($r \approx 500\text{ km}$) are strictly masked invalid ($0.0$) with zero data leakage.
- Runtime benchmark confirms `cKDTree.query` across 4,096 points executes in **$1.33\text{ ms}$**, adding negligible computational overhead to data loading.

