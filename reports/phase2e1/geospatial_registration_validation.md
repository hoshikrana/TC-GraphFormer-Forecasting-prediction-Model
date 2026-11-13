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

### 2.3 Nominal Fallback Regime
- **Found in:** Pilot containers where coordinate variables were not copied during initial packaging (e.g. `environmental_synoptic`).
- **Behavior:** Falls back safely to bilinear/nearest interpolation over the nominal $[-500, +500]\text{ km}$ domain, explicitly tagging metadata with `coord_source: "nominal_fallback"`.

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

### Verification Evidence
- Unit test `test_04_coordinate_aware_swath_resampling_and_distance_masking` validates that a $200\text{ km} \times 200\text{ km}$ swath placed at the center produces valid pixels ($1.0$) in the core, while grid corners ($r \approx 500\text{ km}$) are strictly masked invalid ($0.0$) with zero data leakage.
- Runtime benchmark confirms `cKDTree.query` across 4,096 points executes in **$1.33\text{ ms}$**, adding negligible computational overhead to data loading.
