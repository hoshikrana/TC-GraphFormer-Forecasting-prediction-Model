# Phase 2E.2: All-Modality Coordinate-Aware Registration and Resampling Validation

## 1. Executive Summary

This report documents the validation of coordinate-aware geospatial registration across all observation and environmental modalities in **TC-GraphFormer**.

In satellite meteorology, storm observations originate from distinct orbital geometries:
- Geostationary and polar imagers (IR)
- Conical scanning radiometers (PMW: SSMIS, AMSR2, GMI)
- Cross-track sounders (PMW: ATMS, MHS, AMSUB)
- Active dual-frequency radar (GPM DPR)
- Numerical reanalysis (ERA5)

Phase 2E.2 unifies all observations onto a standardized storm-centered Cartesian reference grid $[-500\text{ km}, +500\text{ km}]^2$ using an exact geodetic tangent plane projection, robust antimeridian wrapping, instrument-specific search radii, and strict coordinate enforcement.

---

## 2. Geodetic Tangent Plane Projection

For a storm center fix at geodetic latitude $\phi_0$ and longitude $\lambda_0$ (degrees), any arbitrary observation point $(\phi, \lambda)$ is projected onto storm-relative Cartesian kilometers $(x, y)$ via the spherical tangent plane approximation ($R_{\text{earth}} = 6371.0\text{ km}$):

$$\Delta \lambda = ((\lambda - \lambda_0 + 180^\circ) \bmod 360^\circ) - 180^\circ$$
$$\bar{\phi} = \frac{\phi + \phi_0}{2} \cdot \frac{\pi}{180}$$
$$x = R_{\text{earth}} \cdot \left(\Delta \lambda \cdot \frac{\pi}{180}\right) \cdot \cos(\bar{\phi})$$
$$y = R_{\text{earth}} \cdot \left((\phi - \phi_0) \cdot \frac{\pi}{180}\right)$$

### 2.1 Antimeridian Discontinuity Handling
Tropical cyclones frequently traverse the International Date Line (e.g. Western/Central Pacific typhoons). Unhandled modular longitude representations create severe projection errors:
- If $\lambda_0 = 179.5^\circ\text{E}$ and an observation point lies at $\lambda = -179.5^\circ\text{W}$, naive subtraction yields $\Delta \lambda = -359^\circ \implies x \approx -39,900\text{ km}$.
- Under the Phase 2E.2 modular wrapping formula:
  $$\Delta \lambda = ((-179.5 - 179.5 + 180) \bmod 360) - 180 = (1 \bmod 360) - 180 = +1.0^\circ$$
  yielding physical storm-relative separation $x \approx +111.2\text{ km} \cdot \cos(\bar{\phi})$.

### 2.2 1D vs 2D Coordinate Handling
For gridded reanalysis (ERA5), coordinates are stored as 1D arrays `latitude(N_lat)` and `longitude(N_lon)`. `_resample_swath_with_coords` automatically detects 1D axes and generates 2D coordinate meshgrids:
```python
if lats.ndim == 1 and lons.ndim == 1:
    lons_2d, lats_2d = np.meshgrid(lons, lats)
```
preventing dimensionality mismatches during KD-Tree indexing.

---

## 3. Instrument-Specific Resampling Distance Thresholds

Because satellite sensors possess varying spatial footprint resolutions, applying a uniform resampling search radius either induces spatial over-smoothing or produces spurious gaps. Phase 2E.2 establishes calibrated maximum distance thresholds (`max_dist_km`):

| Modality | Instrument Footprint Resolution | Maximum Search Radius `max_dist_km` | Physical Justification |
|---|---|:---:|---|
| **IR** | $\approx 4\text{ km}$ (GridSat / ABI) | **15.0 km** | Prevents pixel replication beyond 1 grid cell width on the $15.625\text{ km}$ model grid. |
| **PMW** | $\approx 25\text{--}50\text{ km}$ (SSMIS, AMSR2) | **35.0 km** | Accommodates low-frequency microwave beam convolution while rejecting out-of-swath cells. |
| **GPROF** | $\approx 10\text{--}20\text{ km}$ retrieval footprint | **20.0 km** | Balances footprint smoothing with preservation of convective core gradients. |
| **Radar** | $\approx 5\text{ km}$ (GPM DPR footprint) | **15.0 km** | Enforces tight spatial fidelity; rejects radar echoes beyond 1 cell radius. |
| **ERA5** | $0.25^\circ \approx 28\text{ km}$ regular grid | **40.0 km** | Guarantees continuous synoptic field coverage across the entire $1,000\text{ km}$ domain. |

---

## 4. Strict Scientific Mode vs. Fallback Mode

To ensure rigorous data integrity in production experiments, `TCPRIMEDDataset` supports `strict_mode: bool`:

1. **`strict_mode = False` (Lenient/Exploratory):**
   - If coordinate arrays are missing or unreadable, logs a warning and falls back to index-based volume resizing (`F.interpolate`).
   - Used for quick debugging or when exploring legacy non-georeferenced data.
2. **`strict_mode = True` (Strict Scientific Mode):**
   - If coordinates are absent or invalid, raises an explicit `ValueError`:
     `ValueError("[STRICT MODE] Modality 'pmw' lacks valid latitude/longitude coordinates...")`
   - Halts execution immediately, guaranteeing that uncalibrated or un-registered data can never silently enter training or evaluation pipelines.

---

## 5. Verification Results

In `tests/test_phase2e2_physical_token_integration.py`:
- `test_06_1d_era5_coordinates_and_antimeridian`: Verified that 1D ERA5 axes project correctly across the antimeridian without boundary wrapping artifacts.
- `test_07_modality_resampling_distance_thresholds`: Verified that each modality's resampling call in `__getitem__` invokes `_resample_swath_with_coords` with the exact calibrated threshold (IR 15 km, PMW 35 km, GPROF 20 km, Radar 15 km, ERA5 40 km).
- `test_11_strict_scientific_mode_missing_coords`: Verified that omitting coordinates with `strict_mode=True` immediately raises `ValueError`, stopping corrupted inputs.
