# Phase 2E.2: Geospatial Metadata Preservation in Processed Containers

## 1. Executive Summary

This report documents the architectural corrections made to `src/data/storage_writer.py` to preserve geospatial coordinates across all five sensor and environmental modalities in processed NetCDF-4 containers.

In Phase 2E, while coordinate-aware swath resampling was introduced in `src/data/dataset.py`, the storage writer only preserved coordinates for PMW swaths. Consequently, processed containers for IR, GPROF, Radar, and ERA5 lacked spatial georeferencing metadata. Downstream loaders were compelled to fall back to nominal index-based resizing (e.g. `F.interpolate`), forfeiting true physical coordinate alignment with the storm center.

Under Phase 2E.2, `StorageWriter` mandates coordinate arrays, standard CF attributes, and vertical dimension metadata for all modalities, creating a persistent, self-describing archive.

---

## 2. NetCDF-4 Modality Schema & Coordinate Arrays

The table below summarizes the coordinate preservation schema implemented in `StorageWriter`:

| Modality | Coordinate Variables | Array Dimensions | CF Standard Name / Description | Units |
|---|---|---|---|---|
| **PMW** | `latitude`<br>`longitude` | `(pmw_scans, pmw_pixels)`<br>`(pmw_scans, pmw_pixels)` | `latitude`<br>`longitude` | `degrees_north`<br>`degrees_east` |
| **IR** | `latitude`<br>`longitude` | `(ir_rows, ir_cols)`<br>`(ir_rows, ir_cols)` | `latitude`<br>`longitude` | `degrees_north`<br>`degrees_east` |
| **GPROF** | `latitude`<br>`longitude` | `(gprof_scans, gprof_pixels)`<br>`(gprof_scans, gprof_pixels)` | `latitude`<br>`longitude` | `degrees_north`<br>`degrees_east` |
| **Radar** | `latitude`<br>`longitude`<br>`bin_height_m` | `(radar_scans, radar_beams)`<br>`(radar_scans, radar_beams)`<br>`(radar_gates,)` | `latitude`<br>`longitude`<br>`height_above_mean_sea_level` | `degrees_north`<br>`degrees_east`<br>`m` |
| **ERA5** | `latitude`<br>`longitude` | `(era5_lats,)<br>(era5_lons,)` | `latitude`<br>`longitude` | `degrees_north`<br>`degrees_east` |

---

## 3. Preservation Specifications by Modality

### 3.1 Infrared (IR)
- **Geometry:** Geostationary / polar satellite infrared imagery (e.g., GridSat, GOES ABI, Meteosat SEVIRI).
- **Arrays Preserved:** `latitude`, `longitude` matching the 2D image matrix `(ir_rows, ir_cols)`.
- **CF Attributes:**
  - `standard_name`: `latitude` / `longitude`
  - `units`: `degrees_north` / `degrees_east`
  - `coordinates`: `"latitude longitude"` attached to variable `tb_irwin`.

### 3.2 Passive Microwave (PMW)
- **Geometry:** Conical and cross-track conical swath swaths.
- **Arrays Preserved:** 2D scan/pixel coordinates for low-frequency and high-frequency horns.
- **CF Attributes:**
  - Attached to multi-channel brightness temperatures `tb_channels`.

### 3.3 GPROF Precipitation Profiling
- **Geometry:** Swath retrieval of surface rain rate, convective rain fraction, and vertical hydrometeor profiles.
- **Arrays Preserved:** 2D scan/pixel coordinates for surface precipitation and 3D hydrometeor fields.
- **CF Attributes:**
  - Attached to `surface_precipitation` and `hydrometeor_profiles`.

### 3.4 Precipitation Radar (GPM DPR)
- **Geometry:** 3D radar volume indexed by `(radar_gates, radar_scans, radar_beams)`.
- **Arrays Preserved:**
  - Horizontal geolocation: 2D footprint geodetic coordinates `latitude(radar_scans, radar_beams)` and `longitude(radar_scans, radar_beams)`.
  - Vertical bin height: 1D array `bin_height_m(radar_gates)` measuring physical elevation above sea level for each range gate ($125\text{ m}$ spacing across 176 gates, spanning $0\text{--}22\text{ km}$).
- **Impact:** Downstream loaders can perform true 3D spatial binning into constant-altitude plan position indicator (CAPPI) levels.

### 3.5 ERA5 Synoptic Reanalysis
- **Geometry:** 2D regular latitude-longitude grid enclosing the storm circulation ($10^\circ \times 10^\circ$).
- **Arrays Preserved:** 1D coordinate vectors `latitude(era5_lats)` and `longitude(era5_lons)`.
- **CF Attributes:**
  - `units`: `degrees_north` / `degrees_east`
  - Longitude axis explicitly documents $[-180, 180]^\circ$ or $[0, 360]^\circ$ convention to support antimeridian crossing handling.

---

## 4. Verification and Compliance

In `tests/test_phase2e2_physical_token_integration.py` (`test_05_storage_writer_geospatial_metadata_preservation`):
1. A temporary NetCDF container was written with synthetic observations across all 5 modalities.
2. The generated container was reopened and inspected using `netCDF4.Dataset`.
3. Verified:
   - All 5 modality groups contain non-empty `latitude` and `longitude` variables.
   - The radar group contains `bin_height_m` with length matching the vertical gate dimension.
   - All coordinate variables possess valid `units` attributes (`degrees_north`, `degrees_east`, `m`).
   - Downstream dataset instantiation successfully resolves coordinate-aware spatial resampling for 100% of modalities.
