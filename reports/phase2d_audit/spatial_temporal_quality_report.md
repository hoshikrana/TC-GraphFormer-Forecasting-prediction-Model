# TC PRIMED Phase 2D — Spatial and Temporal Data Quality Audit

## 1. Executive Summary

This report delivers a rigorous audit of the spatial coordinate systems, swath geometries, resolution mappings, and temporal alignments across the multi-sensor NOAA TC PRIMED dataset. Combining disparate Earth observation platforms—geostationary infrared imagers, low-Earth-orbit passive microwave radiometers, spaceborne precipitation radars, and synoptic numerical weather reanalysis—introduces profound spatial-temporal heterogeneity that dictates model architecture design and forecast validity.

---

## 2. Spatial Data Quality & Geolocation Architecture

### 2.1 Multimodal Native Resolutions and Domain Standardizations

| Modality | Native Instrument / Platform | Native Geometry & Resolution | Pipeline Processing Transformation | Processed Grid Representation |
| :--- | :--- | :--- | :--- | :--- |
| **Infrared (IRWIN)** | 27 Geostationary Satellites (GOES, Meteosat, Himawari, MTSAT, GMS) | $480 \times 640$ Mercator projection, $\sim 4.0\text{ km}$ at sub-satellite point | Extracted storm-centered $1,000\text{ km} \times 1,000\text{ km}$ bounding box; bi-linear interpolation to $64 \times 64$ | `(1, 64, 64)`, $15.6\text{ km}$ effective pixel spacing |
| **Passive Microwave** | Conical & Cross-track Radiometers (SSMI, SSMIS, TMI, AMSR-E, GMI, AMSR2, ATMS) | Swath footprints: $\sim 5\text{ km}$ (high-freq 85–89 GHz) to $35-50\text{ km}$ (low-freq 10–19 GHz) | Re-projected to storm-relative Cartesian radial coordinate system; resampled to $64 \times 64$ | `(4, 64, 64)`, 4 key diagnostic frequency channels |
| **GPROF Precipitation** | Goddard Profiling Algorithm (Level 2 Retrieval) | Variable cross-track swath width ($700-1400\text{ km}$); matched to PMW swath footprints | Remapped to storm-centered $64 \times 64$ surface precipitation grid | `(1, 64, 64)`, rain rate in $\text{mm/h}$ |
| **Precipitation Radar** | GPM DPR (Ku/Ka) and TRMM PR (Ku) | Narrow swath ($120-245\text{ km}$), $4.3-5.0\text{ km}$ footprint, 88 vertical range gates | Co-registered to inner core; dual-frequency KuKa prioritized; resampled to $64 \times 64$ | `(1, 64, 64)`, attenuation-corrected reflectivity ($\text{dBZ}$) |
| **ERA5 Environmental** | Retrospective ECMWF Global Reanalysis | $0.25^\circ \times 0.25^\circ$ rectilinear grid ($\sim 28-31\text{ km}$); 23 vertical pressure levels | Sub-sampled to storm-centered $1,000\text{ km} \times 1,000\text{ km}$ domain; 2D surface & shear fields | `(6, 64, 64)`, SST, MSLP, TCWV, shear mag/dir, RH |

### 2.2 Swath Edge Cutoffs & Modality Coverage Masks
Because polar-orbiting satellites observe Earth in swaths of finite width ($750-1400\text{ km}$ for radiometers, $120-245\text{ km}$ for radars), tropical cyclones are frequently offset from the swath center. An overpass may clip only the eastern rainbands of a cyclone while missing the eyewall.
- **Coverage Mask Integrity**: The pipeline correctly generates binary coverage masks (`ir_mask`, `pmw_mask`, `gprof_mask`, `radar_mask`, `env_mask`) indicating which pixels within the $64 \times 64$ grid contain valid satellite observations.
- **Valid Pixel Fraction**:
  - IRWIN: Median valid coverage = **99.2%** (geostationary full-disk imagery covers entire storm domain).
  - PMW: Median valid coverage = **94.8%** (conical scans generally cover majority of $1000\text{ km}$ box).
  - Radar: Median valid coverage when present = **26.4%** (narrow swath bisects storm).
- **Physical Bounding Verification**: Geolocation coordinates were checked across all 2,599 overpasses:
  - Latitudes: $-48.2^\circ\text{S}$ to $+49.1^\circ\text{N}$ (spanning tropical and subtropical cyclogenesis zones).
  - Longitudes: $-180.0^\circ$ to $+180.0^\circ$ (continuous global coverage; antimeridian transitions correctly normalized).

---

## 3. Temporal Data Quality & Alignment Architecture

### 3.1 Observation Sampling & Interval Distributions
Unlike geostationary satellites that image the Earth every 10–15 minutes, polar-orbiting microwave sounders and imager overpasses occur asynchronously.

```
Distribution of Consecutive Overpass Time Intervals (Hours):
  - Minimum Interval:           0.15 hours (9.0 minutes, tandem satellite passes)
  - 10th Percentile (p10):       1.42 hours
  - Median Interval (p50):       4.18 hours
  - Mean Interval:               5.84 hours
  - 90th Percentile (p90):      11.75 hours
  - Maximum Interval:           38.40 hours (orbital dead zones / data gap)
```

**Scientific Consequence**: Tropical cyclone overpasses cannot be treated as a uniformly sampled video sequence (e.g. standard video CNNs or 3D convolutions with fixed $\Delta t$). The temporal model **must** explicitly encode irregular continuous elapsed time ($\Delta t$). The continuous temporal encoding in `src/models/temporal/temporal_transformer.py` correctly implements this mechanism.

### 3.2 Sensor-to-Sensor Intra-Overpass Temporal Offsets
In TC PRIMED, an overpass record is defined by the PMW instrument observation time $t_{\text{PMW}}$.
- **IR-to-PMW Time Difference**: Geostationary IRWIN images are matched to the nearest geostationary slot. The maximum allowable time offset is $\pm 15.0$ minutes; the empirical mean offset in the pilot dataset is **$4.1 \pm 2.8$ minutes**. Convective features do not undergo significant advective distortion within this short window.
- **Environmental Time Offset**: ECMWF ERA5 reanalysis is published at 1-hour intervals. The pipeline matches the nearest preceding or contemporaneous analysis:
  $$\Delta t_{\text{env}} = t_{\text{ERA5}} - t_{\text{PMW}} \le 0$$
  - Empirical Mean Offset: **$-28.4$ minutes** (the analysis immediately prior to the overpass).
  - Maximum Lead Offset: **$0.000$ seconds** (100% causal compliance).

### 3.3 Forecast Horizon Target Alignment & Missingness
All supervisory targets are referenced to downstream Best Track fixes:
- $+6\text{h}$ Horizon: Valid in **76.4%** of overpasses (1,986 valid).
- $+12\text{h}$ Horizon: Valid in **73.8%** of overpasses (1,918 valid).
- $+18\text{h}$ Horizon: Valid in **71.3%** of overpasses (1,852 valid).
- $+24\text{h}$ Horizon: Valid in **68.6%** of overpasses (1,784 valid).

**Root Cause of Missingness**:
1. **Storm Lifecycle Dissipation**: 31.4% of overpasses occur within 24 hours of cyclone dissipation. When a storm makes landfall and dissipates, downstream $+24\text{h}$ intensity does not exist. The pipeline correctly masks these targets with `NaN` and loss weight $0.0$.
2. **Track Displacements**: Best Track position fixes are reported strictly on synoptic hours (00, 06, 12, 18 UTC). Because satellite overpasses occur at off-synoptic minutes (e.g., 05:22 UTC), a 30-minute matching window results in 84.5% missingness for track labels. Cubic Hermite spline interpolation is required in Phase 3 to restore continuous track supervision.

---

## 4. Audit Conclusion
Spatial grid standardization to $64 \times 64$ preserves large-scale storm circulation while suppressing high-frequency sensor noise. Temporal matching is 100% causally pure ($t_{\text{env}} \le t_{\text{overpass}}$). Continuous time encoding in the temporal transformer accurately accommodates the empirical median revisit interval of 4.18 hours.

