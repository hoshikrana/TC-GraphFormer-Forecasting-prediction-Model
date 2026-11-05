# TC PRIMED Pilot Pipeline — Unresolved Scientific Issues & Physical Nuances

This document formally records all identified physical nuances, empirical data characteristics,
and operational limitations in the NOAA TC PRIMED v01r01-final dataset that must be acknowledged
in downstream modeling rather than masked by unverified assumptions.

---

## 1. Environmental Reanalysis Temporal Causality & Operational Validity
* **Issue:** In the raw dataset, each overpass is accompanied by an environmental file containing ERA5 synoptic slices at 6-hourly intervals (00:00, 06:00, 12:00, 18:00 UTC). Naively matching each overpass to the temporally closest synoptic slice results in choosing a **future slice in ~48%–51% of observations** (up to 3.0 hours ahead of the satellite scan).
* **Scientific Resolution:** The pipeline enforces strict causal matching ($t_{\text{env}} \le t_{\text{overpass}}$). For each observation, only past or contemporaneous reanalysis fields are ingested.
* **Operational Limitation:** Retrospective ERA5 fields incorporate observations collected after the cyclone passed (full 4D-Var cycle). Using ERA5 as a model input proves diagnostic capability, but does **not** make the system an operational real-time forecasting tool. Operational real-time deployment requires substituting ERA5 with real-time Global Forecast System (GFS) or ECMWF operational analysis/forecast grids.

---

## 2. R5 Vortex Size vs. Radial Spatial Coordinates
* **Issue:** Early exploratory notes described R5 as a "five-bin radial profile".
* **Scientific Reality:** NetCDF-4 inspection confirms that `/infrared/R5` is a 1D time-series along the storm track dimension `R5_time`, measured in `degrees_latitude` ($1^\circ \text{ lat} \approx 111.14\text{ km}$), representing the radius where the 850-hPa azimuthally averaged tangential wind reaches 5 knots (Knaff et al. 2014).
* **Decoupling:** The spatial coordinate `/infrared/radius` is a separate 30-element coordinate (values 2 to 118 km, units `km`) used solely for `/infrared/brightness_temperature_mean_radial_profile`. The pipeline completely decouples these two variables.

---

## 3. Infrared Multi-Source Provenance & Resolution Heterogeneity
* **Issue:** Geostationary infrared observations originate from two disparate archives:
  1. **CIRA/STAR McIDAS (`flag == 1`):** Nominal 4.0 km resolution on a $480 \times 640$ Mercator grid, with narrow sensor-specific thermal channels (e.g. 10.70 µm for GOES Imagers, 11.20 µm for ABI, 10.40 µm for AHI).
  2. **HURSAT-B1 ISCCP (`flag == 2`):** Nominal 8.0 km resolution on a $301 \times 301$ Equal-Area grid, with broad historical thermal bands ($(10.7 - 12.1\text{ }\mu\text{m})$, $(10.3 - 12.6\text{ }\mu\text{m})$).
* **Scientific Implication:** Array pixel dimensions must not be mistaken for native optical resolution. Ingesting both sources directly into a single neural network without spatial resampling would cause spatial frequency bias. Wavelength descriptors and native resolution metadata are strictly preserved.

---

## 4. Future Track Coordinates Availability
* **Issue:** Overpass NetCDF files contain current center location (`storm_latitude`, `storm_longitude`), but **do not store future track positions** (+6h, +12h, +18h, +24h).
* **Scientific Resolution:** To predict future track or position, future coordinates must be retrieved from the sequential Best Track records in the environmental file's `storm_metadata` group. When a cyclone terminates or dissipates within the forecast window, future coordinates are marked as unavailable rather than synthetically extrapolated.

---

## 5. Missing Target Label Taxonomy
* **Issue:** In the pilot dataset, target missingness ranges from 0% for current $V_{\max}$ to 31.4% for 24-hour intensity change $\Delta V_{24\text{h}}$.
* **Physical Cause:** Missing future targets occur when the cyclone dissipates, transitions to extratropical, or reaches the end of operational agency advisories within the forecast window. Missing central pressure in older cases reflects lack of agency pressure estimates in pre-reanalysis eras (especially in North Indian Ocean and Southern Hemisphere). This is physically meaningful loss, not corrupted data.

---

## 6. Spatial Grid Representation & Mass Conservation Trade-Offs
* **Issue:** Resampling all sensor swaths to a common $256 \times 256$ Cartesian grid at 4 km has a total domain width of $1,024\text{ km}$, extending only $512\text{ km}$ from the storm center. It cannot cover the standard $800\text{ km}$ synoptic radius.
* **Pipeline Decision:** The pipeline **preserves native swath geometries** in the intermediate NetCDF-4/HDF5 format, avoiding premature irreversible resampling.
* **Remapping Mandate for Future Modeling:** When resampling is introduced, precipitation fluxes must use first-order area-weighted conservative remapping to conserve rain mass; bilinear interpolation is mathematically invalid for flux variables.

---

## 7. Storage Footprint for 500 GB Full Deployment
* **Finding:** While the 18.6 GB pilot compresses to approximately 1.8–2.5 GB of standardized NetCDF-4/HDF5 containers (compression ratio ~8–10x), deploying a full 500 GB raw archive alongside its processed derivatives will require at least **550–600 GB of total storage**, exceeding the capacity of a single 500 GB drive.
* **Recommendation:** Full-scale 2000–2025 deployment must utilize a dedicated external SSD (minimum 1 TB) or stream processed batches directly to training storage.
