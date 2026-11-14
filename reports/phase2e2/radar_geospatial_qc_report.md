# Phase 2E.2: Radar Geometry and 12 dBZ Precipitation Echo Quality Control

## 1. Executive Summary

This report documents the scientific correction of precipitation radar processing in **TC-GraphFormer** under `fix/phase2e2-physical-token-integration`.

Dual-frequency precipitation radar (DPR on GPM Core Observatory) provides unmatched 3D vertical profiling of tropical cyclone eyewalls and rainbands. However, Phase 2E contained two scientific discrepancies:
1. **Echo Classification vs. Measurement Validity:** The Phase 2E report claimed a $12\text{ dBZ}$ threshold, but the code merely filtered by instrument bounds ($-20 \le Z \le 100\text{ dBZ}$), blurring the distinction between measurement validity and precipitation detection.
2. **Geometric Disregard:** Raw radar arrays $(176, 49, 49)$ were resized using blind trilinear interpolation across array indices, disregarding the $125\text{ m}$ range gate spacing, scan angles, and earth curvature.

Phase 2E.2 disentangles measurement validity from precipitation echo classification, preserves radar vertical and horizontal georeferencing, and enforces calibrated physical quality control.

---

## 2. Measurement Validity vs. Precipitation Echo Classification

In radar meteorology, low reflectivity values ($-20\text{ dBZ} \le Z < 12\text{ dBZ}$) often represent valid measurements of cloud droplets, cirrus anvil crystals, or the instrument thermal noise floor. Setting a strict cutoff at $12\text{ dBZ}$ as a measurement mask causes valid non-precipitating cloudy regions to be marked as "unobserved" missing data.

Phase 2E.2 implements a clean two-tier quality control formulation:

### 2.1 Tier 1: Measurement Validity Mask
A radar gate is valid if the instrument received a physically valid signal within operational dynamic range:
$$\mathbf{M}_{\text{valid}}^{\text{radar}}(z, y, x) = \begin{cases} 1 & \text{if } -20.0 \le Z_{\text{Ku}} \le 100.0\text{ dBZ} \text{ and } Z_{\text{Ku}} \neq \text{NaN} \\ 0 & \text{otherwise} \end{cases}$$
Valid measurements between $-20$ and $0\text{ dBZ}$ are preserved, allowing the encoder to distinguish between cloud-free clear air and cloud-filled non-precipitating sectors.

### 2.2 Tier 2: Meteorological Precipitation Echo Classification
The standard GPM DPR precipitation detection threshold is $12\text{ dBZ}$ (corresponding to rain rates $\gtrsim 0.2\text{ mm/h}$):
$$\mathbf{E}_{\text{precip}}(z, y, x) = \begin{cases} 1 & \text{if } \mathbf{M}_{\text{valid}}^{\text{radar}} = 1 \text{ and } Z_{\text{Ku}} \ge 12.0\text{ dBZ} \\ 0 & \text{otherwise} \end{cases}$$
This flag indicates convective or stratiform precipitation cores and is preserved as an explicit quality-control attribute.

---

## 3. Physical 3D Radar Geometry Preservation

### 3.1 Vertical Coordinate Discretization
- GPM DPR operates with 176 range gates with a nominal gate length $\Delta h = 125\text{ m}$, spanning an altitude range:
  $$h(k) = k \cdot 0.125\text{ km},\quad k \in \{0, \dots, 175\} \implies h \in [0.0, 21.875]\text{ km}$$
- The 3D radar encoder expects 16 vertical levels:
  $$Z_{\text{levels}} = 16,\quad \Delta z = \frac{22.0\text{ km}}{16} = 1.375\text{ km}$$
  Vertical binning aggregates the 176 high-resolution range gates into 16 physical CAPPI (Constant Altitude Plan Position Indicator) layers:
  $$\text{CAPPI}(l) = [l \cdot 1.375, (l+1) \cdot 1.375]\text{ km}$$

### 3.2 Horizontal Georeferencing
- For each scan line $s \in \{0..48\}$ and beam $b \in \{0..48\}$, geodetic coordinates $(\phi_{s,b}, \lambda_{s,b})$ are converted to storm-centered tangent plane coordinates $(x, y)$ in kilometers.
- The radar swath is projected onto the model horizontal grid using KD-Tree nearest neighbor sampling with a tight search radius `max_dist_km = 15.0 km`.
- Only grid cells falling within the physical footprint of the DPR swath receive valid radar data; all outer cells are assigned validity 0, which are subsequently masked in `CrossModalFusion`.

---

## 4. Verification

In `tests/test_phase2e2_physical_token_integration.py` (`test_08_radar_geometry_and_12dbz_echo_classification`):
1. Synthetic radar test cubes were generated containing:
   - Noise floor: $-15\text{ dBZ}$ (valid measurement, non-precipitating)
   - Weak cloud: $5\text{ dBZ}$ (valid measurement, non-precipitating)
   - Heavy rain: $42\text{ dBZ}$ (valid measurement, precipitating echo)
   - Corrupt values: $-999\text{ dBZ}$ and $+150\text{ dBZ}$ (invalid measurement)
2. Verified:
   - Cells with $-15\text{ dBZ}$ and $+5\text{ dBZ}$ have `mask = 1` and `echo = 0`.
   - Cells with $+42\text{ dBZ}$ have `mask = 1` and `echo = 1`.
   - Cells with $-999$ or $+150$ have `mask = 0` and `echo = 0`.
3. In `scripts/run_pilot_integration_test.py`, radar observations in the 20 GB pilot were evaluated, confirming non-zero autograd gradients across all 3D convolutional kernels when processing the real GPM DPR overpass of storm AL022021.
