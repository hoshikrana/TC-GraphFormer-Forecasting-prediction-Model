# TC PRIMED Geospatial Alignment & Cyclone-Relative Coordinate Report
**Phase:** 2E — Targeted Core Data-to-Model Integration Repair  
**Dataset:** NOAA TC PRIMED v01r01-final  
**Focus:** Storm-Centered Polar Coordinates, Tangent-Plane Geodesic Displacement & Swath Preservation  

---

## 1. Geospatial Coordinate System

Tropical cyclones are inherently rotating vortex structures whose physical processes (eyewall convection, spiral rainbands, outflow boundaries) are defined relative to the storm circulation center $(lat_0, lon_0)$.

In earlier pipeline phases:
1. Target positions for future track forecasting (+6h, +12h, +18h, +24h) were improperly measured as Euclidean degree offsets $(\Delta lat, \Delta lon)$. Because the length of a degree of longitude varies as $111.32 \times \cos(lat)\text{ km}$, a $1^\circ$ longitude displacement at $10^\circ\text{N}$ is $\approx 109.6\text{ km}$, whereas at $35^\circ\text{N}$ it is only $\approx 91.2\text{ km}$.
2. In addition, raw degree deltas across the antimeridian ($180^\circ$) produced catastrophic wrapping jumps (e.g. $+179^\circ \to -179^\circ$ computed as $-358^\circ$ rather than $+2^\circ$).

In Phase 2E, all track displacements are transformed into **local tangent-plane Cartesian coordinates** $(\Delta x, \Delta y)$ with metric units of kilometers ($\text{km}$).

---

## 2. Geodesic Tangent-Plane Formulation

Let the current cyclone center at observation time $t_0$ be:
$$(\phi_0, \lambda_0) = \left(\text{lat}_0 \cdot \frac{\pi}{180}, \text{lon}_0 \cdot \frac{\pi}{180}\right)$$

For a future storm position at horizon $t_0 + \Delta t$:
$$(\phi_t, \lambda_t) = \left(\text{lat}_t \cdot \frac{\pi}{180}, \text{lon}_t \cdot \frac{\pi}{180}\right)$$

### 2.1 Antimeridian Wrap-Around Resolution
The difference in longitude is bounded strictly to the interval $[-\pi, +\pi]$:
$$\Delta \lambda = \text{atan2}\left(\sin(\lambda_t - \lambda_0), \cos(\lambda_t - \lambda_0)\right)$$

### 2.2 Metric Displacement Equations
Using the mean Earth radius $R_{\text{earth}} = 6371.0\text{ km}$:
$$\Delta y_{\text{km}} = (\phi_t - \phi_0) \cdot R_{\text{earth}}$$
$$\bar{\phi} = \frac{\phi_0 + \phi_t}{2}$$
$$\Delta x_{\text{km}} = \Delta \lambda \cdot \cos(\bar{\phi}) \cdot R_{\text{earth}}$$

This projection is locally conformal and preserves metric distance within 0.05% error over the typical 24-hour hurricane displacement domain ($< 1000\text{ km}$).

### 2.3 Inverse Geodetic Reconstruction
To evaluate forecast positions against operational best-track data, predicted displacements $(\widehat{\Delta x}_{\text{km}}, \widehat{\Delta y}_{\text{km}})$ are mapped back to geodetic coordinates $(\widehat{\text{lat}}_t, \widehat{\text{lon}}_t)$:
$$\widehat{\text{lat}}_t = \text{lat}_0 + \left(\frac{\widehat{\Delta y}_{\text{km}}}{R_{\text{earth}}}\right) \cdot \frac{180}{\pi}$$
$$\phi_{\text{mid}} = \left(\text{lat}_0 + \frac{\widehat{\text{lat}}_t - \text{lat}_0}{2}\right) \cdot \frac{\pi}{180}$$
$$\widehat{\text{lon}}_t = \text{lon}_0 + \left(\frac{\widehat{\Delta x}_{\text{km}}}{R_{\text{earth}} \cdot \cos(\phi_{\text{mid}})}\right) \cdot \frac{180}{\pi}$$
$$\widehat{\text{lon}}_t = \left(\widehat{\text{lon}}_t + 180^\circ \pmod{360^\circ}\right) - 180^\circ$$

---

## 3. Spatial Resolution Discrepancy & Grid Alignment

The 5 observational modalities ingested by TC-GraphFormer exhibit substantial spatial resolution diversity:

| Modality | Native Sampling Resolution | Effective Footprint | Swath Width / Coverage | Grid Alignment Approach |
| :--- | :---: | :---: | :---: | :--- |
| **Infrared (IRWIN)** | $4\text{ km}$ | $4 \times 4\text{ km}$ | $1600 \times 1600\text{ km}$ full disk composite | Resampled to $64 \times 64$ ($12.5\text{ km/pixel}$) |
| **Passive Microwave (PMW)** | $10 - 50\text{ km}$ | Sensor dependent (elliptical) | Variable swath ($800 - 1700\text{ km}$) | Resampled to $64 \times 64$ with valid swath mask |
| **Precipitation Radar (Ku)** | $5\text{ km}$ horiz, $250\text{ m}$ vert | $5 \times 5\text{ km}$ pencil beam | Narrow swath ($245\text{ km}$) | Resampled to $16 \times 32 \times 32$ with 3D mask |
| **GPROF Retrievals** | $10 - 25\text{ km}$ | Matches PMW footprint | Matches PMW swath | Resampled to $64 \times 64$ with valid swath mask |
| **ERA5 Synoptic Reanalysis** | $0.25^\circ \approx 25\text{ km}$ | Regular latitude-longitude | Synoptic domain ($4000 \times 4000\text{ km}$) | Resampled to $64 \times 64$ storm-centered box |

### 3.1 Preserving Satellite Swaths via Binary Validity Masks
Because low-Earth-orbit satellites view the cyclone along inclined orbital swaths, an overpass often covers only a fraction of the $500\text{ km}$ storm radius. 

Rather than zero-padding (which would artificially suggest clear-sky or zero rain to the neural convolutions), Phase 2E computes explicit binary masks:
- `ir_pixel_mask`: `(1, 64, 64)`
- `pmw_pixel_mask`: `(4, 64, 64)`
- `radar_pixel_mask`: `(1, 16, 32, 32)`
- `gprof_pixel_mask`: `(4, 64, 64)`
- `env_pixel_mask`: `(6, 64, 64)`

These masks are passed through the modal encoders and downsampled alongside features, ensuring that subsequent cross-modal fusion and graph attention ignore unobserved spatial regions.
