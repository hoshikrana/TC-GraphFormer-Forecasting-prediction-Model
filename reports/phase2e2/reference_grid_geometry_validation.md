# Phase 2E.2: Reference Grid Geometry and Cell-Centering Validation

## 1. Executive Summary

This report documents the mathematical correction of the storm-centered Cartesian reference grid geometry in **TC-GraphFormer** under `fix/phase2e2-physical-token-integration`.

Prior to Phase 2E.2, grid coordinates were generated using `np.linspace(-500.0, 500.0, 64)`. In `np.linspace`, endpoints are included, which produced an irregular cell spacing of $\approx 15.873\text{ km}$ rather than the documented $15.625\text{ km}$, and placed the outermost cell centers directly on the $500\text{ km}$ domain boundary.

Under Phase 2E.2, reference grid coordinates are defined strictly at cell centers, yielding an exact cell width $\Delta x = \Delta y = 15.625\text{ km}$ and cell center bounds at $\pm 492.1875\text{ km}$, with cell boundaries spanning exactly $[-500.0\text{ km}, +500.0\text{ km}]$.

---

## 2. Mathematical Discrepancy in `np.linspace`

When generating $N$ grid points across an interval $[A, B]$ using endpoint-inclusive interpolation:
$$\Delta_{\text{linspace}} = \frac{B - A}{N - 1}$$
For $A = -500\text{ km}$, $B = +500\text{ km}$, and $N = 64$:
$$\Delta_{\text{linspace}} = \frac{1000.0}{63} \approx 15.87301587\text{ km}$$

### 2.1 Geometric Consequences
1. **Grid Spacing Error:** The cell spacing was $15.873\text{ km}$, diverging by $+1.59\%$ from the $15.625\text{ km}$ model and dataset specification.
2. **Domain Spillover:** If the points at $\pm 500.0\text{ km}$ represent pixel centers, each pixel extends half a cell width ($\approx 7.937\text{ km}$) outward:
   $$\text{Domain Bounds} = [-500.0 - 7.937, +500.0 + 7.937] = [-507.937, +507.937]\text{ km}$$
   The grid encompassed an area of $(1015.87\text{ km})^2$, overflowing the standard NOAA TC PRIMED $1,000\text{ km}$ domain.

---

## 3. Cell-Centered Formulation

In a discrete raster grid of $N = 64$ cells spanning $[-L, +L]$ where $L = 500.0\text{ km}$:

$$\Delta x = \frac{2L}{N} = \frac{1000.0\text{ km}}{64} = 15.625000\text{ km}$$

The physical coordinate of cell center $k \in \{0, \dots, N-1\}$ is:
$$x_c(k) = -L + \left(k + \frac{1}{2}\right) \Delta x$$

### 3.1 Coordinate Extremes
- **First Cell Center ($k = 0$):**
  $$x_c(0) = -500.0 + 0.5 \times 15.625 = -492.1875\text{ km}$$
- **Last Cell Center ($k = 63$):**
  $$x_c(63) = -500.0 + 63.5 \times 15.625 = +492.1875\text{ km}$$
- **Domain Coverage:**
  $$[x_c(0) - 0.5\Delta x, x_c(63) + 0.5\Delta x] = [-500.0000, +500.0000]\text{ km}$$

The domain coverage is strictly and exactly $[ -500\text{ km}, +500\text{ km} ]$, with zero spillover.

---

## 4. Radial Distance Alignment with Polar Sectors

The storm-relative radius from storm center $(0, 0)$ to grid cell $(i, j)$ is:
$$r_{i,j} = \sqrt{x_c(j)^2 + y_c(i)^2}$$

| Cell Coordinate Index $(i, j)$ | Physical Location $(x_c, y_c)$ | Radius $r$ | Assigned Polar Ring | Ring Range |
|---|---|---|:---:|---|
| $(31, 31)$ | $(-7.8125, +7.8125)\text{ km}$ | $11.05\text{ km}$ | **Ring 0** (Inner Core) | $0 \le r < 75\text{ km}$ |
| $(32, 32)$ | $(+7.8125, -7.8125)\text{ km}$ | $11.05\text{ km}$ | **Ring 0** (Inner Core) | $0 \le r < 75\text{ km}$ |
| $(27, 27)$ | $(-70.3125, +70.3125)\text{ km}$ | $99.44\text{ km}$ | **Ring 1** (Inner Vortex) | $75 \le r < 180\text{ km}$ |
| $(20, 20)$ | $(-179.6875, +179.6875)\text{ km}$ | $254.12\text{ km}$ | **Ring 2** (Outer Vortex) | $180 \le r < 320\text{ km}$ |
| $(10, 10)$ | $(-335.9375, +335.9375)\text{ km}$ | $475.09\text{ km}$ | **Ring 3** (Synoptic Env) | $320 \le r \le 500\text{ km}$ |
| $(0, 0)$ (Corner) | $(-492.1875, +492.1875)\text{ km}$ | $696.05\text{ km}$ | Outside Disk | $r > 500\text{ km}$ (Zeroed) |

Because the corners of a $1000 \times 1000\text{ km}$ square extend to $r \approx 696\text{ km}$, `PolarSectorPooling` correctly masks grid cells with $r > 500\text{ km}$, ensuring that only physical data within the $500\text{ km}$ TC disk inform the polar graph nodes.

---

## 5. Verification

In `tests/test_phase2e2_physical_token_integration.py` (`test_09_reference_grid_cell_centering`):
1. `_create_reference_grid(extent_km=500.0, grid_size=64)` was evaluated.
2. Verified:
   - Difference between adjacent cell centers is exactly $15.625\text{ km}$ ($\text{max error} < 10^{-12}\text{ km}$).
   - `grid_x.min() == -492.1875` and `grid_x.max() == +492.1875`.
   - `grid_y.min() == -492.1875` and `grid_y.max() == +492.1875`.
   - Grid cell boundaries span $[-500.0, +500.0]\text{ km}$.
