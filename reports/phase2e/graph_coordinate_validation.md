# TC PRIMED Graph Coordinate & Geometric Edge Validation Report
**Phase:** 2E — Targeted Core Data-to-Model Integration Repair  
**Module:** `src/models/graph/spatial_graph.py:SpatialPolarGraph`  
**Polar Node Count ($N$):** 16 nodes  
**Edge Attribute Dimensions ($d_{\text{edge}}$):** 8 geometric features  

---

## 1. Physical Polar Graph Representation

In tropical cyclone dynamics, atmospheric motion is predominantly axisymmetric (vortex tangential circulation) perturbed by asymmetric structural features (shear-induced convective asymmetries, eyewall replacement cycles, and spiral rainband propagation).

Standard Cartesian convolution on rectilinear grids fails to exploit this rotational geometry. In TC-GraphFormer, the multi-modal tokens are mapped onto a **polar coordinate spatial graph** centered on the minimum sea-level pressure vortex center.

### Polar Node Layout (16 Nodes)
The $N=16$ nodes partition the storm circulation into 4 concentric radial rings and 4 azimuthal quadrants:

| Radial Zone | Radial Domain ($r$) | Azimuthal Sectors ($\theta$) | Physical Atmospheric Domain |
| :--- | :---: | :---: | :--- |
| **Ring 0 (Eye & Eyewall)** | $0 - 75\text{ km}$ | $0^\circ, 90^\circ, 180^\circ, 270^\circ$ | Radius of Maximum Winds (RMW), primary convective eyewall |
| **Ring 1 (Inner Rainbands)** | $75 - 200\text{ km}$ | $45^\circ, 135^\circ, 225^\circ, 315^\circ$ | High-vorticity inner core, secondary eyewall formation zone |
| **Ring 2 (Outer Rainbands)** | $200 - 350\text{ km}$ | $0^\circ, 90^\circ, 180^\circ, 270^\circ$ | Spiraling rainbands, latent heat release, moisture inflow |
| **Ring 3 (Synoptic Outflow)** | $350 - 500\text{ km}$ | $45^\circ, 135^\circ, 225^\circ, 315^\circ$ | Anticyclonic outflow layer, environmental shear interaction |

---

## 2. Elimination of Fabricated Edge Attributes

### Defect Identified in Phase 2D Audit
In Phase 2B, `spatial_graph.py` defined 8 edge features, but 4 of the 8 dimensions were filled with hardcoded zeros:
```python
# Legacy Phase 2B fabricated features:
edge_attr = torch.stack([dr, dtheta, dist, torch.zeros_like(dr), ...], dim=-1)
```
This padded the edge attribute tensor with uninformative dummy zeros, diluting graph convolutional weights.

### Phase 2E 8-Dimensional Geometric Attribute System
In Phase 2E, the edge attributes were redesigned to contain **8 genuine physical geometric metrics**:
$$\mathbf{e}_{uv} = \left[ \Delta r, \, \Delta \theta, \, \sin(\Delta \theta), \, \cos(\Delta \theta), \, r_u, \, r_v, \, |\Delta r|, \, d_{uv} \right] \in \mathbb{R}^8$$

Where for directed edge $u \to v$:
1. $\Delta r = r_v - r_u$ (radial displacement, indicating radial inflow/outflow).
2. $\Delta \theta = (\theta_v - \theta_u + \pi) \pmod{2\pi} - \pi$ (azimuthal angular displacement).
3. $\sin(\Delta \theta)$ (continuous cyclic azimuthal offset, resolving cyclonic vs anticyclonic orientation).
4. $\cos(\Delta \theta)$ (angular alignment/collinearity between nodes).
5. $r_u$ (source node radial distance from eye).
6. $r_v$ (target node radial distance from eye).
7. $|\Delta r|$ (absolute radial distance, independent of direction).
8. $d_{uv} = \sqrt{r_u^2 + r_v^2 - 2 r_u r_v \cos(\Delta \theta)}$ (true physical Euclidean distance in km).

---

## 3. Physical Node Coordinates Buffer

To allow the graph module to ground dynamic attention in metric physical coordinates, `SpatialPolarGraph` registers a non-trainable coordinate buffer:
$$\mathbf{X}_{\text{polar}} = \left( r_i \cos\theta_i, \, r_i \sin\theta_i \right) \in \mathbb{R}^{16 \times 2} \quad (\text{km})$$
- Spanning $[-353.5\text{ km}, +353.5\text{ km}]$ along both zonal ($x$) and meridional ($y$) storm-relative axes.
- Dynamic adjacency graphs are constructed using radius thresholding ($d_{uv} \le 200.0\text{ km}$), creating dense connectivity within adjacent sectors and radial rings while avoiding spurious connections between distant outer rainbands.

### Verification Evidence
1. Unit test `test_09_graph_no_fabricated_edge_zeros` verifies that no constant or zero-padded dimensions exist across the edge attribute matrix.
2. Unit test `test_10_graph_node_coordinates_buffer` verifies that `node_coords_km` is registered as a non-trainable buffer and maintains exact geometric consistency with the $(r, \theta)$ node definitions.
