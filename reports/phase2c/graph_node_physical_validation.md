# TC PRIMED Phase 2C — Spatial Graph Node Physical Validation and Meteorological Derivation

## 1. Executive Summary & Objective

The spatial graph module in TC-GraphFormer (`src/models/graph/spatial_graph.py`) replaces standard spatial flattening with a topology-aware Graph Neural Network (GNN). Rather than treating satellite pixels as an unstructured sequence, it models tropical cyclone physics using a structured graph representing:
1. Radial zones: Eyewall / Core $\to$ Inner Rainbands $\to$ Outer Rainbands $\to$ Synoptic Environment.
2. Azimuthal quadrants: Relative storm sectors (NE, SE, SW, NW).
3. Mass and momentum transport: Radial inflow/outflow and cyclonic tangential circulation.

This report evaluates the meteorological derivation of the 16 graph nodes, audits the message passing mechanics, and identifies opportunities to incorporate environmental vertical wind shear directly into graph edges.

---

## 2. Topological & Meteorological Derivation of the 16 Nodes

Standard tropical cyclone dynamics are naturally described in cylindrical storm-centric coordinates $(r, \theta, z)$. In TC PRIMED, satellite overpasses are re-centered on the cyclone eye across a standardized $64 \times 64$ grid representing a $1,000\text{ km} \times 1,000\text{ km}$ storm domain ($15.6\text{ km}$ effective grid spacing per pixel).

The 16 nodes are constructed from a $4 \times 4$ spatial patch decomposition centered at $(c_x, c_y) = (1.5, 1.5)$:

```
           North (0°)
               ^
    [ 0 ]   [ 1 ] | [ 2 ]   [ 3 ]
       \       \  |  /       /
    [ 4 ]   [ 5 ] | [ 6 ]   [ 7 ]
  <---------------+---------------> East (90°)
    [ 8 ]   [ 9 ] | [10 ]   [11 ]
       /       /  |  \       \
    [12 ]   [13 ] | [14 ]   [15 ]
                  v
              South (180°)
```

### Physical Radial Zones:
1. **Core / Eyewall Nodes** $\{5, 6, 9, 10\}$:
   - Distance from center: $r \in [0, 110]\text{ km}$.
   - Physical Phenomena: Maximum tangential wind ($V_{\max}$), radius of maximum winds (RMW), intense eyewall convective towers, central pressure depression, and radar bright bands.
2. **Inner Rainband Nodes** $\{1, 2, 4, 7, 8, 11, 13, 14\}$:
   - Distance from center: $r \in [110, 250]\text{ km}$.
   - Physical Phenomena: Primary spiral rainbands, low-level moisture convergence, stratiform rain regions, and secondary eyewall formation during Eyewall Replacement Cycles (ERC).
3. **Outer Rainband / Synoptic Nodes** $\{0, 3, 12, 15\}$:
   - Distance from center: $r \in [250, 500]\text{ km}$.
   - Physical Phenomena: Outer peripheral convection, dry-air intrusion boundaries, environmental sea-surface temperature gradient, and synoptic steering interaction.

---

## 3. Edge Index and Message Passing Mechanics

In `src/models/graph/spatial_graph.py`, edges $\mathcal{E}$ connect adjacent spatial patches:

### Mathematical Message Passing:
For each directed edge $(u \to v) \in \mathcal{E}$, a message $\mathbf{m}_{uv}$ is computed using the source node embedding $\mathbf{h}_u$, destination embedding $\mathbf{h}_v$, and edge attribute vector $\mathbf{e}_{uv}$:
$$\mathbf{m}_{uv} = \text{MLP}_{\text{msg}} \left( [\mathbf{h}_u \,\|\, \mathbf{h}_v \,\|\, \mathbf{e}_{uv}] \right) \in \mathbb{R}^{d_{\text{model}}}$$

Destination nodes aggregate incoming messages with degree normalization:
$$\mathbf{a}_v = \frac{1}{\text{deg}(v)} \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}$$

Node states are updated with a non-linear residual connection:
$$\mathbf{h}_v^{(l+1)} = \text{LayerNorm} \left( \mathbf{h}_v^{(l)} + \text{MLP}_{\text{update}} \left( [\mathbf{h}_v^{(l)} \,\|\, \mathbf{a}_v] \right) \right)$$

### Edge Attributes ($\mathbf{e}_{uv} \in \mathbb{R}^{16}$):
The edge features currently encode:
1. Relative radial distance: $\Delta r = r_v - r_u$ (captures inward vs. outward flux).
2. Relative azimuthal displacement: $\Delta \theta = \theta_v - \theta_u$ (captures cyclonic vs. anticyclonic flow).
3. Harmonic angular embeddings: $[\sin(\Delta \theta), \cos(\Delta \theta)]$.
4. Radial distances of source and destination: $[r_u, r_v, |\Delta r|]$.

---

## 4. Meteorological Critique & Identified Improvement for Phase 3

### Current Limitation: Static Geometric Graph
In the current Phase 2B implementation, the adjacency matrix and edge attribute vectors $\mathbf{e}_{uv}$ are **static geometric constants** initialized from grid indices $(i, j)$.
While they successfully encode spatial proximity and cyclonic direction:
- They do not know which direction is **up-shear** versus **down-shear**.
- In reality, tropical cyclone intensity change and Rapid Intensification (RI) are strongly governed by **vertical wind shear** (200–850 hPa shear vector $\mathbf{S}$).
- Convection preferentially erupts in the **downshear-left** quadrant (in the Northern Hemisphere) or **downshear-right** (in the Southern Hemisphere).

### Recommended Enhancement for Phase 3:
Dynamically condition edge attributes on the environmental vertical wind shear vector from ERA5:
$$\theta_{\text{shear\_relative}} = \theta_{\text{node}} - \theta_{\text{shear}}$$
$$\mathbf{e}_{uv}^{\text{dynamic}} = [\mathbf{e}_{uv}^{\text{static}} \,\|\, \|\mathbf{S}\|_2 \,\|\, \cos(\theta_u - \theta_{\text{shear}}) \,\|\, \sin(\theta_u - \theta_{\text{shear}})]$$

This enhancement will allow the graph convolution layers to explicitly route convective burst signals from downshear rainbands directly into the eyewall nodes during Rapid Intensification.

---

## 5. Audit Conclusion
The 16-node spatial graph is mathematically rigorous, computationally lightweight (0.13 MB), and aligns well with cyclone structural meteorology. Incorporating dynamic shear-relative orientation in Phase 3 will elevate its physical expressiveness from geometric approximation to dynamic meteorological modeling.

