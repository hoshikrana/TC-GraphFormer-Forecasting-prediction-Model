# TC-GraphFormer Model Architecture Report
**Multimodal Spatiotemporal Graph-Transformer for Tropical Cyclone Forecasting**  
**Dataset:** NOAA TC PRIMED v01r01-final  
**Date:** November 2026

---

## 1. Executive Summary & Design Philosophy

Tropical cyclone (TC) monitoring and prediction from multi-satellite sensor suites presents unique scientific challenges:
1. **Multimodal Asynchrony & Missingness:** Infrared (geostationary ~hourly, 4km/8km) and Passive Microwave (polar-orbiting LEO, 3–12h gaps, 9 disparate sensors) observe cyclones at different frequencies and spatial coverages. Radar (Ku/Ka) is present in only ~3.7% of overpasses.
2. **Asymmetric Vortex Physics:** Cyclonic vortices are governed by asymmetric convective bursts, eyewall replacement cycles (ERC), environmental vertical wind shear tilt, and dry air intrusion. Flat vector concatenation or isotropic CNNs fail to capture directional vorticity and radial momentum transport.
3. **Irregular Continuous-Time Dynamics:** Revisit gaps are non-uniform ($\Delta t \in [1, 24]$ hours). Standard recurrent or discrete-time models distort temporal dynamics by assuming uniform steps.

To solve these challenges without heuristic hand-waving or destructive downscaling, we designed and implemented **TC-GraphFormer**: a unified, modular spatiotemporal graph-transformer architecture.

```
+-------------------------------------------------------------------------------------------------------+
|                                        TC-GraphFormer Architecture                                     |
+-------------------------------------------------------------------------------------------------------+
|  [IRWIN 2D]      [PMW 4-ch]      [GPROF Rain]      [Radar 3D Vol]      [ERA5 Precipitable Water]      |
|       |               |               |                  |                        |                   |
|  [IREncoder]    [PMWEncoder]    [GPROFEncoder]     [RadarEncoder]           [EnvEncoder]              |
|       |               |               |                  |                        |                   |
|   16 tokens       16 tokens       16 tokens          16 tokens                16 tokens               |
|   (B, 16, D)      (B, 16, D)      (B, 16, D)         (B, 16, D)               (B, 16, D)              |
+-------+---------------+---------------+------------------+------------------------+-------------------+
        |               |               |                  |                        |
        +---------------+---------------+------------------+------------------------+
                                        |
                        [Mask-Aware Cross-Modal Fusion]
               (80 Tokens with Modality & 2D Spatial Pos Embeddings)
                 (Self-Attention with -inf Missing Modality Masking)
                                        |
                            (B, 16, 5*D) Spatial Reshaping
                                        |
                       [Spatial Projection & Aggregation]
                                        |
                        [Sparse Spatial Graph MPNN Encoder]
               (16 Cyclone Nodes: Core, Eyewall, Rainbands, Quadrants)
               (Directed Edges: Cyclonic Vorticity & Radial Shear Bias)
                                        |
                           (B, D) Graph State Vector
                                        |
                     [Causal Continuous-Time Temporal Transformer]
                       (Fourier Time-Delta Encoding Delta t = t - t_0)
                      (Strict Causal Mask: t_j <= t_i, Zero Future Leak)
                                        |
                           (B, D) Shared Latent Vector
                                        |
    +-------------------+---------------+-------------------+-------------------+
    |                   |               |                   |                   |
[Intensity Head] [Pressure Head] [Stage Head]      [Delta v Head]         [RI Head]
  (v_max kt)       (p_min hPa)     (6 classes)   (+6h,+12h,+18h,+24h)  (>=30kt / 24h)
```

---

## 2. Modality Encoders (`src/models/encoders/`)

Each satellite modality is processed through dedicated domain encoders preserving physical properties:

### 2.1 Infrared Encoder (`ir_encoder.py`)
- **Input:** $x_{\text{IR}} \in \mathbb{R}^{B \times 1 \times H \times W}$ (Brightness temperature, normalized $[0, 1]$ from 200–300 K).
- **Structure:** 4-stage residual convolutional hierarchy with GELU activations and BatchNorm.
- **Output:** 16 spatial tokens $\in \mathbb{R}^{B \times 16 \times d_{\text{model}}}$.
- **Zero-Mask Enforcement:** If $m_{\text{IR}} = 0$, all output tokens are strictly multiplied by 0.

### 2.2 Passive Microwave Encoder (`pmw_encoder.py`)
- **Input:** $x_{\text{PMW}} \in \mathbb{R}^{B \times 4 \times H \times W}$ (Low-frequency S1 channels ~19/22/37 GHz and high-frequency S2 channels ~85/91 GHz).
- **Structure:** Deep convolutional residual stem capturing hydrometeor scattering and warm core thermal emission.
- **Output:** 16 spatial tokens $\in \mathbb{R}^{B \times 16 \times d_{\text{model}}}$.

### 2.3 GPROF Precipitation Encoder (`gprof_encoder.py`)
- **Input:** $x_{\text{GPROF}} \in \mathbb{R}^{B \times 1 \times H \times W}$ (Surface precipitation rate, $\log(1 + p) / 5.0$).
- **Structure:** 3-stage convolutional residual feature extractor.
- **Output:** 16 spatial tokens $\in \mathbb{R}^{B \times 16 \times d_{\text{model}}}$.

### 2.4 Precipitation Radar Encoder (`radar_encoder.py`)
- **Input:** $x_{\text{Radar}} \in \mathbb{R}^{B \times 1 \times D \times H/2 \times W/2}$ (Volumetric Ku/Ka reflectivities across 16 vertical bins).
- **Structure:** 3D Convolutional Network (`Conv3d`, `BatchNorm3d`, `AdaptiveAvgPool3d`) collapsing vertical cross-sections into horizontal spatial tokens.
- **Sparsity Management:** Because radar is present in only ~3.7% of overpasses, inputs are populated with zeros when absent, and $m_{\text{Radar}} = 0$ zeroes out the representation before fusion.

### 2.5 Environmental Synoptic Encoder (`env_encoder.py`)
- **Input:** $x_{\text{Env}} \in \mathbb{R}^{B \times 1 \times H \times W}$ (ERA5 precipitable water and large-scale atmospheric fields).
- **Structure:** Wide receptive field 2D CNN aligned with cyclone center coordinates.
- **Output:** 16 spatial tokens $\in \mathbb{R}^{B \times 16 \times d_{\text{model}}}$.

---

## 3. Mask-Aware Cross-Modal Fusion (`src/models/fusion/cross_modal_fusion.py`)

- **Total Sequence Length:** $5 \times 16 = 80$ tokens.
- **Embeddings:**
  $$\mathbf{z}_{m, p} = \mathbf{t}_{m, p} + \mathbf{e}_{\text{modality}}(m) + \mathbf{e}_{\text{spatial}}(p)$$
  where $m \in \{0, \dots, 4\}$ indexes modality and $p \in \{0, \dots, 15\}$ indexes 2D grid patch.
- **Masking Mechanism:**
  A boolean key-padding mask $\mathbf{M} \in \{0, 1\}^{B \times 80}$ sets all tokens corresponding to absent modalities ($m_k = 0$) to `True` (ignored).
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}} + \mathbf{M}_{\text{bias}}\right) \mathbf{V}$$
  where $\mathbf{M}_{\text{bias}}(i, j) = -\infty$ for absent modalities. This guarantees that missing modalities contribute zero gradient and cannot distort existing sensor measurements.

---

## 4. Sparse Spatial Graph Encoder (`src/models/graph/spatial_graph.py`)

Rather than treating the cyclone as an isotropic image, we construct a topological graph $G = (V, E)$ over 16 cyclone nodes (representing Core, Eyewall, Rainbands, and 4 Quadrants):
- **Nodes:** $V = \{v_1, \dots, v_{16}\}$ aggregated from multi-modal tokens.
- **Edges:** Directed edges connecting adjacent radial bins and azimuthal cyclonic neighbors.
- **Edge Attributes:** $\mathbf{e}_{ij} = [\Delta r, \Delta \theta, \sin \Delta \theta, \cos \Delta \theta, r_i, r_j, |\Delta r|]$.
- **Message Passing Neural Network (MPNN):**
  $$\mathbf{m}_{ij} = \text{MLP}_{\text{msg}}([\mathbf{h}_i, \mathbf{h}_j, \mathbf{e}_{ij}])$$
  $$\mathbf{h}_i^{(l+1)} = \text{LayerNorm}\left(\mathbf{h}_i^{(l)} + \text{MLP}_{\text{upd}}\left(\left[\mathbf{h}_i^{(l)}, \frac{1}{\text{deg}(i)}\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}\right]\right)\right)$$
- **Graph Pooling:** Attention-based pooling reduces the 16 graph nodes to a single cyclone state vector $\mathbf{h}_{\text{graph}} \in \mathbb{R}^{B \times d_{\text{model}}}$.

---

## 5. Causal Continuous-Time Temporal Transformer (`src/models/temporal/temporal_transformer.py`)

Satellite observations occur at irregular revisit intervals $\Delta t = t_i - t_{i-1} \in [1.2, 18.5]$ hours.
- **Fourier Continuous-Time Encoding:**
  $$\text{PE}(\Delta t, 2k) = \sin\left(\frac{\Delta t}{10000^{2k/d}}\right), \quad \text{PE}(\Delta t, 2k+1) = \cos\left(\frac{\Delta t}{10000^{2k/d}}\right)$$
- **Strict Causality:**
  Upper-triangular causal attention mask enforces that state at overpass $i$ can attend only to overpasses $j \le i$. Zero future observation or future best-track information is accessible during feature extraction.

---

## 6. Multi-Task Prediction Heads (`src/models/heads/task_heads.py`)

All tasks branch from the shared latent representation $\mathbf{z} \in \mathbb{R}^{B \times d_{\text{model}}}$:

| Head | Output Name | Architecture | Target Unit / Classes |
|---|---|---|---|
| **Intensity** | `intensity` | 2-layer MLP (GELU) | Wind speed ($v_{\text{max}}$ in kt) |
| **Pressure** | `pressure` | 2-layer MLP (GELU) | Central pressure ($p_{\text{min}}$ in hPa) |
| **Stage** | `stage_logits` | 2-layer MLP (GELU) | 6 classes (TD, TS, HU, EX, DB, UN) |
| **Intensity Change** | `intensity_change` | 2-layer MLP (GELU) | 4 horizons: $\Delta v_{+6\text{h}}, \Delta v_{+12\text{h}}, \Delta v_{+18\text{h}}, \Delta v_{+24\text{h}}$ (kt) |
| **Rapid Intensification** | `ri_logits` | 2-layer MLP (GELU) | Binary logit ($\Delta v_{24\text{h}} \ge 30$ kt) |
| **Track Displacement** | `track_delta` | 2-layer MLP (GELU) | 8 coordinates ($\Delta \text{lat}, \Delta \text{lon}$ across 4 horizons) |

Total parameters: **1,439,046** (1.44M parameters), striking an optimal balance between capacity, compute efficiency, and prevention of overfitting.

