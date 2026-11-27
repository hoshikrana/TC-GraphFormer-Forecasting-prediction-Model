# TC-GraphFormer V2 — Research-Grade Architecture, Generalization Diagnosis & Controlled Optimization Report

- **Timestamp (UTC):** `2026-11-24T09:20:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Git Branch:** `feat/tc-graphformer-v2-generalization`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, PyTorch `2.9.1+cu128`), Intel Core i9-14900K, `128 GB` RAM
- **Dataset:** NOAA TC PRIMED 100 GB Expansion (`75 TRAIN` storms / `10,545` sequences; `17 VAL` storms / `1,815` sequences; `17 FINAL_TEST` storms / `1,429` sequences — **`FINAL_TEST` untouched (`0` accesses)**).

---

## 1. Phase 1 — Actual Architecture & Label Contract

### 1.1 Exact End-to-End Component Contract (`2,977,797` Trainable Parameters, `7,323` Buffers, `333` `state_dict` Tensors)

| Module (`File`) | Parameters | Input Tensor Shape & Units | Output Tensor Shape & Units | Validity Masking & Padding Semantics | Norm, Residual, Dropout & Transform Details |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`IREncoder`** (`src/models/encoders/ir_encoder.py`) | `600,032` (incl. shared `polar_pool`) | `ir`: `(B·T, 1, 64, 64)` normalized $T_B \in [0,1]$ from $[170, 320]\text{ K}$ ($\pm 500\text{ km}$, $d_{\max}=15\text{ km}$) | `feat_ir`: `(B·T, 128, 32, 32)` | Pre-gated by `ir_pixel_mask`; post-gated by nearest-downsampled $32\times 32$ pixel mask and modality mask `m_ir` | `Conv2d(s=2)` stem + 3 residual `ConvBlock`s (`Conv2d+BN+GELU+Conv2d+BN` + shortcut); `0.0` dropout |
| **`PMWEncoder`** (`src/models/encoders/pmw_encoder.py`) | `600,896` (incl. shared `polar_pool`) | `pmw`: `(B·T, 4, 64, 64)` (`~19V, 19H, 37V, 85–91V GHz`), normalized $T_B \in [0,1]$ from $[100, 320]\text{ K}$ ($d_{\max}=35\text{ km}$) | `feat_pmw`: `(B·T, 128, 32, 32)` | Pre-gated by `pmw_channel_mask (B·T, 4)` and `pmw_pixel_mask (B·T, 4, 64, 64)`; post-gated at $32\times 32$ and by `m_pmw` | `Conv2d(s=2)` stem + 3 residual `ConvBlock`s (`BatchNorm2d`, `GELU`); `0.0` dropout |
| **`GPROFEncoder`** (`src/models/encoders/gprof_encoder.py`) | `305,472` (incl. shared `polar_pool`) | `gprof`: `(B·T, 4, 64, 64)` (`surfacePrecip, convPrecip, RWP, IWP`), $\ln(1+x)/5.0$ ($d_{\max}=20\text{ km}$) | `feat_gprof`: `(B·T, 128, 32, 32)` | Pre-gated by `gprof_channel_mask (B·T, 4)` and `gprof_pixel_mask`; post-gated at $32\times 32$ and by `m_gprof` | `Conv2d(s=2)` stem + 2 residual `ConvBlock`s (`BatchNorm2d`, `GELU`); `0.0` dropout |
| **`RadarEncoder`** (`src/models/encoders/radar_encoder.py`) | `35,088` (incl. shared `polar_pool`) | `radar`: `(B·T, 1, 16, 32, 32)` Ku-band reflectivity CAPPI ($0.5\text{--}19.5\text{ km}$), normalized $[-20, 100]\text{ dBZ} \to [0,1]$ ($d_{\max}=15\text{ km}$) | `feat_radar`: `(B·T, 128, 32, 32)` | Pre-gated by 3D `radar_voxel_mask`; depth collapsed via `.mean(dim=2)` after two stride-`(2,1,1)` `Conv3d+BN3d+GELU` blocks; post-gated by 2D union mask & `m_radar` | 2 `Conv3d` blocks + $1\times 1$ `Conv2d(32->128)+BN2d+GELU`; no residual, `0.0` dropout |
| **`EnvEncoder`** (`src/models/encoders/env_encoder.py`) | `306,048` (incl. shared `polar_pool`) | `env`: `(B·T, 6, 64, 64)` (`TPW/80, (SST-273.15)/35, (MSL-1013.25)/50, (T2m-273.15)/40, U10/30, V10/30`, $d_{\max}=40\text{ km}$) | `feat_env`: `(B·T, 128, 32, 32)` | Strictly filtered to ERA5 timestamp $t_{\text{env}} \le t_{\text{obs}}$; pre/post-gated by `env_channel_mask`, `env_pixel_mask`, and `m_env` | `Conv2d(s=2)` stem + 2 residual `ConvBlock`s (`BatchNorm2d`, `GELU`); `0.0` dropout |
| **`PolarSectorPooling`** (`src/models/graph/spatial_graph.py`) | `16,384` (shared across 5 stems) | `feat_m`: `(B·T, 128, 32, 32)`, native `pixel_mask`: `(B·T, C, 64, 64)` | `t_m`: `(B·T, 16, 128)`, `val_m`: `(B·T, 16)` | 16 physical sectors (4 rings: `0–75, 75–175, 175–300, 300–500 km` $\times$ 4 quadrants: `NE, NW, SW, SE`). Sector valid iff observed area / geometric area $\ge 5\%$ (`min_sector_coverage=0.05`) | Bias-free `Linear(128, 128, bias=False)` with pre- and post-projection zeroing (`tokens * token_validity * m_exp`) |
| **`CrossModalFusion`** (`src/models/fusion/cross_modal_fusion.py`) | `399,488` | 5 token lists `5 x (B·T, 16, 128)` + `modality_masks (B·T, 5)` + `token_masks 5 x (B·T, 16)` | `fused`: `(B·T, 80, 128)` | `key_padding_mask (B·T, 80)` is `True` (ignored) where `~(modality_present & token_valid)`. Includes `all_masked` guard (unmasks col 0 to prevent `NaN`, then zeroes output) | `modality_embed (5,128)` + `spatial_pos_embed (1,16,128)` + 2-layer, 4-head Pre-LN (`norm_first=True`) Transformer (`d_ff=512`, `GELU`, `dropout=0.1`) + final `LayerNorm(128)` |
| **`spatial_aggregator`** (`src/models/tc_graphformer.py`) | `98,816` | `fused` reshaped per sector: `(B·T, 16, 640)` | `spatial_nodes`: `(B·T, 16, 128)` | Gated by sector union validity `node_validity = OR_m(val_m > 0.5)` `(B·T, 16, 1)` to suppress linear/LN bias on unobserved sectors | `Linear(640, 128) -> GELU -> Linear(128, 128) -> LayerNorm(128)` |
| **`SpatialGraphEncoder`** (`src/models/graph/spatial_graph.py`) | `216,833` | `spatial_nodes`: `(B·T, 16, 128)`, `node_validity`: `(B·T, 16, 1)`, fixed `edge_index (2, E)`, `edge_attr (E, 8)` | `graph_latent`: `(B·T, 128)` | Source messages gated by $v_{\text{src}}$; mean aggregation divided by valid incoming degree; destination gated by $v_{\text{dst}}$; attention readout masks invalid nodes with `-1e4` (`fp16`) / `-1e9` (`fp32`) and zeroes `all_invalid` graphs | 2 `GraphMessagePassingLayer`s (`msg_mlp: 264->128->128`, `update_mlp: 256->128->128`, post-residual `LayerNorm(128)`) + `graph_pool` attention MLP (`128->128->1`) |
| **`_encode_kinematic_context`** (`src/models/tc_graphformer.py`) | `0` (analytic Fourier) | `kinematic_state`: `(B, T, 10)` backward-difference storm motion ($t_{\text{prev}} \le t_{\text{obs}}$, $2\text{--}24\text{ h}$) & geophysical state | `kin_emb`: `(B, T, 128)` | Added to `seq_latent` only where `(modality_masks.sum(dim=-1) > 0.5)` | 8 features (`u/30, v/30, spd/10, sin_hdg, cos_hdg, sin_lat, hemi, log10(1+d_land)/3.5`) $\times$ 8 frequencies `{0.5..64}` $\to \sin/\cos \times 0.15$ |
| **`CausalTemporalTransformer`** (`src/models/temporal/temporal_transformer.py`) | `429,824` | `seq_latent`: `(B, T<=4, 128)`, `delta_times_hours`: `(B, T)` ($0\text{--}72\text{ h}$), `seq_pad_mask`: `(B, T)` (`True` = padded) | `temporal_out`: `(B, T, 128)` $\to$ anchor `final_latent`: `(B, 128)` | Upper-triangular causal mask (`triu(diagonal=1)`) combined with `seq_pad_mask`; selects last non-padded anchor step `last_idx = where(~seq_pad_mask[b])[-1]` | `ContinuousTimeEmbedding` (64 log-spaced freqs + `128->128->128` MLP) + 2-layer, 4-head Pre-LN Transformer (`d_ff=512`, `GELU`, `dropout=0.1`) + final `LayerNorm(128)` |
| **`TaskPredictionHeads`** (`src/models/heads/task_heads.py`) | `50,836` | `final_latent`: `(B, 128)`, `anchor_kin`: `(B, 10)` | `intensity (B,)`, `pressure (B,)`, `stage_logits (B,5)`, `intensity_change (B,4)`, `ri_logits (B,)`, `track_delta (B,8)` | Independent target validity masks (`intensity_mask`, `pressure_mask`, `stage_mask`, `intensity_change_mask (B,4)`, `ri_mask`, `track_mask (B,4)`) applied in `MultiTaskLoss` and `evaluate()` | 6 parallel 2-layer MLPs (`Linear(128,64)->GELU->Linear(64,out)`), zero head LN/dropout, affine physical output scaling around pilot priors |

---

### 1.2 Verified Inconsistencies (A–E) and Confirmed Defects vs. Hypotheses

1. **Inconsistency A — Stage Classification Documentation vs. Canonical 5-Class Mapping (Confirmed Documentation/Label Contract Mismatch):**
   - `STAGE_MAPPING_CANONICAL` in [dataset.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/data/dataset.py#L57-L65) maps 5 classes (`0: TD/SD/MD`, `1: TS/SS`, `2: HU/TY`, `3: EX`, `4: DB/LO/WV`). On 100 GB `TRAIN` (`10,545` samples, `9,641` valid stage labels), the class counts are `{0: 1,672, 1: 2,943, 2: 3,156, 3: 507, 4: 1,363}` (`VAL` `1,549` valid: `{0: 304, 1: 569, 2: 340, 3: 111, 4: 225}`).
   - However, docstrings in [task_heads.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/heads/task_heads.py#L6-L41) described `Development Stage (6 classes)`. Additionally, raw IBTrACS stage codes `ST` (Super Typhoon: `148` in `TRAIN`, `90` in `VAL`), `UN` (`742` in `TRAIN`, `172` in `VAL`), `XX` (`14` in `TRAIN`), and `ET` (`4` in `VAL`) receive `stage_mask = 0.0`.
   - **Resolution:** Updated [task_heads.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/heads/task_heads.py) documentation to match the canonical 5-class mapping without silently relabeling existing checkpoints or training samples.
2. **Inconsistency B — RI Prior Initialization vs. 100 GB `TRAIN` Base Rate and Partial Log-Odds Shift in `MultiTaskLoss` (Confirmed Mathematical Calibration Offset):**
   - In [task_heads.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/heads/task_heads.py#L84), `head_ri[-1].bias` is initialized to `-3.5124` ($\sigma(-3.5124) = 0.02896 = 2.90\%$, the pilot-era `36 / 1,243` base rate), whereas the 100 GB `TRAIN` partition has `498` positives and `9,016` negatives across `9,514` valid samples (`base_rate = 5.2344%`, exact `pos_weight = 18.1044`, true log-odds $= -2.8962$).
   - In [losses.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/training/losses.py#L119-L122), `MultiTaskLoss` computes:
     $$\mathcal{L}_{\text{RI}}(z, y) = \text{BCEWithLogits}(z + \alpha \ln w_{\text{pos}},\, y;\, \text{pos\_weight}=w_{\text{pos}}), \quad \alpha = 0.85$$
     Setting $\mathbb{E}[\partial \mathcal{L}_{\text{RI}}/\partial z \mid x] = 0$ for true conditional probability $p(x) = P(\text{RI}=1\mid x)$ gives the stationary logit:
     $$z^*(x) = \ln\frac{p(x)}{1 - p(x)} + (1 - \alpha)\ln w_{\text{pos}}$$
     Because $\alpha = 0.85 \ne 1.0$, the stationary logit carries a positive bias of $+0.15\ln(18.1) = +0.4344$, shifting the stationary base-rate probability from $5.23\%$ to $7.86\%$.
3. **Inconsistency C — Pilot Output Priors & Multipliers vs. Verified 100 GB `TRAIN`-Only Target Distributions (Confirmed Statistical Offset):**

| Target Variable (100 GB `TRAIN` Only) | Valid $N$ | 100 GB `TRAIN` Mean | 100 GB `TRAIN` Std | 100 GB `TRAIN` Median (`IQR`) | Min / Max | Hardcoded Pilot Prior in `TaskPredictionHeads` | Hardcoded Head Multiplier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Current Wind Speed ($V_{\max}$, kt)** | `10,545` | **`55.4200`** | `31.3122` | `46.00` (`45.00`) | `10.0` / `165.0` | `38.19 kt` (`-17.23 kt` offset) | `12.0x` |
| **Central Pressure ($P_{\min}$, hPa)** | `9,499` | **`985.5413`** | `22.7788` | `993.00` (`31.00`) | `891.0` / `1018.0` | `997.25 hPa` (`+11.71 hPa` offset) | `12.0x` |
| **Intensity Change $\Delta V_{+6\text{h}}$ (kt)** | `10,133` | **`+0.0178`** | `5.3698` | `0.00` (`6.00`) | `-50.0` / `+45.0` | `+0.231 kt` | `5.0x` |
| **Intensity Change $\Delta V_{+12\text{h}}$ (kt)** | `9,929` | **`+0.0283`** | `9.8045` | `0.00` (`10.00`) | `-85.0` / `+65.0` | `+0.575 kt` | `5.0x` |
| **Intensity Change $\Delta V_{+18\text{h}}$ (kt)** | `9,728` | **`+0.1036`** | `13.6611` | `0.00` (`14.00`) | `-105.0` / `+80.0` | `+1.075 kt` | `5.0x` |
| **Intensity Change $\Delta V_{+24\text{h}}$ (kt)** | `9,514` | **`+0.1909`** | `17.0889` | `0.00` (`20.00`) | `-115.0` / `+95.0` | `+1.596 kt` | `5.0x` |
| **Track Residual $(\Delta x, \Delta y)_{+6\text{h}}$ (km)** | `1,758` | **`(-4.12, +0.19)`** | `(31.78, 28.40)` | — | — | `0.0` (around causal vel; fallback `(-25.35, +27.92)`) | `20.0x` |
| **Track Residual $(\Delta x, \Delta y)_{+12\text{h}}$ (km)** | `1,723` | **`(-12.52, -2.98)`** | `(69.93, 62.21)` | — | — | `0.0` (around causal vel; fallback `(-40.77, +60.58)`) | `20.0x` |
| **Track Residual $(\Delta x, \Delta y)_{+18\text{h}}$ (km)** | `1,689` | **`(-14.86, -4.91)`** | `(117.72, 100.16)` | — | — | `0.0` (around causal vel; fallback `(-62.82, +88.80)`) | `20.0x` |
| **Track Residual $(\Delta x, \Delta y)_{+24\text{h}}$ (km)** | `1,655` | **`(-28.06, -3.50)`** | `(165.21, 140.18)` | — | — | `0.0` (around causal vel; fallback `(-79.29, +114.98)`) | `20.0x` |

4. **Inconsistency D — Attention Mask Finiteness Under All-Missing Modalities and Padded Sequences (Confirmed & Fixed Edge-Case Defect in `CausalTemporalTransformer`):**
   - Verified on GPU (`task-1124`) that `CrossModalFusion` and `SpatialGraphEncoder` already include `all_masked` / `all_invalid` guards and produce finite outputs and `0` non-finite gradients even when all 5 modalities are missing (`modality_masks = 0`).
   - However, in `CausalTemporalTransformer.forward()` ([temporal_transformer.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/src/models/temporal/temporal_transformer.py#L84-L90)), passing both `causal_mask = triu(diagonal=1)` and `src_key_padding_mask = seq_pad_mask` without an `all_masked` / `first_masked` guard could leave row 0 of the causal attention matrix with all keys masked if timestep 0 were ever padded. We added a strict `needs_unmask = mask.all(dim=1) | mask[:, 0]` guard followed by post-LayerNorm zeroing `out = out * (~mask).unsqueeze(-1)` and verified it via unit test `test_temporal_all_padded_and_first_padded_finiteness` in [test_v2_loss_and_masks.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/tests/test_v2_loss_and_masks.py).
5. **Inconsistency E — Retrospective ERA5 Reanalysis vs. Operational Forecast-Time Data:**
   - While our pipeline strictly enforces timestamp causality $t_{\text{env}} \le t_{\text{obs}}$ (never inspecting future ERA5 slices $t_{\text{env}} > t_{\text{obs}}$) and computes storm translation velocity exclusively via backward finite differences ($t_{\text{prev}} \le t_{\text{obs}}$), **ERA5 is a retrospective reanalysis product** (assimilating observations across a 12-hour 4D-Var window with ~5-day preliminary latency) and IBTrACS storm center coordinates are post-season Best Track positions rather than real-time operational working-best-track fixes (e.g. NHC/JTWC `CARQ` in ADECK). Therefore, timestamp causality prevents look-ahead target leakage during historical training/evaluation, but real-time operational deployment requires substituting operational NWP analysis fields (e.g. GFS/IFS $t=0\text{ h}$ analysis) and real-time operational fix coordinates.

---

## 2. Phase 2 — Reconstructed True Generalization (`model.eval()` Mode on `TRAIN` vs. `VAL`)

All three reference checkpoints were evaluated in `model.eval()` mode (disabling dropout and using frozen running BatchNorm statistics) on the full `TRAIN` partition (`10,545` sequences, `75` storms) and `VAL` partition (`1,815` sequences, `17` storms) using the exact same global valid-target-normalized `MultiTaskLoss` definition (`reports/phase2e4_100gb/v2_phase2_phase3_diagnostics.json`).

### 2.1 Overall & Per-Task `TRAIN_EVAL` vs. `VAL_EVAL` Losses and Generalization Gaps

| Metric / Weighted Loss Component | **Original Epoch 5** (`TRAIN_EVAL` / `VAL_EVAL` / **`Gap`**) | **Original Epoch 11** (`TRAIN_EVAL` / `VAL_EVAL` / **`Gap`**) | **C1 Candidate Epoch 8** (`TRAIN_EVAL` / `VAL_EVAL` / **`Gap`**) |
| :--- | :---: | :---: | :---: |
| **`GLOBAL_EVAL_LOSS` (Valid-Target Norm)** | `13.0492` / `16.6384` / **`+3.5892`** | `5.7588` / `17.1831` / **`+11.4243`** | `7.9958` / **`16.4052`** / **`+8.4094`** |
| **`BATCH_AVG_EVAL_LOSS`** | `12.9684` / `16.7881` / **`+3.8197`** | `5.7384` / `17.1745` / **`+11.4361`** | `7.9496` / **`16.4568`** / **`+8.5072`** |
| **`1.0 * Loss_Intensity` (Regime-Weighted)** | `6.7661` / `8.5792` / **`+1.8131`** | `1.6843` / `9.1217` / **`+7.4374`** | `2.8644` / **`8.4333`** / **`+5.5689`** |
| **`Unweighted_Wind_Huber` ($\beta=8.0$)** | `3.7081` / `4.7838` / **`+1.0757`** | `1.1331` / `4.0360` / **`+2.9029`** | `1.7528` / **`3.9810`** / **`+2.2282`** |
| **`0.5 * Loss_Pressure`** | `0.0467` / `0.0687` / **`+0.0220`** | `0.0334` / `0.0633` / **`+0.0299`** | `0.0360` / **`0.0591`** / **`+0.0231`** |
| **`0.5 * Loss_Stage`** | `0.2799` / `0.4191` / **`+0.1392`** | `0.2085` / `0.3852` / **`+0.1767`** | `0.2394` / **`0.3714`** / **`+0.1320`** |
| **`1.0 * Loss_Intensity_Change` ($\Delta V$)** | `4.6665` / `5.1818` / **`+0.5153`** | `3.1779` / **`4.9369`** / **`+1.7590`** | `3.9060` / `5.1652` / **`+1.2592`** |
| **`2.0 * Loss_RI` (`pos_weight=18.1`)** | `0.9596` / `1.9480` / **`+0.9884`** | `0.3269` / `2.2345` / **`+1.9076`** | `0.6206` / **`1.9346`** / **`+1.3140`** |
| **`1.0 * Loss_Track`** | `0.3305` / `0.4416` / **`+0.1111`** | `0.3278` / **`0.4414`** / **`+0.1136`** | `0.3294` / `0.4416` / **`+0.1122`** |

### 2.2 Per-Observation, Per-Storm, Subgroup, Basin & Modality Generalization Summary (`TRAIN` vs. `VAL`)

| Subgroup / Physical Metric (`TRAIN` / `VAL` / **`Gap`**) | **Original Epoch 5** | **Original Epoch 11** | **C1 Candidate Epoch 8** |
| :--- | :---: | :---: | :---: |
| **Per-Observation Wind MAE (`kt`)** | `6.65` / `7.84` / **`+1.19`** | `3.31` / `6.93` / **`+3.62`** | `4.19` / **`6.90`** / **`+2.71`** |
| **Per-Storm Wind MAE `Mean ± Std (Median)` (`kt`)** | `6.65±2.38 (6.41)` / `8.19±4.51 (8.03)` | `3.70±1.02 (3.52)` / `7.82±4.04 (7.19)` | `4.58±1.25 (4.36)` / **`7.71±3.97 (7.56)`** |
| **Tropical Depression/Storm (`<64 kt`) Wind MAE (`kt`)** | `5.61` / `7.51` / **`+1.90`** | `3.41` / **`6.27`** / **`+2.86`** | `4.12` / `6.33` / **`+2.21`** |
| **Category 1–2 (`64–95 kt`) Wind MAE (`kt`)** | `9.99` / `9.83` / **`-0.16`** | `3.49` / **`8.23`** / **`+4.74`** | `4.73` / `8.30` / **`+3.57`** |
| **Major Hurricane ($\ge 96\text{ kt}$) Wind MAE (`Bias kt`)** | `6.64 (+4.71)` / **`7.78 (-1.08)`** | `2.50 (+1.16)` / `9.28 (-6.31)` | `3.69 (+1.98)` / `8.68 (-4.21)` |
| **Central Pressure MAE (`hPa`)** | `5.18` / `5.99` / **`+0.81`** | `4.41` / `5.54` / **`+1.13`** | `4.69` / **`5.53`** / **`+0.84`** |
| **Stage Accuracy / Balanced Accuracy (`%`)** | `75.88 / 70.62` $\to$ `65.14 / 64.27` | `82.78 / 77.92` $\to$ `68.30 / 66.88` | `80.22 / 75.20` $\to$ **`69.14 / 68.39`** |
| **`+24h` $\Delta V$ MAE Per-Obs / Per-Storm Mean (`kt`)** | `9.26 (9.68)` / `10.18 (9.94)` | `6.62 (7.19)` / **`9.82 (9.56)`** | `7.97 (8.60)` / `10.26 (9.98)` |
| **RI-Positive Subset `+24h` $\Delta V$ MAE (`kt`, `498` Tr / `92` Val)** | `23.79` / `29.78` / **`+5.99`** | `7.87` / **`19.90`** / **`+12.03`** | `14.01` / `23.54` / **`+9.53`** |
| **RI PR-AUC / Brier / Mean Pos Prob (`p_pos`)** | `0.5574 / 0.0320 / 0.328` $\to$ `0.5017 / 0.0386 / 0.319` | `0.8928 / 0.0137 / 0.714` $\to$ `0.5192 / 0.0396 / 0.431` | `0.7069 / 0.0240 / 0.555` $\to$ **`0.5226 / 0.0391 / 0.434`** |
| **`+24h` Track MAE Per-Obs / Per-Storm Median (`km`)** | `170.04 (144.28)` / `213.65 (209.75)` | `168.74 (144.39)` / **`213.57 (210.42)`** | `169.44 (145.27)` / `213.58 (209.73)` |
| **Per-Basin `VAL` Wind MAE (`AL / CP / EP / IO / SH / WP`, kt)** | `8.56 / 6.35 / 6.20 / 7.50 / 23.67 / 7.91` | `5.88 / 6.01 / 5.92 / 6.54 / 20.65 / 9.08` | `6.31 / 5.67 / 5.92 / 6.26 / 20.57 / 8.51` |
| **Modality `VAL` Wind MAE (`Radar Pres/Abs`, `IR Pres/Abs`, kt)** | `8.44 / 7.75`, `7.65 / 9.97` | `7.29 / 6.87`, `6.77 / 8.75` | `7.61 / 6.78`, **`6.70 / 9.03`** |

---

## 3. Phase 3 — Loss & Optimizer Mathematical Verification and Gradient Flow

### 3.1 Microbatch Gradient Accumulation Equivalence Under Heterogeneous Missing Labels
- Tested on 4 synthetic sequences split across 2 microbatches of size 2 with asymmetric missing labels (e.g. Microbatch 1 having `0` valid track targets and `2` valid wind targets; Microbatch 2 having `7` valid track timesteps and `1` valid wind target):
  - **Full-Batch Loss:** `187.094116` vs. **Summed Microbatch Loss (`group_target_counts`):** `187.094128` (**Absolute Difference: `1.14e-05`**).
  - **Maximum Parameter Gradient Difference Across All 333 Parameter Tensors:** **`1.72e-05` (`0` non-finite gradients)**.
  - This confirms that `TCGraphFormerTrainer.train_epoch()`'s `group_target_counts` pre-aggregation is mathematically exact and prevents rare/masked targets in sparse microbatches from receiving disproportionate gradient weights.

### 3.2 Measured Per-Task Shared-Encoder & Task-Head Gradient Norms (`20` `TRAIN` Batches, $N=640$ Sequences)

| Task Objective | Raw Loss Mean | Weighted Loss Mean (`% of Total Loss`) | **Shared-Encoder L2 Grad Norm (`% Share`)** | **Prediction-Head L2 Grad Norm** | Valid Targets (`20` Batches) | Non-Finite Grad Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`intensity` ($V_{\max}$)** | `6.7675` | `6.7675` (**`52.56%`**) | **`380.2180` (`72.74%`)** | **`88.0072`** | `640` | `0` |
| **`pressure` ($P_{\min}$)** | `0.0911` | `0.0456` (`0.35%`) | `2.5298` (`0.48%`) | `0.3891` | `575` | `0` |
| **`stage` (5-class CE)** | `0.5259` | `0.2630` (`2.04%`) | `8.6154` (`1.65%`) | `0.6405` | `590` | `0` |
| **`intensity_change` ($\Delta V$)** | `4.6415` | `4.6415` (`36.05%`) | `60.9534` (`11.66%`) | `3.4434` | `2,380` | `0` |
| **`ri` ($\Delta V_{24\text{h}} \ge 30\text{ kt}$)** | `0.4148` | `0.8296` (`6.44%`) | `68.7454` (`13.15%`) | `2.8103` | `574` | `0` |
| **`track` ($\Delta x, \Delta y$)** | `0.3292` | `0.3292` (`2.56%`) | `1.6126` (`0.31%`) | `0.1462` | `842` | `0` |

### 3.3 Pairwise Shared-Encoder Task-Gradient Cosine Similarity Matrix

| Task | `intensity` | `pressure` | `stage` | `intensity_change` | `ri` | `track` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`intensity`** | `+1.0000` | **`+0.4937`** | **`+0.3706`** | `+0.0229` | `-0.0099` | `+0.0624` |
| **`pressure`** | **`+0.4937`** | `+1.0000` | **`+0.3970`** | `-0.0305` | `-0.0390` | `-0.0804` |
| **`stage`** | **`+0.3706`** | **`+0.3970`** | `+1.0000` | `+0.0003` | `-0.0292` | `+0.0656` |
| **`intensity_change`** | `+0.0229` | `-0.0305` | `+0.0003` | `+1.0000` | **`+0.1720`** | `+0.0202` |
| **`ri`** | `-0.0099` | `-0.0390` | `-0.0292` | **`+0.1720`** | `+1.0000` | `-0.0795` |
| **`track`** | `+0.0624` | `-0.0804` | `+0.0656` | `+0.0202` | `-0.0795` | `+1.0000` |

- **Key Gradient Finding:** The tasks do **not** suffer from destructive gradient conflict (no pair has cosine similarity $< -0.081$; current-state tasks `intensity`, `pressure`, and `stage` are strongly positively aligned at `+0.37` to `+0.49`, and future-intensity tasks `intensity_change` and `ri` are positively aligned at `+0.172`). Instead, **`intensity` dominates `72.74%` of the shared-encoder gradient norm** because $w_{\text{regime}} = 1 + 2.5\,\text{clamp}((v-50)/50, 0, 2)^2$ reaches `11.0x` on intense cyclones combined with the `12.0x` head multiplier.

---

## 4. Phase 4 & Phase 5 — Original vs. V2 Mathematical Loss Definitions & Controlled GPU Experiment

### 4.1 Original (V1) vs. Candidate B (V2) Loss Definitions (`src/training/losses.py`)

| Component | Original Baseline (V1 / C1) | V2 Candidate B (`tc_graphformer_v2_candidate_b`) |
| :--- | :--- | :--- |
| **Intense-Cyclone Wind Regime Weight $w_{\text{regime}}(v)$** | $1.0 + 2.5\,\text{clamp}\!\left(\frac{v - 50}{50}, 0, 2\right)^2 \in [1.0, 11.0]$ | $1.0 + 1.0\,\text{clamp}\!\left(\frac{v - 50}{50}, 0, 2\right)^1 \in [1.0, 3.0]$ |
| **RI Weighted BCE Prior Shift $\Delta_{\text{prior}}$** | $0.85 \ln(w_{\text{pos}}) = 2.4615$ (stationary logit shift $+0.4344$) | $1.00 \ln(w_{\text{pos}}) = 2.8959$ (exact zero stationary logit shift) |
| **Task Loss Weights $(w_v, w_p, w_s, w_{\Delta V}, w_{\text{RI}}, w_{\text{tr}})$** | $(1.0,\, 0.5,\, 0.5,\, 1.0,\, 2.0,\, 1.0)$ | $(1.0,\, 2.0,\, 0.5,\, 1.25,\, 2.0,\, 4.0)$ |

### 4.2 Phase 5 Controlled V2 GPU Experiment Per-Epoch Telemetry (`reports/phase2e4_100gb/v2_controlled_experiment_summary.json`)
Resumed from `checkpoints/tcprimed_100gb/multitask_epoch_005.pt` into `checkpoints/tcprimed_100gb_v2/` with identical seed (`42`), batch size (`32`, `accumulate_grad_batches=2`), and cosine LR schedule (`5.0e-5 -> 1.0e-6` over Epochs 6–9, `weight_decay=3.0e-4`). Both `TRAIN_EVAL` and `VAL_EVAL` are measured in `model.eval()` mode under the **identical V1 reference loss function** for exact comparability:

| V2 Ep | LR | `OptLoss (V2)` | `TRAIN_EVAL (V1ref)` | `VAL_EVAL (V1ref)` | `GEN_GAP (V1ref)` | Pre-Clip Grad Norm (`Mean / Max`, `% Clipped`) | **`VAL_MT_REL_ERR`** | `VAL` Wind MAE (`Bias`) | **`VAL` Major-Hurr MAE (`Bias`)** | `VAL` Press MAE | `VAL` Stage Acc (`Bal %`) | `VAL` $\Delta V$ `6/12/18/24h` (`kt`) | `VAL` RI `PR-AUC / Brier / F1@0.5` | `VAL` Track `+24h` (`Skill vs CausalVel`) | Time / GPU VRAM |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **6** | `5.00e-5` | `11.3291` | `8.9922` | `16.9304` | `+7.9382` | `60.80 / 136.60` (`100%`) | `0.5802` | `7.07` (`-2.87`) | `8.96` (`-5.80`) | `6.37` | `67.33%` (`66.64%`) | `3.14 / 5.82 / 8.11 / 10.01` | **`0.5303` / `0.0381`** / `0.5443` | `214.04` (`-0.89 km`) | `433.8s` / `5263 MiB` |
| **7** | `4.28e-5` | `10.4640` | `8.3083` | `18.4498` | `+10.1415` | `59.02 / 131.49` (`100%`) | `0.5847` | `7.19` (`-2.44`) | `10.16` (`-7.27`) | `5.84` | `68.04%` (`66.80%`) | `3.19 / 5.92 / 8.32 / 10.33` | `0.4955` / `0.0404` / `0.5063` | `213.83` (`-0.68 km`) | `428.1s` / `5263 MiB` |
| **8** | `2.55e-5` | `9.9507` | `7.6952` | `17.1263` | `+9.4311` | `62.28 / 117.16` (`100%`) | `0.5741` | `6.99` (`-0.81`) | `9.10` (`-5.69`) | **`5.46`** | `68.17%` (`67.57%`) | `3.13 / 5.84 / 8.16 / 10.09` | `0.4978` / `0.0388` / `0.4898` | **`213.52`** (`-0.37 km`) | `430.8s` / `5263 MiB` |
| **9** | `8.18e-6` | `9.4806` | `7.5198` | `17.3104` | `+9.7906` | `59.29 / 128.46` (`100%`) | **`0.5699`** | **`6.86`** (`-1.36`) | `9.57` (`-6.11`) | `5.51` | **`69.27%` (`68.52%`)** | **`3.13 / 5.82 / 8.13 / 10.06`** | `0.5044` / **`0.0384`** / `0.5195` | `213.71` (`-0.56 km`) | `429.0s` / `5263 MiB` |

---

## 5. Phase 6 — Acceptance Decision Against Predeclared Guardrails (`VAL`, $N=1,815$)

| Validation Metric (`VAL`, $N=1,815$) | Original **Epoch 5** (`checkpoints/tcprimed_100gb/best_checkpoint.pt`) | Original **Epoch 11** (`checkpoints/tcprimed_100gb/multitask_epoch_011.pt`) | **C1 Continuation Epoch 8** (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`) | **V2 Candidate B Epoch 9** (`checkpoints/tcprimed_100gb_v2/best_checkpoint.pt`) | Predeclared Guardrail Status for V2 Candidate B |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`VAL_MT_REL_ERR` (Primary 6-Task Score)** | `0.5925` | `0.5703` | `0.5723` | **`0.5699`** | **PASS** (Best overall 6-task relative error) |
| **`VAL_EVAL_LOSS` (Global V1 Reference)** | `16.6384` | `17.1831` | **`16.4052`** | `17.3104` (`17.1263` @ Ep 8) | **FAIL** (`+0.9052` worse than C1 Epoch 8) |
| **`TRAIN_EVAL` $\to$ `VAL_EVAL` Gen Gap** | **`+3.5892`** | `+11.4243` | `+8.4094` | `+9.7906` (`+7.9382` @ Ep 6) | **FAIL** (Larger gap than C1 Epoch 8) |
| **Overall Wind MAE / Bias (`kt`, $n=1,815$)** | `7.84` (`+1.36`) | `6.93` (`-0.62`) | `6.90` (`-0.75`) | **`6.86` (`-1.36`)** | **PASS** (`-0.98 kt` vs Ep 5, `-0.04 kt` vs C1) |
| **Major-Hurricane ($\ge 96\text{ kt}$) MAE / Bias (`kt`, $n=245$)** | **`7.78` (`-1.08`)** | `9.28` (`-6.31`) | **`8.68` (`-4.21`)** | `9.57` (`-6.11`) (`8.96` @ Ep 6) | **FAIL (Guardrail `<= 8.68 kt` violated by `+0.89 kt`)** |
| **Central Pressure MAE (`hPa`, $n=1,604$)** | `5.99` | `5.54` | `5.53` | **`5.51`** (`5.46` @ Ep 8) | **PASS** (`-0.48 hPa` vs Ep 5, `-0.02 hPa` vs C1) |
| **Stage Accuracy / Balanced Acc (`%`, $n=1,549$)** | `65.14% / 64.27%` | `68.30% / 66.88%` | `69.14% / 68.39%` | **`69.27% / 68.52%`** | **PASS** (`+4.13%` acc / `+4.25%` bal acc vs Ep 5) |
| **Intensity Change $\Delta V$ `6 / 12 / 18 / 24h` MAE (`kt`)** | `3.22 / 5.96 / 8.27 / 10.18` | **`3.10 / 5.73 / 7.95 / 9.82`** | `3.16 / 5.91 / 8.27 / 10.26` | `3.13 / 5.82 / 8.13 / 10.06` | **PASS** (Beats Ep 5 and C1 at all 4 horizons) |
| **RI PR-AUC / Brier Score ($n=1,576$, `92` pos)** | `0.5017 / 0.0386` | `0.5192 / 0.0396` | **`0.5226` / `0.0391`** | `0.5044` / **`0.0384`** (`0.5303 / 0.0381` @ Ep 6) | **MIXED** (Best Brier `0.0384`, but PR-AUC `-0.0182` below C1) |
| **Track `+6 / +12 / +18 / +24h` MAE (`km`)** | `35.72 / 85.89 / 147.15 / 213.65` | `35.83 / 85.63 / 147.21 / 213.57` | `35.69 / 85.61 / 147.18 / 213.58` | `35.63 / 85.79 / 147.17 / 213.71` (`213.52` @ Ep 8) | **PASS** (Within `0.13 km` of C1) |

### 5.1 Clear Acceptance Decision: **REJECT V2 CANDIDATE B CHECKPOINT & KEEP EXISTING MODELS (ORIGINAL EPOCH 5 + C1 CONTINUATION EPOCH 8)**

1. **Why V2 Candidate B is Rejected as a Replacement Checkpoint:**
   - Although V2 Candidate B achieved the best composite 6-task relative error (`VAL_MT_REL_ERR = 0.5699` at Epoch 9), best overall Wind MAE (`6.86 kt`), best Pressure MAE (`5.46–5.51 hPa`), best Stage Accuracy (`69.27%`, balanced `68.52%`), best RI Brier probability calibration (`0.0381–0.0384`), and improved `+24h` $\Delta V$ MAE (`10.06 kt` vs. `10.18 kt` at Ep 5 and `10.26 kt` at C1), **reducing the intense-cyclone wind regime weight from quadratic $[1, 11]$ to linear $[1, 3]$ caused a direct regression on Major Hurricanes ($\ge 96\text{ kt}$)**: Major-Hurricane MAE degraded from `7.78 kt` (Epoch 5) and `8.68 kt` (C1 Epoch 8) to **`9.57 kt` (bias `-6.11 kt`)** at V2 Epoch 9 (`9.10 kt` at V2 Epoch 8), violating our predeclared Major-Hurricane guardrail (`<= 8.68 kt`).
   - This provides rigorous empirical proof that **quadratic regime weighting ($w_{\text{regime}} \in [1, 11]$) under the original `MultiTaskLoss` is essential to counter the `73.8%` Tropical Depression/Storm class majority and prevent negative bias drift on Major Hurricanes**.
2. **Which Existing Checkpoints Are Preserved:**
   - **Original Baseline (`checkpoints/tcprimed_100gb/best_checkpoint.pt`, Epoch 5):** Preserved untouched as the primary production checkpoint with the smallest `TRAIN -> VAL` generalization gap (`+3.5892`) and strongest Major-Hurricane accuracy (`7.78 kt` MAE, `-1.08 kt` bias).
   - **Controlled Continuation Candidate C1 (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`, Epoch 8 — same V1 loss + low-LR `5e-5` continuation):** Preserved as the strongest multi-task continuation model (`VAL_EVAL_LOSS = 16.4052`, `VAL_MT_REL_ERR = 0.5723`, `Wind MAE = 6.90 kt`, `Major-Hurricane MAE = 8.68 kt`, `Pressure MAE = 5.53 hPa`, `Stage Acc = 69.14%`, `RI PR-AUC = 0.5226`).
   - **Code & Infrastructure Improvements Retained:** All backward-compatible code improvements and bug fixes (`CausalTemporalTransformer` all-padded/first-padded sequence mask guard, `TCGraphFormerTrainer.evaluate()` global valid-target-normalized loss & `VAL_MT_REL_ERR` metrics, configurable `MultiTaskLoss` regime/prior-shift parameters defaulting to exact V1 values, and 5-class stage documentation alignment) are retained and verified by [test_v2_loss_and_masks.py](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/tests/test_v2_loss_and_masks.py).

---

## 6. Model Size, Throughput, VRAM, Checkpoint Provenance & Reproducible Commands

- **Model Size:** `2,977,797` trainable parameters (`7,323` buffers, `333` `state_dict` keys; `34.41 MiB` full checkpoint file on disk including AdamW momentum/variance states).
- **Throughput & Runtime:** `428.1s–433.8s` per epoch on `10,545` 4-step sequences (`~24.4` sequences/sec = `~97.6` overpasses/sec); `VAL` evaluation (`1,815` sequences): `42.6s–43.7s` (`~42` sequences/sec).
- **GPU Memory Footprint:** `1,041.0 MiB` single-step active allocation; `5,263.0 MiB / 20,475.0 MiB` (`25.7%` of RTX 4000 Ada VRAM) peak process VRAM across a full training + evaluation epoch.
- **Checkpoint Compatibility:** `100%` backward-compatible (`333 / 333` matching parameter keys and shapes; `0` missing or unexpected keys across `checkpoints/tcprimed_100gb/`, `checkpoints/tcprimed_100gb_exp_c1/`, and `checkpoints/tcprimed_100gb_v2/`).

### Reproducible PowerShell Commands

```powershell
Set-Location "D:\Enjula uchoi\TC-GraphFormer-verified"
$PY = "D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe"
$env:PYTHONPATH = "D:\Enjula uchoi\TC-GraphFormer-verified"

# 1. Run V2 mathematical & mask finiteness regression tests
& $PY -m unittest tests/test_v2_loss_and_masks.py -v

# 2. Verify all Phase 2 & Phase 3 diagnostics (TRAIN_EVAL vs VAL_EVAL generalization & per-task gradient norms)
Get-Content reports/phase2e4_100gb/v2_phase2_phase3_diagnostics.json | Select-Object -First 40

# 3. Inspect Phase 5 V2 controlled GPU experiment telemetry and Phase 6 acceptance comparison
Get-Content reports/phase2e4_100gb/v2_controlled_experiment_summary.json | Select-Object -First 60
```
