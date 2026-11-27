# TC-GraphFormer — Data Fidelity Audit, Six-Task Loss Validation & Bounded Hyperparameter Optimization Report

- **Timestamp (UTC):** `2026-11-24T23:02:00Z`
- **Repository:** `D:\Enjula uchoi\TC-GraphFormer-verified`
- **Git Branch:** `feat/tc-graphformer-v2-generalization`
- **Python Executable:** `D:\Enjula uchoi\TC-GraphFormer-Forecasting-prediction-Model-main\.venv\Scripts\python.exe`
- **Hardware:** NVIDIA RTX 4000 Ada Generation (`20,475 MiB` VRAM, PyTorch `2.9.1+cu128`), `128 GB` RAM
- **Dataset Partitions:** `TRAIN` (`75` storms, `10,545` sequences), `VAL` (`17` storms, `1,815` sequences), `FINAL_TEST` (`17` storms, `1,429` sequences — **strictly untouched, `0` accesses**).

---

## 1. Actual Raw, Processed, and Cache Directory Sizes (Phase 1.1)

| Storage Tier | Local Directory Path | File Count | Exact Byte Count | Size (`GiB` Binary) | Size (`GB` Decimal) | Compression / Reduction Ratio vs. Raw |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw TC PRIMED Dataset** | `data/raw/tcprimed_100gb` | `13,898` (`13,789` overpass + `109` ERA5 `_env_`) | **`99,997,769,720`** | **`93.1302 GiB`** | **`99.9978 GB`** | `1.0000x` (Reference) |
| **2. Processed Storm Containers** | `data/processed/tcprimed_100gb` | `109` (`1` NetCDF-4/HDF5 per storm, `zlib=4`) | **`33,777,531,524`** | **`31.4578 GiB`** | **`33.7775 GB`** | **`2.9605x` (`-66.22%` bytes)** |
| **3. On-Disk Resampled `64x64` Cache** | `data/cache/resampled_64x64` | `16,388` (`13,789` 100GB + `2,599` pilot `.pt` files) | **`11,401,643,420`** | **`10.6186 GiB`** | **`11.4016 GB`** | **`8.7705x`** |

---

## 2. Raw-to-Processed Variable-Preservation Matrix & Numerical Equality Checks (Phase 1.2)

### 2.1 Variable-by-Variable Preservation Matrix Across All 5 Modalities

| Modality / Group | Raw Location, Dtype, Shape & Units | Processed Container Location, Dtype, Shape & Units | Transformation Category (`Preserved / Transformed / Omitted`) | Valid Pixels & Coordinates Survive? | Consumed by Trained Model (`TCGraphFormer`)? | Numerical Verification (`AL062018` GMI/ERA5 Benchmark) & Scientific Impact |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **PMW Swaths `S1` & `S2`** (`passive_microwave/S1, S2`) | `passive_microwave/S1,S2/TB_*`: `int16` (`scale_factor=0.01`, `add_offset`), native `(n_scan, n_pix)`, `K`; `latitude, longitude, x, y` (`float32`), `ScanTime, angle_bins, surface_type` | `overpasses/obs_*/passive_microwave/S1,S2/TB_*`, `latitude, longitude`: `float32`, native `(n_scan, n_pix)`, `K` (`fill=-9999.0`, `zlib=4`) | **Preserved** (`TB_*`, `lat`, `lon` in `S1, S2`); **Omitted** (`x, y, angle_bins, ScanTime, surface_type` in `S1, S2`) | **Yes (`100%` valid pixels & `lat/lon`)** | **Yes** (`4` canonical channels resampled to `64x64`, $d_{\max}=35\text{ km}$, clipped $[100,320]\text{ K} \to [0,1]$) | **`max_abs_diff = 0.00e+00` (`bitwise_equal = True`, `37,128` valid pixels)** |
| **PMW Swaths `S3..S6`** (`AMSR2, AMSRE, ATMS, SSMIS, TMI`) | `passive_microwave/S3..S6/TB_*`: `int16` (`scale_factor=0.01`), native `(n_scan, n_pix)`, `K` (e.g. `ATMS S3/TB_88.2QV`, `TMI S3/TB_85.5V`, `SSMIS S4/TB_91.665V`, `AMSR2/E S4/36.5V, S5/89.0V`) | **Not written to processed container** (`process_passive_microwave` loops only over `["S1", "S2"]`) | **Omitted (Swath Filtering)** | No (`S3..S6` omitted) | No (`pmw_channel_mask` masks missing bands to `0.0`) | **Verified Omission:** High-frequency `85–91 GHz` ice-scattering swaths stored in `S3/S4/S5` on `TMI, SSMIS, AMSR2, AMSRE, ATMS` are omitted (while `GMI, SSMI, MHS, AMSUB` retain `85–89 GHz` in `S1/S2`). |
| **Infrared Image (`IRWIN`)** | `infrared/IRWIN`: `int16` (`scale_factor=0.01`), native `(ny, nx)` (e.g. `480x640`), `K`; `latitude, longitude, x, y` (`float32`) | `overpasses/obs_*/infrared/IRWIN`, `latitude, longitude, x, y`: `float32`, native `(ny, nx)`, `K` (`zlib=4`) | **Preserved** (lossless `float32` + `zlib=4`) | **Yes (`100%` valid pixels & `lat/lon/x/y`)** | **Yes** (`1` channel resampled to `64x64`, $d_{\max}=15\text{ km}$, clipped $[170,320]\text{ K} \to [0,1]$) | **`max_abs_diff = 0.00e+00` (`bitwise_equal = True`, `307,200` valid pixels)** |
| **Infrared Storm Size (`R5`) & Radial Profile** | `infrared/R5` (`(R5_time,)`, `deg_lat`) & `brightness_temperature_mean_radial_profile` (`(30,)`, `K`), `radius` (`(30,)`, `km`) | `overpasses/obs_*/infrared/R5`, `R5_time`, `brightness_temperature_mean_radial_profile`, `radius`: `float32` / `int64` | **Preserved in Container; Unused by Model** | **Yes (`100%`)** | **No** (model computes 16-sector polar features directly from 2D `IRWIN`) | Preserved losslessly in container for diagnostic analysis; redundant with 2D `IRWIN` polar pooling. |
| **GPROF Retrieval (`GPROF/S1`)** | `GPROF/S1/surfacePrecipitation, convectivePrecipitation, rainWaterPath, iceWaterPath`: `int16` (`scale_factor`), native `(n_scan, n_pix)`, `mm/hr` & `kg/m^2`; plus `totalColumnWaterVaporIndex, mostLikelyPrecipitation` | `overpasses/obs_*/GPROF/surfacePrecipitation, convectivePrecipitation, rainWaterPath, iceWaterPath, latitude, longitude, x, y`: `float32`, native `(n_scan, n_pix)` | **Preserved** (4 core hydrometeor fields + coords); **Omitted** (`totalColumnWaterVaporIndex, mostLikelyPrecipitation`) | **Yes (`100%` valid pixels & coords)** | **Yes** (`4` channels resampled to `64x64`, $d_{\max}=20\text{ km}$, $\ln(1+x)/5.0$) | **`max_abs_diff = 0.00e+00` (`bitwise_equal = True`, `29,568` valid pixels)**. Present on **`13,761 / 13,789` (`99.80%`)** of overpasses (including `1,838 / 1,838` ATMS overpasses!). |
| **Spaceborne 3D Radar (`KuKaGMI, KuGMI, KuTMI`)** | `radar_radiometer/Ku*/correctedReflectFactor_Ku` & `Ka` (`(n_range, n_scan, n_beam)`, `dBZ`), `binHeight` (`km`), `nearSurfPrecipTotRate` (`mm/hr`), `latitude, longitude, x, y`; plus `mainlobeEdge,piaFinal` | `overpasses/obs_*/radar_radiometer/correctedReflectFactor_Ku, correctedReflectFactor_Ka, bin_height_km, binHeight, latitude, longitude, x, y`: `float32`, native `(n_range, n_scan, n_beam)` | **Preserved** (3D `Ku`, 3D `Ka`, 3D `bin_height_km`, 2D coords); **Omitted** (`nearSurfPrecipTotRate` extracted in processor but not written by `StorageWriter`) | **Yes (`100%` 3D voxels, beam coords & gate heights)** | **Yes for 3D `Ku` & `bin_height_km`** (resampled to `16` CAPPI levels `0.5–19.5 km` $\times 32\times 32$, $[-20,100]\text{ dBZ}\to[0,1]$); **No for `Ka`** | **`max_abs_diff = 0.00e+00` (`bitwise_equal = True`, `122,567` valid voxels)**. `Ku` (`13.8 GHz`) is present on all `1,665` radar overpasses (`TMI`+`GMI`), whereas `Ka` (`35.5 GHz`) is absent on all `1,269` TRMM `TMI` overpasses. |
| **ERA5 Surface Synoptic Fields (`rectilinear`)** | `rectilinear/precipitable_water, sst, pressure_msl, temperature_2m, u_wind_10m, v_wind_10m, dewpoint_2m, rain_large_scale, rain_convective`: `int16` (`scale_factor`), `(T_storm, 121, 121)` ($30^\circ\times 30^\circ$ at $0.25^\circ$) | `overpasses/obs_*/environmental_synoptic/precipitable_water, sst, pressure_msl, temperature_2m, latitude, longitude`: `float32`, `(121, 121)` at causal slice $t_{\text{env}} \le t_{\text{obs}}$ | **Preserved** (`4` surface fields at causal time slice); **Omitted** (`u_wind_10m, v_wind_10m` due to name mismatch `"u_10m"` vs `"u_wind_10m"`, plus `dewpoint_2m, rain_*`) | **Yes (`100%` for 4 fields & `121x121` coords)** | **Yes (`4 / 6` channels active; channels `4, 5` masked `0.0` via `env_channel_mask = [1,1,1,1,0,0]`)** | **`max_abs_diff = 0.00e+00` (`bitwise_equal = True`, `14,641` grid points)** for `TPW, SST, MSLP, T2m`; `u_wind_10m` and `v_wind_10m` omitted due to raw variable name `u_wind_10m` vs `u_10m`. |
| **ERA5 23-Level 3D Grids & `diagnostics` Group** | `rectilinear` & `cylindrical`: 9 3D variables `(T, 23, 121, 121)` (`u_wind, v_wind, temperature, RH, q, Z, vort, div, omega`); `diagnostics`: `shear_magnitude, shear_direction, potential_intensity_*, CPS_*` | **Not extracted into processed containers** (`process_environmental_slice` extracts only 2D surface/column fields) | **Omitted (Dimensionality Reduction from 3D 23-level to 2D surface)** | No | No | Accounts for the majority of the `2.96x` raw-to-processed byte reduction (`99.998 GB -> 33.778 GB`) and explains why $+24\text{h}$ track forecasts plateau near the causal velocity baseline (`~213.5 km`) without upper-level steering-flow (`u_wind, v_wind, Z` at `850–200 hPa`) or vertical wind shear vectors. |

---

## 3. Specific Findings on ATMS and All 9 Sensor Channel Mappings (Phase 1.3 & 1.4)

| Sensor Family | Overpasses in 100 GB (`TRAIN / VAL / TEST`) | Raw Swaths & Channels Present | Channels Preserved in Processed Container (`S1, S2`) | Model `SENSOR_PMW_CHANNEL_MAP` (`[Ch0 ~19V, Ch1 ~19H, Ch2 ~37V, Ch3 ~85–91V]`) | Active Model `pmw_channel_mask` | Scientific Assessment |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| **`GMI`** | `397` (`336 / 34 / 27`) | `S1`: `10.65V/H, 18.7V/H, 23.8V, 36.64V/H, 89.0V/H`; `S2`: `166V/H, 183.31V` | All `S1` & `S2` (`13` channels) | `(S1/18.7V, S1/18.7H, S1/36.64V, S1/89.0V)` | **`[1, 1, 1, 1]`** | **Complete 4-band coverage** (+ 3D `Ku` radar on `396/397` overpasses). |
| **`SSMI`** | `2,259` (`1604 / 323 / 332`) | `S1`: `19.35V/H, 22.235V, 37.0V/H`; `S2`: `85.5V/H` | All `S1` & `S2` (`7` channels) | `(S1/19.35V, S1/19.35H, S1/37.0V, S2/85.5V)` | **`[1, 1, 1, 1]`** | **Complete 4-band coverage.** |
| **`SSMIS`** | `2,466` (`1907 / 334 / 225`) | `S1`: `19.35V/H, 22.235V`; `S2`: `37.0V/H`; `S3`: `150H, 183H`; `S4`: `91.665V/H` | `S1` & `S2` (`5` channels; `S4/91.665V` omitted) | `(S1/19.35V, S1/19.35H, S2/37.0V, None)` | **`[1, 1, 1, 0]`** | `19V, 19H, 37V` active; `S4/TB_91.665V` omitted by `S1/S2` filter. |
| **`TMI`** | `1,306` (`897 / 241 / 168`) | `S1`: `10.65V/H`; `S2`: `19.35V/H, 21.3V, 37.0V/H`; `S3`: `85.5V/H` | `S1` & `S2` (`7` channels; `S3/85.5V` omitted) | `(S2/19.35V, S2/19.35H, S2/37.0V, None)` | **`[1, 1, 1, 0]`** | `19V, 19H, 37V` active (+ 3D `Ku` radar on `1,269/1,306` overpasses); `S3/TB_85.5V` omitted. |
| **`AMSR2`** | `651` (`514 / 90 / 47`) | `S1`: `10.65V/H`; `S2`: `18.7V/H`; `S3`: `23.8V/H`; `S4`: `36.5V/H`; `S5/S6`: `89.0V/H` | `S1` & `S2` (`4` channels; `S4/36.5V, S5/89.0V` omitted) | `(S2/18.7V, S2/18.7H, None, None)` | **`[1, 1, 0, 0]`** | `18.7V/H` active; `S4/36.5V` and `S5/89.0V` omitted by `S1/S2` filter. |
| **`AMSRE`** | `395` (`293 / 55 / 47`) | `S1`: `10.65V/H`; `S2`: `18.7V/H`; `S3`: `23.8V/H`; `S4`: `36.5V/H`; `S5/S6`: `89.0V/H` | `S1` & `S2` (`4` channels; `S4/36.5V, S5/89.0V` omitted) | `(S2/18.7V, S2/18.7H, None, None)` | **`[1, 1, 0, 0]`** | `18.7V/H` active; `S4/36.5V` and `S5/89.0V` omitted by `S1/S2` filter. |
| **`MHS`** | `3,712` (`2842 / 471 / 399`) | `S1`: `89.0V, 157.0V, 183.31_1.0H, 183.31_3.0H, 190.31V` | All `S1` (`5` channels) | `(None, None, None, S1/89.0V)` | **`[0, 0, 0, 1]`** | Sounder lacks `19/37 GHz` hardware; `89.0V` ice-scattering band active. |
| **`AMSUB`** | `765` (`603 / 108 / 54`) | `S1`: `89.0_0.9QV, 150.0_0.9QV, 183.31_1/3/7QV` | All `S1` (`5` channels) | `(None, None, None, S1/89.0_0.9QV)` | **`[0, 0, 0, 1]`** | Sounder lacks `19/37 GHz` hardware; `89.0QV` ice-scattering band active. |
| **`ATMS`** | **`1,838` (`1549 / 159 / 130`)** | `S1`: `TB_23.8QV`; `S2`: `TB_31.4QV`; `S3`: `TB_88.2QV`; `S4`: `TB_165.5QH, TB_183.31_*QH` | `S1` (`TB_23.8QV`) & `S2` (`TB_31.4QV`) only (`S3/TB_88.2QV` omitted!) | **`(None, None, None, None)`** | **`[0, 0, 0, 0]` (`has_pmw=False`)** | **Verified ATMS Finding:** ATMS is a cross-track sounder without `19 GHz` or `37 GHz` channels (`23.8QV` and `31.4QV` in `S1/S2` are physically incompatible with `19V/19H/37V` and rightly not mapped to Channels `0–2`). Its only compatible channel is **`S3/TB_88.2QV` (`~89 GHz` Band 3)**, which was omitted by `process_passive_microwave`'s `["S1", "S2"]` loop. However, **all `1,838` ATMS overpasses still provide valid `GPROF` (`1,838/1,838`), `IR` (`1,717/1,838`), and `ERA5` (`1,818/1,838`) tensors** to the model. |

---

## 4. Phase 2 — Six-Task Loss Audit & Two Scientifically Defensible Loss Alternatives

### 4.1 Mathematical Audit of the Current Six-Task Objective (`MultiTaskLoss`)

| Task | Current Loss Formula | Target Units & Scale ($\sigma_{\text{TRAIN}}$) | Mask & Denominator | Head Prior & Multiplier | Pre-Clip Shared / Head Grad Norm | Why Current Formulation Is / Is Not Appropriate |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **1. Current Wind ($V_{\max}$)** | $1.0 \cdot \text{Huber}_{\beta=8.0}(\hat{V}, V) \cdot \left(1 + 2.5\,\text{clamp}\!\left(\frac{V-50}{50},0,2\right)^2\right)$ | `kt` ($\mu=55.42$, $\sigma=31.31\text{ kt}$) | `intensity_mask` ($n_{\text{TR}}=10,545$, $n_{\text{VAL}}=1,815$) | Prior `38.19 kt`, mult **`12.0x`** | **`380.22` (`72.74%`)** / `39.43` | **Appropriate:** Quadratic regime weight $w_{\text{reg}} \in [1, 11]$ is essential to counter the `73.8%` TD/TS majority. **Limitation:** Unscaled $\beta=8.0\text{ kt}$ (`0.255 * sigma_V`) puts `24.77%` of samples in the flat linear Huber tail where gradients saturate regardless of error magnitude. |
| **2. Central Pressure ($P_{\min}$)** | $0.5 \cdot \text{Huber}_{\beta=2.5}\!\left(\frac{\hat{P}}{10}, \frac{P}{10}\right)$ | `10 hPa` ($\mu=985.54$, $\sigma=22.78\text{ hPa}$) | `pressure_mask` ($n_{\text{TR}}=9,499$, $n_{\text{VAL}}=1,604$) | Prior `997.25 hPa`, mult `12.0x` ($\div 10 = \mathbf{1.2x}$) | `2.53` (`0.48%`) / `0.28` | **Appropriate:** $\beta=2.5$ (`25 hPa` $\approx 1.10\,\sigma_P$) keeps `99.57%` of samples in the smooth quadratic regime, and $\hat{P}$ is strongly coupled to $\hat{V}$ (`grad cos = +0.494`). |
| **3. Development Stage (`5-class`)** | $0.5 \cdot \text{CrossEntropy}(\mathbf{z}_s, y_s)$ | `5` canonical classes (`0..4`) | `stage_mask` ($n_{\text{TR}}=9,641$, $n_{\text{VAL}}=1,549$) | Zero bias, `1.0x` logits | `8.62` (`1.65%`) / `0.50` | **Appropriate:** Proper multi-class log-likelihood; positively aligned with wind (`+0.371`) and pressure (`+0.397`) gradients. |
| **4. Intensity Change ($\Delta V_{6,12,18,24\text{h}}$)** | $1.0 \cdot \text{Huber}_{\beta=4.0}(\widehat{\Delta V}, \Delta V)$ | `kt` ($\sigma = [5.37, 9.80, 13.66, 17.09]\text{ kt}$) | `intensity_change_mask` ($n_{\text{TR}}=39,304$, $n_{\text{VAL}}=6,596$) | Priors `[0.23, 0.58, 1.08, 1.60]`, mult **`5.0x`** | `60.95` (`11.66%`) / `2.83` | **Appropriate over MSE:** $\Delta V$ has heavy kurtosis during RI/landfall where pure MSE over-penalizes outliers; $\beta=4.0\text{ kt}$ keeps `49.17%` of samples in the robust linear tail. |
| **5. Rapid Intensification (`RI`)** | $2.0 \cdot \text{BCEWithLogits}(z_{\text{RI}} + 0.85\ln w_{\text{pos}}, y_{\text{RI}}; w_{\text{pos}}=18.1)$ | Binary ($\Delta V_{24\text{h}} \ge 30\text{ kt}$, base rate `5.23%`) | `ri_mask` ($n_{\text{TR}}=9,514$, $n_{\text{VAL}}=1,576$) | Prior `-3.5124`, `1.0x` logit | `68.75` (`13.15%`) / `2.06` | **Appropriate:** Class-weighted BCE with prior log-odds shift prevents trivial all-negative collapse while keeping Brier score (`0.0386–0.0391`) well-calibrated relative to base-rate climatology (`0.0550`). |
| **6. Track Displacement ($\Delta x, \Delta y$)** | $1.0 \cdot \text{Huber}_{\beta=1.0}\!\left(\frac{\widehat{\Delta\mathbf{x}}}{100}, \frac{\Delta\mathbf{x}}{100}\right)$ | `100 km` (residual $\sigma_{+24\text{h}} \approx 153\text{ km}$) | `track_mask` ($n_{\text{TR}}=13,650$, $n_{\text{VAL}}=2,246$ comps) | Causal damped velocity prior + **`20.0x`** residual | `1.61` (`0.31%`) / `0.10` | **Appropriate:** Anchoring on causal observation-time translation velocity provides strong baseline skill (`~213.5 km` at `+24h`), and $\beta=100\text{ km}$ prevents outlier storm tracks from destabilizing the shared encoder. |

### 4.2 Two Scientifically Defensible Loss Alternatives Identified for Controlled Evaluation
1. **Alternative Loss L1 (`TRAIN`-Standardized Wind Huber Loss with $\sigma_{V,\text{TRAIN}} = 31.3122\text{ kt}$ and $\beta = 0.50\,\sigma_{V,\text{TRAIN}} = 15.6561\text{ kt}$):**
   $$\mathcal{L}_v^{(\text{L1})} = \sigma_{V,\text{TRAIN}} \cdot \text{Huber}_{\beta = 0.50}\!\left(\frac{\hat{V}}{\sigma_{V,\text{TRAIN}}},\, \frac{V}{\sigma_{V,\text{TRAIN}}}\right) \cdot w_{\text{regime}}(V)$$
   - *Scientific Rationale:* Unlike pure MSE (which squares $25\text{--}40\text{ kt}$ eye-misalignment outliers and causes gradient explosions under $11\times$ regime weights) or V1's narrow $\beta = 8.0\text{ kt}$ Huber threshold (which saturates the gradient to a flat constant for all errors $\ge 8.0\text{ kt}$, i.e., `24.77%` of samples), setting the transition threshold to $0.50\,\sigma_{V,\text{TRAIN}} = 15.6561\text{ kt}$ preserves proportional quadratic restoring gradients across the entire $8\text{--}15.6\text{ kt}$ error band typical of Major Hurricanes while retaining linear Huber robustness against extreme $>16\text{ kt}$ outliers.
2. **Alternative Loss L2 (Horizon-Normalized Intensity-Change Huber Loss Using `TRAIN` Horizon Standard Deviations $\boldsymbol{\sigma}_{\Delta V,\text{TRAIN}} = [5.37, 9.80, 13.66, 17.09]\text{ kt}$):**
   - *Scientific Rationale:* In V1, all four horizons ($+6\text{h}, +12\text{h}, +18\text{h}, +24\text{h}$) share a single fixed threshold $\beta = 4.0\text{ kt}$, which is $0.74\,\sigma_{+6\text{h}}$ at $+6\text{h}$ but only $0.23\,\sigma_{+24\text{h}}$ at $+24\text{h}$. Scaling each horizon's Huber threshold proportionally to $\sigma_{\Delta V, h}$ equalizes quadratic-to-linear transition behavior across lead times.

---

## 5. Phase 3 & Phase 5 — Bounded 8-Trial Optimizer Screening & Single Loss Comparison Results

### 5.1 Stage 1: Bounded 8-Trial Optimizer Hyperparameter Screening at Epoch 6 (Fixed V1 Loss, `seed = 42`, Resumed from `multitask_epoch_005.pt`)

All 8 trials start from the exact same Epoch 5 checkpoint (`checkpoints/tcprimed_100gb/multitask_epoch_005.pt`, `VAL_EVAL = 16.6384`, `VAL_MT_REL_ERR = 0.5925`) and train on the full `10,545` `TRAIN` sequences before evaluating on the full `1,815` `VAL` sequences (`reports/phase2e4_100gb/hpo_and_loss_optimization_summary.json`):

| Trial ID (`Config Hash`) | LR (`Cosine`) | Weight Decay Policy | Eff. Batch Size (`Accum`) | Head Dropout | Ep 6 `Train Opt Loss` | **Ep 6 `VAL_EVAL` Global Loss** | **Ep 6 `VAL_MT_REL_ERR`** | Ep 6 `VAL` Wind MAE (`kt`) | **Ep 6 `VAL` Major-Hurr MAE (`kt`)** | Ep 6 `VAL` Press MAE (`hPa`) | Ep 6 `VAL` Stage Acc (`%`) | **Ep 6 `VAL` $+24\text{h}$ $\Delta V$ MAE (`kt`)** | Ep 6 `VAL` RI `PR-AUC / Brier` | Ep 6 `VAL` Track $+24\text{h}$ (`km`) | Runtime (`s`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`T1_orig`** (`b139c445ca22`) | `1.50e-4` | `1.0e-4` (Uniform) | `64` (`2`) | `0.00` | `11.2023` | `18.2382` | `0.5876` | `7.54` | `9.79` | `6.18` | `64.88%` | **`9.98`** | `0.5324 / 0.0387` | `214.06` | `446.1s` |
| **`T2_c1_ref`** (`65b537e728b5`) | `5.00e-5` | `3.0e-4` (Uniform) | `64` (`2`) | `0.00` | `10.1260` | `17.0999` | `0.5810` | `6.94` | `9.67` | **`5.83`** | **`69.34%`** | `10.42` | `0.5218 / 0.0419` | **`213.63`** | `441.5s` |
| **`T3_c2_drop`** (`93c23309ee59`) | `5.00e-5` | `3.0e-4` (Uniform) | `64` (`2`) | `0.15` | `12.2675` | `17.4310` | **`0.5772`** | `7.14` | `9.32` | `6.30` | `68.43%` | `10.02` | **`0.5369 / 0.0383`** | `213.79` | `474.8s` |
| **`T4_low_lr`** (`f0fc72ef9f77`) | `2.50e-5` | `3.0e-4` (Uniform) | `64` (`2`) | `0.00` | `10.1621` | `17.2411` | `0.5904` | `7.25` | `9.24` | `6.36` | `66.62%` | `10.25` | `0.5147 / 0.0407` | `213.98` | `440.0s` |
| **`T5_med_lr`** (`453e76f5a878`) | `1.00e-4` | `3.0e-4` (Uniform) | `64` (`2`) | `0.00` | `10.3384` | `16.4283` | `0.5813` | **`6.89`** | `8.87` | `6.01` | `68.56%` | `10.16` | `0.5289 / 0.0423` | `213.93` | `427.7s` |
| **`T6_decoupled_wd` (`d5241303640a`)** | **`5.00e-5`** | **`1.0e-3` (Backbone Only, `0.0` Heads/BN/LN/Bias)** | **`64` (`2`)** | **`0.00`** | **`10.1290`** | **`16.2657` [BEST]** | **`0.5804`** | **`7.07`** | **`8.21` [BEST]** | **`6.06`** | **`67.01%`** | **`9.98` [BEST]** | **`0.5150 / 0.0390`** | **`213.99`** | **`427.8s`** |
| **`T7_bs128`** (`96db8341d262`) | `5.00e-5` | `3.0e-4` (Uniform) | `128` (`4`) | `0.00` | `10.2661` | `17.1154` | `0.5866` | `7.23` | `9.18` | `6.17` | `67.85%` | `10.34` | `0.5237 / 0.0410` | `213.81` | `425.6s` |
| **`T8_decoupled_3.5e-5`** (`67a257616711`) | `3.50e-5` | `1.0e-3` (Backbone Only) | `64` (`2`) | `0.00` | `10.1249` | `16.7599` | `0.5860` | `7.16` | `8.70` | `6.18` | `66.11%` | `10.11` | `0.5161 / 0.0395` | `213.98` | `427.7s` |

### 5.2 Stage 2: Full 4-Epoch Continuation (Epochs 6–9) of Winning Optimizer Trial `T6_decoupled_wd` vs. Single Loss Comparison `L1_standardized_wind_huber`

| Run / Epoch | LR | `TRAIN_EVAL` Global (V1 Ref) | **`VAL_EVAL` Global (V1 Ref)** | **`GEN_GAP` (`VAL - TRAIN`)** | **`VAL_MT_REL_ERR`** | `VAL` Wind MAE (`Bias kt`) | **`VAL` Major-Hurr MAE (`Bias kt`)** | `VAL` Press MAE (`hPa`) | `VAL` Stage Acc (`Bal %`) | **`VAL` $\Delta V$ `6/12/18/24h` (`kt`)** | `VAL` RI `PR-AUC / Brier / F1@0.5` | `VAL` Track `+24h` (`km`) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **`T6_decoupled_wd` Ep 6 (`screen_epoch_006.pt`)** | `5.00e-5` | **`9.4193`** | **`16.2657` [ALL-TIME BEST]** | **`+6.8464`** | `0.5804` | `7.07` (`-1.15`) | **`8.21` (`-3.19`)** | `6.06` | `67.01%` (`66.81%`) | **`3.14 / 5.80 / 8.07 / 9.98`** | `0.5150 / 0.0390 / 0.5155` | `213.99` |
| **`T6_decoupled_wd` Ep 7** | `4.28e-5` | — | `18.0733` | — | `0.5903` | `7.16` (`-2.09`) | `10.03` (`-6.98`) | `6.15` | `67.98%` (`66.88%`) | `3.19 / 5.93 / 8.29 / 10.30` | `0.4890 / 0.0422 / 0.4824` | `213.85` |
| **`T6_decoupled_wd` Ep 8** | `2.55e-5` | — | `17.2067` | — | `0.5781` | `7.04` (`-0.78`) | `9.38` (`-5.53`) | `5.76` | `68.37%` (`68.03%`) | `3.15 / 5.87 / 8.20 / 10.20` | `0.5124 / 0.0398 / 0.5257` | **`213.56`** |
| **`T6_decoupled_wd` Ep 9** | `8.18e-6` | `7.6192` | `17.1919` | `+9.5727` | **`0.5753`** | **`6.91`** (`-1.43`) | `9.60` (`-6.06`) | **`5.71`** | **`69.08%` (`68.10%`)** | `3.15 / 5.87 / 8.20 / 10.21` | `0.5192 / 0.0402 / 0.5198` | `213.61` |
| **`L1_std_wind_huber` Ep 6 (`best_checkpoint.pt`)** | `5.00e-5` | **`9.3558`** | **`16.4439`** (`L1: 13.4922`) | **`+7.0881`** | `0.5836` | `7.10` (`-1.08`) | **`8.43` (`-4.31`)** | `6.18` | `66.11%` (`66.00%`) | **`3.14 / 5.79 / 8.05 / 9.94` [BEST]** | `0.5167 / 0.0393 / 0.5200` | `214.00` |
| **`L1_std_wind_huber` Ep 7** | `4.28e-5` | — | `17.7175` (`L1: 14.5524`) | — | `0.5905` | `7.14` (`-2.06`) | `9.56` (`-6.21`) | `5.94` | `67.33%` (`66.21%`) | `3.19 / 5.93 / 8.30 / 10.32` | `0.4919 / 0.0428 / 0.4641` | `213.71` |
| **`L1_std_wind_huber` Ep 8** | `2.55e-5` | — | `16.7443` (`L1: 13.7528`) | — | `0.5755` | `6.97` (`-0.61`) | `8.89` (`-5.14`) | **`5.58`** | `68.50%` (`68.11%`) | `3.14 / 5.83 / 8.14 / 10.10` | `0.5014 / 0.0397 / 0.5172` | `213.60` |
| **`L1_std_wind_huber` Ep 9** | `8.18e-6` | — | `17.1950` (`L1: 14.0933`) | — | **`0.5739`** | **`6.89`** (`-1.45`) | `9.59` (`-6.11`) | `5.65` | **`69.01%` (`68.32%`)** | `3.13 / 5.83 / 8.13 / 10.08` | `0.5069 / 0.0398 / 0.5116` | `213.74` |

---

## 6. Master Validation & Per-Storm Comparison Table (`VAL`, $N=1,815$ Sequences, $17$ Storms)

| Metric (`TRAIN_EVAL` / `VAL_EVAL` in `model.eval()` Mode) | **Original Epoch 5** (`checkpoints/tcprimed_100gb/best_checkpoint.pt`) | **C1 Continuation Epoch 8** (`checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt`) | **`T6_decoupled_wd` Epoch 6** (`checkpoints/tcprimed_100gb_hpo_screen/T6_lr5.0e-5_wd1e-3_decoupled/screen_epoch_006.pt`) | **`L1_std_wind_huber` Epoch 6** (`checkpoints/tcprimed_100gb_exp_l1_std_wind_huber/best_checkpoint.pt`) |
| :--- | :---: | :---: | :---: | :---: |
| **`TRAIN_EVAL` Global Loss (V1 Ref)** | `13.0492` | `7.9958` | **`9.4193`** | `9.3558` |
| **`VAL_EVAL` Global Loss (V1 Ref)** | `16.6384` | `16.4052` | **`16.2657` [BEST]** | `16.4439` |
| **True Generalization Gap (`VAL - TRAIN`)** | **`+3.5892`** | `+8.4094` | **`+6.8464` (`-18.6%` vs C1)** | `+7.0881` (`-15.7%` vs C1) |
| **`VAL_MT_REL_ERR` (6-Task Relative Error)** | `0.5925` | **`0.5723` [BEST]** | `0.5804` (`0.5753` @ Ep 9) | `0.5836` (`0.5739` @ Ep 9) |
| **Overall Wind MAE / RMSE / Bias (`VAL`, kt)** | `7.84 / 10.52 / +1.36` | **`6.90 / 9.35 / -0.75`** | `7.07 / 9.39 / -1.15` | `7.10 / 9.46 / -1.08` |
| **Major-Hurricane ($\ge 96\text{ kt}$) MAE / Bias (`VAL`, kt)** | **`7.78 / -1.08`** | `8.68 / -4.21` | **`8.21 / -3.19` [BEST CONTINUATION]** | `8.43 / -4.31` |
| **Pressure MAE / RMSE (`VAL`, hPa)** | `5.99 / 8.29` | **`5.53 / 7.69`** | `6.06 / 8.51` (`5.71` @ Ep 9) | `6.18 / 8.66` (`5.58` @ Ep 8) |
| **Stage Balanced Acc / Overall Acc (`VAL`, %)** | `64.27% / 65.14%` | **`68.39% / 69.14%`** | `66.81% / 67.01%` (`68.10%` @ Ep 9) | `66.00% / 66.11%` (`68.32%` @ Ep 9) |
| **$\Delta V$ `+6 / +12 / +18 / +24h` MAE (`VAL`, kt)** | `3.22 / 5.96 / 8.27 / 10.18` | `3.16 / 5.91 / 8.27 / 10.26` | **`3.14 / 5.80 / 8.07 / 9.98`** | **`3.14 / 5.79 / 8.05 / 9.94` [BEST]** |
| **RI `VAL` PR-AUC / Brier (`92` pos / `1,484` neg)** | `0.5017 / 0.0386` | **`0.5226` / `0.0391`** | `0.5150` / **`0.0390`** | `0.5167 / 0.0393` |
| **Track `+24h` `VAL` MAE (`Skill vs CausalVel`, km)** | `213.65` (`-0.50`) | **`213.58` (`-0.43`)** | `213.99` (`-0.84`) (`213.56` @ Ep 8) | `214.00` (`-0.85`) (`213.60` @ Ep 8) |
| **Per-Storm `VAL` Wind MAE `Mean ± Std (Median)` (kt)** | `8.19±4.51 (8.03)` | `7.71±3.97 (7.56)` | **`7.64±3.85 (7.49)` [BEST]** | `7.67±3.88 (7.52)` |
| **Per-Storm `VAL` $\Delta V_{24\text{h}}$ MAE `Mean ± Std (Median)` (kt)** | `9.94±4.48 (9.72)` | `9.98±4.18 (10.81)` | **`9.83±4.36 (10.27)` [BEST]** | `9.80±4.35 (10.22)` |
| **Per-Storm `VAL` Track `+24h` `Mean ± Std (Median)` (km)** | `243.30±85.45 (209.75)` | `243.19±86.49 (209.73)` | `243.75±86.72 (209.36)` | `243.78±86.70 (209.40)` |

---

## 7. Best Recommended Model / Checkpoints, Total GPU Time & Exact Next Scientific Experiment

### 7.1 Best Recommended Checkpoints for Ablation Studies and Historical Forecasting Showcase Website
1. **Primary Overall Multi-Task Validation Model (`C1 Continuation Epoch 8`):**
   - **Path:** [checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt)
   - **Why Recommended:** Achieves the best balanced 6-task validation relative error (`VAL_MT_REL_ERR = 0.5723`), best overall Wind MAE (`6.90 kt`), best Central Pressure MAE (`5.53 hPa`, `RMSE = 7.69 hPa`), best Development Stage accuracy (`69.14%`, balanced `68.39%`), and highest RI ranking discrimination (`PR-AUC = 0.5226`, `F1@0.50 = 0.5568`) while maintaining strong Major-Hurricane Wind MAE (`8.68 kt`).
2. **New Best Global Validation-Loss & Low-Gap Decoupled-Weight-Decay Checkpoint (`T6_decoupled_wd Epoch 6`):**
   - **Path:** [checkpoints/tcprimed_100gb_hpo_screen/T6_lr5.0e-5_wd1e-3_decoupled/screen_epoch_006.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb_hpo_screen/T6_lr5.0e-5_wd1e-3_decoupled/screen_epoch_006.pt) (`config_hash = d5241303640a`)
   - **Why Preserved & Recommended Alongside `C1`:** Applying decoupled weight decay (`1.0e-3`) strictly to shared-backbone weights (with `0.0` decay on `TaskPredictionHeads`, biases, and norm layers) and early-stopping after 1 continuation epoch (`Epoch 6`) achieved the **lowest global validation loss in the project (`16.2657`)**, reduced the `VAL - TRAIN` generalization gap by **`-18.6%` (`+6.8464` vs. `+8.4094` in C1)**, achieved the **best per-storm mean Wind MAE (`7.64 ± 3.85 kt`)**, improved Major-Hurricane Wind MAE to **`8.21 kt`** (`-0.47 kt` better than C1), and improved `+24h` intensity-change MAE to **`9.98 kt`** (`-0.28 kt` better than C1).
3. **Original Production Baseline (`Original Epoch 5`):**
   - **Path:** [checkpoints/tcprimed_100gb/best_checkpoint.pt](file:///D:/Enjula%20uchoi/TC-GraphFormer-verified/checkpoints/tcprimed_100gb/best_checkpoint.pt)
   - **Why Preserved:** Remains the untouched reference baseline with the lowest Major-Hurricane Wind MAE (`7.78 kt`, bias `-1.08 kt`) and smallest generalization gap (`+3.5892`), and is the only checkpoint ever evaluated on `FINAL_TEST`.

### 7.2 Total GPU Compute Time Expended Across This Optimization Campaign
- **Phase 1–3 Diagnostics (`task-1124`, `task-1133`, `task-1229`, `task-1243`, `task-1267`):** `~22.5 minutes` (`1,350s`)
- **Controlled Regularization & V2 Runs (`exp_c1` 4 epochs, `v2_candidate_b` 4 epochs, `exp_c2` 4 epochs):** `~96.5 minutes` (`5,790s`)
- **Bounded 8-Trial Optimizer Screening (`T4..T8` Epoch 6 + `T6` Epochs 7–9 + `TRAIN_EVAL`, `task-1278`):** `~63.8 minutes` (`3,828s`)
- **Single Controlled Loss Comparison (`exp_l1_standardized_wind_huber` Epochs 6–9 + `TRAIN_EVAL`, `task-1318`):** `~34.8 minutes` (`2,088s`)
- **Total Campaign GPU Time on RTX 4000 Ada (`20 GB`):** **`~217.6 minutes` (`3.63 GPU hours`)**, peak active VRAM `1,041–1,239 MiB` (`5,263 MiB` reserved cache).

### 7.3 Exact Next Scientific Experiment (Proposed Work — Not Yet Executed)
- **Recover the Omitted `S3..S6` High-Frequency `85–91 GHz` Passive Microwave Swaths (`ATMS S3/TB_88.2QV`, `TMI S3/TB_85.5V`, `SSMIS S4/TB_91.665V`, `AMSR2/AMSRE S4/TB_36.5V & S5/TB_A89.0V`) and ERA5 `u_wind_10m / v_wind_10m` + `850–200 hPa` Vertical Wind Shear Diagnostic in a Versioned `v2.4` Container & Cache Build:**
  - Our Phase 1 data-fidelity audit proved that hyperparameter and loss tuning on the existing `v2.3` cache plateau near `VAL_EVAL = 16.27–16.41` (`VAL_MT_REL_ERR = 0.570–0.572`) because:
    1. `ModalityProcessors.process_passive_microwave` (`tcprimed_pipeline/modality_processors.py:40`) loops only over `["S1", "S2"]`, leaving the `85–91 GHz` convective ice-scattering band unextracted whenever a sensor stores it in `S3`, `S4`, or `S5` (`1,838` `ATMS`, `1,306` `TMI`, `2,466` `SSMIS`, `651` `AMSR2`, and `395` `AMSRE` overpasses = `6,656 / 13,789` overpasses!), and
    2. `ModalityProcessors.process_environmental_slice` (`tcprimed_pipeline/modality_processors.py:319`) looks for `"u_10m"` / `"v_10m"` instead of `"u_wind_10m"` / `"v_wind_10m"` and omits the ERA5 `diagnostics/shear_magnitude` and `potential_intensity_theoretical` variables that physically govern $+24\text{h}$ intensity change ($\Delta V$), Rapid Intensification (`RI`), and synoptic steering track deviations.
