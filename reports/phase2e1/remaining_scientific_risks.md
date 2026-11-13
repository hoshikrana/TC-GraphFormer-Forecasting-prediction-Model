# TC PRIMED Phase 2E.1 Remaining Scientific & Operational Risks Report
**Phase:** 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Correction  
**Repository:** `hoshikrana/TC-GraphFormer-Forecasting-prediction-Model`  
**Branch:** `fix/phase2e-scientific-integrity`  
**Base Commit:** `e1b79ac25622b99c21ae0ebcf98283983706985a`  
**Status:** Software data-to-model integration verified; operational/scientific scaling risks documented below.

---

## 1. Executive Distinction: Integration Readiness vs. Meteorological Validation

Phase 2E.1 has established complete, verified, defect-free software integration from raw NetCDF files through data loaders, coordinate resampling, spatial polar graphs, causal temporal attention, multi-task losses, and gradient backpropagation.

However, **software integration must not be conflated with trained meteorological forecast skill**. This report details the genuine scientific, data, and operational risks that remain prior to full-scale training on the 2000–2025 dataset.

---

## 2. Scientific & Statistical Risks

### 2.1 Pilot Dataset Sample Limitation (80 Storms, 2,599 Overpasses)
- **Current State:** The 20 GB local pilot contains 80 storms across 6 global basins.
- **Risk:** The TC-GraphFormer architecture contains $\sim 1.2\text{M}$ parameters ($d_{\text{model}} = 128$). Training high-capacity attention networks on 1,742 training samples will rapidly lead to empirical overfitting without massive data scaling.
- **Mitigation:** The verified two-machine staging framework is designed specifically to scale to the full 2000–2025 dataset ($\sim 500\text{ GB}$, $\sim 1,500\text{ storms}$, $> 100,000\text{ overpasses}$) on Machine B (remote GPU node).

### 2.2 Extreme Class Imbalance in Rapid Intensification ($\Delta v \ge 30\text{ kt} / 24\text{h}$)
- **Current State:** Rapid intensification (RI) occurs in fewer than $8\%$ of 24-hour observation windows.
- **Risk:** A standard cross-entropy or BCE loss without extreme positive weighting will converge to predicting "no RI", achieving high nominal accuracy while missing all critical intensification events.
- **Mitigation:** MultiTaskLoss already supports designated RI loss weighting (`weight_ri = 2.0`). For full-scale training, focal loss ($\gamma = 2.0$) or class-balanced re-weighting should be enabled.

### 2.3 Extreme Modality Sparsity in Spaceborne Precipitation Radar
- **Current State:** In the pilot dataset, radar is present in exactly 76 overpasses ($2.9\%$ of samples) because radar instruments only fly on GPM and TRMM core observatories.
- **Risk:** While the mask-aware cross-modal fusion cleanly routes around missing modalities, the `RadarEncoder` parameters receive gradient updates on only $3\%$ of batches. In pretraining or multi-task training, radar encoder features may train much slower than IR or PMW encoders.
- **Mitigation:** Employ modal-balanced batching during self-supervised pretraining (oversampling radar-bearing overpasses for MFM pretraining batches).

### 2.4 Sub-Grid Convective Scale Discrepancies
- **Current State:** The canonical reference grid has cell size $\Delta x = 15.625\text{ km}$ ($64 \times 64$ across $1,000\text{ km}$).
- **Risk:** Infrared imagery has native resolution $\approx 4\text{ km}$. Resampling IR to $15.625\text{ km}$ acts as a spatial low-pass filter, attenuating fine-scale convective cloud-top brightness temperature minimums and overshooting cloud tops.
- **Mitigation:** The $15.625\text{ km}$ grid was chosen to match the physical resolution of the lowest-frequency passive microwave channels ($19\text{ GHz}$ footprint is $\sim 30\text{ km}$). In future phases, a dual-resolution encoder (high-res IR branch at $256 \times 256$ coupled to multi-spectral PMW at $64 \times 64$) may be evaluated.

---

## 3. Operational & Data-Pipeline Risks

### 3.1 Reanalysis Latency in Real-Time Operational Forecasting
- **Current State:** The causal environmental synoptic branch uses ERA5 reanalysis fields (`precipitable_water`, `sst`, `msl`, `t2m`, `u10`, `v10`).
- **Risk:** Operational ERA5 final reanalysis has an approximate 5-day latency; ERA5T (preliminary) has a 5-day to 18-hour latency. In real-time operational deployment, the model cannot access contemporaneous ERA5 fields.
- **Mitigation:** In operational deployment, GFS or ECMWF IFS deterministic numerical weather prediction analysis/forecast fields must be substituted for ERA5, requiring domain shift verification between GFS and ERA5.

### 3.2 Forecasting Horizons Beyond 24 Hours
- **Current State:** Multi-task targets predict intensity change and track displacement up to $+24\text{ hours}$ ($+6\text{h}, +12\text{h}, +18\text{h}, +24\text{h}$).
- **Risk:** Beyond 24 hours, internal convective vortex structure becomes secondary to large-scale environmental steering (subtropical ridge positioning, upper-level trough interactions, baroclinic shear). Satellite imagery alone cannot sustain 48–72h track forecasts without continuous coupled numerical guidance.
- **Mitigation:** Keep the model's primary forecasting horizon focused on the critical $0 - 24\text{ hour}$ short-term rapid intensification and eyewall structural evolution window, where satellite observations provide maximum physical value over NWP.
