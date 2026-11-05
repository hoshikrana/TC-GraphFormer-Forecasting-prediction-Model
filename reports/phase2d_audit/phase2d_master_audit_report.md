# TC PRIMED — Phase 2D: Comprehensive Master Repository, Data, Architecture and Research Integrity Audit

**Audit Phase**: Phase 2D (Comprehensive Read-Only Technical and Scientific Audit)  
**Dataset Under Audit**: NOAA TC PRIMED `v01r01-final` (20 GB Development Pilot, 2,679 NetCDF Files, 80 Storm Sequences)  
**Host Architecture**: Machine A (Friend's Laptop / Windows 11 AMD64 Workstation) & Machine B (Professor's Computer / Remote GPU Compute Node)  
**Audit Scope**: End-to-end provenance, raw NetCDF data, HDF5 containers, preprocessing pipeline, neural architecture, training workflows, checkpoints, baseline comparisons, two-machine staging, and research integrity.

---

## 1. Executive Assessment & Final Audit Decision

Based on comprehensive, read-only empirical auditing across all repository modules, tests, NetCDF granules, HDF5 containers, manifests, logs, and checkpoints, the audit panel renders the following definitive decision:

```
========================================================================================
FINAL AUDIT DECISION: READY FOR CORRECTIVE IMPLEMENTATION
========================================================================================
The foundational engineering of the project is scientifically defensible and robust:
  - The preprocessing pipeline (tcprimed_pipeline/) achieves 100% bitwise exact numerical 
    preservation (0.000000 max absolute error) and 13.7x principled volume reduction.
  - The TC-GraphFormer architecture is completely implemented, acyclic, and fully passes 
    forward, collate, and inference tests (107/107 unit/integration tests passing).
  - Temporal and environmental causality is strictly verified (100% t_env <= t_overpass).
  - Staged containers (82 files, 1.51 GB) match SHA-256 cryptographic hashes with 0 errors.

However, dataset scaling to 500 GB and production training CANNOT proceed until 
a targeted corrective phase resolves three critical infrastructure and experimental blockers:
  1. Internal disk space deficit (-538.4 GB shortage on Machine A internal drives).
  2. Severe test partition imbalance (0 Hurricane-stage overpasses and 0 North Atlantic storms).
  3. Disproven GPU training claims (previous runs executed locally on CPU for 1-2 epochs).

The project is fully mapped, all root causes are conclusively diagnosed, and the system 
is ready for direct corrective implementation before initiating the 2000–2025 download.
========================================================================================
```

---

## 2. Actual Project Phase History & Traceability

The complete reconstructed history from initial dataset discovery through Phase 2D was verified against codebase artifacts:

```
Phase 1A: Discovery & NetCDF Group Inspection         --> [IMPLEMENTED AND VERIFIED]
Phase 1B: S3 Inventory & Pilot Manifest Planning      --> [IMPLEMENTED AND VERIFIED]
Phase 1C: Downloader Debugging & Manifest Audit       --> [IMPLEMENTED AND VERIFIED]
Phase 1D: Local Modality Presence Audit               --> [IMPLEMENTED AND VERIFIED]
Phase 1E: Exploratory Data Analysis (12 Figures)      --> [IMPLEMENTED AND VERIFIED]
Phase 1F: Scientific Consistency & Split Creation     --> [IMPLEMENTED AND VERIFIED]
Phase 2A: Pilot-First Pipeline Implementation (HDF5)  --> [IMPLEMENTED AND VERIFIED]
Phase 2B: TC-GraphFormer & Two-Machine Staging        --> [IMPLEMENTED, NOT FULLY VERIFIED]
Phase 2C: Scientific Integrity Audit & Baselines      --> [IMPLEMENTED AND VERIFIED]
Phase 2D: Comprehensive Read-Only Integrity Audit     --> [IMPLEMENTED AND VERIFIED]
```

Full traceability across source code, configurations, tests, and deliverables is cataloged in [`phase_history_and_traceability.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/phase_history_and_traceability.csv).

---

## 3. Repository Architecture and Code-Flow Description

### 3.1 Software Modularity & Acyclic Design
The repository is structured into two cleanly separated subsystems communicating via a standardized filesystem data contract:
1. **Stage 1 (Preprocessing Pipeline)**: Packaged under `tcprimed_pipeline/` (11 modules). Orchestrated by `pipeline_runner.py` and configurable via `tcprimed_pipeline/config.py`.
2. **Stage 2 (Deep Learning Research System)**: Implemented in `src/models/`, `src/training/`, `src/evaluation/`, `src/data/`, and `src/infrastructure/`.

### 3.2 Code Connection Audit
- **Connected Production Core**: `tcprimed_pipeline` generates standardized HDF5 containers and `reports/pipeline/sample_index.csv`. In turn, `src/data/dataset.py` reads `sample_index.csv`, loads HDF5 arrays, and provides batches to `TCGraphFormer` and `MultiTaskLoss`.
- **Precursor Phase 1 Code**: The root scripts `src/inspect_netcdf.py`, `src/validate_netcdf.py`, `src/prepare_pilot_dataset.py`, `src/audit_pilot_modalities.py`, `src/eda_pilot_dataset.py`, and `src/reconcile_pilot_consistency.py` are historical exploratory utilities. They do not participate in the active Phase 2 training pipeline.
- Detailed architecture and flow diagrams are documented in [`source_execution_dependency_map.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/source_execution_dependency_map.md) and [`repository_architecture_audit.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/repository_architecture_audit.md).

---

## 4. Raw Data Inventory and Quality Findings

### 4.1 Reconciliation of the Pilot Dataset
The local dataset in `data/raw/pilot/` was independently scanned and verified:
- **Valid NetCDF Files**: Exactly **2,679 `.nc` files** (2,599 satellite overpasses + 80 environmental NetCDFs).
- **Exact Byte Volume**: Exactly **19,999,443,986 bytes (19.999 GB / 18.63 GiB)**, matching `reports/pilot_manifest.csv` down to the single byte.
- **The 2,694 File Discrepancy Explained**: A recursive walk counted 2,694 total files. The difference of **15 files** corresponds to orphaned temporary download artifacts (e.g. `.nc.2cF59cF4`, `.nc.defF26fc`) left over from interrupted curl retry attempts during Phase 1C. Zero valid `.nc` files are missing.
- **NetCDF Readability**: An exhaustive batch scan using `netCDF4.Dataset` confirmed that **2,679 out of 2,679 files (100.00%)** open cleanly with zero data corruption.
- Complete audit details are available in [`raw_inventory_reconciliation.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/raw_inventory_reconciliation.csv).

---

## 5. Raw-to-Processed Data Fidelity

### 5.1 Storage Reduction & Container Architecture
- Raw NetCDF pilot volume: **20.74 GB** (including 736 MB fragments) / **19.999 GB** (`.nc` only).
- Processed HDF5 volume: **1,509,537,971 bytes (1.51 GB / 1.41 GiB)** across 80 storm containers.
- Empirical Compression Ratio: **13.74x** (processed data is 7.28% of raw size).

### 5.2 Numerical Bitwise Preservation
Comparative array extractions between raw NetCDF arrays and processed HDF5 arrays verified floating-point preservation:
- **IRWIN Brightness Temp**: `0.000000 K` max absolute difference (**BITWISE EXACT**).
- **PMW S1 / S2 Radiometer TBs**: `0.000000 K` max absolute difference (**BITWISE EXACT**).
- **GPROF Surface Rain Rate**: `0.000000 mm/h` max absolute difference (**BITWISE EXACT**).
- **ERA5 Total Column Water Vapor**: `0.000000 kg/m²` max absolute difference (**BITWISE EXACT**).
- **Radar Reflectivity**: Exactly `0.000000 dBZ` difference when compared against the co-registered dual-frequency `/radar/KuKaGMI/zFactorCorrected` product.
- Complete variable-by-variable lineage is cataloged in [`source_processed_variable_lineage.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/source_processed_variable_lineage.csv).

---

## 6. Feature / Target Lineage and Leakage Analysis

### 6.1 Causal Lineage Verification
- **Environmental Matching Causality**: All 2,599 observations in `sample_index.csv` were verified against matched ERA5 timestamps. Exactly **2,599 / 2,599 (100.00%)** satisfy:
  $$t_{\text{env}} - t_{\text{overpass}} \le 0$$
  Mean backward offset is $-28.4$ minutes (nearest prior hourly reanalysis). Zero forward-looking reanalysis leaked into input tensors.
- **Operational Reality Note**: ERA5 is retrospective reanalysis. In operational deployment, input environmental data must be replaced by real-time operational analysis fields (e.g. ECMWF HRES or GFS analysis).
- Complete data lineage is mapped in [`feature_target_lineage.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/feature_target_lineage.csv).

---

## 7. Split Validity and Representativeness

### 7.1 Split Observation Count Discrepancy Resolved
- The authoritative production split in `reports/pipeline/storm_split_manifest.csv` and `reports/pipeline/sample_index.csv` is:
  $$\text{TRAIN: } 56\text{ storms, } 1,742\text{ overpasses } (67.0\%)$$
  $$\text{VAL: } 12\text{ storms, } 612\text{ overpasses } (23.5\%)$$
  $$\text{TEST: } 12\text{ storms, } 245\text{ overpasses } (9.4\%)$$
  $$\text{TOTAL: } 80\text{ storms, } 2,599\text{ overpasses } (100.0\%)$$
- The number 1,847 cited in Phase 2C text was a typographical error in the narrative report; the underlying table in `storm_partition_distribution.csv` correctly recorded 1,742.
- Split isolation is 100% clean at the storm level.

### 7.2 The Test Partition Representativeness Defect
An exhaustive meteorological audit of the 12 test storms revealed:
1. **Zero North Atlantic (`AL`) Storms**: The test set has 0 storms from the North Atlantic basin.
2. **Zero Hurricane-Stage Labels**: In the categorical WMO stage taxonomy, 0 out of 245 overpasses possess a label of `'HU'` or `'TY'`.
3. **Weak Systems Dominance**: 11 of the 12 test storms never exceeded 57 knots. The only major cyclone in the test split (`SH111989`, peak 125 kt) was labeled `'UN'` (Unknown) due to historical 1989 JTWC conventions.
- Complete split details and the multi-objective stratified re-split protocol are documented in [`dataset_split_reconciliation.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/dataset_split_reconciliation.csv) and [`split_representativeness_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/split_representativeness_report.md).

---

## 8. Model Architecture Audit

### 8.1 Tensor Flow & Dimension Integrity
- **5 Modality Encoders**: Convolutional downsampling maps $(B, C, 64, 64) \to 5 \times (B, 16, 128)$ tokens.
- **Cross-Modal Fusion**: 80 tokens $(B, 80, 128)$ attend across modalities with key-padding masks.
- **The 16-Node Graph Shell Discrepancy Resolved**: The $4 \times 4$ Cartesian grid patch decomposition geometrically produces **three distinct radial shells**:
  - 4 Inner Core / Eyewall nodes ($r \approx 177\text{ km}$)
  - 8 Inner Rainband nodes ($r \approx 395\text{ km}$)
  - 4 Outer Rainband / Synoptic nodes ($r \approx 530\text{ km}$)
  The theoretical description in Phase 2B cited 4 conceptual bands; the physical code implementation uses a $4 \times 4$ Cartesian grid that clusters into 3 geometric distance shells across 4 azimuthal quadrants.
- **Causal Temporal Transformer**: Upper-triangular attention mask strictly enforces causality (future perturbation at $t=2$ causes $0.00000000$ change at $t \le 1$).
- Full tensor dimensions and node details are cataloged in [`model_tensor_trace.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/model_tensor_trace.csv) and [`graph_node_physical_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/graph_node_physical_audit.csv).

---

## 9. Training and Fine-Tuning Audit

### 9.1 Computational Reality: Disproven GPU Claims
- Checkpoint metadata in `checkpoints/best_checkpoint.pt`, `checkpoint_epoch_001.pt`, `checkpoint_epoch_002.pt`, and `finetuned_*.pt` explicitly record:
  $$\text{device: } \mathbf{"cpu"}$$
- `reports/hardware_inventory_gpu_node.json` is a duplicate of Machine A's local hardware report.
- The model has never been executed on a remote GPU. Training consisted of 1–2 integration testing epochs on Machine A's local CPU.
- Verified in [`training_checkpoint_evidence.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/training_checkpoint_evidence.csv) and [`fine_tuning_evidence.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/fine_tuning_evidence.csv).

---

## 10. Baseline and Metric Validity

### 10.1 Evaluation Fairness & Root Cause of Regression Defect
- Baseline comparisons (Persistence, Climatology, Linear Regression, TC-GraphFormer) evaluated on the exact same 245 test samples.
- **Catastrophic Initialization Bias**: The unconverged TC-GraphFormer checkpoint produced an MAE of **995.89 hPa** on pressure and **39.14 kt** on wind. The linear output heads were initialized with zero-mean normal weights without bias offsets, causing the network to predict $\sim 0\text{ hPa}$ instead of $\sim 1000\text{ hPa}$.
- **Stage Classification Skill**: On categorical stage classification, TC-GraphFormer achieved **47.76% accuracy**, significantly beating the Majority-Class baseline (22.45%).
- Detailed benchmark tables and statistical significance analyses are available in [`baseline_fairness_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/baseline_fairness_audit.csv) and [`evaluation_reliability_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/evaluation_reliability_report.md).

---

## 11. Infrastructure and Storage Assessment

### 11.1 Workstation Storage Deficit
- Internal Drive C: **38.86 GB free** (90.2% full).
- Internal Drive D: **31.33 GB free** (59.9% full).
- Total Available Internal Storage: **70.19 GB**.
- Projected 500 GB Dataset Footprint: **608.6 GB** (deficit of **538.4 GB**).

### 11.2 Resolution of the Storage Calculation Discrepancy
- Discrepancy between nominal 557.7 GB and comprehensive budget of 608.6 GB is **50.9 GB**:
  - **40.0 GB**: Interleaved binary shards / fast staging cache (`export_shards.py`).
  - **10.0 GB**: Checkpoints, optimizer states, evaluation runs, and TensorBoard logs across 50 epochs.
  - **0.9 GB**: Container storage rounding based on modern sensor byte growth.
- Documented in [`hardware_storage_audit.json`](file:///D:/Project/Capstone_project/reports/phase2d_audit/hardware_storage_audit.json) and [`infrastructure_readiness.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/infrastructure_readiness.md).

---

## 12. Website / Inference Readiness

1. **Preprocessing Decoupling**: The current pipeline is storm-container-oriented. For real-time inference or an interactive web dashboard, an overpass-level inference wrapper must accept arbitrary NetCDF granules and apply precomputed standardization statistics (`normalization_stats.json`).
2. **Asynchronous Time Handling**: Satellite overpasses arrive irregularly (every 3–12 hours). The web interface must present discrete observation timestamps rather than treating asynchronous overpasses as a uniform 60-minute video stream.
3. **Serving Container**: A lightweight FastAPI or ONNX/TorchScript serving interface is required to output wind speed, central pressure, stage logits, RI probability, and track cones.

---

## 13. Prioritized Risk Register

The complete risk register is cataloged in [`risk_register.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/risk_register.csv):

| Risk ID | Severity | Finding Title | Impact | Blocks 500GB Scaling? |
| :---: | :---: | :--- | :--- | :---: |
| **R-01** | **CRITICAL** | Workstation Internal Storage Deficit (<71 GB free vs 609 GB req.) | OS crash, disk exhaustion | **YES** |
| **R-02** | **CRITICAL** | Disproven GPU Compute on Machine B (Placeholder JSON) | Cannot train 50 epochs on CPU | **NO (Blocks GPU Train)** |
| **R-03** | **CRITICAL** | Test Partition Lacking Hurricanes (0 Hurricanes, 0 AL Storms) | Biased evaluation; invalid metrics | **YES** |
| **R-04** | **HIGH** | Output Layer Initialization Bias (Pressure MAE 995.89 hPa) | Catastrophic regression failure | **NO (Blocks Train/Eval)** |
| **R-05** | **HIGH** | Track Displacement Missingness (84.5% missing) | Insufficient supervision for track | **NO (Blocks Track Eval)** |
| **R-06** | **HIGH** | Rapid Intensification Class Imbalance (2.91% prevalence) | Standard BCE recall = 0% | **NO (Blocks RI Eval)** |
| **R-07** | **HIGH** | Self-Supervised Pretraining Masking Absent Modality Slots | Capacity wasted on zero-prediction | **NO (Blocks Pretrain)** |
| **R-08** | **MEDIUM** | Project Directory Missing Git Version Control | Unversioned code; sync risk | **NO** |
| **R-09** | **MEDIUM** | Staging Manifest Omitting Split and Observation Metadata | Machine B cannot filter splits | **NO** |
| **R-10** | **MEDIUM** | Modern Era +27.5% Storage Inflation | Underestimated storage | **NO** |
| **R-11** | **LOW** | Static Graph Edges Lacking Environmental Wind Shear Vector | Suboptimal convective routing | **NO** |

---

## 14. Exact Corrective Actions (Pre-Scaling Blueprint)

Before downloading the 500 GB dataset, three mandatory engineering corrections must be executed:

```
[CORRECTION 1: STORAGE PROCUREMENT & EXTERNAL ROOT SETUP]
- Procure and mount an External NVMe SSD (>= 1 TB capacity, USB 3.2 Gen 2 / USB4).
- Point configs/pilot.yaml and pipeline_runner.py to the external drive mount (e.g. E:\data).

[CORRECTION 2: MULTI-OBJECTIVE STRATIFIED RE-SPLIT]
- Update tcprimed_pipeline/storm_splitter.py to stratify simultaneously across:
    Basin (6) x Historical Era (4) x Peak Saffir-Simpson Category (TD, TS, Cat 1-2, Major Cat 3-5).
- Regenerate sample_index.csv ensuring Test set contains >= 20% Category 1-5 hurricanes.

[CORRECTION 3: NATIVE REMOTE GPU & RSYNC PIPELINE SETUP]
- Initialize Git repository and push codebase to GitHub/GitLab.
- Clone repository on Machine B (Professor's GPU Node), configure Python venv with CUDA 12.x.
- Replace AnyDesk transport with automated rsync over SSH or high-speed NVMe drive staging.
```

---

## 15. Recommended Phase Sequence

```
========================================================================================
RECOMMENDED WORKFLOW ROADMAP
========================================================================================
1. PHASE 2E (Targeted Pre-Scaling Corrections):
   - Git initialization and requirements.txt specification.
   - Mount external NVMe SSD.
   - Regenerate multi-objective stratified splits (with Category 1-5 hurricanes in Test).
   - Target z-scoring normalization & output head bias initialization.
   - Modality-gated MFM pretraining loss mask.
   - Cubic Hermite spline interpolation for off-synoptic track labels.
   - Machine B native CUDA environment verification.

2. PHASE 3A (Curated 2000–2025 Dataset Download):
   - Curate and download 500 GB raw NetCDF corpus from AWS S3 directly to External SSD.
   - Execute production preprocessing pipeline to produce ~38.6 GB HDF5 containers.

3. PHASE 3B (Distributed GPU Training on Machine B):
   - Stage processed containers to Machine B via rsync/SSH.
   - Execute self-supervised MFM pretraining on TRAIN partition (GPU).
   - Execute multi-task training (50 epochs, GPU) with Class-Ratio Weighted BCE for RI.
   - Evaluate against Baselines on balanced Test partition with bootstrap confidence intervals.
========================================================================================
```

---

## 16. Comprehensive Index of Generated Audit Deliverables

All 25 deliverables for Phase 2D are complete and accessible in [`reports/phase2d_audit/`](file:///D:/Project/Capstone_project/reports/phase2d_audit/):

1. [`phase2d_master_audit_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/phase2d_master_audit_report.md) — Master comprehensive audit report and final decision.
2. [`phase_history_and_traceability.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/phase_history_and_traceability.csv) — Traceability matrix of all project phases (1A through 2D).
3. [`repository_architecture_audit.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/repository_architecture_audit.md) — Detailed software engineering audit (code quality, logging, portability).
4. [`source_execution_dependency_map.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/source_execution_dependency_map.md) — Mermaid execution flow and code coupling map.
5. [`test_quality_assessment.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/test_quality_assessment.csv) — Audit of all 107 unittests with categorization and failure detection scopes.
6. [`raw_inventory_reconciliation.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/raw_inventory_reconciliation.csv) — Reconciliation of 2,679 `.nc` files vs 2,694 total files and byte counts.
7. [`data_quality_summary.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/data_quality_summary.csv) — Empirical statistics, physical ranges, and outlier counts for key variables.
8. [`modality_cooccurrence.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/modality_cooccurrence.csv) — Joint modality availability matrix across 2,599 overpasses.
9. [`spatial_temporal_quality_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/spatial_temporal_quality_report.md) — Audit of spatial resolutions, coordinate grids, and asynchronous time intervals.
10. [`source_processed_variable_lineage.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/source_processed_variable_lineage.csv) — Array-level mapping from raw NetCDF to processed HDF5.
11. [`feature_target_lineage.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/feature_target_lineage.csv) — Feature vs supervisory target lineage and forecast initialization check.
12. [`target_definition_and_quality_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/target_definition_and_quality_audit.csv) — Formal formulas, valid counts, missing reasons for all targets.
13. [`dataset_split_reconciliation.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/dataset_split_reconciliation.csv) — Reconciliation of split versions (1,742 vs 1,847 overpass resolution).
14. [`split_representativeness_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/split_representativeness_report.md) — Meteorological audit of the 12 test storms (0 hurricanes, 0 AL storms).
15. [`model_tensor_trace.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/model_tensor_trace.csv) — Step-by-step tensor dimensions and parameter counts through TC-GraphFormer.
16. [`graph_node_physical_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/graph_node_physical_audit.csv) — Derivation of 16 spatial nodes and resolution of the 3-shell radial geometry.
17. [`model_component_usage.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/model_component_usage.csv) — Audit of all 10 neural submodules, parameters, and gradient flow.
18. [`training_checkpoint_evidence.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/training_checkpoint_evidence.csv) — Audit of all 7 checkpoint files on disk (confirming CPU execution).
19. [`fine_tuning_evidence.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/fine_tuning_evidence.csv) — Audit of intensity change and NIO adaptation fine-tuning checkpoints.
20. [`baseline_fairness_audit.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/baseline_fairness_audit.csv) — Evaluation benchmark of baselines vs TC-GraphFormer.
21. [`evaluation_reliability_report.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/evaluation_reliability_report.md) — Statistical degrees of freedom ($N_{\text{eff}} \approx 12$) and bootstrap uncertainty.
22. [`hardware_storage_audit.json`](file:///D:/Project/Capstone_project/reports/phase2d_audit/hardware_storage_audit.json) — System compute profiles, storage capacities, and 608.6 GB budget.
23. [`infrastructure_readiness.md`](file:///D:/Project/Capstone_project/reports/phase2d_audit/infrastructure_readiness.md) — Audit of Machine A, Machine B, AnyDesk vs SSH, and 50.9 GB arithmetic resolution.
24. [`risk_register.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/risk_register.csv) — Prioritized register of 3 Critical, 4 High, 3 Medium, and 1 Low risks.
25. [`missing_evidence_register.csv`](file:///D:/Project/Capstone_project/reports/phase2d_audit/missing_evidence_register.csv) — Audit of unevidenced or contradicted claims with verified truth.

