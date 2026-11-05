# TC-GraphFormer: Multimodal Spatiotemporal Graph-Transformer for Tropical Cyclone Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-NOAA%20TC%20PRIMED%20v01r01--final-0077b6.svg)](https://registry.opendata.aws/noaa-nesdis-tcprimed-pds/)
[![Tests](https://img.shields.io/badge/Tests-107%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-ready, scientifically validated deep learning and remote sensing data pipeline for tropical cyclone monitoring, intensity estimation, rapid intensification detection, and 6–24-hour track and intensity forecasting using the **NOAA Tropical Cyclone PRecipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED v01r01-final)**.

---

## Architecture Overview

TC-GraphFormer models heterogeneous, asynchronous, multi-platform satellite swaths and atmospheric reanalysis through a unified five-stage spatiotemporal architecture:

```mermaid
flowchart TD
    subgraph Inputs [Multimodal Sensor Swaths & Reanalysis]
        IR["Infrared Window (IRWIN 4km)\n(B, 1, 64, 64)"]
        PMW["Passive Microwave (85/89/166 GHz)\n(B, 4, 64, 64)"]
        GP["GPROF Surface Precipitation\n(B, 1, 64, 64)"]
        RAD["GPM DPR / TRMM PR Radar\n(B, 1, 64, 64)"]
        ENV["ECMWF ERA5 Reanalysis & Shear\n(B, 6, 64, 64)"]
        MASK["Modality Presence Masks\n(B, 5)"]
    end

    subgraph Encoders [Dedicated Modality Encoders]
        E_IR["Conv2d Patch Encoder"]
        E_PMW["Conv2d Patch Encoder"]
        E_GP["Conv2d Patch Encoder"]
        E_RAD["Conv2d Patch Encoder"]
        E_ENV["Conv2d Patch Encoder"]
    end

    subgraph Fusion [Mask-Aware Cross-Modal Attention]
        FUSE["CrossModalFusion (2 Layers, 4 Heads)\n80 Multimodal Tokens (B, 80, 128)"]
    end

    subgraph Graph [Topological Spatial Graph (MPNN)]
        PROJ["Spatial Aggregator (MLP)\nProjects 80 Tokens -> 16 Spatial Nodes"]
        GNN["SpatialGraphEncoder (2 MPNN Layers)\nEyewall, Rainband & Synoptic Radial Shells"]
        POOL["Attention Graph Readout -> (B, 128)"]
    end

    subgraph Temporal [Continuous-Time Causal Transformer]
        TEMP["CausalTemporalTransformer\nUpper-Triangular Mask + Elapsed Time Embedding (dt)"]
    end

    subgraph Heads [Multi-Task Prediction Heads]
        H_V["Intensity Head: V_t0 (knots)"]
        H_P["Pressure Head: P_t0 (hPa)"]
        H_S["Stage Head: WMO Class Logits (6 classes)"]
        H_DV["Intensity Change Head: dV (+6h, +12h, +18h, +24h)"]
        H_RI["Rapid Intensification Head: dV_24h >= 30 kt"]
        H_TR["Track Head: Displacement (dlat, dlon across horizons)"]
    end

    IR --> E_IR
    PMW --> E_PMW
    GP --> E_GP
    RAD --> E_RAD
    ENV --> E_ENV

    E_IR & E_PMW & E_GP & E_RAD & E_ENV --> FUSE
    MASK --> FUSE

    FUSE --> PROJ --> GNN --> POOL
    POOL --> TEMP
    TEMP --> H_V & H_P & H_S & H_DV & H_RI & H_TR
```

### Core Innovations:
1. **Heterogeneous Modality Fusion**: Attends across 5 distinct remote sensing instruments while robustly handling severe sensor missingness (e.g. radar available in only ~3% of overpasses) without crashing or generating NaNs.
2. **Topological Cyclone Graph (16 Nodes)**: Replaces naive spatial flattening with a cyclone-centric Graph Neural Network that structures the storm domain into 3 physical radial shells (Core/Eyewall: $r \sim 177\text{ km}$, Inner Rainbands: $r \sim 395\text{ km}$, Outer Rainbands: $r \sim 530\text{ km}$) and 4 azimuthal quadrants.
3. **Continuous-Time Causal Transformer**: Accommodates irregular, asynchronous satellite revisits (median gap: 4.18 hours) via continuous elapsed time ($\Delta t$) encodings and lower-triangular causal attention masking.
4. **Causal Data Pipeline**: Strictly guarantees temporal causality ($t_{\text{env}} \le t_{\text{overpass}}$), eliminating all temporal lookahead leakage.
5. **Loss Reweighting for Rare Events**: Addresses extreme Rapid Intensification class imbalance (2.91% positive prevalence) using Class-Ratio Weighted BCE ($w = 33.3$) and Focal Loss, elevating RI recall from 0.0% to 73.08%.

---

## Repository Structure

```
TC-GraphFormer-Forecasting-prediction-Model/
├── configs/                     # YAML configuration files
│   ├── pilot.yaml               # Preprocessing pipeline config
│   ├── model.yaml               # TC-GraphFormer architectural dimensions
│   ├── pretraining.yaml         # Self-supervised MFM pretraining config
│   ├── multitask_training.yaml  # Multi-task supervised training config
│   ├── finetuning.yaml          # Task fine-tuning & domain adaptation
│   ├── workstation.yaml         # Workstation compute node configuration
│   └── gpu_node.yaml            # Remote GPU compute node configuration
├── data/                        # Dataset mount point (gitignored)
│   ├── raw/                     # Raw NOAA TC PRIMED NetCDF files
│   └── processed/               # Standardized HDF5 containers (*_processed.nc)
├── tcprimed_pipeline/           # Stage 1: Preprocessing & Data Extraction Package
│   ├── __main__.py              # Pipeline CLI entry point
│   ├── pipeline_runner.py       # Orchestration engine with resume support
│   ├── netcdf_reader.py         # NetCDF-4 extraction and attribute parser
│   ├── metadata_extractor.py    # Temporal, coordinate, and Best Track extraction
│   ├── environmental_matcher.py # Causal ERA5 reanalysis matching (t_env <= t_overpass)
│   ├── modality_processors.py   # Sensor spatial re-gridding and channel slicing
│   ├── quality_control.py       # Coverage masking, fill values, QA filtering
│   ├── storage_writer.py        # HDF5 container writer with float32 preservation
│   ├── storm_splitter.py        # 70/15/15 storm-level partition splitting
│   ├── inventory.py             # S3 bucket and local filesystem inventory
│   └── reporting.py             # Manifest generation and sample_index.csv creation
├── src/                         # Stage 2: Deep Learning System
│   ├── data/                    # PyTorch Dataset and Multimodal Collate modules
│   │   ├── dataset.py           # TCPRIMEDDataset reading HDF5 and sample_index.csv
│   │   └── collate.py           # collate_multimodal_batch packing multimodal tensors
│   ├── models/                  # TC-GraphFormer Neural Network
│   │   ├── tc_graphformer.py    # Unifying model class
│   │   ├── encoders/            # Modality encoders (IR, PMW, GPROF, Radar, Env)
│   │   ├── fusion/              # CrossModalFusion transformer
│   │   ├── graph/               # SpatialGraphEncoder & message passing layers
│   │   ├── temporal/            # CausalContinuousTimeTemporalTransformer
│   │   └── heads/               # Multi-task heads for intensity, pressure, stage, track
│   ├── training/                # Training engines and loss formulations
│   │   ├── losses.py            # MultiTaskLoss with dynamic loss masking
│   │   ├── trainer.py           # Optimization, gradient clipping, checkpoint saving
│   │   ├── pretrain.py          # Self-supervised Masked Feature Modeling (MFM)
│   │   ├── train_multitask.py   # Multi-task supervised training workflow
│   │   ├── finetune_task.py     # Fine-tuning & North Indian Ocean (NIO) adaptation
│   │   └── ri_experiments.py    # Rapid Intensification class imbalance benchmarks
│   ├── evaluation/              # Benchmark evaluation suite
│   │   └── evaluate_baselines.py# Persistence, Climatology, Linear & TC-GraphFormer eval
│   └── infrastructure/          # Distributed two-machine staging utilities
│       ├── hardware_info.py     # CPU, RAM, CUDA hardware discovery
│       ├── export_shards.py     # Fast binary shard export for high-throughput GPU training
│       └── verify_transfer.py   # SHA-256 cryptographic transfer validation
├── tests/                       # Complete Unit & Integration Test Suite (107 Tests)
├── scripts/                     # Operational automation scripts (PowerShell & Bash)
│   ├── verify_system.py         # System pre-flight diagnostic check
│   ├── run_pipeline.ps1 / .sh   # Preprocessing runner scripts
│   ├── run_training.ps1 / .sh   # Training runner scripts
│   └── run_evaluation.ps1 / .sh # Benchmark evaluation runner scripts
├── reports/                     # Audit matrices, EDA reports, figures, and manifests
│   ├── phase2d_audit/           # Comprehensive Phase 2D technical audit (25 files)
│   ├── phase2c/                 # Phase 2C scientific integrity reports (19 files)
│   ├── eda/                     # Exploratory data analysis reports & figures
│   └── pipeline/                # Production manifests and sample_index.csv
├── docs/                        # Technical documentation & research notes
├── pyproject.toml               # Modern PEP 517/621 package build definition
├── setup.py                     # Backwards-compatible editable installation
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Testing, linting, and development dependencies
├── Makefile                     # Standard developer commands
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model.git
cd TC-GraphFormer-Forecasting-prediction-Model

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
# Production dependencies
pip install -r requirements.txt

# Install TC-GraphFormer in editable development mode
pip install -e .
```

### 3. Verify Environment
```bash
python scripts/verify_system.py
```

### 4. Run Test Suite
```bash
# Execute all 107 unit and integration tests
python -m unittest discover tests
```

---

## Data Pipeline Execution

The pipeline ingests raw NOAA TC PRIMED NetCDF-4 files and writes standardized, compressed HDF5 containers with zero floating-point precision loss (compression factor: 13.7x; raw 20.74 GB $\to$ processed 1.51 GB).

```bash
# Run preprocessing pipeline on pilot dataset
python -m tcprimed_pipeline --config configs/pilot.yaml

# Resume an interrupted preprocessing job
python -m tcprimed_pipeline --config configs/pilot.yaml --resume
```

### Generated Data Products:
- `data/processed/pilot/<year>/<basin>/<storm_id>/<storm_id>_processed.nc`: Standardized HDF5 containers containing cropped $64 \times 64$ sensor fields and Best Track labels.
- `reports/pipeline/sample_index.csv`: Master dataset index (2,599 observations) recording partition assignments, timestamps, sensor flags, and target values.

---

## Model Training & Workflows

### 1. Self-Supervised Pretraining (Masked Feature Modeling)
Pretrains cross-modal encoders and fusion layers on the `TRAIN` partition without requiring supervisory labels:
```bash
python src/training/pretrain.py --config configs/pretraining.yaml
```

### 2. Multi-Task Supervised Training
Trains TC-GraphFormer end-to-end across all 6 prediction tasks:
```bash
python src/training/train_multitask.py --config configs/multitask_training.yaml
```

### 3. Rapid Intensification Imbalance Optimization
Evaluates class-weighting and focal loss configurations for rare RI events:
```bash
python src/training/ri_experiments.py
```

### 4. Task Fine-Tuning & Domain Adaptation
Freezes lower modality encoders and fine-tunes prediction heads on specific tasks or regional basins (e.g. North Indian Ocean `IO`):
```bash
# Fine-tune on multi-horizon intensity change:
python src/training/finetune_task.py --task intensity_change --epochs 3

# Adapt to North Indian Ocean domain:
python src/training/finetune_task.py --task nio_adaptation --epochs 3
```

---

## Evaluation & Benchmarks

Benchmark TC-GraphFormer against standardized operational and statistical baselines (Persistence, Climatology, Linear Regression, Majority-Class):
```bash
python src/evaluation/evaluate_baselines.py
```

### Benchmark Summary (Pilot Test Partition):

| Model | Task | Metric | Test Partition Performance | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Persistence** | Current Intensity | MAE | 15.51 kt | Uses current wind as forecast |
| **Climatology** | Central Pressure | MAE | 5.14 hPa | Uses training set mean pressure |
| **Majority Class** | Development Stage | Accuracy | 22.45% | Predicts most frequent class (TS) |
| **TC-GraphFormer** | Development Stage | Accuracy | **47.76%** | **Outperforms Majority Baseline (+25.3%)** |
| **TC-GraphFormer (Class-Ratio BCE)** | Rapid Intensification | Recall | **73.08%** | **PR-AUC: 0.1873** (Elevates recall from 0.0%) |

---

## Two-Machine Compute Architecture

To scale from the 20 GB development pilot to the full 500 GB curated corpus (2000–2025), the project utilizes a two-machine distributed framework:

```
[Machine A: Workstation / Research Laptop]            [Machine B: Dedicated GPU Compute Node]
  - S3 Data Inventory & Download                       - Native PyTorch with CUDA 12.x
  - Preprocessing (tcprimed_pipeline)                  - High-throughput NVMe Shard Caches
  - HDF5 Container Generation                          - Distributed Self-Supervised Pretraining
  - QC Filtering & Manifest Creation                   - 50-Epoch Multi-Task GPU Training
               |                                                        ^
               +================= Automated rsync / SSH ================+
```

Data staging generates bitwise cryptographic manifests (`staging/staging_manifest.json`) verifying 100% SHA-256 integrity before model training commences on Machine B.

---

## Comprehensive Integrity Audit Deliverables

All technical and scientific audit reports, empirical data distributions, and failure analyses generated during Phase 2C and Phase 2D are preserved in `reports/`:
- [`phase2d_master_audit_report.md`](reports/phase2d_audit/phase2d_master_audit_report.md) — Comprehensive technical master report.
- [`risk_register.csv`](reports/phase2d_audit/risk_register.csv) — Prioritized 11-point engineering risk register.
- [`source_execution_dependency_map.md`](reports/phase2d_audit/source_execution_dependency_map.md) — Module execution flow.
- [`test_quality_assessment.csv`](reports/phase2d_audit/test_quality_assessment.csv) — Comprehensive breakdown of all 107 tests.
- [`hardware_storage_audit.json`](reports/phase2d_audit/hardware_storage_audit.json) — Workstation storage audit and 608.6 GB budget.

---

## Citation & References

If you use TC-GraphFormer or this data pipeline in your research, please cite:

```bibtex
@software{rana2026tcgraphformer,
  author       = {Hoshik Rana},
  title        = {TC-GraphFormer: Multimodal Spatiotemporal Graph-Transformer for Tropical Cyclone Forecasting},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/hoshikrana/TC-GraphFormer-Forecasting-prediction-Model}
}
```

### Data Citation
> Rozoff, C., et al. (2021). *NOAA Tropical Cyclone PRecipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED v01r01-final)*. NOAA National Centers for Environmental Information. Available via AWS Open Data Sponsorship: `s3://noaa-nesdis-tcprimed-pds/`.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

