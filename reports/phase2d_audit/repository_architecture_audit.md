# TC PRIMED Phase 2D — Repository Architecture and Software Quality Audit

## 1. Executive Summary & Codebase Health

This audit performs an end-to-end static code analysis and architectural review of the NOAA TC PRIMED machine-learning codebase. The repository contains 228 non-virtualenv files comprising two primary subsystems: the preprocessing pipeline (`tcprimed_pipeline/`) and the deep learning research framework (`src/`).

Overall, the software exhibits strong architectural modularity, clear separation of concerns, and clean acyclic dependency graphs. However, critical software-engineering gaps exist in version control, structured logging, configuration unification, and cross-platform path handling.

---

## 2. Package Organization & Separation of Responsibilities

```
D:\Project\Capstone_project
├── tcprimed_pipeline/           # Stage 1: Production Preprocessing Engine (11 modules)
│   ├── netcdf_reader.py         # Low-level NetCDF-4/HDF5 reader with attribute handling
│   ├── metadata_extractor.py    # Temporal, coordinate, and Best Track extraction
│   ├── environmental_matcher.py # Causal reanalysis matching (t_env <= t_overpass)
│   ├── modality_processors.py   # Sensor-specific spatial re-gridding & channel slicing
│   ├── quality_control.py       # Coverage masking, fill-value filtering, QC flags
│   ├── storage_writer.py        # HDF5 container creation with float32 preservation
│   ├── storm_splitter.py        # 70/15/15 storm-level partition splitting
│   ├── reporting.py             # Manifest generation & sample_index.csv creation
│   ├── inventory.py             # S3 bucket and local directory inventory parsing
│   ├── config.py                # Dataclass-based pipeline configuration
│   └── pipeline_runner.py       # Orchestration engine with resume capability
├── src/                         # Stage 2: Spatiotemporal Deep Learning System
│   ├── data/                    # PyTorch Dataset and Multimodal Collate functions
│   ├── models/                  # TC-GraphFormer Neural Architecture (Encoders, Fusion, Graph, Heads)
│   ├── training/                # Losses, Trainer, Pretraining, Multi-Task, Fine-Tuning
│   ├── evaluation/              # Baseline benchmarks and multi-task evaluation
│   ├── infrastructure/          # Hardware discovery, staging manifests, shard export
│   └── (precursor scripts)      # Standalone Phase 1 exploratory utilities
├── configs/                     # YAML configuration files for pipeline, models, and nodes
├── tests/                       # 107 Unittests across 7 test suites (100% passing)
├── reports/                     # Audit reports, manifests, CSVs, and EDA figures
├── checkpoints/                 # Saved PyTorch model checkpoints (.pt)
└── staging/                     # Staged HDF5 containers and SHA-256 transfer manifest
```

### Module Coupling Assessment
- **Decoupled Engine Contract**: `tcprimed_pipeline` and `src` communicate strictly through standard HDF5 containers and `reports/pipeline/sample_index.csv`. There are zero cyclic imports between the two packages.
- **Acyclic Internal Graphs**: Within `src/models/`, submodules strictly follow a hierarchical dependency structure: Encoders/Fusion/Graph/Temporal/Heads $\to$ `tc_graphformer.py`.

---

## 3. Detailed Software Quality Criteria Assessment

| Criterion | Audit Finding | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Version Control** | **No Git repository found (`.git` absent).** Project directory is unversioned. | **CRITICAL** | Initialize Git repository immediately (`git init`), create `.gitignore` (ignoring `.venv`, `data/`, `*.pt`, `__pycache__`), and commit codebase. |
| **Configuration Management** | Configs are split between YAML (`configs/*.yaml`), dataclasses (`tcprimed_pipeline/config.py`), and CLI overrides. Some scripts (`ri_experiments.py`, `evaluate_baselines.py`) have hardcoded paths. | **HIGH** | Consolidate all paths and training hyperparameters into a unified Hydra or OmegaConf schema. |
| **Path Handling & OS Portability** | In several locations (e.g. `staging_manifest.json`), Windows backslashes (`\`) were serialized into JSON paths. PyTorch loaders handle backslashes on Windows, but this breaks on Linux GPU nodes (Machine B). | **HIGH** | Standardize on `pathlib.Path.as_posix()` or `os.path.normpath` across all manifests and logging tools. |
| **Error Handling & Resilience** | Preprocessing uses explicit `try...except` per overpass with logging to `reports/pipeline/processing_failures.csv`. Corrupt granules do not halt execution. | **LOW** | Maintained across all pipeline stages; clean fail-safe design. |
| **Logging vs. Standard Output** | Extensive use of `print()` statements across `src/training/` and `src/models/` without `-u` unbuffered flag, causing delayed telemetry in background runners. | **MEDIUM** | Migrate to standard Python `logging` module with console and file handlers. |
| **Type Annotations** | `tcprimed_pipeline` and `src/data/` have extensive type hints (`Dict[str, Any]`, `torch.Tensor`, `Optional[int]`). Some training loops lack return annotations. | **LOW** | Add complete mypy validation in CI. |
| **Duplicate Logic** | S3 inventory parsing exists in both `src/prepare_pilot_dataset.py` and `tcprimed_pipeline/inventory.py`. Downloader logic in Phase 1B/1C is separate from Phase 2A. | **MEDIUM** | Consolidate S3 downloading and inventory logic into `tcprimed_pipeline/inventory.py`. |
| **PyTorch Nested Tensor Warnings** | Training triggers `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True` in fusion and temporal transformers. | **LOW** | Explicitly pass `enable_nested_tensor=False` to `nn.TransformerEncoder` to suppress warning. |

---

## 4. Machine Portability & Configuration Readiness

Can the pipeline run seamlessly when moved from the Friend's Windows Laptop (Machine A) to the Professor's Remote Compute Node (Machine B)?

### Identified Portability Blockers:
1. **Hardcoded Drive Letters (`D:\...`)**: Several generated summary JSON files contain absolute Windows paths `D:\Project\Capstone_project\...`. While source code utilizes relative paths, checkpoint loaders and datasets must always accept an explicit `--data-root` or `--index-csv` argument.
2. **Path Separator Normalization**: Manifests generated on Windows contain backslashes (`data\processed\pilot\...`). When Machine B (Linux) parses the manifest, `os.path.exists()` fails unless paths are normalized to POSIX forward slashes.
3. **Missing Package Installation Manifest**: The repository lacks a `requirements.txt` or `pyproject.toml` specifying pinned versions for PyTorch, torchvision, netCDF4, h5py, pyyaml, and pandas. Machine B may experience library version mismatch upon deployment.

**Action Item**: Create `requirements.txt` and a containerized / conda environment specification for Machine B prior to Phase 3.

