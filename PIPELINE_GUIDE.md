# TC PRIMED Pilot-First Data Preprocessing Pipeline

A real, modular, configurable, and restartable preprocessing pipeline for the NOAA TC PRIMED (`v01r01-final`) multimodal tropical cyclone dataset.

---

## 1. Architecture Overview

The pipeline standardizes raw satellite overpass NetCDF files and storm-aligned ERA5 environmental reanalysis files into per-storm NetCDF-4/HDF5 containers. It preserves floating-point scientific measurements, native coordinates, and sensor resolutions without lossy image conversion or premature spatial remapping.

```
tcprimed_pipeline/
├── __init__.py               # Package metadata and version
├── config.py                 # YAML configuration parser and disk space validator
├── inventory.py              # File discovery, metadata parsing, and manifest generation
├── netcdf_reader.py          # Safe NetCDF4 reading with auto-decoding scale/offset once
├── metadata_extractor.py     # Overpass metadata, contemporaneous Best Track, sequential track
├── modality_processors.py    # Independent processors for PMW, IR, GPROF, Radar, Env
├── environmental_matcher.py  # Strictly causal ERA5 matching (t_env <= t_op)
├── quality_control.py        # Physical bounds checks, duplicate detection, QC reporting
├── storm_splitter.py         # Reproducible 70/15/15 storm-wise partition (0 leakage)
├── sample_builder.py         # Forecasting sample indexing and target availability
├── storage_writer.py         # Standardized NetCDF-4/HDF5 per-storm container writer
├── reporting.py              # Processing ledger, markdown reports, and benchmark export
├── pipeline_runner.py        # Master coordinator with resume support
└── __main__.py               # CLI interface (python -m tcprimed_pipeline)
```

---

## 2. Prerequisites & Environment

- **Operating System:** Windows 11, Linux, or macOS
- **Python Version:** Python 3.10+ (tested on Python 3.13.12)
- **Key Dependencies:** `netCDF4` (>=1.7.0), `numpy` (>=2.0.0), `pyyaml` (>=6.0)
- **Virtual Environment:** Recommended `.venv`

---

## 3. Configuration Reference (`configs/pilot.yaml`)

```yaml
dataset:
  root: "data/raw/pilot"            # Directory containing raw NetCDF files
  manifest: "reports/pilot_manifest.csv"  # Optional inventory manifest
  date_start: null                  # Optional YYYYMMDD filter start
  date_end: null                    # Optional YYYYMMDD filter end
  include_storms: null              # Optional list of storm IDs (e.g. ['AL122015'])
  exclude_storms: null              # Optional list of storm IDs to skip
  include_basins: null              # Optional list of basins (e.g. ['AL', 'WP'])

output:
  root: "data/processed/pilot"      # Output directory for processed storm containers
  reports_dir: "reports/pipeline"   # Output directory for manifests and audit reports
  format: "netcdf4_hdf5"            # Output format
  compression_level: 4              # Zlib compression level (0-9)
  min_disk_free_gb: 2.0             # Safety check threshold for disk space

processing:
  workers: 1                        # Number of worker processes (1 for Windows safety)
  batch_size: 16                    # Processing batch size
  resume: true                      # Skip already-processed storm containers
  preserve_native_grids: true       # Avoid lossy spatial interpolation
  atomic_writes: true               # Write to .tmp before atomic rename

splitting:
  train: 0.70                       # Proportion of storms for training
  validation: 0.15                  # Proportion of storms for validation
  test: 0.15                        # Proportion of storms for test
  seed: 42                          # Fixed seed for reproducibility

targets:
  horizons_hours: [6, 12, 18, 24]   # Forecasting horizons
  ri_threshold_kt: 30.0             # Rapid Intensification threshold (kt / 24h)

quality_control:
  ir_mw_max_diff_minutes: 60.0      # Maximum allowed IR-MW timing offset
  enforce_causal_environmental: true # Prohibit future-time ERA5 slices
  min_valid_pixel_fraction: 0.05    # Minimum valid pixel fraction
  tb_min_k: 50.0                    # Minimum plausible PMW brightness temperature
  tb_max_k: 350.0                   # Maximum plausible PMW brightness temperature
  ir_min_k: 150.0                   # Minimum plausible IR brightness temperature
  ir_max_k: 350.0                   # Maximum plausible IR brightness temperature
```

---

## 4. Command-Line Usage

All operations are accessible through the Python module runner:

```bash
# 1. Dataset Discovery & Inventory
python -m tcprimed_pipeline inventory --config configs/pilot.yaml

# 2. Quality Control Audit
python -m tcprimed_pipeline validate --config configs/pilot.yaml

# 3. Storm-Wise Partitioning
python -m tcprimed_pipeline split --config configs/pilot.yaml

# 4. Standardized Processing (Incremental)
python -m tcprimed_pipeline process --config configs/pilot.yaml

# 5. Build Forecasting Sample Indexes
python -m tcprimed_pipeline build-samples --config configs/pilot.yaml

# 6. Execute End-to-End Pipeline with Resume
python -m tcprimed_pipeline run --config configs/pilot.yaml --resume
```

---

## 5. Storage & Large-Dataset Reusability (2000–2025 Deployment)

The pipeline is designed to process the planned ~500 GB selection (2000–2025) without code changes:
1. **Configurable Roots:** Adjust `dataset.root` and `output.root` in `configs/pilot.yaml` (or create `configs/deployment_500gb.yaml`).
2. **Date Filtering:** Set `date_start: "20000101"` and `date_end: "20251231"`.
3. **Storage Projection:** Benchmark results show an effective compression ratio of ~8–10x. Processing a 500 GB raw archive will produce ~50–65 GB of standardized NetCDF-4/HDF5 containers. Total disk space required during operation is approximately 570 GB (requiring at least a 1 TB external SSD).
4. **Crash-Resilience:** Per-storm atomic writes and `--resume` guarantee that interrupted runs resume from the last completed storm without data corruption.

---

## 6. Testing

Run unit and real-data integration tests:

```bash
# Unit tests with synthetic NetCDF files
python -m unittest tests.test_pipeline_units -v

# Real pilot observation tests (GMI, TMI, AMSR2, Radar, No-IR, Environmental)
python -m unittest tests.test_pipeline_real_data -v

# Full repository test suite
python -m unittest discover tests -v
```

