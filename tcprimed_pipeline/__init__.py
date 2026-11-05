"""
tcprimed_pipeline - Modular, Configurable, and Restartable Data Pipeline
for the NOAA TC PRIMED v01r01-final Dataset.

Components:
  - config: Pipeline configuration loader and validator
  - inventory: Raw file cataloging and metadata indexing
  - netcdf_reader: Safe NetCDF4 reading with mask/scale preservation
  - metadata_extractor: Storm and overpass metadata extraction
  - modality_processors: Independent processors for PMW, IR, GPROF, Radar, Env
  - environmental_matcher: Causal ERA5 matching without future-time leakage
  - quality_control: Physical and metadata QC validation
  - storm_splitter: Reproducible storm-wise train/val/test partitioning
  - sample_builder: Forecasting sample indexing and target validation
  - storage_writer: Standardized NetCDF4/HDF5 per-storm container writer
  - reporting: Comprehensive metrics, CSV manifests, and markdown reports
  - pipeline_runner: End-to-end execution coordinator with resume support
"""

__version__ = "1.0.0"
__author__ = "TC PRIMED Intelligence Project"

