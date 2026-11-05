"""
Pipeline Configuration module for TC PRIMED.
Loads, validates, and provides access to pipeline configuration parameters.
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class DatasetConfig:
    root: str = "data/raw/pilot"
    manifest: Optional[str] = "reports/pilot_manifest.csv"
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    include_storms: Optional[List[str]] = None
    exclude_storms: Optional[List[str]] = None
    include_basins: Optional[List[str]] = None


@dataclass
class OutputConfig:
    root: str = "data/processed/pilot"
    reports_dir: str = "reports/pipeline"
    format: str = "netcdf4_hdf5"
    compression_level: int = 4
    min_disk_free_gb: float = 2.0


@dataclass
class ProcessingConfig:
    workers: int = 1
    batch_size: int = 16
    resume: bool = True
    preserve_native_grids: bool = True
    atomic_writes: bool = True


@dataclass
class SplittingConfig:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    seed: int = 42


@dataclass
class TargetsConfig:
    horizons_hours: List[int] = field(default_factory=lambda: [6, 12, 18, 24])
    ri_threshold_kt: float = 30.0


@dataclass
class QCConfig:
    ir_mw_max_diff_minutes: float = 60.0
    enforce_causal_environmental: bool = True
    min_valid_pixel_fraction: float = 0.05
    tb_min_k: float = 50.0
    tb_max_k: 350.0 = 350.0
    ir_min_k: float = 150.0
    ir_max_k: float = 350.0


class PipelineConfig:
    """Master configuration container for the TC PRIMED pipeline."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        cfg = config_dict or {}

        ds = cfg.get("dataset", {})
        self.dataset = DatasetConfig(
            root=ds.get("root", "data/raw/pilot"),
            manifest=ds.get("manifest", "reports/pilot_manifest.csv"),
            date_start=ds.get("date_start"),
            date_end=ds.get("date_end"),
            include_storms=ds.get("include_storms"),
            exclude_storms=ds.get("exclude_storms"),
            include_basins=ds.get("include_basins"),
        )

        out = cfg.get("output", {})
        self.output = OutputConfig(
            root=out.get("root", "data/processed/pilot"),
            reports_dir=out.get("reports_dir", "reports/pipeline"),
            format=out.get("format", "netcdf4_hdf5"),
            compression_level=int(out.get("compression_level", 4)),
            min_disk_free_gb=float(out.get("min_disk_free_gb", 2.0)),
        )

        prc = cfg.get("processing", {})
        self.processing = ProcessingConfig(
            workers=int(prc.get("workers", 1)),
            batch_size=int(prc.get("batch_size", 16)),
            resume=bool(prc.get("resume", True)),
            preserve_native_grids=bool(prc.get("preserve_native_grids", True)),
            atomic_writes=bool(prc.get("atomic_writes", True)),
        )

        splt = cfg.get("splitting", {})
        self.splitting = SplittingConfig(
            train=float(splt.get("train", 0.70)),
            validation=float(splt.get("validation", 0.15)),
            test=float(splt.get("test", 0.15)),
            seed=int(splt.get("seed", 42)),
        )

        tgt = cfg.get("targets", {})
        self.targets = TargetsConfig(
            horizons_hours=tgt.get("horizons_hours", [6, 12, 18, 24]),
            ri_threshold_kt=float(tgt.get("ri_threshold_kt", 30.0)),
        )

        qc = cfg.get("quality_control", {})
        self.quality_control = QCConfig(
            ir_mw_max_diff_minutes=float(qc.get("ir_mw_max_diff_minutes", 60.0)),
            enforce_causal_environmental=bool(qc.get("enforce_causal_environmental", True)),
            min_valid_pixel_fraction=float(qc.get("min_valid_pixel_fraction", 0.05)),
            tb_min_k=float(qc.get("tb_min_k", 50.0)),
            tb_max_k=float(qc.get("tb_max_k", 350.0)),
            ir_min_k=float(qc.get("ir_min_k", 150.0)),
            ir_max_k=float(qc.get("ir_max_k", 350.0)),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": {
                "root": self.dataset.root,
                "manifest": self.dataset.manifest,
                "date_start": self.dataset.date_start,
                "date_end": self.dataset.date_end,
                "include_storms": self.dataset.include_storms,
                "exclude_storms": self.dataset.exclude_storms,
                "include_basins": self.dataset.include_basins,
            },
            "output": {
                "root": self.output.root,
                "reports_dir": self.output.reports_dir,
                "format": self.output.format,
                "compression_level": self.output.compression_level,
                "min_disk_free_gb": self.output.min_disk_free_gb,
            },
            "processing": {
                "workers": self.processing.workers,
                "batch_size": self.processing.batch_size,
                "resume": self.processing.resume,
                "preserve_native_grids": self.processing.preserve_native_grids,
                "atomic_writes": self.processing.atomic_writes,
            },
            "splitting": {
                "train": self.splitting.train,
                "validation": self.splitting.validation,
                "test": self.splitting.test,
                "seed": self.splitting.seed,
            },
            "targets": {
                "horizons_hours": self.targets.horizons_hours,
                "ri_threshold_kt": self.targets.ri_threshold_kt,
            },
            "quality_control": {
                "ir_mw_max_diff_minutes": self.quality_control.ir_mw_max_diff_minutes,
                "enforce_causal_environmental": self.quality_control.enforce_causal_environmental,
                "min_valid_pixel_fraction": self.quality_control.min_valid_pixel_fraction,
                "tb_min_k": self.quality_control.tb_min_k,
                "tb_max_k": self.quality_control.tb_max_k,
                "ir_min_k": self.quality_control.ir_min_k,
                "ir_max_k": self.quality_control.ir_max_k,
            },
        }

    def ensure_directories(self) -> None:
        """Create all configured output directories."""
        os.makedirs(self.output.root, exist_ok=True)
        os.makedirs(self.output.reports_dir, exist_ok=True)

    def check_disk_space(self) -> Dict[str, float]:
        """Check available disk space at output location."""
        target_path = self.output.root
        # Find closest existing parent if target_path does not exist yet
        check_dir = target_path
        while check_dir and not os.path.exists(check_dir):
            parent = os.path.dirname(check_dir)
            if parent == check_dir:
                break
            check_dir = parent
        check_dir = check_dir or "."

        total, used, free = shutil.disk_usage(check_dir)
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)

        if free_gb < self.output.min_disk_free_gb:
            raise RuntimeError(
                f"Insufficient disk space on {check_dir}: "
                f"{free_gb:.2f} GB free, but {self.output.min_disk_free_gb:.2f} GB required."
            )

        return {"total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb}

