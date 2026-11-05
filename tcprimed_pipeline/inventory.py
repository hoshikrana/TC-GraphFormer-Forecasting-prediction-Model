"""
Inventory module for TC PRIMED.
Discovers, catalogues, and indexes raw pilot NetCDF files across storm folders.
"""

import os
import re
import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from .config import PipelineConfig


def is_environmental_filename(filename: str) -> bool:
    """Determine if a filename represents an environmental NetCDF file."""
    base = os.path.basename(filename)
    return "_env_" in base or base.endswith("_env.nc") or "environmental" in base.lower()


def parse_filename_metadata(filename: str) -> Dict[str, Any]:
    """
    Parse metadata fields from a standard TC PRIMED NetCDF filename.
    Format Overpass:
      TCPRIMED_v01r01-final_{storm_id}_{instrument}_{platform}_{granule}_{timestamp}.nc
      Example: TCPRIMED_v01r01-final_WP082022_GMI_GPM_047909_20220804082925.nc
    Format Environmental:
      TCPRIMED_v01r01-final_{storm_id}_env_s{start}_e{end}.nc
      Example: TCPRIMED_v01r01-final_CP011987_env_s19870826000000_e19870829180000.nc
    """
    base = os.path.basename(filename)
    if not base.endswith(".nc"):
        return {"is_valid": False, "filename": base}

    is_env = is_environmental_filename(base)
    clean_name = base[:-3]  # Strip .nc
    parts = clean_name.split("_")

    # Environmental file
    if is_env:
        # Expected parts: ['TCPRIMED', 'v01r01-final', storm_id, 'env', 's...', 'e...']
        storm_id = parts[2] if len(parts) >= 3 else "UNKNOWN"
        basin = storm_id[:2] if len(storm_id) >= 2 else "UN"
        season = int(storm_id[4:8]) if len(storm_id) >= 8 and storm_id[4:8].isdigit() else 0
        cyclone_num = storm_id[2:4] if len(storm_id) >= 4 else "00"
        return {
            "is_valid": True,
            "filename": base,
            "is_environmental": True,
            "storm_id": storm_id,
            "basin": basin,
            "season": season,
            "cyclone_number": cyclone_num,
            "instrument": "ERA5",
            "platform": "reanalysis",
            "granule_number": None,
            "timestamp_str": None,
            "observation_time_utc": None,
        }

    # Overpass file
    # Expected parts: ['TCPRIMED', 'v01r01-final', storm_id, instrument, platform, granule, timestamp]
    if len(parts) >= 7:
        storm_id = parts[2]
        basin = storm_id[:2]
        season = int(storm_id[4:8]) if storm_id[4:8].isdigit() else 0
        cyclone_num = storm_id[2:4]
        instrument = parts[3]
        platform = parts[4]
        granule = parts[5]
        timestamp_str = parts[6]

        obs_time_utc = None
        if len(timestamp_str) >= 14 and timestamp_str[:14].isdigit():
            try:
                dt = datetime.strptime(timestamp_str[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                obs_time_utc = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except ValueError:
                pass

        return {
            "is_valid": True,
            "filename": base,
            "is_environmental": False,
            "storm_id": storm_id,
            "basin": basin,
            "season": season,
            "cyclone_number": cyclone_num,
            "instrument": instrument,
            "platform": platform,
            "granule_number": granule,
            "timestamp_str": timestamp_str,
            "observation_time_utc": obs_time_utc,
        }

    # Fallback for short or alternative naming
    storm_id = parts[2] if len(parts) >= 3 else "UNKNOWN"
    return {
        "is_valid": True,
        "filename": base,
        "is_environmental": False,
        "storm_id": storm_id,
        "basin": storm_id[:2] if len(storm_id) >= 2 else "UN",
        "season": int(storm_id[4:8]) if len(storm_id) >= 8 and storm_id[4:8].isdigit() else 0,
        "cyclone_number": storm_id[2:4] if len(storm_id) >= 4 else "00",
        "instrument": parts[3] if len(parts) >= 4 else "UNKNOWN",
        "platform": parts[4] if len(parts) >= 5 else "UNKNOWN",
        "granule_number": parts[5] if len(parts) >= 6 else "UNKNOWN",
        "timestamp_str": parts[6] if len(parts) >= 7 else None,
        "observation_time_utc": None,
    }


class DatasetInventory:
    """Manages raw dataset indexing, filtering, and manifest generation."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.files: List[Dict[str, Any]] = []
        self.storms: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "storm_id": "",
            "season": 0,
            "basin": "",
            "overpass_files": [],
            "env_file": None,
        })

    def scan(self) -> "DatasetInventory":
        """
        Scan dataset files from disk or manifest according to configuration filters.
        """
        root = self.config.dataset.root
        manifest_path = self.config.dataset.manifest

        raw_records = []
        # Prefer manifest if it exists
        if manifest_path and os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loc = row.get("local_destination", "")
                    # Normalize path
                    norm_path = os.path.normpath(loc) if loc else ""
                    if not os.path.exists(norm_path):
                        # Try relative to project root or dataset root
                        candidate = os.path.join(root, os.path.basename(loc))
                        if os.path.exists(candidate):
                            norm_path = candidate
                        else:
                            # Reconstruct from s3_key
                            s3_k = row.get("s3_key", "")
                            rel = s3_k[len("v01r01/final/"):] if s3_k.startswith("v01r01/final/") else s3_k
                            candidate = os.path.join(root, rel.replace("/", os.sep))
                            if os.path.exists(candidate):
                                norm_path = candidate
                    
                    if os.path.exists(norm_path):
                        sz = int(row.get("size_bytes", os.path.getsize(norm_path)))
                        raw_records.append((norm_path, sz))
        else:
            # Walk directory directly
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if fname.endswith(".nc"):
                        fp = os.path.join(dirpath, fname)
                        raw_records.append((fp, os.path.getsize(fp)))

        # Process and filter records
        self.files = []
        self.storms = defaultdict(lambda: {
            "storm_id": "",
            "season": 0,
            "basin": "",
            "overpass_files": [],
            "env_file": None,
        })

        for fpath, fsize in raw_records:
            meta = parse_filename_metadata(fpath)
            if not meta["is_valid"]:
                continue

            sid = meta["storm_id"]
            basin = meta["basin"]
            season = meta["season"]

            # Apply filters
            if self.config.dataset.include_storms and sid not in self.config.dataset.include_storms:
                continue
            if self.config.dataset.exclude_storms and sid in self.config.dataset.exclude_storms:
                continue
            if self.config.dataset.include_basins and basin not in self.config.dataset.include_basins:
                continue
            if self.config.dataset.date_start and meta["timestamp_str"] and meta["timestamp_str"][:8] < self.config.dataset.date_start:
                continue
            if self.config.dataset.date_end and meta["timestamp_str"] and meta["timestamp_str"][:8] > self.config.dataset.date_end:
                continue

            record = {
                "file_path": fpath,
                "file_name": os.path.basename(fpath),
                "file_size_bytes": fsize,
                "file_type": "environmental" if meta["is_environmental"] else "overpass",
                "storm_id": sid,
                "season": season,
                "basin": basin,
                "cyclone_number": meta["cyclone_number"],
                "instrument": meta["instrument"],
                "platform": meta["platform"],
                "granule_number": meta["granule_number"],
                "timestamp_str": meta["timestamp_str"],
                "observation_time_utc": meta["observation_time_utc"],
            }
            self.files.append(record)

            st = self.storms[sid]
            st["storm_id"] = sid
            st["season"] = season
            st["basin"] = basin
            if meta["is_environmental"]:
                st["env_file"] = record
            else:
                st["overpass_files"].append(record)

        return self

    def generate_manifests(self, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Write file_manifest.csv and observation_manifest.csv to output_dir.
        """
        out_dir = output_dir or self.config.output.reports_dir
        os.makedirs(out_dir, exist_ok=True)

        file_manifest_path = os.path.join(out_dir, "file_manifest.csv")
        obs_manifest_path = os.path.join(out_dir, "observation_manifest.csv")

        # 1. file_manifest.csv
        file_fields = [
            "file_path", "file_name", "file_size_bytes", "file_type",
            "storm_id", "season", "basin", "cyclone_number",
            "instrument", "platform", "granule_number",
            "timestamp_str", "observation_time_utc"
        ]
        with open(file_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=file_fields)
            writer.writeheader()
            for rec in self.files:
                writer.writerow({k: rec.get(k, "") for k in file_fields})

        # 2. observation_manifest.csv (all overpass observations paired with storm env file)
        obs_fields = [
            "observation_id", "storm_id", "season", "basin", "cyclone_number",
            "instrument", "platform", "granule_number", "timestamp_str",
            "observation_time_utc", "overpass_file_path", "overpass_file_size",
            "has_environmental_file", "env_file_path"
        ]
        with open(obs_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=obs_fields)
            writer.writeheader()
            for sid, sdata in sorted(self.storms.items()):
                env_rec = sdata["env_file"]
                has_env = env_rec is not None
                env_path = env_rec["file_path"] if has_env else ""

                for op in sdata["overpass_files"]:
                    obs_id = f"{sid}_{op['instrument']}_{op['platform']}_{op['timestamp_str']}"
                    writer.writerow({
                        "observation_id": obs_id,
                        "storm_id": sid,
                        "season": op["season"],
                        "basin": op["basin"],
                        "cyclone_number": op["cyclone_number"],
                        "instrument": op["instrument"],
                        "platform": op["platform"],
                        "granule_number": op["granule_number"],
                        "timestamp_str": op["timestamp_str"],
                        "observation_time_utc": op["observation_time_utc"],
                        "overpass_file_path": op["file_path"],
                        "overpass_file_size": op["file_size_bytes"],
                        "has_environmental_file": has_env,
                        "env_file_path": env_path,
                    })

        return file_manifest_path, obs_manifest_path

