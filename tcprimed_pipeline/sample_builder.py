"""
Sample Builder module for TC PRIMED.
Builds candidate forecasting samples with verified prediction targets,
task-specific availability auditing, and provenance tracking.
"""

import os
import csv
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from .config import TargetsConfig


class SampleBuilder:
    """Constructs forecasting sample records and compiles task target availability tables."""

    def __init__(self, targets_config: Optional[TargetsConfig] = None):
        self.config = targets_config or TargetsConfig()
        self.samples: List[Dict[str, Any]] = []

    def add_sample(self,
                   observation_id: str,
                   storm_id: str,
                   partition: str,
                   season: int,
                   basin: str,
                   overpass_meta: Dict[str, Any],
                   targets: Dict[str, Any],
                   future_track: Dict[str, Optional[Tuple[float, float]]],
                   modality_flags: Dict[str, bool],
                   env_match: Dict[str, Any],
                   storage_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Record a validated observation sample with its targets and provenance.
        """
        t_epoch = overpass_meta.get("time_epoch")
        t_utc = overpass_meta.get("time_utc", "")

        sample_rec = {
            "sample_id": f"SMP_{observation_id}",
            "observation_id": observation_id,
            "storm_id": storm_id,
            "partition": partition,
            "season": season,
            "basin": basin,
            "timestamp_epoch": t_epoch,
            "timestamp_utc": t_utc,
            "instrument": overpass_meta.get("instrument", ""),
            "platform": overpass_meta.get("platform", ""),
            # Current state targets
            "storm_latitude": targets.get("storm_latitude"),
            "storm_longitude": targets.get("storm_longitude"),
            "intensity_vmax_kt": targets.get("intensity_vmax_kt"),
            "central_min_pressure_hpa": targets.get("central_min_pressure_hpa"),
            "development_stage_raw": targets.get("development_stage_raw", "UNKNOWN"),
            "storm_speed_mps": targets.get("storm_speed_mps"),
            "storm_heading_deg": targets.get("storm_heading_deg"),
            "distance_to_land_km": targets.get("distance_to_land_km"),
            # Future intensity change targets
            "intensity_change_6h_kt": targets.get("intensity_change_6h_kt"),
            "intensity_change_12h_kt": targets.get("intensity_change_12h_kt"),
            "intensity_change_18h_kt": targets.get("intensity_change_18h_kt"),
            "intensity_change_24h_kt": targets.get("intensity_change_24h_kt"),
            "is_rapid_intensification_24h": targets.get("is_rapid_intensification_24h"),
            # Future track positions (from sequential Best Track, None if dissipated)
            "future_lat_6h": future_track.get("future_track_6h", (None, None))[0] if future_track.get("future_track_6h") else None,
            "future_lon_6h": future_track.get("future_track_6h", (None, None))[1] if future_track.get("future_track_6h") else None,
            "future_lat_12h": future_track.get("future_track_12h", (None, None))[0] if future_track.get("future_track_12h") else None,
            "future_lon_12h": future_track.get("future_track_12h", (None, None))[1] if future_track.get("future_track_12h") else None,
            "future_lat_18h": future_track.get("future_track_18h", (None, None))[0] if future_track.get("future_track_18h") else None,
            "future_lon_18h": future_track.get("future_track_18h", (None, None))[1] if future_track.get("future_track_18h") else None,
            "future_lat_24h": future_track.get("future_track_24h", (None, None))[0] if future_track.get("future_track_24h") else None,
            "future_lon_24h": future_track.get("future_track_24h", (None, None))[1] if future_track.get("future_track_24h") else None,
            # Modality presence
            "has_pmw": modality_flags.get("has_pmw", False),
            "has_ir": modality_flags.get("has_ir", False),
            "has_gprof": modality_flags.get("has_gprof", False),
            "has_radar": modality_flags.get("has_radar", False),
            "has_causal_env": env_match.get("has_match", False),
            "env_time_diff_hours": env_match.get("time_diff_hours"),
            # Storage references
            "processed_container_path": storage_info.get("container_path", ""),
            "container_group_path": storage_info.get("group_path", ""),
        }
        self.samples.append(sample_rec)
        return sample_rec

    def export_sample_index(self, output_path: str) -> str:
        """Save sample_index.csv."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not self.samples:
            return output_path

        fields = list(self.samples[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in self.samples:
                writer.writerow(s)

        return output_path

    def export_task_target_availability(self, output_path: str) -> str:
        """
        Save task_target_availability.csv with valid/missing stats by task and partition.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        tasks = [
            ("Current Intensity (V_max)", "intensity_vmax_kt", "kt", "Continuous 1-min sustained wind"),
            ("Current Min Pressure (P_min)", "central_min_pressure_hpa", "hPa", "Omitted in historical agency records"),
            ("Intensity Change (+6h)", "intensity_change_6h_kt", "kt", "Missing near cyclogenesis or dissipation within 6h"),
            ("Intensity Change (+12h)", "intensity_change_12h_kt", "kt", "Missing near cyclogenesis or dissipation within 12h"),
            ("Intensity Change (+18h)", "intensity_change_18h_kt", "kt", "Missing near cyclogenesis or dissipation within 18h"),
            ("Intensity Change (+24h)", "intensity_change_24h_kt", "kt", "Missing near cyclogenesis or dissipation within 24h"),
            ("Rapid Intensification (>=30kt/24h)", "is_rapid_intensification_24h", "binary", "Binary event defined as dV_24h >= 30 kt"),
            ("Future Track (+6h)", "future_lat_6h", "deg", "Extracted from sequential Best Track; missing if dissipated"),
            ("Future Track (+12h)", "future_lat_12h", "deg", "Extracted from sequential Best Track; missing if dissipated"),
            ("Future Track (+18h)", "future_lat_18h", "deg", "Extracted from sequential Best Track; missing if dissipated"),
            ("Future Track (+24h)", "future_lat_24h", "deg", "Extracted from sequential Best Track; missing if dissipated"),
        ]

        partitions = ["ALL", "TRAIN", "VAL", "TEST"]
        out_fields = [
            "task_name", "target_variable", "units", "partition",
            "total_samples", "valid_count", "missing_count", "missing_pct",
            "mean", "std", "median", "min", "max", "scientific_missing_reason"
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()

            for t_name, var_name, units, reason in tasks:
                for part in partitions:
                    if part == "ALL":
                        subset = self.samples
                    else:
                        subset = [s for s in self.samples if s["partition"] == part]

                    n_total = len(subset)
                    vals = [s[var_name] for s in subset if s[var_name] is not None]
                    n_valid = len(vals)
                    n_missing = n_total - n_valid
                    missing_pct = round(100.0 * n_missing / n_total, 2) if n_total > 0 else 0.0

                    if n_valid > 0:
                        arr = np.array(vals, dtype=np.float64)
                        mean_val = round(float(np.mean(arr)), 2)
                        std_val = round(float(np.std(arr)), 2)
                        median_val = round(float(np.median(arr)), 2)
                        min_val = round(float(np.min(arr)), 2)
                        max_val = round(float(np.max(arr)), 2)
                    else:
                        mean_val = std_val = median_val = min_val = max_val = None

                    writer.writerow({
                        "task_name": t_name,
                        "target_variable": var_name,
                        "units": units,
                        "partition": part,
                        "total_samples": n_total,
                        "valid_count": n_valid,
                        "missing_count": n_missing,
                        "missing_pct": missing_pct,
                        "mean": mean_val,
                        "std": std_val,
                        "median": median_val,
                        "min": min_val,
                        "max": max_val,
                        "scientific_missing_reason": reason,
                    })

        return output_path

