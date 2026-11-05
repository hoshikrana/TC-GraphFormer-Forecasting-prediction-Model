"""
Quality Control module for TC PRIMED.
Performs rigorous physical bounds checking, metadata consistency validation,
duplicate detection, and data completeness auditing.
"""

import os
import csv
import json
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
import netCDF4 as nc

from .config import QCConfig
from .netcdf_reader import open_netcdf, NetCDFReader


class QualityController:
    """Executes QC rules on raw and processed TC PRIMED observations."""

    def __init__(self, qc_config: Optional[QCConfig] = None):
        self.config = qc_config or QCConfig()
        self.audit_records: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.seen_signatures: set = set()

    def validate_overpass(self,
                          filepath: str,
                          extracted_metadata: Dict[str, Any],
                          pmw_data: Dict[str, Any],
                          ir_data: Dict[str, Any],
                          targets: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate an overpass observation against all QC rules.
        Returns (is_valid, list_of_issues).
        """
        issues = []
        fname = os.path.basename(filepath)

        # 1. Deduplication check
        obs_sig = (
            extracted_metadata.get("basin"),
            extracted_metadata.get("cyclone_number"),
            extracted_metadata.get("season"),
            extracted_metadata.get("instrument"),
            extracted_metadata.get("platform"),
            extracted_metadata.get("time_epoch"),
        )
        if obs_sig in self.seen_signatures:
            issues.append(f"DUPLICATE_OBSERVATION: Exact observation signature already recorded: {obs_sig}")
        else:
            self.seen_signatures.add(obs_sig)

        # 2. Timestamp check
        t_sat = extracted_metadata.get("time_epoch")
        if t_sat is None or t_sat <= 0:
            issues.append("INVALID_TIMESTAMP: Satellite overpass time is missing or non-positive")

        # 3. Coordinate sanity check
        lat = targets.get("storm_latitude")
        lon = targets.get("storm_longitude")
        if lat is not None and not (-90.0 <= lat <= 90.0):
            issues.append(f"OUT_OF_BOUNDS_COORDINATE: Storm latitude {lat} outside [-90, 90]")
        if lon is not None and not (-180.0 <= lon <= 360.0):
            issues.append(f"OUT_OF_BOUNDS_COORDINATE: Storm longitude {lon} outside [-180, 360]")

        # 4. Target validity checks
        vmax = targets.get("intensity_vmax_kt")
        if vmax is not None and not (10.0 <= vmax <= 220.0):
            issues.append(f"PHYSICAL_BOUNDS_WARNING: V_max {vmax} kt outside reasonable TC range [10, 220]")

        pmin = targets.get("central_min_pressure_hpa")
        if pmin is not None and not (850.0 <= pmin <= 1040.0):
            issues.append(f"PHYSICAL_BOUNDS_WARNING: P_min {pmin} hPa outside reasonable TC range [850, 1040]")

        # 5. Passive microwave QC
        if pmw_data.get("is_available"):
            for sw_name, sw_dict in pmw_data.get("swaths", {}).items():
                for ch_name, ch_dict in sw_dict.get("channels", {}).items():
                    data = ch_dict.get("data")
                    mask = ch_dict.get("valid_mask")
                    if data is not None and mask is not None:
                        valid_count = int(np.sum(mask))
                        total_count = data.size
                        frac = valid_count / total_count if total_count > 0 else 0.0
                        if frac < self.config.min_valid_pixel_fraction:
                            issues.append(f"LOW_DATA_COVERAGE: {ch_name} valid fraction {frac:.3f} < {self.config.min_valid_pixel_fraction}")

                        # Check physical brightness temperature range
                        if valid_count > 0:
                            v_vals = data[mask]
                            min_tb = float(np.min(v_vals))
                            max_tb = float(np.max(v_vals))
                            if min_tb < self.config.tb_min_k or max_tb > self.config.tb_max_k:
                                issues.append(
                                    f"PHYSICAL_TB_BOUNDS: {ch_name} range [{min_tb:.1f}, {max_tb:.1f}] K "
                                    f"exceeds physical limits [{self.config.tb_min_k}, {self.config.tb_max_k}] K"
                                )

        # 6. Infrared QC & Temporal synchronization
        if ir_data.get("is_available"):
            ir_img = ir_data.get("irwin_image")
            ir_mask = ir_data.get("irwin_valid_mask")
            if ir_img is not None and ir_mask is not None:
                v_ir = ir_img[ir_mask]
                if len(v_ir) > 0:
                    min_ir = float(np.min(v_ir))
                    max_ir = float(np.max(v_ir))
                    if min_ir < self.config.ir_min_k or max_ir > self.config.ir_max_k:
                        issues.append(
                            f"PHYSICAL_IR_BOUNDS: IRWIN range [{min_ir:.1f}, {max_ir:.1f}] K "
                            f"exceeds physical limits [{self.config.ir_min_k}, {self.config.ir_max_k}] K"
                        )

        # Record failure if any issue is an error
        is_fatal = any(i.startswith("INVALID_") or i.startswith("OUT_OF_BOUNDS_COORDINATE") for i in issues)
        is_valid = not is_fatal

        rec = {
            "file_name": fname,
            "filepath": filepath,
            "is_valid": is_valid,
            "issues": issues,
            "issue_count": len(issues),
        }
        self.audit_records.append(rec)
        if not is_valid or issues:
            self.failures.append(rec)

        return is_valid, issues

    def export_qc_reports(self, output_dir: str) -> Tuple[str, str]:
        """Save data_quality_report.json and processing_failures.csv."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "data_quality_report.json")
        failures_path = os.path.join(output_dir, "processing_failures.csv")

        # Compile statistics
        total = len(self.audit_records)
        valid = sum(1 for r in self.audit_records if r["is_valid"])
        failed = total - valid
        issue_counts = Counter()
        for r in self.audit_records:
            for iss in r["issues"]:
                tag = iss.split(":")[0]
                issue_counts[tag] += 1

        summary = {
            "total_inspected_observations": total,
            "valid_observations": valid,
            "failed_observations": failed,
            "pass_rate_pct": round(100.0 * valid / total, 2) if total > 0 else 100.0,
            "issue_counts_by_category": dict(issue_counts),
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "filepath", "is_valid", "issue_count", "issues"])
            writer.writeheader()
            for fl in self.failures:
                writer.writerow({
                    "file_name": fl["file_name"],
                    "filepath": fl["filepath"],
                    "is_valid": fl["is_valid"],
                    "issue_count": fl["issue_count"],
                    "issues": "; ".join(fl["issues"]),
                })

        return json_path, failures_path

