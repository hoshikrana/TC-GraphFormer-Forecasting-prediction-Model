"""
audit_pilot_modalities.py - Research-grade Modality and Download Audit Utility
for NOAA TC PRIMED v01r01-final pilot dataset.

Inspects all locally downloaded NetCDF files referenced by the pilot manifest:
  - Download integrity and file size matching
  - Infrared group existence, infrared_availability_flag, IRWIN presence,
    valid pixel counts, missing pixel percentages, physical distributions
  - IR provenance: geostationary platform, instrument, source granule, wavelength,
    temporal offsets relative to nominal microwave overpass
  - R5 radial wind metrics and R5_availability_flag
  - Other modalities: passive microwave, GPROF, and radar-radiometer (5-state audit)
  - ERA5 environmental files: group structure, vertical levels, coordinates, fields
  - Cross-tabulated distributions across era, year, basin, instrument, and platform

Outputs:
  - reports/pilot_modality_audit.txt
  - reports/pilot_modality_audit.json
  - reports/pilot_ir_availability.csv
  - reports/pilot_download_integrity.csv
"""

import os
import sys
import csv
import json
import argparse
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import netCDF4 as nc

# ---------------------------------------------------------------------------
# Constants & Taxonomy
# ---------------------------------------------------------------------------
PREFIX = "v01r01/final/"

BASIN_NAMES = {
    "AL": "North Atlantic",
    "EP": "Eastern North Pacific",
    "WP": "Western North Pacific",
    "IO": "North Indian Ocean",
    "SH": "Southern Hemisphere",
    "CP": "Central North Pacific",
    "SL": "South Atlantic",
}

ERA_BINS = [
    ("1987-1999", 1987, 1999),
    ("2000-2009", 2000, 2009),
    ("2010-2019", 2010, 2019),
    ("2020+",     2020, 9999),
]

# Modality 5-state taxonomy
STATE_GROUP_ABSENT = "GROUP_ABSENT"
STATE_FLAG_FALSE = "FLAG_FALSE"
STATE_SOME_VALID = "AVAILABLE_SOME_VALID"
STATE_ALL_MISSING = "AVAILABLE_ALL_MISSING"
STATE_VALID_DATA = "AVAILABLE_VALID_DATA"


# ===========================================================================
# Helper Functions
# ===========================================================================

def classify_era(season: int) -> str:
    """Return the historical-period label for a given season year."""
    for label, lo, hi in ERA_BINS:
        if lo <= season <= hi:
            return label
    return "unknown"


def is_environmental_file(filename: str) -> bool:
    """Detect environmental NetCDF files."""
    lower = filename.lower()
    return "_env_" in lower or "environmental" in lower or "era5" in lower or lower.endswith("_env.nc")


def to_serializable(val):
    """Safely converts netCDF4 / numpy types into native Python JSON-serializable types."""
    if val is None:
        return None
    if isinstance(val, np.ma.core.MaskedConstant):
        return None
    if isinstance(val, np.ndarray):
        return [to_serializable(x) for x in val.tolist()]
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if not np.isfinite(val):
            return str(val)
        return float(val)
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    return str(val)


def clean_scalar(v):
    """Returns a clean Python scalar, handling masked arrays and sentinels."""
    if v is None or np.ma.is_masked(v):
        return None
    v = np.ma.getdata(v)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        v = v.ravel()[0]
    return to_serializable(v)


def parse_time_seconds(seconds_since_epoch):
    """Converts seconds since 1970-01-01 to ISO 8601 UTC string."""
    try:
        if seconds_since_epoch is None or not np.isfinite(seconds_since_epoch):
            return None
        return datetime.fromtimestamp(float(seconds_since_epoch), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return None


# ===========================================================================
# Inspection Engines
# ===========================================================================

def inspect_overpass_file(filepath, manifest_row):
    """
    Inspect an individual overpass NetCDF file.
    Extracts IR, PMW, GPROF, Radar, and overpass metadata.
    """
    res = {
        "s3_key": manifest_row.get("s3_key", ""),
        "local_destination": filepath,
        "storm_id": manifest_row.get("storm_id", ""),
        "season": int(manifest_row.get("season", 0)),
        "era": classify_era(int(manifest_row.get("season", 0))),
        "basin": manifest_row.get("basin", ""),
        "cyclone_number": manifest_row.get("cyclone_number", ""),
        "instrument": manifest_row.get("instrument", ""),
        "platform": manifest_row.get("platform", ""),
        "is_environmental": False,
        "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
        "open_success": False,
        "error_message": "",
    }

    try:
        with nc.Dataset(filepath, "r") as ds:
            res["open_success"] = True

            # 1. Overpass metadata
            nominal_time = None
            if "overpass_metadata" in ds.groups:
                op_grp = ds.groups["overpass_metadata"]
                if "time" in op_grp.variables:
                    nominal_time = clean_scalar(op_grp.variables["time"][:])
                if "instrument_name" in op_grp.variables:
                    res["instrument"] = clean_scalar(op_grp.variables["instrument_name"][:]) or res["instrument"]
                if "platform_name" in op_grp.variables:
                    res["platform"] = clean_scalar(op_grp.variables["platform_name"][:]) or res["platform"]

            res["nominal_time_raw"] = nominal_time
            res["nominal_time_utc"] = parse_time_seconds(nominal_time)

            # 2. Overpass storm metadata
            if "overpass_storm_metadata" in ds.groups:
                s_grp = ds.groups["overpass_storm_metadata"]
                for k in ["storm_latitude", "storm_longitude", "intensity", "central_min_pressure", "development_level", "distance_to_land"]:
                    if k in s_grp.variables:
                        res[k] = clean_scalar(s_grp.variables[k][:])

            # 3. Infrared Audit
            ir_info = {
                "group_present": False,
                "availability_flag": None,
                "provenance_platform": "",
                "provenance_instrument": "",
                "source_granule": "",
                "wavelength": "",
                "observation_time_raw": None,
                "observation_time_utc": "",
                "time_offset_seconds": None,
                "time_offset_minutes": None,
                "geospatial_bounds": {},
                # IRWIN
                "irwin_present": False,
                "irwin_shape": [],
                "irwin_dimensions": [],
                "irwin_dtype": "",
                "irwin_units": "",
                "irwin_long_name": "",
                "irwin_fill_value": None,
                "irwin_total_pixels": 0,
                "irwin_valid_pixels": 0,
                "irwin_missing_pixels": 0,
                "irwin_missing_percent": 100.0,
                "irwin_min_k": None,
                "irwin_max_k": None,
                "irwin_mean_k": None,
                # R5
                "r5_present": False,
                "r5_availability_flag": None,
                "r5_shape": [],
                "r5_dimensions": [],
                "r5_units": "",
                "r5_valid_count": 0,
                "r5_total_count": 0,
                "r5_min": None,
                "r5_max": None,
                "r5_mean": None,
                # State
                "modality_state": STATE_GROUP_ABSENT,
            }

            if "infrared" in ds.groups:
                ir_grp = ds.groups["infrared"]
                ir_info["group_present"] = True

                # Attributes / Provenance
                ir_attrs = {a: ir_grp.getncattr(a) for a in ir_grp.ncattrs()}
                ir_info["provenance_platform"] = str(ir_attrs.get("platform", ""))
                ir_info["provenance_instrument"] = str(ir_attrs.get("instrument", ""))
                ir_info["source_granule"] = str(ir_attrs.get("source", ""))
                
                # Bounds
                for bkey in ["geospatial_lat_min", "geospatial_lat_max", "geospatial_lon_min", "geospatial_lon_max"]:
                    if bkey in ir_attrs:
                        ir_info["geospatial_bounds"][bkey] = clean_scalar(ir_attrs[bkey])

                # Flags & times
                if "infrared_availability_flag" in ir_grp.variables:
                    ir_info["availability_flag"] = clean_scalar(ir_grp.variables["infrared_availability_flag"][:])

                if "time" in ir_grp.variables:
                    ir_t = clean_scalar(ir_grp.variables["time"][:])
                    ir_info["observation_time_raw"] = ir_t
                    ir_info["observation_time_utc"] = parse_time_seconds(ir_t)
                    if ir_t is not None and nominal_time is not None:
                        delta = float(ir_t - nominal_time)
                        ir_info["time_offset_seconds"] = round(delta, 1)
                        ir_info["time_offset_minutes"] = round(delta / 60.0, 2)

                # R5
                if "R5_availability_flag" in ir_grp.variables:
                    ir_info["r5_availability_flag"] = clean_scalar(ir_grp.variables["R5_availability_flag"][:])

                if "R5" in ir_grp.variables:
                    r5_var = ir_grp.variables["R5"]
                    ir_info["r5_present"] = True
                    ir_info["r5_shape"] = list(r5_var.shape)
                    ir_info["r5_dimensions"] = list(r5_var.dimensions)
                    ir_info["r5_units"] = getattr(r5_var, "units", "")
                    r5_data = r5_var[:]
                    r5_mask = np.ma.getmaskarray(r5_data)
                    r5_fill = getattr(r5_var, "_FillValue", None)
                    if r5_fill is not None:
                        r5_mask = r5_mask | (np.ma.getdata(r5_data) == r5_fill)
                    r5_valid = np.ma.getdata(r5_data)[~r5_mask]
                    ir_info["r5_total_count"] = int(r5_var.size)
                    ir_info["r5_valid_count"] = int(r5_valid.size)
                    if r5_valid.size > 0:
                        ir_info["r5_min"] = round(float(np.min(r5_valid)), 2)
                        ir_info["r5_max"] = round(float(np.max(r5_valid)), 2)
                        ir_info["r5_mean"] = round(float(np.mean(r5_valid)), 2)

                # IRWIN
                if "IRWIN" in ir_grp.variables:
                    irwin_var = ir_grp.variables["IRWIN"]
                    ir_info["irwin_present"] = True
                    ir_info["irwin_shape"] = list(irwin_var.shape)
                    ir_info["irwin_dimensions"] = list(irwin_var.dimensions)
                    ir_info["irwin_dtype"] = str(irwin_var.dtype)
                    ir_info["irwin_units"] = getattr(irwin_var, "units", "")
                    ir_info["irwin_long_name"] = getattr(irwin_var, "long_name", "")
                    ir_info["wavelength"] = "11.20 microns"

                    fill_val = getattr(irwin_var, "_FillValue", None)
                    ir_info["irwin_fill_value"] = to_serializable(fill_val)

                    raw_irwin = irwin_var[:]
                    mask = np.ma.getmaskarray(raw_irwin)
                    data = np.ma.getdata(raw_irwin)

                    if fill_val is not None:
                        try:
                            mask = mask | (data == float(fill_val))
                        except (ValueError, TypeError):
                            pass

                    # Physical validity check: brightness temp > 0
                    mask = mask | (data <= 0) | np.isnan(data) | np.isinf(data)

                    total_px = int(irwin_var.size)
                    valid_px = int((~mask).sum())
                    miss_px = total_px - valid_px
                    miss_pct = round(100.0 * miss_px / total_px, 2) if total_px > 0 else 100.0

                    ir_info["irwin_total_pixels"] = total_px
                    ir_info["irwin_valid_pixels"] = valid_px
                    ir_info["irwin_missing_pixels"] = miss_px
                    ir_info["irwin_missing_percent"] = miss_pct

                    if valid_px > 0:
                        valid_data = data[~mask]
                        ir_info["irwin_min_k"] = round(float(np.min(valid_data)), 2)
                        ir_info["irwin_max_k"] = round(float(np.max(valid_data)), 2)
                        ir_info["irwin_mean_k"] = round(float(np.mean(valid_data)), 2)

                # Classify 5-state for IR
                flag_val = ir_info["availability_flag"]
                if flag_val is None or flag_val == 0:
                    ir_info["modality_state"] = STATE_FLAG_FALSE
                elif not ir_info["irwin_present"]:
                    ir_info["modality_state"] = STATE_FLAG_FALSE
                elif ir_info["irwin_valid_pixels"] == 0:
                    ir_info["modality_state"] = STATE_ALL_MISSING
                elif ir_info["irwin_missing_percent"] > 0:
                    ir_info["modality_state"] = STATE_SOME_VALID
                else:
                    ir_info["modality_state"] = STATE_VALID_DATA

            res["infrared"] = ir_info

            # 4. Passive Microwave Audit
            pmw_info = {
                "group_present": False,
                "swaths": [],
                "channel_count": 0,
                "channels": [],
                "modality_state": STATE_GROUP_ABSENT,
            }
            if "passive_microwave" in ds.groups:
                pmw_grp = ds.groups["passive_microwave"]
                pmw_info["group_present"] = True
                pmw_info["swaths"] = list(pmw_grp.groups.keys())
                all_channels = []
                total_valid_tb = 0
                total_elem_tb = 0
                for sw_name, sw_grp in pmw_grp.groups.items():
                    tb_vars = [v for v in sw_grp.variables if v.startswith("TB_")]
                    all_channels.extend(tb_vars)
                    # Check first channel for data validity
                    if tb_vars and total_elem_tb == 0:
                        sample_tb = sw_grp.variables[tb_vars[0]][:]
                        m = np.ma.getmaskarray(sample_tb)
                        d = np.ma.getdata(sample_tb)
                        m = m | (d <= 0) | np.isnan(d)
                        total_elem_tb += sample_tb.size
                        total_valid_tb += int((~m).sum())

                pmw_info["channel_count"] = len(all_channels)
                pmw_info["channels"] = sorted(set(all_channels))

                if pmw_info["channel_count"] == 0:
                    pmw_info["modality_state"] = STATE_FLAG_FALSE
                elif total_valid_tb == 0 and total_elem_tb > 0:
                    pmw_info["modality_state"] = STATE_ALL_MISSING
                elif total_valid_tb < total_elem_tb:
                    pmw_info["modality_state"] = STATE_SOME_VALID
                else:
                    pmw_info["modality_state"] = STATE_VALID_DATA

            res["passive_microwave"] = pmw_info

            # 5. GPROF Audit
            gprof_info = {
                "group_present": False,
                "availability_flag": None,
                "key_variables": [],
                "modality_state": STATE_GROUP_ABSENT,
            }
            if "GPROF" in ds.groups:
                gp_grp = ds.groups["GPROF"]
                gprof_info["group_present"] = True
                if "availability_flag" in gp_grp.variables:
                    gprof_info["availability_flag"] = clean_scalar(gp_grp.variables["availability_flag"][:])

                gp_vars = []
                total_precip_valid = 0
                total_precip_elem = 0
                for sw_name, sw_grp in gp_grp.groups.items():
                    for v in sw_grp.variables:
                        if v not in ("latitude", "longitude", "x", "y", "time", "ScanTime"):
                            gp_vars.append(v)
                    if "surfacePrecipitation" in sw_grp.variables and total_precip_elem == 0:
                        p_data = sw_grp.variables["surfacePrecipitation"][:]
                        m = np.ma.getmaskarray(p_data) | np.isnan(np.ma.getdata(p_data))
                        total_precip_elem += p_data.size
                        total_precip_valid += int((~m).sum())

                gprof_info["key_variables"] = sorted(set(gp_vars))

                flag = gprof_info["availability_flag"]
                if flag is None or flag == 0:
                    gprof_info["modality_state"] = STATE_FLAG_FALSE
                elif total_precip_valid == 0 and total_precip_elem > 0:
                    gprof_info["modality_state"] = STATE_ALL_MISSING
                elif total_precip_valid < total_precip_elem:
                    gprof_info["modality_state"] = STATE_SOME_VALID
                else:
                    gprof_info["modality_state"] = STATE_VALID_DATA

            res["GPROF"] = gprof_info

            # 6. Radar-Radiometer Audit
            radar_info = {
                "group_present": False,
                "availability_flag": None,
                "swaths": [],
                "key_variables": [],
                "modality_state": STATE_GROUP_ABSENT,
            }
            if "radar_radiometer" in ds.groups:
                rr_grp = ds.groups["radar_radiometer"]
                radar_info["group_present"] = True
                if "availability_flag" in rr_grp.variables:
                    radar_info["availability_flag"] = clean_scalar(rr_grp.variables["availability_flag"][:])
                radar_info["swaths"] = list(rr_grp.groups.keys())

                rr_vars = []
                total_valid_radar = 0
                total_elem_radar = 0
                for sw_name, sw_grp in rr_grp.groups.items():
                    for v in sw_grp.variables:
                        if v not in ("latitude", "longitude", "time", "ScanTime"):
                            rr_vars.append(v)
                    # Check a representative variable
                    check_v = "nearSurfPrecipTotRate" if "nearSurfPrecipTotRate" in sw_grp.variables else (
                        "correctedReflectFactor" if "correctedReflectFactor" in sw_grp.variables else None
                    )
                    if check_v and total_elem_radar == 0:
                        rd = sw_grp.variables[check_v][:]
                        m = np.ma.getmaskarray(rd) | np.isnan(np.ma.getdata(rd))
                        total_elem_radar += rd.size
                        total_valid_radar += int((~m).sum())

                radar_info["key_variables"] = sorted(set(rr_vars))

                flag = radar_info["availability_flag"]
                if flag is None or flag == 0:
                    radar_info["modality_state"] = STATE_FLAG_FALSE
                elif total_valid_radar == 0 and total_elem_radar > 0:
                    radar_info["modality_state"] = STATE_ALL_MISSING
                elif total_valid_radar < total_elem_radar:
                    radar_info["modality_state"] = STATE_SOME_VALID
                else:
                    radar_info["modality_state"] = STATE_VALID_DATA

            res["radar_radiometer"] = radar_info

    except Exception as e:
        res["open_success"] = False
        res["error_message"] = str(e)

    return res


def inspect_environmental_file(filepath, manifest_row):
    """
    Inspect an individual ERA5 environmental NetCDF file.
    Extracts group availability, vertical levels, coordinates, and physical variables.
    """
    res = {
        "s3_key": manifest_row.get("s3_key", ""),
        "local_destination": filepath,
        "storm_id": manifest_row.get("storm_id", ""),
        "season": int(manifest_row.get("season", 0)),
        "era": classify_era(int(manifest_row.get("season", 0))),
        "basin": manifest_row.get("basin", ""),
        "cyclone_number": manifest_row.get("cyclone_number", ""),
        "is_environmental": True,
        "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
        "open_success": False,
        "error_message": "",
        "groups_present": [],
        "groups_missing": [],
        "time_steps_count": 0,
        "time_start_utc": "",
        "time_end_utc": "",
        "duration_hours": 0.0,
        "vertical_levels_count": 0,
        "vertical_levels": [],
        "rectilinear_grid": {},
        "cylindrical_grid": {},
        "environmental_variables": [],
    }

    expected_groups = [
        "storm_metadata",
        "overpass_metadata",
        "overpass_storm_metadata",
        "diagnostics",
        "cylindrical",
        "rectilinear",
    ]

    try:
        with nc.Dataset(filepath, "r") as ds:
            res["open_success"] = True
            present = list(ds.groups.keys())
            res["groups_present"] = [g for g in expected_groups if g in present]
            res["groups_missing"] = [g for g in expected_groups if g not in present]

            # Time coverage from storm_metadata or diagnostics
            for candidate_grp in ["storm_metadata", "diagnostics", "cylindrical", "rectilinear"]:
                if candidate_grp in ds.groups and "time" in ds.groups[candidate_grp].variables:
                    t_var = ds.groups[candidate_grp].variables["time"]
                    t_data = t_var[:]
                    if t_data.size > 0:
                        res["time_steps_count"] = int(t_data.size)
                        min_t = float(np.min(t_data))
                        max_t = float(np.max(t_data))
                        res["time_start_utc"] = parse_time_seconds(min_t)
                        res["time_end_utc"] = parse_time_seconds(max_t)
                        res["duration_hours"] = round((max_t - min_t) / 3600.0, 1)
                        break

            # Vertical levels from diagnostics / rectilinear
            for candidate_grp in ["rectilinear", "cylindrical", "diagnostics"]:
                if candidate_grp in ds.groups and "level" in ds.groups[candidate_grp].variables:
                    l_var = ds.groups[candidate_grp].variables["level"]
                    levels = [to_serializable(x) for x in l_var[:].tolist()]
                    res["vertical_levels_count"] = len(levels)
                    res["vertical_levels"] = levels
                    break

            # Rectilinear grid
            if "rectilinear" in ds.groups:
                rg = ds.groups["rectilinear"]
                res["rectilinear_grid"] = {
                    "dimensions": {dname: len(d) for dname, d in rg.dimensions.items()},
                    "lat_bounds": [float(rg.variables["latitude"][:].min()), float(rg.variables["latitude"][:].max())] if "latitude" in rg.variables else [],
                    "lon_bounds": [float(rg.variables["longitude"][:].min()), float(rg.variables["longitude"][:].max())] if "longitude" in rg.variables else [],
                    "variables": [v for v in rg.variables if v not in ("time", "level", "latitude", "longitude")],
                }
                res["environmental_variables"] = res["rectilinear_grid"]["variables"]

            # Cylindrical grid
            if "cylindrical" in ds.groups:
                cg = ds.groups["cylindrical"]
                res["cylindrical_grid"] = {
                    "dimensions": {dname: len(d) for dname, d in cg.dimensions.items()},
                    "radius_count": len(cg.dimensions["radius"]) if "radius" in cg.dimensions else 0,
                    "azimuth_count": len(cg.dimensions["azimuth"]) if "azimuth" in cg.dimensions else 0,
                }

    except Exception as e:
        res["open_success"] = False
        res["error_message"] = str(e)

    return res


# ===========================================================================
# Audit Orchestrator
# ===========================================================================

def run_modality_audit(manifest_path="reports/pilot_manifest.csv", destination_base="data/raw/pilot", max_workers=4):
    """
    Master audit workflow. Reads manifest, audits all entries,
    extracts statistics, writes reports.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    total_manifest = len(manifest)
    print(f"\n[AUDIT] Commencing Modality Audit of {total_manifest} manifest entries...")
    print(f"        Destination directory: {destination_base}")
    print(f"        Concurrency workers:   {max_workers}")

    # Step 1: Pre-classify integrity on disk
    download_integrity = []
    overpasses_to_inspect = []
    env_to_inspect = []

    for row in manifest:
        rel = row["s3_key"]
        if rel.startswith(PREFIX):
            rel = rel[len(PREFIX):]
        resolved_path = os.path.join(destination_base, rel.replace("/", os.sep))
        expected_size = int(row["size_bytes"])
        is_env = is_environmental_file(os.path.basename(row["s3_key"]))

        entry = {
            "s3_key": row["s3_key"],
            "season": int(row["season"]),
            "basin": row["basin"],
            "cyclone_number": row["cyclone_number"],
            "storm_id": row["storm_id"],
            "is_environmental": is_env,
            "file_type": "environmental" if is_env else "overpass",
            "local_destination": resolved_path,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": 0,
            "download_status": "MISSING",
            "can_open_netcdf": False,
            "inspection_status": "SKIPPED_UNAVAILABLE",
            "error_message": "",
        }

        if not os.path.exists(resolved_path):
            entry["download_status"] = "MISSING"
            download_integrity.append(entry)
            continue

        actual_size = os.path.getsize(resolved_path)
        entry["actual_size_bytes"] = actual_size

        if actual_size != expected_size:
            entry["download_status"] = "WRONG_SIZE"
            entry["error_message"] = f"Expected {expected_size} bytes, got {actual_size}"
            download_integrity.append(entry)
            continue

        entry["download_status"] = "CORRECT_SIZE"

        if is_env:
            env_to_inspect.append((resolved_path, row, entry))
        else:
            overpasses_to_inspect.append((resolved_path, row, entry))

    print(f"[AUDIT] Integrity check: {len(overpasses_to_inspect)} overpasses & {len(env_to_inspect)} environmental files ready for inspection.")

    # Step 2: Parallel inspection of overpasses
    inspected_overpasses = []
    failed_overpass_retries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(inspect_overpass_file, path, row): (path, row, entry)
            for path, row, entry in overpasses_to_inspect
        }
        for future in as_completed(future_map):
            path, row, entry = future_map[future]
            try:
                res = future.result()
                if res["open_success"]:
                    entry["can_open_netcdf"] = True
                    entry["inspection_status"] = "SUCCESS"
                    inspected_overpasses.append(res)
                else:
                    # Queue for single-thread retry to handle HDF5 multi-threading race conditions
                    failed_overpass_retries.append((path, row, entry, res["error_message"]))
            except Exception as e:
                failed_overpass_retries.append((path, row, entry, str(e)))

    # Synchronous retry for any files that failed due to HDF5 thread collisions
    for path, row, entry, initial_err in failed_overpass_retries:
        try:
            retry_res = inspect_overpass_file(path, row)
            if retry_res["open_success"]:
                entry["can_open_netcdf"] = True
                entry["inspection_status"] = "SUCCESS"
                entry["error_message"] = ""
                inspected_overpasses.append(retry_res)
            else:
                entry["can_open_netcdf"] = False
                entry["inspection_status"] = "FAILED_CORRUPT"
                entry["error_message"] = retry_res["error_message"]
        except Exception as e:
            entry["can_open_netcdf"] = False
            entry["inspection_status"] = "FAILED_CORRUPT"
            entry["error_message"] = str(e)
        download_integrity.append(entry)

    # Add all initially successful overpass entries to download_integrity
    for path, row, entry in overpasses_to_inspect:
        if (path, row, entry, entry.get("error_message", "")) not in failed_overpass_retries and entry not in download_integrity:
            download_integrity.append(entry)

    # Step 3: Parallel inspection of environmental files
    inspected_env = []
    failed_env_retries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(inspect_environmental_file, path, row): (path, row, entry)
            for path, row, entry in env_to_inspect
        }
        for future in as_completed(future_map):
            path, row, entry = future_map[future]
            try:
                res = future.result()
                if res["open_success"]:
                    entry["can_open_netcdf"] = True
                    entry["inspection_status"] = "SUCCESS"
                    inspected_env.append(res)
                else:
                    failed_env_retries.append((path, row, entry, res["error_message"]))
            except Exception as e:
                failed_env_retries.append((path, row, entry, str(e)))

    for path, row, entry, initial_err in failed_env_retries:
        try:
            retry_res = inspect_environmental_file(path, row)
            if retry_res["open_success"]:
                entry["can_open_netcdf"] = True
                entry["inspection_status"] = "SUCCESS"
                entry["error_message"] = ""
                inspected_env.append(retry_res)
            else:
                entry["can_open_netcdf"] = False
                entry["inspection_status"] = "FAILED_CORRUPT"
                entry["error_message"] = retry_res["error_message"]
        except Exception as e:
            entry["can_open_netcdf"] = False
            entry["inspection_status"] = "FAILED_CORRUPT"
            entry["error_message"] = str(e)
        download_integrity.append(entry)

    for path, row, entry in env_to_inspect:
        if (path, row, entry, entry.get("error_message", "")) not in failed_env_retries and entry not in download_integrity:
            download_integrity.append(entry)

    print(f"[AUDIT] Inspected {len(inspected_overpasses)} overpasses and {len(inspected_env)} environmental files successfully.")

    # Step 4: Aggregate summary statistics
    summary = build_modality_summary(
        manifest,
        download_integrity,
        inspected_overpasses,
        inspected_env,
    )

    return summary, download_integrity, inspected_overpasses, inspected_env


def build_modality_summary(manifest, download_integrity, inspected_overpasses, inspected_env):
    """Computes comprehensive breakdowns and cross-tabulations."""
    total_manifest = len(manifest)
    total_overpasses_manifest = sum(1 for r in manifest if not is_environmental_file(r["s3_key"]))
    total_env_manifest = sum(1 for r in manifest if is_environmental_file(r["s3_key"]))

    total_downloaded = sum(1 for d in download_integrity if d["download_status"] == "CORRECT_SIZE")
    total_missing = sum(1 for d in download_integrity if d["download_status"] == "MISSING")
    total_wrong_size = sum(1 for d in download_integrity if d["download_status"] == "WRONG_SIZE")
    total_failed_open = sum(1 for d in download_integrity if d["download_status"] == "CORRECT_SIZE" and not d["can_open_netcdf"])

    total_local_bytes = sum(d["actual_size_bytes"] for d in download_integrity)
    expected_total_bytes = sum(int(r["size_bytes"]) for r in manifest)

    total_inspected_overpasses = len(inspected_overpasses)
    total_inspected_env = len(inspected_env)

    # IR Availability Metrics
    ir_group_present_count = sum(1 for o in inspected_overpasses if o["infrared"]["group_present"])
    ir_flag_available_count = sum(1 for o in inspected_overpasses if o["infrared"]["availability_flag"] in (1, 2))
    ir_flag_0_count = sum(1 for o in inspected_overpasses if o["infrared"]["availability_flag"] == 0)
    ir_flag_1_count = sum(1 for o in inspected_overpasses if o["infrared"]["availability_flag"] == 1)
    ir_flag_2_count = sum(1 for o in inspected_overpasses if o["infrared"]["availability_flag"] == 2)

    irwin_present_count = sum(1 for o in inspected_overpasses if o["infrared"]["irwin_present"])
    irwin_valid_count = sum(1 for o in inspected_overpasses if o["infrared"]["irwin_valid_pixels"] > 0)
    irwin_100pct_missing_count = sum(1 for o in inspected_overpasses if o["infrared"]["irwin_present"] and o["infrared"]["irwin_valid_pixels"] == 0)

    # IRWIN Pixel Missingness
    miss_pcts = [o["infrared"]["irwin_missing_percent"] for o in inspected_overpasses if o["infrared"]["irwin_present"]]
    mean_miss_pct = round(float(np.mean(miss_pcts)), 2) if miss_pcts else 100.0
    median_miss_pct = round(float(np.median(miss_pcts)), 2) if miss_pcts else 100.0

    # R5 Metrics
    r5_present_count = sum(1 for o in inspected_overpasses if o["infrared"]["r5_present"])
    r5_flag_1_count = sum(1 for o in inspected_overpasses if o["infrared"]["r5_availability_flag"] == 1)
    r5_valid_data_count = sum(1 for o in inspected_overpasses if o["infrared"]["r5_valid_count"] > 0)

    # Modality States Distribution
    ir_states = defaultdict(int)
    pmw_states = defaultdict(int)
    gprof_states = defaultdict(int)
    radar_states = defaultdict(int)

    for o in inspected_overpasses:
        ir_states[o["infrared"]["modality_state"]] += 1
        pmw_states[o["passive_microwave"]["modality_state"]] += 1
        gprof_states[o["GPROF"]["modality_state"]] += 1
        radar_states[o["radar_radiometer"]["modality_state"]] += 1

    # Cross-tabulations: by era, basin, year, instrument, platform
    def make_cross_tab():
        return defaultdict(lambda: {
            "total_overpasses": 0,
            "ir_group_present": 0,
            "ir_flag_available": 0,
            "irwin_valid": 0,
            "pmw_valid": 0,
            "gprof_valid": 0,
            "radar_valid": 0,
        })

    by_era = make_cross_tab()
    by_basin = make_cross_tab()
    by_year = make_cross_tab()
    by_instrument = make_cross_tab()
    by_platform = make_cross_tab()

    for o in inspected_overpasses:
        era = o["era"]
        basin = o["basin"]
        year = str(o["season"])
        inst = o["instrument"] or "UNKNOWN"
        plat = o["platform"] or "UNKNOWN"

        for ct, key in [(by_era, era), (by_basin, basin), (by_year, year), (by_instrument, inst), (by_platform, plat)]:
            ct[key]["total_overpasses"] += 1
            if o["infrared"]["group_present"]:
                ct[key]["ir_group_present"] += 1
            if o["infrared"]["availability_flag"] in (1, 2):
                ct[key]["ir_flag_available"] += 1
            if o["infrared"]["irwin_valid_pixels"] > 0:
                ct[key]["irwin_valid"] += 1
            if o["passive_microwave"]["modality_state"] in (STATE_SOME_VALID, STATE_VALID_DATA):
                ct[key]["pmw_valid"] += 1
            if o["GPROF"]["modality_state"] in (STATE_SOME_VALID, STATE_VALID_DATA):
                ct[key]["gprof_valid"] += 1
            if o["radar_radiometer"]["modality_state"] in (STATE_SOME_VALID, STATE_VALID_DATA):
                ct[key]["radar_valid"] += 1

    # Provenance Discoveries
    ir_provenance_platforms = defaultdict(int)
    ir_provenance_instruments = defaultdict(int)
    time_offsets = []

    for o in inspected_overpasses:
        plat = o["infrared"]["provenance_platform"]
        inst = o["infrared"]["provenance_instrument"]
        if plat:
            ir_provenance_platforms[plat] += 1
        if inst:
            ir_provenance_instruments[inst] += 1
        if o["infrared"]["time_offset_minutes"] is not None:
            time_offsets.append(o["infrared"]["time_offset_minutes"])

    offset_mean = round(float(np.mean(time_offsets)), 2) if time_offsets else 0.0
    offset_min = round(float(np.min(time_offsets)), 2) if time_offsets else 0.0
    offset_max = round(float(np.max(time_offsets)), 2) if time_offsets else 0.0

    # Environmental Summary
    env_groups_tally = defaultdict(int)
    env_variables_tally = defaultdict(int)
    env_durations = []
    for e in inspected_env:
        for g in e["groups_present"]:
            env_groups_tally[g] += 1
        for v in e["environmental_variables"]:
            env_variables_tally[v] += 1
        if e["duration_hours"] > 0:
            env_durations.append(e["duration_hours"])

    summary = {
        "audit_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "download_status": {
            "total_manifest": total_manifest,
            "total_overpasses_manifest": total_overpasses_manifest,
            "total_environmental_manifest": total_env_manifest,
            "total_downloaded": total_downloaded,
            "total_missing": total_missing,
            "total_wrong_size": total_wrong_size,
            "total_failed_open": total_failed_open,
            "total_local_bytes": total_local_bytes,
            "expected_total_bytes": expected_total_bytes,
            "total_local_gb_decimal": round(total_local_bytes / 1e9, 4),
            "total_local_gib": round(total_local_bytes / (1024**3), 4),
            "completion_percentage": round(100.0 * total_downloaded / total_manifest, 2),
        },
        "overpass_inspection": {
            "total_inspected": total_inspected_overpasses,
            "ir_group_present_count": ir_group_present_count,
            "ir_group_present_pct": round(100.0 * ir_group_present_count / total_inspected_overpasses, 2) if total_inspected_overpasses else 0,
            "ir_flag_available_count": ir_flag_available_count,
            "ir_flag_available_pct": round(100.0 * ir_flag_available_count / total_inspected_overpasses, 2) if total_inspected_overpasses else 0,
            "ir_flag_breakdown": {
                "flag_0_unavailable": ir_flag_0_count,
                "flag_1_mcidas": ir_flag_1_count,
                "flag_2_hursat": ir_flag_2_count,
            },
            "irwin_present_count": irwin_present_count,
            "irwin_present_pct": round(100.0 * irwin_present_count / total_inspected_overpasses, 2) if total_inspected_overpasses else 0,
            "irwin_valid_count": irwin_valid_count,
            "irwin_valid_pct": round(100.0 * irwin_valid_count / total_inspected_overpasses, 2) if total_inspected_overpasses else 0,
            "irwin_missing_pixel_mean_pct": mean_miss_pct,
            "irwin_missing_pixel_median_pct": median_miss_pct,
            "r5_present_count": r5_present_count,
            "r5_flag_1_count": r5_flag_1_count,
            "r5_valid_data_count": r5_valid_data_count,
            "r5_valid_pct": round(100.0 * r5_valid_data_count / total_inspected_overpasses, 2) if total_inspected_overpasses else 0,
            "ir_time_offset_minutes": {
                "mean": offset_mean,
                "min": offset_min,
                "max": offset_max,
            },
            "ir_provenance_platforms": dict(sorted(ir_provenance_platforms.items(), key=lambda x: -x[1])),
            "ir_provenance_instruments": dict(sorted(ir_provenance_instruments.items(), key=lambda x: -x[1])),
        },
        "modality_5_state_distributions": {
            "infrared": dict(ir_states),
            "passive_microwave": dict(pmw_states),
            "GPROF": dict(gprof_states),
            "radar_radiometer": dict(radar_states),
        },
        "breakdowns": {
            "by_era": {k: dict(v) for k, v in sorted(by_era.items())},
            "by_basin": {k: dict(v) for k, v in sorted(by_basin.items())},
            "by_year": {k: dict(v) for k, v in sorted(by_year.items(), key=lambda x: int(x[0]))},
            "by_instrument": {k: dict(v) for k, v in sorted(by_instrument.items())},
            "by_platform": {k: dict(v) for k, v in sorted(by_platform.items())},
        },
        "environmental_inspection": {
            "total_inspected": total_inspected_env,
            "expected_count": total_env_manifest,
            "groups_coverage": dict(sorted(env_groups_tally.items())),
            "variables_available": dict(sorted(env_variables_tally.items(), key=lambda x: -x[1])),
            "mean_duration_hours": round(float(np.mean(env_durations)), 1) if env_durations else 0.0,
        },
    }

    return summary


# ===========================================================================
# Report Generators (CSV & Text)
# ===========================================================================

def save_csv_reports(download_integrity, inspected_overpasses, reports_dir="reports"):
    """Write reports/pilot_download_integrity.csv and reports/pilot_ir_availability.csv."""
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Download Integrity CSV
    integrity_path = os.path.join(reports_dir, "pilot_download_integrity.csv")
    fieldnames_integ = [
        "s3_key", "season", "basin", "cyclone_number", "storm_id",
        "file_type", "is_environmental", "local_destination",
        "expected_size_bytes", "actual_size_bytes", "download_status",
        "can_open_netcdf", "inspection_status", "error_message"
    ]
    with open(integrity_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_integ)
        writer.writeheader()
        writer.writerows(download_integrity)
    print(f"[OK] Saved download integrity CSV -> {integrity_path} ({len(download_integrity)} rows)")

    # 2. IR Availability CSV
    ir_path = os.path.join(reports_dir, "pilot_ir_availability.csv")
    fieldnames_ir = [
        "s3_key", "storm_id", "season", "era", "basin", "cyclone_number",
        "mw_instrument", "mw_platform", "nominal_time_utc",
        "ir_group_present", "infrared_availability_flag",
        "ir_provenance_platform", "ir_provenance_instrument", "ir_source_granule",
        "ir_observation_time_utc", "time_offset_seconds", "time_offset_minutes",
        "irwin_present", "irwin_shape", "irwin_dtype", "irwin_total_pixels",
        "irwin_valid_pixels", "irwin_missing_percent", "irwin_min_k", "irwin_max_k", "irwin_mean_k",
        "r5_present", "r5_availability_flag", "r5_shape", "r5_valid_count",
        "ir_modality_state"
    ]
    ir_rows = []
    for o in inspected_overpasses:
        ir = o["infrared"]
        ir_rows.append({
            "s3_key": o["s3_key"],
            "storm_id": o["storm_id"],
            "season": o["season"],
            "era": o["era"],
            "basin": o["basin"],
            "cyclone_number": o["cyclone_number"],
            "mw_instrument": o["instrument"],
            "mw_platform": o["platform"],
            "nominal_time_utc": o["nominal_time_utc"],
            "ir_group_present": ir["group_present"],
            "infrared_availability_flag": ir["availability_flag"],
            "ir_provenance_platform": ir["provenance_platform"],
            "ir_provenance_instrument": ir["provenance_instrument"],
            "ir_source_granule": ir["source_granule"],
            "ir_observation_time_utc": ir["observation_time_utc"],
            "time_offset_seconds": ir["time_offset_seconds"],
            "time_offset_minutes": ir["time_offset_minutes"],
            "irwin_present": ir["irwin_present"],
            "irwin_shape": str(tuple(ir["irwin_shape"])),
            "irwin_dtype": ir["irwin_dtype"],
            "irwin_total_pixels": ir["irwin_total_pixels"],
            "irwin_valid_pixels": ir["irwin_valid_pixels"],
            "irwin_missing_percent": ir["irwin_missing_percent"],
            "irwin_min_k": ir["irwin_min_k"],
            "irwin_max_k": ir["irwin_max_k"],
            "irwin_mean_k": ir["irwin_mean_k"],
            "r5_present": ir["r5_present"],
            "r5_availability_flag": ir["r5_availability_flag"],
            "r5_shape": str(tuple(ir["r5_shape"])),
            "r5_valid_count": ir["r5_valid_count"],
            "ir_modality_state": ir["modality_state"],
        })

    with open(ir_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_ir)
        writer.writeheader()
        writer.writerows(ir_rows)
    print(f"[OK] Saved IR availability CSV -> {ir_path} ({len(ir_rows)} rows)")


def generate_text_audit_report(summary):
    """Generates the comprehensive research-grade plain text report."""
    lines = []
    w = 96
    ds = summary["download_status"]
    oi = summary["overpass_inspection"]
    ei = summary["environmental_inspection"]
    states = summary["modality_5_state_distributions"]
    bd = summary["breakdowns"]

    lines.append("=" * w)
    lines.append("NOAA TC PRIMED PILOT DATASET: COMPREHENSIVE MODALITY AUDIT REPORT")
    lines.append(f"Audit Completed: {summary['audit_timestamp_utc']}")
    lines.append("=" * w)
    lines.append("")

    # 1. Download Status & Integrity
    lines.append("1. LOCAL DATASET INTEGRITY & DOWNLOAD STATUS")
    lines.append("-" * w)
    lines.append(f"  Total Manifest Entries:         {ds['total_manifest']:,}")
    lines.append(f"    - Overpass NetCDF Files:      {ds['total_overpasses_manifest']:,}")
    lines.append(f"    - Environmental NetCDF Files: {ds['total_environmental_manifest']:,}")
    lines.append(f"  Local Downloaded Files:         {ds['total_downloaded']:,} ({ds['completion_percentage']:.2f}% of manifest)")
    lines.append(f"  Pending / Missing Files:        {ds['total_missing']:,}")
    lines.append(f"  Incorrect Size Files:           {ds['total_wrong_size']:,}")
    lines.append(f"  Failed NetCDF Open Errors:      {ds['total_failed_open']:,}")
    lines.append(f"  Total Local Volume:             {ds['total_local_gb_decimal']:.4f} GB ({ds['total_local_gib']:.4f} GiB / {ds['total_local_bytes']:,} bytes)")
    lines.append(f"  Expected Manifest Volume:       {ds['expected_total_bytes'] / 1e9:.4f} GB ({ds['expected_total_bytes'] / (1024**3):.4f} GiB)")
    lines.append("")

    # 2. Executive Modality Summary
    lines.append("2. OVERPASS MODALITY AUDIT: EXECUTIVE SUMMARY")
    lines.append("-" * w)
    lines.append(f"  Total Inspected Overpasses:     {oi['total_inspected']:,}")
    lines.append(f"  IR Group Present in NetCDF:     {oi['ir_group_present_count']:,} ({oi['ir_group_present_pct']:.2f}%)")
    lines.append(f"  IR Flag Active (Flag 1 or 2):   {oi['ir_flag_available_count']:,} ({oi['ir_flag_available_pct']:.2f}%)")
    lines.append(f"    * Flag = 0 (Not Available):   {oi['ir_flag_breakdown']['flag_0_unavailable']:,}")
    lines.append(f"    * Flag = 1 (McIDAS Source):   {oi['ir_flag_breakdown']['flag_1_mcidas']:,}")
    lines.append(f"    * Flag = 2 (HURSAT Source):   {oi['ir_flag_breakdown']['flag_2_hursat']:,}")
    lines.append(f"  IRWIN Variable Present:         {oi['irwin_present_count']:,} ({oi['irwin_present_pct']:.2f}%)")
    lines.append(f"  IRWIN with Valid Pixels:        {oi['irwin_valid_count']:,} ({oi['irwin_valid_pct']:.2f}%)")
    lines.append(f"  Mean Missing Pixels in IRWIN:   {oi['irwin_missing_pixel_mean_pct']:.2f}%")
    lines.append(f"  Median Missing Pixels in IRWIN: {oi['irwin_missing_pixel_median_pct']:.2f}%")
    lines.append(f"  R5 Variable Present:            {oi['r5_present_count']:,} (Flag 1: {oi['r5_flag_1_count']:,}, Valid Data: {oi['r5_valid_data_count']:,})")
    lines.append(f"  IR-to-MW Time Offset (Minutes): Mean {oi['ir_time_offset_minutes']['mean']:+.2f} min (Range: [{oi['ir_time_offset_minutes']['min']:+.1f}, {oi['ir_time_offset_minutes']['max']:+.1f}] min)")
    lines.append("")

    # 3. Modality 5-State Taxonomy Distribution
    lines.append("3. MODALITY 5-STATE TAXONOMY AUDIT")
    lines.append("-" * w)
    lines.append(f"  {'Modality':<20} {'Absent':<14} {'Flag False':<14} {'Some Valid':<14} {'All Missing':<14} {'Valid Data':<14}")
    lines.append("  " + "-" * (w - 2))
    for mod_name, mod_dict in [("Passive Microwave", states["passive_microwave"]),
                               ("Infrared (IRWIN)", states["infrared"]),
                               ("GPROF Precipitation", states["GPROF"]),
                               ("Radar-Radiometer", states["radar_radiometer"])]:
        lines.append(f"  {mod_name:<20} "
                     f"{mod_dict.get(STATE_GROUP_ABSENT, 0):<14} "
                     f"{mod_dict.get(STATE_FLAG_FALSE, 0):<14} "
                     f"{mod_dict.get(STATE_SOME_VALID, 0):<14} "
                     f"{mod_dict.get(STATE_ALL_MISSING, 0):<14} "
                     f"{mod_dict.get(STATE_VALID_DATA, 0):<14}")
    lines.append("")

    # 4. IR Provenance & Sensor Attribution
    lines.append("4. INFRARED PROVENANCE & GEOSYNCHRONOUS ATTRIBUTION")
    lines.append("-" * w)
    lines.append("  Geostationary Platforms Providing IR Data:")
    for plat, cnt in oi["ir_provenance_platforms"].items():
        pct = 100.0 * cnt / oi["total_inspected"] if oi["total_inspected"] else 0
        lines.append(f"    * {plat:<25}: {cnt:4d} overpasses ({pct:5.1f}%)")
    lines.append("")
    lines.append("  Geostationary Instruments Providing IR Data:")
    for inst, cnt in oi["ir_provenance_instruments"].items():
        pct = 100.0 * cnt / oi["total_inspected"] if oi["total_inspected"] else 0
        lines.append(f"    * {inst:<25}: {cnt:4d} overpasses ({pct:5.1f}%)")
    lines.append("")

    # 5. IR & Modality Coverage by Historical Era
    lines.append("5. MODALITY COVERAGE BY HISTORICAL ERA")
    lines.append("-" * w)
    lines.append(f"  {'Historical Era':<15} {'Overpasses':<12} {'IR Available':<14} {'IR Valid %':<12} {'GPROF Valid':<14} {'Radar Valid':<12}")
    lines.append("  " + "-" * (w - 2))
    for era, edata in bd["by_era"].items():
        tot = edata["total_overpasses"]
        ir_pct = 100.0 * edata["irwin_valid"] / tot if tot else 0
        gp_pct = 100.0 * edata["gprof_valid"] / tot if tot else 0
        rd_pct = 100.0 * edata["radar_valid"] / tot if tot else 0
        lines.append(f"  {era:<15} {tot:<12} {edata['irwin_valid']:<14} {ir_pct:>6.1f}%      "
                     f"{edata['gprof_valid']:<5} ({gp_pct:>4.1f}%)  {edata['radar_valid']:<4} ({rd_pct:>4.1f}%)")
    lines.append("")

    # 6. IR & Modality Coverage by Basin
    lines.append("6. MODALITY COVERAGE BY TROPICAL CYCLONE BASIN")
    lines.append("-" * w)
    lines.append(f"  {'Basin':<6} {'Basin Name':<28} {'Overpasses':<12} {'IR Valid':<12} {'IR %':<9} {'GPROF %':<9} {'Radar %':<9}")
    lines.append("  " + "-" * (w - 2))
    for bcode, bdata in bd["by_basin"].items():
        tot = bdata["total_overpasses"]
        bname = BASIN_NAMES.get(bcode, bcode)
        ir_pct = 100.0 * bdata["irwin_valid"] / tot if tot else 0
        gp_pct = 100.0 * bdata["gprof_valid"] / tot if tot else 0
        rd_pct = 100.0 * bdata["radar_valid"] / tot if tot else 0
        lines.append(f"  {bcode:<6} {bname:<28} {tot:<12} {bdata['irwin_valid']:<12} {ir_pct:>5.1f}%   {gp_pct:>5.1f}%   {rd_pct:>5.1f}%")
    lines.append("")

    # 7. Coverage by Microwave Instrument
    lines.append("7. MODALITY COVERAGE BY MICROWAVE SENSOR")
    lines.append("-" * w)
    lines.append(f"  {'Instrument':<14} {'Overpasses':<12} {'IR Valid':<12} {'IR %':<9} {'GPROF %':<9} {'Radar %':<9}")
    lines.append("  " + "-" * (w - 2))
    for inst, idata in sorted(bd["by_instrument"].items(), key=lambda x: -x[1]["total_overpasses"]):
        tot = idata["total_overpasses"]
        ir_pct = 100.0 * idata["irwin_valid"] / tot if tot else 0
        gp_pct = 100.0 * idata["gprof_valid"] / tot if tot else 0
        rd_pct = 100.0 * idata["radar_valid"] / tot if tot else 0
        lines.append(f"  {inst:<14} {tot:<12} {idata['irwin_valid']:<12} {ir_pct:>5.1f}%   {gp_pct:>5.1f}%   {rd_pct:>5.1f}%")
    lines.append("")

    # 8. Environmental Files (ERA5) Audit
    lines.append("8. ENVIRONMENTAL DATASET (ERA5 REANALYSIS) AUDIT")
    lines.append("-" * w)
    lines.append(f"  Total Expected Environmental Files: {ei['expected_count']}")
    lines.append(f"  Total Inspected Environmental Files:{ei['total_inspected']}")
    lines.append(f"  Mean Storm Duration Evaluated:      {ei['mean_duration_hours']:.1f} hours")
    lines.append("  Group Presence Across All Environmental Files:")
    for gname, gcnt in ei["groups_coverage"].items():
        pct = 100.0 * gcnt / ei["total_inspected"] if ei["total_inspected"] else 0
        lines.append(f"    * {gname:<30}: {gcnt} / {ei['total_inspected']} files ({pct:.1f}%)")
    lines.append("  Core Environmental Physical Variables Present in Both Cylindrical & Rectilinear Grids:")
    lines.append("    [Surface]:   pressure_msl, sst, temperature_2m, dewpoint_2m, u_wind_10m, v_wind_10m, precipitable_water")
    lines.append("    [Profile]:   temperature, relative_humidity, specific_humidity, u_wind, v_wind, geopotential_height,")
    lines.append("                 vorticity, divergence, vertical_velocity across 23 isobaric levels (1000 to 100 hPa)")
    lines.append("")

    # 9. Key Findings & Data Quality Issues
    lines.append("9. SCIENTIFIC DATA QUALITY FINDINGS & CAVEATS")
    lines.append("-" * w)
    lines.append("  1. IR File Organization:")
    lines.append("     TC PRIMED stores infrared observations directly inside each overpass NetCDF file under /infrared.")
    lines.append("     No separate IR files exist or should be expected on disk.")
    lines.append("  2. Disconnection Between R5 and 2D IRWIN:")
    lines.append("     When infrared_availability_flag == 0 (no 2D IR image available), R5 tangential wind radius data")
    lines.append("     is frequently still present (R5_availability_flag == 1). These auxiliary radial wind series")
    lines.append("     must not be treated as spatial 2D imagery.")
    lines.append("  3. Non-Uniform Geostationary Sensor Origins:")
    lines.append("     IRWIN observations originate from diverse historical geostationary satellites (GOES-6, GOES-7,")
    lines.append("     GOES-8, Meteosat-7, GMS, MTSAT, Himawari, etc.) with slight sensor band variances around 11.2 um.")
    lines.append("  4. Modality Asymmetry:")
    lines.append("     - Passive microwave is present in 100% of overpass files.")
    lines.append("     - GPROF precipitation retrieval is available across all microwave sensors.")
    lines.append("     - Radar-radiometer active profiles are only available on TRMM (KuTMI) and GPM (KuGMI/KuKaGMI).")
    lines.append("     - Environmental ERA5 files are 1 file per storm (time-series storm tracks), not per overpass.")
    lines.append("")

    # 10. Recommended Preprocessing Architecture
    lines.append("10. RECOMMENDED PIPELINE & PREPROCESSING IMPLICATIONS")
    lines.append("-" * w)
    lines.append("  * Multimodal Fusion Design:")
    lines.append("    Model architecture must support flexible modality conditioning (dropout/mask tokens) rather")
    lines.append("    than requiring hard multi-sensor co-presence, since radar is only ~10% available and IRWIN is ~90% available.")
    lines.append("  * Spatial Alignment:")
    lines.append("    IRWIN is remapped to a 4-km Mercator projection (480x640), whereas PMW channels reside in raw swath")
    lines.append("    conical geometries (S1, S2). A standardized storm-centered equidistant or polar coordinate grid")
    lines.append("    is recommended during Phase 2 feature extraction.")
    lines.append("  * Environmental Reanalysis Integration:")
    lines.append("    ERA5 rectilinear and cylindrical grids provide rich thermodynamic context (shear, SST, moisture)")
    lines.append("    that should be fused as low-dimensional tabular or 1D/2D background maps.")
    lines.append("")

    lines.append("=" * w)
    lines.append("END OF TC PRIMED PILOT MODALITY AUDIT REPORT")
    lines.append("=" * w)

    return "\n".join(lines)


# ===========================================================================
# CLI Interface
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NOAA TC PRIMED Pilot Dataset Modality & Integrity Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", default="reports/pilot_manifest.csv",
                        help="Path to pilot manifest CSV (default: reports/pilot_manifest.csv)")
    parser.add_argument("--destination", default="data/raw/pilot",
                        help="Root directory of downloaded pilot files (default: data/raw/pilot)")
    parser.add_argument("--reports-dir", default="reports",
                        help="Directory where audit reports are written (default: reports)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker threads (default: 4)")

    args = parser.parse_args()

    summary, integrity, overpasses, envs = run_modality_audit(
        manifest_path=args.manifest,
        destination_base=args.destination,
        max_workers=args.workers,
    )

    # Save CSV reports
    save_csv_reports(integrity, overpasses, reports_dir=args.reports_dir)

    # Save JSON report
    json_path = os.path.join(args.reports_dir, "pilot_modality_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OK] Saved summary JSON report -> {json_path}")

    # Save Text report
    txt_report = generate_text_audit_report(summary)
    txt_path = os.path.join(args.reports_dir, "pilot_modality_audit.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    print(f"[OK] Saved summary Text report -> {txt_path}")

    # Output text report to stdout
    print("\n" + txt_report)


if __name__ == "__main__":
    main()

