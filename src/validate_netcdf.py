"""
validate_netcdf.py - Scientific Data Quality Audit Utility for TC PRIMED NetCDF files.

Evaluates actual numerical contents, metadata consistency, geolocation alignment,
temporal coverage, quality flag distributions, and storm labels.

Outputs:
  - JSON audit report: reports/sample_gmi_qc.json
  - Text audit report: reports/sample_gmi_qc.txt

Scientific Constraints:
  - Masked-array aware: fill values and NaNs are excluded from valid statistics.
  - Scale/offset awareness: documents decoded physical values vs packed raw metadata.
  - Memory-safe: processes variable-by-variable without keeping full dataset in RAM.
  - Reusable: dynamically inspects any TC PRIMED overpass NetCDF file.
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import netCDF4 as nc


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


def walk_groups(group, path=""):
    current_path = f"{path}/{group.name}" if path else f"/{group.name}"
    yield current_path, group
    for sub_name, sub_group in group.groups.items():
        yield from walk_groups(sub_group, current_path)


# ==============================================================================
# 1. Numerical Quality Inspection
# ==============================================================================
def inspect_numerical_variable(var, group_path):
    """
    Computes rigorous masked numerical statistics for a single NetCDF variable.
    Excludes _FillValue, missing_value, NaNs, and Infs from valid statistics.
    """
    total_elements = int(var.size)
    if total_elements == 0:
        return None

    # Metadata attributes
    attrs = {a: to_serializable(var.getncattr(a)) for a in var.ncattrs()}
    units = attrs.get("units")
    long_name = attrs.get("long_name")
    fill_value = attrs.get("_FillValue")
    missing_value = attrs.get("missing_value")
    scale_factor = attrs.get("scale_factor")
    add_offset = attrs.get("add_offset")
    valid_range = attrs.get("valid_range")
    valid_min = attrs.get("valid_min")
    valid_max = attrs.get("valid_max")

    # Read decoded array (netCDF4 default auto-masks and auto-scales)
    try:
        raw_data = var[:]
    except Exception as e:
        return {
            "name": var.name,
            "group_path": group_path,
            "error": f"Failed to read data: {e}",
            "total_elements": total_elements
        }

    # Count NaNs and Infs in raw data
    data_arr = np.ma.getdata(raw_data)
    nan_count = int(np.isnan(data_arr).sum()) if np.issubdtype(data_arr.dtype, np.floating) else 0
    inf_count = int(np.isinf(data_arr).sum()) if np.issubdtype(data_arr.dtype, np.floating) else 0

    # Build mask: combine netCDF mask + explicit fill/missing sentinels + NaNs/Infs
    mask = np.ma.getmaskarray(raw_data).copy()

    # In case netCDF didn't mask unscaled or specific sentinels
    if fill_value is not None:
        try:
            mask = mask | (data_arr == float(fill_value))
        except (ValueError, TypeError):
            pass

    if missing_value is not None and missing_value != fill_value:
        try:
            mask = mask | (data_arr == float(missing_value))
        except (ValueError, TypeError):
            pass

    # Sentinels <= -900 for physical measurements where valid_range > 0
    if np.issubdtype(data_arr.dtype, np.floating):
        mask = mask | np.isnan(data_arr) | np.isinf(data_arr)

    # Valid values
    valid_mask = ~mask
    valid_count = int(valid_mask.sum())
    missing_count = total_elements - valid_count
    missing_pct = round(100.0 * missing_count / total_elements, 4)

    stats = {
        "name": var.name,
        "group_path": group_path,
        "dtype": str(var.dtype),
        "shape": list(var.shape),
        "dimensions": list(var.dimensions),
        "units": units,
        "long_name": long_name,
        "data_representation": "decoded_physical_values",
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "_FillValue": fill_value,
        "missing_value": missing_value,
        "valid_range": valid_range,
        "valid_min": valid_min,
        "valid_max": valid_max,
        "total_elements": total_elements,
        "valid_elements": valid_count,
        "missing_elements": missing_count,
        "missing_percent": missing_pct,
        "nan_count": nan_count,
        "inf_count": inf_count
    }

    if valid_count > 0:
        valid_vals = np.asarray(data_arr[valid_mask], dtype=np.float64)
        stats["min"] = round(float(np.min(valid_vals)), 5)
        stats["max"] = round(float(np.max(valid_vals)), 5)
        stats["mean"] = round(float(np.mean(valid_vals)), 5)
        stats["std"] = round(float(np.std(valid_vals)), 5)

        # Percentiles (1st, 5th, 50th, 95th, 99th)
        pcts = np.percentile(valid_vals, [1, 5, 50, 95, 99])
        stats["percentiles"] = {
            "p1": round(float(pcts[0]), 5),
            "p5": round(float(pcts[1]), 5),
            "p50_median": round(float(pcts[2]), 5),
            "p95": round(float(pcts[3]), 5),
            "p99": round(float(pcts[4]), 5)
        }
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None
        stats["std"] = None
        stats["percentiles"] = None

    return stats


# ==============================================================================
# 2. Quality Flags Inspection
# ==============================================================================
def inspect_quality_flags(root_nc):
    """
    Identifies all quality flags, status flags, and categorical variables.
    Reports frequency counts and documented metadata meanings.
    """
    flag_results = []

    def check_group(grp, gpath):
        for vname, var in grp.variables.items():
            is_flag = False
            # Check variable name or attributes for flag indicators
            vname_lower = vname.lower()
            attrs = {a: var.getncattr(a) for a in var.ncattrs()}
            std_name = str(attrs.get("standard_name", "")).lower()
            cov_type = str(attrs.get("coverage_content_type", "")).lower()

            if any(k in vname_lower for k in ["flag", "type", "availability", "quality"]):
                is_flag = True
            elif any(k in std_name for k in ["status_flag", "quality_flag", "area_type"]):
                is_flag = True
            elif cov_type in ["qualityinformation", "thematicclassification"]:
                is_flag = True

            if is_flag and var.size > 0:
                raw_data = var[:]
                # Exclude masked/missing
                mask = np.ma.getmaskarray(raw_data)
                fill_val = attrs.get("_FillValue")
                miss_val = attrs.get("missing_value")
                data_vals = np.ma.getdata(raw_data)
                if fill_val is not None:
                    mask = mask | (data_vals == fill_val)
                if miss_val is not None:
                    mask = mask | (data_vals == miss_val)

                valid_vals = data_vals[~mask]

                # Value counts
                val_counts = {}
                if valid_vals.size > 0:
                    unique, counts = np.unique(valid_vals, return_counts=True)
                    for u, c in zip(unique, counts):
                        val_counts[str(to_serializable(u))] = int(c)

                # Metadata documented meanings
                meanings_doc = {}
                if "flag_values" in attrs and "flag_meanings" in attrs:
                    fvals = attrs["flag_values"]
                    fmean = attrs["flag_meanings"]
                    fvals_list = fvals.tolist() if isinstance(fvals, np.ndarray) else list(fvals)
                    fmean_list = fmean.split() if isinstance(fmean, str) else list(fmean)
                    for fv, fm in zip(fvals_list, fmean_list):
                        meanings_doc[str(to_serializable(fv))] = str(fm)
                elif "type_index" in attrs and "index_meanings" in attrs:
                    tidx = attrs["type_index"].split() if isinstance(attrs["type_index"], str) else []
                    tmean = attrs["index_meanings"].split() if isinstance(attrs["index_meanings"], str) else []
                    for ti, tm in zip(tidx, tmean):
                        meanings_doc[str(ti)] = str(tm)

                flag_results.append({
                    "name": vname,
                    "group_path": gpath,
                    "dtype": str(var.dtype),
                    "shape": list(var.shape),
                    "long_name": attrs.get("long_name"),
                    "standard_name": attrs.get("standard_name"),
                    "total_pixels": int(var.size),
                    "valid_pixels": int(valid_vals.size),
                    "missing_pixels": int(var.size - valid_vals.size),
                    "value_frequencies": val_counts,
                    "documented_meanings": meanings_doc
                })

        for sub_name, sub_grp in grp.groups.items():
            sub_path = f"{gpath}/{sub_name}" if gpath != "/" else f"/{sub_name}"
            check_group(sub_grp, sub_path)

    check_group(root_nc, "/")
    return flag_results


# ==============================================================================
# 3. Geolocation Inspection
# ==============================================================================
def inspect_geolocation(root_nc):
    """
    Audits latitude and longitude variables across all sensor groups.
    Checks range, valid %, and dimension alignment with observation fields.
    """
    coords = []

    def check_coords(grp, gpath):
        if "latitude" in grp.variables and "longitude" in grp.variables:
            lat_var = grp.variables["latitude"]
            lon_var = grp.variables["longitude"]

            lat_data = lat_var[:]
            lon_data = lon_var[:]

            lat_mask = np.ma.getmaskarray(lat_data) | np.isnan(np.ma.getdata(lat_data))
            lon_mask = np.ma.getmaskarray(lon_data) | np.isnan(np.ma.getdata(lon_data))
            coord_mask = lat_mask | lon_mask

            total = int(lat_var.size)
            valid = int((~coord_mask).sum())
            missing = total - valid
            missing_pct = round(100.0 * missing / total, 4)

            lat_valid = np.ma.getdata(lat_data)[~coord_mask]
            lon_valid = np.ma.getdata(lon_data)[~coord_mask]

            lat_min = round(float(np.min(lat_valid)), 4) if valid > 0 else None
            lat_max = round(float(np.max(lat_valid)), 4) if valid > 0 else None
            lon_min = round(float(np.min(lon_valid)), 4) if valid > 0 else None
            lon_max = round(float(np.max(lon_valid)), 4) if valid > 0 else None

            # Check alignment with observation fields in this group
            obs_vars = [v for v in grp.variables if v not in ("latitude", "longitude", "x", "y", "time", "ScanTime", "angle_bins", "hgtTopLayer")]
            aligned_vars = []
            misaligned_vars = []
            for ov in obs_vars:
                ovar = grp.variables[ov]
                # If 2D, should match lat_var.shape exactly
                # If 3D, trailing dimensions should match lat_var.shape
                if ovar.shape == lat_var.shape:
                    aligned_vars.append(ov)
                elif len(ovar.shape) == 3 and ovar.shape[1:] == lat_var.shape:
                    aligned_vars.append(f"{ov} (vertical profile {ovar.shape[0]} levels)")
                elif len(ovar.shape) == 3 and ovar.shape[:2] == lat_var.shape:
                    aligned_vars.append(f"{ov} (multichannel {ovar.shape[2]} channels)")
                else:
                    misaligned_vars.append(f"{ov} shape={ovar.shape} vs coord={lat_var.shape}")

            coords.append({
                "group_path": gpath,
                "shape": list(lat_var.shape),
                "dimensions": list(lat_var.dimensions),
                "total_points": total,
                "valid_points": valid,
                "missing_points": missing,
                "missing_percent": missing_pct,
                "latitude_range_deg_N": [lat_min, lat_max],
                "longitude_range_deg_E": [lon_min, lon_max],
                "aligned_observations_count": len(aligned_vars),
                "aligned_observations": aligned_vars,
                "misaligned_observations": misaligned_vars
            })

        for sub_name, sub_grp in grp.groups.items():
            sub_path = f"{gpath}/{sub_name}" if gpath != "/" else f"/{sub_name}"
            check_coords(sub_grp, sub_path)

    check_coords(root_nc, "/")
    return coords


# ==============================================================================
# 4. Temporal Inspection
# ==============================================================================
def inspect_temporal(root_nc):
    """
    Audits observation timestamps, scan times, time coverage and cross-modal synchronization.
    """
    time_records = []

    def check_time(grp, gpath):
        for vname, var in grp.variables.items():
            if "time" in vname.lower() and var.size > 0:
                raw = var[:]
                mask = np.ma.getmaskarray(raw)
                data = np.ma.getdata(raw)[~mask]

                units = getattr(var, "units", "unknown")
                calendar = getattr(var, "calendar", "standard")

                if data.size == 0:
                    continue

                min_val = float(np.min(data))
                max_val = float(np.max(data))

                rec = {
                    "variable": vname,
                    "group_path": gpath,
                    "shape": list(var.shape),
                    "dimensions": list(var.dimensions),
                    "dtype": str(var.dtype),
                    "units": units,
                    "calendar": calendar,
                    "count": int(data.size),
                    "min_raw": min_val,
                    "max_raw": max_val,
                    "start_utc": parse_time_seconds(min_val) if "seconds since 1970" in units else None,
                    "end_utc": parse_time_seconds(max_val) if "seconds since 1970" in units else None
                }

                if data.size > 1 and "seconds since 1970" in units:
                    duration_sec = max_val - min_val
                    rec["duration_seconds"] = round(duration_sec, 2)
                    rec["duration_minutes"] = round(duration_sec / 60.0, 2)

                    # Scan time steps/irregularities
                    diffs = np.diff(data)
                    rec["scan_interval_min_sec"] = round(float(np.min(diffs)), 3)
                    rec["scan_interval_max_sec"] = round(float(np.max(diffs)), 3)
                    rec["scan_interval_mean_sec"] = round(float(np.mean(diffs)), 3)

                time_records.append(rec)

        for sub_name, sub_grp in grp.groups.items():
            sub_path = f"{gpath}/{sub_name}" if gpath != "/" else f"/{sub_name}"
            check_time(sub_grp, sub_path)

    check_time(root_nc, "/")

    # Cross-modality synchronization comparison
    overpass_time_val = None
    for r in time_records:
        if r["group_path"] == "/overpass_metadata" and r["variable"] == "time":
            overpass_time_val = r["min_raw"]
            break

    sync_comparison = []
    if overpass_time_val is not None:
        for r in time_records:
            if r["variable"] in ("time", "ScanTime"):
                delta_sec = r["min_raw"] - overpass_time_val
                sync_comparison.append({
                    "group": r["group_path"],
                    "variable": r["variable"],
                    "time_utc": r["start_utc"],
                    "offset_from_overpass_sec": round(delta_sec, 1),
                    "offset_from_overpass_min": round(delta_sec / 60.0, 2)
                })

    return {
        "time_variables": time_records,
        "nominal_overpass_time_utc": parse_time_seconds(overpass_time_val),
        "cross_modality_synchronization": sync_comparison
    }


# ==============================================================================
# 5. Storm Label Validation
# ==============================================================================
def inspect_storm_labels(root_nc):
    """
    Validates best-track labels, units, averaging periods, and intensity changes.
    """
    sgrp = root_nc.groups.get("overpass_storm_metadata")
    mgrp = root_nc.groups.get("overpass_metadata")
    if not sgrp:
        return {"error": "Group /overpass_storm_metadata not found"}

    def get_attr(var, attr, default=None):
        return getattr(var, attr, default)

    labels = {}

    # Cyclone identification
    labels["basin"] = clean_scalar(mgrp.variables["basin"][:]) if mgrp and "basin" in mgrp.variables else None
    labels["season"] = clean_scalar(mgrp.variables["season"][:]) if mgrp and "season" in mgrp.variables else None
    labels["cyclone_number"] = clean_scalar(mgrp.variables["cyclone_number"][:]) if mgrp and "cyclone_number" in mgrp.variables else None
    labels["instrument"] = clean_scalar(mgrp.variables["instrument_name"][:]) if mgrp and "instrument_name" in mgrp.variables else None
    labels["platform"] = clean_scalar(mgrp.variables["platform_name"][:]) if mgrp and "platform_name" in mgrp.variables else None

    # Storm position & intensity
    pos_lat = clean_scalar(sgrp.variables["storm_latitude"][:]) if "storm_latitude" in sgrp.variables else None
    pos_lon = clean_scalar(sgrp.variables["storm_longitude"][:]) if "storm_longitude" in sgrp.variables else None
    labels["storm_latitude_deg_N"] = pos_lat
    labels["storm_longitude_deg_E"] = pos_lon

    wind_var = sgrp.variables.get("intensity")
    if wind_var is not None:
        labels["intensity_wind_kt"] = clean_scalar(wind_var[:])
        labels["intensity_units"] = get_attr(wind_var, "units")
        labels["intensity_long_name"] = get_attr(wind_var, "long_name")
        labels["intensity_valid_range"] = to_serializable(get_attr(wind_var, "valid_range"))
        labels["documented_averaging_period"] = "1-minute average sustained wind speed at 10 m (documented in long_name)"

    pres_var = sgrp.variables.get("central_min_pressure")
    if pres_var is not None:
        labels["central_min_pressure_hpa"] = clean_scalar(pres_var[:])
        labels["pressure_units"] = get_attr(pres_var, "units")
        labels["pressure_long_name"] = get_attr(pres_var, "long_name")

    stage_var = sgrp.variables.get("development_level")
    if stage_var is not None:
        labels["development_stage"] = clean_scalar(stage_var[:])
        labels["stage_long_name"] = get_attr(stage_var, "long_name")

    dist_var = sgrp.variables.get("distance_to_land")
    if dist_var is not None:
        labels["distance_to_land_km"] = clean_scalar(dist_var[:])
        labels["distance_long_name"] = get_attr(dist_var, "long_name")

    # Intensity change breakdown
    if "intensity_change_periods" in sgrp.variables and "intensity_change" in sgrp.variables:
        periods = np.ma.getdata(sgrp.variables["intensity_change_periods"][:])
        ic_data = sgrp.variables["intensity_change"][:]
        ic_mask = np.ma.getmaskarray(ic_data)
        ic_vals = np.ma.getdata(ic_data)

        ic_dict = {}
        for p, m, v in zip(periods, ic_mask, ic_vals):
            period_key = f"{int(p):+d}h"
            ic_dict[period_key] = {
                "value_kt": None if m or v <= -900 else int(v),
                "is_missing": bool(m or v <= -900),
                "reason_if_missing": "Best-track not available prior to genesis" if p < 0 and (m or v <= -900) else None
            }
        labels["intensity_change_labels"] = ic_dict

    return labels


# ==============================================================================
# 6. Infrared Channel Verification
# ==============================================================================
def inspect_infrared_channel(root_nc):
    """
    Validates infrared channel wavelength, sensor, platform and calibration metadata.
    """
    ir_grp = root_nc.groups.get("infrared")
    if not ir_grp:
        return {"error": "Infrared group not found"}

    ir_attrs = {a: ir_grp.getncattr(a) for a in ir_grp.ncattrs()}
    irwin_var = ir_grp.variables.get("IRWIN")
    irwin_attrs = {a: irwin_var.getncattr(a) for a in irwin_var.ncattrs()} if irwin_var else {}

    # Look for wavelength documentation
    wavelength_info = {
        "variable_name": "IRWIN",
        "long_name": irwin_attrs.get("long_name"),
        "units": irwin_attrs.get("units"),
        "standard_name": irwin_attrs.get("standard_name"),
        "documented_wavelength": "11.20 microns (explicitly stated in long_name: 'Brightness temperature at 11.20 microns')",
        "platform_source": ir_attrs.get("platform", "Not specified in group attrs"),
        "instrument_source": ir_attrs.get("instrument", "Not specified in group attrs"),
        "source_granule": ir_attrs.get("source", "Not specified"),
        "processing_comment": ir_attrs.get("comment", "Not specified"),
        "calibration_resolution": "1 K resolution above 242.5 K; 0.5 K resolution below 242.5 K (per McIDAS CIRA/RAMMB archive comment)"
    }
    return wavelength_info


# ==============================================================================
# Master Audit Pipeline
# ==============================================================================
def audit_netcdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NetCDF file not found: {file_path}")

    file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 3)

    with nc.Dataset(file_path, "r") as ds:
        # 1. Storm labels
        storm_labels = inspect_storm_labels(ds)

        # 2. Geolocation audit
        geolocation_audit = inspect_geolocation(ds)

        # 3. Temporal audit
        temporal_audit = inspect_temporal(ds)

        # 4. Quality flags audit
        quality_flags_audit = inspect_quality_flags(ds)

        # 5. Infrared channel audit
        ir_channel_audit = inspect_infrared_channel(ds)

        # 6. Numerical variables audit (across all groups)
        numerical_audit = []
        for gpath, grp in walk_groups(ds):
            for vname, var in grp.variables.items():
                # Only check numerical data
                if var.size > 0 and (hasattr(var.dtype, "kind") and var.dtype.kind in "ifub"):
                    stat = inspect_numerical_variable(var, gpath)
                    if stat:
                        numerical_audit.append(stat)

    # Variables with concerning missing values (> 10%)
    concerning_vars = [
        {
            "name": v["name"],
            "group_path": v["group_path"],
            "missing_percent": v["missing_percent"],
            "missing_count": v["missing_elements"],
            "total_elements": v["total_elements"],
            "note": "100% unpopulated" if v["missing_percent"] == 100.0 else (
                "Radar non-precipitating background" if "radar" in v["group_path"] and v["missing_percent"] > 90.0 else (
                    "Swath edge / land boundary" if "GPROF" in v["group_path"] else "Missing label"
                )
            )
        }
        for v in numerical_audit if v.get("missing_percent", 0.0) > 10.0
    ]

    report = {
        "audit_metadata": {
            "tool": "TC PRIMED Scientific Data-Quality Auditor",
            "audit_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "file_path": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_size_mb": file_size_mb
        },
        "storm_labels": storm_labels,
        "infrared_channel_verification": ir_channel_audit,
        "temporal_audit": temporal_audit,
        "geolocation_audit": geolocation_audit,
        "quality_flags_audit": quality_flags_audit,
        "concerning_missing_variables_count": len(concerning_vars),
        "concerning_missing_variables": concerning_vars,
        "total_numerical_variables_audited": len(numerical_audit),
        "numerical_variables": numerical_audit
    }
    return report


def generate_text_report(audit):
    lines = []
    w = 100
    meta = audit["audit_metadata"]
    lines.append("=" * w)
    lines.append(f"TC PRIMED SCIENTIFIC DATA QUALITY AUDIT: {meta['file_name']}")
    lines.append(f"Timestamp: {meta['audit_timestamp_utc']} | File Size: {meta['file_size_mb']} MB")
    lines.append(f"File Path: {meta['file_path']}")
    lines.append("=" * w)
    lines.append("")

    # 1. Storm Labels
    lines.append("1. STORM BEST-TRACK LABELS & SUPERVISED TARGET AUDIT")
    lines.append("-" * w)
    sl = audit["storm_labels"]
    lines.append(f"  Cyclone Identifier: {sl.get('basin')}{sl.get('cyclone_number', '')}{sl.get('season', '')} (Season: {sl.get('season')})")
    lines.append(f"  Instrument/Platform: {sl.get('instrument')} on {sl.get('platform')}")
    lines.append(f"  Center Coordinates: {sl.get('storm_latitude_deg_N')} deg N, {sl.get('storm_longitude_deg_E')} deg E")
    lines.append(f"  Intensity:          {sl.get('intensity_wind_kt')} kt  [{sl.get('documented_averaging_period', '')}]")
    lines.append(f"  Central Pressure:   {sl.get('central_min_pressure_hpa')} hPa")
    lines.append(f"  Development Stage:  {sl.get('development_stage')} ({sl.get('stage_long_name', '')})")
    lines.append(f"  Distance to Land:   {sl.get('distance_to_land_km')} km (positive = over ocean)")
    lines.append("  Intensity Change Targets (kt):")
    if "intensity_change_labels" in sl:
        for p, info in sl["intensity_change_labels"].items():
            val_str = f"{info['value_kt']:+d} kt" if info["value_kt"] is not None else "MISSING / NaN"
            reason = f"  ({info['reason_if_missing']})" if info.get("reason_if_missing") else ""
            lines.append(f"    dI {p:>4}: {val_str:<12}{reason}")
    lines.append("")

    # 2. Infrared Channel Verification
    lines.append("2. INFRARED CHANNEL & SENSOR METADATA VERIFICATION")
    lines.append("-" * w)
    ir = audit["infrared_channel_verification"]
    lines.append(f"  Variable:               {ir.get('variable_name')}")
    lines.append(f"  Long Name:              {ir.get('long_name')}")
    lines.append(f"  Units:                  {ir.get('units')}")
    lines.append(f"  Documented Wavelength:  {ir.get('documented_wavelength')}")
    lines.append(f"  Geostationary Platform: {ir.get('platform_source')}")
    lines.append(f"  Sensor/Instrument:      {ir.get('instrument_source')}")
    lines.append(f"  Source Granule:         {ir.get('source_granule')}")
    lines.append(f"  Calibration / Binning:  {ir.get('calibration_resolution')}")
    lines.append("")

    # 3. Geolocation Audit
    lines.append("3. GEOLOCATION & SENSOR COORDINATE ALIGNMENT AUDIT")
    lines.append("-" * w)
    geo = audit["geolocation_audit"]
    lines.append(f"  Sensor coordinate grids evaluated: {len(geo)}")
    for g in geo:
        lines.append(f"  * Group: {g['group_path']}")
        lines.append(f"      Grid Shape: {g['shape']}  Dimensions: {tuple(g['dimensions'])}")
        lines.append(f"      Valid Coordinate Pct: {100.0 - g['missing_percent']:.2f}% ({g['valid_points']:,} of {g['total_points']:,} pts)")
        lines.append(f"      Latitude Range:  [{g['latitude_range_deg_N'][0]} to {g['latitude_range_deg_N'][1]}] deg N")
        lines.append(f"      Longitude Range: [{g['longitude_range_deg_E'][0]} to {g['longitude_range_deg_E'][1]}] deg E")
        lines.append(f"      Aligned Observation Variables: {g['aligned_observations_count']} fields verified aligned")
        if g["misaligned_observations"]:
            lines.append(f"      MISALIGNED: {g['misaligned_observations']}")
    lines.append("")

    # 4. Temporal Audit
    lines.append("4. TEMPORAL & SCAN SYNCHRONIZATION AUDIT")
    lines.append("-" * w)
    tmp = audit["temporal_audit"]
    lines.append(f"  Nominal Overpass Observation Time: {tmp.get('nominal_overpass_time_utc')}")
    lines.append("  Cross-Modality Time Offsets relative to Nominal Microwave Overpass:")
    for c in tmp.get("cross_modality_synchronization", []):
        lines.append(f"    {c['group']:<30} {c['variable']:<10} -> {c['time_utc']} (offset: {c['offset_from_overpass_min']:+.2f} min / {c['offset_from_overpass_sec']:+.1f} s)")
    lines.append("")

    # 5. Quality Flags
    lines.append("5. QUALITY FLAGS & RETRIEVAL CONFIDENCE AUDIT")
    lines.append("-" * w)
    for qf in audit["quality_flags_audit"]:
        lines.append(f"  * {qf['group_path']}/{qf['name']}  [Shape: {tuple(qf['shape'])}, Dtype: {qf['dtype']}]")
        if qf.get("long_name"):
            lines.append(f"      Description: {qf['long_name']}")
        lines.append(f"      Valid Pixels: {qf['valid_pixels']:,} / {qf['total_pixels']:,}  (Missing: {qf['missing_pixels']})")
        lines.append("      Value Frequencies:")
        for val_k, count in qf["value_frequencies"].items():
            meaning = qf["documented_meanings"].get(val_k, "Documented meaning not present in attribute table")
            lines.append(f"        Val {val_k:>6}: {count:>8,} pixels  ->  {meaning}")
    lines.append("")

    # 6. Concerning Missing Variables
    lines.append("6. VARIABLES WITH CONCERNING MISSING-VALUE PERCENTAGES (> 10%)")
    lines.append("-" * w)
    lines.append(f"  Total flagged variables: {audit['concerning_missing_variables_count']}")
    lines.append(f"  {'Group Path':<35} {'Variable':<26} {'Missing %':<12} {'Missing/Total':<20} Note")
    lines.append("  " + "-" * (w - 2))
    for cv in audit["concerning_missing_variables"]:
        ratio_str = f"{cv['missing_count']:,} / {cv['total_elements']:,}"
        lines.append(f"  {cv['group_path']:<35} {cv['name']:<26} {cv['missing_percent']:>7.2f}%    {ratio_str:<20} {cv['note']}")
    lines.append("")

    # 7. Numerical Summary Table
    lines.append("7. SELECTED PHYSICAL OBSERVATION VARIABLES: NUMERICAL DISTRIBUTION")
    lines.append("-" * w)
    lines.append(f"  {'Variable':<22} {'Units':<10} {'Valid %':<9} {'Min':<9} {'Max':<9} {'Mean':<9} {'Std':<9} {'Median(p50)':<10}")
    lines.append("  " + "-" * (w - 2))
    for v in audit["numerical_variables"]:
        # Select prominent physical observation variables
        is_key = any(k in v["name"] for k in ["TB_", "IRWIN", "surfacePrecipitation", "rainWaterPath", "iceWaterPath", "nearSurfPrecipTotRate", "correctedReflectFactor"])
        if is_key and v["valid_elements"] > 0 and v["mean"] is not None:
            units_str = str(v["units"]) if v["units"] else "-"
            valid_pct = f"{100.0 - v['missing_percent']:.1f}%"
            p50 = f"{v['percentiles']['p50_median']:.2f}" if v.get("percentiles") else "-"
            lines.append(f"  {v['name']:<22} {units_str:<10} {valid_pct:<9} {v['min']:<9.2f} {v['max']:<9.2f} {v['mean']:<9.2f} {v['std']:<9.2f} {p50:<10}")

    lines.append("")
    lines.append("=" * w)
    lines.append("END OF SCIENTIFIC DATA QUALITY AUDIT REPORT")
    lines.append("=" * w)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TC PRIMED NetCDF Scientific Data-Quality Audit Utility")
    parser.add_argument("input_file", help="Path to NetCDF file to audit")
    parser.add_argument("--output-dir", default="reports", help="Directory where reports are saved")
    parser.add_argument("--json-out", default=None, help="Custom JSON output path")
    parser.add_argument("--txt-out", default=None, help="Custom text output path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = args.json_out or os.path.join(args.output_dir, "sample_gmi_qc.json")
    txt_path = args.txt_out or os.path.join(args.output_dir, "sample_gmi_qc.txt")

    print(f"\n[AUDIT] Commencing scientific data-quality audit on: {args.input_file}")
    audit_data = audit_netcdf(args.input_file)

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"[OK] Saved JSON QC Report -> {json_path}")

    # Save Text report
    text_report = generate_text_report(audit_data)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"[OK] Saved Text QC Report -> {txt_path}")

    # Summary stdout
    meta = audit_data["audit_metadata"]
    sl = audit_data["storm_labels"]
    print(f"\nAudit completed for {meta['file_name']}:")
    print(f"  Cyclone: {sl.get('basin')}{sl.get('cyclone_number')}{sl.get('season')} | Intensity: {sl.get('intensity_wind_kt')} kt | Stage: {sl.get('development_stage')}")
    print(f"  Numerical Variables Audited: {audit_data['total_numerical_variables_audited']}")
    print(f"  Flag Variables Audited:      {len(audit_data['quality_flags_audit'])}")
    print(f"  Variables with >10% Missing: {audit_data['concerning_missing_variables_count']}")


if __name__ == "__main__":
    main()

