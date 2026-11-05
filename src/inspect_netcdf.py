"""
inspect_netcdf.py - Research-grade NetCDF inspection utility for TC PRIMED dataset.

Inspects any NetCDF-4 file recursively and extracts complete metadata,
hierarchical group structures, variable definitions, and modality availability.

Produces:
  - JSON inventory report
  - Human-readable text report
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
import numpy as np
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
    """Returns a clean Python scalar, handling masked arrays and sentinels."""
    if v is None or np.ma.is_masked(v):
        return None
    v = np.ma.getdata(v)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        v = v.ravel()[0]
    return to_serializable(v)


def walk_groups(group, path=""):
    """Recursively yields (group_path, group_object)."""
    current_path = f"{path}/{group.name}" if path else f"/{group.name}"
    yield current_path, group
    for sub_name, sub_group in group.groups.items():
        yield from walk_groups(sub_group, current_path)


def inspect_variable(var, group_path):
    """Extracts all required metadata attributes for a single variable."""
    # Attribute extraction helpers
    attrs = {}
    for a in var.ncattrs():
        try:
            attrs[a] = to_serializable(var.getncattr(a))
        except Exception as e:
            attrs[a] = f"<Error reading attr {a}: {e}>"

    # Specific required fields
    units = attrs.get("units")
    long_name = attrs.get("long_name")
    
    fill_value = None
    if hasattr(var, "_FillValue"):
        fill_value = to_serializable(var._FillValue)
    elif "_FillValue" in attrs:
        fill_value = attrs["_FillValue"]
    
    missing_value = attrs.get("missing_value")
    scale_factor = attrs.get("scale_factor")
    add_offset = attrs.get("add_offset")
    valid_range = attrs.get("valid_range")

    return {
        "name": var.name,
        "group_path": group_path,
        "shape": list(var.shape),
        "dimensions": list(var.dimensions),
        "dtype": str(var.dtype),
        "units": units,
        "long_name": long_name,
        "fill_value": fill_value,
        "missing_value": missing_value,
        "scale_factor": scale_factor,
        "add_offset": add_offset,
        "valid_range": valid_range,
        "attributes": attrs
    }


def analyze_modalities(root_nc):
    """
    Identifies available observation groups and detects missing or flagged modalities.
    Standard TC PRIMED overpass modalities:
      1. passive_microwave (PMW brightness temperatures)
      2. infrared (Geostationary IR brightness temperatures)
      3. GPROF (Precipitation retrievals)
      4. radar_radiometer (Active radar precipitation)
    """
    all_groups = set(root_nc.groups.keys())
    standard_modalities = {
        "passive_microwave": "Passive Microwave Imagery",
        "infrared": "Geostationary Infrared (IR)",
        "GPROF": "NASA GPROF Precipitation Retrievals",
        "radar_radiometer": "TRMM/GPM Radar-Radiometer Observation"
    }

    available = []
    missing = []
    modality_details = {}

    for mod_key, mod_desc in standard_modalities.items():
        if mod_key in all_groups:
            grp = root_nc.groups[mod_key]
            detail = {
                "present_in_file": True,
                "description": mod_desc,
                "subgroups": list(grp.groups.keys()),
                "variable_count": len(grp.variables)
            }

            # Check availability flags when present
            flag = None
            if mod_key == "infrared":
                if "infrared_availability_flag" in grp.variables:
                    flag = clean_scalar(grp.variables["infrared_availability_flag"][:])
                elif "R5_availability_flag" in grp.variables:
                    flag = clean_scalar(grp.variables["R5_availability_flag"][:])
            elif mod_key in ("GPROF", "radar_radiometer"):
                if "availability_flag" in grp.variables:
                    flag = clean_scalar(grp.variables["availability_flag"][:])

            detail["availability_flag"] = flag

            # Check channels in passive microwave
            if mod_key == "passive_microwave":
                swaths = {}
                for sw_name, sw_grp in grp.groups.items():
                    channels = [v for v in sw_grp.variables if v.startswith("TB_")]
                    swaths[sw_name] = {
                        "channels": channels,
                        "num_channels": len(channels),
                        "spatial_shape": list(sw_grp.variables["latitude"].shape) if "latitude" in sw_grp.variables else None
                    }
                detail["swaths"] = swaths

            # Modality is effectively available if group exists and flag != 0
            is_usable = (flag is None) or (isinstance(flag, (int, float)) and flag > 0)
            detail["is_usable"] = is_usable
            available.append(mod_key)
            modality_details[mod_key] = detail
        else:
            missing.append(mod_key)
            modality_details[mod_key] = {
                "present_in_file": False,
                "description": mod_desc,
                "is_usable": False
            }

    return {
        "standard_modalities_checked": list(standard_modalities.keys()),
        "available_observation_groups": available,
        "missing_observation_groups": missing,
        "modality_details": modality_details
    }


def inspect_file(file_path):
    """
    Performs full recursive inspection of a NetCDF file.
    Does not modify the original file or alter scientific data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    with nc.Dataset(file_path, "r") as root:
        # Root Global Attributes
        global_attributes = {}
        for a in root.ncattrs():
            try:
                global_attributes[a] = to_serializable(root.getncattr(a))
            except Exception as e:
                global_attributes[a] = f"<Error: {e}>"

        # Root Dimensions
        root_dimensions = {name: len(dim) for name, dim in root.dimensions.items()}

        # Top-level Groups
        top_level_groups = list(root.groups.keys())

        # Collect all groups recursively
        all_group_paths = ["/"]
        all_variables = []

        # Root variables
        for vname, var in root.variables.items():
            all_variables.append(inspect_variable(var, "/"))

        # Recursive groups
        group_tree = {}
        for gpath, grp in walk_groups(root):
            all_group_paths.append(gpath)
            group_tree[gpath] = {
                "name": grp.name,
                "dimensions": {dname: len(d) for dname, d in grp.dimensions.items()},
                "variable_count": len(grp.variables),
                "subgroups": list(grp.groups.keys())
            }
            for vname, var in grp.variables.items():
                all_variables.append(inspect_variable(var, gpath))

        # Modalities analysis
        modalities_analysis = analyze_modalities(root)

        # Basic Overpass Metadata / Labels extraction if available
        storm_summary = {}
        meta_grp = root.groups.get("overpass_metadata")
        storm_grp = root.groups.get("overpass_storm_metadata")
        if meta_grp:
            for k in ["basin", "season", "cyclone_number", "instrument_name", "platform_name", "coverage_fraction"]:
                if k in meta_grp.variables:
                    storm_summary[k] = clean_scalar(meta_grp.variables[k][:])
        if storm_grp:
            for k in ["storm_latitude", "storm_longitude", "intensity", "central_min_pressure", "development_level", "distance_to_land"]:
                if k in storm_grp.variables:
                    storm_summary[k] = clean_scalar(storm_grp.variables[k][:])

    inventory = {
        "inspection_metadata": {
            "tool_name": "TC PRIMED NetCDF Inspector",
            "inspection_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target_file": os.path.abspath(file_path),
            "file_name": os.path.basename(file_path),
            "file_size_bytes": file_size_bytes,
            "file_size_mb": file_size_mb
        },
        "global_attributes": global_attributes,
        "root_dimensions": root_dimensions,
        "top_level_groups": top_level_groups,
        "all_group_paths": all_group_paths,
        "group_tree": group_tree,
        "modalities_summary": modalities_analysis,
        "storm_summary": storm_summary,
        "total_variables_count": len(all_variables),
        "variables": all_variables
    }
    return inventory


def generate_text_report(inv):
    """Builds a comprehensive, readable plain-text report."""
    meta = inv["inspection_metadata"]
    lines = []
    w = 95
    lines.append("=" * w)
    lines.append(f"TC PRIMED NetCDF Inspection Report: {meta['file_name']}")
    lines.append(f"Inspection Timestamp: {meta['inspection_time_utc']}")
    lines.append(f"Full Path: {meta['target_file']}")
    lines.append(f"File Size: {meta['file_size_mb']} MB ({meta['file_size_bytes']:,} bytes)")
    lines.append("=" * w)
    lines.append("")

    # 1. Storm Summary
    ss = inv.get("storm_summary", {})
    if ss:
        lines.append("1. STORM IDENTIFICATION & BEST-TRACK LABELS")
        lines.append("-" * w)
        lines.append(f"  Basin:              {ss.get('basin')} | Season: {ss.get('season')} | Storm ID: {ss.get('cyclone_number')}")
        lines.append(f"  Instrument/Platform: {ss.get('instrument_name')} on {ss.get('platform_name')}")
        lines.append(f"  Position:           {ss.get('storm_latitude')} deg N, {ss.get('storm_longitude')} deg E")
        lines.append(f"  Intensity:          {ss.get('intensity')} knots (1-min max sustained wind)")
        lines.append(f"  Central Min Pressure:{ss.get('central_min_pressure')} hPa")
        lines.append(f"  Development Stage:  {ss.get('development_level')}")
        lines.append(f"  Distance to Land:   {ss.get('distance_to_land')} km")
        lines.append(f"  Coverage Fraction:  {ss.get('coverage_fraction')}%")
        lines.append("")

    # 2. Modality Analysis
    lines.append("2. MODALITY AUDIT (AVAILABLE VS MISSING OBSERVATIONS)")
    lines.append("-" * w)
    mod = inv["modalities_summary"]
    for k in mod["standard_modalities_checked"]:
        info = mod["modality_details"].get(k, {})
        status = "[PRESENT & USABLE]" if info.get("is_usable") else ("[PRESENT BUT FLAGGED/UNUSABLE]" if info.get("present_in_file") else "[MISSING]")
        lines.append(f"  * {k.upper():<20} {status:<30} - {info.get('description', '')}")
        if info.get("present_in_file"):
            if "availability_flag" in info and info["availability_flag"] is not None:
                lines.append(f"      Availability Flag: {info['availability_flag']}")
            if "swaths" in info:
                for sw, sdata in info["swaths"].items():
                    chans = ", ".join(sdata["channels"]) if sdata["channels"] else "None"
                    shape_str = f", shape: {sdata['spatial_shape']}" if sdata['spatial_shape'] else ""
                    lines.append(f"      Swath '{sw}': {sdata['num_channels']} channels ({chans}{shape_str})")
            if "subgroups" in info and info["subgroups"] and "swaths" not in info:
                lines.append(f"      Subgroups: {', '.join(info['subgroups'])}")
    lines.append("")

    # 3. Global Attributes
    lines.append("3. ROOT GLOBAL ATTRIBUTES")
    lines.append("-" * w)
    for k, v in sorted(inv["global_attributes"].items()):
        val_str = str(v)
        if len(val_str) > 120:
            val_str = val_str[:117] + "..."
        lines.append(f"  {k:<35}: {val_str}")
    lines.append("")

    # 4. Root Dimensions
    lines.append("4. ROOT DIMENSIONS")
    lines.append("-" * w)
    if inv["root_dimensions"]:
        for dname, dlen in inv["root_dimensions"].items():
            lines.append(f"  {dname} = {dlen}")
    else:
        lines.append("  (No root-level dimensions defined; dimensions reside within groups)")
    lines.append("")

    # 5. Group Hierarchy
    lines.append("5. DISCOVERED GROUP HIERARCHY")
    lines.append("-" * w)
    lines.append(f"  Total groups discovered: {len(inv['all_group_paths'])}")
    for gpath in inv["all_group_paths"]:
        if gpath == "/":
            lines.append("  / (root)")
        else:
            gdata = inv["group_tree"].get(gpath, {})
            dims_str = ", ".join(f"{d}={l}" for d, l in gdata.get("dimensions", {}).items())
            lines.append(f"  {gpath}  [Vars: {gdata.get('variable_count', 0)}, Dims: ({dims_str})]")
    lines.append("")

    # 6. Detailed Variable Inventory
    lines.append("6. COMPLETE VARIABLE INVENTORY")
    lines.append("-" * w)
    lines.append(f"  Total variables: {inv['total_variables_count']}")
    lines.append("")

    # Group variables by group_path for readability
    by_group = {}
    for v in inv["variables"]:
        gp = v["group_path"]
        by_group.setdefault(gp, []).append(v)

    col_fmt = "  {:<26} {:<16} {:<12} {:<24} {:<12}"
    for gp, vlist in sorted(by_group.items()):
        lines.append(f"  === Group: {gp} ({len(vlist)} variables) ===")
        lines.append(col_fmt.format("Variable Name", "Shape", "Dtype", "Dimensions", "Units"))
        lines.append("  " + "-" * (w - 2))
        for v in vlist:
            shape_str = str(tuple(v["shape"]))
            dims_str = ",".join(v["dimensions"])
            if len(dims_str) > 22:
                dims_str = dims_str[:19] + "..."
            units_str = str(v["units"]) if v["units"] is not None else "-"
            if len(units_str) > 11:
                units_str = units_str[:10] + "."
            lines.append(col_fmt.format(v["name"], shape_str, v["dtype"], dims_str, units_str))
            
            # Print additional line if long_name, fill_value, or scale exists
            extra = []
            if v["long_name"]:
                extra.append(f"long_name: '{v['long_name']}'")
            if v["fill_value"] is not None:
                extra.append(f"_FillValue: {v['fill_value']}")
            if v["missing_value"] is not None:
                extra.append(f"missing_value: {v['missing_value']}")
            if v["scale_factor"] is not None:
                extra.append(f"scale_factor: {v['scale_factor']}")
            if v["add_offset"] is not None:
                extra.append(f"add_offset: {v['add_offset']}")
            if extra:
                lines.append(f"    -> {'; '.join(extra)}")
        lines.append("")

    lines.append("=" * w)
    lines.append("END OF INSPECTION REPORT")
    lines.append("=" * w)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TC PRIMED NetCDF Inspection Utility")
    parser.add_argument("input_file", help="Path to the NetCDF file to inspect")
    parser.add_argument("--output-dir", default="reports", help="Directory where reports are saved")
    parser.add_argument("--json-out", default=None, help="Custom path for JSON output")
    parser.add_argument("--txt-out", default=None, help="Custom path for text output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = args.json_out or os.path.join(args.output_dir, "sample_netcdf_inventory.json")
    txt_path = args.txt_out or os.path.join(args.output_dir, "sample_netcdf_inventory.txt")

    print(f"\n[INSPECT] Opening and analyzing: {args.input_file}")
    inventory = inspect_file(args.input_file)

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"[OK] Saved JSON inventory -> {json_path}")

    # Save Text report
    report_text = generate_text_report(inventory)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Saved text report   -> {txt_path}")

    # Output executive summary to stdout
    meta = inventory["inspection_metadata"]
    mods = inventory["modalities_summary"]
    ss = inventory.get("storm_summary", {})
    print(f"\nExecutive Summary for {meta['file_name']}:")
    print(f"  File Size: {meta['file_size_mb']} MB")
    print(f"  Basin: {ss.get('basin', 'N/A')} | Storm: {ss.get('cyclone_number', 'N/A')} | Season: {ss.get('season', 'N/A')}")
    print(f"  Instrument / Platform: {ss.get('instrument_name', 'N/A')} on {ss.get('platform_name', 'N/A')}")
    print(f"  Intensity: {ss.get('intensity', 'N/A')} kt | Stage: {ss.get('development_level', 'N/A')}")
    print(f"  Discovered Groups: {len(inventory['all_group_paths'])} groups")
    print(f"  Discovered Variables: {inventory['total_variables_count']} variables")
    print(f"  Observation Groups Available: {mods['available_observation_groups']}")
    print(f"  Observation Groups Missing:   {mods['missing_observation_groups']}")


if __name__ == "__main__":
    main()
