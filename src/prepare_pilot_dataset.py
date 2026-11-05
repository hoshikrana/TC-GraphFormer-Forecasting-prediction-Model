"""
prepare_pilot_dataset.py - Stratified 20 GB Pilot Dataset Planner and Downloader
for TC PRIMED (Tropical Cyclone PRecipitation, Infrared, Microwave, and Environmental Dataset).

Operations:
  plan     - Discover S3 NetCDF files, produce a balanced pilot manifest (no downloads).
  download - Download only the files recorded in the approved manifest.

Usage:
  python src/prepare_pilot_dataset.py plan --target-gb 20 --seed 42
  python src/prepare_pilot_dataset.py download --manifest reports/pilot_manifest.csv --destination data/raw/pilot --workers 4
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import shutil
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKET = "noaa-nesdis-tcprimed-pds"
PREFIX = "v01r01/final/"
S3_URI = f"s3://{BUCKET}/{PREFIX}"

DEFAULT_TARGET_BYTES = 20_000_000_000  # 20 GB decimal
DEFAULT_SEED = 42
DEFAULT_WORKERS = 4
SAFETY_BUFFER_BYTES = 2_000_000_000  # 2 GB
MAX_RETRIES = 3
RETRY_DELAY_SEC = 5

# TC PRIMED basins (from IBTrACS convention)
BASIN_NAMES = {
    "AL": "North Atlantic",
    "EP": "Eastern North Pacific",
    "WP": "Western North Pacific",
    "IO": "North Indian Ocean",
    "SH": "Southern Hemisphere",
    "CP": "Central North Pacific",
    "SL": "South Atlantic",  # rare
}

# Historical period strata
ERA_BINS = [
    ("1987-1999", 1987, 1999),
    ("2000-2009", 2000, 2009),
    ("2010-2019", 2010, 2019),
    ("2020+",     2020, 9999),
]

# Known TC PRIMED instruments (for parsing filenames)
KNOWN_INSTRUMENTS = [
    "GMI", "TMI", "SSMI", "SSMIS", "AMSR2", "AMSRE", "AMSUB", "AMSU-B",
    "AMSUA", "AMSU-A", "MHS", "ATMS", "SAPHIR", "MWRI", "MWHS2",
    "TEMPEST-D", "TEMPEST", "COWVR",
]

KNOWN_PLATFORMS = [
    "GPM", "TRMM", "F08", "F10", "F11", "F13", "F14", "F15", "F16",
    "F17", "F18", "F19", "F20", "GCOMW1", "AQUA", "NOAA15", "NOAA16",
    "NOAA17", "NOAA18", "NOAA19", "NOAA20", "NOAA21", "METOPA", "METOPB",
    "METOPC", "MEGHAT", "NPP", "SNPP", "ISS", "FY3D", "FY3E",
]


# ===========================================================================
# Utility Functions
# ===========================================================================

def classify_era(season: int) -> str:
    """Return the historical-period label for a given season year."""
    for label, lo, hi in ERA_BINS:
        if lo <= season <= hi:
            return label
    return "unknown"


def parse_filename_instrument_platform(filename: str):
    """
    Parse instrument and platform from a TC PRIMED filename.

    Typical filename patterns:
      TCPRIMED_v01r01-final_{basin}{number}{season}_{instrument}_{platform}_{orbit}_{timestamp}.nc
      TCPRIMED_v01r01-final_{basin}{number}{season}_era5_environmental.nc
    """
    instrument = None
    platform = None

    base = filename.replace(".nc", "")
    parts = base.split("_")

    # Environmental files
    if is_environmental_file(filename):
        return "ERA5", "reanalysis"

    # Try to find instrument and platform tokens
    for i, part in enumerate(parts):
        part_upper = part.upper()
        for inst in KNOWN_INSTRUMENTS:
            if part_upper == inst.upper() or part_upper == inst.replace("-", "").upper():
                instrument = inst
                # platform is typically the next token
                if i + 1 < len(parts):
                    platform = parts[i + 1]
                break
        if instrument:
            break

    return instrument, platform


def is_environmental_file(filename: str) -> bool:
    """
    Detect environmental NetCDF files.
    TC PRIMED standard: TCPRIMED_v01r01-final_{storm_id}_env_s{start}_e{end}.nc
    Also matches 'environmental', 'era5', or '_env.nc'.
    """
    lower = filename.lower()
    return "_env_" in lower or "environmental" in lower or "era5" in lower or lower.endswith("_env.nc")


def parse_storm_folder(key: str):
    """
    Extract (season, basin, cyclone_number, storm_folder) from an S3 key.

    Expected path structure:
      v01r01/final/{season}/{basin}/{cyclone_number}/filename.nc
    """
    # Remove leading prefix if present
    rel = key
    if rel.startswith(PREFIX):
        rel = rel[len(PREFIX):]

    parts = rel.split("/")
    if len(parts) < 4:
        return None

    season_str, basin, cyclone_num = parts[0], parts[1], parts[2]
    try:
        season = int(season_str)
    except ValueError:
        return None

    storm_folder = f"{PREFIX}{season_str}/{basin}/{cyclone_num}"
    filename = parts[-1]
    return {
        "season": season,
        "basin": basin,
        "cyclone_number": cyclone_num,
        "storm_folder": storm_folder,
        "filename": filename,
    }


def storm_id_from_parts(season, basin, cyclone_number):
    """Create a unique storm identifier."""
    return f"{basin}{str(cyclone_number).zfill(2)}{season}"


def get_aws_cli_cmd():
    """
    Get the command prefix to run AWS CLI using the active Python interpreter.
    Always returns [sys.executable, "-m", "awscli"] to guarantee consistency,
    preventing Windows .cmd / PATH / WinError 2 issues.
    """
    return [sys.executable, "-m", "awscli"]


# ===========================================================================
# S3 Inventory
# ===========================================================================

def run_s3_listing():
    """
    Run the AWS CLI to list all objects under the TC PRIMED final prefix.
    Returns raw listing lines. Handles pagination automatically (AWS CLI does this).
    """
    cmd = get_aws_cli_cmd() + [
        "s3", "ls", S3_URI,
        "--recursive", "--no-sign-request",
    ]

    print(f"[INVENTORY] Running: {' '.join(cmd)}")
    print("[INVENTORY] This may take several minutes for the complete bucket listing...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
    except subprocess.TimeoutExpired:
        print("[ERROR] S3 listing timed out after 1 hour.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] AWS CLI not found. Please install and configure AWS CLI.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"[ERROR] AWS CLI failed with return code {result.returncode}", file=sys.stderr)
        print(f"[ERROR] stderr: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = result.stdout.strip().split("\n")
    lines = [l for l in lines if l.strip()]
    print(f"[INVENTORY] Received {len(lines)} raw listing lines.")
    return lines


def parse_s3_listing_line(line: str):
    """
    Parse a single line from `aws s3 ls --recursive` output.

    Format: "2024-01-15 12:34:56    12345678 v01r01/final/2014/IO/03/file.nc"

    Returns dict with key and size, or None if line is unparseable / not a file.
    """
    line = line.strip()
    if not line:
        return None

    # Pattern: date time size key
    match = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\d+)\s+(.+)$",
        line,
    )
    if not match:
        return None

    size_bytes = int(match.group(3))
    key = match.group(4).strip()

    return {"key": key, "size_bytes": size_bytes}


def build_inventory(listing_lines):
    """
    Parse all listing lines and build the NetCDF inventory.
    Returns list of inventory dicts.
    """
    inventory = []
    skipped = 0

    for line in listing_lines:
        parsed = parse_s3_listing_line(line)
        if parsed is None:
            skipped += 1
            continue

        key = parsed["key"]
        size_bytes = parsed["size_bytes"]

        # Only .nc files
        if not key.lower().endswith(".nc"):
            skipped += 1
            continue

        # Only files under v01r01/final/
        if not key.startswith(PREFIX):
            skipped += 1
            continue

        # Parse storm folder structure
        storm_info = parse_storm_folder(key)
        if storm_info is None:
            skipped += 1
            continue

        filename = storm_info["filename"]
        instrument, platform = parse_filename_instrument_platform(filename)
        env_file = is_environmental_file(filename)

        inventory.append({
            "s3_key": key,
            "size_bytes": size_bytes,
            "season": storm_info["season"],
            "basin": storm_info["basin"],
            "cyclone_number": storm_info["cyclone_number"],
            "storm_folder": storm_info["storm_folder"],
            "filename": filename,
            "instrument": instrument or "",
            "platform": platform or "",
            "is_environmental": env_file,
        })

    print(f"[INVENTORY] Parsed {len(inventory)} NetCDF objects, skipped {skipped} non-matching lines.")
    return inventory


def save_inventory_csv(inventory, path):
    """Save the full S3 inventory to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "s3_key", "size_bytes", "season", "basin", "cyclone_number",
        "storm_folder", "filename", "instrument", "platform", "is_environmental",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)
    print(f"[OK] Saved S3 inventory -> {path} ({len(inventory)} rows)")


def build_storm_catalogue(inventory):
    """
    Group inventory by storm folder and compute per-storm summary.
    Returns list of storm-catalogue dicts.
    """
    storms = defaultdict(lambda: {
        "files": [],
        "total_bytes": 0,
        "file_count": 0,
        "overpass_count": 0,
        "env_count": 0,
        "instruments": set(),
        "platforms": set(),
    })

    for row in inventory:
        sf = row["storm_folder"]
        storms[sf]["files"].append(row)
        storms[sf]["total_bytes"] += row["size_bytes"]
        storms[sf]["file_count"] += 1
        if row["is_environmental"]:
            storms[sf]["env_count"] += 1
        else:
            storms[sf]["overpass_count"] += 1
        if row["instrument"]:
            storms[sf]["instruments"].add(row["instrument"])
        if row["platform"]:
            storms[sf]["platforms"].add(row["platform"])

    catalogue = []
    for sf, data in sorted(storms.items()):
        first = data["files"][0]
        storm_id = storm_id_from_parts(first["season"], first["basin"], first["cyclone_number"])
        catalogue.append({
            "storm_folder": sf,
            "storm_id": storm_id,
            "season": first["season"],
            "basin": first["basin"],
            "cyclone_number": first["cyclone_number"],
            "era": classify_era(first["season"]),
            "total_bytes": data["total_bytes"],
            "file_count": data["file_count"],
            "overpass_count": data["overpass_count"],
            "env_count": data["env_count"],
            "instruments": ";".join(sorted(data["instruments"])),
            "platforms": ";".join(sorted(data["platforms"])),
        })

    return catalogue


def save_storm_catalogue_csv(catalogue, path):
    """Save the storm catalogue to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "storm_folder", "storm_id", "season", "basin", "cyclone_number",
        "era", "total_bytes", "file_count", "overpass_count", "env_count",
        "instruments", "platforms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalogue)
    print(f"[OK] Saved storm catalogue -> {path} ({len(catalogue)} storms)")


# ===========================================================================
# Stratified Pilot Selection
# ===========================================================================

def stratified_select(catalogue, target_bytes, seed, nio_fraction=0.25):
    """
    Perform stratified selection of complete storm folders.

    Strategy:
      1. Partition storms by (era, basin).
      2. Reserve ~25% of budget for IO basin.
      3. Distribute remaining ~75% across other basins.
      4. Within each allocation, balance across eras (~25% each).
      5. Within each (era, basin) cell, randomly select complete storm folders.
      6. If a folder would exceed remaining budget, skip it and try another.
      7. Never exceed target_bytes.

    Returns (selected_storms, selection_report).
    """
    rng = random.Random(seed)

    # Group storms by (era, basin)
    cells = defaultdict(list)
    for storm in catalogue:
        era = storm["era"]
        basin = storm["basin"]
        cells[(era, basin)].append(storm)

    # Shuffle each cell deterministically
    for key in cells:
        rng.shuffle(cells[key])

    # Identify available basins and eras
    available_basins = sorted(set(s["basin"] for s in catalogue))
    available_eras = sorted(set(s["era"] for s in catalogue))

    # Compute budgets
    io_budget = int(target_bytes * nio_fraction) if "IO" in available_basins else 0
    other_budget = target_bytes - io_budget

    other_basins = [b for b in available_basins if b != "IO"]
    num_other_basins = len(other_basins)

    # Per-basin budget for non-IO basins (roughly equal)
    basin_budgets = {}
    if "IO" in available_basins:
        basin_budgets["IO"] = io_budget
    if num_other_basins > 0:
        per_other = other_budget // num_other_basins
        for b in other_basins:
            basin_budgets[b] = per_other

    # Per-era budget within each basin
    num_eras = len(available_eras)
    era_basin_budgets = {}
    for basin, basin_total in basin_budgets.items():
        per_era = basin_total // max(num_eras, 1)
        for era in available_eras:
            era_basin_budgets[(era, basin)] = per_era

    # Selection pass: iterate through cells and pick storms
    selected = []
    selected_bytes = 0
    selected_folders = set()
    cell_used_bytes = defaultdict(int)

    # First pass: try to fill each cell up to its budget
    for (era, basin) in sorted(era_basin_budgets.keys()):
        cell_budget = era_basin_budgets[(era, basin)]
        cell_storms = cells.get((era, basin), [])

        for storm in cell_storms:
            sf = storm["storm_folder"]
            if sf in selected_folders:
                continue
            storm_bytes = storm["total_bytes"]

            # Would adding this storm exceed the cell budget?
            if cell_used_bytes[(era, basin)] + storm_bytes > cell_budget:
                continue

            # Would adding this storm exceed total target?
            if selected_bytes + storm_bytes > target_bytes:
                continue

            selected.append(storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes
            cell_used_bytes[(era, basin)] += storm_bytes

    # Second pass: try to fill remaining budget from underrepresented cells
    # This helps when some cells have large storms that didn't fit in the first pass
    remaining = target_bytes - selected_bytes
    if remaining > 0:
        # Collect all unselected storms, shuffle them
        unselected = [s for s in catalogue if s["storm_folder"] not in selected_folders]
        rng.shuffle(unselected)

        # Try to add storms preferring underrepresented eras/basins
        # Sort by (era representation deficit, basin representation deficit)
        era_bytes = defaultdict(int)
        basin_bytes_used = defaultdict(int)
        for s in selected:
            era_bytes[s["era"]] += s["total_bytes"]
            basin_bytes_used[s["basin"]] += s["total_bytes"]

        target_per_era = target_bytes / max(num_eras, 1)
        target_per_basin_io = io_budget
        target_per_other = other_budget / max(num_other_basins, 1) if num_other_basins > 0 else 0

        def deficit_score(storm):
            era_deficit = target_per_era - era_bytes.get(storm["era"], 0)
            if storm["basin"] == "IO":
                basin_deficit = target_per_basin_io - basin_bytes_used.get("IO", 0)
            else:
                basin_deficit = target_per_other - basin_bytes_used.get(storm["basin"], 0)
            return era_deficit + basin_deficit

        # Sort unselected by deficit (highest deficit first = most underrepresented)
        unselected.sort(key=deficit_score, reverse=True)

        for storm in unselected:
            sf = storm["storm_folder"]
            if sf in selected_folders:
                continue
            storm_bytes = storm["total_bytes"]
            if selected_bytes + storm_bytes > target_bytes:
                continue
            selected.append(storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes
            era_bytes[storm["era"]] += storm_bytes
            basin_bytes_used[storm["basin"]] += storm_bytes

    # Third pass: greedy fill with any remaining storms that fit
    remaining = target_bytes - selected_bytes
    if remaining > 0:
        leftover = [s for s in catalogue if s["storm_folder"] not in selected_folders]
        rng.shuffle(leftover)
        for storm in leftover:
            sf = storm["storm_folder"]
            if sf in selected_folders:
                continue
            storm_bytes = storm["total_bytes"]
            if selected_bytes + storm_bytes > target_bytes:
                continue
            selected.append(storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes

    # Build selection report
    report = build_selection_report(selected, catalogue, target_bytes, available_basins, available_eras)

    return selected, report


def build_selection_report(selected, catalogue, target_bytes, available_basins, available_eras):
    """Build a detailed report about the selection."""

    total_selected_bytes = sum(s["total_bytes"] for s in selected)
    total_files = sum(s["file_count"] for s in selected)
    total_overpasses = sum(s["overpass_count"] for s in selected)
    total_env = sum(s["env_count"] for s in selected)
    total_storms = len(selected)

    # Per-basin
    basin_stats = defaultdict(lambda: {"storms": 0, "bytes": 0, "files": 0})
    for s in selected:
        basin_stats[s["basin"]]["storms"] += 1
        basin_stats[s["basin"]]["bytes"] += s["total_bytes"]
        basin_stats[s["basin"]]["files"] += s["file_count"]

    # Per-era
    era_stats = defaultdict(lambda: {"storms": 0, "bytes": 0, "files": 0})
    for s in selected:
        era_stats[s["era"]]["storms"] += 1
        era_stats[s["era"]]["bytes"] += s["total_bytes"]
        era_stats[s["era"]]["files"] += s["file_count"]

    # Per-year
    year_bytes = defaultdict(int)
    for s in selected:
        year_bytes[s["season"]] += s["total_bytes"]

    # Instruments and platforms
    instruments = defaultdict(int)
    platforms = defaultdict(int)
    for s in selected:
        for inst in s["instruments"].split(";"):
            if inst:
                instruments[inst] += 1
        for plat in s["platforms"].split(";"):
            if plat:
                platforms[plat] += 1

    # Era x Basin matrix
    matrix = {}
    for era in sorted(set(s["era"] for s in selected)):
        matrix[era] = {}
        for basin in sorted(set(s["basin"] for s in selected)):
            cell_storms = [s for s in selected if s["era"] == era and s["basin"] == basin]
            cell_bytes = sum(s["total_bytes"] for s in cell_storms)
            matrix[era][basin] = {
                "storms": len(cell_storms),
                "bytes": cell_bytes,
            }

    # Largest single-year contribution
    max_year = max(year_bytes, key=year_bytes.get) if year_bytes else None
    max_year_bytes = year_bytes[max_year] if max_year else 0
    max_year_pct = (max_year_bytes / total_selected_bytes * 100) if total_selected_bytes > 0 else 0

    # NIO allocation check
    io_bytes = basin_stats.get("IO", {}).get("bytes", 0)
    io_pct = (io_bytes / total_selected_bytes * 100) if total_selected_bytes > 0 else 0

    # Distinct seasons
    distinct_seasons = sorted(set(s["season"] for s in selected))
    distinct_eras = sorted(set(s["era"] for s in selected))
    distinct_basins = sorted(set(s["basin"] for s in selected))

    # Limitations
    limitations = []
    if total_selected_bytes < 18_000_000_000:
        limitations.append(f"Selected {total_selected_bytes / 1e9:.2f} GB is below the 18 GB minimum target.")
    if len(distinct_eras) < len(available_eras):
        missing_eras = set(available_eras) - set(distinct_eras)
        limitations.append(f"Missing historical periods: {missing_eras}")
    if len(distinct_basins) < len(available_basins):
        missing_basins = set(available_basins) - set(distinct_basins)
        limitations.append(f"Missing basins: {missing_basins}")
    if max_year_pct > 10:
        limitations.append(f"Year {max_year} contributes {max_year_pct:.1f}% of selected bytes (target: <=10%).")
    if abs(io_pct - 25) > 10:
        limitations.append(f"NIO allocation is {io_pct:.1f}% (target: ~25%).")

    report = {
        "target_bytes": target_bytes,
        "target_gb": target_bytes / 1e9,
        "selected_bytes": total_selected_bytes,
        "selected_gb_decimal": round(total_selected_bytes / 1e9, 4),
        "selected_gib": round(total_selected_bytes / (1024**3), 4),
        "utilization_pct": round(total_selected_bytes / target_bytes * 100, 2) if target_bytes > 0 else 0,
        "total_storms": total_storms,
        "total_files": total_files,
        "total_overpass_files": total_overpasses,
        "total_environmental_files": total_env,
        "distinct_seasons": distinct_seasons,
        "distinct_seasons_count": len(distinct_seasons),
        "distinct_eras": distinct_eras,
        "distinct_eras_count": len(distinct_eras),
        "distinct_basins": distinct_basins,
        "distinct_basins_count": len(distinct_basins),
        "basin_stats": {k: dict(v) for k, v in sorted(basin_stats.items())},
        "era_stats": {k: dict(v) for k, v in sorted(era_stats.items())},
        "era_basin_matrix": matrix,
        "year_bytes": dict(sorted(year_bytes.items())),
        "instruments": dict(sorted(instruments.items())),
        "platforms": dict(sorted(platforms.items())),
        "nio_pct": round(io_pct, 2),
        "nio_achieved": abs(io_pct - 25) <= 10,
        "largest_single_year": max_year,
        "largest_single_year_pct": round(max_year_pct, 2),
        "limitations": limitations,
        "available_basins_in_catalogue": sorted(available_basins),
        "available_eras_in_catalogue": sorted(available_eras),
        "total_catalogue_storms": len([]),  # filled by caller
    }

    return report


# ===========================================================================
# Manifest Generation
# ===========================================================================

def build_manifest(selected_storms, inventory, destination_base="data/raw/pilot"):
    """
    Build the pilot manifest with one row per NetCDF file from selected storms.
    """
    selected_folders = set(s["storm_folder"] for s in selected_storms)

    manifest = []
    for row in inventory:
        if row["storm_folder"] not in selected_folders:
            continue

        # Build local path preserving hierarchy
        # v01r01/final/{season}/{basin}/{cyclone_number}/filename.nc
        # -> data/raw/pilot/{season}/{basin}/{cyclone_number}/filename.nc
        rel = row["s3_key"]
        if rel.startswith(PREFIX):
            rel = rel[len(PREFIX):]
        local_dest = os.path.join(destination_base, rel.replace("/", os.sep))

        storm_id = storm_id_from_parts(row["season"], row["basin"], row["cyclone_number"])

        manifest.append({
            "s3_key": row["s3_key"],
            "size_bytes": row["size_bytes"],
            "season": row["season"],
            "basin": row["basin"],
            "cyclone_number": row["cyclone_number"],
            "storm_id": storm_id,
            "instrument": row["instrument"],
            "platform": row["platform"],
            "is_environmental": row["is_environmental"],
            "local_destination": local_dest,
            "storm_folder": row["storm_folder"],
        })

    return manifest


def save_manifest_csv(manifest, path):
    """Save the pilot manifest to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "s3_key", "size_bytes", "season", "basin", "cyclone_number",
        "storm_id", "instrument", "platform", "is_environmental",
        "local_destination", "storm_folder",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[OK] Saved pilot manifest -> {path} ({len(manifest)} files)")


def load_manifest_csv(path):
    """Load the pilot manifest from CSV."""
    manifest = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["size_bytes"] = int(row["size_bytes"])
            row["season"] = int(row["season"])
            row["is_environmental"] = row["is_environmental"].lower() in ("true", "1", "yes")
            manifest.append(row)
    return manifest


# ===========================================================================
# Summary Reports
# ===========================================================================

def generate_plan_summary_text(report):
    """Generate the human-readable plan summary."""
    lines = []
    w = 90
    lines.append("=" * w)
    lines.append("TC PRIMED PILOT DATASET PLAN SUMMARY")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("=" * w)
    lines.append("")

    lines.append("1. SIZE SUMMARY")
    lines.append("-" * w)
    lines.append(f"  Target Size:       {report['target_gb']:.2f} GB ({report['target_bytes']:,} bytes)")
    lines.append(f"  Selected Size:     {report['selected_gb_decimal']:.4f} GB = {report['selected_gib']:.4f} GiB ({report['selected_bytes']:,} bytes)")
    lines.append(f"  Utilization:       {report['utilization_pct']:.2f}%")
    lines.append("")

    lines.append("2. CONTENT COUNTS")
    lines.append("-" * w)
    lines.append(f"  Complete storms:         {report['total_storms']}")
    lines.append(f"  Total NetCDF files:      {report['total_files']}")
    lines.append(f"  Overpass files:          {report['total_overpass_files']}")
    lines.append(f"  Environmental files:     {report['total_environmental_files']}")
    lines.append(f"  Distinct seasons:        {report['distinct_seasons_count']}")
    lines.append(f"  Historical periods:      {report['distinct_eras_count']}")
    lines.append(f"  Basins represented:      {report['distinct_basins_count']}")
    lines.append("")

    lines.append("3. STORMS BY BASIN")
    lines.append("-" * w)
    for basin, stats in sorted(report["basin_stats"].items()):
        pct = (stats["bytes"] / report["selected_bytes"] * 100) if report["selected_bytes"] > 0 else 0
        basin_name = BASIN_NAMES.get(basin, basin)
        lines.append(f"  {basin} ({basin_name:30s}): {stats['storms']:4d} storms, "
                      f"{stats['bytes'] / 1e9:8.3f} GB ({pct:5.1f}%), {stats['files']} files")
    lines.append("")

    lines.append("4. STORMS BY HISTORICAL ERA")
    lines.append("-" * w)
    for era, stats in sorted(report["era_stats"].items()):
        pct = (stats["bytes"] / report["selected_bytes"] * 100) if report["selected_bytes"] > 0 else 0
        lines.append(f"  {era:12s}: {stats['storms']:4d} storms, "
                      f"{stats['bytes'] / 1e9:8.3f} GB ({pct:5.1f}%), {stats['files']} files")
    lines.append("")

    lines.append("5. ERA x BASIN MATRIX (Storm Count / GB)")
    lines.append("-" * w)
    all_basins_sel = sorted(set(s for era_data in report["era_basin_matrix"].values() for s in era_data))
    header = f"  {'Era':12s}"
    for b in all_basins_sel:
        header += f" | {b:>12s}"
    header += " | Total"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for era in sorted(report["era_basin_matrix"].keys()):
        row_str = f"  {era:12s}"
        era_total_storms = 0
        era_total_bytes = 0
        for b in all_basins_sel:
            cell = report["era_basin_matrix"][era].get(b, {"storms": 0, "bytes": 0})
            row_str += f" | {cell['storms']:3d}/{cell['bytes']/1e9:6.2f}GB"
            era_total_storms += cell["storms"]
            era_total_bytes += cell["bytes"]
        row_str += f" | {era_total_storms:3d}/{era_total_bytes/1e9:6.2f}GB"
        lines.append(row_str)
    lines.append("")

    lines.append("6. INSTRUMENT COUNTS (storms containing instrument)")
    lines.append("-" * w)
    for inst, count in sorted(report["instruments"].items(), key=lambda x: -x[1]):
        lines.append(f"  {inst:20s}: {count:4d} storms")
    lines.append("")

    lines.append("7. PLATFORM COUNTS (storms containing platform)")
    lines.append("-" * w)
    for plat, count in sorted(report["platforms"].items(), key=lambda x: -x[1]):
        lines.append(f"  {plat:20s}: {count:4d} storms")
    lines.append("")

    lines.append("8. NIO ALLOCATION")
    lines.append("-" * w)
    lines.append(f"  NIO (IO) percentage of selected bytes: {report['nio_pct']:.2f}%")
    lines.append(f"  NIO allocation achieved (within 10% of 25%): {'YES' if report['nio_achieved'] else 'NO'}")
    lines.append("")

    lines.append("9. YEAR CONCENTRATION")
    lines.append("-" * w)
    lines.append(f"  Largest single-year contribution: {report['largest_single_year']} "
                  f"({report['largest_single_year_pct']:.2f}%)")
    lines.append("")

    if report["limitations"]:
        lines.append("10. LIMITATIONS AND DEVIATIONS")
        lines.append("-" * w)
        for lim in report["limitations"]:
            lines.append(f"  ! {lim}")
        lines.append("")

    lines.append("11. VALIDATION CHECKLIST")
    lines.append("-" * w)
    lines.append(f"  [{'✓' if report['distinct_seasons_count'] > 1 else 'X'}] Multiple distinct seasons: {report['distinct_seasons_count']}")
    lines.append(f"  [{'✓' if report['distinct_eras_count'] > 1 else 'X'}] Multiple historical periods: {report['distinct_eras_count']}")
    lines.append(f"  [{'✓' if report['distinct_basins_count'] > 1 else 'X'}] Multiple basins: {report['distinct_basins_count']}")
    lines.append(f"  [{'✓' if report['selected_bytes'] <= report['target_bytes'] else 'X'}] Within target budget")
    lines.append(f"  [{'✓' if report['selected_bytes'] >= 18_000_000_000 else '!'} ] At least 18 GB selected")
    lines.append(f"  [{'✓' if report['nio_achieved'] else '!'}] NIO allocation ~25%")
    lines.append(f"  [{'✓' if report['largest_single_year_pct'] <= 10 else '!'}] No single year > 10%")
    lines.append("")

    lines.append("=" * w)
    lines.append("END OF PLAN SUMMARY")
    lines.append("=" * w)
    return "\n".join(lines)


def validate_plan_diversity(report, available_basins, available_eras):
    """
    Validate diversity requirements before allowing download.
    Returns (is_valid, issues).
    """
    issues = []
    multi_basin_available = len(available_basins) > 1
    multi_era_available = len(available_eras) > 1

    if report["distinct_basins_count"] <= 1 and multi_basin_available:
        issues.append("Plan contains only one basin when multiple are available.")
    if report["distinct_eras_count"] <= 1 and multi_era_available:
        issues.append("Plan contains only one historical period when multiple are available.")
    if report["selected_bytes"] > report["target_bytes"]:
        issues.append(f"Selection exceeds target: {report['selected_bytes']} > {report['target_bytes']}")

    return len(issues) == 0, issues


# ===========================================================================
# Download Operations
# ===========================================================================

def check_disk_space(destination, required_bytes):
    """Check available disk space. Returns (has_space, free_bytes)."""
    dest_path = Path(destination)
    # Create base dir if needed to check space on the correct drive
    dest_path.mkdir(parents=True, exist_ok=True)

    total, used, free = shutil.disk_usage(str(dest_path))
    needed = required_bytes + SAFETY_BUFFER_BYTES
    return free >= needed, free


def download_single_file(s3_key, local_dest, expected_size, retries=MAX_RETRIES):
    """
    Download a single S3 object using AWS CLI via sys.executable -m awscli.
    Returns (success, actual_size, error_msg).
    """
    # Check if file exists and has correct size
    if os.path.exists(local_dest):
        actual_size = os.path.getsize(local_dest)
        if actual_size == expected_size:
            return True, actual_size, "skipped (already exists with correct size)"

    # Ensure parent directory exists
    dest_dir = os.path.dirname(os.path.abspath(local_dest))
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as e:
        return False, 0, f"Failed to create directory '{dest_dir}': {e}"

    s3_path = f"s3://{BUCKET}/{s3_key}"
    cmd = [
        sys.executable,
        "-m",
        "awscli",
        "s3",
        "cp",
        s3_path,
        local_dest,
        "--no-sign-request",
    ]

    cwd = os.getcwd()
    last_error = "Unknown error"

    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per file
                cwd=cwd,
            )

            if result.returncode == 0:
                if os.path.exists(local_dest):
                    actual_size = os.path.getsize(local_dest)
                    if actual_size == expected_size:
                        return True, actual_size, "downloaded"
                    else:
                        last_error = f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes"
                        # Clean up wrong-sized file before retry
                        try:
                            os.remove(local_dest)
                        except OSError:
                            pass
                        if attempt < retries:
                            time.sleep(RETRY_DELAY_SEC)
                            continue
                        return False, actual_size, last_error
                else:
                    last_error = f"AWS CLI exited with 0 but local file missing at '{local_dest}'"
                    if attempt < retries:
                        time.sleep(RETRY_DELAY_SEC)
                        continue
                    return False, 0, last_error
            else:
                stderr_clean = result.stderr.strip() if result.stderr else ""
                stdout_clean = result.stdout.strip() if result.stdout else ""
                last_error = (
                    f"AWS CLI failed (rc={result.returncode}): "
                    f"stderr='{stderr_clean}', stdout='{stdout_clean}'"
                )
                if attempt < retries:
                    time.sleep(RETRY_DELAY_SEC)
                    continue
                return False, 0, last_error

        except FileNotFoundError as e:
            # Explicitly catch WinError 2 / missing executable with complete context
            diagnostic_msg = (
                f"FileNotFoundError [WinError 2]: Executable '{cmd[0]}' not found. "
                f"cwd='{cwd}', destination='{local_dest}', s3_key='{s3_key}'. "
                f"Python sys.executable='{sys.executable}'. Details: {e}"
            )
            return False, 0, diagnostic_msg

        except subprocess.TimeoutExpired:
            last_error = f"Download timed out after 600s (attempt {attempt}/{retries})"
            if attempt < retries:
                time.sleep(RETRY_DELAY_SEC)
                continue
            return False, 0, last_error

        except Exception as e:
            last_error = f"Unexpected download exception ({type(e).__name__}): {e}"
            if attempt < retries:
                time.sleep(RETRY_DELAY_SEC)
                continue
            return False, 0, last_error

    return False, 0, f"Max retries ({retries}) exceeded: {last_error}"


def run_download(manifest, destination, workers=DEFAULT_WORKERS):
    """
    Download all files in the manifest using parallel workers.
    Returns download report dict.
    """
    total_expected_bytes = sum(row["size_bytes"] for row in manifest)
    total_files = len(manifest)

    print(f"\n[DOWNLOAD] Starting download of {total_files} files "
          f"({total_expected_bytes / 1e9:.3f} GB) with {workers} workers")

    # Check disk space
    has_space, free_bytes = check_disk_space(destination, total_expected_bytes)
    if not has_space:
        print(f"[ERROR] Insufficient disk space!", file=sys.stderr)
        print(f"  Required:  {(total_expected_bytes + SAFETY_BUFFER_BYTES) / 1e9:.2f} GB "
              f"(data + {SAFETY_BUFFER_BYTES / 1e9:.0f} GB safety buffer)", file=sys.stderr)
        print(f"  Available: {free_bytes / 1e9:.2f} GB", file=sys.stderr)
        sys.exit(1)

    print(f"[DOWNLOAD] Disk space check passed: {free_bytes / 1e9:.2f} GB free")

    completed = 0
    downloaded_bytes = 0
    skipped = 0
    skipped_bytes = 0
    failed = []
    succeeded = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {}
        for row in manifest:
            future = executor.submit(
                download_single_file,
                row["s3_key"],
                row["local_destination"],
                row["size_bytes"],
            )
            future_to_row[future] = row

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                success, actual_size, msg = future.result()
                completed += 1

                if success:
                    if "skipped" in msg:
                        skipped += 1
                        skipped_bytes += actual_size
                    else:
                        downloaded_bytes += actual_size
                    succeeded.append(row)
                else:
                    failed.append({**row, "error": msg})

                # Progress update every 10 files or on failures
                if completed % 10 == 0 or not success:
                    elapsed = time.time() - start_time
                    pct = completed / total_files * 100
                    status = "OK" if success else f"FAILED: {msg}"
                    print(f"  [{completed}/{total_files} {pct:.0f}%] "
                          f"{os.path.basename(row['s3_key'])} -> {status} "
                          f"({elapsed:.0f}s elapsed)")

            except Exception as e:
                completed += 1
                failed.append({**row, "error": str(e)})

    elapsed_total = time.time() - start_time

    # Storm counts
    selected_storms = set(row["storm_id"] for row in manifest)
    downloaded_storms = set(row["storm_id"] for row in succeeded)

    # Instrument counts in downloaded
    instruments = defaultdict(int)
    for row in succeeded:
        if row.get("instrument"):
            instruments[row["instrument"]] += 1

    report = {
        "download_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "expected_bytes": total_expected_bytes,
        "expected_gb": round(total_expected_bytes / 1e9, 4),
        "downloaded_bytes": downloaded_bytes,
        "downloaded_gb": round(downloaded_bytes / 1e9, 4),
        "skipped_bytes": skipped_bytes,
        "skipped_gb": round(skipped_bytes / 1e9, 4),
        "total_files": total_files,
        "completed_files": len(succeeded),
        "skipped_files": skipped,
        "failed_files": len(failed),
        "failed_details": [{"s3_key": f["s3_key"], "error": f["error"]} for f in failed],
        "selected_storm_count": len(selected_storms),
        "downloaded_storm_count": len(downloaded_storms),
        "elapsed_seconds": round(elapsed_total, 2),
        "instrument_counts": dict(sorted(instruments.items())),
    }

    return report


def generate_download_report_text(report):
    """Generate human-readable download report."""
    lines = []
    w = 80
    lines.append("=" * w)
    lines.append("TC PRIMED PILOT DOWNLOAD REPORT")
    lines.append(f"Completed: {report['download_timestamp_utc']}")
    lines.append("=" * w)
    lines.append("")
    lines.append(f"  Expected:    {report['expected_gb']:.4f} GB ({report['expected_bytes']:,} bytes)")
    lines.append(f"  Downloaded:  {report['downloaded_gb']:.4f} GB ({report['downloaded_bytes']:,} bytes)")
    lines.append(f"  Skipped:     {report['skipped_gb']:.4f} GB ({report['skipped_bytes']:,} bytes)")
    lines.append(f"  Total files: {report['total_files']}")
    lines.append(f"  Completed:   {report['completed_files']}")
    lines.append(f"  Skipped:     {report['skipped_files']}")
    lines.append(f"  Failed:      {report['failed_files']}")
    lines.append(f"  Storms:      {report['downloaded_storm_count']} / {report['selected_storm_count']}")
    lines.append(f"  Elapsed:     {report['elapsed_seconds']:.1f} seconds")
    lines.append("")

    if report["instrument_counts"]:
        lines.append("  Instrument Counts:")
        for inst, count in report["instrument_counts"].items():
            lines.append(f"    {inst:20s}: {count}")
        lines.append("")

    if report["failed_details"]:
        lines.append("  FAILED DOWNLOADS:")
        for f in report["failed_details"]:
            lines.append(f"    {f['s3_key']}")
            lines.append(f"      Error: {f['error']}")
        lines.append("")

    lines.append("=" * w)
    lines.append("END OF DOWNLOAD REPORT")
    lines.append("=" * w)
    return "\n".join(lines)


def audit_local_downloads(manifest, destination_base="data/raw/pilot"):
    """
    Compare manifest entries against local files on disk.
    Classifies entries as:
      - correctly downloaded (file exists and size matches expected size)
      - incomplete or incorrect size (file exists but size != expected)
      - missing (file does not exist)
      - unexpected local file (file exists in destination tree but not in manifest)
    """
    manifest_by_normpath = {}
    for row in manifest:
        dest = row["local_destination"]
        if destination_base:
            rel = row["s3_key"]
            if rel.startswith(PREFIX):
                rel = rel[len(PREFIX):]
            dest = os.path.join(destination_base, rel.replace("/", os.sep))
        norm = os.path.normcase(os.path.abspath(dest))
        manifest_by_normpath[norm] = {**row, "resolved_path": dest}

    correct = []
    incomplete = []
    missing = []
    unexpected = []

    for norm, item in manifest_by_normpath.items():
        dest = item["resolved_path"]
        expected_size = int(item["size_bytes"])
        if not os.path.exists(dest):
            missing.append({
                "s3_key": item["s3_key"],
                "local_destination": dest,
                "expected_size": expected_size,
                "storm_id": item.get("storm_id", ""),
                "is_environmental": item.get("is_environmental", False),
            })
        else:
            try:
                actual_size = os.path.getsize(dest)
            except OSError:
                actual_size = -1

            if actual_size == expected_size:
                correct.append({
                    "s3_key": item["s3_key"],
                    "local_destination": dest,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "storm_id": item.get("storm_id", ""),
                    "is_environmental": item.get("is_environmental", False),
                })
            else:
                incomplete.append({
                    "s3_key": item["s3_key"],
                    "local_destination": dest,
                    "expected_size": expected_size,
                    "actual_size": actual_size,
                    "storm_id": item.get("storm_id", ""),
                    "is_environmental": item.get("is_environmental", False),
                })

    # Search destination_base for unexpected files
    dest_abs = os.path.abspath(destination_base)
    if os.path.exists(dest_abs):
        for root, _, files in os.walk(dest_abs):
            for f in files:
                full_path = os.path.join(root, f)
                norm = os.path.normcase(os.path.abspath(full_path))
                if norm not in manifest_by_normpath:
                    try:
                        sz = os.path.getsize(full_path)
                    except OSError:
                        sz = -1
                    unexpected.append({
                        "local_path": full_path,
                        "size_bytes": sz,
                    })

    expected_total_bytes = sum(int(r["size_bytes"]) for r in manifest)
    correct_bytes = sum(c["actual_size"] for c in correct)
    missing_bytes = sum(m["expected_size"] for m in missing)
    incomplete_expected_bytes = sum(inc["expected_size"] for inc in incomplete)
    incomplete_actual_bytes = sum(inc["actual_size"] for inc in incomplete if inc["actual_size"] > 0)
    unexpected_bytes = sum(u["size_bytes"] for u in unexpected if u["size_bytes"] > 0)

    audit_report = {
        "audit_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "destination_root": os.path.abspath(destination_base),
        "total_manifest_entries": len(manifest),
        "expected_total_bytes": expected_total_bytes,
        "expected_total_gb": round(expected_total_bytes / 1e9, 4),
        "expected_total_gib": round(expected_total_bytes / (1024**3), 4),
        "correct_count": len(correct),
        "correct_bytes": correct_bytes,
        "correct_gb": round(correct_bytes / 1e9, 4),
        "correct_gib": round(correct_bytes / (1024**3), 4),
        "missing_count": len(missing),
        "missing_bytes": missing_bytes,
        "missing_gb": round(missing_bytes / 1e9, 4),
        "missing_gib": round(missing_bytes / (1024**3), 4),
        "incomplete_count": len(incomplete),
        "incomplete_expected_bytes": incomplete_expected_bytes,
        "incomplete_actual_bytes": incomplete_actual_bytes,
        "unexpected_count": len(unexpected),
        "unexpected_bytes": unexpected_bytes,
        "unexpected_files": unexpected,
        "correct_sample": correct[:5],
        "missing_sample": missing[:5],
        "incomplete_sample": incomplete[:5],
    }
    return audit_report


def generate_audit_report_text(report):
    """Generate human-readable local file audit text report."""
    lines = []
    w = 88
    lines.append("=" * w)
    lines.append("TC PRIMED PILOT DATASET: LOCAL FILE AUDIT REPORT")
    lines.append(f"Audit Timestamp: {report['audit_timestamp_utc']}")
    lines.append(f"Destination:     {report['destination_root']}")
    lines.append("=" * w)
    lines.append("")
    lines.append("1. AUDIT CLASSIFICATION SUMMARY")
    lines.append("-" * w)
    lines.append(f"  Total Manifest Entries:         {report['total_manifest_entries']:,}")
    lines.append(f"  Expected Manifest Size:         {report['expected_total_gb']:.4f} GB ({report['expected_total_gib']:.4f} GiB / {report['expected_total_bytes']:,} bytes)")
    lines.append("")
    lines.append(f"  [CORRECT]    Fully Downloaded:  {report['correct_count']:,} files ({report['correct_gb']:.4f} GB / {report['correct_gib']:.4f} GiB)")
    lines.append(f"  [MISSING]    Missing on Disk:   {report['missing_count']:,} files ({report['missing_gb']:.4f} GB / {report['missing_gib']:.4f} GiB)")
    lines.append(f"  [INCOMPLETE] Wrong Size:        {report['incomplete_count']:,} files ({report['incomplete_expected_bytes']:,} bytes expected)")
    lines.append(f"  [UNEXPECTED] Extraneous Files:  {report['unexpected_count']:,} files ({report['unexpected_bytes'] / 1e9:.4f} GB / {report['unexpected_bytes']:,} bytes)")
    lines.append("")

    lines.append("2. DOWNLOAD PROGRESS")
    lines.append("-" * w)
    pct_files = (report['correct_count'] / report['total_manifest_entries'] * 100) if report['total_manifest_entries'] > 0 else 0
    pct_bytes = (report['correct_bytes'] / report['expected_total_bytes'] * 100) if report['expected_total_bytes'] > 0 else 0
    lines.append(f"  Completed Files:  {pct_files:6.2f}% ({report['correct_count']:,} of {report['total_manifest_entries']:,})")
    lines.append(f"  Completed Bytes:  {pct_bytes:6.2f}% ({report['correct_gb']:.4f} of {report['expected_total_gb']:.4f} GB)")
    lines.append("")

    if report["unexpected_files"]:
        lines.append("3. UNEXPECTED LOCAL FILES (INCOMPLETE / TEMPORARY AWS CLI ARTIFACTS)")
        lines.append("-" * w)
        for u in report["unexpected_files"]:
            lines.append(f"  * {u['local_path']} ({u['size_bytes']:,} bytes)")
        lines.append("")

    lines.append("=" * w)
    lines.append("END OF LOCAL FILE AUDIT REPORT")
    lines.append("=" * w)
    return "\n".join(lines)


def cmd_audit(args):
    """Execute the 'audit' command: audit local files against the manifest."""
    manifest_path = args.manifest
    destination = args.destination
    reports_dir = args.reports_dir

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest_csv(manifest_path)
    if not manifest:
        print("[ERROR] Manifest is empty.", file=sys.stderr)
        return 1

    audit_report = audit_local_downloads(manifest, destination_base=destination)

    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "pilot_local_file_audit.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, default=str)
    print(f"[OK] Saved local file audit JSON -> {json_path}")

    txt_path = os.path.join(reports_dir, "pilot_local_file_audit.txt")
    txt_report = generate_audit_report_text(audit_report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    print(f"[OK] Saved local file audit text -> {txt_path}")

    print("\n" + txt_report)
    return 0


# ===========================================================================
# CLI Entry Points
# ===========================================================================

def cmd_plan(args):
    """Execute the 'plan' command: inventory + stratified selection + manifest."""
    target_bytes = int(args.target_gb * 1e9)
    seed = args.seed
    reports_dir = args.reports_dir

    os.makedirs(reports_dir, exist_ok=True)

    # Step 1: S3 inventory
    print("\n" + "=" * 70)
    print("PHASE 1: S3 INVENTORY")
    print("=" * 70)
    listing_lines = run_s3_listing()
    inventory = build_inventory(listing_lines)

    inventory_path = os.path.join(reports_dir, "s3_inventory.csv")
    save_inventory_csv(inventory, inventory_path)

    # Step 2: Storm catalogue
    print("\n" + "=" * 70)
    print("PHASE 2: STORM CATALOGUE")
    print("=" * 70)
    catalogue = build_storm_catalogue(inventory)
    catalogue_path = os.path.join(reports_dir, "storm_catalogue.csv")
    save_storm_catalogue_csv(catalogue, catalogue_path)

    total_nc_bytes = sum(row["size_bytes"] for row in inventory)
    print(f"[CATALOGUE] {len(catalogue)} storm folders, "
          f"{len(inventory)} NetCDF files, "
          f"{total_nc_bytes / 1e9:.2f} GB total")

    # Step 3: Stratified selection
    print("\n" + "=" * 70)
    print("PHASE 3: STRATIFIED PILOT SELECTION")
    print("=" * 70)
    selected_storms, report = stratified_select(catalogue, target_bytes, seed)
    report["total_catalogue_storms"] = len(catalogue)

    available_basins = sorted(set(s["basin"] for s in catalogue))
    available_eras = sorted(set(s["era"] for s in catalogue))

    # Validate diversity
    is_valid, issues = validate_plan_diversity(report, available_basins, available_eras)
    if not is_valid:
        print("[WARNING] Plan diversity validation issues:")
        for issue in issues:
            print(f"  ! {issue}")
        report["limitations"].extend(issues)

    # Step 4: Build manifest
    print("\n" + "=" * 70)
    print("PHASE 4: MANIFEST GENERATION")
    print("=" * 70)
    manifest = build_manifest(selected_storms, inventory)
    manifest_path = os.path.join(reports_dir, "pilot_manifest.csv")
    save_manifest_csv(manifest, manifest_path)

    # Step 5: Save summaries
    summary_text = generate_plan_summary_text(report)

    txt_path = os.path.join(reports_dir, "pilot_plan_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[OK] Saved plan summary text -> {txt_path}")

    json_path = os.path.join(reports_dir, "pilot_plan_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] Saved plan summary JSON -> {json_path}")

    # Print summary to stdout
    print("\n" + summary_text)

    # Final validation display
    print("\n[VALIDATION] Pre-download checks:")
    print(f"  Distinct seasons:        {report['distinct_seasons_count']}")
    print(f"  Historical periods:      {report['distinct_eras_count']}")
    print(f"  Basins:                  {report['distinct_basins_count']}")
    print(f"  Complete storms:         {report['total_storms']}")
    print(f"  Selected size:           {report['selected_gb_decimal']:.4f} GB")
    for era in sorted(report['era_stats']):
        pct = report['era_stats'][era]['bytes'] / report['selected_bytes'] * 100 if report['selected_bytes'] > 0 else 0
        print(f"  Era {era:12s}: {pct:5.1f}%")
    for basin in sorted(report['basin_stats']):
        pct = report['basin_stats'][basin]['bytes'] / report['selected_bytes'] * 100 if report['selected_bytes'] > 0 else 0
        print(f"  Basin {basin:5s}: {pct:5.1f}%")
    print(f"  Max single-year:         {report['largest_single_year']} ({report['largest_single_year_pct']:.1f}%)")

    return 0


def cmd_download(args):
    """Execute the 'download' command: download manifest entries."""
    manifest_path = args.manifest
    destination = args.destination
    workers = args.workers

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest_csv(manifest_path)
    if not manifest:
        print("[ERROR] Manifest is empty.", file=sys.stderr)
        return 1

    # Update local destinations if custom destination provided
    if destination:
        for row in manifest:
            # Reconstruct from s3_key
            rel = row["s3_key"]
            if rel.startswith(PREFIX):
                rel = rel[len(PREFIX):]
            row["local_destination"] = os.path.join(destination, rel.replace("/", os.sep))

    print(f"[DOWNLOAD] Loaded manifest with {len(manifest)} files")

    # Validate plan diversity before downloading
    basins = set(row["basin"] for row in manifest)
    seasons = set(row["season"] for row in manifest)
    eras = set(classify_era(int(s)) for s in seasons)

    # Check if there's only one era or basin available - warn but proceed if data limited
    if len(basins) <= 1 and len(set(row["basin"] for row in manifest)) > 1:
        print("[ERROR] Plan contains only one basin when others are available. Aborting.", file=sys.stderr)
        return 1
    if len(eras) <= 1 and len(set(classify_era(int(row["season"])) for row in manifest)) > 1:
        print("[ERROR] Plan contains only one historical period when others are available. Aborting.", file=sys.stderr)
        return 1

    dest = destination or os.path.dirname(manifest[0]["local_destination"])
    report = run_download(manifest, dest, workers)

    # Save reports
    reports_dir = os.path.dirname(manifest_path) or "reports"
    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "pilot_download_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[OK] Saved download report JSON -> {json_path}")

    txt_path = os.path.join(reports_dir, "pilot_download_report.txt")
    report_text = generate_download_report_text(report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Saved download report text -> {txt_path}")

    print("\n" + report_text)

    return 0 if report["failed_files"] == 0 else 1


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TC PRIMED Stratified Pilot Dataset Planner & Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/prepare_pilot_dataset.py plan --target-gb 20 --seed 42
  python src/prepare_pilot_dataset.py download --manifest reports/pilot_manifest.csv --destination data/raw/pilot --workers 4
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Plan command
    plan_parser = subparsers.add_parser("plan", help="Discover S3 files and produce a balanced pilot manifest")
    plan_parser.add_argument("--target-gb", type=float, default=20.0,
                             help="Target size in decimal GB (default: 20)")
    plan_parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                             help="Random seed for deterministic selection (default: 42)")
    plan_parser.add_argument("--reports-dir", default="reports",
                             help="Directory for output reports (default: reports)")

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download files from an approved manifest")
    dl_parser.add_argument("--manifest", required=True,
                           help="Path to pilot_manifest.csv")
    dl_parser.add_argument("--destination", default=None,
                           help="Local destination base directory (default: from manifest)")
    dl_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                           help="Number of concurrent downloads (default: 4)")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit local files against manifest")
    audit_parser.add_argument("--manifest", default="reports/pilot_manifest.csv",
                              help="Path to pilot_manifest.csv (default: reports/pilot_manifest.csv)")
    audit_parser.add_argument("--destination", default="data/raw/pilot",
                              help="Local destination base directory (default: data/raw/pilot)")
    audit_parser.add_argument("--reports-dir", default="reports",
                              help="Directory for output reports (default: reports)")

    args = parser.parse_args()

    if args.command == "plan":
        sys.exit(cmd_plan(args))
    elif args.command == "download":
        sys.exit(cmd_download(args))
    elif args.command == "audit":
        sys.exit(cmd_audit(args))


if __name__ == "__main__":
    main()

