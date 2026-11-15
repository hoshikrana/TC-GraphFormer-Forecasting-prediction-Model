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
import hashlib
import json
import os
import re
import subprocess
import sys
import shutil
import time
import random
import yaml
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

# Supported TC PRIMED radar instruments
RADAR_INSTRUMENTS = {"DPR", "PR", "KUKAGMI", "KUGMI", "KUTMI"}

# Known TC PRIMED instruments (for parsing filenames)
KNOWN_INSTRUMENTS = [
    "GMI", "TMI", "SSMI", "SSMIS", "AMSR2", "AMSRE", "AMSUB", "AMSU-B",
    "AMSUA", "AMSU-A", "MHS", "ATMS", "SAPHIR", "MWRI", "MWHS2",
    "TEMPEST-D", "TEMPEST", "COWVR", "DPR", "PR", "KuKaGMI", "KuGMI", "KuTMI",
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


def load_inventory_csv(path):
    """Load S3 inventory from CSV."""
    inventory = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inventory.append({
                "s3_key": row["s3_key"],
                "size_bytes": int(row["size_bytes"]),
                "season": int(row["season"]),
                "basin": row["basin"],
                "cyclone_number": row["cyclone_number"],
                "storm_folder": row["storm_folder"],
                "filename": row["filename"],
                "instrument": row.get("instrument") or None,
                "platform": row.get("platform") or None,
                "is_environmental": row["is_environmental"].lower() in ("true", "1", "yes") if isinstance(row["is_environmental"], str) else bool(row["is_environmental"]),
            })
    return inventory


def load_storm_catalogue_csv(path):
    """Load storm catalogue from CSV."""
    catalogue = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            catalogue.append({
                "storm_folder": row["storm_folder"],
                "storm_id": row["storm_id"],
                "season": int(row["season"]),
                "basin": row["basin"],
                "cyclone_number": row["cyclone_number"],
                "era": row.get("era") or classify_era(int(row["season"])),
                "total_bytes": int(row["total_bytes"]),
                "file_count": int(row["file_count"]),
                "overpass_count": int(row["overpass_count"]),
                "env_count": int(row["env_count"]),
                "instruments": row.get("instruments", ""),
                "platforms": row.get("platforms", ""),
            })
    return catalogue


def deterministic_tie_breaker(seed: int, storm_folder: str) -> float:
    """
    Deterministic cross-process tie-breaker using SHA-256.
    Ensures identical float scoring across separate Python processes regardless of PYTHONHASHSEED.
    """
    digest = hashlib.sha256(f"{seed}:{storm_folder}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 10000.0


def get_storm_sensors(storm) -> set:
    """
    Extract distinct remote-sensing instrument identifiers for a storm,
    excluding environmental/reanalysis models ('ERA5', 'reanalysis').
    """
    inst_str = storm.get("instruments", "")
    if not inst_str:
        return set()
    raw_insts = [i.strip() for i in inst_str.split(";") if i.strip()]
    return {i for i in raw_insts if i.upper() not in {"ERA5", "REANALYSIS", "UNKNOWN", "ENVIRONMENTAL"}}


def storm_has_radar(storm) -> bool:
    """Return True if storm possesses recognized radar instrument observations."""
    sensors = get_storm_sensors(storm)
    for s in sensors:
        if s.upper() in RADAR_INSTRUMENTS or "RADAR" in s.upper():
            return True
    return False


def filter_eligible_catalogue(catalogue, modality_requirements=None):
    """
    Filter candidate storms against explicit modality requirements before stratified selection.

    Enforces:
      - require_environmental: true -> env_count >= 1
      - min_sensors_per_storm: int -> distinct remote-sensing instrument count >= min_sensors_per_storm
      - prioritize_radar_overpasses: tracked for preference in stratified_select

    Returns:
      eligible_catalogue: List of qualifying complete storms
      modality_audit: Comprehensive audit statistics covering original/eligible counts,
                      exclusion reasons, sensor coverage, and radar availability.
    """
    reqs = modality_requirements or {}
    req_env = bool(reqs.get("require_environmental", False))
    min_sensors = int(reqs.get("min_sensors_per_storm", 0)) if reqs.get("min_sensors_per_storm") is not None else 0
    prioritize_radar = bool(reqs.get("prioritize_radar_overpasses", False))

    original_count = len(catalogue)
    original_bytes = sum(s["total_bytes"] for s in catalogue)

    eligible = []
    excluded_count = 0
    exclusion_reasons = defaultdict(int)
    excluded_storm_details = []

    storms_with_env = 0
    storms_with_radar = 0
    sensor_counts = []

    for storm in catalogue:
        sensors = get_storm_sensors(storm)
        n_sensors = len(sensors)
        has_env = (storm.get("env_count", 0) >= 1)
        has_radar = storm_has_radar(storm)

        if has_env:
            storms_with_env += 1
        if has_radar:
            storms_with_radar += 1
        sensor_counts.append(n_sensors)

        # Check eligibility
        reasons = []
        if req_env and not has_env:
            reasons.append("missing_environmental")
            exclusion_reasons["missing_environmental"] += 1

        if min_sensors > 0 and n_sensors < min_sensors:
            reasons.append(f"insufficient_sensors (found {n_sensors} < required {min_sensors})")
            exclusion_reasons["insufficient_sensors"] += 1

        if reasons:
            excluded_count += 1
            excluded_storm_details.append({
                "storm_folder": storm["storm_folder"],
                "reasons": reasons,
                "sensors": sorted(sensors),
                "env_count": storm.get("env_count", 0),
            })
        else:
            eligible.append(storm)

    eligible_bytes = sum(s["total_bytes"] for s in eligible)
    eligible_radar_count = sum(1 for s in eligible if storm_has_radar(s))

    modality_audit = {
        "modality_requirements_applied": {
            "require_environmental": req_env,
            "min_sensors_per_storm": min_sensors,
            "prioritize_radar_overpasses": prioritize_radar,
        },
        "original_catalogue_storms": original_count,
        "original_catalogue_bytes": original_bytes,
        "eligible_catalogue_storms": len(eligible),
        "eligible_catalogue_bytes": eligible_bytes,
        "excluded_storms_count": excluded_count,
        "exclusion_reasons": dict(exclusion_reasons),
        "excluded_storm_details": excluded_storm_details,
        "environmental_coverage_pct": round(100.0 * storms_with_env / max(1, original_count), 2),
        "multi_sensor_coverage_pct": round(100.0 * sum(1 for c in sensor_counts if c >= max(2, min_sensors)) / max(1, original_count), 2),
        "radar_storms_original_count": storms_with_radar,
        "radar_storms_eligible_count": eligible_radar_count,
        "original_radar_storms_count": storms_with_radar,
        "eligible_radar_storms_count": eligible_radar_count,
        "radar_priority_enabled": prioritize_radar,
    }

    return eligible, modality_audit


# ===========================================================================
# Stratified Pilot Selection
# ===========================================================================

def stratified_select(
    catalogue,
    target_bytes,
    seed,
    nio_fraction=0.25,
    basin_proportions=None,
    era_proportions=None,
    single_year_max_pct=20.0,
    enforce_complete_storms=True,
    enforce_single_year_cap=False,
    modality_requirements=None,
    modality_audit=None,
):
    """
    Perform stratified selection of complete storm folders.

    Strategy:
      1. Partition storms by (era, basin).
      2. If basin_proportions is specified, allocate budget proportionally across basins.
         Otherwise, reserve ~25% for IO basin and split remaining equally.
      3. If era_proportions is specified, balance eras within each basin proportionally.
         Otherwise, divide equally across available historical periods.
      4. Within each (era, basin) cell, randomly select complete storm folders.
      5. Never exceed target_bytes. Complete storms are strictly preserved.

    Returns (selected_storms, selection_report).
    """
    if modality_requirements and modality_audit is None:
        catalogue, modality_audit = filter_eligible_catalogue(catalogue, modality_requirements)

    prioritize_radar = bool(modality_requirements.get("prioritize_radar_overpasses", False)) if modality_requirements else False

    rng = random.Random(seed)

    # Validate proportions if supplied
    if basin_proportions:
        total_bp = sum(basin_proportions.values())
        if not (0.95 <= total_bp <= 1.05):
            raise ValueError(f"Target basin proportions must sum to ~1.0 (got {total_bp:.4f})")
    if era_proportions:
        total_ep = sum(era_proportions.values())
        if not (0.95 <= total_ep <= 1.05):
            raise ValueError(f"Target era proportions must sum to ~1.0 (got {total_ep:.4f})")

    # Group storms by (era, basin)
    cells = defaultdict(list)
    for storm in catalogue:
        era = storm["era"]
        basin = storm["basin"]
        cells[(era, basin)].append(storm)

    # Shuffle each cell deterministically, prioritizing radar overpasses if requested
    for key in cells:
        rng.shuffle(cells[key])
        if prioritize_radar:
            cells[key].sort(key=lambda s: (not storm_has_radar(s)))

    # Identify available basins and eras
    available_basins = sorted(set(s["basin"] for s in catalogue))
    available_eras = sorted(set(s["era"] for s in catalogue))

    # Compute per-basin budgets and normalized proportions
    basin_budgets = {}
    if basin_proportions:
        norm_b = {b: float(basin_proportions.get(b, 0.0)) for b in available_basins}
        sum_b = sum(norm_b.values())
        if sum_b > 0:
            norm_b = {b: norm_b[b] / sum_b for b in available_basins}
        else:
            norm_b = {b: 1.0 / max(1, len(available_basins)) for b in available_basins}
        basin_budgets = {b: int(target_bytes * norm_b[b]) for b in available_basins}
    else:
        norm_b = {}
        if "IO" in available_basins:
            norm_b["IO"] = float(nio_fraction) if nio_fraction is not None else 0.25
            other_basins = [b for b in available_basins if b != "IO"]
            rem_frac = max(0.0, 1.0 - norm_b["IO"])
            for b in other_basins:
                norm_b[b] = rem_frac / max(1, len(other_basins))
        else:
            for b in available_basins:
                norm_b[b] = 1.0 / max(1, len(available_basins))
        basin_budgets = {b: int(target_bytes * norm_b[b]) for b in available_basins}

    # Compute per-era budgets within each basin and normalized proportions
    if era_proportions:
        norm_e = {e: float(era_proportions.get(e, 0.0)) for e in available_eras}
        sum_e = sum(norm_e.values())
        if sum_e > 0:
            norm_e = {e: norm_e[e] / sum_e for e in available_eras}
        else:
            norm_e = {e: 1.0 / max(1, len(available_eras)) for e in available_eras}
    else:
        norm_e = {e: 1.0 / max(1, len(available_eras)) for e in available_eras}

    era_basin_budgets = {}
    for b, b_tot in basin_budgets.items():
        for e in available_eras:
            era_basin_budgets[(e, b)] = int(b_tot * norm_e[e])

    # Single-year concentration limit (in bytes)
    max_single_year_bytes = int(target_bytes * (single_year_max_pct / 100.0)) if single_year_max_pct > 0 else 0

    # Selection pass 1: iterate through cells and pick storms up to cell budget
    selected = []
    selected_bytes = 0
    selected_folders = set()
    cell_used_bytes = defaultdict(int)
    year_bytes = defaultdict(int)

    for (era, basin) in sorted(era_basin_budgets.keys()):
        cell_budget = era_basin_budgets[(era, basin)]
        cell_storms = cells.get((era, basin), [])

        for storm in cell_storms:
            sf = storm["storm_folder"]
            if sf in selected_folders:
                continue
            storm_bytes = storm["total_bytes"]
            s_year = storm.get("season", storm.get("year"))

            if cell_used_bytes[(era, basin)] + storm_bytes > cell_budget:
                continue

            if selected_bytes + storm_bytes > target_bytes:
                continue

            if max_single_year_bytes > 0 and (year_bytes[s_year] + storm_bytes > max_single_year_bytes):
                continue

            selected.append(storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes
            cell_used_bytes[(era, basin)] += storm_bytes
            year_bytes[s_year] += storm_bytes

    # Selection pass 2: iterative deficit-based filling (preserving configured era and basin proportions)
    remaining = target_bytes - selected_bytes
    if remaining > 0:
        unselected = [s for s in catalogue if s["storm_folder"] not in selected_folders]

        era_bytes = defaultdict(int)
        basin_bytes_used = defaultdict(int)
        for s in selected:
            era_bytes[s["era"]] += s["total_bytes"]
            basin_bytes_used[s["basin"]] += s["total_bytes"]

        while selected_bytes < target_bytes:
            best_storm = None
            best_score = -float("inf")

            for storm in unselected:
                sf = storm["storm_folder"]
                if sf in selected_folders:
                    continue
                storm_bytes = storm["total_bytes"]
                if selected_bytes + storm_bytes > target_bytes:
                    continue

                s_year = storm.get("season", storm.get("year"))
                year_violation = (max_single_year_bytes > 0 and (year_bytes[s_year] + storm_bytes > max_single_year_bytes))
                if enforce_single_year_cap and year_violation:
                    continue

                b_target = basin_budgets.get(storm["basin"], target_bytes * norm_b.get(storm["basin"], 0))
                b_def = b_target - basin_bytes_used.get(storm["basin"], 0)
                e_target = target_bytes * norm_e.get(storm["era"], 0)
                e_def = e_target - era_bytes.get(storm["era"], 0)
                score = b_def + e_def

                if prioritize_radar and storm_has_radar(storm):
                    score += 1000.0

                if year_violation:
                    score -= 1e12

                # Deterministic cross-process tie-breaker using SHA-256
                score += deterministic_tie_breaker(seed, sf)

                if score > best_score:
                    best_score = score
                    best_storm = storm

            if best_storm is None or best_score <= -1e11:
                break

            sf = best_storm["storm_folder"]
            storm_bytes = best_storm["total_bytes"]
            s_year = best_storm.get("season", best_storm.get("year"))

            selected.append(best_storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes
            era_bytes[best_storm["era"]] += storm_bytes
            basin_bytes_used[best_storm["basin"]] += storm_bytes
            year_bytes[s_year] += storm_bytes

    # Selection pass 3: greedy fill with any remaining storms sorted by deficit score
    remaining = target_bytes - selected_bytes
    if remaining > 0:
        leftover = [s for s in catalogue if s["storm_folder"] not in selected_folders]

        while selected_bytes < target_bytes:
            best_storm = None
            best_score = -float("inf")

            for storm in leftover:
                sf = storm["storm_folder"]
                if sf in selected_folders:
                    continue
                storm_bytes = storm["total_bytes"]
                if selected_bytes + storm_bytes > target_bytes:
                    continue

                s_year = storm.get("season", storm.get("year"))
                year_violation = (max_single_year_bytes > 0 and (year_bytes[s_year] + storm_bytes > max_single_year_bytes))
                if enforce_single_year_cap and year_violation:
                    continue

                b_target = basin_budgets.get(storm["basin"], target_bytes * norm_b.get(storm["basin"], 0))
                b_def = b_target - basin_bytes_used.get(storm["basin"], 0)
                e_target = target_bytes * norm_e.get(storm["era"], 0)
                e_def = e_target - era_bytes.get(storm["era"], 0)
                score = b_def + e_def

                if prioritize_radar and storm_has_radar(storm):
                    score += 1000.0

                if year_violation:
                    score -= 1e12

                score += deterministic_tie_breaker(seed, sf)

                if score > best_score:
                    best_score = score
                    best_storm = storm

            if best_storm is None and not enforce_single_year_cap:
                # If no candidate fits without year violation and hard cap is disabled, relax year violation to fill remaining budget
                for storm in leftover:
                    sf = storm["storm_folder"]
                    if sf in selected_folders:
                        continue
                    if selected_bytes + storm["total_bytes"] <= target_bytes:
                        best_storm = storm
                        break

            if best_storm is None or (enforce_single_year_cap and best_score <= -1e11):
                break

            sf = best_storm["storm_folder"]
            storm_bytes = best_storm["total_bytes"]
            s_year = best_storm.get("season", best_storm.get("year"))

            selected.append(best_storm)
            selected_folders.add(sf)
            selected_bytes += storm_bytes
            era_bytes[best_storm["era"]] += storm_bytes
            basin_bytes_used[best_storm["basin"]] += storm_bytes
            year_bytes[s_year] += storm_bytes

    # Build selection report
    report = build_selection_report(
        selected,
        catalogue,
        target_bytes,
        available_basins,
        available_eras,
        basin_proportions=basin_proportions,
        era_proportions=era_proportions,
        nio_fraction=nio_fraction,
        single_year_max_pct=single_year_max_pct,
        enforce_complete_storms=enforce_complete_storms,
        enforce_single_year_cap=enforce_single_year_cap,
        modality_audit=modality_audit,
    )

    return selected, report


def build_selection_report(
    selected,
    catalogue,
    target_bytes,
    available_basins,
    available_eras,
    basin_proportions=None,
    era_proportions=None,
    nio_fraction=0.25,
    single_year_max_pct=20.0,
    enforce_complete_storms=True,
    enforce_single_year_cap=False,
    modality_audit=None,
):
    """Build a detailed report about the selection."""

    total_selected_bytes = sum(s["total_bytes"] for s in selected)
    total_files = sum(s.get("file_count", s.get("total_files", 0)) for s in selected)
    total_overpasses = sum(s.get("overpass_count", 0) for s in selected)
    total_env = sum(s.get("env_count", 0) for s in selected)
    total_storms = len(selected)

    # Per-basin
    basin_stats = defaultdict(lambda: {"storms": 0, "bytes": 0, "files": 0})
    for s in selected:
        basin_stats[s["basin"]]["storms"] += 1
        basin_stats[s["basin"]]["bytes"] += s["total_bytes"]
        basin_stats[s["basin"]]["files"] += s.get("file_count", s.get("total_files", 0))

    # Per-era
    era_stats = defaultdict(lambda: {"storms": 0, "bytes": 0, "files": 0})
    for s in selected:
        era_stats[s["era"]]["storms"] += 1
        era_stats[s["era"]]["bytes"] += s["total_bytes"]
        era_stats[s["era"]]["files"] += s.get("file_count", s.get("total_files", 0))

    # Target proportions calculation
    if basin_proportions:
        norm_b_sum = sum(basin_proportions.values())
        norm_b_targets = {b: (basin_proportions.get(b, 0.0) / norm_b_sum * 100.0) if norm_b_sum > 0 else 0.0 for b in available_basins}
        if "IO" in basin_proportions:
            nio_target_pct = (basin_proportions["IO"] / norm_b_sum * 100.0) if norm_b_sum > 0 else 25.0
        else:
            nio_target_pct = 0.0
    else:
        norm_b_targets = {}
        target_nio_frac = float(nio_fraction) if nio_fraction is not None else 0.25
        if "IO" in available_basins:
            norm_b_targets["IO"] = target_nio_frac * 100.0
            other_b = [b for b in available_basins if b != "IO"]
            rem_pct = max(0.0, 100.0 - norm_b_targets["IO"])
            for b in other_b:
                norm_b_targets[b] = rem_pct / max(1, len(other_b))
        else:
            for b in available_basins:
                norm_b_targets[b] = 100.0 / max(1, len(available_basins))
        nio_target_pct = norm_b_targets.get("IO", target_nio_frac * 100.0)

    if era_proportions:
        norm_e_sum = sum(era_proportions.values())
        norm_e_targets = {e: (era_proportions.get(e, 0.0) / norm_e_sum * 100.0) if norm_e_sum > 0 else 0.0 for e in available_eras}
    else:
        norm_e_targets = {e: 100.0 / max(1, len(available_eras)) for e in available_eras}

    # Populate target_pct, achieved_pct, deviation_pct in basin_stats
    for b in available_basins:
        t_pct = norm_b_targets.get(b, 0.0)
        a_pct = (basin_stats[b]["bytes"] / total_selected_bytes * 100.0) if total_selected_bytes > 0 else 0.0
        basin_stats[b]["target_pct"] = round(t_pct, 2)
        basin_stats[b]["achieved_pct"] = round(a_pct, 2)
        basin_stats[b]["deviation_pct"] = round(a_pct - t_pct, 2)

    # Populate target_pct, achieved_pct, deviation_pct in era_stats
    for e in available_eras:
        t_pct = norm_e_targets.get(e, 0.0)
        a_pct = (era_stats[e]["bytes"] / total_selected_bytes * 100.0) if total_selected_bytes > 0 else 0.0
        era_stats[e]["target_pct"] = round(t_pct, 2)
        era_stats[e]["achieved_pct"] = round(a_pct, 2)
        era_stats[e]["deviation_pct"] = round(a_pct - t_pct, 2)

    # Per-year
    year_bytes = defaultdict(int)
    for s in selected:
        year_bytes[s.get("season", s.get("year", 0))] += s["total_bytes"]

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
    max_year_budget_pct = (max_year_bytes / target_bytes * 100.0) if target_bytes > 0 else 0.0
    max_year_selected_pct = (max_year_bytes / total_selected_bytes * 100.0) if total_selected_bytes > 0 else 0.0

    # Basin stats and NIO allocation check
    io_bytes = basin_stats.get("IO", {}).get("bytes", 0)
    io_pct = (io_bytes / total_selected_bytes * 100) if total_selected_bytes > 0 else 0
    nio_achieved = abs(io_pct - nio_target_pct) <= 10.0

    # Distinct seasons
    distinct_seasons = sorted(set(s.get("season", s.get("year", 0)) for s in selected))
    distinct_eras = sorted(set(s["era"] for s in selected))
    distinct_basins = sorted(set(s["basin"] for s in selected))

    # Limitations
    limitations = []
    min_bytes_expected = int(target_bytes * 0.85)
    unfilled_bytes = max(0, target_bytes - total_selected_bytes)
    unfilled_budget_reason = None
    max_single_year_bytes = int(target_bytes * (single_year_max_pct / 100.0)) if single_year_max_pct > 0 else 0
    selected_folders = set(s["storm_folder"] for s in selected)

    if unfilled_bytes > 0:
        if modality_audit and modality_audit.get("eligible_catalogue_bytes", total_selected_bytes) < target_bytes:
            orig_gb = modality_audit.get("original_catalogue_bytes", 0) / 1e9
            elig_gb = modality_audit.get("eligible_catalogue_bytes", 0) / 1e9
            reasons_dict = modality_audit.get("exclusion_reasons", {})
            reasons_str = f" ({', '.join(f'{k}: {v}' for k, v in sorted(reasons_dict.items()))})" if reasons_dict else ""
            unfilled_budget_reason = (
                f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB: modality requirements "
                f"excluded {modality_audit.get('excluded_storms_count', 0)} storms{reasons_str}, reducing available catalogue "
                f"from {orig_gb:.2f} GB to {elig_gb:.2f} GB."
            )
        elif enforce_single_year_cap:
            blocked_by_year_cap = any(
                (s["storm_folder"] not in selected_folders)
                and (s["total_bytes"] <= unfilled_bytes)
                and (max_single_year_bytes > 0 and (year_bytes[s.get("season", s.get("year"))] + s["total_bytes"] > max_single_year_bytes))
                for s in catalogue
            )
            if blocked_by_year_cap:
                unfilled_budget_reason = (
                    f"Unfilled budget of {unfilled_bytes / 1e9:.2f} GB ({unfilled_bytes:,} bytes): "
                    f"single-year concentration cap ({single_year_max_pct}% of target budget) strictly enforced and candidate storms from other years exhausted."
                )
        if not unfilled_budget_reason and total_selected_bytes < min_bytes_expected and target_bytes > 0:
            unfilled_budget_reason = (
                f"Selected {total_selected_bytes / 1e9:.2f} GB is below 85% of target budget ({min_bytes_expected / 1e9:.2f} GB) "
                f"due to limited qualifying candidate storms in catalogue."
            )
    if unfilled_budget_reason:
        limitations.append(unfilled_budget_reason)

    if total_selected_bytes < min_bytes_expected and target_bytes > 0 and not unfilled_budget_reason:
        limitations.append(f"Selected {total_selected_bytes / 1e9:.2f} GB is below 85% of target budget ({min_bytes_expected / 1e9:.2f} GB).")
    if len(distinct_eras) < len(available_eras):
        missing_eras = set(available_eras) - set(distinct_eras)
        limitations.append(f"Missing historical periods: {missing_eras}")
    if len(distinct_basins) < len(available_basins):
        missing_basins = set(available_basins) - set(distinct_basins)
        limitations.append(f"Missing basins: {missing_basins}")

    # Single-year concentration limitation: evaluated relative to target budget ceiling
    if max_year_budget_pct > single_year_max_pct:
        limitations.append(f"Year {max_year} contributes {max_year_budget_pct:.1f}% of requested target budget (target ceiling: <={single_year_max_pct}%).")
    elif total_selected_bytes < target_bytes and max_year_selected_pct > single_year_max_pct:
        limitations.append(f"Note: Year {max_year} contributes {max_year_budget_pct:.1f}% of requested target budget (within ceiling <={single_year_max_pct}%), representing {max_year_selected_pct:.1f}% of realized underfilled dataset.")

    if abs(io_pct - nio_target_pct) > 10.0:
        limitations.append(f"NIO allocation is {io_pct:.1f}% (target: {nio_target_pct:.1f}% ± 10%).")

    report = {
        "target_bytes": target_bytes,
        "target_gb": target_bytes / 1e9,
        "selected_bytes": total_selected_bytes,
        "selected_gb_decimal": round(total_selected_bytes / 1e9, 4),
        "selected_gib": round(total_selected_bytes / (1024**3), 4),
        "utilization_pct": round(total_selected_bytes / target_bytes * 100, 2) if target_bytes > 0 else 0,
        "unfilled_bytes": unfilled_bytes,
        "unfilled_budget_reason": unfilled_budget_reason,
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
        "nio_target_pct": round(nio_target_pct, 2),
        "nio_pct": round(io_pct, 2),
        "nio_achieved": nio_achieved,
        "basin_proportions_target": {b: round(norm_b_targets.get(b, 0.0), 2) for b in available_basins},
        "era_proportions_target": {e: round(norm_e_targets.get(e, 0.0), 2) for e in available_eras},
        "largest_single_year": max_year,
        "largest_single_year_bytes": max_year_bytes,
        "largest_single_year_budget_pct": round(max_year_budget_pct, 2),
        "largest_single_year_selected_pct": round(max_year_selected_pct, 2),
        "largest_single_year_pct": round(max_year_budget_pct, 2),
        "single_year_max_pct": single_year_max_pct,
        "single_year_cap_basis": "target_budget",
        "max_single_year_budget_bytes": max_single_year_bytes,
        "enforce_single_year_cap": enforce_single_year_cap,
        "limitations": limitations,
        "available_basins_in_catalogue": sorted(available_basins),
        "available_eras_in_catalogue": sorted(available_eras),
        "configured_basin_proportions": basin_proportions,
        "configured_era_proportions": era_proportions,
        "enforce_complete_storms": enforce_complete_storms,
        "byte_budget_exceeded": (total_selected_bytes > target_bytes),
        "modality_audit": modality_audit,
        "planning_enforced_criteria": {
            "enforce_complete_storms": enforce_complete_storms,
            "enforce_single_year_cap": enforce_single_year_cap,
            "single_year_max_pct": single_year_max_pct,
            "single_year_cap_basis": "target_budget",
            "basin_stratification": bool(basin_proportions or nio_fraction is not None),
            "era_stratification": bool(era_proportions),
            "require_environmental": bool(modality_audit.get("modality_requirements_applied", {}).get("require_environmental", False)) if modality_audit else False,
            "min_sensors_per_storm": int(modality_audit.get("modality_requirements_applied", {}).get("min_sensors_per_storm", 0)) if modality_audit else 0,
            "prioritize_radar_overpasses": bool(modality_audit.get("modality_requirements_applied", {}).get("prioritize_radar_overpasses", False)) if modality_audit else False,
            "note": "Planning selection strictly enforces complete lifecycles, basin/era targets, single-year caps, and configured modality eligibility from S3 catalogue metadata.",
        },
        "post_processing_validation_criteria": {
            "forecasting_horizons_hours": [6, 12, 18, 24],
            "intensity_change_verified_at": "preprocessing/sample_index",
            "rapid_intensification_verified_at": "preprocessing/sample_index",
            "physical_valid_masks_verified_at": "dataset_loading",
            "note": "Multitask forecasting horizons, RI occurrence, and valid-pixel masks are validated during NetCDF ingestion and sample indexing, not at S3 catalogue planning.",
        },
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


def build_manifest_from_catalogue(selected_storms, destination_base="data/raw/pilot"):
    """
    Construct manifest entries from catalogue summary when detailed inventory is not available.
    """
    manifest = []
    for s in selected_storms:
        sf = s["storm_folder"]
        rel = f"{s['season']}/{s['basin']}/{s['cyclone_number']}"
        local_dest = os.path.join(destination_base, rel.replace("/", os.sep))
        manifest.append({
            "s3_key": f"{PREFIX}{sf}/",
            "size_bytes": s["total_bytes"],
            "season": s["season"],
            "basin": s["basin"],
            "cyclone_number": s["cyclone_number"],
            "storm_id": s["storm_id"],
            "instrument": s.get("instruments", ""),
            "platform": s.get("platforms", ""),
            "is_environmental": False,
            "local_destination": local_dest,
            "storm_folder": sf,
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

    lines.append("3. STORMS BY BASIN (Target vs Achieved)")
    lines.append("-" * w)
    lines.append(f"  {'Basin':6s} | {'Name':26s} | {'Storms':6s} | {'Selected':10s} | {'Achieved':8s} | {'Target':8s} | {'Deviation':9s}")
    lines.append("  " + "-" * (w - 4))
    for basin, stats in sorted(report["basin_stats"].items()):
        basin_name = BASIN_NAMES.get(basin, basin)
        achieved = stats.get("achieved_pct", (stats["bytes"] / report["selected_bytes"] * 100) if report["selected_bytes"] > 0 else 0)
        target = stats.get("target_pct", 0.0)
        dev = stats.get("deviation_pct", achieved - target)
        sign = "+" if dev > 0 else ""
        lines.append(f"  {basin:6s} | {basin_name:26s} | {stats['storms']:6d} | "
                     f"{stats['bytes'] / 1e9:7.3f} GB | {achieved:7.2f}% | {target:7.2f}% | {sign}{dev:7.2f}%")
    lines.append("")

    lines.append("4. STORMS BY HISTORICAL ERA (Target vs Achieved)")
    lines.append("-" * w)
    lines.append(f"  {'Era':14s} | {'Storms':6s} | {'Selected':10s} | {'Achieved':8s} | {'Target':8s} | {'Deviation':9s}")
    lines.append("  " + "-" * (w - 4))
    for era, stats in sorted(report["era_stats"].items()):
        achieved = stats.get("achieved_pct", (stats["bytes"] / report["selected_bytes"] * 100) if report["selected_bytes"] > 0 else 0)
        target = stats.get("target_pct", 0.0)
        dev = stats.get("deviation_pct", achieved - target)
        sign = "+" if dev > 0 else ""
        lines.append(f"  {era:14s} | {stats['storms']:6d} | {stats['bytes'] / 1e9:7.3f} GB | "
                     f"{achieved:7.2f}% | {target:7.2f}% | {sign}{dev:7.2f}%")
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
    nio_t = report.get('nio_target_pct', 25.0)
    lines.append(f"  NIO (IO) target percentage:          {nio_t:.2f}%")
    lines.append(f"  NIO (IO) achieved percentage:        {report['nio_pct']:.2f}%")
    lines.append(f"  NIO allocation achieved (within 10% of {nio_t:.1f}%): {'YES' if report['nio_achieved'] else 'NO'}")
    lines.append("")

    lines.append("9. YEAR CONCENTRATION")
    lines.append("-" * w)
    cap_max = report.get('single_year_max_pct', 20.0)
    b_pct = report.get('largest_single_year_budget_pct', report.get('largest_single_year_pct', 0.0))
    s_pct = report.get('largest_single_year_selected_pct', b_pct)
    lines.append(f"  Largest single-year: {report['largest_single_year']}")
    lines.append(f"  Single-year cap basis: {report.get('single_year_cap_basis', 'target_budget')} (limit: {cap_max:.1f}%)")
    lines.append(f"  Fraction of target budget:    {b_pct:.2f}%")
    lines.append(f"  Fraction of realized dataset: {s_pct:.2f}%")
    lines.append("")

    if report.get("modality_audit"):
        ma = report["modality_audit"]
        lines.append("9b. MODALITY AUDIT & ELIGIBILITY")
        lines.append("-" * w)
        reqs = ma.get("modality_requirements_applied", {})
        lines.append(f"  Requirements: require_env={reqs.get('require_environmental')}, min_sensors={reqs.get('min_sensors_per_storm')}, prioritize_radar={reqs.get('prioritize_radar_overpasses')}")
        lines.append(f"  Original catalogue:  {ma.get('original_catalogue_storms')} storms ({ma.get('original_catalogue_bytes', 0)/1e9:.2f} GB)")
        lines.append(f"  Eligible catalogue:  {ma.get('eligible_catalogue_storms')} storms ({ma.get('eligible_catalogue_bytes', 0)/1e9:.2f} GB)")
        lines.append(f"  Excluded storms:     {ma.get('excluded_storms_count')}")
        if ma.get("exclusion_reasons"):
            lines.append(f"  Exclusion reasons:   {ma.get('exclusion_reasons')}")
        lines.append(f"  Radar storms:        {ma.get('eligible_radar_storms_count')} eligible / {ma.get('original_radar_storms_count')} original")
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
    lines.append(f"  [{'✓' if report['nio_achieved'] else '!'}] NIO allocation ~{report.get('nio_target_pct', 25.0):.0f}%")
    lines.append(f"  [{'✓' if report['largest_single_year_pct'] <= report.get('single_year_max_pct', 20.0) else '!'}] No single year > {report.get('single_year_max_pct', 20.0):.0f}%")
    lines.append("")

    lines.append("12. PLANNING VS POST-PROCESSING CRITERIA")
    lines.append("-" * w)
    lines.append("  Planning-Enforced (S3 Acquisition):")
    lines.append("    * Complete storm lifecycle preservation (never split storm files)")
    lines.append("    * Basin stratification targets")
    lines.append("    * Historical era stratification targets")
    lines.append("    * Single-year concentration cap enforcement")
    lines.append("    * Environmental and overpass file presence from catalogue metadata")
    lines.append("  Post-Processing Validation (Verified During Preprocessing / Indexing):")
    lines.append("    * Best-track intensity change and RI labels (evaluated from NetCDF variables)")
    lines.append("    * Multitask forecasting horizons (6h, 12h, 18h, 24h)")
    lines.append("    * Physical valid-pixel mask coverage and QC filtering")
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
    cfg = {}
    if getattr(args, "config", None):
        if not os.path.exists(args.config):
            print(f"[ERROR] Configuration file not found: {args.config}", file=sys.stderr)
            return 1
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    acq_cfg = cfg.get("acquisition", {})
    paths_cfg = cfg.get("paths", {})
    strat_cfg = cfg.get("stratification_criteria", {})
    modality_cfg = cfg.get("modality_requirements", {})

    # Parameter resolution: CLI explicit > YAML config > fallback default
    if getattr(args, "target_gb", None) is not None:
        target_bytes = int(args.target_gb * 1e9)
    elif "target_bytes" in acq_cfg:
        target_bytes = int(acq_cfg["target_bytes"])
    elif "target_gb" in acq_cfg:
        target_bytes = int(acq_cfg["target_gb"] * 1e9)
    else:
        target_bytes = DEFAULT_TARGET_BYTES

    seed = args.seed if getattr(args, "seed", None) is not None else acq_cfg.get("seed", DEFAULT_SEED)
    reports_dir = args.reports_dir if getattr(args, "reports_dir", None) is not None else paths_cfg.get("acquisition_reports_dir", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    manifest_path = args.manifest if getattr(args, "manifest", None) is not None else paths_cfg.get("manifest_csv", os.path.join(reports_dir, "pilot_manifest.csv"))
    inventory_path = paths_cfg.get("inventory_csv", os.path.join(reports_dir, "s3_inventory.csv"))
    catalogue_path = paths_cfg.get("catalogue_csv", os.path.join(reports_dir, "storm_catalogue.csv"))
    txt_path = paths_cfg.get("plan_summary_text", os.path.join(reports_dir, "pilot_plan_summary.txt"))
    json_path = paths_cfg.get("plan_summary_json", os.path.join(reports_dir, "pilot_plan_summary.json"))
    raw_dest = paths_cfg.get("raw_destination_dir", "data/raw/pilot")

    basin_proportions = strat_cfg.get("basins", {}).get("target_proportions")
    era_proportions = strat_cfg.get("eras", {}).get("target_proportions")
    single_year_max_pct = strat_cfg.get("single_year_max_pct", 20.0)
    enforce_complete_storms = acq_cfg.get("enforce_complete_storms", True)
    enforce_single_year_cap = strat_cfg.get("enforce_single_year_cap", True)

    inventory = None
    catalogue = None

    if getattr(args, "catalogue_csv", None) and os.path.exists(args.catalogue_csv):
        print(f"[CATALOGUE] Loading pre-existing catalogue from {args.catalogue_csv}")
        catalogue = load_storm_catalogue_csv(args.catalogue_csv)
    elif getattr(args, "inventory_csv", None) and os.path.exists(args.inventory_csv):
        print(f"[INVENTORY] Loading pre-existing inventory from {args.inventory_csv}")
        inventory = load_inventory_csv(args.inventory_csv)
        catalogue = build_storm_catalogue(inventory)
    else:
        # Step 1: S3 inventory
        print("\n" + "=" * 70)
        print("PHASE 1: S3 INVENTORY")
        print("=" * 70)
        listing_lines = run_s3_listing()
        inventory = build_inventory(listing_lines)
        save_inventory_csv(inventory, inventory_path)

        # Step 2: Storm catalogue
        print("\n" + "=" * 70)
        print("PHASE 2: STORM CATALOGUE")
        print("=" * 70)
        catalogue = build_storm_catalogue(inventory)
        save_storm_catalogue_csv(catalogue, catalogue_path)

    total_nc_bytes = sum(s["total_bytes"] for s in catalogue)
    print(f"[CATALOGUE] {len(catalogue)} storm folders, "
          f"{total_nc_bytes / 1e9:.2f} GB total")

    # Step 3: Stratified selection
    print("\n" + "=" * 70)
    print("PHASE 3: STRATIFIED SELECTION")
    print("=" * 70)
    selected_storms, report = stratified_select(
        catalogue=catalogue,
        target_bytes=target_bytes,
        seed=seed,
        basin_proportions=basin_proportions,
        era_proportions=era_proportions,
        single_year_max_pct=single_year_max_pct,
        enforce_complete_storms=enforce_complete_storms,
        enforce_single_year_cap=enforce_single_year_cap,
        modality_requirements=modality_cfg,
    )
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
    if inventory is not None:
        manifest = build_manifest(selected_storms, inventory, destination_base=raw_dest)
    else:
        manifest = build_manifest_from_catalogue(selected_storms, destination_base=raw_dest)

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    save_manifest_csv(manifest, manifest_path)

    # Step 5: Save summaries
    summary_text = generate_plan_summary_text(report)

    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"[OK] Saved plan summary text -> {txt_path}")

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
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
    plan_parser.add_argument("--config", default=None,
                             help="Path to YAML acquisition configuration file")
    plan_parser.add_argument("--target-gb", type=float, default=None,
                             help="Target size in decimal GB")
    plan_parser.add_argument("--seed", type=int, default=None,
                             help="Random seed for deterministic selection")
    plan_parser.add_argument("--reports-dir", default=None,
                             help="Directory for output reports")
    plan_parser.add_argument("--manifest", default=None,
                             help="Output path for manifest CSV (overrides config/default)")
    plan_parser.add_argument("--inventory-csv", default=None,
                             help="Path to pre-existing S3 inventory CSV (offline planning)")
    plan_parser.add_argument("--catalogue-csv", default=None,
                             help="Path to pre-existing storm catalogue CSV (offline planning)")

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

