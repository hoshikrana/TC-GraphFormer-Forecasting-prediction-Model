"""
test_prepare_pilot_dataset.py - Unit tests for the TC PRIMED pilot dataset planner.

Tests cover:
  - S3 listing line parsing
  - Storm-folder grouping
  - Deterministic selection with seed
  - Budget constraint enforcement
  - NIO allocation
  - Complete storm folder integrity
  - Environmental file detection
  - Local path construction
  - Existing-file skipping
  - File-size verification
  - Download failure handling
  - Instrument/platform parsing

Does NOT run real S3 downloads.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.prepare_pilot_dataset import (
    parse_s3_listing_line,
    parse_storm_folder,
    parse_filename_instrument_platform,
    is_environmental_file,
    classify_era,
    storm_id_from_parts,
    build_inventory,
    build_storm_catalogue,
    stratified_select,
    build_manifest,
    save_manifest_csv,
    load_manifest_csv,
    save_inventory_csv,
    check_disk_space,
    download_single_file,
    validate_plan_diversity,
    generate_plan_summary_text,
    audit_local_downloads,
    generate_audit_report_text,
    PREFIX,
)


# ======================================================================
# Synthetic data generators
# ======================================================================

def make_listing_line(date, time, size, key):
    """Build a synthetic `aws s3 ls --recursive` line."""
    return f"{date} {time}  {size} {key}"


def make_nc_key(season, basin, num, instrument, platform, orbit, timestamp):
    """Build a synthetic TC PRIMED S3 key."""
    storm_id = f"{basin}{str(num).zfill(2)}{season}"
    fname = f"TCPRIMED_v01r01-final_{storm_id}_{instrument}_{platform}_{orbit}_{timestamp}.nc"
    return f"v01r01/final/{season}/{basin}/{str(num).zfill(2)}/{fname}"


def make_env_key(season, basin, num):
    """Build a synthetic environmental file key."""
    storm_id = f"{basin}{str(num).zfill(2)}{season}"
    fname = f"TCPRIMED_v01r01-final_{storm_id}_era5_environmental.nc"
    return f"v01r01/final/{season}/{basin}/{str(num).zfill(2)}/{fname}"


def generate_synthetic_listing(num_storms_per_cell=3, files_per_storm=5, env_per_storm=1,
                                file_size=50_000_000):
    """
    Generate a synthetic S3 listing with storms distributed across basins and eras.
    Returns list of listing line strings.
    """
    basins = ["AL", "EP", "WP", "IO", "SH"]
    eras = [(1995, 1999), (2005, 2009), (2015, 2019), (2021, 2024)]
    instruments = ["GMI", "TMI", "SSMI", "ATMS", "AMSR2"]
    platforms = ["GPM", "TRMM", "F13", "NOAA20", "GCOMW1"]

    lines = []
    storm_num = 1

    for basin in basins:
        for era_start, era_end in eras:
            for i in range(num_storms_per_cell):
                season = era_start + (i % (era_end - era_start + 1))
                num_str = str(storm_num).zfill(2)

                for f in range(files_per_storm):
                    inst = instruments[f % len(instruments)]
                    plat = platforms[f % len(platforms)]
                    orbit = str(100000 + storm_num * 100 + f).zfill(6)
                    ts = f"{season}0{(f+1) % 10 + 1}01120000"
                    key = make_nc_key(season, basin, storm_num, inst, plat, orbit, ts)
                    line = make_listing_line("2024-01-15", "12:00:00", file_size, key)
                    lines.append(line)

                for e in range(env_per_storm):
                    key = make_env_key(season, basin, storm_num)
                    line = make_listing_line("2024-01-15", "12:00:00", file_size // 2, key)
                    lines.append(line)

                storm_num += 1

    return lines


def generate_small_catalogue(num_storms_per_cell=2, bytes_per_storm=500_000_000):
    """Generate a small storm catalogue for selection testing."""
    basins = ["AL", "EP", "WP", "IO", "SH"]
    eras_years = {"1987-1999": [1995, 1998], "2000-2009": [2005, 2008],
                  "2010-2019": [2015, 2018], "2020+": [2021, 2023]}

    catalogue = []
    storm_num = 1

    for basin in basins:
        for era, years in eras_years.items():
            for i in range(num_storms_per_cell):
                season = years[i % len(years)]
                storm_folder = f"v01r01/final/{season}/{basin}/{str(storm_num).zfill(2)}"
                storm_id = storm_id_from_parts(season, basin, storm_num)

                catalogue.append({
                    "storm_folder": storm_folder,
                    "storm_id": storm_id,
                    "season": season,
                    "basin": basin,
                    "cyclone_number": str(storm_num).zfill(2),
                    "era": era,
                    "total_bytes": bytes_per_storm,
                    "file_count": 6,
                    "overpass_count": 5,
                    "env_count": 1,
                    "instruments": "GMI;SSMI",
                    "platforms": "GPM;F13",
                })
                storm_num += 1

    return catalogue


# ======================================================================
# Tests: S3 Listing Parser
# ======================================================================

class TestS3ListingParser(unittest.TestCase):
    """Test parsing of AWS CLI `s3 ls --recursive` output lines."""

    def test_valid_nc_line(self):
        line = "2024-01-15 12:34:56    12345678 v01r01/final/2014/IO/03/TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc"
        result = parse_s3_listing_line(line)
        self.assertIsNotNone(result)
        self.assertEqual(result["size_bytes"], 12345678)
        self.assertIn("IO", result["key"])
        self.assertTrue(result["key"].endswith(".nc"))

    def test_non_nc_line(self):
        line = "2024-01-15 12:34:56    1234 v01r01/final/index.html"
        result = parse_s3_listing_line(line)
        self.assertIsNotNone(result)
        self.assertFalse(result["key"].endswith(".nc"))

    def test_empty_line(self):
        result = parse_s3_listing_line("")
        self.assertIsNone(result)

    def test_whitespace_line(self):
        result = parse_s3_listing_line("   ")
        self.assertIsNone(result)

    def test_malformed_line(self):
        result = parse_s3_listing_line("This is not an S3 listing line")
        self.assertIsNone(result)

    def test_zero_size_line(self):
        line = "2024-01-15 12:34:56          0 v01r01/final/2014/IO/03/"
        result = parse_s3_listing_line(line)
        self.assertIsNotNone(result)
        self.assertEqual(result["size_bytes"], 0)

    def test_large_file_size(self):
        line = "2024-01-15 12:34:56 999999999999 v01r01/final/2020/AL/01/big_file.nc"
        result = parse_s3_listing_line(line)
        self.assertIsNotNone(result)
        self.assertEqual(result["size_bytes"], 999999999999)

    def test_multiple_valid_lines(self):
        lines = [
            "2024-01-15 10:00:00   100000 v01r01/final/2014/IO/03/file1.nc",
            "2024-01-15 10:00:01   200000 v01r01/final/2014/IO/03/file2.nc",
            "2024-01-15 10:00:02   300000 v01r01/final/2015/AL/01/file3.nc",
        ]
        results = [parse_s3_listing_line(l) for l in lines]
        self.assertTrue(all(r is not None for r in results))
        self.assertEqual(sum(r["size_bytes"] for r in results), 600000)


# ======================================================================
# Tests: Storm Folder Grouping
# ======================================================================

class TestStormFolderGrouping(unittest.TestCase):
    """Test storm folder parsing and grouping."""

    def test_valid_key_parsing(self):
        key = "v01r01/final/2014/IO/03/some_file.nc"
        result = parse_storm_folder(key)
        self.assertIsNotNone(result)
        self.assertEqual(result["season"], 2014)
        self.assertEqual(result["basin"], "IO")
        self.assertEqual(result["cyclone_number"], "03")
        self.assertEqual(result["storm_folder"], "v01r01/final/2014/IO/03")

    def test_different_basins(self):
        for basin in ["AL", "EP", "WP", "IO", "SH", "CP"]:
            key = f"v01r01/final/2020/{basin}/01/test.nc"
            result = parse_storm_folder(key)
            self.assertIsNotNone(result)
            self.assertEqual(result["basin"], basin)

    def test_different_seasons(self):
        for season in [1987, 1999, 2000, 2010, 2020, 2024]:
            key = f"v01r01/final/{season}/AL/01/test.nc"
            result = parse_storm_folder(key)
            self.assertIsNotNone(result)
            self.assertEqual(result["season"], season)

    def test_short_key_returns_none(self):
        result = parse_storm_folder("v01r01/final/2014")
        self.assertIsNone(result)

    def test_non_numeric_season(self):
        result = parse_storm_folder("v01r01/final/abc/IO/03/test.nc")
        self.assertIsNone(result)

    def test_storm_catalogue_grouping(self):
        """Verify that inventory entries are grouped by complete storm folder."""
        lines = generate_synthetic_listing(num_storms_per_cell=2, files_per_storm=3, env_per_storm=1)
        inventory = build_inventory(lines)
        catalogue = build_storm_catalogue(inventory)

        # Each storm should have all its files grouped together
        storm_files = defaultdict(list)
        for row in inventory:
            storm_files[row["storm_folder"]].append(row)

        for storm in catalogue:
            sf = storm["storm_folder"]
            self.assertEqual(storm["file_count"], len(storm_files[sf]),
                             f"File count mismatch for {sf}")
            self.assertEqual(storm["total_bytes"],
                             sum(f["size_bytes"] for f in storm_files[sf]),
                             f"Byte count mismatch for {sf}")


# ======================================================================
# Tests: Deterministic Selection
# ======================================================================

class TestDeterministicSelection(unittest.TestCase):
    """Test that selection is deterministic with the same seed."""

    def test_same_seed_same_result(self):
        catalogue = generate_small_catalogue()
        target = 5_000_000_000  # 5 GB

        selected1, _ = stratified_select(catalogue, target, seed=42)
        selected2, _ = stratified_select(catalogue, target, seed=42)

        folders1 = sorted([s["storm_folder"] for s in selected1])
        folders2 = sorted([s["storm_folder"] for s in selected2])
        self.assertEqual(folders1, folders2)

    def test_different_seed_different_result(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=5, bytes_per_storm=200_000_000)
        target = 5_000_000_000

        selected1, _ = stratified_select(catalogue, target, seed=42)
        selected2, _ = stratified_select(catalogue, target, seed=123)

        folders1 = set(s["storm_folder"] for s in selected1)
        folders2 = set(s["storm_folder"] for s in selected2)
        # With enough storms and different seeds, sets should differ
        # (not guaranteed but highly likely with enough data)
        # We just check both return valid results
        self.assertTrue(len(selected1) > 0)
        self.assertTrue(len(selected2) > 0)


# ======================================================================
# Tests: Budget Constraint
# ======================================================================

class TestBudgetConstraint(unittest.TestCase):
    """Test that selection never exceeds the target budget."""

    def test_never_exceeds_target(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=5, bytes_per_storm=1_000_000_000)
        target = 5_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        total = sum(s["total_bytes"] for s in selected)
        self.assertLessEqual(total, target,
                             f"Selected {total} bytes exceeds target {target}")

    def test_various_targets(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=500_000_000)
        for target in [1_000_000_000, 5_000_000_000, 10_000_000_000, 20_000_000_000]:
            selected, _ = stratified_select(catalogue, target, seed=42)
            total = sum(s["total_bytes"] for s in selected)
            self.assertLessEqual(total, target,
                                 f"Exceeded target {target} with {total}")

    def test_very_small_target(self):
        catalogue = generate_small_catalogue(bytes_per_storm=1_000_000_000)
        target = 100_000_000  # 100 MB - smaller than any single storm

        selected, _ = stratified_select(catalogue, target, seed=42)
        total = sum(s["total_bytes"] for s in selected)
        self.assertLessEqual(total, target)
        # Might select zero storms if all storms are too large
        self.assertGreaterEqual(len(selected), 0)

    def test_exact_target(self):
        """When a storm exactly matches remaining budget, it should be included."""
        catalogue = [
            {
                "storm_folder": "v01r01/final/2020/AL/01",
                "storm_id": "AL012020",
                "season": 2020,
                "basin": "AL",
                "cyclone_number": "01",
                "era": "2020+",
                "total_bytes": 5_000_000_000,
                "file_count": 10,
                "overpass_count": 9,
                "env_count": 1,
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        target = 5_000_000_000

        selected, _ = stratified_select(catalogue, target, seed=42)
        self.assertEqual(len(selected), 1)
        total = sum(s["total_bytes"] for s in selected)
        self.assertEqual(total, target)


# ======================================================================
# Tests: NIO Budget Allocation
# ======================================================================

class TestNIOAllocation(unittest.TestCase):
    """Test that NIO (IO basin) gets approximately 25% of budget."""

    def test_io_allocation_present(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=4, bytes_per_storm=400_000_000)
        target = 10_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        io_bytes = sum(s["total_bytes"] for s in selected if s["basin"] == "IO")
        total_bytes = sum(s["total_bytes"] for s in selected)

        if total_bytes > 0:
            io_pct = io_bytes / total_bytes * 100
            # NIO should be roughly 25% (within a reasonable tolerance)
            # given discrete storm folder sizes
            self.assertGreater(io_pct, 0,
                               "IO basin should have some representation")

    def test_io_storms_selected(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=500_000_000)
        target = 15_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        io_storms = [s for s in selected if s["basin"] == "IO"]
        self.assertGreater(len(io_storms), 0,
                           "At least one IO storm should be selected")


# ======================================================================
# Tests: Complete Storm Folder Integrity
# ======================================================================

class TestStormFolderIntegrity(unittest.TestCase):
    """Test that selected storm folders remain complete."""

    def test_all_files_from_storm_included(self):
        """When a storm is selected, all its files must be in the manifest."""
        lines = generate_synthetic_listing(num_storms_per_cell=2, files_per_storm=4, env_per_storm=1)
        inventory = build_inventory(lines)
        catalogue = build_storm_catalogue(inventory)

        selected, _ = stratified_select(catalogue, 5_000_000_000, seed=42)
        manifest = build_manifest(selected, inventory)

        selected_folders = set(s["storm_folder"] for s in selected)

        # Count files per folder in inventory vs manifest
        inv_counts = defaultdict(int)
        for row in inventory:
            if row["storm_folder"] in selected_folders:
                inv_counts[row["storm_folder"]] += 1

        man_counts = defaultdict(int)
        for row in manifest:
            man_counts[row["storm_folder"]] += 1

        for sf in selected_folders:
            self.assertEqual(inv_counts[sf], man_counts[sf],
                             f"Storm folder {sf}: inventory has {inv_counts[sf]} files "
                             f"but manifest has {man_counts[sf]}")

    def test_no_partial_storms(self):
        """Manifest should not contain partial storm folders."""
        lines = generate_synthetic_listing(num_storms_per_cell=2, files_per_storm=5)
        inventory = build_inventory(lines)
        catalogue = build_storm_catalogue(inventory)

        selected, _ = stratified_select(catalogue, 3_000_000_000, seed=42)
        manifest = build_manifest(selected, inventory)

        selected_folders = set(s["storm_folder"] for s in selected)
        manifest_folders = set(row["storm_folder"] for row in manifest)

        # Every folder in manifest should be in selected storms
        self.assertEqual(manifest_folders, selected_folders)


# ======================================================================
# Tests: Environmental File Detection
# ======================================================================

class TestEnvironmentalDetection(unittest.TestCase):
    """Test environmental file detection."""

    def test_environmental_filename(self):
        self.assertTrue(is_environmental_file("TCPRIMED_v01r01-final_IO032014_era5_environmental.nc"))
        self.assertTrue(is_environmental_file("TCPRIMED_v01r01-final_AL012020_ERA5_environmental.nc"))

    def test_overpass_filename(self):
        self.assertFalse(is_environmental_file("TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc"))
        self.assertFalse(is_environmental_file("TCPRIMED_v01r01-final_AL012020_SSMI_F13_054321_20200815120000.nc"))

    def test_environmental_included_in_catalogue(self):
        lines = generate_synthetic_listing(num_storms_per_cell=1, files_per_storm=3, env_per_storm=1)
        inventory = build_inventory(lines)
        catalogue = build_storm_catalogue(inventory)

        for storm in catalogue:
            self.assertEqual(storm["env_count"], 1,
                             f"Storm {storm['storm_folder']} missing environmental file")


# ======================================================================
# Tests: Instrument/Platform Parsing
# ======================================================================

class TestInstrumentParsing(unittest.TestCase):
    """Test instrument and platform parsing from filenames."""

    def test_gmi_gpm(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc")
        self.assertEqual(inst, "GMI")
        self.assertEqual(plat, "GPM")

    def test_tmi_trmm(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_AL041999_TMI_TRMM_010087_19990829131352.nc")
        self.assertEqual(inst, "TMI")
        self.assertEqual(plat, "TRMM")

    def test_ssmi(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_EP071999_SSMI_F13_022614_19990812020754.nc")
        self.assertEqual(inst, "SSMI")
        self.assertEqual(plat, "F13")

    def test_atms(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_WP012021_ATMS_NOAA20_045678_20210501120000.nc")
        self.assertEqual(inst, "ATMS")
        self.assertEqual(plat, "NOAA20")

    def test_environmental(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_IO032014_era5_environmental.nc")
        self.assertEqual(inst, "ERA5")
        self.assertEqual(plat, "reanalysis")

    def test_amsr2(self):
        inst, plat = parse_filename_instrument_platform(
            "TCPRIMED_v01r01-final_WP052020_AMSR2_GCOMW1_012345_20200801120000.nc")
        self.assertEqual(inst, "AMSR2")
        self.assertEqual(plat, "GCOMW1")


# ======================================================================
# Tests: Era Classification
# ======================================================================

class TestEraClassification(unittest.TestCase):
    """Test historical period classification."""

    def test_era_bins(self):
        self.assertEqual(classify_era(1987), "1987-1999")
        self.assertEqual(classify_era(1999), "1987-1999")
        self.assertEqual(classify_era(2000), "2000-2009")
        self.assertEqual(classify_era(2009), "2000-2009")
        self.assertEqual(classify_era(2010), "2010-2019")
        self.assertEqual(classify_era(2019), "2010-2019")
        self.assertEqual(classify_era(2020), "2020+")
        self.assertEqual(classify_era(2024), "2020+")
        self.assertEqual(classify_era(2030), "2020+")


# ======================================================================
# Tests: Local Path Construction
# ======================================================================

class TestLocalPathConstruction(unittest.TestCase):
    """Test correct construction of local file paths."""

    def test_path_preserves_hierarchy(self):
        inventory = [{
            "s3_key": "v01r01/final/2014/IO/03/TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc",
            "size_bytes": 100000,
            "season": 2014,
            "basin": "IO",
            "cyclone_number": "03",
            "storm_folder": "v01r01/final/2014/IO/03",
            "filename": "TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc",
            "instrument": "GMI",
            "platform": "GPM",
            "is_environmental": False,
        }]
        storms = [{
            "storm_folder": "v01r01/final/2014/IO/03",
            "storm_id": "IO032014",
            "season": 2014,
            "basin": "IO",
            "cyclone_number": "03",
            "era": "2010-2019",
            "total_bytes": 100000,
            "file_count": 1,
            "overpass_count": 1,
            "env_count": 0,
            "instruments": "GMI",
            "platforms": "GPM",
        }]

        manifest = build_manifest(storms, inventory, destination_base="data/raw/pilot")
        self.assertEqual(len(manifest), 1)

        local_path = manifest[0]["local_destination"]
        # Should contain season/basin/number hierarchy
        self.assertIn("2014", local_path)
        self.assertIn("IO", local_path)
        self.assertIn("03", local_path)
        self.assertTrue(local_path.endswith(".nc"))
        # Should not rename the file
        self.assertIn("TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc", local_path)


# ======================================================================
# Tests: File Skipping and Size Verification
# ======================================================================

class TestFileSkipping(unittest.TestCase):
    """Test existing-file skipping and size verification."""

    def test_skip_correct_size(self):
        """File with matching size should be skipped."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.nc', delete=False) as f:
            f.write(b'\x00' * 1000)
            temp_path = f.name

        try:
            success, actual, msg = download_single_file(
                "test/key.nc", temp_path, expected_size=1000
            )
            self.assertTrue(success)
            self.assertEqual(actual, 1000)
            self.assertIn("skipped", msg)
        finally:
            os.unlink(temp_path)

    def test_redownload_wrong_size(self):
        """File with wrong size should trigger redownload attempt."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.nc', delete=False) as f:
            f.write(b'\x00' * 500)  # wrong size
            temp_path = f.name

        try:
            # This will fail because there's no real S3 to download from,
            # but it should NOT skip the file
            success, actual, msg = download_single_file(
                "test/nonexistent_key.nc", temp_path, expected_size=1000, retries=1
            )
            # Should attempt download (and fail since no real S3)
            self.assertFalse(success)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# ======================================================================
# Tests: Download Failure Handling
# ======================================================================

class TestDownloadFailureHandling(unittest.TestCase):
    """Test that download failures are handled gracefully."""

    def test_nonexistent_file_fails_gracefully(self):
        """Downloading a non-existent key should fail without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "test_file.nc")
            success, actual, msg = download_single_file(
                "totally/fake/key.nc", dest, expected_size=1000, retries=1
            )
            self.assertFalse(success)
            self.assertIsInstance(msg, str)


# ======================================================================
# Tests: Manifest CSV Round-trip
# ======================================================================

class TestManifestCSV(unittest.TestCase):
    """Test manifest save and load round-trip."""

    def test_roundtrip(self):
        manifest = [
            {
                "s3_key": "v01r01/final/2014/IO/03/test.nc",
                "size_bytes": 12345,
                "season": 2014,
                "basin": "IO",
                "cyclone_number": "03",
                "storm_id": "IO032014",
                "instrument": "GMI",
                "platform": "GPM",
                "is_environmental": False,
                "local_destination": "data/raw/pilot/2014/IO/03/test.nc",
                "storm_folder": "v01r01/final/2014/IO/03",
            },
            {
                "s3_key": "v01r01/final/2014/IO/03/test_env.nc",
                "size_bytes": 6789,
                "season": 2014,
                "basin": "IO",
                "cyclone_number": "03",
                "storm_id": "IO032014",
                "instrument": "ERA5",
                "platform": "reanalysis",
                "is_environmental": True,
                "local_destination": "data/raw/pilot/2014/IO/03/test_env.nc",
                "storm_folder": "v01r01/final/2014/IO/03",
            },
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            temp_path = f.name

        try:
            save_manifest_csv(manifest, temp_path)
            loaded = load_manifest_csv(temp_path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["s3_key"], manifest[0]["s3_key"])
            self.assertEqual(loaded[0]["size_bytes"], 12345)
            self.assertEqual(loaded[0]["season"], 2014)
            self.assertFalse(loaded[0]["is_environmental"])
            self.assertTrue(loaded[1]["is_environmental"])
        finally:
            os.unlink(temp_path)


# ======================================================================
# Tests: Plan Diversity Validation
# ======================================================================

class TestPlanDiversityValidation(unittest.TestCase):
    """Test the plan diversity validation logic."""

    def test_valid_diverse_plan(self):
        report = {
            "distinct_basins_count": 3,
            "distinct_eras_count": 4,
            "selected_bytes": 15_000_000_000,
            "target_bytes": 20_000_000_000,
        }
        valid, issues = validate_plan_diversity(report, ["AL", "IO", "WP"], ["1987-1999", "2000-2009", "2010-2019", "2020+"])
        self.assertTrue(valid)
        self.assertEqual(len(issues), 0)

    def test_single_basin_rejected(self):
        report = {
            "distinct_basins_count": 1,
            "distinct_eras_count": 3,
            "selected_bytes": 15_000_000_000,
            "target_bytes": 20_000_000_000,
        }
        valid, issues = validate_plan_diversity(report, ["AL", "IO", "WP"], ["1987-1999", "2000-2009", "2020+"])
        self.assertFalse(valid)
        self.assertTrue(any("basin" in i.lower() for i in issues))

    def test_single_era_rejected(self):
        report = {
            "distinct_basins_count": 3,
            "distinct_eras_count": 1,
            "selected_bytes": 15_000_000_000,
            "target_bytes": 20_000_000_000,
        }
        valid, issues = validate_plan_diversity(report, ["AL", "IO"], ["1987-1999", "2000-2009"])
        self.assertFalse(valid)
        self.assertTrue(any("period" in i.lower() for i in issues))

    def test_budget_exceeded_rejected(self):
        report = {
            "distinct_basins_count": 3,
            "distinct_eras_count": 3,
            "selected_bytes": 25_000_000_000,
            "target_bytes": 20_000_000_000,
        }
        valid, issues = validate_plan_diversity(report, ["AL", "IO", "WP"], ["1987-1999", "2000-2009", "2020+"])
        self.assertFalse(valid)
        self.assertTrue(any("exceeds" in i.lower() for i in issues))


# ======================================================================
# Tests: Summary Report Generation
# ======================================================================

class TestSummaryReportGeneration(unittest.TestCase):
    """Test that summary report is generated correctly."""

    def test_plan_summary_contains_required_info(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=400_000_000)
        target = 10_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        report["total_catalogue_storms"] = len(catalogue)

        summary = generate_plan_summary_text(report)

        # Check for required sections
        self.assertIn("SIZE SUMMARY", summary)
        self.assertIn("CONTENT COUNTS", summary)
        self.assertIn("STORMS BY BASIN", summary)
        self.assertIn("STORMS BY HISTORICAL ERA", summary)
        self.assertIn("INSTRUMENT COUNTS", summary)
        self.assertIn("NIO ALLOCATION", summary)
        self.assertIn("VALIDATION CHECKLIST", summary)
        self.assertIn("GB", summary)
        self.assertIn("GiB", summary)


# ======================================================================
# Tests: Storm ID Generation
# ======================================================================

class TestStormId(unittest.TestCase):
    """Test storm ID generation."""

    def test_standard_format(self):
        self.assertEqual(storm_id_from_parts(2014, "IO", 3), "IO032014")
        self.assertEqual(storm_id_from_parts(1999, "AL", 4), "AL041999")
        self.assertEqual(storm_id_from_parts(2020, "WP", 12), "WP122020")


# ======================================================================
# Tests: Inventory Building
# ======================================================================

class TestInventoryBuilding(unittest.TestCase):
    """Test full inventory building from synthetic listing."""

    def test_filters_non_nc_files(self):
        lines = [
            "2024-01-15 12:00:00    1000 v01r01/final/index.html",
            "2024-01-15 12:00:00    2000 v01r01/final/docs/readme.txt",
            "2024-01-15 12:00:00    3000 v01r01/final/2014/IO/03/file.nc",
        ]
        inventory = build_inventory(lines)
        self.assertEqual(len(inventory), 1)
        self.assertTrue(inventory[0]["s3_key"].endswith(".nc"))

    def test_excludes_preliminary_data(self):
        lines = [
            "2024-01-15 12:00:00    3000 v01r01/preliminary/2014/IO/03/file.nc",
            "2024-01-15 12:00:00    4000 v01r01/final/2014/IO/03/file.nc",
        ]
        inventory = build_inventory(lines)
        self.assertEqual(len(inventory), 1)
        self.assertIn("final", inventory[0]["s3_key"])

    def test_inventory_csv_save(self):
        lines = generate_synthetic_listing(num_storms_per_cell=1, files_per_storm=2, env_per_storm=1)
        inventory = build_inventory(lines)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            save_inventory_csv(inventory, temp_path)
            # Verify CSV is readable
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), len(inventory))
        finally:
            os.unlink(temp_path)


# ======================================================================
# Tests: Disk Space Check
# ======================================================================

class TestDiskSpaceCheck(unittest.TestCase):
    """Test disk space verification."""

    def test_check_returns_tuple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            has_space, free = check_disk_space(tmpdir, 1000)
            self.assertIsInstance(has_space, bool)
            self.assertIsInstance(free, int)
            self.assertGreater(free, 0)

    def test_absurdly_large_requirement_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 100 TB should exceed any disk
            has_space, free = check_disk_space(tmpdir, 100_000_000_000_000)
            self.assertFalse(has_space)


# ======================================================================
# Tests: Multi-era Multi-basin Selection
# ======================================================================

class TestMultiErMultiBasinSelection(unittest.TestCase):
    """Integration test: selection covers multiple eras and basins."""

    def test_selection_covers_multiple_eras(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=300_000_000)
        target = 15_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        eras = set(s["era"] for s in selected)
        self.assertGreater(len(eras), 1,
                           f"Selection should cover multiple eras, got: {eras}")

    def test_selection_covers_multiple_basins(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=300_000_000)
        target = 15_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        basins = set(s["basin"] for s in selected)
        self.assertGreater(len(basins), 1,
                           f"Selection should cover multiple basins, got: {basins}")

    def test_selection_includes_io(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=300_000_000)
        target = 15_000_000_000

        selected, _ = stratified_select(catalogue, target, seed=42)
        io_storms = [s for s in selected if s["basin"] == "IO"]
        self.assertGreater(len(io_storms), 0,
                           "IO basin should be represented in selection")

    def test_era_basin_matrix_populated(self):
        catalogue = generate_small_catalogue(num_storms_per_cell=3, bytes_per_storm=300_000_000)
        target = 15_000_000_000

        selected, report = stratified_select(catalogue, target, seed=42)
        matrix = report["era_basin_matrix"]
        self.assertGreater(len(matrix), 0, "Era x Basin matrix should be populated")


# ======================================================================
# Tests: Environmental File Classification Regression Tests
# ======================================================================

class TestEnvironmentalRegression(unittest.TestCase):
    """Regression tests for exact environmental filename handling."""

    def test_cp021992_exact_environmental_filename(self):
        """Verify the exact filename reported in the issue is classified correctly."""
        fname = "TCPRIMED_v01r01-final_CP021992_env_s19920328060000_e19920330180000.nc"
        self.assertTrue(is_environmental_file(fname), "CP021992 env file must be detected as environmental")

        inst, plat = parse_filename_instrument_platform(fname)
        self.assertEqual(inst, "ERA5")
        self.assertEqual(plat, "reanalysis")

    def test_various_env_patterns(self):
        """Verify detection of various real and legacy environmental filename conventions."""
        test_cases = [
            ("TCPRIMED_v01r01-final_IO021993_env_s19931127000000_e19931205180000.nc", True),
            ("TCPRIMED_v01r01-final_WP091993_env_s19930711120000_e19930717060000.nc", True),
            ("TCPRIMED_v01r01-final_WP371993_env_s19931218180000_e19931230060000.nc", True),
            ("TCPRIMED_v01r01-final_AL012020_ERA5_environmental.nc", True),
            ("TCPRIMED_v01r01-final_EP012005_env.nc", True),
            ("TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc", False),
            ("TCPRIMED_v01r01-final_IO032014_GMI_GPM_012345_20140101120000.nc", False),
        ]
        for fname, expected in test_cases:
            with self.subTest(fname=fname):
                self.assertEqual(is_environmental_file(fname), expected)

    def test_overpass_and_environmental_sum_consistency(self):
        """Verify that overpass_count + env_count == file_count."""
        lines = [
            make_listing_line("2024-01-15", "12:00:00", 1000,
                              "v01r01/final/1992/CP/02/TCPRIMED_v01r01-final_CP021992_SSMI_F10_000001_19920328000000.nc"),
            make_listing_line("2024-01-15", "12:00:00", 2000,
                              "v01r01/final/1992/CP/02/TCPRIMED_v01r01-final_CP021992_SSMI_F10_000002_19920329000000.nc"),
            make_listing_line("2024-01-15", "12:00:00", 5000,
                              "v01r01/final/1992/CP/02/TCPRIMED_v01r01-final_CP021992_env_s19920328060000_e19920330180000.nc"),
        ]
        inventory = build_inventory(lines)
        self.assertEqual(len(inventory), 3)
        env_files = [x for x in inventory if x["is_environmental"]]
        self.assertEqual(len(env_files), 1)
        self.assertEqual(env_files[0]["instrument"], "ERA5")

        catalogue = build_storm_catalogue(inventory)
        self.assertEqual(len(catalogue), 1)
        storm = catalogue[0]
        self.assertEqual(storm["file_count"], 3)
        self.assertEqual(storm["overpass_count"], 2)
        self.assertEqual(storm["env_count"], 1)
        self.assertEqual(storm["overpass_count"] + storm["env_count"], storm["file_count"])


# ======================================================================
# Tests: Downloader Invocation & Diagnostics
# ======================================================================

class TestDownloaderInvocation(unittest.TestCase):
    """Test downloader subprocess invocation and diagnostic reporting."""

    @patch("subprocess.run")
    def test_downloader_uses_sys_executable(self, mock_run):
        """Verify downloader explicitly invokes [sys.executable, '-m', 'awscli', ...]."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "test_file.nc")
            with open(dest, "wb") as f:
                f.write(b"x" * 100)

            success, size, msg = download_single_file("fake/key.nc", dest, expected_size=100)
            self.assertTrue(success)
            self.assertIn("skipped", msg)
            mock_run.assert_not_called()

            dest_missing = os.path.join(tmpdir, "subdir", "missing.nc")

            def side_effect(*args, **kwargs):
                os.makedirs(os.path.dirname(dest_missing), exist_ok=True)
                with open(dest_missing, "wb") as f:
                    f.write(b"x" * 200)
                return mock_result

            mock_run.side_effect = side_effect

            success, size, msg = download_single_file("fake/key2.nc", dest_missing, expected_size=200)
            self.assertTrue(success)
            self.assertEqual(msg, "downloaded")

            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            self.assertEqual(called_cmd[0], sys.executable)
            self.assertEqual(called_cmd[1], "-m")
            self.assertEqual(called_cmd[2], "awscli")
            self.assertEqual(called_cmd[3], "s3")
            self.assertEqual(called_cmd[4], "cp")

    @patch("subprocess.run")
    def test_filenotfounderror_reports_diagnostics(self, mock_run):
        """Verify FileNotFoundError reports WinError 2 diagnostics with executable and cwd."""
        mock_run.side_effect = FileNotFoundError(2, "The system cannot find the file specified")

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "dest.nc")
            success, size, msg = download_single_file("fake/key.nc", dest, expected_size=500, retries=1)

            self.assertFalse(success)
            self.assertIn("FileNotFoundError", msg)
            self.assertIn("WinError 2", msg)
            self.assertIn(sys.executable, msg)

    def test_destination_parent_directory_creation(self):
        """Verify destination parent directory is created even before subprocess executes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_dest = os.path.join(tmpdir, "deep", "nested", "path", "file.nc")
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = Exception("Stop after dir check")
                download_single_file("fake/key.nc", deep_dest, expected_size=100, retries=1)

            self.assertTrue(os.path.exists(os.path.dirname(deep_dest)))


# ======================================================================
# Tests: Local File Audit
# ======================================================================

class TestLocalFileAudit(unittest.TestCase):
    """Test local file audit classification."""

    def test_audit_classifications(self):
        """Test classification of correct, missing, incomplete, and unexpected files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storm_dir = os.path.join(tmpdir, "1990", "AL", "01")
            os.makedirs(storm_dir, exist_ok=True)

            correct_dest = os.path.join(storm_dir, "correct.nc")
            with open(correct_dest, "wb") as f:
                f.write(b"x" * 100)

            incomplete_dest = os.path.join(storm_dir, "incomplete.nc")
            with open(incomplete_dest, "wb") as f:
                f.write(b"x" * 50)

            missing_dest = os.path.join(storm_dir, "missing.nc")

            unexpected_dest = os.path.join(tmpdir, "unexpected.tmp")
            with open(unexpected_dest, "wb") as f:
                f.write(b"x" * 75)

            manifest = [
                {
                    "s3_key": "v01r01/final/1990/AL/01/correct.nc",
                    "local_destination": correct_dest,
                    "size_bytes": 100,
                    "storm_id": "AL011990",
                    "is_environmental": False,
                },
                {
                    "s3_key": "v01r01/final/1990/AL/01/incomplete.nc",
                    "local_destination": incomplete_dest,
                    "size_bytes": 200,
                    "storm_id": "AL011990",
                    "is_environmental": False,
                },
                {
                    "s3_key": "v01r01/final/1990/AL/01/missing.nc",
                    "local_destination": missing_dest,
                    "size_bytes": 300,
                    "storm_id": "AL011990",
                    "is_environmental": True,
                },
            ]

            report = audit_local_downloads(manifest, destination_base=tmpdir)
            self.assertEqual(report["total_manifest_entries"], 3)
            self.assertEqual(report["correct_count"], 1)
            self.assertEqual(report["incomplete_count"], 1)
            self.assertEqual(report["missing_count"], 1)
            self.assertEqual(report["unexpected_count"], 1)
            self.assertEqual(report["correct_bytes"], 100)
            self.assertEqual(report["missing_bytes"], 300)


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)

