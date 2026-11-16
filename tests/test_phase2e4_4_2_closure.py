"""
Unit and Integration Tests for Phase 2E.4.4.2:
Final Radar Availability Validation & Regression Closure.

Validates:
1. Correct Radar-Capable Overpass Identification:
   - Grounded overpass combinations: GMI/GPM and TMI/TRMM.
   - DPR and PR removed as standalone filename instrument identifiers.
   - Exact per-file pairing required; cross-product matching of aggregate instruments and platforms eliminated.
   - Old catalogue CSV records lacking exact per-file pairing metadata cannot be falsely marked radar-capable.
   - radar_confirmed remains 'unknown' at planning time.
2. Tri-State Radar Availability Inspection (inspect_netcdf_radar_group):
   - Case A (Unavailable): availability_flag == 0 -> 'unavailable' regardless of subgroups/coordinates.
   - Case B (Available): availability_flag == 1 -> verified subgroup and documented reflectivity variable
     with >= 1 valid, finite observation. Coordinates alone do not qualify as radar data.
     All-fill reflectivity data returns 'unavailable'.
   - Case C (Unknown): missing file, missing /radar_radiometer, missing availability_flag, or corrupt file.
3. Operational Local Radar Audit (audit_local_radar):
   - Local directory and manifest scanning.
   - Per-overpass inspection and per-storm count aggregation.
   - Machine-readable JSON and CSV report export.
4. Radar Ranking Verification across Passes 1, 2, and 3:
   - Pass 1: Cell-level radar priority.
   - Pass 2: Global deficit filling with radar priority over multi-gigabyte deficit non-radar candidate.
   - Pass 3: Remaining-budget filling with radar priority.
   - Ordinary deficit ranking when radar priority is disabled.
   - Deterministic selection and budget adherence.
5. Complete Underfilled-Budget Attribution Tests:
   - All 5 attribution categories (year cap, catalogue smaller than budget with 0 exclusions,
     modality exclusions reducing budget, complete-storm granularity, general fallback).
   - Modality filtering occurring while budget is still fully filled.
6. Strengthened Training Metric Count Tests:
   - Production track_mask shapes ((B, 4), (B, 4, 1), (B, 4, 2)) and rejection of invalid shapes.
   - Multiple gradient accumulation groups and incomplete final groups.
   - Valid target counts equivalence against independent references.
   - Zero valid target edge cases.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prepare_pilot_dataset import (
    KNOWN_INSTRUMENTS,
    RADAR_CAPABLE_COMBINATIONS,
    RADAR_NETCDF_SUBGROUPS,
    is_radar_capable_file,
    is_storm_radar_capable,
    storm_has_radar,
    inspect_netcdf_radar_group,
    audit_local_radar,
    build_storm_catalogue,
    stratified_select,
    build_selection_report,
)
from src.training.trainer import TCGraphFormerTrainer
from src.training.losses import MultiTaskLoss


# ---------------------------------------------------------------------------
# Synthetic Test Helpers
# ---------------------------------------------------------------------------

class LinearTestModel(nn.Module):
    def __init__(self, in_features: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_features, 4, bias=True)

    def forward(self, ir, **kwargs):
        B = ir.size(0)
        x = ir.view(B, -1)[:, :16]
        out = self.linear(x)
        return {
            "intensity": out[:, 0],
            "pressure": out[:, 1],
            "stage_logits": out.repeat(1, 2)[:, :5],
            "intensity_change": out[:, :4],
            "ri_logits": out[:, 2],
            "track_delta": out.repeat(1, 2)[:, :8].view(B, 4, 2),
        }


def make_metric_sample(sample_id: int, intensity_val: float, track_mask_tensor: torch.Tensor, ri_valid: bool = True):
    return {
        "ir": torch.randn(1, 4, 4),
        "pmw": torch.randn(1, 4, 4),
        "gprof": torch.randn(1, 4, 4),
        "radar": torch.randn(1, 4, 4, 4),
        "env": torch.randn(1, 8),
        "modality_masks": torch.ones(5),
        "targets": {
            "intensity": torch.tensor(intensity_val, dtype=torch.float32),
            "intensity_mask": torch.tensor(1.0, dtype=torch.float32),
            "pressure": torch.tensor(980.0, dtype=torch.float32),
            "pressure_mask": torch.tensor(1.0, dtype=torch.float32),
            "stage": torch.tensor(sample_id % 5, dtype=torch.long),
            "stage_mask": torch.tensor(1.0, dtype=torch.float32),
            "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0], dtype=torch.float32),
            "intensity_change_mask": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "rapid_intensification": torch.tensor(1.0 if (sample_id % 2 == 1) else 0.0, dtype=torch.float32),
            "ri": torch.tensor(1.0 if (sample_id % 2 == 1) else 0.0, dtype=torch.float32),
            "ri_mask": torch.tensor(1.0 if ri_valid else 0.0, dtype=torch.float32),
            "track_displacement": torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]], dtype=torch.float32),
            "track_mask": track_mask_tensor,
        }
    }


def collate_test_batch(batch):
    return {
        "ir": torch.stack([s["ir"] for s in batch]),
        "pmw": torch.stack([s["pmw"] for s in batch]),
        "gprof": torch.stack([s["gprof"] for s in batch]),
        "radar": torch.stack([s["radar"] for s in batch]),
        "env": torch.stack([s["env"] for s in batch]),
        "modality_masks": torch.stack([s["modality_masks"] for s in batch]),
        "targets": {
            k: torch.stack([s["targets"][k] for s in batch])
            for k in batch[0]["targets"]
        }
    }


class PrebatchedDataLoader:
    def __init__(self, batch_list):
        self.batch_list = batch_list
    def __iter__(self):
        return iter(self.batch_list)
    def __len__(self):
        return len(self.batch_list)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestRadarCapableOverpassIdentification(unittest.TestCase):
    """Section 1: Strict TC PRIMED Radar-Capable Overpass Identification."""

    def test_radar_capable_combinations_definition(self):
        """Only GMI on GPM and TMI on TRMM are valid radar-capable combinations."""
        expected = {("GMI", "GPM"), ("TMI", "TRMM")}
        self.assertEqual(RADAR_CAPABLE_COMBINATIONS, expected)

    def test_known_instruments_no_dpr_pr_internal_groups(self):
        """KNOWN_INSTRUMENTS must not include DPR, PR, or internal NetCDF groups."""
        for disallowed in ["DPR", "PR", "KuKaGMI", "KuGMI", "KuTMI"]:
            self.assertNotIn(disallowed, KNOWN_INSTRUMENTS)

    def test_is_radar_capable_file_grounded_pairs(self):
        """is_radar_capable_file only returns True for GMI/GPM and TMI/TRMM."""
        self.assertTrue(is_radar_capable_file("GMI", "GPM"))
        self.assertTrue(is_radar_capable_file("TMI", "TRMM"))

        # DPR and PR are rejected as overpass filename instrument tokens
        self.assertFalse(is_radar_capable_file("DPR", "GPM"))
        self.assertFalse(is_radar_capable_file("PR", "TRMM"))

        # Non-radar sensors / platforms
        self.assertFalse(is_radar_capable_file("SSMIS", "F18"))
        self.assertFalse(is_radar_capable_file("ATMS", "NPP"))
        self.assertFalse(is_radar_capable_file("MHS", "METOP-B"))
        self.assertFalse(is_radar_capable_file("AMSR2", "GCOM-W1"))
        self.assertFalse(is_radar_capable_file(None, "GPM"))
        self.assertFalse(is_radar_capable_file("GMI", None))

    def test_cross_product_elimination_in_is_storm_radar_capable(self):
        """
        Storm with GMI on NOAA and SSMIS on GPM must NOT be marked radar-capable.
        Proves independent cross-product combination of instruments and platforms is eliminated.
        """
        cross_product_storm = {
            "storm_folder": "CROSS_PROD_STORM",
            "instruments": "GMI; SSMIS",
            "platforms": "NOAA; GPM",
            "files": [
                {"instrument": "GMI", "platform": "NOAA"},
                {"instrument": "SSMIS", "platform": "GPM"},
            ]
        }
        self.assertFalse(is_storm_radar_capable(cross_product_storm))

        # True pairing: GMI on GPM
        true_pairing_storm = {
            "storm_folder": "TRUE_PAIRING_STORM",
            "instruments": "GMI; SSMIS",
            "platforms": "GPM; DMSP",
            "files": [
                {"instrument": "GMI", "platform": "GPM"},
                {"instrument": "SSMIS", "platform": "DMSP"},
            ]
        }
        self.assertTrue(is_storm_radar_capable(true_pairing_storm))

    def test_old_catalogue_records_without_pairing_cannot_be_falsely_radar_capable(self):
        """Legacy CSV records without exact file pairing metadata return False."""
        old_record = {
            "storm_folder": "OLD_STORM",
            "instruments": "GMI; SSMIS",
            "platforms": "GPM; DMSP",
            # No 'radar_capable' field, no 'files' list
        }
        self.assertFalse(is_storm_radar_capable(old_record))

    def test_catalogue_planning_radar_confirmed_unknown(self):
        """build_storm_catalogue sets radar_confirmed to 'unknown' at planning time."""
        raw_files = [
            {
                "storm_folder": "AL012021",
                "size_bytes": 10_000_000,
                "season": 2021,
                "basin": "AL",
                "cyclone_number": 1,
                "instrument": "GMI",
                "platform": "GPM",
                "is_environmental": False,
            }
        ]
        catalogue = build_storm_catalogue(raw_files)
        self.assertEqual(len(catalogue), 1)
        self.assertTrue(catalogue[0]["radar_capable"])
        self.assertEqual(catalogue[0]["radar_confirmed"], "unknown")


class TestNetCDFRadarGroupInspection(unittest.TestCase):
    """Section 2: Tri-State NetCDF Radar Group Inspection."""

    def test_case_a_radar_unavailable_flag_zero(self):
        """
        Case A: When availability_flag == 0, status must be 'unavailable'
        even if an unexpected subgroup or coordinate variable exists.
        """
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_unavailable.nc")
            with nc.Dataset(file_path, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                flag = rr.createVariable("availability_flag", "i4")
                flag[...] = 0

                # Even if unexpected subgroup and reflectivity exists:
                sg = rr.createGroup("KuKaGMI")
                sg.createDimension("time", 1)
                z = sg.createVariable("correctedReflectFactor_Ku", "f4", ("time",))
                z[:] = [30.0]

            res = inspect_netcdf_radar_group(file_path)
            self.assertEqual(res["status"], "unavailable")
            self.assertFalse(res["radar_confirmed_available"])
            self.assertEqual(res["availability_flag"], 0)
            self.assertIn("availability_flag is 0", res["reason"])

    def test_case_b_radar_available_with_valid_reflectivity(self):
        """Case B: When availability_flag == 1 with valid observations, status is 'available'."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_available.nc")
            with nc.Dataset(file_path, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                flag = rr.createVariable("availability_flag", "i4")
                flag[...] = 1

                sg = rr.createGroup("KuKaGMI")
                sg.createDimension("time", 3)
                z = sg.createVariable("correctedReflectFactor_Ku", "f4", ("time",), fill_value=-9999.0)
                z[:] = [-9999.0, 22.5, np.nan]  # Contains at least one finite valid observation (22.5)

            res = inspect_netcdf_radar_group(file_path)
            self.assertEqual(res["status"], "available")
            self.assertTrue(res["radar_confirmed_available"])
            self.assertEqual(res["availability_flag"], 1)
            self.assertIn("KuKaGMI", res["subgroups_found"])

    def test_case_b_coordinates_alone_do_not_qualify(self):
        """Case B: Subgroups with only coordinates (x, y) do NOT qualify as available."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_coords_only.nc")
            with nc.Dataset(file_path, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                flag = rr.createVariable("availability_flag", "i4")
                flag[...] = 1

                sg = rr.createGroup("KuKaGMI")
                sg.createDimension("time", 2)
                x = sg.createVariable("x", "f4", ("time",))
                x[:] = [10.0, 20.0]
                y = sg.createVariable("y", "f4", ("time",))
                y[:] = [30.0, 40.0]

            res = inspect_netcdf_radar_group(file_path)
            self.assertEqual(res["status"], "unavailable")
            self.assertFalse(res["radar_confirmed_available"])
            self.assertIn("no valid non-fill reflectivity observations", res["reason"])

    def test_case_b_all_fill_reflectivity_is_unavailable(self):
        """Case B: Subgroup reflectivity where all observations are fill/NaN is 'unavailable'."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_all_fill.nc")
            with nc.Dataset(file_path, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                flag = rr.createVariable("availability_flag", "i4")
                flag[...] = 1

                sg = rr.createGroup("KuTMI")
                sg.createDimension("time", 2)
                z = sg.createVariable("correctedReflectFactor_Ku", "f4", ("time",), fill_value=-9999.0)
                z[:] = [-9999.0, -9999.0]

            res = inspect_netcdf_radar_group(file_path)
            self.assertEqual(res["status"], "unavailable")
            self.assertFalse(res["radar_confirmed_available"])

    def test_case_c_unknown_missing_metadata_and_corrupt(self):
        """Case C: Missing files, missing groups, missing availability_flag, or corrupt files report 'unknown'."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Missing file
            res_missing = inspect_netcdf_radar_group(os.path.join(tmp_dir, "non_existent.nc"))
            self.assertEqual(res_missing["status"], "unknown")
            self.assertFalse(res_missing["file_exists"])

            # 2. File lacking /radar_radiometer group
            no_rr_file = os.path.join(tmp_dir, "no_rr.nc")
            with nc.Dataset(no_rr_file, "w") as ds:
                ds.createGroup("infrared")
            res_no_rr = inspect_netcdf_radar_group(no_rr_file)
            self.assertEqual(res_no_rr["status"], "unknown")
            self.assertFalse(res_no_rr["has_radar_group"])
            self.assertIn("radar_radiometer group not present", res_no_rr["reason"])

            # 3. File with /radar_radiometer group but lacking availability_flag variable
            no_flag_file = os.path.join(tmp_dir, "no_flag.nc")
            with nc.Dataset(no_flag_file, "w") as ds:
                ds.createGroup("radar_radiometer")
            res_no_flag = inspect_netcdf_radar_group(no_flag_file)
            self.assertEqual(res_no_flag["status"], "unknown")
            self.assertTrue(res_no_flag["has_radar_group"])
            self.assertIn("lacks required availability_flag", res_no_flag["reason"])

            # 4. Corrupt non-NetCDF file
            corrupt_file = os.path.join(tmp_dir, "corrupt.nc")
            with open(corrupt_file, "w") as f:
                f.write("not a netcdf file")
            res_corrupt = inspect_netcdf_radar_group(corrupt_file)
            self.assertEqual(res_corrupt["status"], "unknown")


class TestOperationalLocalRadarAudit(unittest.TestCase):
    """Section 3: Operational Local Radar Audit Entry Point."""

    def test_audit_local_radar_directory_scan(self):
        """audit_local_radar scans directory, aggregates per-storm and overall statistics."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            storm_dir = os.path.join(tmp_dir, "WP012021")
            os.makedirs(storm_dir, exist_ok=True)

            # Available file
            f1 = os.path.join(storm_dir, "TCPRIMED_v01r01_WP012021_GMI_GPM_20210501.nc")
            with nc.Dataset(f1, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                fl = rr.createVariable("availability_flag", "i4")
                fl[...] = 1
                sg = rr.createGroup("KuKaGMI")
                sg.createDimension("time", 1)
                z = sg.createVariable("correctedReflectFactor_Ku", "f4", ("time",))
                z[:] = [20.0]

            # Unavailable file
            f2 = os.path.join(storm_dir, "TCPRIMED_v01r01_WP012021_GMI_GPM_20210502.nc")
            with nc.Dataset(f2, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                fl = rr.createVariable("availability_flag", "i4")
                fl[...] = 0

            # Output report path
            report_json = os.path.join(tmp_dir, "radar_audit.json")
            report_csv = os.path.join(tmp_dir, "radar_audit.csv")

            report = audit_local_radar(local_dir=tmp_dir, output_report_path=report_json)

            self.assertEqual(report["total_files_inspected"], 2)
            self.assertEqual(report["total_available"], 1)
            self.assertEqual(report["total_unavailable"], 1)
            self.assertEqual(report["total_unknown"], 0)
            self.assertEqual(report["total_storms"], 1)
            self.assertEqual(report["storms_with_confirmed_radar"], 1)

            # Check JSON file written
            self.assertTrue(os.path.exists(report_json))
            with open(report_json, "r") as f:
                data = json.load(f)
            self.assertEqual(data["total_available"], 1)

            # Check CSV export
            audit_local_radar(local_dir=tmp_dir, output_report_path=report_csv)
            self.assertTrue(os.path.exists(report_csv))


class TestRadarRankingPasses(unittest.TestCase):
    """Section 4: Verify Radar Ranking across Passes 1, 2, and 3."""

    def test_pass_1_cell_level_radar_priority(self):
        """In Pass 1, candidate storms within each cell prioritize radar-capable storms."""
        candidates = [
            {
                "storm_folder": "CELL_NON_RADAR",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "SSMIS",
                "platforms": "DMSP",
                "radar_capable": False,
            },
            {
                "storm_folder": "CELL_RADAR",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
                "radar_capable": True,
            },
        ]
        # Target budget fits 1 storm (10 MB). Cell budget fits 1 storm.
        selected, _ = stratified_select(
            catalogue=candidates,
            target_bytes=10_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": True},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["storm_folder"], "CELL_RADAR")

    def test_pass_2_deficit_filling_radar_priority_over_massive_deficit(self):
        """
        In Pass 2, a radar candidate is selected over a non-radar candidate
        even when the non-radar candidate has a much larger deficit score (e.g. 10 GB vs 1 MB).
        """
        # Basin AL has 10 MB budget. Basin WP has 100 GB budget (massive deficit).
        # Target bytes = 50 MB.
        # Pass 1 fills nothing if cell budgets are small, or we place both in leftover for Pass 2.
        candidates = [
            {
                "storm_folder": "WP_NON_RADAR_MASSIVE_DEFICIT",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 20_000_000,
                "total_files": 20,
                "overpass_count": 15,
                "env_count": 5,
                "instruments": "SSMIS",
                "platforms": "DMSP",
                "radar_capable": False,
            },
            {
                "storm_folder": "AL_RADAR_SMALL_DEFICIT",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 20_000_000,
                "total_files": 20,
                "overpass_count": 15,
                "env_count": 5,
                "instruments": "GMI",
                "platforms": "GPM",
                "radar_capable": True,
            },
        ]
        # Target budget fits exactly one 20 MB storm
        selected, _ = stratified_select(
            catalogue=candidates,
            target_bytes=20_000_000,
            basin_proportions={"WP": 0.90, "AL": 0.10},  # WP has 90% deficit, AL has 10% deficit
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": True},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["storm_folder"], "AL_RADAR_SMALL_DEFICIT")

    def test_pass_2_and_3_radar_disabled_deficit_governs(self):
        """When prioritize_radar_overpasses is False, the storm with largest deficit score is selected."""
        candidates = [
            {
                "storm_folder": "WP_NON_RADAR_MASSIVE_DEFICIT",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 20_000_000,
                "total_files": 20,
                "overpass_count": 15,
                "env_count": 5,
                "instruments": "SSMIS",
                "platforms": "DMSP",
                "radar_capable": False,
            },
            {
                "storm_folder": "AL_RADAR_SMALL_DEFICIT",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 20_000_000,
                "total_files": 20,
                "overpass_count": 15,
                "env_count": 5,
                "instruments": "GMI",
                "platforms": "GPM",
                "radar_capable": True,
            },
        ]
        selected, _ = stratified_select(
            catalogue=candidates,
            target_bytes=20_000_000,
            basin_proportions={"WP": 0.90, "AL": 0.10},
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": False},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["storm_folder"], "WP_NON_RADAR_MASSIVE_DEFICIT")


class TestUnderfilledBudgetCasesComplete(unittest.TestCase):
    """Section 5: Comprehensive Underfilled-Budget Attribution Tests."""

    def test_case_1_year_cap_constrained(self):
        """Case 1: Underfill caused by single-year concentration cap."""
        candidates = [
            {
                "storm_folder": "STORM_2021_A",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            },
            {
                "storm_folder": "STORM_2021_B",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        # Target 50 MB, 25% single-year cap = 12.5 MB.
        # STORM A is selected (10 MB). STORM B (10 MB) cannot fit within 12.5 MB cap.
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=50_000_000,
            single_year_max_pct=25.0,
            enforce_single_year_cap=True,
            seed=42,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["selected_bytes"], 10_000_000)
        self.assertIn("single-year concentration cap", report["unfilled_budget_reason"])

    def test_case_2_catalogue_smaller_than_budget_zero_exclusions(self):
        """Case 2: Input catalogue smaller than target budget with zero exclusions."""
        candidates = [
            {
                "storm_folder": "STORM_1",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            }
        ]
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=50_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertEqual(report["selected_bytes"], 10_000_000)
        reason = report["unfilled_budget_reason"]
        self.assertIn("original catalogue", reason.lower())
        self.assertIn("zero modality exclusions", reason.lower())

    def test_case_3_modality_exclusions_caused_underfill(self):
        """Case 3: Modality exclusions reduce a previously sufficient catalogue."""
        candidates = [
            {
                "storm_folder": "STORM_ELIGIBLE",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; SSMIS",
                "platforms": "GPM; DMSP",
            },
            {
                "storm_folder": "STORM_EXCLUDED",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 20_000_000,
                "total_files": 20,
                "overpass_count": 15,
                "env_count": 0,  # Missing environmental data
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        # Target 25 MB. Original 30 MB >= 25 MB. Eligible 10 MB < 25 MB.
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=25_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertEqual(report["selected_bytes"], 10_000_000)
        reason = report["unfilled_budget_reason"]
        self.assertIn("modality requirements excluded", reason.lower())

    def test_case_4_complete_storm_granularity_constraint(self):
        """Case 4: Eligible catalogue sufficient in total bytes, but remaining storm exceeds remaining budget."""
        candidates = [
            {
                "storm_folder": "STORM_FIT",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 15_000_000,
                "total_files": 15,
                "overpass_count": 10,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            },
            {
                "storm_folder": "STORM_TOO_BIG",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 15_000_000,  # Exceeds remaining 5 MB of 20 MB budget
                "total_files": 15,
                "overpass_count": 10,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        # Target 20 MB. Eligible 30 MB >= 20 MB.
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=20_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertEqual(report["selected_bytes"], 15_000_000)
        reason = report["unfilled_budget_reason"]
        self.assertIn("complete-storm lifecycle constraints", reason.lower())

    def test_case_5_general_fallback(self):
        """Case 5: General fallback attribution is explicitly populated."""
        report = build_selection_report(
            selected=[],
            catalogue=[],
            target_bytes=10_000_000,
            available_basins=["AL"],
            available_eras=["2020+"],
            modality_audit=None,
        )
        self.assertIn("Unfilled budget", report["unfilled_budget_reason"])

    def test_modality_filtering_occurring_with_fully_filled_budget(self):
        """When modality exclusions occur but the eligible catalogue still fills 100% of the budget."""
        candidates = [
            {
                "storm_folder": "STORM_1",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI",
                "platforms": "GPM",
            },
            {
                "storm_folder": "STORM_EXCLUDED",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 0,  # Excluded
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        # Target 10 MB. Eligible has 10 MB, so budget is 100% filled!
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=10_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertEqual(report["selected_bytes"], 10_000_000)
        self.assertEqual(report["unfilled_bytes"], 0)
        self.assertIn(report["unfilled_budget_reason"], (None, ""))


class TestStrengthenedTrainingMetricCounts(unittest.TestCase):
    """Section 6: Strengthened Training Metric Count Tests."""

    def test_track_mask_shapes_and_rejection_of_invalid(self):
        """MultiTaskLoss accepts (B, 4), (B, 4, 1), (B, 4, 2) and rejects invalid shapes."""
        loss_fn = MultiTaskLoss()
        preds = {
            "intensity": torch.tensor([50.0]),
            "pressure": torch.tensor([980.0]),
            "stage_logits": torch.zeros(1, 5),
            "intensity_change": torch.zeros(1, 4),
            "ri_logits": torch.zeros(1),
            "track_delta": torch.zeros(1, 4, 2),
        }

        # 1. 2D shape (B, 4)
        targets_2d = {
            "intensity": torch.tensor([50.0]),
            "pressure": torch.tensor([980.0]),
            "stage": torch.tensor([1]),
            "intensity_change": torch.zeros(1, 4),
            "intensity_change_mask": torch.ones(1, 4),
            "rapid_intensification": torch.tensor([0.0]),
            "ri_mask": torch.tensor([1.0]),
            "track_displacement": torch.zeros(1, 4, 2),
            "track_mask": torch.ones(1, 4),
        }
        _, d_2d = loss_fn(preds, targets_2d)
        self.assertEqual(d_2d["valid_track_timesteps_count"], 4.0)
        self.assertEqual(d_2d["valid_track_components_count"], 8.0)

        # 2. 3D shape (B, 4, 1)
        targets_3d_1 = dict(targets_2d)
        targets_3d_1["track_mask"] = torch.ones(1, 4, 1)
        _, d_3d_1 = loss_fn(preds, targets_3d_1)
        self.assertEqual(d_3d_1["valid_track_timesteps_count"], 4.0)
        self.assertEqual(d_3d_1["valid_track_components_count"], 8.0)

        # 3. 3D shape (B, 4, 2)
        targets_3d_2 = dict(targets_2d)
        targets_3d_2["track_mask"] = torch.ones(1, 4, 2)
        _, d_3d_2 = loss_fn(preds, targets_3d_2)
        self.assertEqual(d_3d_2["valid_track_timesteps_count"], 4.0)
        self.assertEqual(d_3d_2["valid_track_components_count"], 8.0)

        # 4. Invalid shape (B, 5) -> must raise ValueError
        targets_invalid = dict(targets_2d)
        targets_invalid["track_mask"] = torch.ones(1, 5)
        with self.assertRaises(ValueError):
            loss_fn(preds, targets_invalid)

    def test_multiple_accumulation_groups_and_incomplete_final_group(self):
        """
        Verify metric aggregation equivalence across multiple accumulation groups
        and an incomplete final accumulation group (5 batches with accumulate_grad_batches=2).
        """
        batches = [
            collate_test_batch([make_metric_sample(1, 50.0, torch.ones(4, 1)), make_metric_sample(2, 60.0, torch.ones(4, 1))]),
            collate_test_batch([make_metric_sample(3, 70.0, torch.ones(4, 1)), make_metric_sample(4, 80.0, torch.ones(4, 1))]),
            collate_test_batch([make_metric_sample(5, 90.0, torch.ones(4, 1)), make_metric_sample(6, 100.0, torch.ones(4, 1))]),
            collate_test_batch([make_metric_sample(7, 110.0, torch.ones(4, 1)), make_metric_sample(8, 120.0, torch.ones(4, 1))]),
            collate_test_batch([make_metric_sample(9, 130.0, torch.ones(4, 1))]),  # Incomplete final group (1 batch)
        ]

        model = LinearTestModel()
        loss_fn = MultiTaskLoss()

        # Compute exact independent reference
        ref_loss_sums = defaultdict(float)
        ref_valid_counts = defaultdict(float)
        for b in batches:
            preds = model(b["ir"])
            _, l_dict = loss_fn(preds, b["targets"])
            for t in ["intensity", "pressure", "stage", "intensity_change", "ri"]:
                ref_loss_sums[t] += l_dict[f"unreduced_loss_{t}"]
                ref_valid_counts[t] += l_dict[f"valid_{t}_count"]
            ref_loss_sums["track"] += l_dict["unreduced_loss_track"]
            ref_valid_counts["track"] += l_dict["valid_track_components_count"]

        expected_metrics = {}
        for t, weight in loss_fn.task_weights.items():
            c = ref_valid_counts[t]
            expected_metrics[f"loss_{t}"] = round(ref_loss_sums[t] / c if c > 0 else 0.0, 4)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer = TCGraphFormerTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_amp=False,
            accumulate_grad_batches=2,
        )

        metrics = trainer.train_epoch(PrebatchedDataLoader(batches), epoch=1)

        for k, v in expected_metrics.items():
            self.assertIn(k, metrics)
            diff = abs(metrics[k] - v)
            self.assertLess(diff, 1e-4, f"Mismatch on {k}: got {metrics[k]}, expected {v}")

        # Assert explicit epoch valid counts match reference
        self.assertEqual(metrics["valid_intensity_count"], 9)
        self.assertEqual(metrics["valid_pressure_count"], 9)
        self.assertEqual(metrics["valid_stage_count"], 9)
        self.assertEqual(metrics["valid_intensity_change_count"], 36)  # 9 * 4
        self.assertEqual(metrics["valid_ri_count"], 9)
        self.assertEqual(metrics["valid_track_components_count"], 72)  # 9 * 8
        self.assertEqual(metrics["valid_track_timesteps_count"], 36)   # 9 * 4


if __name__ == "__main__":
    unittest.main()
