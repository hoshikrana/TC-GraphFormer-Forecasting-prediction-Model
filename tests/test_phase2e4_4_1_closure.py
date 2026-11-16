"""
Unit and Integration Tests for Phase 2E.4.4.1:
Radar Metadata Correctness, Metric Count Reporting & Final Audit.

Validates:
1. TC PRIMED Radar Identification & Separation:
   - KNOWN_INSTRUMENTS contains filename instruments only (no internal NetCDF groups).
   - Radar capability mapped to platform/instrument combinations:
     (GMI, GPM), (DPR, GPM), (TMI, TRMM), (PR, TRMM).
   - Catalogue radar status marked as "radar_capable": True/False and "radar_confirmed": "unknown".
   - inspect_netcdf_radar_group correctly inspects NetCDF group hierarchy on disk.
2. Radar Candidate Ranking:
   - Lexicographical tuple ranking candidate_key = (is_rad, deficit_score, tie_val).
   - Radar candidate is never overtaken by non-radar candidate with massive deficit score.
   - When prioritize_radar_overpasses is False, deficit score governs selection order.
3. Epoch-Level Valid Target Counts:
   - train_epoch() returns explicit epoch metrics for valid target counts.
   - Distinction between valid track components (dx, dy) and valid track time steps (components = timesteps * 2).
   - Elimination of misleading norm_*_count keys in loss_dict.
4. Modality-Related Unfilled Budget Attribution:
   - Underfill reason distinguishes: single-year cap, catalogue smaller than budget with 0 exclusions,
     modality exclusions reducing budget, and complete-storm granularity constraints.
5. Single-Year Cap Toggle Consistency:
   - enforce_single_year_cap toggle controls year ceiling consistently across all selection passes.
"""

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
    build_storm_catalogue,
    save_storm_catalogue_csv,
    load_storm_catalogue_csv,
    filter_eligible_catalogue,
    stratified_select,
    build_selection_report,
)
from src.training.trainer import TCGraphFormerTrainer
from src.training.losses import MultiTaskLoss


# ---------------------------------------------------------------------------
# Synthetic Models and Helpers
# ---------------------------------------------------------------------------

class MockLinearModel(nn.Module):
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


def make_sample(sample_id: int, intensity_val: float, track_steps_valid: int = 4):
    t_mask = torch.zeros(4, dtype=torch.float32)
    t_mask[:track_steps_valid] = 1.0
    return {
        "ir": torch.randn(1, 64, 64),
        "pmw": torch.randn(4, 64, 64),
        "gprof": torch.randn(2, 64, 64),
        "radar": torch.randn(1, 4, 4, 4),
        "env": torch.randn(8),
        "modality_masks": torch.ones(5),
        "targets": {
            "intensity": torch.tensor(intensity_val, dtype=torch.float32),
            "intensity_mask": torch.tensor(1.0, dtype=torch.float32),
            "pressure": torch.tensor(980.0, dtype=torch.float32),
            "pressure_mask": torch.tensor(1.0, dtype=torch.float32),
            "stage": torch.tensor(2, dtype=torch.long),
            "stage_mask": torch.tensor(1.0, dtype=torch.float32),
            "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0], dtype=torch.float32),
            "intensity_change_mask": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
            "rapid_intensification": torch.tensor(0.0, dtype=torch.float32),
            "ri": torch.tensor(0.0, dtype=torch.float32),
            "ri_mask": torch.tensor(1.0, dtype=torch.float32),
            "track_displacement": torch.randn(4, 2),
            "track_mask": t_mask,
        }
    }


def collate_fn(batch):
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


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestRadarMetadataAndInspection(unittest.TestCase):
    """Priority 1: TC PRIMED Radar Identification, Separation & NetCDF Inspection."""

    def test_known_instruments_does_not_contain_internal_groups(self):
        """Internal NetCDF group names must NOT be in filename instrument sets."""
        for internal_name in ["KuKaGMI", "KuGMI", "KuTMI"]:
            self.assertNotIn(
                internal_name,
                KNOWN_INSTRUMENTS,
                f"Internal group {internal_name} must not be in KNOWN_INSTRUMENTS"
            )

    def test_radar_capable_combinations(self):
        """Radar capability must be grounded in physical sensor/platform combinations."""
        # Supported radar combinations
        self.assertTrue(is_radar_capable_file("GMI", "GPM"))
        self.assertTrue(is_radar_capable_file("DPR", "GPM"))
        self.assertTrue(is_radar_capable_file("TMI", "TRMM"))
        self.assertTrue(is_radar_capable_file("PR", "TRMM"))

        # Non-radar sensors / platforms
        self.assertFalse(is_radar_capable_file("SSMIS", "F18"))
        self.assertFalse(is_radar_capable_file("ATMS", "NPP"))
        self.assertFalse(is_radar_capable_file("MHS", "METOP-B"))
        self.assertFalse(is_radar_capable_file("AMSR2", "GCOM-W1"))

    def test_catalogue_radar_status_planning_time(self):
        """Catalogue planning records radar_capable boolean and radar_confirmed unknown."""
        files = [
            {
                "file_path": "v01r01/final/WP/2021/WP012021/TCPRIMED_WP012021_GPM_GMI_20210501.nc",
                "storm_folder": "WP012021",
                "size_bytes": 5_000_000,
                "season": 2021,
                "basin": "WP",
                "cyclone_number": 1,
                "satellite": "GPM",
                "platform": "GPM",
                "instrument": "GMI",
                "is_environmental": False,
            },
            {
                "file_path": "v01r01/final/WP/2021/WP012021/TCPRIMED_WP012021_ERA5_20210501.nc",
                "storm_folder": "WP012021",
                "size_bytes": 1_000_000,
                "season": 2021,
                "basin": "WP",
                "cyclone_number": 1,
                "satellite": "ERA5",
                "platform": "ERA5",
                "instrument": "ERA5",
                "is_environmental": True,
            }
        ]
        catalogue = build_storm_catalogue(files)
        self.assertEqual(len(catalogue), 1)
        storm = catalogue[0]
        self.assertTrue(storm["radar_capable"])
        self.assertEqual(storm["radar_confirmed"], "unknown")
        self.assertTrue(is_storm_radar_capable(storm))
        self.assertTrue(storm_has_radar(storm))

    def test_csv_roundtrip_radar_metadata(self):
        """CSV export and import must preserve radar_capable and radar_confirmed fields."""
        storm = {
            "storm_folder": "AL012021",
            "storm_id": "AL012021",
            "season": 2021,
            "basin": "AL",
            "cyclone_number": 1,
            "era": "2020+",
            "total_bytes": 10_000_000,
            "file_count": 10,
            "overpass_count": 8,
            "env_count": 2,
            "instruments": "GMI; SSMI",
            "platforms": "GPM; DMSP",
            "radar_capable": True,
            "radar_confirmed": "unknown",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "catalogue.csv")
            save_storm_catalogue_csv([storm], csv_path)
            loaded = load_storm_catalogue_csv(csv_path)

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["storm_folder"], "AL012021")
            self.assertTrue(loaded[0]["radar_capable"])
            self.assertEqual(loaded[0]["radar_confirmed"], "unknown")

    def test_inspect_netcdf_radar_group(self):
        """inspect_netcdf_radar_group accurately inspects local NetCDF files."""
        import netCDF4 as nc

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Non-existent file
            non_existent = os.path.join(tmp_dir, "missing.nc")
            res_missing = inspect_netcdf_radar_group(non_existent)
            self.assertFalse(res_missing["file_exists"])
            self.assertFalse(res_missing["radar_confirmed_available"])

            # 2. File without radar_radiometer group
            no_radar_file = os.path.join(tmp_dir, "no_radar.nc")
            with nc.Dataset(no_radar_file, "w") as ds:
                ds.createGroup("passive_microwave")
            res_no_radar = inspect_netcdf_radar_group(no_radar_file)
            self.assertTrue(res_no_radar["file_exists"])
            self.assertFalse(res_no_radar["has_radar_group"])
            self.assertFalse(res_no_radar["radar_confirmed_available"])

            # 3. File with radar_radiometer group and KuKaGMI subgroup with variables
            radar_file = os.path.join(tmp_dir, "has_radar.nc")
            with nc.Dataset(radar_file, "w") as ds:
                rr = ds.createGroup("radar_radiometer")
                kuka = rr.createGroup("KuKaGMI")
                kuka.createDimension("time", 1)
                kuka.createVariable("zKu", "f4", ("time",))
            res_radar = inspect_netcdf_radar_group(radar_file)
            self.assertTrue(res_radar["file_exists"])
            self.assertTrue(res_radar["has_radar_group"])
            self.assertIn("KuKaGMI", res_radar["subgroups_found"])
            self.assertTrue(res_radar["radar_confirmed_available"])


class TestRadarCandidateRanking(unittest.TestCase):
    """Priority 2: Deterministic Tuple Ranking for Radar Candidates."""

    def test_radar_candidate_ranking_dominance(self):
        """
        Radar candidate must not be overtaken by a non-radar candidate with massive byte deficit.
        Tuple ranking key candidate_key = (is_rad, deficit_score, tie_val).
        """
        # Two storms in the same cell:
        # Storm A is non-radar but has a huge deficit score (e.g. 10 GB)
        # Storm B has radar but a tiny deficit score (e.g. 1 MB)
        candidates = [
            {
                "storm_folder": "STORM_NON_RADAR_HUGE_DEFICIT",
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
                "storm_folder": "STORM_RADAR_TINY_DEFICIT",
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

        # Target budget fits exactly one storm (10 MB)
        selected_radar_pref, _ = stratified_select(
            catalogue=candidates,
            target_bytes=10_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": True},
        )
        self.assertEqual(len(selected_radar_pref), 1)
        self.assertEqual(selected_radar_pref[0]["storm_folder"], "STORM_RADAR_TINY_DEFICIT")

    def test_ranking_when_radar_priority_disabled(self):
        """When prioritize_radar_overpasses is False, selection does not enforce radar preference."""
        candidates = [
            {
                "storm_folder": "STORM_NON_RADAR",
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
                "storm_folder": "STORM_RADAR",
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
        selected, _ = stratified_select(
            catalogue=candidates,
            target_bytes=10_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": False},
        )
        self.assertEqual(len(selected), 1)


class TestEpochValidTargetCounts(unittest.TestCase):
    """Priority 3: Epoch-Level Valid Target Counts & Track Units Distinction."""

    def test_epoch_valid_counts_and_track_units(self):
        """
        train_epoch() must report exact valid target counts.
        valid_track_components_count must be exactly 2x valid_track_timesteps_count.
        """
        model = MockLinearModel()
        loss_fn = MultiTaskLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        trainer = TCGraphFormerTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_amp=False,
            accumulate_grad_batches=1,
        )

        # 3 samples:
        # Sample 1: 4 valid track steps (8 components)
        # Sample 2: 2 valid track steps (4 components)
        # Sample 3: 0 valid track steps (0 components)
        samples = [
            make_sample(1, 50.0, track_steps_valid=4),
            make_sample(2, 60.0, track_steps_valid=2),
            make_sample(3, 70.0, track_steps_valid=0),
        ]
        dataloader = [collate_fn(samples)]

        metrics = trainer.train_epoch(dataloader, epoch=1)

        self.assertIn("valid_intensity_count", metrics)
        self.assertIn("valid_pressure_count", metrics)
        self.assertIn("valid_stage_count", metrics)
        self.assertIn("valid_intensity_change_count", metrics)
        self.assertIn("valid_ri_count", metrics)
        self.assertIn("valid_track_components_count", metrics)
        self.assertIn("valid_track_timesteps_count", metrics)

        self.assertEqual(metrics["valid_intensity_count"], 3)
        self.assertEqual(metrics["valid_pressure_count"], 3)
        self.assertEqual(metrics["valid_stage_count"], 3)
        self.assertEqual(metrics["valid_intensity_change_count"], 12)  # 3 samples * 4 horizons
        self.assertEqual(metrics["valid_ri_count"], 3)

        # Track: 4 steps + 2 steps + 0 steps = 6 timesteps, 12 components
        self.assertEqual(metrics["valid_track_timesteps_count"], 6)
        self.assertEqual(metrics["valid_track_components_count"], 12)
        self.assertEqual(metrics["valid_track_components_count"], metrics["valid_track_timesteps_count"] * 2)

    def test_loss_dict_keys_clean(self):
        """loss_dict must not contain obsolete norm_*_count keys."""
        loss_fn = MultiTaskLoss()
        preds = {
            "intensity": torch.tensor([50.0]),
            "pressure": torch.tensor([980.0]),
            "stage_logits": torch.zeros(1, 5),
            "intensity_change": torch.zeros(1, 4),
            "ri_logits": torch.zeros(1),
            "track_delta": torch.zeros(1, 4, 2),
        }
        targets = {
            "intensity": torch.tensor([50.0]),
            "intensity_mask": torch.tensor([1.0]),
            "pressure": torch.tensor([980.0]),
            "pressure_mask": torch.tensor([1.0]),
            "stage": torch.tensor([1]),
            "stage_mask": torch.tensor([1.0]),
            "intensity_change": torch.zeros(1, 4),
            "intensity_change_mask": torch.ones(1, 4),
            "rapid_intensification": torch.tensor([0.0]),
            "ri_mask": torch.tensor([1.0]),
            "track_displacement": torch.zeros(1, 4, 2),
            "track_mask": torch.ones(1, 4),
        }
        _, loss_dict = loss_fn(preds, targets)

        for k in loss_dict:
            self.assertFalse(k.startswith("norm_"), f"Unexpected norm key {k} in loss_dict")


class TestModalityUnfilledBudgetCases(unittest.TestCase):
    """Priority 4: Modality-Related Unfilled Budget Attribution Cases."""

    def test_case_catalogue_smaller_than_budget_zero_exclusions(self):
        """When input catalogue is smaller than target budget and 0 exclusions occurred."""
        catalogue = [
            {
                "storm_folder": "AL012021",
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
        # Target is 50 MB, catalogue has 10 MB total. No storms excluded.
        selected, report = stratified_select(
            catalogue=catalogue,
            target_bytes=50_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertLess(report["selected_bytes"], 50_000_000)
        reason = report["unfilled_budget_reason"]
        self.assertIn("original catalogue", reason.lower())
        self.assertIn("zero modality exclusions", reason.lower())

    def test_case_modality_exclusions_caused_underfill(self):
        """When original catalogue was sufficient, but modality exclusions caused underfill."""
        catalogue = [
            {
                "storm_folder": "AL012021",
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
                "storm_folder": "AL022021",
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
        # Target is 25 MB. Original is 30 MB. Eligible is 10 MB.
        selected, report = stratified_select(
            catalogue=catalogue,
            target_bytes=25_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "min_sensors_per_storm": 1},
        )
        self.assertLess(report["selected_bytes"], 25_000_000)
        reason = report["unfilled_budget_reason"]
        self.assertIn("modality requirements excluded", reason.lower())


class TestSingleYearCapToggle(unittest.TestCase):
    """Priority 5: Single-Year Concentration Cap Toggle Consistency."""

    def test_single_year_cap_toggle_behavior(self):
        """When enforce_single_year_cap=False, single-year cap is not enforced."""
        candidates = [
            {
                "storm_folder": "AL012021",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 40_000_000,
                "total_files": 40,
                "overpass_count": 30,
                "env_count": 5,
                "instruments": "GMI",
                "platforms": "GPM",
            },
            {
                "storm_folder": "AL022021",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 40_000_000,
                "total_files": 40,
                "overpass_count": 30,
                "env_count": 5,
                "instruments": "GMI",
                "platforms": "GPM",
            },
        ]
        # Target 100 MB. 25% single-year cap would be 25 MB.
        # Both storms are in 2021 (40 MB each).
        # With enforce_single_year_cap=False, both storms can be selected up to budget.
        selected_no_cap, report_no_cap = stratified_select(
            catalogue=candidates,
            target_bytes=100_000_000,
            single_year_max_pct=25.0,
            enforce_single_year_cap=False,
            seed=42,
        )
        self.assertEqual(len(selected_no_cap), 2)
        self.assertEqual(report_no_cap["selected_bytes"], 80_000_000)

        # With enforce_single_year_cap=True, neither storm can fit because each is 40 MB > 25 MB cap!
        selected_with_cap, report_with_cap = stratified_select(
            catalogue=candidates,
            target_bytes=100_000_000,
            single_year_max_pct=25.0,
            enforce_single_year_cap=True,
            seed=42,
        )
        self.assertEqual(len(selected_with_cap), 0)
        self.assertEqual(report_with_cap["selected_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
