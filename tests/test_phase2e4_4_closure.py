"""
Unit and Integration Tests for Phase 2E.4.4: Modality Enforcement, Training Metric Integrity & Final Verification.

Validates:
1. Modality Requirements Enforcement:
   - filter_eligible_catalogue excludes storms missing environmental data (env_count < 1).
   - filter_eligible_catalogue excludes storms with fewer sensors than min_sensors_per_storm.
   - Modality audit properly tallies original vs eligible storms, bytes, exclusion reasons, and radar counts.
   - Deterministic radar overpass prioritization in stratified_select.
   - Transparent reporting of modality filtering in unfilled_budget_reason.
2. MultiTaskLoss Training Metric Aggregation Equivalence:
   - Multi-batch training epoch computes exact task loss ratios: sum(unreduced_loss) / sum(valid_targets).
   - Metrics match manually calculated combined single-batch reference within < 1e-4 on float32.
   - Handles zero-valid-target edge cases safely without division by zero.
3. Production Configuration Precedence Resolvers:
   - resolve_evaluation_config verifies CLI explicit > YAML config > fallback default.
   - resolve_evaluation_config supports negative boolean flags (--single-obs / --use-sequences, --no-strict-checkpoint, --smoke-test).
   - resolve_training_config correctly preserves training.num_workers: 0 without override from machine.num_workers.
   - resolve_training_config supports negative boolean flags (--no-amp, --no-pin-memory, --no-persistent-workers, --no-allow-legacy-resume, --single-obs, --use-sequences).
4. Single-Year Concentration Cap Semantics:
   - Cap basis is strictly defined as requested target budget ceiling: target_bytes * (single_year_max_pct / 100).
   - Underfilled dataset distinguishes budget percentage from realized dataset percentage without false violation flags.
"""

import os
import sys
import copy
import json
import tempfile
import unittest
from typing import Dict, Any, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.trainer import TCGraphFormerTrainer
from src.training.losses import MultiTaskLoss
from src.prepare_pilot_dataset import (
    filter_eligible_catalogue,
    stratified_select,
    build_selection_report,
    get_storm_sensors,
    storm_has_radar,
    RADAR_INSTRUMENTS,
)
from src.evaluation.evaluate_baselines import (
    build_parser as build_eval_parser,
    resolve_evaluation_config,
)
from src.training.train_multitask import (
    build_parser as build_train_parser,
    resolve_training_config,
)


# ---------------------------------------------------------------------------
# Synthetic Test Model and Dataset for Metric Aggregation Checks
# ---------------------------------------------------------------------------

class LinearTestModel(nn.Module):
    """Simple linear regression model for deterministic mathematical equivalence checks."""
    def __init__(self, in_features: int = 16, out_features: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, ir, pmw=None, gprof=None, radar=None, env=None, **kwargs):
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


class MockDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def mock_collate(batch):
    B = len(batch)
    return {
        "ir": torch.stack([item["ir"] for item in batch]),
        "pmw": torch.stack([item["pmw"] for item in batch]),
        "gprof": torch.stack([item["gprof"] for item in batch]),
        "radar": torch.stack([item["radar"] for item in batch]),
        "env": torch.stack([item["env"] for item in batch]),
        "modality_masks": torch.ones(B, 5, dtype=torch.bool),
        "targets": {
            k: torch.stack([item["targets"][k] for item in batch])
            for k in batch[0]["targets"].keys()
        },
    }


def make_synthetic_sample(
    sample_id: int,
    intensity_val: float,
    v_valid: bool = True,
    p_valid: bool = True,
    s_valid: bool = True,
    dv_valid: bool = True,
    ri_valid: bool = True,
    tr_valid: bool = True,
):
    """Create reproducible synthetic sample with controllable target masks."""
    torch.manual_seed(2000 + sample_id)
    return {
        "ir": torch.randn(1, 4, 4),
        "pmw": torch.randn(1, 4, 4),
        "gprof": torch.randn(1, 4, 4),
        "radar": torch.randn(1, 4, 4, 4),
        "env": torch.randn(1, 8),
        "targets": {
            "intensity": torch.tensor(intensity_val, dtype=torch.float32),
            "intensity_mask": torch.tensor(1.0 if v_valid else 0.0, dtype=torch.float32),
            "pressure": torch.tensor(1000.0 - intensity_val * 0.5, dtype=torch.float32),
            "pressure_mask": torch.tensor(1.0 if p_valid else 0.0, dtype=torch.float32),
            "stage": torch.tensor(sample_id % 5, dtype=torch.long),
            "stage_mask": torch.tensor(1.0 if s_valid else 0.0, dtype=torch.float32),
            "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0], dtype=torch.float32),
            "intensity_change_mask": torch.tensor([1.0 if dv_valid else 0.0] * 4, dtype=torch.float32),
            "rapid_intensification": torch.tensor(1.0 if (sample_id % 2 == 1) else 0.0, dtype=torch.float32),
            "ri_mask": torch.tensor(1.0 if ri_valid else 0.0, dtype=torch.float32),
            "track_displacement": torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]], dtype=torch.float32),
            "track_mask": torch.tensor([[1.0 if tr_valid else 0.0]] * 4, dtype=torch.float32),
        },
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestModalityRequirements(unittest.TestCase):
    """Test acquisition modality requirement enforcement and radar preference."""

    def setUp(self):
        self.mock_catalogue = [
            {
                "storm_folder": "AL012021",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; SSMI; MHS",
                "platforms": "GPM; DMSP; NOAA",
            },
            {
                "storm_folder": "AL022021",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 12_000_000,
                "total_files": 12,
                "overpass_count": 12,
                "env_count": 0,  # Missing environmental data
                "instruments": "GMI; DPR; SSMI",
                "platforms": "GPM; DMSP",
            },
            {
                "storm_folder": "EP012020",
                "basin": "EP",
                "era": "2020+",
                "year": 2020,
                "total_bytes": 8_000_000,
                "total_files": 8,
                "overpass_count": 7,
                "env_count": 1,
                "instruments": "MHS",  # Single sensor
                "platforms": "NOAA",
            },
            {
                "storm_folder": "WP012021",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 15_000_000,
                "total_files": 15,
                "overpass_count": 13,
                "env_count": 2,
                "instruments": "KuKaGMI; GMI; SSMI",  # Radar instrument
                "platforms": "GPM; DMSP",
            },
        ]

    def test_filter_eligible_catalogue_environmental(self):
        """Storm with env_count < 1 must be excluded when require_environmental=True."""
        reqs = {"require_environmental": True, "min_sensors_per_storm": 1}
        eligible, audit = filter_eligible_catalogue(self.mock_catalogue, reqs)
        folders = [s["storm_folder"] for s in eligible]

        self.assertNotIn("AL022021", folders)
        self.assertIn("AL012021", folders)
        self.assertEqual(audit["excluded_storms_count"], 1)
        self.assertEqual(audit["exclusion_reasons"]["missing_environmental"], 1)

    def test_filter_eligible_catalogue_min_sensors(self):
        """Storms with fewer sensors than min_sensors_per_storm must be excluded."""
        reqs = {"require_environmental": False, "min_sensors_per_storm": 2}
        eligible, audit = filter_eligible_catalogue(self.mock_catalogue, reqs)
        folders = [s["storm_folder"] for s in eligible]

        self.assertNotIn("EP012020", folders)  # Has only 1 sensor (MHS)
        self.assertIn("AL012021", folders)    # Has 3 sensors
        self.assertIn("WP012021", folders)    # Has 3 sensors
        self.assertEqual(audit["exclusion_reasons"]["insufficient_sensors"], 1)

    def test_filter_eligible_catalogue_modality_audit_metadata(self):
        """Modality audit must accurately record counts, radar presence, and exclusion reasons."""
        reqs = {"require_environmental": True, "min_sensors_per_storm": 2, "prioritize_radar_overpasses": True}
        eligible, audit = filter_eligible_catalogue(self.mock_catalogue, reqs)

        self.assertEqual(audit["original_catalogue_storms"], 4)
        self.assertEqual(audit["eligible_catalogue_storms"], 2)  # AL012021 and WP012021
        self.assertEqual(audit["excluded_storms_count"], 2)
        self.assertEqual(audit["original_radar_storms_count"], 2)  # AL022021 and WP012021
        self.assertEqual(audit["eligible_radar_storms_count"], 1)  # WP012021
        self.assertTrue(audit["modality_requirements_applied"]["prioritize_radar_overpasses"])

    def test_stratified_select_radar_priority(self):
        """When prioritize_radar_overpasses is True, radar storm is selected preferentially."""
        # Create two competing storms in the same cell (basin WP, era 2020+)
        candidates = [
            {
                "storm_folder": "WP_NO_RADAR",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; SSMI",
                "platforms": "GPM; DMSP",
            },
            {
                "storm_folder": "WP_WITH_RADAR",
                "basin": "WP",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; DPR; SSMI",  # DPR is recognized radar
                "platforms": "GPM; DMSP",
            },
        ]
        # Set budget to fit exactly one storm (10 MB)
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=10_000_000,
            seed=42,
            modality_requirements={"require_environmental": True, "prioritize_radar_overpasses": True},
        )
        selected_folders = [s["storm_folder"] for s in selected]
        self.assertEqual(len(selected_folders), 1)
        self.assertEqual(selected_folders[0], "WP_WITH_RADAR")

    def test_underfilled_budget_modality_exclusion_reason(self):
        """Underfilled budget due to modality exclusions must be reported in unfilled_budget_reason."""
        # Catalogue has 45 MB total, but all storms except one (10 MB) have env_count = 0
        candidates = [
            {
                "storm_folder": f"STORM_{i}",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 10_000_000,
                "total_files": 10,
                "overpass_count": 10,
                "env_count": 1 if i == 0 else 0,
                "instruments": "GMI; SSMI",
                "platforms": "GPM; DMSP",
            }
            for i in range(4)
        ]
        selected, report = stratified_select(
            catalogue=candidates,
            target_bytes=40_000_000,  # 40 MB requested
            seed=42,
            modality_requirements={"require_environmental": True},
        )
        self.assertEqual(len(selected), 1)
        self.assertIn("modality requirements", report["unfilled_budget_reason"])
        self.assertIn("missing_environmental: 3", report["unfilled_budget_reason"])


class TestTrainingMetricAggregation(unittest.TestCase):
    """Test MultiTaskLoss training metric aggregation equivalence against combined-batch reference."""

    def test_metric_aggregation_mathematical_equivalence(self):
        """
        Verify that accumulating unreduced task loss sums and valid target counts across microbatches
        produces epoch metrics mathematically identical to a combined single-batch evaluation (< 1e-4).
        """
        model = LinearTestModel(in_features=16, out_features=4)
        loss_fn = MultiTaskLoss()

        # Create 3 microbatches of varying batch sizes and varying valid masks
        batch1_samples = [
            make_synthetic_sample(1, 45.0, v_valid=True, p_valid=True, dv_valid=True, ri_valid=True, tr_valid=True),
            make_synthetic_sample(2, 65.0, v_valid=True, p_valid=False, dv_valid=True, ri_valid=False, tr_valid=True),
            make_synthetic_sample(3, 85.0, v_valid=False, p_valid=True, dv_valid=False, ri_valid=True, tr_valid=False),
        ]
        batch2_samples = [
            make_synthetic_sample(4, 30.0, v_valid=True, p_valid=True, dv_valid=True, ri_valid=True, tr_valid=True),
            make_synthetic_sample(5, 55.0, v_valid=True, p_valid=True, dv_valid=True, ri_valid=True, tr_valid=False),
        ]
        batch3_samples = [
            make_synthetic_sample(6, 95.0, v_valid=False, p_valid=False, dv_valid=True, ri_valid=True, tr_valid=True),
            make_synthetic_sample(7, 110.0, v_valid=True, p_valid=True, dv_valid=True, ri_valid=True, tr_valid=True),
            make_synthetic_sample(8, 70.0, v_valid=True, p_valid=True, dv_valid=False, ri_valid=False, tr_valid=True),
            make_synthetic_sample(9, 40.0, v_valid=True, p_valid=False, dv_valid=True, ri_valid=True, tr_valid=True),
        ]

        all_samples = batch1_samples + batch2_samples + batch3_samples

        # Microbatch dataloader: batch_size=3 produces microbatches [3, 2, 4]
        # To strictly test the 3 discrete batches, use a custom list of batches
        batches = [
            mock_collate(batch1_samples),
            mock_collate(batch2_samples),
            mock_collate(batch3_samples),
        ]

        # 1. Compute manual reference on each microbatch to accumulate exact unreduced sums and counts
        model.eval()
        with torch.no_grad():
            ref_loss_sums = defaultdict(float)
            ref_valid_counts = defaultdict(float)

            for b in batches:
                preds = model(b["ir"])
                targets = b["targets"]
                _, l_dict = loss_fn(preds, targets)

                ref_loss_sums["intensity"] += l_dict["unreduced_loss_intensity"]
                ref_valid_counts["intensity"] += l_dict["valid_intensity_count"]

                ref_loss_sums["pressure"] += l_dict["unreduced_loss_pressure"]
                ref_valid_counts["pressure"] += l_dict["valid_pressure_count"]

                ref_loss_sums["stage"] += l_dict["unreduced_loss_stage"]
                ref_valid_counts["stage"] += l_dict["valid_stage_count"]

                ref_loss_sums["intensity_change"] += l_dict["unreduced_loss_intensity_change"]
                ref_valid_counts["intensity_change"] += l_dict["valid_intensity_change_count"]

                ref_loss_sums["ri"] += l_dict["unreduced_loss_ri"]
                ref_valid_counts["ri"] += l_dict["valid_ri_count"]

                ref_loss_sums["track"] += l_dict["unreduced_loss_track"]
                ref_valid_counts["track"] += l_dict["valid_track_count"]

            # Compute combined reference per-task losses and total weighted loss
            expected_metrics = {}
            expected_total = 0.0
            task_weights = loss_fn.task_weights
            for task, weight in task_weights.items():
                c = ref_valid_counts[task]
                task_loss = ref_loss_sums[task] / c if c > 0 else 0.0
                expected_metrics[f"loss_{task}"] = round(task_loss, 4)
                expected_total += weight * task_loss

            expected_metrics["loss_total"] = round(expected_total, 4)

        # 2. Run trainer.train_epoch() using a DataLoader containing the 3 batches
        class PrebatchedDataLoader:
            def __init__(self, batch_list):
                self.batch_list = batch_list
            def __iter__(self):
                return iter(self.batch_list)
            def __len__(self):
                return len(self.batch_list)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer = TCGraphFormerTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_amp=False,
            accumulate_grad_batches=2,  # Exercise gradient accumulation group handling
        )

        metrics = trainer.train_epoch(PrebatchedDataLoader(batches), epoch=1)

        # 3. Assert metrics match expected combined-batch reference within < 1e-4
        for k in expected_metrics:
            self.assertIn(k, metrics)
            diff = abs(metrics[k] - expected_metrics[k])
            self.assertLess(diff, 1e-4, f"Metric mismatch on {k}: got {metrics[k]}, expected {expected_metrics[k]} (diff={diff})")

    def test_zero_valid_targets_safe_handling(self):
        """When all samples in an epoch have valid_count=0 for a task, reported loss is safely 0.0."""
        model = LinearTestModel(in_features=16, out_features=4)
        loss_fn = MultiTaskLoss()

        # All samples have ri_valid=False
        samples = [
            make_synthetic_sample(1, 50.0, ri_valid=False),
            make_synthetic_sample(2, 60.0, ri_valid=False),
        ]
        dataloader = [mock_collate(samples)]

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        trainer = TCGraphFormerTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            use_amp=False,
            accumulate_grad_batches=1,
        )

        metrics = trainer.train_epoch(dataloader, epoch=1)
        self.assertEqual(metrics["loss_ri"], 0.0)
        self.assertGreater(metrics["loss_total"], 0.0)


class TestProductionConfigResolvers(unittest.TestCase):
    """Test resolve_evaluation_config and resolve_training_config production resolvers."""

    def test_evaluation_resolver_cli_precedence(self):
        """CLI explicit arguments must strictly override YAML config in resolve_evaluation_config."""
        eval_parser = build_eval_parser()
        args = eval_parser.parse_args([
            "--device", "cpu",
            "--batch-size", "32",
            "--single-obs",
            "--no-strict-checkpoint",
            "--smoke-test",
        ])

        yaml_cfg = {
            "machine": {"device": "cuda"},
            "evaluation": {"batch_size": 16, "strict_checkpoint": True, "smoke_test": False},
            "training": {"use_sequences": True},
        }

        resolved = resolve_evaluation_config(args, cfg=yaml_cfg)
        self.assertEqual(resolved["device"], "cpu")
        self.assertEqual(resolved["batch_size"], 32)
        self.assertFalse(resolved["use_sequences"])
        self.assertFalse(resolved["strict_checkpoint"])
        self.assertTrue(resolved["smoke_test"])

    def test_evaluation_resolver_yaml_fallback(self):
        """When CLI flag is omitted, resolve_evaluation_config must adopt YAML configuration."""
        eval_parser = build_eval_parser()
        args = eval_parser.parse_args([])

        yaml_cfg = {
            "machine": {"device": "cuda:1"},
            "evaluation": {"batch_size": 8, "use_sequences": False, "strict_checkpoint": False, "smoke_test": True},
        }

        resolved = resolve_evaluation_config(args, cfg=yaml_cfg)
        self.assertEqual(resolved["device"], "cuda:1")
        self.assertEqual(resolved["batch_size"], 8)
        self.assertFalse(resolved["use_sequences"])
        self.assertFalse(resolved["strict_checkpoint"])
        self.assertTrue(resolved["smoke_test"])

    def test_training_resolver_num_workers_zero_preserved(self):
        """training.num_workers: 0 in YAML must NOT be overridden by machine.num_workers: 8."""
        train_parser = build_train_parser()
        args = train_parser.parse_args([])

        yaml_cfg = {
            "training": {"num_workers": 0},
            "machine": {"num_workers": 8},
        }

        resolved = resolve_training_config(args, cfg=yaml_cfg)
        self.assertEqual(resolved["num_workers"], 0, "num_workers: 0 was improperly overridden by mach_cfg")

    def test_training_resolver_negative_cli_flags(self):
        """Negative CLI flags (--no-amp, --no-pin-memory, etc.) must override positive YAML values."""
        train_parser = build_train_parser()
        args = train_parser.parse_args([
            "--no-amp",
            "--no-pin-memory",
            "--no-persistent-workers",
            "--no-allow-legacy-resume",
            "--single-obs",
        ])

        yaml_cfg = {
            "training": {
                "mixed_precision": True,
                "pin_memory": True,
                "persistent_workers": True,
                "allow_legacy_resume": True,
                "use_sequences": True,
            },
            "machine": {
                "precision": "fp16",
            },
        }

        resolved = resolve_training_config(args, cfg=yaml_cfg)
        self.assertFalse(resolved["use_amp"])
        self.assertFalse(resolved["pin_memory"])
        self.assertFalse(resolved["persistent_workers"])
        self.assertFalse(resolved["allow_legacy_resume"])
        self.assertFalse(resolved["use_sequences"])


class TestSingleYearCapSemantics(unittest.TestCase):
    """Test single-year concentration cap semantics relative to target budget ceiling."""

    def test_underfilled_dataset_cap_basis_semantics(self):
        """
        When realized dataset is underfilled (e.g. 40 GB selected of 100 GB target budget),
        a single year contributing 20 GB is 20% of target budget ceiling, representing 50% of realized dataset.
        No cap violation limitation should be reported since it is within target ceiling.
        """
        # 4 storms from 2021 (5 GB each = 20 GB) and 4 storms from 2020 (5 GB each = 20 GB)
        # Total catalogue available = 40 GB. Target budget = 100 GB.
        catalogue = []
        for i in range(4):
            catalogue.append({
                "storm_folder": f"STORM_2021_{i}",
                "basin": "AL",
                "era": "2020+",
                "year": 2021,
                "total_bytes": 5_000_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; SSMI",
                "platforms": "GPM; DMSP",
            })
            catalogue.append({
                "storm_folder": f"STORM_2020_{i}",
                "basin": "AL",
                "era": "2020+",
                "year": 2020,
                "total_bytes": 5_000_000_000,
                "total_files": 10,
                "overpass_count": 8,
                "env_count": 2,
                "instruments": "GMI; SSMI",
                "platforms": "GPM; DMSP",
            })

        target_bytes = 100_000_000_000  # 100 GB
        single_year_max_pct = 20.0       # 20 GB ceiling

        selected, report = stratified_select(
            catalogue=catalogue,
            target_bytes=target_bytes,
            single_year_max_pct=single_year_max_pct,
            enforce_single_year_cap=True,
            seed=42,
        )

        self.assertEqual(report["selected_bytes"], 40_000_000_000)
        self.assertEqual(report["largest_single_year_budget_pct"], 20.0)
        self.assertEqual(report["largest_single_year_selected_pct"], 50.0)
        self.assertEqual(report["single_year_cap_basis"], "target_budget")
        self.assertEqual(report["max_single_year_budget_bytes"], 20_000_000_000)

        # Limitation list must not flag year 2021 as violating the single year cap
        cap_violations = [lim for lim in report["limitations"] if "target ceiling: <=20.0%" in lim]
        self.assertEqual(len(cap_violations), 0)


if __name__ == "__main__":
    unittest.main()
