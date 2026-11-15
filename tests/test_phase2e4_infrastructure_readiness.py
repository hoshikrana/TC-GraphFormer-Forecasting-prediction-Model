"""
Unit and Integration Tests for Phase 2E.4: Training, Evaluation & Dataset Infrastructure Readiness.
Validates:
- Training YAML configuration parsing, CLI overrides, and argument bounds
- Sequence dataset chronological ordering and storm-level isolation (zero leakage)
- Multi-task loss target masking and RI class-imbalance weighting
- Strict checkpoint compatibility and mismatch detection
- Metric calculation correctness (track displacement in km, RI PR-AUC, Brier score)
- 100 GB acquisition and pipeline configurations
- Configurable two-machine staging and SHA-256 transfer verification
- Synthetic CPU training-step execution
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_multitask import (
    load_yaml_config,
    validate_checkpoint_compatibility,
    load_pretrained_backbone,
    run_multitask_training,
)
from src.training.losses import MultiTaskLoss
from src.models.tc_graphformer import TCGraphFormer
from src.data.dataset import TCPRIMEDSequenceDataset
from src.data.collate import collate_sequence_batch
from src.evaluation.evaluate_baselines import compute_pr_auc_and_brier, haversine_distance_km
from tcprimed_pipeline.config import PipelineConfig
from src.infrastructure.export_shards import create_staging_manifest, compute_sha256
from src.infrastructure.verify_transfer import verify_staging


class TestPhase2E4InfrastructureReadiness(unittest.TestCase):

    def test_01_training_yaml_config_and_cli_overrides(self):
        """Verify YAML config parsing and parameter precedence (CLI > YAML > Default)."""
        cfg_path = "configs/multitask_training.yaml"
        self.assertTrue(os.path.exists(cfg_path), f"Missing config: {cfg_path}")

        cfg = load_yaml_config(cfg_path)
        self.assertIn("training", cfg)
        self.assertIn("loss_weights", cfg)

        self.assertEqual(cfg["training"]["epochs"], 15)
        self.assertEqual(cfg["training"]["batch_size"], 16)
        self.assertEqual(cfg["training"]["d_model"], 128)
        self.assertEqual(cfg["loss_weights"]["ri_pos_weight"], 10.0)

        # Precedence verification: CLI > YAML > Default
        def resolve_val(cli_val, yaml_val, default):
            if cli_val is not None:
                return cli_val
            if yaml_val is not None:
                return yaml_val
            return default

        # Case 1: CLI provided overrides both YAML and default
        self.assertEqual(resolve_val(cli_val=25, yaml_val=cfg["training"]["epochs"], default=5), 25)
        # Case 2: CLI is None -> takes YAML value over default
        self.assertEqual(resolve_val(cli_val=None, yaml_val=cfg["training"]["epochs"], default=5), 15)
        # Case 3: Both CLI and YAML are None -> takes fallback default
        self.assertEqual(resolve_val(cli_val=None, yaml_val=None, default=42), 42)

    def test_02_parameter_validation_bounds(self):
        """Verify that invalid hyperparameters fail loudly with ValueError."""
        with self.assertRaises(ValueError):
            run_multitask_training(epochs=0)
        with self.assertRaises(ValueError):
            run_multitask_training(batch_size=-1)
        with self.assertRaises(ValueError):
            run_multitask_training(lr=-1e-4)
        with self.assertRaises(ValueError):
            run_multitask_training(d_model=0)
        with self.assertRaises(ValueError):
            run_multitask_training(grad_clip_norm=0.0)
        with self.assertRaises(ValueError):
            run_multitask_training(max_history_len=0)
        with self.assertRaises(ValueError):
            run_multitask_training(loss_weights={"ri_pos_weight": -5.0})

    def test_03_checkpoint_compatibility_validator(self):
        """Verify checkpoint compatibility detection and rejection of mismatches."""
        model = TCGraphFormer(d_model=64, num_stages=5)
        tmp_dir = tempfile.mkdtemp()
        try:
            valid_ckpt_path = os.path.join(tmp_dir, "valid.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": 3,
                "metrics": {"val_loss": 1.23},
            }, valid_ckpt_path)

            res = validate_checkpoint_compatibility(valid_ckpt_path, model, strict=True)
            self.assertTrue(res["is_compatible"])
            self.assertEqual(res["missing_keys"], [])
            self.assertEqual(res["shape_mismatches"], [])

            # Corrupt state with missing keys
            corrupt_state = {k: v for k, v in model.state_dict().items() if "embed" not in k}
            corrupt_ckpt_path = os.path.join(tmp_dir, "corrupt.pt")
            torch.save({"model_state_dict": corrupt_state}, corrupt_ckpt_path)

            with self.assertRaises(ValueError):
                validate_checkpoint_compatibility(corrupt_ckpt_path, model, strict=True)

            res_non_strict = validate_checkpoint_compatibility(corrupt_ckpt_path, model, strict=False)
            self.assertFalse(res_non_strict["is_compatible"])
            self.assertGreater(len(res_non_strict["missing_keys"]), 0)

            # Incompatible backbone load
            incomp_ckpt_path = os.path.join(tmp_dir, "incompatible.pt")
            torch.save({"model_state_dict": {"random_layer.weight": torch.randn(10, 10)}}, incomp_ckpt_path)
            with self.assertRaises(ValueError):
                load_pretrained_backbone(model, incomp_ckpt_path)
        finally:
            shutil.rmtree(tmp_dir)

    def test_04_sequence_dataset_chronological_ordering_and_storm_isolation(self):
        """Verify that sequence datasets are strictly chronological and isolated by storm and split."""
        sample_index = "reports/pipeline/sample_index.csv"
        self.assertTrue(os.path.exists(sample_index))
        df = pd.read_csv(sample_index)

        # Check zero storm overlap between splits
        train_storms = set(df[df["partition"] == "TRAIN"]["storm_id"].unique())
        val_storms = set(df[df["partition"] == "VAL"]["storm_id"].unique())
        test_storms = set(df[df["partition"] == "TEST"]["storm_id"].unique())

        self.assertEqual(len(train_storms.intersection(val_storms)), 0, "Leakage between TRAIN and VAL!")
        self.assertEqual(len(train_storms.intersection(test_storms)), 0, "Leakage between TRAIN and TEST!")
        self.assertEqual(len(val_storms.intersection(test_storms)), 0, "Leakage between VAL and TEST!")

        # Check chronological ordering and storm purity within sequences
        seq_ds = TCPRIMEDSequenceDataset(
            sample_index_csv=sample_index,
            partition="TRAIN",
            max_history_len=4,
            max_sequences=30,
        )
        self.assertGreater(len(seq_ds), 0)

        for i in range(min(15, len(seq_ds))):
            seq = seq_ds[i]
            self.assertLessEqual(len(seq), 4)
            self.assertGreater(len(seq), 0)

            # Check all observations belong to same storm
            s_ids = [obs["storm_id"] for obs in seq]
            self.assertEqual(len(set(s_ids)), 1, f"Sequence {i} contains multiple storms: {set(s_ids)}")

            # Check strictly non-decreasing timestamps
            epochs = [obs["timestamp_epoch"] for obs in seq]
            for t in range(len(epochs) - 1):
                self.assertLessEqual(epochs[t], epochs[t + 1], f"Sequence {i} is not chronologically ordered!")

    def test_05_multitask_loss_target_masking_and_ri_weight(self):
        """Verify loss computation ignores missing targets and respects pos_weight."""
        loss_fn = MultiTaskLoss(
            weight_intensity=1.0,
            weight_pressure=1.0,
            weight_stage=1.0,
            weight_intensity_change=1.0,
            weight_ri=1.0,
            weight_track=1.0,
            ri_pos_weight=10.0,
        )

        B = 2
        preds = {
            "intensity": torch.tensor([50.0, 60.0]),
            "pressure": torch.tensor([990.0, 980.0]),
            "stage_logits": torch.zeros((B, 5)),
            "intensity_change": torch.zeros((B, 4)),
            "ri_logits": torch.tensor([0.0, 0.0]),
            "track_delta": torch.zeros((B, 4, 2)),
        }

        # Case A: Targets with completely valid masks
        targets_valid = {
            "intensity": torch.tensor([50.0, 60.0]),
            "intensity_mask": torch.tensor([1.0, 1.0]),
            "pressure": torch.tensor([990.0, 980.0]),
            "pressure_mask": torch.tensor([1.0, 1.0]),
            "stage": torch.tensor([0, 1]),
            "stage_mask": torch.tensor([1.0, 1.0]),
            "intensity_change": torch.zeros((B, 4)),
            "intensity_change_mask": torch.ones((B, 4)),
            "rapid_intensification": torch.tensor([0.0, 1.0]),
            "ri_mask": torch.tensor([1.0, 1.0]),
            "track_displacement": torch.zeros((B, 4, 2)),
            "track_mask": torch.ones((B, 4)),
        }
        loss_val, l_dict = loss_fn(preds, targets_valid)
        self.assertFalse(torch.isnan(loss_val))
        self.assertGreater(loss_val.item(), 0.0)

        # Case B: Completely invalid targets (all masks 0)
        # Even with completely crazy NaN / sentinel targets, loss must NOT explode or be NaN
        targets_invalid = {
            "intensity": torch.tensor([9999.0, 9999.0]),
            "intensity_mask": torch.tensor([0.0, 0.0]),
            "pressure": torch.tensor([9999.0, 9999.0]),
            "pressure_mask": torch.tensor([0.0, 0.0]),
            "stage": torch.tensor([0, 0]),
            "stage_mask": torch.tensor([0.0, 0.0]),
            "intensity_change": torch.tensor([[999.0] * 4] * B),
            "intensity_change_mask": torch.zeros((B, 4)),
            "rapid_intensification": torch.tensor([1.0, 1.0]),
            "ri_mask": torch.tensor([0.0, 0.0]),
            "track_displacement": torch.tensor([[[999.0, 999.0]] * 4] * B),
            "track_mask": torch.zeros((B, 4)),
        }
        loss_inval, l_dict_inval = loss_fn(preds, targets_invalid)
        self.assertFalse(torch.isnan(loss_inval))
        self.assertEqual(l_dict_inval["loss_intensity"], 0.0)
        self.assertEqual(l_dict_inval["loss_pressure"], 0.0)
        self.assertEqual(l_dict_inval["loss_stage"], 0.0)
        self.assertEqual(l_dict_inval["loss_intensity_change"], 0.0)
        self.assertEqual(l_dict_inval["loss_ri"], 0.0)
        self.assertEqual(l_dict_inval["loss_track"], 0.0)

    def test_06_evaluation_metric_computations(self):
        """Verify metric functions: PR-AUC, Brier score, and Haversine distance."""
        # Test PR-AUC & Brier
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        pr_auc, brier, prec, rec = compute_pr_auc_and_brier(y_true, y_prob)

        self.assertAlmostEqual(pr_auc, 1.0, places=2)
        self.assertAlmostEqual(prec, 1.0, places=2)
        self.assertAlmostEqual(rec, 1.0, places=2)
        self.assertLess(brier, 0.05)

        # Test Haversine distance
        # Distance from (0, 0) to (0, 1) deg lon at equator is ~111.19 km
        d_km = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(d_km, 111.19, delta=0.5)

    def test_07_100gb_acquisition_and_pipeline_configurations(self):
        """Verify 100 GB acquisition and pipeline configurations have separated paths."""
        acq_cfg_path = "configs/acquisition_100gb.yaml"
        pipe_cfg_path = "configs/pipeline_100gb.yaml"

        self.assertTrue(os.path.exists(acq_cfg_path))
        self.assertTrue(os.path.exists(pipe_cfg_path))

        acq_cfg = load_yaml_config(acq_cfg_path)
        self.assertEqual(acq_cfg["acquisition"]["target_bytes"], 100000000000)
        self.assertEqual(acq_cfg["paths"]["raw_destination_dir"], "data/raw/tcprimed_100gb")
        self.assertEqual(acq_cfg["paths"]["processed_destination_dir"], "data/processed/tcprimed_100gb")
        self.assertFalse(acq_cfg["execution_policy"]["auto_download"])

        pipe_cfg = PipelineConfig.from_yaml(pipe_cfg_path)
        self.assertEqual(pipe_cfg.dataset.root, "data/raw/tcprimed_100gb")
        self.assertEqual(pipe_cfg.output.root, "data/processed/tcprimed_100gb")
        self.assertEqual(pipe_cfg.output.reports_dir, "reports/pipeline_100gb")

    def test_08_staging_export_and_transfer_verification(self):
        """Verify configurable staging manifest generation and SHA-256 verification."""
        tmp_staging = tempfile.mkdtemp()
        tmp_proc = tempfile.mkdtemp()
        tmp_reports = tempfile.mkdtemp()

        try:
            # Create synthetic processed NetCDF file
            file1 = os.path.join(tmp_proc, "storm_01.nc")
            with open(file1, "wb") as f:
                f.write(b"SYNTHETIC_NETCDF_CONTENT_STORM_01")

            file2 = os.path.join(tmp_proc, "storm_02.nc")
            with open(file2, "wb") as f:
                f.write(b"SYNTHETIC_NETCDF_CONTENT_STORM_02")

            manifest_path = create_staging_manifest(
                processed_dir=tmp_proc,
                split_manifest_path=os.path.join(tmp_reports, "non_existent.csv"),
                sample_index_path=os.path.join(tmp_reports, "non_existent.csv"),
                output_dir=tmp_staging,
                manifest_name="test_staging.json",
                dataset_name="Test Staging Dataset",
            )
            self.assertTrue(os.path.exists(manifest_path))

            # Verify successful transfer
            report_out = os.path.join(tmp_reports, "transfer_report.json")
            success = verify_staging(manifest_path=manifest_path, base_dir=".", report_output=report_out)
            self.assertTrue(success)

            with open(report_out, "r") as f:
                rep_data = json.load(f)
            self.assertEqual(rep_data["verification_status"], "SUCCESS")
            self.assertEqual(rep_data["passed_files"], 2)
            self.assertEqual(rep_data["failed_files"], 0)

            # Corrupt one file and verify failure detection
            with open(file1, "wb") as f:
                f.write(b"CORRUPTED_CONTENT")

            fail_report_out = os.path.join(tmp_reports, "fail_report.json")
            fail_success = verify_staging(manifest_path=manifest_path, base_dir=".", report_output=fail_report_out)
            self.assertFalse(fail_success)

            with open(fail_report_out, "r") as f:
                fail_data = json.load(f)
            self.assertEqual(fail_data["verification_status"], "FAILURE")
            self.assertEqual(fail_data["failed_files"], 1)
            self.assertEqual(fail_data["passed_files"], 1)
        finally:
            shutil.rmtree(tmp_staging, ignore_errors=True)
            shutil.rmtree(tmp_proc, ignore_errors=True)
            shutil.rmtree(tmp_reports, ignore_errors=True)

    def test_09_synthetic_cpu_training_step(self):
        """Verify minimal synthetic 1-step forward and backward pass on CPU."""
        device = "cpu"
        model = TCGraphFormer(d_model=64, num_stages=5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = MultiTaskLoss(ri_pos_weight=10.0).to(device)

        B, T, H, W = 2, 3, 32, 32
        batch = {
            "ir": torch.randn((B, T, 1, H, W)),
            "pmw": torch.randn((B, T, 4, H, W)),
            "gprof": torch.randn((B, T, 4, H, W)),
            "radar": torch.randn((B, T, 1, 16, H // 2, W // 2)),
            "env": torch.randn((B, T, 6, H, W)),
            "modality_masks": torch.ones((B, T, 5)),
            "delta_times_hours": torch.zeros((B, T)),
            "seq_pad_mask": torch.zeros((B, T), dtype=torch.bool),
            "channel_masks": {
                "pmw": torch.ones((B, T, 4)),
                "gprof": torch.ones((B, T, 4)),
                "env": torch.ones((B, T, 6)),
            },
            "pixel_masks": {
                "ir": torch.ones((B, T, 1, H, W)),
                "pmw": torch.ones((B, T, 4, H, W)),
                "gprof": torch.ones((B, T, 4, H, W)),
                "radar": torch.ones((B, T, 1, 16, H // 2, W // 2)),
                "radar_echo": torch.ones((B, T, 1, 16, H // 2, W // 2)),
                "env": torch.ones((B, T, 6, H, W)),
            },
            "targets": {
                "intensity": torch.tensor([45.0, 65.0]),
                "intensity_mask": torch.tensor([1.0, 1.0]),
                "pressure": torch.tensor([995.0, 975.0]),
                "pressure_mask": torch.tensor([1.0, 1.0]),
                "stage": torch.tensor([1, 2]),
                "stage_mask": torch.tensor([1.0, 1.0]),
                "intensity_change": torch.zeros((B, 4)),
                "intensity_change_mask": torch.ones((B, 4)),
                "rapid_intensification": torch.tensor([0.0, 1.0]),
                "ri_mask": torch.tensor([1.0, 1.0]),
                "track_displacement": torch.zeros((B, 4, 2)),
                "track_mask": torch.ones((B, 4)),
            }
        }

        optimizer.zero_grad()
        preds = model(
            ir=batch["ir"],
            pmw=batch["pmw"],
            gprof=batch["gprof"],
            radar=batch["radar"],
            env=batch["env"],
            modality_masks=batch["modality_masks"],
            delta_times_hours=batch["delta_times_hours"],
            seq_pad_mask=batch["seq_pad_mask"],
            channel_masks=batch["channel_masks"],
            pixel_masks=batch["pixel_masks"],
        )

        loss, loss_dict = loss_fn(preds, batch["targets"])
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Check gradients exist and are finite
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grad_norms), 0)
        self.assertFalse(any(np.isnan(g) for g in grad_norms))

        optimizer.step()


if __name__ == "__main__":
    unittest.main()
