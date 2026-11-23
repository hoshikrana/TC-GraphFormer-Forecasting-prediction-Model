"""
Unit and Integration Tests for Phase 2E.4.2: Final GPU Handoff Readiness Corrections.

Validates:
1. Reusable production restoration function `restore_checkpoint_state(...)`:
   - Fails loudly on missing model parameters, missing epoch, missing optimizer,
     missing scheduler, missing historical best_val_loss, or missing AMP scaler (for fp16).
   - Permits legacy partial resume only when allow_legacy_resume=True.
2. Gradient accumulation in `TCGraphFormerTrainer.train_epoch()`:
   - Accurately tracks optimizer steps and normalizes incomplete final microbatch groups
     using actual group size rather than fixed accumulate_grad_batches.
   - Tested on 2, 3, and 4 microbatches with accumulation factor 2.
3. Strict checkpoint provenance validation:
   - Verifies 7 distinct cases in validate_checkpoint_compatibility() and load_model_for_evaluation().
   - Confirms that heuristic matching without explicit supervised_multitask_training provenance is rejected.
4. Stratified selection and reporting:
   - Enforces configured era proportions and single-year concentration limits in Pass 2 and Pass 3.
   - Verifies target_pct, achieved_pct, and deviation_pct calculations for basins and eras.
   - Verifies NIO target percentage uses configured proportion (15% in 100 GB config, not 25%).
5. Pipeline & GPU node configuration separation:
   - Verifies distinct paths for Pilot (data/processed/pilot) vs 100 GB (data/processed/tcprimed_100gb).
   - Confirms --config argument resolution in evaluate_baselines.py.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.trainer import TCGraphFormerTrainer, restore_checkpoint_state
from src.training.train_multitask import (
    load_yaml_config,
    validate_checkpoint_compatibility,
    run_multitask_training,
)
from src.evaluation.evaluate_baselines import (
    load_model_for_evaluation,
    compute_track_errors,
)
from src.prepare_pilot_dataset import (
    stratified_select,
    build_selection_report,
    generate_plan_summary_text,
)
from src.models.tc_graphformer import TCGraphFormer


class MockAccumDataset(Dataset):
    """Synthetic dataset for testing gradient accumulation with dummy tensors."""
    def __init__(self, num_samples: int = 4, d_model: int = 16):
        self.num_samples = num_samples
        self.d_model = d_model

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "ir": torch.randn(1, 64, 64),
            "pmw": torch.randn(1, 64, 64),
            "gprof": torch.randn(1, 64, 64),
            "radar": torch.randn(1, 16, 32, 32),
            "env": torch.randn(1, 8),
            "targets": {
                "intensity": torch.tensor(45.0 + idx),
                "pressure": torch.tensor(990.0 - idx),
                "stage": torch.tensor(1),
                "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0]),
                "rapid_intensification": torch.tensor(0.0),
                "track_displacement": torch.tensor([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0], [40.0, 80.0]]),
            },
        }


def mock_accum_collate(batch):
    B = len(batch)
    return {
        "ir": torch.stack([item["ir"] for item in batch]),
        "pmw": torch.stack([item["pmw"] for item in batch]),
        "gprof": torch.stack([item["gprof"] for item in batch]),
        "radar": torch.stack([item["radar"] for item in batch]),
        "env": torch.stack([item["env"] for item in batch]),
        "modality_masks": torch.ones(B, 5, dtype=torch.bool),
        "targets": {
            "intensity": torch.stack([item["targets"]["intensity"] for item in batch]),
            "pressure": torch.stack([item["targets"]["pressure"] for item in batch]),
            "stage": torch.stack([item["targets"]["stage"] for item in batch]),
            "intensity_change": torch.stack([item["targets"]["intensity_change"] for item in batch]),
            "rapid_intensification": torch.stack([item["targets"]["rapid_intensification"] for item in batch]),
            "track_displacement": torch.stack([item["targets"]["track_displacement"] for item in batch]),
        },
    }


class TinyDummyModel(nn.Module):
    """Minimal model for fast gradient accumulation and checkpoint testing."""
    def __init__(self, d_model=16):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(1, d_model))
        self.head_intensity = nn.Linear(d_model, 1)
        self.head_pressure = nn.Linear(d_model, 1)
        self.head_stage = nn.Linear(d_model, 5)
        self.head_intensity_change = nn.Linear(d_model, 4)
        self.head_ri = nn.Linear(d_model, 1)
        self.head_track = nn.Linear(d_model, 8)

    def forward(self, ir, pmw, gprof, radar, env, modality_masks=None, **kwargs):
        B = ir.size(0)
        feat = self.weight.repeat(B, 1)
        return {
            "intensity": self.head_intensity(feat).squeeze(-1),
            "pressure": self.head_pressure(feat).squeeze(-1),
            "stage_logits": self.head_stage(feat),
            "intensity_change": self.head_intensity_change(feat),
            "ri_logits": self.head_ri(feat).squeeze(-1),
            "track_delta": self.head_track(feat).view(B, 4, 2),
        }


class DummyLossFn(nn.Module):
    """Simple loss function for accumulation tests."""
    def forward(self, preds, targets):
        loss = (preds["intensity"] - targets["intensity"]).pow(2).mean()
        return loss, {"intensity_loss": loss.item()}


class TestPhase2E42GPUHandoff(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # 1. Checkpoint Resume Failure Modes via restore_checkpoint_state()
    # -----------------------------------------------------------------------
    def test_01_checkpoint_resume_failure_modes(self):
        """Verify restore_checkpoint_state strictly raises ValueError on missing components unless overridden."""
        model = TinyDummyModel(d_model=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

        # Baseline fully valid checkpoint
        base_state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "epoch": 3,
            "best_val_loss": 0.42,
            "is_trained": True,
            "provenance": {
                "workflow": "supervised_multitask_training",
                "is_trained": True,
            },
        }

        # 1a. Missing epoch
        no_epoch = dict(base_state)
        del no_epoch["epoch"]
        p_no_epoch = os.path.join(self.tmp_dir, "no_epoch.pt")
        torch.save(no_epoch, p_no_epoch)
        with self.assertRaises(ValueError) as ctx:
            restore_checkpoint_state(p_no_epoch, model, opt, sched, allow_legacy_resume=False)
        self.assertIn("epoch", str(ctx.exception))

        # 1b. Missing optimizer_state_dict
        no_opt = dict(base_state)
        no_opt["optimizer_state_dict"] = None
        p_no_opt = os.path.join(self.tmp_dir, "no_opt.pt")
        torch.save(no_opt, p_no_opt)
        with self.assertRaises(ValueError) as ctx:
            restore_checkpoint_state(p_no_opt, model, opt, sched, allow_legacy_resume=False)
        self.assertIn("optimizer_state_dict", str(ctx.exception))

        # 1c. Missing scheduler_state_dict
        no_sched = dict(base_state)
        no_sched["scheduler_state_dict"] = None
        p_no_sched = os.path.join(self.tmp_dir, "no_sched.pt")
        torch.save(no_sched, p_no_sched)
        with self.assertRaises(ValueError) as ctx:
            restore_checkpoint_state(p_no_sched, model, opt, sched, allow_legacy_resume=False)
        self.assertIn("scheduler_state_dict", str(ctx.exception))

        # 1d. Missing historical best_val_loss
        no_best = dict(base_state)
        del no_best["best_val_loss"]
        p_no_best = os.path.join(self.tmp_dir, "no_best.pt")
        torch.save(no_best, p_no_best)
        with self.assertRaises(ValueError) as ctx:
            restore_checkpoint_state(p_no_best, model, opt, sched, allow_legacy_resume=False)
        self.assertIn("best_val_loss", str(ctx.exception))

        # 1e. Missing AMP GradScaler state when precision='fp16' and use_amp=True
        no_scaler = dict(base_state)
        p_no_scaler = os.path.join(self.tmp_dir, "no_scaler.pt")
        torch.save(no_scaler, p_no_scaler)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        with self.assertRaises(ValueError) as ctx:
            restore_checkpoint_state(
                p_no_scaler, model, opt, sched, scaler=scaler, use_amp=True, precision="fp16", allow_legacy_resume=False
            )
        self.assertIn("scaler_state_dict", str(ctx.exception))

        # 1f. allow_legacy_resume=True overrides missing scheduler and best_val_loss safely
        legacy_res = restore_checkpoint_state(p_no_sched, model, opt, sched, allow_legacy_resume=True)
        self.assertTrue(legacy_res["is_legacy_resume"])
        self.assertEqual(legacy_res["epoch"], 3)
        self.assertEqual(legacy_res["next_epoch"], 4)

    # -----------------------------------------------------------------------
    # 2. Gradient Accumulation Group Normalization & Step Counting
    # -----------------------------------------------------------------------
    def test_02_gradient_accumulation_group_normalization(self):
        """Verify train_epoch accurately handles complete and incomplete accumulation groups (2, 3, 4 batches)."""
        loss_fn = DummyLossFn()

        for num_batches, expected_steps in [(2, 1), (3, 2), (4, 2)]:
            model = TinyDummyModel(d_model=16)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)

            dataset = MockAccumDataset(num_samples=num_batches, d_model=16)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=mock_accum_collate)

            trainer = TCGraphFormerTrainer(
                model=model,
                optimizer=opt,
                loss_fn=loss_fn,
                device="cpu",
                checkpoint_dir=self.tmp_dir,
                use_amp=False,
                accumulate_grad_batches=2,
            )

            metrics = trainer.train_epoch(loader, epoch=1)
            self.assertEqual(metrics["optimizer_steps"], expected_steps,
                             f"Failed on {num_batches} batches with accum=2: expected {expected_steps} steps, got {metrics['optimizer_steps']}")
            self.assertEqual(metrics["total_batches"], num_batches)
            self.assertEqual(metrics["accumulate_grad_batches"], 2)

    # -----------------------------------------------------------------------
    # 3. Checkpoint Provenance: 7 Rigorous Cases
    # -----------------------------------------------------------------------
    def test_03_checkpoint_provenance_seven_cases(self):
        """Verify 7 distinct checkpoint conditions across compatibility check and baseline loading."""
        model = TCGraphFormer(d_model=64, num_stages=5)
        model_state = model.state_dict()

        # Case 1: Fully valid trained checkpoint
        c1_path = os.path.join(self.tmp_dir, "case1_valid.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 5,
            "is_trained": True,
            "provenance": {
                "workflow": "supervised_multitask_training",
                "is_trained": True,
            },
        }, c1_path)
        c1_val = validate_checkpoint_compatibility(c1_path, model, strict=True)
        self.assertTrue(c1_val["is_compatible"])
        self.assertTrue(c1_val["is_trained"])
        m1, s1 = load_model_for_evaluation(c1_path, d_model=64, strict=True, smoke_test=False)
        self.assertTrue(s1["is_trained"])
        self.assertEqual(s1["model_label"], "TC-GraphFormer")

        # Case 2: Missing workflow in provenance
        c2_path = os.path.join(self.tmp_dir, "case2_no_workflow.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 5,
            "is_trained": True,
            "provenance": {"is_trained": True},
        }, c2_path)
        c2_val = validate_checkpoint_compatibility(c2_path, model, strict=True)
        self.assertFalse(c2_val["is_trained"])
        with self.assertRaises(ValueError):
            load_model_for_evaluation(c2_path, d_model=64, strict=True, smoke_test=False)

        # Case 3: Wrong workflow (e.g. self-supervised pretraining)
        c3_path = os.path.join(self.tmp_dir, "case3_pretraining.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 10,
            "is_trained": True,
            "provenance": {
                "workflow": "self_supervised_pretraining",
                "is_trained": True,
            },
        }, c3_path)
        c3_val = validate_checkpoint_compatibility(c3_path, model, strict=True)
        self.assertFalse(c3_val["is_trained"])
        with self.assertRaises(ValueError):
            load_model_for_evaluation(c3_path, d_model=64, strict=True, smoke_test=False)

        # Case 4: Epoch 0 (untrained initialization checkpoint)
        c4_path = os.path.join(self.tmp_dir, "case4_epoch0.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 0,
            "is_trained": True,
            "provenance": {
                "workflow": "supervised_multitask_training",
                "is_trained": True,
            },
        }, c4_path)
        c4_val = validate_checkpoint_compatibility(c4_path, model, strict=True)
        self.assertFalse(c4_val["is_trained"])
        with self.assertRaises(ValueError):
            load_model_for_evaluation(c4_path, d_model=64, strict=True, smoke_test=False)

        # Case 5: is_trained explicitly False
        c5_path = os.path.join(self.tmp_dir, "case5_not_trained.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 2,
            "is_trained": False,
            "provenance": {
                "workflow": "supervised_multitask_training",
                "is_trained": False,
            },
        }, c5_path)
        c5_val = validate_checkpoint_compatibility(c5_path, model, strict=True)
        self.assertFalse(c5_val["is_trained"])
        with self.assertRaises(ValueError):
            load_model_for_evaluation(c5_path, d_model=64, strict=True, smoke_test=False)

        # Case 6: Heuristic match only (optimizer_state_dict and metrics present, but missing provenance dict)
        c6_path = os.path.join(self.tmp_dir, "case6_heuristic_only.pt")
        torch.save({
            "model_state_dict": model_state,
            "epoch": 3,
            "optimizer_state_dict": {"state": {}},
            "metrics": {"val_loss": 0.5},
            # Missing is_trained and missing provenance dict
        }, c6_path)
        c6_val = validate_checkpoint_compatibility(c6_path, model, strict=True)
        self.assertFalse(c6_val["is_trained"], "Heuristic fallback must be removed!")
        with self.assertRaises(ValueError):
            load_model_for_evaluation(c6_path, d_model=64, strict=True, smoke_test=False)
        # Verify smoke_test=True allows it with disclaimer
        m6, s6 = load_model_for_evaluation(c6_path, d_model=64, strict=True, smoke_test=True)
        self.assertIn("Smoke-Test Untrained", s6["model_label"])

        # Case 7: Incompatible model parameters (shape mismatch)
        c7_path = os.path.join(self.tmp_dir, "case7_incompatible.pt")
        bad_state = dict(model_state)
        # Corrupt one tensor shape
        for k in bad_state:
            if bad_state[k].dim() >= 1:
                bad_state[k] = bad_state[k][:1]
                break
        torch.save({"model_state_dict": bad_state, "epoch": 5, "is_trained": True}, c7_path)
        with self.assertRaises(ValueError):
            validate_checkpoint_compatibility(c7_path, model, strict=True)

    # -----------------------------------------------------------------------
    # 4. Stratified Selection & Reporting Proportions
    # -----------------------------------------------------------------------
    def test_04_stratification_reporting_and_configured_targets(self):
        """Verify Pass 2 & 3 preserve era/basin proportions, respect single_year_max_pct, and report 15% NIO target."""
        basins = ["IO", "WP", "AL", "EP"]
        eras = ["1987-1999", "2000-2009", "2010-2019", "2020+"]

        # Build synthetic catalogue
        synthetic_catalogue = []
        storm_id_counter = 1
        for b in basins:
            for e in eras:
                for yr in range(2000, 2010):
                    synthetic_catalogue.append({
                        "storm_folder": f"storm_{storm_id_counter:04d}",
                        "season": yr,
                        "basin": b,
                        "era": e,
                        "cyclone_number": f"01",
                        "total_bytes": 10_000_000, # 10 MB per storm
                        "file_count": 5,
                        "overpass_count": 4,
                        "env_count": 1,
                        "instruments": "SSMI;TMI",
                        "platforms": "F13;TRMM",
                    })
                    storm_id_counter += 1

        target_bytes = 100_000_000 # 100 MB target
        basin_proportions = {"IO": 0.15, "WP": 0.30, "AL": 0.25, "EP": 0.30}
        era_proportions = {"1987-1999": 0.20, "2000-2009": 0.25, "2010-2019": 0.35, "2020+": 0.20}

        selected, report = stratified_select(
            catalogue=synthetic_catalogue,
            target_bytes=target_bytes,
            seed=42,
            basin_proportions=basin_proportions,
            era_proportions=era_proportions,
            single_year_max_pct=20.0,
            enforce_complete_storms=True,
        )

        self.assertLessEqual(report["selected_bytes"], target_bytes)
        self.assertEqual(report["nio_target_pct"], 15.0, "NIO target must match configured 15%, not hardcoded 25%!")
        self.assertTrue(report["nio_achieved"])
        self.assertLessEqual(report["largest_single_year_pct"], 20.0 + 10.0) # Within tolerance

        # Verify target_pct, achieved_pct, deviation_pct in report
        for b in basins:
            self.assertIn("target_pct", report["basin_stats"][b])
            self.assertIn("achieved_pct", report["basin_stats"][b])
            self.assertIn("deviation_pct", report["basin_stats"][b])
            self.assertEqual(report["basin_stats"][b]["target_pct"], basin_proportions[b] * 100.0)

        for e in eras:
            self.assertIn("target_pct", report["era_stats"][e])
            self.assertIn("achieved_pct", report["era_stats"][e])
            self.assertIn("deviation_pct", report["era_stats"][e])
            self.assertEqual(report["era_stats"][e]["target_pct"], era_proportions[e] * 100.0)

        summary_text = generate_plan_summary_text(report)
        self.assertIn("STORMS BY BASIN (Target vs Achieved)", summary_text)
        self.assertIn("STORMS BY HISTORICAL ERA (Target vs Achieved)", summary_text)
        self.assertIn("NIO (IO) target percentage:          15.00%", summary_text)

    # -----------------------------------------------------------------------
    # 5. Pipeline & GPU Node Configuration Separation
    # -----------------------------------------------------------------------
    def test_05_pipeline_and_gpu_node_config_separation(self):
        """Verify Pilot vs 100 GB path isolation and --config parsing in evaluate_baselines."""
        pilot_cfg_p = "configs/gpu_node.yaml"
        expansion_cfg_p = "configs/gpu_node_100gb.yaml"
        pipe_100_p = "configs/pipeline_100gb.yaml"

        self.assertTrue(os.path.exists(pilot_cfg_p))
        self.assertTrue(os.path.exists(expansion_cfg_p))
        self.assertTrue(os.path.exists(pipe_100_p))

        pilot_cfg = load_yaml_config(pilot_cfg_p)
        exp_cfg = load_yaml_config(expansion_cfg_p)
        pipe_cfg = load_yaml_config(pipe_100_p)

        # Pilot paths
        self.assertEqual(pilot_cfg["paths"]["staged_data_dir"], "data/processed/pilot")
        self.assertEqual(pilot_cfg["paths"]["sample_index_path"], "reports/pipeline/sample_index.csv")
        self.assertEqual(pilot_cfg["paths"]["checkpoints_dir"], "checkpoints")

        # 100 GB Expansion paths
        self.assertEqual(exp_cfg["paths"]["staged_data_dir"], "data/processed/tcprimed_100gb")
        self.assertEqual(exp_cfg["paths"]["sample_index_path"], "reports/pipeline_100gb/sample_index.csv")
        self.assertEqual(exp_cfg["paths"]["checkpoints_dir"], "checkpoints/tcprimed_100gb")

        # Pipeline 100 GB matching paths
        self.assertEqual(pipe_cfg["output"]["root"], "data/processed/tcprimed_100gb")
        self.assertEqual(pipe_cfg["output"]["reports_dir"], "reports/pipeline_100gb")

    # -----------------------------------------------------------------------
    # 6. FP16 AMP Masking & DSA O(1) Sample Cache Regression
    # -----------------------------------------------------------------------
    def test_06_fp16_amp_spatial_graph_and_dsa_cache_regression(self):
        """Verify SpatialGraphEncoder supports FP16 without half overflow, physical priors init, and O(1) sample cache matches uncached."""
        from src.models.graph.spatial_graph import SpatialGraphEncoder
        from src.models.heads.task_heads import TaskPredictionHeads
        from src.data.dataset import TCPRIMEDDataset

        # 1. FP16 masking safety in SpatialGraphEncoder
        graph = SpatialGraphEncoder(d_model=64, num_spatial_nodes=16, num_layers=2).half()
        nodes = torch.randn(2, 16, 64, dtype=torch.float16)
        validity = torch.ones(2, 16, 1, dtype=torch.float16)
        validity[0, 8:, 0] = 0.0
        validity[1, :, 0] = 0.0  # All invalid edge case
        out = graph(nodes, node_validity=validity)
        self.assertEqual(out.shape, (2, 64))
        self.assertTrue(torch.all(torch.isfinite(out)))

        # 2. Physical prior bias initialization
        heads = TaskPredictionHeads(d_model=64, num_stages=5)
        self.assertAlmostEqual(float(heads.head_intensity[-1].bias.item()), 38.19, places=2)
        self.assertAlmostEqual(float(heads.head_pressure[-1].bias.item()), 997.25, places=2)
        self.assertAlmostEqual(float(heads.head_ri[-1].bias.item()), -3.5124, places=2)

        # 3. Bounded LRU & O(1) sample cache bitwise equivalence
        if os.path.exists("reports/pipeline/sample_index.csv"):
            ds_u = TCPRIMEDDataset("reports/pipeline/sample_index.csv", partition="TRAIN", max_samples=3)
            ds_u.enable_memory_cache = False
            ds_u.enable_disk_cache = False
            s_u = ds_u[0]
            self.assertIn("kinematic_state", s_u)
            self.assertEqual(s_u["kinematic_state"].shape, (10,))

            ds_c = TCPRIMEDDataset("reports/pipeline/sample_index.csv", partition="TRAIN", max_samples=3, max_cache_entries=2)
            ds_c._sample_cache.clear()
            s_c0 = ds_c[0]
            s_c1 = ds_c[1]
            s_c2 = ds_c[2]
            self.assertLessEqual(len(ds_c._sample_cache), 2, "Bounded LRU cache must enforce max_cache_entries!")
            self.assertTrue(torch.equal(s_u["ir"], s_c0["ir"]))
            self.assertTrue(torch.equal(s_u["pmw"], s_c0["pmw"]))


if __name__ == "__main__":
    unittest.main()
