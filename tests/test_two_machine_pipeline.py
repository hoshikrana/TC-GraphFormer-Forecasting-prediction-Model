"""
Unit and Integration Tests for Two-Machine Pipeline, TC-GraphFormer, and Staging.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Ensure root directory on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.infrastructure.hardware_info import discover_hardware
from src.infrastructure.export_shards import compute_sha256
from src.data.dataset import TCPRIMEDDataset, STAGE_MAPPING
from src.data.collate import collate_multimodal_batch
from src.models.tc_graphformer import TCGraphFormer
from src.models.encoders.ir_encoder import IREncoder
from src.models.encoders.pmw_encoder import PMWEncoder
from src.models.fusion.cross_modal_fusion import CrossModalFusion
from src.models.graph.spatial_graph import SpatialGraphEncoder
from src.models.temporal.temporal_transformer import CausalTemporalTransformer
from src.training.losses import MultiTaskLoss


class TestInfrastructure(unittest.TestCase):
    def test_hardware_discovery(self):
        inv = discover_hardware("workstation")
        self.assertIn("compute_profile", inv)
        self.assertIn("operating_system", inv)
        self.assertIn("cpu_and_memory", inv)
        self.assertIn("storage_drives", inv)
        self.assertIn("python_environment", inv)
        self.assertIn("gpu_acceleration", inv)
        self.assertEqual(inv["compute_profile"]["role"], "workstation")
        self.assertIsInstance(inv["cpu_and_memory"]["logical_cpu_cores"], int)

    def test_sha256_computation(self):
        test_file = "reports/pipeline/storm_split_manifest.csv"
        if os.path.exists(test_file):
            h1 = compute_sha256(test_file)
            h2 = compute_sha256(test_file)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)


class TestDatasetAndCollate(unittest.TestCase):
    def test_dataset_sample(self):
        index_file = "reports/pipeline/sample_index.csv"
        if os.path.exists(index_file):
            ds = TCPRIMEDDataset(sample_index_csv=index_file, partition="TRAIN", max_samples=4, grid_size=64)
            self.assertEqual(len(ds), 4)
            sample = ds[0]
            self.assertIn("ir", sample)
            self.assertIn("pmw", sample)
            self.assertIn("modality_masks", sample)
            self.assertIn("targets", sample)
            self.assertEqual(sample["ir"].shape, (1, 64, 64))
            self.assertEqual(sample["pmw"].shape, (4, 64, 64))
            self.assertEqual(sample["modality_masks"].shape, (5,))

    def test_collate_batch(self):
        index_file = "reports/pipeline/sample_index.csv"
        if os.path.exists(index_file):
            ds = TCPRIMEDDataset(sample_index_csv=index_file, partition="TRAIN", max_samples=4, grid_size=64)
            batch = collate_multimodal_batch([ds[0], ds[1]])
            self.assertEqual(batch["ir"].shape, (2, 1, 64, 64))
            self.assertEqual(batch["pmw"].shape, (2, 4, 64, 64))
            self.assertEqual(batch["modality_masks"].shape, (2, 5))
            self.assertIn("intensity", batch["targets"])
            self.assertEqual(batch["targets"]["intensity"].shape, (2,))


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.model = TCGraphFormer(d_model=self.d_model)

    def test_forward_pass_shapes(self):
        B = 2
        ir = torch.randn(B, 1, 64, 64)
        pmw = torch.randn(B, 4, 64, 64)
        gprof = torch.randn(B, 1, 64, 64)
        radar = torch.randn(B, 1, 16, 32, 32)
        env = torch.randn(B, 1, 64, 64)
        masks = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]])

        preds = self.model(ir, pmw, gprof, radar, env, masks)
        self.assertEqual(preds["intensity"].shape, (B,))
        self.assertEqual(preds["pressure"].shape, (B,))
        self.assertEqual(preds["stage_logits"].shape, (B, self.model.task_heads.num_stages))
        self.assertEqual(preds["intensity_change"].shape, (B, 4))
        self.assertEqual(preds["ri_logits"].shape, (B,))
        self.assertEqual(preds["track_delta"].shape, (B, 8))
        self.assertEqual(preds["latent"].shape, (B, self.d_model))

    def test_masked_modality_zero_leakage(self):
        # Mask out PMW completely and verify no NaNs or crash
        B = 1
        ir = torch.randn(B, 1, 64, 64)
        pmw = torch.zeros(B, 4, 64, 64)
        gprof = torch.zeros(B, 1, 64, 64)
        radar = torch.zeros(B, 1, 16, 32, 32)
        env = torch.randn(B, 1, 64, 64)
        masks = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]])

        preds = self.model(ir, pmw, gprof, radar, env, masks)
        self.assertFalse(torch.isnan(preds["intensity"]).any())
        self.assertFalse(torch.isnan(preds["pressure"]).any())

    def test_loss_computation(self):
        loss_fn = MultiTaskLoss()
        preds = {
            "intensity": torch.tensor([45.0, 65.0]),
            "pressure": torch.tensor([990.0, 970.0]),
            "stage_logits": torch.randn(2, 5),
            "intensity_change": torch.tensor([[5.0, 10.0, 15.0, 20.0], [0.0, 0.0, 0.0, 0.0]]),
            "ri_logits": torch.tensor([-1.5, 2.1]),
        }
        targets = {
            "intensity": torch.tensor([50.0, 60.0]),
            "pressure": torch.tensor([985.0, 975.0]),
            "stage": torch.tensor([1, 2]),
            "intensity_change": torch.tensor([[5.0, 10.0, 10.0, 25.0], [0.0, 0.0, 0.0, 0.0]]),
            "intensity_change_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]),
            "rapid_intensification": torch.tensor([0.0, 1.0]),
            "ri_mask": torch.tensor([1.0, 1.0]),
        }
        loss, loss_dict = loss_fn(preds, targets)
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_intensity", loss_dict)
        self.assertIn("loss_pressure", loss_dict)
        self.assertIn("loss_stage", loss_dict)
        self.assertIn("loss_intensity_change", loss_dict)
        self.assertIn("loss_ri", loss_dict)


if __name__ == "__main__":
    unittest.main()

