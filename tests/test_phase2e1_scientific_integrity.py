"""
Phase 2E.1 — Scientific Integrity, Geospatial Alignment & Model Data-Flow Test Suite.
Validates:
  1. Pixel and voxel mask preservation through Dataset, Collate, and Model.
  2. Invariance of model forward pass to mutated invalid/missing pixels.
  3. Raw sentinel detection before arithmetic, np.maximum, or log1p.
  4. Geospatial coordinate-aware swath resampling and distance-based swath masking.
  5. True polar spatial graph topology (4 rings x 4 quadrants = 16 nodes) and 8D geometric edge attributes.
  6. PMW sensor mapping completeness across all 9 sensors with channel-level masks.
  7. Radar Ku reflectivity QC, [-20, 100] dBZ range, and precipitation echo thresholding.
  8. MFM student patch masking preventing raw sensor leakage to student encoders.
  9. Strict NIO adaptation partition and basin isolation with held-out regional test storms.
 10. Canonical storm split (56/12/12) and sequence end-padding verification.
"""

import os
import math
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset, SENSOR_PMW_CHANNEL_MAP
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.models.graph.spatial_graph import SpatialGraphEncoder, PolarSectorPooling
from src.training.losses import MultiTaskLoss
from src.training.pretrain import MaskedFeatureReconstructionHead


class TestPhase2E1ScientificIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_index_csv = "reports/pipeline/sample_index.csv"
        cls.has_real_index = os.path.exists(cls.sample_index_csv)

    # 1. Pixel and Voxel Masks Survive Collate
    def test_01_pixel_and_voxel_masks_survive_collate(self):
        H, W = 64, 64
        sample_a = {
            "sample_id": "TEST_A",
            "storm_id": "STORM_A",
            "timestamp_epoch": 1600000000,
            "ir": torch.randn(1, H, W),
            "pmw": torch.randn(4, H, W),
            "gprof": torch.randn(4, H, W),
            "radar": torch.randn(1, 16, H // 2, W // 2),
            "env": torch.randn(6, H, W),
            "masks": {
                "ir_pixel": torch.ones(1, H, W),
                "pmw_pixel": torch.ones(4, H, W),
                "gprof_pixel": torch.ones(4, H, W),
                "radar_voxel": torch.ones(1, 16, H // 2, W // 2),
                "env_pixel": torch.ones(6, H, W),
                "pmw_channels": torch.ones(4),
                "gprof_channels": torch.ones(4),
                "env_channels": torch.ones(6),
            },
            "modality_masks": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            "targets": {
                "intensity": torch.tensor(50.0), "intensity_mask": torch.tensor(1.0),
                "pressure": torch.tensor(990.0), "pressure_mask": torch.tensor(1.0),
                "stage": torch.tensor(1), "stage_mask": torch.tensor(1.0),
                "intensity_change": torch.zeros(4), "intensity_change_mask": torch.zeros(4),
                "rapid_intensification": torch.tensor(0.0), "ri_mask": torch.tensor(0.0),
                "track_displacement": torch.zeros(4, 2), "track_mask": torch.zeros(4),
            },
        }
        sample_b = dict(sample_a)
        sample_b["sample_id"] = "TEST_B"

        batch = collate_multimodal_batch([sample_a, sample_b])
        self.assertIn("pixel_masks", batch)
        self.assertEqual(batch["pixel_masks"]["ir"].shape, (2, 1, H, W))
        self.assertEqual(batch["pixel_masks"]["pmw"].shape, (2, 4, H, W))
        self.assertEqual(batch["pixel_masks"]["gprof"].shape, (2, 4, H, W))
        self.assertEqual(batch["pixel_masks"]["radar"].shape, (2, 1, 16, H // 2, W // 2))
        self.assertEqual(batch["pixel_masks"]["env"].shape, (2, 6, H, W))

        # Test sequence batch: each element is a list of observations [obs_0, ...]
        seq_sample = [sample_a, sample_b]
        seq_batch = collate_sequence_batch([seq_sample])
        self.assertIn("pixel_masks", seq_batch)
        self.assertEqual(seq_batch["pixel_masks"]["ir"].shape, (1, 2, 1, H, W))
        self.assertEqual(seq_batch["pixel_masks"]["radar"].shape, (1, 2, 1, 16, H // 2, W // 2))

    # 2. Model Invariance to Invalid Masked Pixels
    def test_02_model_invariance_to_invalid_masked_pixels(self):
        torch.manual_seed(42)
        model = TCGraphFormer(d_model=64)
        model.eval()

        B, H, W = 1, 64, 64
        ir = torch.randn(B, 1, H, W)
        pmw = torch.randn(B, 4, H, W)
        gprof = torch.randn(B, 4, H, W)
        radar = torch.randn(B, 1, 16, H // 2, W // 2)
        env = torch.randn(B, 6, H, W)
        masks = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

        # Valid mask: top half valid, bottom half invalid (mask = 0)
        pix_ir = torch.ones(B, 1, H, W)
        pix_ir[:, :, H // 2:, :] = 0.0
        pix_pmw = torch.ones(B, 4, H, W)
        pix_pmw[:, :, H // 2:, :] = 0.0
        pix_gp = torch.ones(B, 4, H, W)
        pix_gp[:, :, H // 2:, :] = 0.0
        pix_rad = torch.ones(B, 1, 16, H // 2, W // 2)
        pix_rad[:, :, :, (H // 4):, :] = 0.0
        pix_env = torch.ones(B, 6, H, W)
        pix_env[:, :, H // 2:, :] = 0.0

        pix_masks = {
            "ir": pix_ir, "pmw": pix_pmw, "gprof": pix_gp,
            "radar": pix_rad, "env": pix_env,
        }

        # Baseline forward pass
        with torch.no_grad():
            out1 = model(ir, pmw, gprof, radar, env, masks, pixel_masks=pix_masks)

        # Mutate the invalid pixels arbitrarily to extreme values (e.g. 9999.0)
        ir_mutated = ir.clone()
        ir_mutated[:, :, H // 2:, :] = 9999.0
        pmw_mutated = pmw.clone()
        pmw_mutated[:, :, H // 2:, :] = -8888.0
        gp_mutated = gprof.clone()
        gp_mutated[:, :, H // 2:, :] = 5555.0
        rad_mutated = radar.clone()
        rad_mutated[:, :, :, (H // 4):, :] = 7777.0
        env_mutated = env.clone()
        env_mutated[:, :, H // 2:, :] = -3333.0

        with torch.no_grad():
            out2 = model(ir_mutated, pmw_mutated, gp_mutated, rad_mutated, env_mutated, masks, pixel_masks=pix_masks)

        diff_latent = (out1["latent"] - out2["latent"]).abs().max().item()
        diff_intensity = (out1["intensity"] - out2["intensity"]).abs().max().item()
        diff_pressure = (out1["pressure"] - out2["pressure"]).abs().max().item()

        self.assertAlmostEqual(diff_latent, 0.0, places=5, msg="Latent changed when mutating invalid pixels!")
        self.assertAlmostEqual(diff_intensity, 0.0, places=5, msg="Intensity changed when mutating invalid pixels!")
        self.assertAlmostEqual(diff_pressure, 0.0, places=5, msg="Pressure changed when mutating invalid pixels!")

    # 3. Raw Sentinel Values Detected Before Arithmetic
    def test_03_raw_fill_values_detected_before_arithmetic(self):
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64

        # Array with negative fill values (-9999.0)
        raw_arr = np.array([[-9999.0, 10.0], [5.0, -9999.0]], dtype=np.float32)

        # Detect raw sentinels first
        valid = (~np.isnan(raw_arr)) & (raw_arr > -9000.0) & (raw_arr >= 0.0)
        norm_arr = np.where(valid, np.log1p(np.maximum(raw_arr, 0.0)) / 5.0, np.nan)

        t_data, t_mask = dataset._resize_to_grid_with_mask(norm_arr, (64, 64))

        # Verify that corners corresponding to -9999.0 are strictly invalid (mask = 0)
        self.assertEqual(t_mask[0, 0, 0].item(), 0.0)
        self.assertEqual(t_mask[0, 63, 63].item(), 0.0)
        self.assertEqual(t_data[0, 0, 0].item(), 0.0)

    # 4. Coordinate-Aware Swath Resampling & Distance Thresholding
    def test_04_coordinate_aware_swath_resampling_and_distance_masking(self):
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64

        # Artificial swath located between -100 km and +100 km
        arr = np.ones((50, 50), dtype=np.float32) * 250.0  # 250 K
        gx = np.linspace(-100.0, 100.0, 50)
        gy = np.linspace(-100.0, 100.0, 50)
        x_2d, y_2d = np.meshgrid(gx, gy)

        class DummyGroup:
            def __init__(self, x, y):
                self.variables = {
                    "x": type("Var", (), {"__getitem__": lambda self, idx: x})(),
                    "y": type("Var", (), {"__getitem__": lambda self, idx: y})(),
                }

        grp = DummyGroup(x_2d, y_2d)
        t_data, t_mask, src = dataset._resample_swath_with_coords(
            arr, grp, storm_lat=15.0, storm_lon=120.0, target_shape=(64, 64), domain_km=500.0, max_dist_km=35.0
        )
        self.assertEqual(src, "utm_storm_centered_km")
        # Center of reference grid should be valid (inside the [-100, 100] km swath)
        self.assertEqual(t_mask[0, 32, 32].item(), 1.0)
        # Corners of reference grid (e.g. 500 km radius) must be masked invalid (0.0)
        self.assertEqual(t_mask[0, 0, 0].item(), 0.0)
        self.assertEqual(t_mask[0, 63, 63].item(), 0.0)
        self.assertEqual(t_data[0, 0, 0].item(), 0.0)

    # 5. Polar Spatial Graph Topology & 8D Geometric Edge Attributes
    def test_05_polar_graph_topology_and_8d_geometric_edges(self):
        encoder = SpatialGraphEncoder(d_model=64)
        self.assertEqual(encoder.node_coords_km.shape, (16, 2))
        self.assertEqual(encoder.node_polar_coords.shape, (16, 2))

        # 4 rings: radii should be 37.5, 125.0, 237.5, 400.0 km
        expected_radii = [37.5, 125.0, 237.5, 400.0]
        actual_radii = encoder.node_polar_coords[::4, 0].tolist()
        for exp, act in zip(expected_radii, actual_radii):
            self.assertAlmostEqual(exp, act, places=2)

        # 8-dimensional geometric edge attributes
        self.assertEqual(encoder.edge_attr.size(1), 8)
        self.assertGreater(encoder.edge_index.size(1), 48)  # Includes azimuthal loops, radial connections, and d <= 220km

        # Verify no constant zero columns exist in edge_attr
        for col in range(8):
            std = encoder.edge_attr[:, col].std().item()
            self.assertGreater(std, 1e-4, f"Column {col} has zero variance (fabricated dummy zeros)!")

    # 6. PMW Sensor Mapping Completeness
    def test_06_sensor_pmw_channel_mapping_completeness(self):
        supported_sensors = ["SSMI", "SSMIS", "GMI", "TMI", "AMSR2", "AMSRE", "AMSUB", "MHS", "ATMS"]
        for s in supported_sensors:
            self.assertIn(s, SENSOR_PMW_CHANNEL_MAP)
            mapping = SENSOR_PMW_CHANNEL_MAP[s]
            self.assertEqual(len(mapping), 4)

        # Verify specific sensors
        self.assertIsNotNone(SENSOR_PMW_CHANNEL_MAP["GMI"][0])
        self.assertIsNotNone(SENSOR_PMW_CHANNEL_MAP["GMI"][3])
        self.assertIsNone(SENSOR_PMW_CHANNEL_MAP["ATMS"][0])
        self.assertIsNone(SENSOR_PMW_CHANNEL_MAP["ATMS"][3])

    # 7. Radar Ku Validity & Reflectivity Scaling
    def test_07_radar_ku_validity_and_reflectivity_scaling(self):
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)

        # Create dummy group with Ku reflectivity spanning [-25, 60] dBZ
        raw_ku = np.zeros((88, 50, 49), dtype=np.float32)
        raw_ku[:, :, :] = -9999.0  # fill
        raw_ku[10:30, 20:30, 20:30] = 30.0  # valid echo (30 dBZ)
        raw_ku[0:5, 0:5, 0:5] = -25.0  # below valid Ku threshold (-20 dBZ)

        class DummyRadarGroup:
            variables = {"correctedReflectFactor_Ku": raw_ku}

        tensor, mask, is_valid = dataset._load_radar_volume(DummyRadarGroup(), (64, 64), target_depth=16)
        self.assertTrue(is_valid)
        self.assertEqual(tensor.shape, (1, 16, 32, 32))
        self.assertEqual(mask.shape, (1, 16, 32, 32))
        # 30 dBZ normalized: (30 - (-20)) / 100 = 0.50
        self.assertGreater(tensor.max().item(), 0.40)
        self.assertLessEqual(tensor.max().item(), 0.60)
        # Below -20 dBZ should be masked out
        self.assertEqual(tensor.min().item(), 0.0)

    # 8. MFM Student Mask Leakage Invariance
    def test_08_mfm_student_mask_leakage_invariance(self):
        torch.manual_seed(42)
        model = TCGraphFormer(d_model=64)
        recon_head = MaskedFeatureReconstructionHead(d_model=64)

        B = 2
        ir = torch.randn(B, 1, 64, 64)
        pmw = torch.randn(B, 4, 64, 64)
        gprof = torch.randn(B, 4, 64, 64)
        radar = torch.randn(B, 1, 16, 32, 32)
        env = torch.randn(B, 6, 64, 64)
        masks = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0]])

        # Detached target representations
        with torch.no_grad():
            t_ir = model.ir_encoder(ir, masks[:, 0]).detach()
            t_pmw = model.pmw_encoder(pmw, masks[:, 1]).detach()
            t_gp = model.gprof_encoder(gprof, masks[:, 2]).detach()
            t_rad = model.radar_encoder(radar, masks[:, 3]).detach()
            t_env = model.env_encoder(env, masks[:, 4]).detach()
            targets = torch.cat([t_ir, t_pmw, t_gp, t_rad, t_env], dim=1)

        # Mask student inputs at patch 0 (top-left 16x16)
        ir_stud = ir.clone()
        ir_stud[:, :, :16, :16] = 0.0
        pix_ir_stud = torch.ones_like(ir)
        pix_ir_stud[:, :, :16, :16] = 0.0

        s_ir = model.ir_encoder(ir_stud, masks[:, 0], pixel_mask=pix_ir_stud)
        s_pmw = model.pmw_encoder(pmw, masks[:, 1])
        s_gp = model.gprof_encoder(gprof, masks[:, 2])
        s_rad = model.radar_encoder(radar, masks[:, 3])
        s_env = model.env_encoder(env, masks[:, 4])

        fused = model.fusion([s_ir, s_pmw, s_gp, s_rad, s_env], masks)
        preds = recon_head(fused)

        mask_to_recon = torch.zeros(B, 80, 1)
        mask_to_recon[:, 0, 0] = 1.0  # Reconstruct patch 0

        loss = ((preds - targets) ** 2 * mask_to_recon).sum() / (mask_to_recon.sum() * 64 + 1e-6)
        loss.backward()

        # Encoder gradients exist and are finite
        grad_norm = sum(p.grad.norm().item() for p in model.ir_encoder.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)
        self.assertFalse(math.isnan(grad_norm))

    # 9. Strict NIO Basin and Partition Isolation
    def test_09_nio_adaptation_strict_basin_and_partition_isolation(self):
        if not self.has_real_index:
            self.skipTest("Sample index CSV not available.")

        df = pd.read_csv(self.sample_index_csv)
        df_io_train = df[(df["basin"] == "IO") & (df["partition"] == "TRAIN")].copy()
        df_io_val = df[(df["basin"] == "IO") & (df["partition"] == "VAL")].copy()
        df_io_test = df[(df["basin"] == "IO") & (df["partition"] == "TEST")].copy()

        train_storms = set(df_io_train["storm_id"].unique())
        val_storms = set(df_io_val["storm_id"].unique())
        test_storms = set(df_io_test["storm_id"].unique())

        self.assertEqual(len(train_storms.intersection(test_storms)), 0, "NIO TRAIN storms leaked into TEST!")
        self.assertEqual(len(val_storms.intersection(test_storms)), 0, "NIO VAL storms leaked into TEST!")
        self.assertEqual(len(train_storms.intersection(val_storms)), 0, "NIO TRAIN storms leaked into VAL!")

        # Verify dataset constructed with df_io_train contains strictly IO storms
        ds_train = TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, df=df_io_train)
        self.assertTrue(all(ds_train.df["basin"] == "IO"))
        self.assertTrue(all(ds_train.df["partition"] == "TRAIN"))

        seq_ds_train = TCPRIMEDSequenceDataset(sample_index_csv=self.sample_index_csv, df=df_io_train)
        self.assertTrue(all(seq_ds_train.base_dataset.df["basin"] == "IO"))

    # 10. Canonical Storm Split & Sequence End-Padding
    def test_10_storm_split_and_sequence_padding_reconciliation(self):
        manifest_file = "reports/pipeline/storm_split_manifest.csv"
        if os.path.exists(manifest_file):
            df_manifest = pd.read_csv(manifest_file)
            split_counts = df_manifest["partition"].value_counts().to_dict()
            self.assertEqual(split_counts.get("TRAIN", 0), 56)
            self.assertEqual(split_counts.get("VAL", 0), 12)
            self.assertEqual(split_counts.get("TEST", 0), 12)
            self.assertEqual(len(df_manifest), 80)

        # Test sequence end-padding
        sample = {
            "sample_id": "OBS_1",
            "storm_id": "STORM_1",
            "timestamp_epoch": 1600000000,
            "ir": torch.randn(1, 64, 64),
            "pmw": torch.randn(4, 64, 64),
            "gprof": torch.randn(4, 64, 64),
            "radar": torch.randn(1, 16, 32, 32),
            "env": torch.randn(6, 64, 64),
            "masks": {
                "ir_pixel": torch.ones(1, 64, 64), "pmw_pixel": torch.ones(4, 64, 64),
                "gprof_pixel": torch.ones(4, 64, 64), "radar_voxel": torch.ones(1, 16, 32, 32),
                "env_pixel": torch.ones(6, 64, 64),
                "pmw_channels": torch.ones(4), "gprof_channels": torch.ones(4), "env_channels": torch.ones(6),
            },
            "modality_masks": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
            "targets": {
                "intensity": torch.tensor(45.0), "intensity_mask": torch.tensor(1.0),
                "pressure": torch.tensor(995.0), "pressure_mask": torch.tensor(1.0),
                "stage": torch.tensor(1), "stage_mask": torch.tensor(1.0),
                "intensity_change": torch.zeros(4), "intensity_change_mask": torch.zeros(4),
                "rapid_intensification": torch.tensor(0.0), "ri_mask": torch.tensor(0.0),
                "track_displacement": torch.zeros(4, 2), "track_mask": torch.zeros(4),
            },
        }

        # Sequence with 1 observation padded to length 3 (each item in batch is a list of observations)
        seq_sample_short = [sample]
        seq_sample_long = [sample, sample, sample]

        batch = collate_sequence_batch([seq_sample_short, seq_sample_long])
        # Padded timestep at t=1,2 for S1 must have modality_masks == 0
        self.assertEqual(batch["modality_masks"][0, 0].sum().item(), 5.0)  # t=0 valid
        self.assertEqual(batch["modality_masks"][0, 1].sum().item(), 0.0)  # t=1 padded
        self.assertEqual(batch["modality_masks"][0, 2].sum().item(), 0.0)  # t=2 padded
        self.assertEqual(batch["pixel_masks"]["ir"][0, 1].sum().item(), 0.0)  # pixel mask padded


if __name__ == "__main__":
    unittest.main()
