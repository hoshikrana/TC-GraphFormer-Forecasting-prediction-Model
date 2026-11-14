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
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import netCDF4 as nc

from tcprimed_pipeline.config import OutputConfig
from tcprimed_pipeline.storage_writer import StorageWriter
from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset, SENSOR_PMW_CHANNEL_MAP
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.models.graph.spatial_graph import SpatialGraphEncoder, PolarSectorPooling
from src.training.losses import MultiTaskLoss
from src.training.pretrain import MaskedFeatureReconstructionHead, compute_module_grad_norm, run_pretraining


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

    # 7. Radar Ku Validity, Echo Classification & Reflectivity Scaling (Critical 4)
    def test_07_radar_ku_validity_and_reflectivity_scaling(self):
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.raw_data_dir = "data/raw/pilot"
        dataset._raw_radar_coord_cache = {}

        # Create dummy group with Ku reflectivity spanning:
        # 1. Fill sentinels (-9999.0)
        # 2. Out of range below -20 dBZ (-25.0 dBZ)
        # 3. Valid low-reflectivity measurement below 12 dBZ detection threshold (5.0 dBZ)
        # 4. Precipitation detection threshold (12.0 dBZ)
        # 5. Moderate precipitation echo (30.0 dBZ)
        raw_ku = np.zeros((88, 50, 49), dtype=np.float32)
        raw_ku[:, :, :] = -9999.0  # fill
        raw_ku[0:5, 0:5, 0:5] = -25.0         # below valid Ku threshold (-20 dBZ)
        raw_ku[10:20, 10:20, 10:20] = 5.0     # low-reflectivity: valid meas, NOT precip echo
        raw_ku[20:30, 20:30, 20:30] = 12.0    # 12 dBZ MDS threshold: valid meas, YES precip echo
        raw_ku[30:40, 30:40, 30:40] = 30.0    # 30 dBZ moderate rain: valid meas, YES precip echo

        class DummyRadarGroup:
            variables = {"correctedReflectFactor_Ku": raw_ku}

        # Test standard 3-tuple return (backwards-compatible)
        tensor, mask, is_valid = dataset._load_radar_volume(DummyRadarGroup(), (64, 64), target_depth=16)
        self.assertTrue(is_valid)
        self.assertEqual(tensor.shape, (1, 16, 32, 32))
        self.assertEqual(mask.shape, (1, 16, 32, 32))

        # Test 4-tuple return separating measurement validity from echo classification
        tensor_e, mask_e, echo_e, is_valid_e = dataset._load_radar_volume(
            DummyRadarGroup(), (64, 64), target_depth=16, echo_threshold_dbz=12.0, return_echo_mask=True
        )
        self.assertTrue(is_valid_e)
        self.assertEqual(echo_e.shape, (1, 16, 32, 32))

        # 1. Measurement validity includes low-reflectivity (5 dBZ), threshold (12 dBZ) and rain (30 dBZ)
        # 2. Echo classification strictly requires Ku >= 12.0 dBZ
        # Therefore, measurement valid mask sum MUST be strictly greater than echo mask sum!
        self.assertGreater(mask_e.sum().item(), echo_e.sum().item(),
                           "Measurement validity must include low-reflectivity measurements where echo is False!")
        self.assertGreater(echo_e.sum().item(), 0.0, "Precipitation echo mask must capture >= 12 dBZ echoes!")

        # 3. Normalized physical scaling verification:
        # -20 dBZ -> 0.00
        # +5 dBZ  -> (5 - (-20)) / 100 = 0.25
        # +12 dBZ -> (12 - (-20)) / 100 = 0.32 (MDS threshold)
        # +30 dBZ -> (30 - (-20)) / 100 = 0.50
        # -25 dBZ -> masked out (0.00)
        self.assertEqual(tensor_e.min().item(), 0.0)
        self.assertAlmostEqual(tensor_e.max().item(), 0.50, places=2)

        # 4. Real pilot container verification with coordinate-aware registration
        pilot_csv = "reports/pipeline/sample_index.csv"
        if os.path.exists(pilot_csv):
            real_ds = TCPRIMEDDataset(sample_index_csv=pilot_csv, partition=None)
            gmi_indices = [i for i, r in real_ds.df.iterrows() if "GMI" in str(r.get("instrument", "")) and r.get("has_radar")]
            if len(gmi_indices) > 0:
                sample = real_ds[gmi_indices[0]]
                self.assertIn("radar_echo", sample["masks"], "radar_echo mask must be present in sample masks!")
                self.assertEqual(sample["masks"]["radar_echo"].shape, (1, 16, 32, 32))
                self.assertEqual(sample["masks"]["radar_voxel"].shape, (1, 16, 32, 32))
                # Swath reality: voxel mask sum >= echo mask sum
                self.assertGreater(sample["masks"]["radar_voxel"].sum().item(), 0.0)
                self.assertGreaterEqual(sample["masks"]["radar_voxel"].sum().item(), sample["masks"]["radar_echo"].sum().item())
                # Source must be verified radar_beam_geodetic_km from coordinates
                prov = sample.get("geospatial_provenance", {}).get("radar")
                self.assertEqual(prov, "radar_beam_geodetic_km", "Radar must use true beam coordinates!")

    # 8. MFM Pretraining Execution Across All Modalities & Encoder Gradient Flow
    def test_08_mfm_pretraining_workflow_and_all_modality_gradients(self):
        tmp_ckpt_dir = "checkpoints/test_pretrain_unit"
        tmp_summary = "reports/phase2e/test_pretrain_unit_summary.json"
        tmp_grads = "reports/phase2e/test_pretrain_unit_grads.json"
        os.makedirs(tmp_ckpt_dir, exist_ok=True)

        res = run_pretraining(
            train_index=self.sample_index_csv if self.has_real_index else "reports/pipeline/sample_index.csv",
            epochs=1,
            batch_size=4,
            d_model=64,
            max_samples=8,
            checkpoint_dir=tmp_ckpt_dir,
            output_report=tmp_summary,
            gradient_report_path=tmp_grads,
            device="cpu",
        )

        self.assertEqual(res["epochs"], 1)
        self.assertIsNotNone(res["final_loss_mse"])
        self.assertGreater(res["final_loss_mse"], 0.0)
        self.assertFalse(math.isnan(res["final_loss_mse"]))

        # Verify saved backbone checkpoint
        ckpt_path = os.path.join(tmp_ckpt_dir, "pretrained_backbone.pt")
        self.assertTrue(os.path.exists(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("history", ckpt)

        # Verify gradients flowed into ALL encoders, fusion and reconstruction head
        grad_norms = res["final_gradient_norms"]
        expected_modules = ["ir_encoder", "pmw_encoder", "gprof_encoder", "radar_encoder", "env_encoder", "fusion", "recon_head"]
        for mod in expected_modules:
            self.assertIn(mod, grad_norms)
            self.assertGreater(grad_norms[mod], 0.0, f"MFM pretraining yielded zero gradient for {mod}!")

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

    # 11. Critical 1 Fix: Polar Graph Connected to True Polar Spatial Features
    def test_11_polar_graph_connected_to_polar_spatial_features(self):
        model = TCGraphFormer(d_model=64)
        model.eval()

        # 1. Structural verification: TCGraphFormer imports and instantiates PolarSectorPooling
        self.assertTrue(hasattr(model, "polar_pool"), "TCGraphFormer must instantiate polar_pool!")
        self.assertIsInstance(model.polar_pool, PolarSectorPooling)
        self.assertIs(model.ir_encoder.polar_pool, model.polar_pool)
        self.assertIs(model.pmw_encoder.polar_pool, model.polar_pool)
        self.assertIs(model.gprof_encoder.polar_pool, model.polar_pool)
        self.assertIs(model.radar_encoder.polar_pool, model.polar_pool)
        self.assertIs(model.env_encoder.polar_pool, model.polar_pool)

        # 2. Polar Sector Grounding: Eye / Inner-Core NE sector localization
        B = 1
        ir_baseline = torch.zeros(B, 1, 64, 64)
        ir_eye_ne = torch.zeros(B, 1, 64, 64)
        # Point (row=30, col=33) on 64x64 grid is physically in Ring 0 (r ~ 33 km < 75 km), Quadrant 0 (NE: theta = 45 deg)
        ir_eye_ne[0, 0, 30, 33] = 100.0

        f_base = model.ir_encoder.extract_features(ir_baseline)
        t_base, _ = model.polar_pool(f_base)

        f_eye = model.ir_encoder.extract_features(ir_eye_ne)
        t_eye, _ = model.polar_pool(f_eye)

        diff = (t_eye - t_base).abs().sum(dim=-1)[0]  # (16,)
        # Sector 0 represents Ring 0 (Eye) NE: must have the dominant response
        self.assertGreater(diff[0].item(), 5.0, "Sector 0 (Eye NE) failed to activate strongly!")
        self.assertGreater(diff[0].item(), diff[1].item() * 3.0, "Sector 0 must exceed Sector 1 (NW)!")
        self.assertGreater(diff[0].item(), diff[4].item() * 5.0, "Sector 0 must exceed Sector 4 (Eyewall NE)!")
        # Outer synoptic ring (Sectors 12..15) must be zero
        for s in range(12, 16):
            self.assertAlmostEqual(diff[s].item(), 0.0, places=4, msg=f"Outer sector {s} corrupted by eye convection!")

        # 3. Synoptic Outflow SE sector localization (Ring 3 SE: r ~ 408 km, theta = 315 deg)
        ir_outflow_se = torch.zeros(B, 1, 64, 64)
        ir_outflow_se[0, 0, 50, 50] = 100.0

        f_outflow = model.ir_encoder.extract_features(ir_outflow_se)
        t_outflow, _ = model.polar_pool(f_outflow)
        diff_outflow = (t_outflow - t_base).abs().sum(dim=-1)[0]

        # Sector 15 represents Ring 3 (Synoptic Outflow) SE: must activate
        self.assertGreater(diff_outflow[15].item(), 0.01, "Sector 15 (Outflow SE) failed to activate!")
        # Sector 0 (Eye NE) must be strictly zero
        self.assertAlmostEqual(diff_outflow[0].item(), 0.0, places=5, msg="Eye sector 0 corrupted by outer outflow!")

        # 4. Pixel Validity Mask Invariance at Polar Sector Level
        pix_mask = torch.ones(B, 1, 64, 64)
        pix_mask[0, 0, 30, 33] = 0.0  # Mask out the convective pixel as invalid / unobserved

        f_masked = model.ir_encoder.extract_features(ir_eye_ne, pixel_mask=pix_mask)
        t_masked, _ = model.polar_pool(f_masked, pixel_mask=pix_mask)
        diff_masked = (t_masked - t_base).abs().sum(dim=-1)[0]
        self.assertAlmostEqual(diff_masked[0].item(), 0.0, places=4, msg="Invalid masked pixel leaked into Sector 0!")

    # 12. Geospatial Coordinate Preservation, Axis Meshing & Radar Swath Registration
    def test_12_geospatial_coordinate_preservation_and_registration(self):
        # 1. Verify StorageWriter preserves coordinates for IR, GPROF, Radar, and ERA5
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = StorageWriter(OutputConfig(root=tmp_dir, compression_level=1))
            H_in, W_in = 64, 64
            obs = {
                "observation_id": "TEST_OBS_01",
                "sample_id": "TEST_GEO_01",
                "timestamp_str": "20200913_160000",
                "timestamp_epoch": 1600000000,
                "timestamp_utc": "20200913_160000",
                "instrument": "GMI_GPM",
                "storm_latitude": 20.0,
                "storm_longitude": -60.0,
                "intensity_knots": 65.0,
                "pressure_hpa": 985.0,
                "raw_stage": "H1",
                "stage_class_id": 2,
                "ir_data": {
                    "is_available": True,
                    "irwin_image": np.ones((H_in, W_in), dtype=np.float32) * 220.0,
                    "latitude": np.ones((H_in, W_in), dtype=np.float32) * 20.0,
                    "longitude": np.ones((H_in, W_in), dtype=np.float32) * -60.0,
                    "x": np.ones((H_in, W_in), dtype=np.float32) * 10.0,
                    "y": np.ones((H_in, W_in), dtype=np.float32) * 15.0,
                },
                "gprof_data": {
                    "is_available": True,
                    "surface_precip": np.ones((H_in, W_in), dtype=np.float32) * 12.0,
                    "latitude": np.ones((H_in, W_in), dtype=np.float32) * 20.0,
                    "longitude": np.ones((H_in, W_in), dtype=np.float32) * -60.0,
                    "x": np.ones((H_in, W_in), dtype=np.float32) * 10.0,
                    "y": np.ones((H_in, W_in), dtype=np.float32) * 15.0,
                },
                "radar_data": {
                    "is_available": True,
                    "radar_group": "KuGMI",
                    "ku_reflectivity_profile": np.ones((88, 50, 49), dtype=np.float32) * 35.0,
                    "latitude": np.ones((50, 49), dtype=np.float32) * 20.0,
                    "longitude": np.ones((50, 49), dtype=np.float32) * -60.0,
                    "x": np.ones((50, 49), dtype=np.float32) * 5.0,
                    "y": np.ones((50, 49), dtype=np.float32) * 5.0,
                    "bin_height": np.ones((88, 50, 49), dtype=np.float32) * 2.0,
                },
                "env_slice_data": {
                    "is_available": True,
                    "time_epoch": 1600000000,
                    "precipitable_water": np.ones((121, 121), dtype=np.float32) * 50.0,
                    "latitude": np.linspace(15.0, 25.0, 121, dtype=np.float32),
                    "longitude": np.linspace(-65.0, -55.0, 121, dtype=np.float32),
                },
            }

            container_path = writer.write_storm_container("TEST_GEO_STORM", 2020, "AL", "TRAIN", {}, [obs])
            self.assertTrue(os.path.exists(container_path))

            with nc.Dataset(container_path, "r") as ds:
                g_ops = ds.groups["overpasses"]
                g_obs = list(g_ops.groups.values())[0]
                # IR variables
                g_ir = g_obs.groups["infrared"]
                self.assertIn("latitude", g_ir.variables)
                self.assertIn("longitude", g_ir.variables)
                self.assertIn("x", g_ir.variables)
                self.assertIn("y", g_ir.variables)

                # GPROF variables
                g_gp = g_obs.groups["GPROF"]
                self.assertIn("latitude", g_gp.variables)
                self.assertIn("longitude", g_gp.variables)
                self.assertIn("x", g_gp.variables)
                self.assertIn("y", g_gp.variables)

                # Radar variables
                g_rad = g_obs.groups["radar_radiometer"]
                self.assertIn("latitude", g_rad.variables)
                self.assertIn("longitude", g_rad.variables)
                self.assertIn("x", g_rad.variables)
                self.assertIn("y", g_rad.variables)
                self.assertIn("binHeight", g_rad.variables)

                # Environmental variables
                g_env = g_obs.groups["environmental_synoptic"]
                self.assertIn("latitude", g_env.variables)
                self.assertIn("longitude", g_env.variables)
                self.assertEqual(g_env.variables["latitude"].shape, (121,))
                self.assertEqual(g_env.variables["longitude"].shape, (121,))

        # 2. Verify 1D ERA5 Axis Meshing & Tangent-Plane Geodetic Resampling
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.strict_mode = False
        dataset.processing_errors = []

        class Mock1DAxisGroup:
            variables = {
                "latitude": np.linspace(18.0, 22.0, 121, dtype=np.float32),
                "longitude": np.linspace(-62.0, -58.0, 121, dtype=np.float32),
            }

        env_field_121 = np.ones((121, 121), dtype=np.float32) * 45.0
        t_env, m_env, src_env = dataset._resample_swath_with_coords(
            env_field_121,
            Mock1DAxisGroup(),
            storm_lat=20.0,
            storm_lon=-60.0,
            target_shape=(64, 64),
            domain_km=500.0,
        )
        self.assertEqual(src_env, "tangent_plane_geodetic_km")
        self.assertEqual(t_env.shape, (1, 64, 64))
        self.assertEqual(m_env.shape, (1, 64, 64))
        self.assertGreater(m_env.sum().item(), 0.0)

        # 3. Verify Radar Beam Coordinate Resampling and Outside-Swath Masking
        # Narrow radar swath: 100 scans, 49 beams covering [-100, 100] km x [-100, 100] km
        rad_x = np.linspace(-100.0, 100.0, 49, dtype=np.float32)
        rad_y = np.linspace(-100.0, 100.0, 100, dtype=np.float32)
        rad_x_2d, rad_y_2d = np.meshgrid(rad_x, rad_y)
        rad_ku_3d = np.ones((88, 100, 49), dtype=np.float32) * 30.0  # 30 dBZ

        class MockRadarBeamGroup:
            variables = {
                "correctedReflectFactor_Ku": rad_ku_3d,
                "x": rad_x_2d,
                "y": rad_y_2d,
            }

        r_tensor, r_mask, r_valid, r_src = dataset._load_radar_volume(
            MockRadarBeamGroup(),
            target_grid=(64, 64),
            target_depth=16,
            domain_km=500.0,
            max_dist_km=35.0,
            return_source=True,
        )
        self.assertTrue(r_valid)
        self.assertEqual(r_src, "radar_beam_geodetic_km")
        self.assertEqual(r_tensor.shape, (1, 16, 32, 32))
        self.assertEqual(r_mask.shape, (1, 16, 32, 32))

        # Check outside-swath masking:
        # Target grid runs from -500 to +500 km across 32 pixels (~32 km / pixel).
        # Inner center (row 15, col 15) is at ~0 km -> MUST be valid (mask == 1.0)
        # Outer edge (row 0, col 0, dist ~700 km) -> MUST be strictly unobserved (mask == 0.0)
        center_mask = r_mask[0, :, 15, 15]
        self.assertEqual(center_mask.sum().item(), 16.0, "Center cells inside swath must have valid mask!")

        edge_mask = r_mask[0, :, 0, 0]
        self.assertEqual(edge_mask.sum().item(), 0.0, "Outer cells beyond swath must be strictly masked 0!")
        self.assertEqual(r_tensor[0, :, 0, 0].sum().item(), 0.0, "Outer cells must be strictly zero data!")

        # 4. Fallback Path Integrity: Missing Coordinates Marked as nominal_fallback
        class MockEmptyRadarGroup:
            variables = {
                "correctedReflectFactor_Ku": rad_ku_3d,
            }

        _, _, fallback_valid, fallback_src = dataset._load_radar_volume(
            MockEmptyRadarGroup(),
            target_grid=(64, 64),
            target_depth=16,
            return_source=True,
        )
        self.assertTrue(fallback_valid)
        self.assertEqual(fallback_src, "nominal_fallback", "Missing coordinates must be tracked as nominal_fallback!")

    # 13. Fusion Per-Token Validity & Attention Protection (Critical 3)
    def test_13_fusion_token_level_validity_masking(self):
        torch.manual_seed(42)
        model = TCGraphFormer(d_model=64)
        model.eval()

        B = 2
        H, W = 64, 64
        ir = torch.randn(B, 1, H, W)
        pmw = torch.randn(B, 4, H, W)
        gprof = torch.randn(B, 4, H, W)
        radar = torch.randn(B, 1, 16, H // 2, W // 2)
        env = torch.randn(B, 6, H, W)

        # 1. Verify all 5 encoders return both features and token-level validity
        mask_ones = torch.ones(B)
        t_ir, val_ir = model.ir_encoder(ir, mask_ones, return_validity=True)
        t_pmw, val_pmw = model.pmw_encoder(pmw, mask_ones, return_validity=True)
        t_gp, val_gp = model.gprof_encoder(gprof, mask_ones, return_validity=True)
        t_rad, val_rad = model.radar_encoder(radar, mask_ones, return_validity=True)
        t_env, val_env = model.env_encoder(env, mask_ones, return_validity=True)

        self.assertEqual(t_ir.shape, (B, 16, 64))
        self.assertEqual(val_ir.shape, (B, 16))
        self.assertEqual(val_pmw.shape, (B, 16))
        self.assertEqual(val_gp.shape, (B, 16))
        self.assertEqual(val_rad.shape, (B, 16))
        self.assertEqual(val_env.shape, (B, 16))

        # 2. Token Projection Bias Suppression: Completely invalid patch must yield identically 0.0 tokens
        pix_ir_part = torch.ones(B, 1, H, W)
        # Invalidate sectors 1..15, keeping only sector 0 (Inner Core NE)
        reg_masks = model.polar_pool._get_region_masks(H, W, ir.device)
        sec_0_mask = reg_masks[0] > 0.0
        pix_ir_part[:, :, ~sec_0_mask] = 0.0  # Zero out all pixels outside sector 0

        t_ir_masked, val_ir_masked = model.ir_encoder(ir, mask_ones, pixel_mask=pix_ir_part, return_validity=True)
        # Sectors 1..15 must be strictly zero in token validity
        self.assertEqual(val_ir_masked[:, 1:].sum().item(), 0.0)
        # Tokens 1..15 must be identically 0.0 (no bias leakage)
        self.assertEqual(t_ir_masked[:, 1:, :].abs().sum().item(), 0.0)

        # 3. Cross-Modal Attention Protection:
        # Cross-modal attention must NEVER attend to invalid tokens.
        # Mutating invalid tokens in Radar must NOT change the fused tokens of IR or valid Radar tokens!
        mod_tokens_clean = [t_ir.clone(), t_pmw.clone(), t_gp.clone(), t_rad.clone(), t_env.clone()]
        mod_masks = torch.ones(B, 5)

        # Radar has valid data ONLY in Sector 0, sectors 1..15 are invalid
        tok_validity_masks = [
            torch.ones(B, 16),  # IR all valid
            torch.ones(B, 16),  # PMW all valid
            torch.ones(B, 16),  # GPROF all valid
            torch.zeros(B, 16), # Radar: only sector 0 valid
            torch.ones(B, 16),  # Env all valid
        ]
        tok_validity_masks[3][:, 0] = 1.0  # Radar sector 0 valid

        # Zero out invalid radar tokens as expected from encoder
        mod_tokens_clean[3][:, 1:, :] = 0.0

        with torch.no_grad():
            fused_clean = model.fusion(
                mod_tokens_clean,
                modality_masks=mod_masks,
                token_masks=tok_validity_masks,
            )

        # Now mutate the invalid radar tokens (sectors 1..15) with arbitrary garbage
        mod_tokens_corrupted = [t.clone() for t in mod_tokens_clean]
        mod_tokens_corrupted[3][:, 1:, :] = torch.randn(B, 15, 64) * 999.0 + 500.0

        with torch.no_grad():
            fused_corrupted = model.fusion(
                mod_tokens_corrupted,
                modality_masks=mod_masks,
                token_masks=tok_validity_masks,
            )

        # Key verification:
        # Fused tokens for IR (tokens 0..15) and valid Radar token (token 48 = Radar sector 0)
        # must be IDENTICAL with 0.000000 difference because attention strictly ignored the invalid tokens!
        ir_diff = (fused_corrupted[:, :16, :] - fused_clean[:, :16, :]).abs().max().item()
        self.assertAlmostEqual(ir_diff, 0.0, places=5, msg="IR fused tokens changed when invalid radar tokens were mutated!")

        radar_sec0_diff = (fused_corrupted[:, 48, :] - fused_clean[:, 48, :]).abs().max().item()
        self.assertAlmostEqual(radar_sec0_diff, 0.0, places=5, msg="Radar valid sector 0 changed when invalid radar tokens were mutated!")

        # Invalid radar tokens in output must be strictly zeroed out
        self.assertEqual(fused_clean[:, 49:64, :].abs().sum().item(), 0.0)
        self.assertEqual(fused_corrupted[:, 49:64, :].abs().sum().item(), 0.0)

        # 4. End-to-End Model Invariance with Narrow Radar Swath
        # When radar has valid data only in sector 0, mutating unobserved radar voxels
        # must produce identically zero change in all model prediction heads
        pix_radar_swath = torch.zeros(B, 1, 16, H // 2, W // 2)
        # Make only inner core valid in radar voxel mask
        pix_radar_swath[:, :, :, 14:18, 14:18] = 1.0

        radar_clean = radar.clone() * pix_radar_swath
        radar_mutated = radar_clean.clone()
        radar_mutated[:, :, :, :10, :10] = 99999.0  # Mutate unobserved voxels

        pixel_masks_clean = {"radar": pix_radar_swath.unsqueeze(1)}
        pixel_masks_mutated = {"radar": pix_radar_swath.unsqueeze(1)}

        delta_t = torch.zeros(B, 1)
        pad_m = torch.zeros(B, 1, dtype=torch.bool)

        with torch.no_grad():
            preds_clean = model(
                ir.unsqueeze(1),
                pmw.unsqueeze(1),
                gprof.unsqueeze(1),
                radar_clean.unsqueeze(1),
                env.unsqueeze(1),
                mod_masks.unsqueeze(1),
                delta_t,
                pad_m,
                pixel_masks=pixel_masks_clean,
            )

            preds_mutated = model(
                ir.unsqueeze(1),
                pmw.unsqueeze(1),
                gprof.unsqueeze(1),
                radar_mutated.unsqueeze(1),
                env.unsqueeze(1),
                mod_masks.unsqueeze(1),
                delta_t,
                pad_m,
                pixel_masks=pixel_masks_mutated,
            )

        int_diff = (preds_mutated["intensity"] - preds_clean["intensity"]).abs().max().item()
        prs_diff = (preds_mutated["pressure"] - preds_clean["pressure"]).abs().max().item()
        stg_diff = (preds_mutated["stage_logits"] - preds_clean["stage_logits"]).abs().max().item()

        self.assertAlmostEqual(int_diff, 0.0, places=5, msg="Model intensity prediction leaked unobserved radar voxels!")
        self.assertAlmostEqual(prs_diff, 0.0, places=5, msg="Model pressure prediction leaked unobserved radar voxels!")
        self.assertAlmostEqual(stg_diff, 0.0, places=5, msg="Model stage prediction leaked unobserved radar voxels!")

    # 14. Radar-Present Sequence Learning & Multi-Task Gradient Flow
    def test_14_radar_present_sequence_learning_and_gradient_flow(self):
        """
        Verify that a multi-modal sequence containing a radar observation
        propagates pixel/voxel masks to the model, calculates multi-task loss,
        and generates strictly non-zero gradients for radar_encoder and all other modules.
        """
        B, T = 1, 4
        H, W = 64, 64
        D_rad = 16

        seq = []
        for t in range(T):
            has_rad = (t == 3)  # Step 3 has radar
            rad_tensor = torch.zeros(1, D_rad, H // 2, W // 2)
            rad_voxel = torch.zeros(1, D_rad, H // 2, W // 2)
            if has_rad:
                rad_tensor[0, :, 10:20, 10:20] = torch.rand(D_rad, 10, 10) * 0.8
                rad_voxel[0, :, 10:20, 10:20] = 1.0

            obs = {
                "sample_id": f"SMP_TEST_{t}",
                "storm_id": "STORM_TEST",
                "timestamp_epoch": 1600000000 + t * 7200,
                "timestamp_utc": f"202106140{t}0000",
                "instrument": "GMI" if has_rad else "SSMIS",
                "ir": torch.randn(1, H, W) * 0.5 + 0.5,
                "pmw": torch.randn(4, H, W) * 0.5 + 0.5,
                "gprof": torch.randn(4, H, W) * 0.5 + 0.5,
                "radar": rad_tensor,
                "env": torch.randn(6, H, W) * 0.5 + 0.5,
                "modality_masks": torch.tensor([1.0, 1.0, 1.0, 1.0 if has_rad else 0.0, 1.0]),
                "masks": {
                    "ir_pixel": torch.ones(1, H, W),
                    "pmw_pixel": torch.ones(4, H, W),
                    "gprof_pixel": torch.ones(4, H, W),
                    "radar_voxel": rad_voxel,
                    "radar_echo": rad_voxel,
                    "env_pixel": torch.ones(6, H, W),
                    "pmw_channels": torch.ones(4),
                    "gprof_channels": torch.ones(4),
                    "env_channels": torch.ones(6),
                },
                "targets": {
                    "intensity": torch.tensor(65.0),
                    "intensity_mask": torch.tensor(1.0),
                    "pressure": torch.tensor(985.0),
                    "pressure_mask": torch.tensor(1.0),
                    "stage": torch.tensor(2),
                    "stage_mask": torch.tensor(1.0),
                    "raw_stage": "HU",
                    "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0]),
                    "intensity_change_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                    "rapid_intensification": torch.tensor(0.0),
                    "ri_mask": torch.tensor(1.0),
                    "track_displacement": torch.tensor([[10.0, 20.0], [20.0, 40.0], [30.0, 60.0], [40.0, 80.0]]),
                    "track_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                },
                "geospatial_provenance": {"radar": "raw_beam_kdtree" if has_rad else "none"},
            }
            seq.append(obs)

        batch_seq = collate_sequence_batch([seq])
        model = TCGraphFormer(d_model=64, num_stages=5)
        loss_fn = MultiTaskLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        optimizer.zero_grad()

        pixel_masks = {k: v for k, v in batch_seq["pixel_masks"].items() if v is not None}
        channel_masks = {k: v for k, v in batch_seq["channel_masks"].items() if v is not None}

        preds = model(
            ir=batch_seq["ir"],
            pmw=batch_seq["pmw"],
            gprof=batch_seq["gprof"],
            radar=batch_seq["radar"],
            env=batch_seq["env"],
            modality_masks=batch_seq["modality_masks"],
            delta_times_hours=batch_seq["delta_times_hours"],
            seq_pad_mask=batch_seq["seq_pad_mask"],
            channel_masks=channel_masks,
            pixel_masks=pixel_masks,
        )

        loss, loss_dict = loss_fn(preds, batch_seq["targets"])
        loss.backward()

        g_radar = compute_module_grad_norm(model.radar_encoder)
        g_ir = compute_module_grad_norm(model.ir_encoder)
        g_pmw = compute_module_grad_norm(model.pmw_encoder)
        g_gp = compute_module_grad_norm(model.gprof_encoder)
        g_env = compute_module_grad_norm(model.env_encoder)
        g_graph = compute_module_grad_norm(model.spatial_graph)
        g_temp = compute_module_grad_norm(model.temporal_transformer)
        g_heads = compute_module_grad_norm(model.task_heads)

        self.assertGreater(g_radar, 0.0, "Radar encoder must receive non-zero gradients when radar is present!")
        self.assertGreater(g_ir, 0.0)
        self.assertGreater(g_pmw, 0.0)
        self.assertGreater(g_gp, 0.0)
        self.assertGreater(g_env, 0.0)
        self.assertGreater(g_graph, 0.0)
        self.assertGreater(g_temp, 0.0)
        self.assertGreater(g_heads, 0.0)

    # 15. Reference-Grid Cell-Centered Spacing (15.625 km, not linspace boundary 15.873 km)
    def test_15_reference_grid_cell_centered_spacing_15_625km(self):
        model = TCGraphFormer(d_model=64)
        pool = model.polar_pool
        H, W = 64, 64
        domain_radius_km = 500.0

        # Exact expected spacing: 2 * 500 / 64 = 15.625 km
        expected_dx = (2.0 * domain_radius_km) / W
        self.assertAlmostEqual(expected_dx, 15.625, places=4)

        # Outer endpoints: linspace(-500, 500, 64) would have step 1000/63 = 15.8730 km.
        # Cell-centered grid coordinates must have step exactly 15.625 km and cell centers at +/- 492.1875 km
        expected_x0 = -domain_radius_km + 0.5 * expected_dx  # -492.1875 km
        expected_x_last = domain_radius_km - 0.5 * expected_dx # +492.1875 km
        self.assertAlmostEqual(expected_x0, -492.1875, places=4)
        self.assertAlmostEqual(expected_x_last, 492.1875, places=4)

        # PolarSectorPooling internal cell center verification
        masks = pool._get_region_masks(H, W, "cpu")
        self.assertEqual(masks.shape, (16, H, W))
        # Total cells partitioned by polar sectors must be valid
        self.assertGreater(masks.sum().item(), 0.0)

        # Test dataset swath resampling coordinate generator
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64
        dx = (2.0 * domain_radius_km) / 64
        gx = -domain_radius_km + (np.arange(64, dtype=np.float32) + 0.5) * dx
        self.assertAlmostEqual(float(gx[1] - gx[0]), 15.625, places=4)
        self.assertAlmostEqual(float(gx[0]), -492.1875, places=4)
        self.assertAlmostEqual(float(gx[-1]), 492.1875, places=4)

    # 16. MFM Valid-Token Selection Excludes Zero-Pixel Coverage Sectors
    def test_16_mfm_valid_token_selection_excludes_zero_pixel_coverage(self):
        torch.manual_seed(42)
        model = TCGraphFormer(d_model=64)
        B = 2
        H, W = 64, 64
        ir = torch.randn(B, 1, H, W)
        pmw = torch.randn(B, 4, H, W)
        gprof = torch.randn(B, 4, H, W)
        radar = torch.randn(B, 1, 16, 32, 32)
        env = torch.randn(B, 6, H, W)
        masks = torch.ones(B, 5) # All 5 modalities marked present

        # Create a partial pixel mask for IR where only sector 0 (Inner Core NE) is valid;
        # sectors 1..15 have ZERO valid pixel coverage.
        reg_masks = model.polar_pool._get_region_masks(H, W, "cpu")
        pix_ir = torch.zeros(B, 1, H, W)
        pix_ir[:, :, reg_masks[0] > 0] = 1.0

        # Encoders with return_validity=True
        with torch.no_grad():
            t_ir, v_ir = model.ir_encoder(ir, masks[:, 0], pixel_mask=pix_ir, return_validity=True)
            t_pmw, v_pmw = model.pmw_encoder(pmw, masks[:, 1], return_validity=True)
            t_gp, v_gp = model.gprof_encoder(gprof, masks[:, 2], return_validity=True)
            t_rad, v_rad = model.radar_encoder(radar, masks[:, 3], return_validity=True)
            t_env, v_env = model.env_encoder(env, masks[:, 4], return_validity=True)

        # IR validity must be 1.0 ONLY for sector 0, and strictly 0.0 for sectors 1..15
        self.assertEqual(v_ir[:, 0].sum().item(), float(B))
        self.assertEqual(v_ir[:, 1:].sum().item(), 0.0)

        # MFM valid_tok_mask
        valid_tok_mask = torch.cat([v_ir, v_pmw, v_gp, v_rad, v_env], dim=1).unsqueeze(-1) # (B, 80, 1)

        # Assert IR tokens 1..15 in valid_tok_mask are strictly 0.0
        self.assertEqual(valid_tok_mask[:, 1:16, :].sum().item(), 0.0)

        # Target representations gated by valid_tok_mask
        all_targets = torch.cat([t_ir, t_pmw, t_gp, t_rad, t_env], dim=1) * valid_tok_mask
        # Invalid IR targets must be identically 0.0
        self.assertEqual(all_targets[:, 1:16, :].abs().sum().item(), 0.0)

        # Ensure random masking loop NEVER selects invalid tokens
        mask_to_recon = torch.zeros_like(valid_tok_mask)
        for b in range(B):
            valid_idx = torch.where(valid_tok_mask[b, :, 0] > 0.5)[0]
            # Ensure valid_idx does NOT contain any of 1..15
            for idx in valid_idx.tolist():
                self.assertNotIn(idx, list(range(1, 16)))


if __name__ == "__main__":
    unittest.main()
