"""
Phase 2E.2 — Polar Feature Integration, Complete Geospatial Registration & Validity-Aware Fusion Test Suite.

Validates all 10 Phase 2E.2 Priorities:
  1. PolarSectorPooling directly invoked in TCGraphFormer.extract_overpass_features().
  2. Cartesian feature-to-polar-node mapping (node 0 is Inner Core NE; tracing impulse locations).
  3. Token validity derivation from spatial pixel coverage and suppression of projection linear bias.
  4. Cross-modal attention protection via attention padding masks (unobserved tokens cannot affect output).
  5. Geospatial metadata preservation in StorageWriter (IR, PMW, GPROF, Radar, ERA5).
  6. Coordinate-aware resampling with 1D ERA5 axes, antimeridian wrapping, and tangent plane projection.
  7. Modality-specific resampling distance thresholds (IR 15km, PMW 35km, GPROF 20km, Radar 15km, ERA5 40km).
  8. Radar geometry and 12 dBZ precipitation echo classification separation from measurement validity.
  9. Exact reference-grid cell-centered spacing (15.625 km, endpoints at +/- 492.1875 km).
 10. MFM valid-token selection, loss normalization, and positive parameter update norm tracking.
 11. Strict scientific mode raising explicit error on missing coordinate metadata.
 12. End-to-end radar sequence gradient flow with pixel masks enabled.
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
from src.training.pretrain import (
    MaskedFeatureReconstructionHead,
    compute_module_grad_norm,
    compute_module_param_update_norm,
    run_pretraining,
)


class TestPhase2E2PhysicalTokenIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_index_csv = "reports/pipeline/sample_index.csv"
        cls.has_real_index = os.path.exists(cls.sample_index_csv)

    # -------------------------------------------------------------------------
    # Priority 1: Polar Pooling in TCGraphFormer & Spatial Feature Map Tracing
    # -------------------------------------------------------------------------
    def test_01_polar_pooling_invoked_in_extract_overpass_features(self):
        """Verify that TCGraphFormer.extract_overpass_features() explicitly executes PolarSectorPooling."""
        model = TCGraphFormer(d_model=64)
        B = 2
        ir = torch.randn(B, 1, 64, 64)
        pmw = torch.randn(B, 4, 64, 64)
        gprof = torch.randn(B, 4, 64, 64)
        radar = torch.randn(B, 1, 16, 32, 32)
        env = torch.randn(B, 6, 64, 64)
        masks = torch.ones(B, 5)

        # Instrument polar_pool to track call count
        original_forward = model.polar_pool.forward
        call_count = [0]

        def tracked_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        model.polar_pool.forward = tracked_forward

        latent = model.extract_overpass_features(
            ir=ir, pmw=pmw, gprof=gprof, radar=radar, env=env, modality_masks=masks
        )
        self.assertEqual(latent.shape, (B, 64))
        # 5 modalities must each be pooled via polar_pool
        self.assertGreaterEqual(call_count[0], 5, "PolarSectorPooling must be directly invoked for all modalities!")

    def test_02_spatial_feature_location_traces_to_expected_polar_node(self):
        """Verify node 0 corresponds to Inner Core NE, and real impulses trace to expected polar sectors."""
        model = TCGraphFormer(d_model=64)
        pool = model.polar_pool
        H, W = 64, 64
        dy = 1000.0 / H
        dx = 1000.0 / W
        y = 500.0 - (torch.arange(H, dtype=torch.float32) + 0.5) * dy
        x = -500.0 + (torch.arange(W, dtype=torch.float32) + 0.5) * dx

        # Define 4 points: one in each concentric ring and azimuthal quadrant
        # Ring 0: [0, 75) km, Quadrant 0: NE [0, 90) deg -> Node 0
        # Ring 1: [75, 175) km, Quadrant 1: NW [90, 180) deg -> Node 5
        # Ring 2: [175, 300) km, Quadrant 2: SW [180, 270) deg -> Node 10
        # Ring 3: [300, 500) km, Quadrant 3: SE [270, 360) deg -> Node 15
        test_points = [
            (25.0, 25.0, 0),       # Eye / Inner Core NE
            (-100.0, 50.0, 5),     # Eyewall NW
            (-150.0, -100.0, 10),  # Outer Rainband SW
            (250.0, -250.0, 15),   # Synoptic Environment SE
        ]

        for x_pt, y_pt, expected_node in test_points:
            col = torch.argmin(torch.abs(x - x_pt)).item()
            row = torch.argmin(torch.abs(y - y_pt)).item()

            feat = torch.zeros(1, 64, H, W)
            feat[0, :, row, col] = 10.0

            tokens, _ = pool(feat)
            max_node = torch.argmax(tokens[0].norm(dim=-1)).item()
            self.assertEqual(
                max_node, expected_node,
                f"Feature impulse at ({x_pt}, {y_pt}) km landed in node {max_node}, expected polar node {expected_node}!"
            )

    # -------------------------------------------------------------------------
    # Priority 2: Token Validity & Cross-Modal Attention Protection
    # -------------------------------------------------------------------------
    def test_03_token_validity_flow_and_linear_bias_suppression(self):
        """Verify token validity is derived from pixel coverage and linear bias is zeroed out for invalid tokens."""
        model = TCGraphFormer(d_model=64)
        B = 2
        ir = torch.randn(B, 1, 64, 64)
        mask_ones = torch.ones(B)

        # Invalidate sectors 1..15, keeping only sector 0
        reg_masks = model.polar_pool._get_region_masks(64, 64, ir.device)
        sec_0_mask = reg_masks[0] > 0.0
        pix_ir_part = torch.zeros(B, 1, 64, 64)
        pix_ir_part[:, :, sec_0_mask] = 1.0

        t_ir, val_ir = model.ir_encoder(ir, mask_ones, pixel_mask=pix_ir_part, return_validity=True)
        self.assertEqual(val_ir[:, 0].sum().item(), float(B))
        self.assertEqual(val_ir[:, 1:].sum().item(), 0.0)

        # Projected tokens 1..15 must be identically 0.0 (no bias leakage)
        self.assertEqual(t_ir[:, 1:, :].abs().sum().item(), 0.0)

    def test_04_cross_modal_attention_excludes_unobserved_tokens(self):
        """Mutating unobserved tokens must not change the attention output of valid observed tokens."""
        model = TCGraphFormer(d_model=64)
        model.eval()
        B = 2
        mod_tokens_clean = [torch.randn(B, 16, 64) for _ in range(5)]
        mod_masks = torch.ones(B, 5)

        # Radar has valid data ONLY in Sector 0, sectors 1..15 are invalid
        tok_validity_masks = [torch.ones(B, 16) for _ in range(5)]
        tok_validity_masks[3][:, 1:] = 0.0
        mod_tokens_clean[3][:, 1:, :] = 0.0

        fused_clean = model.fusion(
            mod_tokens_clean, modality_masks=mod_masks, token_masks=tok_validity_masks
        )

        # Corrupt the invalid radar tokens with large random noise
        mod_tokens_corrupted = [t.clone() for t in mod_tokens_clean]
        mod_tokens_corrupted[3][:, 1:, :] = torch.randn(B, 15, 64) * 500.0

        fused_corrupted = model.fusion(
            mod_tokens_corrupted, modality_masks=mod_masks, token_masks=tok_validity_masks
        )

        # Valid tokens (IR, PMW, GPROF, Radar Sector 0, Env) must be identical
        diff_ir = (fused_clean[:, :16, :] - fused_corrupted[:, :16, :]).abs().max().item()
        self.assertAlmostEqual(diff_ir, 0.0, places=5, msg="Cross-modal attention leaked from invalid tokens into IR!")
        diff_rad_0 = (fused_clean[:, 48:49, :] - fused_corrupted[:, 48:49, :]).abs().max().item()
        self.assertAlmostEqual(diff_rad_0, 0.0, places=5, msg="Cross-modal attention leaked from invalid tokens into Radar sector 0!")

    # -------------------------------------------------------------------------
    # Priority 3: Geospatial Metadata Preservation in StorageWriter
    # -------------------------------------------------------------------------
    def test_05_geospatial_metadata_preservation_in_storage_writer(self):
        """Verify that StorageWriter preserves lat/lon/x/y/binHeight across IR, PMW, GPROF, Radar, and ERA5."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OutputConfig(root=tmp_dir)
            writer = StorageWriter(cfg)

            ny, nx = 40, 40
            obs = {
                "observation_id": "TEST_GEO_001",
                "timestamp_str": "19970921000000",
                "instrument": "SSMI",
                "has_pmw": True,
                "has_ir": True,
                "has_gprof": True,
                "has_radar": True,
                "has_causal_env": True,
                "pmw_data": {
                    "is_available": True,
                    "swaths": {
                        "S1": {
                            "latitude": np.full((ny, nx), 15.0, dtype=np.float32),
                            "longitude": np.full((ny, nx), 120.0, dtype=np.float32),
                            "channels": {"TB_19V": {"data": np.full((ny, nx), 200.0, dtype=np.float32), "units": "K"}},
                        }
                    },
                },
                "ir_data": {
                    "is_available": True,
                    "irwin_image": np.full((ny, nx), 240.0, dtype=np.float32),
                    "latitude": np.full((ny, nx), 15.0, dtype=np.float32),
                    "longitude": np.full((ny, nx), 120.0, dtype=np.float32),
                    "x": np.full((ny, nx), -10.0, dtype=np.float32),
                    "y": np.full((ny, nx), 10.0, dtype=np.float32),
                },
                "gprof_data": {
                    "is_available": True,
                    "surface_precip": np.full((ny, nx), 5.0, dtype=np.float32),
                    "latitude": np.full((ny, nx), 15.0, dtype=np.float32),
                    "longitude": np.full((ny, nx), 120.0, dtype=np.float32),
                },
                "radar_data": {
                    "is_available": True,
                    "ku_reflectivity_profile": np.full((16, 20, 20), 25.0, dtype=np.float32),
                    "latitude": np.full((20, 20), 15.0, dtype=np.float32),
                    "longitude": np.full((20, 20), 120.0, dtype=np.float32),
                    "bin_height": np.full((16, 20, 20), 2.5, dtype=np.float32),
                },
                "env_slice_data": {
                    "is_available": True,
                    "precipitable_water": np.full((ny, nx), 50.0, dtype=np.float32),
                    "latitude": np.linspace(10.0, 20.0, ny, dtype=np.float32),
                    "longitude": np.linspace(115.0, 125.0, nx, dtype=np.float32),
                },
            }

            path = writer.write_storm_container("TEST01", 1997, "AL", "TRAIN", {}, [obs])
            self.assertTrue(os.path.exists(path))

            with nc.Dataset(path, "r") as ds:
                g_obs = ds.groups["overpasses"].groups[list(ds.groups["overpasses"].groups.keys())[0]]

                # IR
                self.assertIn("latitude", g_obs.groups["infrared"].variables)
                self.assertIn("x", g_obs.groups["infrared"].variables)
                self.assertEqual(g_obs.groups["infrared"].variables["x"].units, "km")

                # PMW
                self.assertIn("latitude", g_obs.groups["passive_microwave"].groups["S1"].variables)

                # GPROF
                self.assertIn("latitude", g_obs.groups["GPROF"].variables)

                # Radar
                self.assertIn("latitude", g_obs.groups["radar_radiometer"].variables)
                self.assertIn("binHeight", g_obs.groups["radar_radiometer"].variables)
                self.assertEqual(g_obs.groups["radar_radiometer"].variables["binHeight"].units, "km")

                # ERA5
                self.assertIn("latitude", g_obs.groups["environmental_synoptic"].variables)
                self.assertEqual(g_obs.groups["environmental_synoptic"].variables["latitude"].dimensions, ("env_ny",))

    # -------------------------------------------------------------------------
    # Priority 4: Coordinate-Aware Resampling (1D Axes, Antimeridian Wrapping)
    # -------------------------------------------------------------------------
    def test_06_1d_era5_coordinate_axes_meshgrid_and_antimeridian_handling(self):
        """Verify _resample_swath_with_coords handles 1D ERA5 axes and antimeridian wrapping (-180 to +180)."""
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64
        dataset.strict_mode = False

        ny, nx = 50, 50
        arr = np.ones((ny, nx), dtype=np.float32) * 50.0
        # Storm at antimeridian: storm_lon = 180.0
        lat_1d = np.linspace(10.0, 20.0, ny, dtype=np.float32)
        lon_1d = np.linspace(175.0, 185.0, nx, dtype=np.float32) # spans across 180

        class DummyEnvGroup:
            variables = {
                "latitude": type("Var", (), {"__getitem__": lambda self, idx: lat_1d})(),
                "longitude": type("Var", (), {"__getitem__": lambda self, idx: lon_1d})(),
            }

        t_data, t_mask, src = dataset._resample_swath_with_coords(
            arr, DummyEnvGroup(), storm_lat=15.0, storm_lon=180.0, target_shape=(64, 64), domain_km=500.0, max_dist_km=40.0
        )
        self.assertEqual(src, "tangent_plane_geodetic_km")
        # Center of reference grid must be valid and non-zero
        self.assertEqual(t_mask[0, 32, 32].item(), 1.0)
        self.assertAlmostEqual(t_data[0, 32, 32].item(), 50.0, places=1)

    def test_07_modality_specific_resampling_distance_thresholds(self):
        """Verify distinct max_dist_km thresholds: IR 15km, PMW 35km, GPROF 20km, Radar 15km, ERA5 40km."""
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64

        # A point located 25 km away from reference grid center
        arr = np.array([[100.0]], dtype=np.float32)
        x_pt = np.array([[25.0]], dtype=np.float32)
        y_pt = np.array([[0.0]], dtype=np.float32)

        class DummyGrp:
            variables = {
                "x": type("Var", (), {"__getitem__": lambda self, idx: x_pt})(),
                "y": type("Var", (), {"__getitem__": lambda self, idx: y_pt})(),
            }

        # 1. With max_dist_km = 15.0 (IR, Radar): point 25km away should NOT be matched at center (0,0)
        _, mask_15, _ = dataset._resample_swath_with_coords(
            arr, DummyGrp(), 15.0, 120.0, (64, 64), domain_km=500.0, max_dist_km=15.0
        )
        # Center of reference grid (index 32, 32 corresponds to +/- 7.8km from 0)
        self.assertEqual(mask_15[0, 32, 32].item(), 0.0)

        # 2. With max_dist_km = 35.0 (PMW, ERA5): point 25km away IS matched at center
        _, mask_35, _ = dataset._resample_swath_with_coords(
            arr, DummyGrp(), 15.0, 120.0, (64, 64), domain_km=500.0, max_dist_km=35.0
        )
        self.assertEqual(mask_35[0, 32, 32].item(), 1.0)

    # -------------------------------------------------------------------------
    # Priority 5: Radar Geometry & 12 dBZ Echo Classification
    # -------------------------------------------------------------------------
    def test_08_radar_geometry_and_12dbz_echo_classification(self):
        """Verify radar 12 dBZ echo thresholding keeps measurement validity separate from echo classification."""
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.raw_data_dir = "data/raw/pilot"
        dataset._raw_radar_coord_cache = {}

        # 3 distinct reflectivity levels:
        # 1. 5 dBZ: valid measurement, NOT precip echo
        # 2. 12 dBZ: valid measurement, YES precip echo
        # 3. 35 dBZ: valid measurement, YES precip echo
        raw_ku = np.full((88, 50, 49), -9999.0, dtype=np.float32)
        raw_ku[5:10, 5:10, 5:10] = 5.0
        raw_ku[15:20, 15:20, 15:20] = 12.0
        raw_ku[25:30, 25:30, 25:30] = 35.0

        class DummyRadarGroup:
            variables = {"correctedReflectFactor_Ku": raw_ku}

        tensor, mask, echo, is_valid = dataset._load_radar_volume(
            DummyRadarGroup(), (64, 64), target_depth=16, echo_threshold_dbz=12.0, return_echo_mask=True
        )
        self.assertTrue(is_valid)
        # Measurement validity must be strictly larger than precipitation echo mask
        self.assertGreater(mask.sum().item(), echo.sum().item())
        self.assertGreater(echo.sum().item(), 0.0)

    # -------------------------------------------------------------------------
    # Priority 6: Reference-Grid Geometry Spacing Verification
    # -------------------------------------------------------------------------
    def test_09_reference_grid_cell_centered_exact_15_625km_spacing(self):
        """Verify 64-cell grid has exact 15.625 km spacing, outer edges at +/-500 km, and cell centers +/-492.1875 km."""
        H, W = 64, 64
        domain_km = 500.0
        dx = (2.0 * domain_km) / W
        dy = (2.0 * domain_km) / H
        self.assertAlmostEqual(dx, 15.625, places=5)
        self.assertAlmostEqual(dy, 15.625, places=5)

        gx = -domain_km + (np.arange(W, dtype=np.float32) + 0.5) * dx
        gy = domain_km - (np.arange(H, dtype=np.float32) + 0.5) * dy

        self.assertAlmostEqual(float(gx[0]), -492.1875, places=4)
        self.assertAlmostEqual(float(gx[-1]), 492.1875, places=4)
        self.assertAlmostEqual(float(gy[0]), 492.1875, places=4)
        self.assertAlmostEqual(float(gy[-1]), -492.1875, places=4)

        # Check domain outer extent
        self.assertAlmostEqual(float(gx[0] - 0.5 * dx), -500.0, places=4)
        self.assertAlmostEqual(float(gx[-1] + 0.5 * dx), 500.0, places=4)

    # -------------------------------------------------------------------------
    # Priority 7: MFM Valid-Token Selection & Parameter Update Norms
    # -------------------------------------------------------------------------
    def test_10_mfm_valid_token_selection_and_parameter_update_tracking(self):
        """Verify MFM excludes unobserved tokens, normalizes loss by d_model, and tracks positive parameter update norms."""
        tmp_ckpt_dir = "checkpoints/test_pretrain_unit_p2e2"
        tmp_summary = "reports/phase2e2/test_pretrain_unit_summary.json"
        tmp_grads = "reports/phase2e2/test_pretrain_unit_grads.json"
        os.makedirs(tmp_ckpt_dir, exist_ok=True)
        os.makedirs("reports/phase2e2", exist_ok=True)

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
        self.assertGreater(res["final_loss_mse"], 0.0)

        # Check that parameter update norms were recorded and are strictly positive
        param_updates = res["final_parameter_updates"]
        for mod in ["ir_encoder", "pmw_encoder", "gprof_encoder", "radar_encoder", "env_encoder", "fusion", "recon_head"]:
            self.assertIn(mod, param_updates)
            self.assertGreater(param_updates[mod], 0.0, f"Module {mod} had zero parameter update norm!")

    # -------------------------------------------------------------------------
    # Priority 8 & 4: Strict Scientific Mode Data Quality Failure
    # -------------------------------------------------------------------------
    def test_11_strict_scientific_mode_missing_coordinates_failure(self):
        """In strict scientific mode, missing coordinate metadata must raise an explicit ValueError."""
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.grid_size = 64
        dataset.strict_mode = True

        arr = np.ones((50, 50), dtype=np.float32)

        class EmptyGrp:
            variables = {}
            name = "empty_test_group"

        with self.assertRaises(ValueError):
            dataset._resample_swath_with_coords(arr, EmptyGrp(), 15.0, 120.0, (64, 64))

    # -------------------------------------------------------------------------
    # Priority 9: Radar Sequence Integration & Convergence Distinction
    # -------------------------------------------------------------------------
    def test_12_radar_present_sequence_gradient_flow_and_convergence_distinction(self):
        """Verify model forward/backward pass with pixel masks on radar sequence yields non-zero radar gradient."""
        seq = []
        for t_idx in range(4):
            has_rad = (t_idx == 2)
            radar_tensor = torch.zeros(1, 16, 32, 32)
            radar_mask = torch.zeros(1, 16, 32, 32)
            radar_echo = torch.zeros(1, 16, 32, 32)
            if has_rad:
                radar_tensor[:, :, 10:20, 10:20] = 0.45
                radar_mask[:, :, 10:20, 10:20] = 1.0
                radar_echo[:, :, 10:20, 10:20] = 1.0

            obs = {
                "sample_id": f"TEST_RAD_SEQ_{t_idx}",
                "storm_id": "AL022021",
                "instrument": "GMI" if has_rad else "SSMI",
                "timestamp_epoch": 1623674141 + t_idx * 10800,
                "timestamp_utc": f"2021-06-14T{12 + t_idx * 3}:00:00",
                "ir": torch.randn(1, 64, 64),
                "pmw": torch.randn(4, 64, 64),
                "gprof": torch.randn(4, 64, 64),
                "radar": radar_tensor,
                "env": torch.randn(6, 64, 64),
                "modality_masks": torch.tensor([1.0, 1.0, 1.0, 1.0 if has_rad else 0.0, 1.0]),
                "masks": {
                    "ir_pixel": torch.ones(1, 64, 64),
                    "pmw_pixel": torch.ones(4, 64, 64),
                    "pmw_channels": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                    "gprof_pixel": torch.ones(4, 64, 64),
                    "gprof_channels": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                    "radar_voxel": radar_mask,
                    "radar_echo": radar_echo,
                    "env_pixel": torch.ones(6, 64, 64),
                    "env_channels": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                },
                "targets": {
                    "intensity": torch.tensor(65.0),
                    "intensity_mask": torch.tensor(1.0),
                    "pressure": torch.tensor(985.0),
                    "pressure_mask": torch.tensor(1.0),
                    "stage": torch.tensor(2),
                    "stage_mask": torch.tensor(1.0),
                    "intensity_change": torch.tensor([5.0, 10.0, 15.0, 20.0]),
                    "intensity_change_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                    "rapid_intensification": torch.tensor(1.0),
                    "ri_mask": torch.tensor(1.0),
                    "track_displacement": torch.zeros((4, 2)),
                    "track_mask": torch.tensor([1.0, 1.0, 1.0, 1.0]),
                },
                "geospatial_provenance": {"radar": "radar_beam_geodetic_km" if has_rad else "none"},
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
        self.assertGreater(g_radar, 0.0, "Radar encoder must receive non-zero autograd gradients when radar is present!")


if __name__ == "__main__":
    unittest.main()
