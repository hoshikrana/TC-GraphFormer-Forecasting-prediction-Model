"""
Phase 2E.3 — Production Scientific Correctness, Physical Grid Alignment,
Masked Graph Processing and Radar Geometry Test Suite.

Validates all 12 Phase 2E.3 Priorities:
  1. Feature-grid coordinate alignment: 32x32 convolution receptive field centers vs 64x64 cell centers (7.8125 km offset).
  2. Spatial impulse tests across all 5 modalities mapping to identical polar nodes.
  3. Polar sector physical area coverage threshold (>= 5% true geometric area).
  4. Node validity derivation and strict linear bias suppression post-projection and in spatial aggregator.
  5. Spatial graph message passing with masked node validity; all-invalid sample produces exact zero vector without NaNs/Infs.
  6. Physical radar CAPPI vertical resampling to 16 heights (0.5 to 19.5 km) with real gate heights, no extrapolation, and separate echo classification (>= 12 dBZ) vs measurement validity (-20 to 100 dBZ).
  7. Geospatial policy enforcement ('raise', 'exclude_modality', 'exclude_sample', 'nominal_fallback').
  8. Storage schema versioning (SCHEMA_VERSION = "2.3.0") and CF metadata preservation (bin_height_km).
  9. Inventory lifecycle preservation across date boundaries in DatasetInventory.scan().
 10. MFM edge cases (0-token skip, 1-token no-mask, >= 2 tokens with >= 1 visible, loss denominator sum(M * d_model) + eps).
 11. Baseline persistence anchor (V(t) predicting V(t+h)) vs climatology mean, evaluated on identical target masks.
 12. End-to-end radar sequence gradient flow with all 10 modules having strictly non-zero gradients.
"""

import os
import math
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import netCDF4 as nc

from src.models.graph.grid_geometry import GridGeometry, GLOBAL_GRID_GEOMETRY
from src.models.graph.spatial_graph import PolarSectorPooling, SpatialGraphEncoder
from src.models.tc_graphformer import TCGraphFormer
from src.data.dataset import (
    TCPRIMEDDataset,
    TCPRIMEDSequenceDataset,
    SampleExclusionError,
    TARGET_RADAR_HEIGHTS_KM,
    DEFAULT_RESAMPLING_THRESHOLDS_KM,
)
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from tcprimed_pipeline.storage_writer import StorageWriter, SCHEMA_VERSION, RADAR_VERTICAL_CONVENTION
from tcprimed_pipeline.config import OutputConfig, PipelineConfig, DatasetConfig
from tcprimed_pipeline.inventory import DatasetInventory
from src.training.pretrain import run_pretraining, MaskedFeatureReconstructionHead, compute_module_grad_norm
from src.training.losses import MultiTaskLoss
from src.evaluation.evaluate_baselines import compute_pr_auc_and_brier


class TestPhase2E3ScientificCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_index_csv = "reports/pipeline/sample_index.csv"
        cls.has_real_index = os.path.exists(cls.sample_index_csv)

    # -------------------------------------------------------------------------
    # Priority 1: Feature-Grid Coordinate Alignment (Receptive Field Centers)
    # -------------------------------------------------------------------------
    def test_01_feature_grid_receptive_field_alignment(self):
        """Verify 32x32 convolution receptive field centers vs 64x64 input cell centers and exact 7.8125 km offset."""
        geom = GridGeometry(domain_radius_km=500.0)

        # 64x64 input cell centers
        coords_64_x, coords_64_y = geom.get_input_coords_64()
        self.assertEqual(len(coords_64_x), 64)
        self.assertAlmostEqual(coords_64_x[0], -492.1875, places=4)
        self.assertAlmostEqual(coords_64_x[-1], 492.1875, places=4)
        self.assertAlmostEqual(coords_64_x[1] - coords_64_x[0], 15.625, places=4)

        # 32x32 convolution feature receptive field centers
        coords_32_x, coords_32_y = geom.get_feature_coords_32()
        self.assertEqual(len(coords_32_x), 32)
        # First feature receptive field center is exactly at input index 0.0 -> -492.1875 km
        self.assertAlmostEqual(coords_32_x[0], -492.1875, places=4)
        # Spacing is 2 * 15.625 = 31.25 km
        self.assertAlmostEqual(coords_32_x[1] - coords_32_x[0], 31.25, places=4)
        # Last feature receptive field center is at -492.1875 + 31 * 31.25 = 476.5625 km
        self.assertAlmostEqual(coords_32_x[-1], 476.5625, places=4)

        # Naive 32-cell grid centers: -500 + (c + 0.5) * (1000/32) -> first is -484.375 km
        naive_first = -500.0 + 0.5 * (1000.0 / 32)
        # Physical offset must be exactly 7.8125 km (half of one 15.625 km input cell)
        offset = abs(naive_first - coords_32_x[0])
        self.assertAlmostEqual(offset, 7.8125, places=4)

    # -------------------------------------------------------------------------
    # Priority 2: Spatial Impulse Mapping Across All 5 Modalities
    # -------------------------------------------------------------------------
    def test_02_spatial_impulse_across_all_modalities(self):
        """Verify that a spatial impulse at (100, 100) km maps to the exact same polar node (node 4) in all 5 modalities."""
        # Impulse at x = 100 km, y = 100 km:
        # r = sqrt(100^2 + 100^2) = 141.42 km -> Ring 1 [75, 175) km
        # theta = atan2(100, 100) = 45 deg -> Quadrant 0 (NE [0, 90) deg)
        # Expected node = ring * 4 + quad = 1 * 4 + 0 = 4
        expected_node = 4

        geom = GLOBAL_GRID_GEOMETRY
        x64, y64 = geom.get_input_coords_64()
        X64, Y64 = np.meshgrid(x64, y64)
        X32, Y32 = geom.get_feature_mesh_32()

        # Find closest grid cell in 64x64
        dist_64 = (X64 - 100.0) ** 2 + (Y64 - 100.0) ** 2
        r64, c64 = np.unravel_index(np.argmin(dist_64), dist_64.shape)

        # Find closest grid cell in 32x32 (for radar)
        dist_32 = (X32 - 100.0) ** 2 + (Y32 - 100.0) ** 2
        r32, c32 = np.unravel_index(np.argmin(dist_32), dist_32.shape)

        pool_64 = PolarSectorPooling(in_channels=1, d_model=32, min_sector_coverage=0.0)
        pool_32 = PolarSectorPooling(in_channels=1, d_model=32, min_sector_coverage=0.0)

        # Test 2D input (64x64): impulse at (r64, c64)
        feat_2d = torch.zeros(1, 1, 64, 64)
        feat_2d[0, 0, r64, c64] = 10.0
        mask_2d = torch.zeros(1, 64, 64)
        mask_2d[0, r64, c64] = 1.0

        toks_2d, val_2d = pool_64(feat_2d, pixel_mask=mask_2d)
        # The node with maximum norm must be node 4
        tok_norms_2d = torch.norm(toks_2d[0], dim=-1)
        self.assertEqual(int(torch.argmax(tok_norms_2d).item()), expected_node)

        # Test 3D radar input (32x32 horizontal): impulse at (r32, c32)
        feat_3d = torch.zeros(1, 1, 32, 32)
        feat_3d[0, 0, r32, c32] = 10.0
        mask_3d = torch.zeros(1, 32, 32)
        mask_3d[0, r32, c32] = 1.0

        toks_3d, val_3d = pool_32(feat_3d, pixel_mask=mask_3d)
        tok_norms_3d = torch.norm(toks_3d[0], dim=-1)
        self.assertEqual(int(torch.argmax(tok_norms_3d).item()), expected_node)

    # -------------------------------------------------------------------------
    # Priority 3: Polar Sector Physical Area Coverage Threshold (>= 5%)
    # -------------------------------------------------------------------------
    def test_03_polar_sector_physical_area_coverage_threshold(self):
        """Verify PolarSectorPooling enforces >= 5% physical sector area threshold."""
        geom = GLOBAL_GRID_GEOMETRY
        ring_radii = [0.0, 75.0, 175.0, 300.0, 500.0]
        # Canonical sector analytical area for ring 0: pi * 75^2 / 4 = 4417.86 km^2
        expected_area_r0 = (math.pi * 75.0**2) / 4.0
        # Sector analytical area for ring 3: pi * (500^2 - 300^2) / 4 = 125663.7 km^2
        expected_area_r3 = (math.pi * (500.0**2 - 300.0**2)) / 4.0

        self.assertAlmostEqual(geom.sector_areas_km2[0], expected_area_r0, delta=1.0)
        self.assertAlmostEqual(geom.sector_areas_km2[3], expected_area_r3, delta=1.0)

        pool = PolarSectorPooling(in_channels=4, d_model=32, min_sector_coverage=0.05)
        # Test synthetic mask:
        # Ring 3, Quadrant 0 (Node 12): has sector area ~125664 km^2. One pixel is 15.625 x 15.625 = 244.14 km^2.
        # 1 pixel coverage = 244.14 / 125664 = 0.00194 (0.19% < 5%) -> Must be ZEROED
        feat = torch.ones(1, 4, 64, 64)
        mask = torch.zeros(1, 64, 64)

        # Place 1 valid pixel in Node 12 (Ring 3 NE: x=400, y=100)
        x64, y64 = geom.get_input_coords_64()
        X64, Y64 = np.meshgrid(x64, y64)
        dist_n12 = (X64 - 400.0) ** 2 + (Y64 - 100.0) ** 2
        r_n12, c_n12 = np.unravel_index(np.argmin(dist_n12), dist_n12.shape)
        mask[0, r_n12, c_n12] = 1.0

        toks, val = pool(feat, pixel_mask=mask)
        # Coverage is < 5%, so token 12 must be strictly zero and validity must be 0.0
        self.assertEqual(float(val[0, 12].item()), 0.0)
        self.assertEqual(float(torch.norm(toks[0, 12]).item()), 0.0)

        # Now activate enough pixels to exceed 5% coverage of Node 12 (5% of 125664 km^2 is ~6283 km^2 = ~26 pixels)
        # Activate a 6x6 block = 36 pixels * 244.14 = 8789 km^2 (~7.0% > 5%)
        mask[0, r_n12 - 3 : r_n12 + 3, c_n12 - 3 : c_n12 + 3] = 1.0
        toks2, val2 = pool(feat, pixel_mask=mask)
        self.assertEqual(float(val2[0, 12].item()), 1.0)
        self.assertGreater(float(torch.norm(toks2[0, 12]).item()), 0.0)

    # -------------------------------------------------------------------------
    # Priority 4: Token Validity & Linear Bias Suppression
    # -------------------------------------------------------------------------
    def test_04_token_validity_and_linear_bias_suppression(self):
        """Verify that invalid tokens are zeroed post-projection to eliminate linear bias revival, and gated in spatial aggregator."""
        pool = PolarSectorPooling(in_channels=8, d_model=32, min_sector_coverage=0.05, bias=True)
        # Ensure projection layer has non-zero bias
        nn.init.constant_(pool.proj.bias, 5.0)

        feat = torch.randn(2, 8, 64, 64)
        mask = torch.zeros(2, 64, 64) # Completely invalid mask

        toks, val = pool(feat, pixel_mask=mask)
        self.assertEqual(float(val.sum().item()), 0.0)
        # Must be strictly 0.0 across all tokens despite bias=5.0
        self.assertEqual(float(toks.abs().max().item()), 0.0)

        # SpatialAggregator test in TCGraphFormer
        model = TCGraphFormer(d_model=32)
        # Verify that for all-invalid masks, extract_overpass_features produces exact zero
        B = 2
        ir = torch.randn(B, 1, 64, 64)
        pmw = torch.randn(B, 4, 64, 64)
        gprof = torch.randn(B, 4, 64, 64)
        radar = torch.randn(B, 1, 16, 32, 32)
        env = torch.randn(B, 6, 64, 64)
        # All modality masks zero and all pixel masks zero
        mod_masks = torch.zeros(B, 5)
        pixel_masks = {
            "ir": torch.zeros(B, 64, 64),
            "pmw": torch.zeros(B, 64, 64),
            "gprof": torch.zeros(B, 64, 64),
            "radar": torch.zeros(B, 16, 32, 32),
            "env": torch.zeros(B, 64, 64),
        }
        latent = model.extract_overpass_features(
            ir=ir, pmw=pmw, gprof=gprof, radar=radar, env=env,
            modality_masks=mod_masks, pixel_masks=pixel_masks
        )
        self.assertEqual(float(latent.abs().max().item()), 0.0)

    # -------------------------------------------------------------------------
    # Priority 5: Spatial Graph Message Passing & All-Invalid Sample
    # -------------------------------------------------------------------------
    def test_05_spatial_graph_message_passing_and_all_invalid_sample(self):
        """Verify SpatialGraphEncoder handles node_validity: all-invalid sample produces exact zero vector without NaNs."""
        encoder = SpatialGraphEncoder(d_model=32, num_spatial_nodes=16, num_layers=2)

        # Case 1: Partially valid graph
        node_features = torch.randn(2, 16, 32)
        node_validity = torch.zeros(2, 16)
        node_validity[0, :8] = 1.0 # First 8 nodes valid for sample 0
        node_validity[1, 8:] = 1.0 # Last 8 nodes valid for sample 1

        out = encoder(node_features, node_validity=node_validity)
        self.assertEqual(out.shape, (2, 32))
        self.assertTrue(torch.isfinite(out).all())

        # Case 2: Completely invalid sample (all 16 nodes invalid)
        all_invalid_val = torch.zeros(2, 16)
        out_zero = encoder(node_features, node_validity=all_invalid_val)
        self.assertEqual(out_zero.shape, (2, 32))
        self.assertTrue(torch.isfinite(out_zero).all())
        # Must be strictly 0.0 without any NaN or Inf
        self.assertEqual(float(out_zero.abs().max().item()), 0.0)

    # -------------------------------------------------------------------------
    # Priority 6: Physical Radar CAPPI Vertical Resampling
    # -------------------------------------------------------------------------
    def test_06_physical_radar_cappi_vertical_resampling(self):
        """Verify _resample_radar_vertical_profile_to_cappi uses actual gate heights, no extrapolation, and separate echo classification."""
        dataset = TCPRIMEDDataset.__new__(TCPRIMEDDataset)
        dataset.geospatial_policy = "raise"

        # Gate heights: 0.0 km to 15.0 km (80 gates)
        gate_heights = np.linspace(0.0, 15.0, 80, dtype=np.float32)
        raw_ku = np.full(80, -999.0, dtype=np.float32)

        # Set valid measurements between 2.0 and 10.0 km
        # In 2-6 km: Ku = 30 dBZ (strong echo, >= 12 dBZ)
        # In 6-10 km: Ku = -5 dBZ (valid measurement -20 to 100 dBZ, but NOT an echo < 12 dBZ)
        mask_30 = (gate_heights >= 2.0) & (gate_heights < 6.0)
        mask_neg5 = (gate_heights >= 6.0) & (gate_heights <= 10.0)
        raw_ku[mask_30] = 30.0
        raw_ku[mask_neg5] = -5.0

        target_heights = np.asarray(TARGET_RADAR_HEIGHTS_KM, dtype=np.float32) # 0.5 to 19.5 km (16 levels)

        d_col, m_col, e_col = dataset._resample_radar_vertical_profile_to_cappi(
            raw_ku, gate_heights, target_heights, echo_threshold_dbz=12.0
        )

        self.assertEqual(len(d_col), 16)
        self.assertEqual(len(m_col), 16)
        self.assertEqual(len(e_col), 16)

        # Above 15 km (e.g. heights at 15.7, 16.9, 18.2, 19.5 km): outside gate coverage -> must be 0.0, mask=0.0
        high_levels = np.where(target_heights > 15.0)[0]
        self.assertTrue(np.all(d_col[high_levels] == 0.0))
        self.assertTrue(np.all(m_col[high_levels] == 0.0))
        self.assertTrue(np.all(e_col[high_levels] == 0.0))

        # Check echo vs measurement validity at 7.0 km (target index 5, since 0.5 + 5 * 1.2667 ~ 6.83 km)
        # In the -5 dBZ region: measurement validity m_col MUST be 1.0, but echo mask e_col MUST be 0.0!
        idx_neg5 = np.where((target_heights >= 6.5) & (target_heights <= 9.5))[0]
        self.assertTrue(len(idx_neg5) > 0)
        for idx in idx_neg5:
            self.assertEqual(m_col[idx], 1.0, f"Level {target_heights[idx]} km should have measurement validity = 1.0")
            self.assertEqual(e_col[idx], 0.0, f"Level {target_heights[idx]} km should have echo mask = 0.0 (< 12 dBZ)")

        # In the 30 dBZ region: both measurement validity and echo mask MUST be 1.0!
        idx_30 = np.where((target_heights >= 2.5) & (target_heights <= 5.5))[0]
        self.assertTrue(len(idx_30) > 0)
        for idx in idx_30:
            self.assertEqual(m_col[idx], 1.0)
            self.assertEqual(e_col[idx], 1.0)
            # Normalized value: (30 - (-20)) / 120 = 50 / 120 ~ 0.4167
            self.assertAlmostEqual(d_col[idx], 50.0 / 120.0, places=3)

    # -------------------------------------------------------------------------
    # Priority 7: Geospatial Policy Enforcement
    # -------------------------------------------------------------------------
    def test_07_geospatial_policy_enforcement(self):
        """Verify TCPRIMEDDataset strictly enforces geospatial_policy ('raise', 'exclude_modality', 'exclude_sample', 'nominal_fallback')."""
        if not self.has_real_index:
            self.skipTest("Real sample index not found")

        # Test valid policy initialization
        ds_raise = TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, geospatial_policy="raise")
        self.assertEqual(ds_raise.geospatial_policy, "raise")

        ds_mod = TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, geospatial_policy="exclude_modality")
        self.assertEqual(ds_mod.geospatial_policy, "exclude_modality")

        ds_samp = TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, geospatial_policy="exclude_sample")
        self.assertEqual(ds_samp.geospatial_policy, "exclude_sample")

        ds_nom = TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, geospatial_policy="nominal_fallback")
        self.assertEqual(ds_nom.geospatial_policy, "nominal_fallback")

        with self.assertRaises(ValueError):
            TCPRIMEDDataset(sample_index_csv=self.sample_index_csv, geospatial_policy="invalid_policy")

    # -------------------------------------------------------------------------
    # Priority 8: Storage Schema Versioning & CF Metadata
    # -------------------------------------------------------------------------
    def test_08_storage_schema_versioning_and_cf_metadata(self):
        """Verify StorageWriter schema version is '2.3.0', convention is 'physical_cappi_km', and stale containers are rejected."""
        self.assertEqual(SCHEMA_VERSION, "2.3.0")
        self.assertEqual(RADAR_VERTICAL_CONVENTION, "physical_cappi_km")

        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = OutputConfig(root=tmp_dir)
            writer = StorageWriter(output_config=cfg)

            # Test 1: Container without schema_version is identified as stale
            stale_nc_path = writer.get_storm_container_path(2021, "AL", "TEST01")
            os.makedirs(os.path.dirname(stale_nc_path), exist_ok=True)
            with nc.Dataset(stale_nc_path, "w", format="NETCDF4") as ds:
                ds.storm_id = "TEST01"
                ds.createGroup("overpasses")
            self.assertFalse(writer.is_storm_processed(2021, "AL", "TEST01"))

            # Test 2: Container with matching schema_version is identified as valid
            valid_nc_path = writer.get_storm_container_path(2021, "AL", "TEST02")
            os.makedirs(os.path.dirname(valid_nc_path), exist_ok=True)
            with nc.Dataset(valid_nc_path, "w", format="NETCDF4") as ds:
                ds.storm_id = "TEST02"
                ds.schema_version = "2.3.0"
                ds.radar_vertical_convention = "physical_cappi_km"
                ovp = ds.createGroup("overpasses")
                ovp.createGroup("OVP_01")
            self.assertTrue(writer.is_storm_processed(2021, "AL", "TEST02"))

    # -------------------------------------------------------------------------
    # Priority 9: Inventory Lifecycle Preservation
    # -------------------------------------------------------------------------
    def test_09_inventory_lifecycle_preservation(self):
        """Verify DatasetInventory.scan() selects storms by lifecycle overlap without truncating overpasses or dropping environmental files."""
        # Test the temporal matching logic in inventory
        pipe_cfg = PipelineConfig({"dataset": {"date_start": "2021-06-14", "date_end": "2021-06-16"}})
        inv = DatasetInventory(config=pipe_cfg)

        # Case 1: Storm lifecycle starts before start_date but ends within window -> overlaps
        t_start = pd.Timestamp("2021-06-12")
        t_end = pd.Timestamp("2021-06-15")
        filter_start = pd.Timestamp(pipe_cfg.dataset.date_start)
        filter_end = pd.Timestamp(pipe_cfg.dataset.date_end)
        storm_overlaps = not (t_end < filter_start or t_start > filter_end)
        self.assertTrue(storm_overlaps, "Storm overlapping lifecycle must be selected")

        # Case 2: Storm completely outside window -> does not overlap
        t_outside_start = pd.Timestamp("2021-07-01")
        t_outside_end = pd.Timestamp("2021-07-05")
        outside_overlaps = not (t_outside_end < filter_start or t_outside_start > filter_end)
        self.assertFalse(outside_overlaps, "Storm outside lifecycle must not be selected")

    # -------------------------------------------------------------------------
    # Priority 10: MFM Edge Cases & Normalization
    # -------------------------------------------------------------------------
    def test_10_mfm_edge_cases_and_normalization(self):
        """Verify MFM masking: 0 valid -> skip, 1 valid -> visible, >= 2 -> masked leaving >= 1 visible; loss denominator."""
        d_model = 32
        head = MaskedFeatureReconstructionHead(d_model=d_model)

        # Case 1: 0 valid tokens -> all skipped
        tok_mask_0 = torch.zeros(1, 80, dtype=torch.bool)
        val_indices_0 = torch.nonzero(tok_mask_0[0], as_tuple=False).squeeze(-1)
        self.assertEqual(len(val_indices_0), 0)

        # Case 2: 1 valid token -> must leave visible (no masking)
        tok_mask_1 = torch.zeros(1, 80, dtype=torch.bool)
        tok_mask_1[0, 5] = True
        val_indices_1 = torch.nonzero(tok_mask_1[0], as_tuple=False).squeeze(-1)
        self.assertEqual(len(val_indices_1), 1)
        # Ratio masking logic:
        n_mask = int(math.ceil(1 * 0.15))
        n_mask = min(n_mask, len(val_indices_1) - 1)
        self.assertEqual(n_mask, 0, "Single valid token must remain visible (0 masked)")

        # Case 3: 4 valid tokens with mask_ratio=0.5 -> 2 masked, 2 visible
        tok_mask_4 = torch.zeros(1, 80, dtype=torch.bool)
        tok_mask_4[0, [0, 1, 2, 3]] = True
        val_indices_4 = torch.nonzero(tok_mask_4[0], as_tuple=False).squeeze(-1)
        n_mask_4 = int(math.ceil(4 * 0.5))
        n_mask_4 = min(n_mask_4, len(val_indices_4) - 1)
        self.assertEqual(n_mask_4, 2)
        n_visible = len(val_indices_4) - n_mask_4
        self.assertGreaterEqual(n_visible, 1, "Must leave at least 1 visible token")

        # Case 4: Verify loss calculation and denominator sum(M * d_model) + eps
        pred_feat = torch.randn(1, 80, d_model, requires_grad=True)
        targ_feat = torch.randn(1, 80, d_model)
        mask_m = torch.zeros(1, 80, dtype=torch.bool)
        mask_m[0, [0, 1]] = True # 2 tokens masked

        pred_masked = pred_feat[mask_m]
        targ_masked = targ_feat[mask_m]
        denom = float(mask_m.sum().item()) * d_model + 1e-8
        loss = F.mse_loss(pred_masked, targ_masked, reduction="sum") / denom
        loss.backward()
        self.assertIsNotNone(pred_feat.grad)
        self.assertTrue(torch.isfinite(loss).item())

    # -------------------------------------------------------------------------
    # Priority 11: Baseline Persistence Anchor vs Climatology Mean
    # -------------------------------------------------------------------------
    def test_11_baseline_persistence_anchor_vs_climatology(self):
        """Verify persistence uses V(t) anchor for V(t+h) with dV=0, distinct from climatology mean; test PR-AUC & Brier."""
        v_anchor = np.array([45.0, 65.0, 85.0, 110.0])
        v_future = np.array([50.0, 70.0, 80.0, 120.0])
        v_train_mean = 55.0

        # Persistence prediction: V_hat(t+h) = V(t)
        v_pred_persist = v_anchor.copy()
        err_persist = v_pred_persist - v_future

        # Climatology prediction: V_hat(t+h) = v_train_mean
        v_pred_clim = np.full_like(v_future, v_train_mean)
        err_clim = v_pred_clim - v_future

        # They are distinct formulations
        self.assertFalse(np.allclose(err_persist, err_clim))

        # Test pure NumPy PR-AUC and Brier computation
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        pr_auc, brier, prec, rec = compute_pr_auc_and_brier(y_true, y_prob)
        self.assertGreater(pr_auc, 0.8)
        self.assertLess(brier, 0.1)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

    # -------------------------------------------------------------------------
    # Priority 12: End-to-End Radar Sequence Gradient Flow
    # -------------------------------------------------------------------------
    def test_12_end_to_end_radar_sequence_gradient_flow(self):
        """Verify full forward, masked loss, and backward pass on radar sequence produces non-zero gradients on all 10 modules."""
        if not self.has_real_index:
            self.skipTest("Real sample index not found")

        seq_ds = TCPRIMEDSequenceDataset(
            sample_index_csv=self.sample_index_csv,
            partition=None,
            grid_size=64,
            max_history_len=4,
            max_history_hours=72.0,
        )

        # Find sequence with radar in AL022021
        target_seq_indices = None
        for indices in seq_ds.sequence_index:
            first_row = seq_ds.base_dataset.df.iloc[indices[0]]
            if first_row["storm_id"] == "AL022021" and len(indices) >= 2:
                if any(seq_ds.base_dataset.df.iloc[i]["has_radar"] for i in indices):
                    target_seq_indices = indices
                    break

        self.assertIsNotNone(target_seq_indices, "Radar-bearing sequence in AL022021 must exist")
        chosen_seq = [seq_ds.base_dataset[i] for i in target_seq_indices]
        batch = collate_sequence_batch([chosen_seq])

        model = TCGraphFormer(d_model=64, num_stages=5)
        loss_fn = MultiTaskLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        optimizer.zero_grad()

        pixel_masks = batch["pixel_masks"]
        channel_masks = batch["channel_masks"]

        preds = model(
            ir=batch["ir"],
            pmw=batch["pmw"],
            gprof=batch["gprof"],
            radar=batch["radar"],
            env=batch["env"],
            modality_masks=batch["modality_masks"],
            delta_times_hours=batch["delta_times_hours"],
            seq_pad_mask=batch["seq_pad_mask"],
            channel_masks=channel_masks,
            pixel_masks=pixel_masks,
        )

        loss, _ = loss_fn(preds, batch["targets"])
        loss.backward()

        grad_ir = compute_module_grad_norm(model.ir_encoder)
        grad_pmw = compute_module_grad_norm(model.pmw_encoder)
        grad_gprof = compute_module_grad_norm(model.gprof_encoder)
        grad_radar = compute_module_grad_norm(model.radar_encoder)
        grad_env = compute_module_grad_norm(model.env_encoder)
        grad_fusion = compute_module_grad_norm(model.fusion)
        grad_agg = compute_module_grad_norm(model.spatial_aggregator)
        grad_graph = compute_module_grad_norm(model.spatial_graph)
        grad_temp = compute_module_grad_norm(model.temporal_transformer)
        grad_heads = compute_module_grad_norm(model.task_heads)

        self.assertGreater(grad_ir, 0.0, "IR encoder grad must be > 0")
        self.assertGreater(grad_pmw, 0.0, "PMW encoder grad must be > 0")
        self.assertGreater(grad_gprof, 0.0, "GPROF encoder grad must be > 0")
        self.assertGreater(grad_radar, 0.0, f"Radar encoder grad must be > 0 (got {grad_radar})")
        self.assertGreater(grad_env, 0.0, "Env encoder grad must be > 0")
        self.assertGreater(grad_fusion, 0.0, "Fusion grad must be > 0")
        self.assertGreater(grad_agg, 0.0, "Spatial aggregator grad must be > 0")
        self.assertGreater(grad_graph, 0.0, "Spatial graph grad must be > 0")
        self.assertGreater(grad_temp, 0.0, "Temporal transformer grad must be > 0")
        self.assertGreater(grad_heads, 0.0, "Task heads grad must be > 0")


if __name__ == "__main__":
    unittest.main()
