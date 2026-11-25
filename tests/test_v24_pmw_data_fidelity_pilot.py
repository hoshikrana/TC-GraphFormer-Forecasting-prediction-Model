"""
Unit tests for Phase 3 Versioned Data-Fidelity Recovery Pilot (v2.4 S3..S6 PMW 36.5 & 85-91 GHz Channels).
Verifies:
  1. Backward compatibility of v2.3 schema, cache keys, and existing S24 checkpoints.
  2. Versioned v2.4 channel mapping across all 9 satellite sensor families without mixing incompatible frequencies.
  3. Recovery of S3..S6 swaths (ATMS S3/TB_88.2QV, TMI S3/TB_85.5V, SSMIS S4/TB_91.665V,
     AMSR2 S4/TB_36.5V + S5/TB_A89.0V, AMSRE S4/TB_36.5V + S6/TB_B89.0V fallback) on representative TRAIN samples.
  4. Scientific units ([100, 320] K -> [0, 1]), input shapes ((4, 64, 64)), sensor masks (pmw_channel_mask, modality_masks),
     geospatial resampling provenance (tangent_plane_geodetic_km / utm_storm_centered_km), causal timestamps,
     numerical finite values, and GPU forward/backward pass + S24 checkpoint state_dict compatibility.
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch

from src.data.dataset import (
    TCPRIMEDDataset,
    TCPRIMEDSequenceDataset,
    SENSOR_PMW_CHANNEL_MAP_V23,
    SENSOR_PMW_CHANNEL_MAP_V24,
)
from src.data.collate import collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss


class TestV24PMWDataFidelityPilot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_index_csv = "reports/pipeline_100gb/sample_index.csv"
        cls.df_train = pd.read_csv(cls.sample_index_csv)
        cls.df_train = cls.df_train[cls.df_train["partition"] == "TRAIN"].reset_index(drop=True)

    def test_01_channel_map_frequency_integrity_and_versioning(self):
        """Verify v2.3 vs v2.4 channel mappings and confirm no incompatible frequencies (23.8/31.4 GHz) enter Ch 0-2."""
        self.assertEqual(TCPRIMEDDataset.CACHE_PREPROCESSING_VERSION, "v3_strict_causal_kin_cappi16_grid64")
        self.assertEqual(TCPRIMEDDataset.CACHE_PREPROCESSING_VERSION_V24, "v4_pmw_s3s6_recovered_cappi16_grid64")

        # ATMS has no 19 GHz or 37 GHz channels (S1=23.8QV, S2=31.4QV must remain None in Ch 0, 1, 2)
        self.assertIsNone(SENSOR_PMW_CHANNEL_MAP_V24["ATMS"][0])
        self.assertIsNone(SENSOR_PMW_CHANNEL_MAP_V24["ATMS"][1])
        self.assertIsNone(SENSOR_PMW_CHANNEL_MAP_V24["ATMS"][2])
        self.assertEqual(SENSOR_PMW_CHANNEL_MAP_V24["ATMS"][3], ("S3", "TB_88.2QV"))

        # TMI, SSMIS, AMSR2, AMSRE recover 36.5 GHz and 85.5-91.665 GHz
        self.assertEqual(SENSOR_PMW_CHANNEL_MAP_V24["TMI"][3], ("S3", "TB_85.5V"))
        self.assertEqual(SENSOR_PMW_CHANNEL_MAP_V24["SSMIS"][3], ("S4", "TB_91.665V"))
        self.assertEqual(SENSOR_PMW_CHANNEL_MAP_V24["AMSR2"][2], ("S4", "TB_36.5V"))
        self.assertEqual(SENSOR_PMW_CHANNEL_MAP_V24["AMSRE"][2], ("S4", "TB_36.5V"))
        self.assertIsInstance(SENSOR_PMW_CHANNEL_MAP_V24["AMSRE"][3], list)

    def test_02_train_subset_sensor_recovery_and_geospatial_provenance(self):
        """
        Verify on a representative TRAIN-only subset spanning all 5 affected sensors (ATMS, TMI, SSMIS, AMSR2, AMSRE)
        that v2.3 masks the missing channels while v2.4 recovers valid physical [0, 1] brightness temperatures
        with coordinate-registered geospatial resampling and non-Simultaneous preservation of GPROF/IR/Env/Targets.
        """
        target_sensors = {
            "ATMS": [0.0, 0.0, 0.0, 1.0],
            "TMI": [1.0, 1.0, 1.0, 1.0],
            "SSMIS": [1.0, 1.0, 1.0, 1.0],
            "AMSR2": [1.0, 1.0, 1.0, 1.0],
            "AMSRE": [1.0, 1.0, 1.0, 1.0],
        }
        selected_rows = []
        for inst in target_sensors:
            sub = self.df_train[self.df_train["instrument"] == inst]
            self.assertGreater(len(sub), 0, f"No TRAIN rows found for {inst}")
            selected_rows.append(sub.iloc[0])
        df_sub = pd.DataFrame(selected_rows).reset_index(drop=True)

        ds_v23 = TCPRIMEDDataset(
            sample_index_csv=self.sample_index_csv,
            partition="TRAIN",
            df=df_sub,
            pmw_schema_version="v2.3",
            disk_cache_dir=None,
        )
        ds_v23.enable_memory_cache = False
        ds_v23.enable_disk_cache = False

        ds_v24 = TCPRIMEDDataset(
            sample_index_csv=self.sample_index_csv,
            partition="TRAIN",
            df=df_sub,
            pmw_schema_version="v2.4",
            disk_cache_dir=None,
        )
        ds_v24.enable_memory_cache = False
        ds_v24.enable_disk_cache = False

        for i, inst in enumerate(target_sensors.keys()):
            s23 = ds_v23[i]
            s24 = ds_v24[i]

            # Shape and finiteness checks
            self.assertEqual(tuple(s24["pmw"].shape), (4, 64, 64))
            self.assertEqual(tuple(s24["masks"]["pmw_pixel"].shape), (4, 64, 64))
            self.assertEqual(tuple(s24["masks"]["pmw_channels"].shape), (4,))
            self.assertTrue(torch.isfinite(s24["pmw"]).all().item())

            # Normalized brightness temperatures must lie in [0.0, 1.0]
            self.assertGreaterEqual(float(s24["pmw"].min().item()), 0.0)
            self.assertLessEqual(float(s24["pmw"].max().item()), 1.0)

            # In v2.3, Channel 3 (85-91 GHz) was 0.0 for all 5 of these sensors
            self.assertEqual(float(s23["masks"]["pmw_channels"][3].item()), 0.0)
            self.assertEqual(float(s23["pmw"][3].abs().sum().item()), 0.0)

            # In v2.4, Channel 3 (and Channel 2 for AMSR2/AMSRE) is recovered with valid non-zero pixels
            expected_ch_mask = torch.tensor(target_sensors[inst], dtype=torch.float32)
            self.assertTrue(
                torch.equal(s24["masks"]["pmw_channels"], expected_ch_mask),
                f"{inst} v2.4 channel mask {s24['masks']['pmw_channels'].tolist()} != {expected_ch_mask.tolist()}",
            )
            self.assertEqual(float(s24["modality_masks"][1].item()), 1.0, f"{inst} v2.4 has_pmw must be 1.0")
            self.assertGreater(float(s24["pmw"][3].sum().item()), 5.0, f"{inst} v2.4 Ch3 must have non-zero TBs")
            self.assertGreater(float(s24["masks"]["pmw_pixel"][3].sum().item()), 50.0)

            # Geospatial provenance must be coordinate-registered (not nominal_fallback)
            self.assertIn(
                s24["geospatial_provenance"]["passive_microwave"],
                ("tangent_plane_geodetic_km", "utm_storm_centered_km"),
            )

            # Existing valid Ch 0/1 in TMI/SSMIS/AMSR2/AMSRE and other modalities must be preserved identically
            if inst in ("TMI", "SSMIS", "AMSR2", "AMSRE"):
                self.assertTrue(torch.allclose(s23["pmw"][0], s24["pmw"][0], atol=1e-6))
                self.assertTrue(torch.allclose(s23["pmw"][1], s24["pmw"][1], atol=1e-6))
            self.assertTrue(torch.allclose(s23["ir"], s24["ir"], atol=1e-6))
            self.assertTrue(torch.allclose(s23["gprof"], s24["gprof"], atol=1e-6))
            self.assertTrue(torch.allclose(s23["env"], s24["env"], atol=1e-6))
            self.assertTrue(torch.allclose(s23["kinematic_state"], s24["kinematic_state"], atol=1e-6))

    def test_03_causal_sequence_gpu_forward_backward_and_s24_checkpoint_compat(self):
        """
        Verify causal sequence timestamps, GPU forward/backward pass, finite gradients on PMWEncoder,
        and 100% state_dict compatibility with saved S24 Epoch 10 checkpoint.
        """
        # Select 1 TRAIN storm with ATMS + TMI/SSMIS/AMSR2 overpasses
        df_storm = self.df_train[self.df_train["storm_id"] == "AL062018"].iloc[:12].copy().reset_index(drop=True)
        seq_ds = TCPRIMEDSequenceDataset(
            sample_index_csv=self.sample_index_csv,
            partition="TRAIN",
            df=df_storm,
            max_history_len=4,
            max_history_hours=72.0,
            pmw_schema_version="v2.4",
            disk_cache_dir=None,
        )
        seq_ds.base_dataset.enable_memory_cache = False
        seq_ds.base_dataset.enable_disk_cache = False

        batch_items = [seq_ds[i] for i in range(min(4, len(seq_ds)))]
        for seq in batch_items:
            epochs = [obs["timestamp_epoch"] for obs in seq]
            self.assertEqual(epochs, sorted(epochs), "Sequence timestamps must be strictly chronological (causal)")

        batch = collate_sequence_batch(batch_items)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TCGraphFormer(d_model=128, num_stages=5, head_dropout=0.0).to(device)
        s24_ckpt = "checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt"
        if os.path.exists(s24_ckpt):
            ckpt = torch.load(s24_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=True)

        loss_fn = MultiTaskLoss().to(device)
        model.train()
        preds = model(
            ir=batch["ir"].to(device),
            pmw=batch["pmw"].to(device),
            gprof=batch["gprof"].to(device),
            radar=batch["radar"].to(device),
            env=batch["env"].to(device),
            modality_masks=batch["modality_masks"].to(device),
            delta_times_hours=batch["delta_times_hours"].to(device),
            seq_pad_mask=batch["seq_pad_mask"].to(device),
            channel_masks={k: v.to(device) for k, v in batch["channel_masks"].items()},
            pixel_masks={k: v.to(device) for k, v in batch["pixel_masks"].items()},
            kinematic_state=batch["kinematic_state"].to(device),
        )
        targets = {k: v.to(device) for k, v in batch["targets"].items() if isinstance(v, torch.Tensor)}
        loss, _ = loss_fn(preds, targets)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()

        pmw_stem_grad = model.pmw_encoder.stem[0].weight.grad
        self.assertIsNotNone(pmw_stem_grad)
        self.assertTrue(torch.isfinite(pmw_stem_grad).all().item())
        self.assertGreater(float(pmw_stem_grad[:, 3].abs().sum().item()), 0.0, "PMW Channel 3 must receive non-zero gradient")


if __name__ == "__main__":
    unittest.main()

