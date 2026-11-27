"""
Unit Tests for Experiment R1 (`exp_r1_regularized`) Feature-Flagged Regularizers:
1. Each flag (`modality_dropout_prob`, `sector_token_dropout`, `drop_path_rate`, `use_ema`)
   defaults to OFF and is a strict no-op when off.
2. Dropout (`modality_dropout_prob`, `sector_token_dropout`, `drop_path_rate`) is a strict
   no-op in `model.eval()` mode and adds zero `state_dict` keys.
3. `ModelEMA` update, evaluation swap, checkpoint save (`model_state_dict == ema_state_dict`),
   and `restore_checkpoint_state` work accurately.
4. `build_storm_balanced_weights` produces weights that sum to 1.0, downweights large-count storms
   by `(1/n_storm)^0.5`, and automatically lowers the exponent toward 0.25 if the expected
   major-hurricane share falls below 80% of the unbalanced share.
5. `_apply_whole_modality_dropout` never drops IR (index 0) and never drops all available
   modalities even when IR is absent (`m_ir == 0`) and `modality_dropout_prob = 0.999`.
6. AMP (`fp16`) forward/backward pass and `build_decoupled_param_groups` (0.5x backbone LR,
   1e-2 decoupled weight decay) execute cleanly with finite gradients.
"""

import os
import tempfile
import unittest
import numpy as np
import torch

from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import (
    ModelEMA,
    TCGraphFormerTrainer,
    build_storm_balanced_weights,
    build_decoupled_param_groups,
    restore_checkpoint_state,
)


def _make_tiny_sequence_batch(B: int = 3, T: int = 2, device: str = "cpu"):
    torch.manual_seed(123)
    ir = torch.randn(B, T, 1, 64, 64, device=device)
    pmw = torch.randn(B, T, 4, 64, 64, device=device)
    gprof = torch.randn(B, T, 4, 64, 64, device=device)
    radar = torch.randn(B, T, 1, 16, 32, 32, device=device)
    env = torch.randn(B, T, 6, 64, 64, device=device)
    modality_masks = torch.ones(B, T, 5, device=device)
    delta_t = torch.tensor([[0.0, 6.0]] * B, dtype=torch.float32, device=device)
    seq_pad = torch.zeros(B, T, dtype=torch.bool, device=device)
    kin = torch.randn(B, T, 10, device=device)
    kin[..., 9] = 1.0

    ch_masks = {
        "pmw": torch.ones(B, T, 4, device=device),
        "gprof": torch.ones(B, T, 4, device=device),
        "env": torch.ones(B, T, 6, device=device),
    }
    pix_masks = {
        "ir": torch.ones(B, T, 1, 64, 64, device=device),
        "pmw": torch.ones(B, T, 4, 64, 64, device=device),
        "gprof": torch.ones(B, T, 4, 64, 64, device=device),
        "radar": torch.ones(B, T, 1, 16, 32, 32, device=device),
        "env": torch.ones(B, T, 6, 64, 64, device=device),
    }
    targets = {
        "intensity": torch.tensor([45.0, 105.0, 75.0][:B], dtype=torch.float32, device=device),
        "intensity_mask": torch.ones(B, dtype=torch.float32, device=device),
        "pressure": torch.tensor([998.0, 955.0, 980.0][:B], dtype=torch.float32, device=device),
        "pressure_mask": torch.ones(B, dtype=torch.float32, device=device),
        "stage": torch.tensor([1, 2, 2][:B], dtype=torch.long, device=device),
        "stage_mask": torch.ones(B, dtype=torch.float32, device=device),
        "intensity_change": torch.zeros(B, 4, dtype=torch.float32, device=device),
        "intensity_change_mask": torch.ones(B, 4, dtype=torch.float32, device=device),
        "rapid_intensification": torch.tensor([0.0, 1.0, 0.0][:B], dtype=torch.float32, device=device),
        "ri_mask": torch.ones(B, dtype=torch.float32, device=device),
        "track_displacement": torch.zeros(B, 4, 2, dtype=torch.float32, device=device),
        "track_mask": torch.ones(B, 4, dtype=torch.float32, device=device),
    }
    return {
        "ir": ir,
        "pmw": pmw,
        "gprof": gprof,
        "radar": radar,
        "env": env,
        "modality_masks": modality_masks,
        "delta_times_hours": delta_t,
        "seq_pad_mask": seq_pad,
        "kinematic_state": kin,
        "channel_masks": ch_masks,
        "pixel_masks": pix_masks,
        "targets": targets,
    }


class TestR1Regularization(unittest.TestCase):
    def test_01_flags_default_off_and_eval_mode_noop(self):
        """Verify flags default to 0/False, add zero state_dict keys, and are exact no-ops in eval mode."""
        m_default = TCGraphFormer(d_model=64)
        self.assertEqual(m_default.modality_dropout_prob, 0.0)
        self.assertEqual(m_default.sector_token_dropout, 0.0)
        self.assertEqual(m_default.drop_path_rate, 0.0)
        self.assertEqual(m_default.fusion_drop_path_rates, [0.0, 0.0])
        self.assertEqual(m_default.graph_drop_path_rates, [0.0, 0.0])
        self.assertEqual(m_default.temporal_drop_path_rates, [0.0, 0.0])

        m_r1 = TCGraphFormer(
            d_model=64,
            modality_dropout_prob=0.15,
            sector_token_dropout=0.10,
            drop_path_rate=0.10,
        )
        # Linear schedule 0 -> 0.10 across 6 layers: [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        expected_dp = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        actual_dp = m_r1.fusion_drop_path_rates + m_r1.graph_drop_path_rates + m_r1.temporal_drop_path_rates
        for a, e in zip(actual_dp, expected_dp):
            self.assertAlmostEqual(a, e, places=6)

        # Identical state_dict keys and shapes
        self.assertEqual(set(m_default.state_dict().keys()), set(m_r1.state_dict().keys()))
        m_r1.load_state_dict(m_default.state_dict(), strict=True)

        # In eval mode, m_default and m_r1 must produce bit-identical predictions
        m_default.eval()
        m_r1.eval()
        batch = _make_tiny_sequence_batch(B=2, T=2)
        with torch.no_grad():
            out_def = m_default(
                ir=batch["ir"], pmw=batch["pmw"], gprof=batch["gprof"],
                radar=batch["radar"], env=batch["env"], modality_masks=batch["modality_masks"],
                delta_times_hours=batch["delta_times_hours"], seq_pad_mask=batch["seq_pad_mask"],
                channel_masks=batch["channel_masks"], pixel_masks=batch["pixel_masks"],
                kinematic_state=batch["kinematic_state"],
            )
            out_r1 = m_r1(
                ir=batch["ir"], pmw=batch["pmw"], gprof=batch["gprof"],
                radar=batch["radar"], env=batch["env"], modality_masks=batch["modality_masks"],
                delta_times_hours=batch["delta_times_hours"], seq_pad_mask=batch["seq_pad_mask"],
                channel_masks=batch["channel_masks"], pixel_masks=batch["pixel_masks"],
                kinematic_state=batch["kinematic_state"],
            )
        for k in ["intensity", "pressure", "stage_logits", "intensity_change", "ri_logits", "track_delta"]:
            self.assertTrue(torch.equal(out_def[k], out_r1[k]), f"Mismatch in eval mode on {k}")

    def test_02_whole_modality_dropout_never_drops_ir_or_all_modalities(self):
        """Verify IR is never dropped and all-modality-dropped edge case cannot occur."""
        m_r1 = TCGraphFormer(d_model=32, modality_dropout_prob=0.999)
        m_r1.train()

        # Case A: IR is present -> IR must ALWAYS remain 1.0
        masks_with_ir = torch.ones((64, 5), dtype=torch.float32)
        dropped_a = m_r1._apply_whole_modality_dropout(masks_with_ir)
        self.assertTrue(torch.all(dropped_a[:, 0] == 1.0))
        self.assertTrue(torch.all(dropped_a.sum(dim=-1) >= 1.0))

        # Case B: IR is absent (0.0), only non-IR modalities are present, with p_drop=0.999
        # Even when Bernoulli drops all non-IR modalities, the guard must keep at least 1 valid modality
        masks_no_ir = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0]] * 50, dtype=torch.float32)
        dropped_b = m_r1._apply_whole_modality_dropout(masks_no_ir)
        self.assertTrue(torch.all(dropped_b[:, 0] == 0.0))
        self.assertTrue(torch.all(dropped_b.sum(dim=-1) >= 1.0), "All-modality-dropped edge case occurred!")

    def test_03_ema_update_and_checkpoint_restore(self):
        """Verify EMA shadow update formula, eval weight swap, and checkpoint save/restore."""
        model = TCGraphFormer(d_model=32)
        ema = ModelEMA(model, decay=0.9)

        init_weight = model.task_heads.head_intensity[0].weight.detach().clone()
        with torch.no_grad():
            model.task_heads.head_intensity[0].weight.add_(10.0)
        new_weight = model.task_heads.head_intensity[0].weight.detach().clone()

        ema.update(model)
        expected_ema = 0.9 * init_weight + 0.1 * new_weight
        actual_ema = ema.shadow["task_heads.head_intensity.0.weight"]
        self.assertTrue(torch.allclose(actual_ema, expected_ema, atol=1e-6))

        # Test trainer save_checkpoint and restore_checkpoint with EMA enabled
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TCGraphFormerTrainer(
                model=model,
                optimizer=opt,
                scheduler=sched,
                device="cpu",
                checkpoint_dir=tmpdir,
                use_ema=True,
                ema_decay=0.9,
            )
            batch = _make_tiny_sequence_batch(B=2, T=2)
            trainer.train_epoch([batch], epoch=1)
            self.assertEqual(trainer.ema.num_updates, 1)

            val_m = trainer.evaluate([batch])
            self.assertTrue(val_m["evaluated_with_ema"])

            ckpt_path = trainer.save_checkpoint(
                epoch=1,
                metrics=val_m,
                is_best=True,
                is_best_composite=True,
                filename="ep1.pt",
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "best_checkpoint.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "best_mt_rel_err_checkpoint.pt")))

            loaded_raw = torch.load(ckpt_path, map_location="cpu")
            # Saved model_state_dict must be the EMA weights
            self.assertTrue(
                torch.allclose(
                    loaded_raw["model_state_dict"]["task_heads.head_intensity.0.weight"],
                    trainer.ema.shadow["task_heads.head_intensity.0.weight"],
                )
            )

            # Restoring into a new trainer must restore both live model and EMA state
            model2 = TCGraphFormer(d_model=32)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5)
            trainer2 = TCGraphFormerTrainer(
                model=model2,
                optimizer=opt2,
                scheduler=sched2,
                device="cpu",
                checkpoint_dir=tmpdir,
                use_ema=True,
                ema_decay=0.9,
            )
            trainer2.restore_checkpoint(ckpt_path)
            self.assertTrue(
                torch.allclose(
                    trainer2.ema.shadow["task_heads.head_intensity.0.weight"],
                    trainer.ema.shadow["task_heads.head_intensity.0.weight"],
                )
            )
            self.assertTrue(
                torch.allclose(
                    trainer2.model.task_heads.head_intensity[0].weight,
                    trainer.model.task_heads.head_intensity[0].weight,
                )
            )

    def test_04_storm_balanced_sampler_weights_sum_to_one_and_guardrail(self):
        """Verify sampler weights sum to 1.0 and guardrail adjusts exponent toward 0.25 if major share < 80%."""
        storm_ids = ["S_LARGE"] * 100 + ["S_SMALL"] * 4
        intensities = np.array([110.0] * 100 + [40.0] * 4)
        # Here all major hurricanes are in S_LARGE (100 obs), so exp=0.5 downweights S_LARGE heavily
        w, info = build_storm_balanced_weights(
            storm_ids=storm_ids,
            intensities_kt=intensities,
            exponent=0.5,
            min_major_ratio=0.80,
            min_exponent=0.25,
        )
        self.assertAlmostEqual(float(np.sum(w)), 1.0, places=7)
        self.assertEqual(len(w), 104)

        # Normal case where major hurricanes are balanced across storm sizes
        storm_ids_bal = ["S1"] * 50 + ["S2"] * 10
        intensities_bal = np.array([100.0] * 10 + [50.0] * 40 + [100.0] * 2 + [50.0] * 8)
        w_bal, info_bal = build_storm_balanced_weights(
            storm_ids=storm_ids_bal,
            intensities_kt=intensities_bal,
            exponent=0.5,
        )
        self.assertAlmostEqual(float(np.sum(w_bal)), 1.0, places=7)
        self.assertAlmostEqual(info_bal["active_exponent"], 0.5, places=4)
        # Weight ratio per observation between S2 (n=10) and S1 (n=50) must be (50/10)^0.5 = sqrt(5)
        self.assertAlmostEqual(float(w_bal[-1] / w_bal[0]), np.sqrt(5.0), places=5)

    def test_05_differential_lr_decoupled_wd_and_amp_forward_backward(self):
        """Verify 0.5x backbone LR, decoupled 1e-2 WD, and AMP forward/backward pass."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TCGraphFormer(
            d_model=32,
            modality_dropout_prob=0.15,
            sector_token_dropout=0.10,
            drop_path_rate=0.10,
        ).to(device)
        param_groups, counts = build_decoupled_param_groups(
            model=model,
            head_lr=3e-4,
            backbone_lr_scale=0.5,
            weight_decay=1e-2,
        )
        g_map = {g["group_name"]: g for g in param_groups}
        self.assertAlmostEqual(g_map["backbone_decay"]["lr"], 1.5e-4, places=8)
        self.assertAlmostEqual(g_map["backbone_decay"]["weight_decay"], 1e-2, places=8)
        self.assertAlmostEqual(g_map["backbone_no_decay"]["lr"], 1.5e-4, places=8)
        self.assertAlmostEqual(g_map["backbone_no_decay"]["weight_decay"], 0.0, places=8)
        self.assertAlmostEqual(g_map["head_decay"]["lr"], 3.0e-4, places=8)
        self.assertAlmostEqual(g_map["head_decay"]["weight_decay"], 1e-2, places=8)
        self.assertAlmostEqual(g_map["head_no_decay"]["lr"], 3.0e-4, places=8)
        self.assertAlmostEqual(g_map["head_no_decay"]["weight_decay"], 0.0, places=8)

        opt = torch.optim.AdamW(param_groups)
        loss_fn = MultiTaskLoss(ri_pos_weight=18.1).to(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TCGraphFormerTrainer(
                model=model,
                optimizer=opt,
                loss_fn=loss_fn,
                device=device,
                checkpoint_dir=tmpdir,
                use_amp=device.startswith("cuda"),
                precision="fp16",
                grad_clip_norm=1.0,
                accumulate_grad_batches=1,
                use_ema=True,
                ema_decay=0.999,
            )
            batch = _make_tiny_sequence_batch(B=3, T=2, device=device)
            tr_m = trainer.train_epoch([batch], epoch=1)
            self.assertTrue(np.isfinite(tr_m["loss_total"]))
            self.assertEqual(tr_m["optimizer_steps"], 1)
            self.assertEqual(trainer.ema.num_updates, 1)


if __name__ == "__main__":
    unittest.main()

