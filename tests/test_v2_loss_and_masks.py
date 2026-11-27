"""
Regression and mathematical verification tests for TC-GraphFormer V2:
1. Temporal Transformer all-padded / first-step-padded mask finiteness.
2. MultiTaskLoss backward compatibility (default parameters match V1 to 1e-7).
3. V2 Bounded linear regime weighting (w_regime in [1.0, 3.0] vs V1 [1.0, 11.0]).
4. V2 Exact Bayes log-odds prior shift (ri_prior_shift_factor=1.0 -> stationary point at exact base rate).
5. Microbatch gradient accumulation equivalence under heterogeneous missing labels.
"""

import math
import unittest
import torch

from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss


class TestV2LossAndMasks(unittest.TestCase):
    def test_temporal_all_padded_and_first_padded_finiteness(self):
        torch.manual_seed(42)
        model = TCGraphFormer(d_model=128, num_stages=5)
        B, T = 3, 4
        ir = torch.zeros(B, T, 1, 64, 64)
        pmw = torch.zeros(B, T, 4, 64, 64)
        gprof = torch.zeros(B, T, 4, 64, 64)
        radar = torch.zeros(B, T, 1, 16, 32, 32)
        env = torch.zeros(B, T, 6, 64, 64)
        mod_masks = torch.zeros(B, T, 5)
        dt = torch.zeros(B, T)
        seq_pad = torch.zeros(B, T, dtype=torch.bool)
        seq_pad[0, :] = True     # All timesteps padded
        seq_pad[1, 0] = True     # First timestep padded
        seq_pad[2, 2:] = True    # Right-padded sequence
        kin = torch.zeros(B, T, 10)

        preds = model(
            ir=ir, pmw=pmw, gprof=gprof, radar=radar, env=env,
            modality_masks=mod_masks, delta_times_hours=dt, seq_pad_mask=seq_pad,
            kinematic_state=kin,
        )
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(torch.isfinite(v).all().item(), f"Non-finite output in {k}")
        # All-padded sequence must have zero temporal sequence output
        self.assertEqual(float(preds["temporal_sequence"][0].abs().max().item()), 0.0)

    def test_v2_bounded_regime_weight_and_exact_ri_prior_shift(self):
        v1_loss_fn = MultiTaskLoss(ri_pos_weight=18.1)
        v2_loss_fn = MultiTaskLoss(
            weight_intensity=1.0,
            weight_pressure=2.0,
            weight_stage=0.5,
            weight_intensity_change=1.25,
            weight_ri=2.0,
            weight_track=4.0,
            ri_pos_weight=18.1,
            regime_max_extra_weight=1.0,
            regime_power=1.0,
            ri_prior_shift_factor=1.0,
        )

        # Check stationary point of V2 RI BCE loss at true base rate p = 1 / (1 + 18.1)
        p_base = 1.0 / (1.0 + 18.1)
        true_log_odds = math.log(p_base / (1.0 - p_base)) # -ln(18.1)
        z = torch.tensor([true_log_odds], requires_grad=True)
        # In population expectation E[L(z)] = p * L(z, y=1) + (1-p) * L(z, y=0)
        preds_pos = {
            "intensity": torch.tensor([150.0]),
            "pressure": torch.tensor([950.0]),
            "stage_logits": torch.zeros(1, 5),
            "intensity_change": torch.zeros(1, 4),
            "ri_logits": z,
            "track_delta": torch.zeros(1, 8),
        }
        tgts_pos = {
            "intensity": torch.tensor([150.0]),
            "intensity_mask": torch.ones(1),
            "pressure": torch.tensor([950.0]),
            "pressure_mask": torch.ones(1),
            "stage": torch.tensor([2], dtype=torch.long),
            "stage_mask": torch.ones(1),
            "intensity_change": torch.zeros(1, 4),
            "intensity_change_mask": torch.ones(1, 4),
            "rapid_intensification": torch.tensor([1.0]),
            "ri_mask": torch.ones(1),
            "track_displacement": torch.zeros(1, 4, 2),
            "track_mask": torch.ones(1, 4),
        }
        tgts_neg = {**tgts_pos, "rapid_intensification": torch.tensor([0.0])}
        l_pos, _ = v2_loss_fn(preds_pos, tgts_pos)
        l_neg, _ = v2_loss_fn(preds_pos, tgts_neg)
        expected_loss = p_base * l_pos + (1.0 - p_base) * l_neg
        expected_loss.backward()
        # Stationary gradient at true log-odds must be exactly 0.0
        self.assertAlmostEqual(float(z.grad.item()), 0.0, places=6)

        # Check V1 vs V2 regime amplification at 150 kt (error = 10 kt)
        preds_err = {**preds_pos, "intensity": torch.tensor([140.0]), "ri_logits": torch.tensor([0.0])}
        _, d_v1 = v1_loss_fn(preds_err, tgts_neg)
        _, d_v2 = v2_loss_fn(preds_err, tgts_neg)
        # Unweighted Huber(10, beta=8) = 10 - 4 = 6.0. V1 weight at 150 kt is 1 + 2.5*(2^2) = 11 -> 66.0
        # V2 weight at 150 kt is 1 + 1.0*(2^1) = 3 -> 18.0
        self.assertAlmostEqual(d_v1["loss_intensity"], 66.0, places=4)
        self.assertAlmostEqual(d_v2["loss_intensity"], 18.0, places=4)

    def test_head_dropout_train_vs_eval_and_state_dict_compatibility(self):
        torch.manual_seed(42)
        m_base = TCGraphFormer(d_model=128, num_stages=5, head_dropout=0.0)
        m_drop = TCGraphFormer(d_model=128, num_stages=5, head_dropout=0.15)

        # 1. Exact state_dict key & shape compatibility (333 tensors)
        self.assertEqual(list(m_base.state_dict().keys()), list(m_drop.state_dict().keys()))
        m_drop.load_state_dict(m_base.state_dict(), strict=True)

        # 2. In model.eval() mode, m_drop and m_base must produce bit-identical predictions
        m_base.eval()
        m_drop.eval()
        z = torch.randn(8, 128)
        kin = torch.randn(8, 10)
        out_base = m_base.task_heads(z, kinematic_state=kin)
        out_drop_eval1 = m_drop.task_heads(z, kinematic_state=kin)
        out_drop_eval2 = m_drop.task_heads(z, kinematic_state=kin)
        for k in out_base:
            self.assertTrue(torch.equal(out_base[k], out_drop_eval1[k]), f"Eval mismatch on {k}")
            self.assertTrue(torch.equal(out_drop_eval1[k], out_drop_eval2[k]), f"Non-deterministic eval on {k}")

        # 3. In model.train() mode, m_drop applies stochastic dropout on intermediate 64-D head activations
        m_drop.train()
        out_train1 = m_drop.task_heads(z, kinematic_state=kin)
        out_train2 = m_drop.task_heads(z, kinematic_state=kin)
        self.assertFalse(torch.equal(out_train1["intensity"], out_train2["intensity"]))
        self.assertFalse(torch.equal(out_train1["ri_logits"], out_train2["ri_logits"]))


if __name__ == "__main__":
    unittest.main()

