"""
Unit tests for `src.evaluation.canonical_eval`:
Verifies single-path evaluation metrics, FINAL_TEST guard, and paired storm_bootstrap
handling of zero-positive RI resamples.
"""

import unittest
import numpy as np
from src.evaluation.canonical_eval import evaluate, storm_bootstrap


class TestCanonicalEval(unittest.TestCase):
    def test_01_final_test_is_blocked(self):
        with self.assertRaises(PermissionError):
            evaluate("checkpoints/tcprimed_100gb/best_checkpoint.pt", split="test")
        with self.assertRaises(PermissionError):
            evaluate("checkpoints/tcprimed_100gb/best_checkpoint.pt", split="FINAL_TEST")

    def test_02_storm_bootstrap_drops_and_counts_zero_positive_ri_resamples(self):
        # Construct 4 synthetic storms where only 1 storm has RI positives, so some resamples have 0 positives
        storm_ids = np.array(["S1"] * 5 + ["S2"] * 5 + ["S3"] * 5 + ["S4"] * 5)
        sample_ids = np.array([f"SMP_{i}" for i in range(20)])
        ri_true = np.zeros(20, dtype=np.float32)
        ri_true[:2] = 1.0  # Only S1 has positives

        base_payload = {
            "sample_ids": sample_ids,
            "storm_ids": storm_ids,
            "v_pred": np.full(20, 50.0, dtype=np.float32),
            "v_true": np.full(20, 55.0, dtype=np.float32),
            "v_mask": np.ones(20, dtype=bool),
            "maj_mask": np.array([True] * 5 + [False] * 15, dtype=bool),
            "p_pred": np.full(20, 990.0, dtype=np.float32),
            "p_true": np.full(20, 995.0, dtype=np.float32),
            "p_mask": np.ones(20, dtype=bool),
            "s_pred": np.zeros(20, dtype=np.int64),
            "s_true": np.zeros(20, dtype=np.int64),
            "s_mask": np.ones(20, dtype=bool),
            "dv_pred": np.zeros((20, 4), dtype=np.float32),
            "dv_true": np.ones((20, 4), dtype=np.float32),
            "dv_mask": np.ones((20, 4), dtype=bool),
            "ri_prob": np.linspace(0.9, 0.05, 20, dtype=np.float32),
            "ri_true": ri_true,
            "ri_mask": np.ones(20, dtype=bool),
            "tr_mask": np.ones((20, 4), dtype=bool),
            "track_err_nn": np.full((20, 4), 100.0, dtype=np.float32),
            "track_err_vel": np.full((20, 4), 110.0, dtype=np.float32),
        }
        preds_a = dict(base_payload)
        preds_b = dict(base_payload)
        preds_b["v_pred"] = np.full(20, 48.0, dtype=np.float32)

        res = storm_bootstrap(preds_a, preds_b, n=200, seed=0)
        self.assertGreater(res["ri_zero_positive_resamples_dropped"], 0)
        self.assertEqual(
            res["ri_zero_positive_resamples_dropped"] + res["ri_valid_resamples_used"],
            200,
        )
        self.assertEqual(
            res["paired_deltas_a_minus_b"]["ri_pr_auc"]["dropped_resamples"],
            res["ri_zero_positive_resamples_dropped"],
        )
        self.assertAlmostEqual(
            res["paired_deltas_a_minus_b"]["wind_mae_kt"]["point_delta_a_minus_b"],
            -2.0,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()

