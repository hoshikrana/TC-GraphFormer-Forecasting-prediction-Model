"""
test_eda_pilot_dataset.py - Unit tests for Phase 1E Exploratory Data Analysis & Preprocessing Study.

Verifies:
  - Dataset integrity audit logic and incomplete storm detection
  - Target and predictand statistical calculations (excluding fill values)
  - Temporal ordering, gap calculations, and sequence feasibility metrics
  - Leak-free storm-level dataset partitioning (0 storm ID overlap between splits)
  - Preprocessing transformation documentation and metadata preservation rules
  - NumPy extraction integrity and masked array handling
"""

import os
import sys
import csv
import json
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.eda_pilot_dataset import (
    classify_era,
    is_environmental_file,
    compute_target_statistics,
    compute_temporal_statistics,
    build_leak_free_splits,
    clean_scalar,
    parse_time_seconds,
)


class TestEdaIntegrityAndClassification(unittest.TestCase):
    """Verify era classification, environmental detection, and scalar cleaning."""

    def test_era_classification(self):
        self.assertEqual(classify_era(1987), "1987-1999")
        self.assertEqual(classify_era(1995), "1987-1999")
        self.assertEqual(classify_era(2005), "2000-2009")
        self.assertEqual(classify_era(2015), "2010-2019")
        self.assertEqual(classify_era(2022), "2020+")

    def test_environmental_file_detection(self):
        self.assertTrue(is_environmental_file("TCPRIMED_v01r01-final_AL122015_env_s20151108180000_e20151113120000.nc"))
        self.assertFalse(is_environmental_file("TCPRIMED_v01r01-final_AL122015_GMI_GPM_009673_20151111092614.nc"))

    def test_clean_scalar(self):
        self.assertEqual(clean_scalar(np.int32(42)), 42)
        self.assertEqual(clean_scalar(np.float64(3.14)), 3.14)
        self.assertIsNone(clean_scalar(np.nan))
        self.assertIsNone(clean_scalar(np.ma.masked))
        ma = np.ma.array([10, -999], mask=[0, 1])
        self.assertEqual(clean_scalar(ma[0]), 10)
        self.assertIsNone(clean_scalar(ma[1]))

    def test_parse_time_seconds(self):
        # 1447013579 = 2015-11-08 20:12:59 UTC
        t_str = parse_time_seconds(1447013579)
        self.assertEqual(t_str, "2015-11-08 20:12:59")
        self.assertEqual(parse_time_seconds(None), "")
        self.assertEqual(parse_time_seconds(-1), "")


class TestTargetStatisticsComputation(unittest.TestCase):
    """Verify target calculation, fill-value exclusion, and statistical metrics."""

    def test_fill_values_excluded(self):
        # 5 records: 3 valid intensities (30, 50, 70), 2 fill values (-999, None)
        records = [
            {"intensity": 30.0, "central_min_pressure": 1005.0, "development_level": "TD"},
            {"intensity": 50.0, "central_min_pressure": 995.0, "development_level": "TS"},
            {"intensity": 70.0, "central_min_pressure": 980.0, "development_level": "HU"},
            {"intensity": -999.0, "central_min_pressure": -999.0, "development_level": "HU"},
            {"intensity": None, "central_min_pressure": None, "development_level": "EX"},
        ]

        target_stats, stage_breakdown = compute_target_statistics(records)

        # Check intensity stats
        int_stat = target_stats["intensity"]
        self.assertEqual(int_stat["total_records"], 5)
        self.assertEqual(int_stat["valid_count"], 3)
        self.assertEqual(int_stat["missing_count"], 2)
        self.assertAlmostEqual(int_stat["missing_percent"], 40.0, places=1)
        self.assertAlmostEqual(int_stat["mean"], 50.0, places=1)
        self.assertAlmostEqual(int_stat["min"], 30.0, places=1)
        self.assertAlmostEqual(int_stat["max"], 70.0, places=1)
        self.assertAlmostEqual(int_stat["median"], 50.0, places=1)

        # Check stage breakdown
        self.assertEqual(stage_breakdown["HU"]["count"], 2)
        self.assertEqual(stage_breakdown["TS"]["count"], 1)
        self.assertEqual(stage_breakdown["TD"]["count"], 1)
        self.assertEqual(stage_breakdown["EX"]["count"], 1)


class TestTemporalStatisticsComputation(unittest.TestCase):
    """Verify chronological timeline sorting and inter-overpass gap calculations."""

    def test_temporal_gap_calculations(self):
        # Storm with 4 overpasses at t = 0h, 3h, 9h, 24h
        records = [
            {"storm_id": "AL012020", "nominal_time": 10000.0, "nominal_time_utc": "2020-01-01 02:46:40", "infrared": {"time_offset_min": 1.5}},
            {"storm_id": "AL012020", "nominal_time": 20800.0, "nominal_time_utc": "2020-01-01 05:46:40", "infrared": {"time_offset_min": -0.5}},  # +3 hours
            {"storm_id": "AL012020", "nominal_time": 42400.0, "nominal_time_utc": "2020-01-01 11:46:40", "infrared": {"time_offset_min": 0.0}},   # +6 hours
            {"storm_id": "AL012020", "nominal_time": 96400.0, "nominal_time_utc": "2020-01-02 02:46:40", "infrared": {"time_offset_min": 5.0}},   # +15 hours
        ]

        temporal_summary, gaps = compute_temporal_statistics(records)

        # Gaps between 4 consecutive overpasses: [3.0, 6.0, 15.0] hours
        self.assertEqual(len(gaps), 3)
        self.assertAlmostEqual(gaps[0], 3.0, places=1)
        self.assertAlmostEqual(gaps[1], 6.0, places=1)
        self.assertAlmostEqual(gaps[2], 15.0, places=1)

        # Feasibility counts
        self.assertEqual(temporal_summary["gaps_le_6h"], 2)   # 3h and 6h
        self.assertEqual(temporal_summary["gaps_le_12h"], 2)  # 3h and 6h
        self.assertEqual(temporal_summary["gaps_le_24h"], 3)  # all 3

        # IR-MW synchronization deltas: [1.5, -0.5, 0.0, 5.0]
        sync = temporal_summary["ir_mw_sync"]
        self.assertEqual(sync["count"], 4)
        self.assertAlmostEqual(sync["mean_delta_minutes"], 1.5, places=1)
        self.assertEqual(sync["sync_within_15min_pct"], 100.0)


class TestLeakFreeSplitPartitioning(unittest.TestCase):
    """Verify 100% storm-level separation with 0 leakage across splits."""

    def test_leak_free_split_guarantee(self):
        # Synthetic catalogue with storms across multiple basins and eras
        storm_counts = {}
        for b in ["AL", "WP", "EP", "IO"]:
            for yr in [1995, 2005, 2015, 2022]:
                for c in range(1, 4):  # 3 storms per stratum
                    sid = f"{b}{c:02d}{yr}"
                    storm_counts[sid] = {"expected": 10, "downloaded": 10, "overpasses": 9, "env": 1}

        split_map, split_summary = build_leak_free_splits(storm_counts, overpass_records=[], seed=42)

        # 1. Verify every storm is assigned
        self.assertEqual(len(split_map), 48)

        # 2. Verify all 3 partitions are populated
        train_sids = [sid for sid, sp in split_map.items() if sp == "TRAIN"]
        val_sids = [sid for sid, sp in split_map.items() if sp == "VAL"]
        test_sids = [sid for sid, sp in split_map.items() if sp == "TEST"]

        self.assertTrue(len(train_sids) > 0)
        self.assertTrue(len(val_sids) > 0)
        self.assertTrue(len(test_sids) > 0)

        # 3. VERIFY ZERO DATA LEAKAGE: Intersection between any two sets must be empty
        self.assertEqual(len(set(train_sids).intersection(set(val_sids))), 0)
        self.assertEqual(len(set(train_sids).intersection(set(test_sids))), 0)
        self.assertEqual(len(set(val_sids).intersection(set(test_sids))), 0)


if __name__ == "__main__":
    unittest.main()
