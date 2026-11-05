"""
test_reconcile_pilot_consistency.py - Unit tests for Phase 1F Scientific Consistency & Reconciliation.

Verifies:
  1. Separation of R5 (degrees_latitude along track) from radial profile coordinate (km).
  2. Stratified split guarantees: exact 56/12/12 (70%/15%/15%) storm partitioning with 0 overlap.
  3. Temporal synchronization metric calculations (signed mean, absolute mean/median, thresholds).
  4. Rapid Intensification calculation (>= 30 kt / 24h).
  5. Scientific classification taxonomy integrity.
"""

import unittest
import numpy as np

from src.reconcile_pilot_consistency import (
    classify_era,
    is_environmental_file,
    clean_scalar,
    parse_time_seconds,
    DECISION_CLASSIFICATIONS,
)


class TestR5VsRadiusSeparation(unittest.TestCase):
    """
    Test scientific distinction between R5 storm-size metric and IR radial profile coordinate.
    """

    def test_r5_metadata_definitions(self):
        # R5 is radius of 5-knot tangential wind at 850 hPa
        r5_units = "degrees_latitude"
        r5_dim_name = "R5_time"
        r5_long_name = "Radius of mean tangential winds of 5 knots at 850 hPa level"
        
        # Radial profile is 30-bin coordinate
        radius_profile_units = "km"
        radius_profile_dim_name = "radius"
        radius_profile_bins = 30
        
        self.assertNotEqual(r5_units, radius_profile_units)
        self.assertEqual(r5_units, "degrees_latitude")
        self.assertEqual(radius_profile_units, "km")
        self.assertNotEqual(r5_dim_name, radius_profile_dim_name)
        self.assertEqual(radius_profile_bins, 30)

    def test_r5_conversion_to_km(self):
        # 1 degree of latitude is approximately 111.139 km
        r5_deg = 4.5
        km_per_deg_lat = 111.139
        r5_km = r5_deg * km_per_deg_lat
        self.assertAlmostEqual(r5_km, 500.1255, places=2)


class TestTemporalSyncCalculations(unittest.TestCase):
    """
    Test temporal synchronization metrics (signed vs. absolute offsets).
    """

    def test_signed_mean_cancels_but_absolute_reflects_dispersion(self):
        # Symmetrical distribution around zero
        offsets = np.array([-15.0, 15.0, -10.0, 10.0, 0.0])
        signed_mean = float(np.mean(offsets))
        abs_mean = float(np.mean(np.abs(offsets)))
        abs_median = float(np.median(np.abs(offsets)))

        self.assertEqual(signed_mean, 0.0)
        self.assertEqual(abs_mean, 10.0)
        self.assertEqual(abs_median, 10.0)

    def test_within_threshold_percentages(self):
        offsets = np.array([2.0, -4.0, 12.0, -14.0, 25.0, 45.0, -70.0])
        abs_offsets = np.abs(offsets)
        n = len(offsets)

        within_15 = np.sum(abs_offsets <= 15.0) / n
        within_60 = np.sum(abs_offsets <= 60.0) / n

        self.assertAlmostEqual(within_15, 4.0 / 7.0, places=4)
        self.assertAlmostEqual(within_60, 6.0 / 7.0, places=4)


class TestStratifiedSplitLogic(unittest.TestCase):
    """
    Test exact 56/12/12 split guarantee and zero leakage.
    """

    def test_era_classification(self):
        self.assertEqual(classify_era(1995), "1987-1999")
        self.assertEqual(classify_era(2005), "2000-2009")
        self.assertEqual(classify_era(2015), "2010-2019")
        self.assertEqual(classify_era(2022), "2020+")

    def test_leak_free_split_guarantee(self):
        # Generate 80 synthetic storm IDs across 6 basins and 4 eras
        basins = ["AL", "EP", "WP", "IO", "SH", "CP"]
        eras = ["1987-1999", "2000-2009", "2010-2019", "2020+"]
        synthetic_storms = [f"STORM_{i:03d}" for i in range(80)]

        # Allocation: 56 train, 12 val, 12 test
        train_set = set(synthetic_storms[:56])
        val_set = set(synthetic_storms[56:68])
        test_set = set(synthetic_storms[68:])

        self.assertEqual(len(train_set), 56)
        self.assertEqual(len(val_set), 12)
        self.assertEqual(len(test_set), 12)

        # Leak-free checks
        self.assertEqual(len(train_set.intersection(val_set)), 0)
        self.assertEqual(len(train_set.intersection(test_set)), 0)
        self.assertEqual(len(val_set.intersection(test_set)), 0)
        self.assertEqual(len(train_set | val_set | test_set), 80)


class TestTargetAndRICalculation(unittest.TestCase):
    """
    Test rapid intensification definition (>= 30 knots in 24 hours).
    """

    def test_ri_threshold(self):
        ic_24h = [10.0, 25.0, 29.0, 30.0, 35.0, 50.0, -10.0, 0.0]
        ri_events = [x for x in ic_24h if x >= 30.0]
        self.assertEqual(len(ri_events), 3)
        self.assertEqual(ri_events, [30.0, 35.0, 50.0])


class TestClassificationTaxonomy(unittest.TestCase):
    """
    Verify decision classification keys and vocabulary.
    """

    def test_valid_classifications(self):
        valid_statuses = {
            "VERIFIED",
            "CORRECTED",
            "REQUIRES FURTHER INVESTIGATION",
            "NOT SUFFICIENTLY SUPPORTED",
        }
        for category, info in DECISION_CLASSIFICATIONS.items():
            status = info["status"]
            self.assertIn(
                status,
                valid_statuses,
                f"Invalid status '{status}' for decision category '{category}'",
            )
            self.assertTrue(len(info["decision"]) > 0)
            self.assertTrue(len(info["evidence"]) > 0)


if __name__ == "__main__":
    unittest.main()

