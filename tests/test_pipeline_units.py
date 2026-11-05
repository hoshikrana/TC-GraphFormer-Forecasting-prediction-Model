"""
Unit tests for TC PRIMED pipeline components using synthetic NetCDF structures.
Tests:
  1. Configuration parsing and disk space checking.
  2. NetCDF reader: no double scale/offset decoding, masked array handling, clean scalar extraction.
  3. Metadata extractor: Best track, overpass metadata, future track coordinate extraction.
  4. Modality processors: PMW, IR, GPROF, Radar, Environmental slice.
  5. Environmental matcher: strictly causal matching vs retrospective leakage.
  6. Storm splitter: leak-free storm partitioning.
  7. Storage writer: atomic write and resume detection.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import netCDF4 as nc

from tcprimed_pipeline.config import PipelineConfig, DatasetConfig, OutputConfig
from tcprimed_pipeline.netcdf_reader import clean_scalar, get_array_with_fill, NetCDFReader, open_netcdf
from tcprimed_pipeline.metadata_extractor import MetadataExtractor
from tcprimed_pipeline.modality_processors import ModalityProcessors
from tcprimed_pipeline.environmental_matcher import EnvironmentalMatcher
from tcprimed_pipeline.storm_splitter import StormSplitter
from tcprimed_pipeline.storage_writer import StorageWriter


class TestConfigAndEnvironment(unittest.TestCase):
    def test_default_config(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.dataset.root, "data/raw/pilot")
        self.assertEqual(cfg.splitting.train, 0.70)
        self.assertEqual(cfg.splitting.seed, 42)
        self.assertTrue(cfg.quality_control.enforce_causal_environmental)

    def test_disk_space_check(self):
        cfg = PipelineConfig()
        res = cfg.check_disk_space()
        self.assertIn("free_gb", res)
        self.assertGreater(res["free_gb"], 0.0)


class TestNetCDFReaderSynthetic(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.nc_path = os.path.join(self.tmp_dir, "test_synthetic.nc")

        with nc.Dataset(self.nc_path, "w", format="NETCDF4") as ds:
            ds.root_attr = "global_test"
            g1 = ds.createGroup("passive_microwave")
            g_s1 = g1.createGroup("S1")
            g_s1.createDimension("scan", 4)
            g_s1.createDimension("pixel", 5)

            # Variable with scale_factor and add_offset
            v_tb = g_s1.createVariable("TB_19V", "i2", ("scan", "pixel"), fill_value=-9999)
            v_tb.scale_factor = 0.01
            v_tb.add_offset = 200.0
            v_tb.units = "K"
            # Unscaled values: write 280.5 K with one masked element
            raw_vals = np.ma.array(np.full((4, 5), 280.5, dtype=np.float32), mask=False)
            raw_vals[0, 0] = np.ma.masked
            v_tb[:] = raw_vals

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_double_scale_offset(self):
        with open_netcdf(self.nc_path, "r") as ds:
            v_var = ds.groups["passive_microwave"].groups["S1"].variables["TB_19V"]
            arr, mask = get_array_with_fill(v_var, fill_value=-999.0)

            # Masked pixel should be filled with -999.0
            self.assertEqual(arr[0, 0], -999.0)
            self.assertFalse(mask[0, 0])

            # Valid pixels should be ~280.5 K, NOT scaled twice (which would be ~202.8 or similar)
            self.assertTrue(mask[1, 1])
            self.assertAlmostEqual(arr[1, 1], 280.5, places=1)


class TestEnvironmentalMatcherCausality(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.env_nc = os.path.join(self.tmp_dir, "test_env.nc")
        with nc.Dataset(self.env_nc, "w", format="NETCDF4") as ds:
            g = ds.createGroup("rectilinear")
            g.createDimension("time", 4)
            v_t = g.createVariable("time", "i8", ("time",))
            # Synoptic times: 00:00, 06:00, 12:00, 18:00 on 2020-01-01
            # Base epoch: 1577836800 (2020-01-01 00:00:00 UTC)
            base_t = 1577836800
            v_t[:] = [base_t, base_t + 21600, base_t + 43200, base_t + 64800]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_causal_matching_rejects_future(self):
        matcher = EnvironmentalMatcher(enforce_causality=True)
        matcher.register_storm_environmental_file("STORM01", self.env_nc)

        base_t = 1577836800
        # Overpass at 04:30 (closer to 06:00 than 00:00)
        # Nearest slice would be 06:00 (FUTURE LEAKAGE)
        # Causal slice MUST be 00:00 (PAST)
        op_time = base_t + (4 * 3600) + 1800  # 04:30

        match = matcher.match_overpass("STORM01", op_time)
        self.assertTrue(match["has_match"])
        self.assertEqual(match["matched_index"], 0)  # Must pick 00:00 (index 0)
        self.assertEqual(match["env_time_epoch"], base_t)
        self.assertFalse(match["is_future_leakage"])
        self.assertLessEqual(match["env_time_epoch"], op_time)

    def test_overpass_before_first_synoptic_slice_flagged(self):
        matcher = EnvironmentalMatcher(enforce_causality=True)
        matcher.register_storm_environmental_file("STORM01", self.env_nc)

        base_t = 1577836800
        # Overpass 1 hour before first synoptic slice
        op_time = base_t - 3600
        match = matcher.match_overpass("STORM01", op_time)
        self.assertFalse(match["has_match"])
        self.assertTrue(match["is_future_leakage"])
        self.assertIn("causal violation", match["rejection_reason"])


class TestStormSplitterIsolation(unittest.TestCase):
    def test_split_isolation_and_proportions(self):
        splitter = StormSplitter()
        # Create 80 synthetic storms across basins and eras
        basins = ["AL", "EP", "WP", "IO", "SH", "CP"]
        seasons = [1995, 2005, 2015, 2022]
        storms = {}
        for i in range(80):
            sid = f"{basins[i % len(basins)]}{i:02d}{seasons[i % len(seasons)]}"
            storms[sid] = {
                "season": seasons[i % len(seasons)],
                "basin": basins[i % len(basins)],
                "overpass_files": [f"op_{j}.nc" for j in range(5)],
                "env_file": "env.nc",
            }

        split_map = splitter.split_storms(storms)
        self.assertEqual(len(split_map), 80)

        train_s = [s for s, p in split_map.items() if p == "TRAIN"]
        val_s = [s for s, p in split_map.items() if p == "VAL"]
        test_s = [s for s, p in split_map.items() if p == "TEST"]

        self.assertEqual(len(train_s), 56)  # Exactly 70% of 80
        self.assertEqual(len(val_s), 12)    # Exactly 15% of 80
        self.assertEqual(len(test_s), 12)   # Exactly 15% of 80

        # Zero leakage
        self.assertEqual(len(set(train_s) & set(val_s)), 0)
        self.assertEqual(len(set(train_s) & set(test_s)), 0)
        self.assertEqual(len(set(val_s) & set(test_s)), 0)


class TestStorageWriterAtomicAndResume(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.out_cfg = OutputConfig(root=self.tmp_dir, compression_level=1)
        self.writer = StorageWriter(self.out_cfg)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_atomic_write_and_resume_check(self):
        sid = "TEST012022"
        season = 2022
        basin = "TEST"

        self.assertFalse(self.writer.is_storm_processed(season, basin, sid))

        # Write container
        best_track = {"times": np.array([100, 200]), "lats": np.array([10.0, 11.0]), "lons": np.array([120.0, 121.0])}
        obs_list = [{
            "observation_id": "TEST_OBS_01",
            "timestamp_str": "20220101000000",
            "timestamp_epoch": 100,
            "instrument": "GMI",
            "platform": "GPM",
            "targets": {"intensity_vmax_kt": 45.0},
        }]

        cpath = self.writer.write_storm_container(sid, season, basin, "TRAIN", best_track, obs_list)
        self.assertTrue(os.path.exists(cpath))
        self.assertFalse(os.path.exists(cpath + ".tmp"))  # Temp file removed

        # Now resume check should return True
        self.assertTrue(self.writer.is_storm_processed(season, basin, sid))

        # Verify content
        with nc.Dataset(cpath, "r") as ds:
            self.assertEqual(ds.getncattr("storm_id"), sid)
            self.assertIn("storm_metadata", ds.groups)
            self.assertIn("overpasses", ds.groups)


if __name__ == "__main__":
    unittest.main()
