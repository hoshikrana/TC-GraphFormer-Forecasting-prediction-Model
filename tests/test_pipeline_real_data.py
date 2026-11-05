"""
Real observation tests for TC PRIMED pipeline components.
Tests real files from the pilot dataset:
  1. GMI observation (GPM GMI)
  2. TMI observation (TRMM TMI)
  3. AMSR2 observation (GCOM-W1 AMSR2)
  4. Observation without valid IR
  5. Radar-containing observation (GPM DPR Ku/Ka)
  6. Environmental file (ERA5)
"""

import os
import unittest
import netCDF4 as nc

from tcprimed_pipeline.netcdf_reader import open_netcdf
from tcprimed_pipeline.metadata_extractor import MetadataExtractor
from tcprimed_pipeline.modality_processors import ModalityProcessors


class TestRealPilotObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gmi_path = "data/raw/pilot/2015/AL/12/TCPRIMED_v01r01-final_AL122015_GMI_GPM_009633_20151108201259.nc"
        cls.tmi_path = "data/raw/pilot/1998/AL/03/TCPRIMED_v01r01-final_AL031998_TMI_TRMM_004206_19980821142248.nc"
        cls.amsr2_path = "data/raw/pilot/2013/AL/08/TCPRIMED_v01r01-final_AL082013_AMSR2_GCOMW1_006958_20130907084935.nc"
        cls.no_ir_path = "data/raw/pilot/1988/IO/01/TCPRIMED_v01r01-final_IO011988_SSMI_F08_005047_19880611140806.nc"
        cls.radar_path = "data/raw/pilot/2022/WP/08/TCPRIMED_v01r01-final_WP082022_GMI_GPM_047909_20220804082925.nc"
        cls.env_path = "data/raw/pilot/1987/CP/01/TCPRIMED_v01r01-final_CP011987_env_s19870826000000_e19870829180000.nc"

    def test_real_gmi_observation(self):
        self.assertTrue(os.path.exists(self.gmi_path), f"GMI file missing: {self.gmi_path}")
        with open_netcdf(self.gmi_path, "r") as ds:
            meta = MetadataExtractor.extract_overpass_metadata(ds)
            self.assertEqual(meta["instrument"], "GMI")
            self.assertEqual(meta["platform"], "GPM")
            self.assertEqual(meta["season"], 2015)
            self.assertEqual(meta["basin"], "AL")

            pmw = ModalityProcessors.process_passive_microwave(ds)
            self.assertTrue(pmw["is_available"])
            self.assertIn("S1", pmw["swaths"])
            self.assertIn("S2", pmw["swaths"])

            # Check TB channels
            chans = pmw["channel_names"]
            self.assertTrue(any("TB_10.65" in c for c in chans))
            self.assertTrue(any("TB_89.0" in c or "TB_166.0" in c for c in chans))

    def test_real_tmi_observation(self):
        self.assertTrue(os.path.exists(self.tmi_path), f"TMI file missing: {self.tmi_path}")
        with open_netcdf(self.tmi_path, "r") as ds:
            meta = MetadataExtractor.extract_overpass_metadata(ds)
            self.assertEqual(meta["instrument"], "TMI")
            self.assertEqual(meta["platform"], "TRMM")

            pmw = ModalityProcessors.process_passive_microwave(ds)
            self.assertTrue(pmw["is_available"])
            self.assertIn("S1", pmw["swaths"])
            chans = pmw["channel_names"]
            self.assertTrue(any("TB_10.65" in c for c in chans))

    def test_real_amsr2_observation(self):
        self.assertTrue(os.path.exists(self.amsr2_path), f"AMSR2 file missing: {self.amsr2_path}")
        with open_netcdf(self.amsr2_path, "r") as ds:
            meta = MetadataExtractor.extract_overpass_metadata(ds)
            self.assertEqual(meta["instrument"], "AMSR2")
            self.assertEqual(meta["platform"], "GCOMW1")

            pmw = ModalityProcessors.process_passive_microwave(ds)
            self.assertTrue(pmw["is_available"])

    def test_real_no_ir_observation(self):
        self.assertTrue(os.path.exists(self.no_ir_path), f"No-IR file missing: {self.no_ir_path}")
        with open_netcdf(self.no_ir_path, "r") as ds:
            ir = ModalityProcessors.process_infrared(ds)
            # IRWIN image must not be available
            self.assertFalse(ir["is_available"])
            self.assertEqual(ir["irwin_image"], None)

    def test_real_radar_observation(self):
        self.assertTrue(os.path.exists(self.radar_path), f"Radar file missing: {self.radar_path}")
        with open_netcdf(self.radar_path, "r") as ds:
            radar = ModalityProcessors.process_radar(ds)
            self.assertTrue(radar["is_available"])
            self.assertIn(radar["radar_group"], ["KuGMI", "KuKaGMI"])
            self.assertIsNotNone(radar["ku_reflectivity_profile"])
            self.assertEqual(len(radar["ku_reflectivity_profile"].shape), 3)

    def test_real_environmental_file(self):
        self.assertTrue(os.path.exists(self.env_path), f"Env file missing: {self.env_path}")
        with open_netcdf(self.env_path, "r") as ds:
            bt = MetadataExtractor.extract_environmental_best_track(ds)
            self.assertGreater(len(bt["times"]), 0)
            self.assertGreater(len(bt["lats"]), 0)
            self.assertGreater(len(bt["intensities"]), 0)

            # Test slice processing
            slice_0 = ModalityProcessors.process_environmental_slice(ds, 0)
            self.assertTrue(slice_0["is_available"])
            self.assertIsNotNone(slice_0["precipitable_water"])


if __name__ == "__main__":
    unittest.main()

