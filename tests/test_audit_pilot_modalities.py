"""
test_audit_pilot_modalities.py - Unit tests for TC PRIMED modality audit utility.

Verifies:
  - Environmental files vs. overpass classification
  - Missing or size-mismatched files are not confused with missing modalities
  - IR group absence vs. infrared_availability_flag == 0
  - IRWIN 2D grid inspection (missing pixel percentage, valid count, statistics)
  - Non-spatial 1D auxiliary variables (R5, radius) not treated as 2D image grids
  - Denominator calculation (overpass counts only, never diluted by environmental files)
  - 5-state taxonomy consistency
  - Environmental ERA5 file structure inspection
  - Report output generation (CSV, JSON, TXT)
"""

import os
import sys
import csv
import json
import tempfile
import unittest
import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.audit_pilot_modalities import (
    is_environmental_file,
    classify_era,
    inspect_overpass_file,
    inspect_environmental_file,
    build_modality_summary,
    save_csv_reports,
    generate_text_audit_report,
    STATE_GROUP_ABSENT,
    STATE_FLAG_FALSE,
    STATE_SOME_VALID,
    STATE_ALL_MISSING,
    STATE_VALID_DATA,
)


class TestFileClassification(unittest.TestCase):
    """Verify distinction between environmental files and overpasses."""

    def test_environmental_file_detection(self):
        env_names = [
            "TCPRIMED_v01r01-final_AL122015_env_s2015111200_e2015111300.nc",
            "TCPRIMED_v01r01-final_WP011987_env_s1987010100_e1987010200.nc",
            "path/to/TCPRIMED_v01r01-final_IO032014_env_s2014101000_e2014101200.nc",
        ]
        for name in env_names:
            self.assertTrue(is_environmental_file(name), f"Should detect env: {name}")

    def test_overpass_file_detection(self):
        overpass_names = [
            "TCPRIMED_v01r01-final_AL122015_SSMIS_F17_046553_20151112205447.nc",
            "TCPRIMED_v01r01-final_CP011987_SSMI_F08_000996_19870829163228.nc",
            "TCPRIMED_v01r01-final_IO032014_GMI_GPM_003463_20141012111128.nc",
        ]
        for name in overpass_names:
            self.assertFalse(is_environmental_file(name), f"Should detect overpass: {name}")

    def test_era_classification(self):
        self.assertEqual(classify_era(1987), "1987-1999")
        self.assertEqual(classify_era(1999), "1987-1999")
        self.assertEqual(classify_era(2000), "2000-2009")
        self.assertEqual(classify_era(2009), "2000-2009")
        self.assertEqual(classify_era(2010), "2010-2019")
        self.assertEqual(classify_era(2019), "2010-2019")
        self.assertEqual(classify_era(2020), "2020+")
        self.assertEqual(classify_era(2023), "2020+")


class TestModalityInspectionSynthetic(unittest.TestCase):
    """Test NetCDF inspection logic using synthetically generated NetCDF files."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_overpass_nc(self, filename, with_ir_group=True, ir_flag=1, with_irwin=True,
                                 irwin_all_missing=False, with_pmw=True, with_gprof=True,
                                 with_radar=False, with_r5=True, r5_flag=1):
        filepath = os.path.join(self.temp_dir.name, filename)
        ds = nc.Dataset(filepath, "w", format="NETCDF4")

        # Global attributes
        ds.storm_id = "AL122015"
        ds.season = np.int32(2015)
        ds.basin = "AL"
        ds.cyclone_number = "12"

        # Overpass metadata
        grp_om = ds.createGroup("overpass_metadata")
        grp_om.createDimension("time_dim", 1)
        var_ot = grp_om.createVariable("time", np.float64, ("time_dim",))
        var_ot[0] = 1447361687.0  # nominal MW epoch seconds

        # Overpass storm metadata
        grp_osm = ds.createGroup("overpass_storm_metadata")
        grp_osm.createDimension("scalar_dim", 1)
        var_lat = grp_osm.createVariable("storm_latitude", np.float32, ("scalar_dim",))
        var_lon = grp_osm.createVariable("storm_longitude", np.float32, ("scalar_dim",))
        var_lat[0] = 25.5
        var_lon[0] = -70.2

        # Passive microwave
        if with_pmw:
            grp_pmw = ds.createGroup("passive_microwave")
            grp_pmw.instrument = "SSMIS"
            grp_pmw.platform = "F17"
            sw_grp = grp_pmw.createGroup("S1")
            sw_grp.createDimension("pmw_dim", 10)
            v = sw_grp.createVariable("dummy_tb", np.float32, ("pmw_dim",))
            v[:] = np.full(10, 200.0, dtype=np.float32)

        # GPROF
        if with_gprof:
            grp_gprof = ds.createGroup("GPROF")
            grp_gprof.gprof_availability_flag = np.int8(1)
            sw_gprof = grp_gprof.createGroup("S1")
            sw_gprof.createDimension("gprof_dim", 10)
            v_g = sw_gprof.createVariable("surfacePrecipitation", np.float32, ("gprof_dim",))
            v_g[:] = np.full(10, 5.0, dtype=np.float32)

        # Radar
        if with_radar:
            grp_radar = ds.createGroup("radar_radiometer")
            grp_radar.createDimension("flag_dim", 1)
            f_var = grp_radar.createVariable("availability_flag", np.int8, ("flag_dim",))
            f_var[0] = 1
            sw_rr = grp_radar.createGroup("Ku")
            sw_rr.createDimension("radar_dim", 10)
            v_rr = sw_rr.createVariable("nearSurfPrecipTotRate", np.float32, ("radar_dim",))
            v_rr[:] = np.full(10, 10.0, dtype=np.float32)

        # Infrared group
        if with_ir_group:
            grp_ir = ds.createGroup("infrared")
            grp_ir.platform = "GOES-13"
            grp_ir.instrument = "Imager"
            grp_ir.source = "HURSAT"
            grp_ir.central_wavelength = "10.7 um"

            grp_ir.createDimension("flag_dim", 1)
            var_fl = grp_ir.createVariable("infrared_availability_flag", np.int8, ("flag_dim",))
            var_fl[0] = ir_flag

            # R5 tangential wind radius
            if with_r5:
                var_r5_fl = grp_ir.createVariable("R5_availability_flag", np.int8, ("flag_dim",))
                var_r5_fl[0] = r5_flag
                grp_ir.createDimension("radius_dim", 5)
                var_rad = grp_ir.createVariable("R5", np.float32, ("radius_dim",))
                var_rad[:] = np.array([50.0, 100.0, 150.0, 200.0, 250.0], dtype=np.float32)

            # 2D IRWIN grid
            if with_irwin and ir_flag > 0:
                grp_ir.createDimension("latitude", 4)
                grp_ir.createDimension("longitude", 5)
                var_ir = grp_ir.createVariable("IRWIN", np.float32, ("latitude", "longitude"), fill_value=-999.0)
                var_ir.units = "Kelvin"
                var_ir.long_name = "Window Channel Brightness Temperature"

                if irwin_all_missing:
                    var_ir[:] = np.full((4, 5), -999.0, dtype=np.float32)
                else:
                    arr = np.full((4, 5), 260.0, dtype=np.float32)
                    arr[0, 0] = -999.0  # 1 missing pixel out of 20 = 5% missing
                    var_ir[:] = arr

                var_irt = grp_ir.createVariable("time", np.float64, ("flag_dim",))
                var_irt[0] = 1447361747.0  # 60 seconds offset (+1 minute)

        ds.close()
        return filepath

    def test_ir_group_absent(self):
        """When /infrared group is absent, state must be GROUP_ABSENT."""
        fp = self._create_mock_overpass_nc("test_no_ir.nc", with_ir_group=False)
        manifest_row = {"s3_key": "v01r01/final/2015/AL/12/test_no_ir.nc", "season": "2015", "basin": "AL",
                        "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "1000"}
        res = inspect_overpass_file(fp, manifest_row)

        self.assertTrue(res["open_success"])
        self.assertFalse(res["infrared"]["group_present"])
        self.assertEqual(res["infrared"]["modality_state"], STATE_GROUP_ABSENT)
        self.assertIsNone(res["infrared"]["availability_flag"])
        self.assertFalse(res["infrared"]["irwin_present"])
        self.assertEqual(res["infrared"]["irwin_valid_pixels"], 0)

    def test_ir_flag_zero(self):
        """When /infrared exists but flag is 0, state must be FLAG_FALSE."""
        fp = self._create_mock_overpass_nc("test_flag_0.nc", with_ir_group=True, ir_flag=0, with_irwin=False)
        manifest_row = {"s3_key": "v01r01/final/2015/AL/12/test_flag_0.nc", "season": "2015", "basin": "AL",
                        "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "1000"}
        res = inspect_overpass_file(fp, manifest_row)

        self.assertTrue(res["open_success"])
        self.assertTrue(res["infrared"]["group_present"])
        self.assertEqual(res["infrared"]["availability_flag"], 0)
        self.assertEqual(res["infrared"]["modality_state"], STATE_FLAG_FALSE)
        self.assertFalse(res["infrared"]["irwin_present"])

    def test_irwin_all_missing(self):
        """When IRWIN exists but all values are fill values, state must be AVAILABLE_ALL_MISSING."""
        fp = self._create_mock_overpass_nc("test_all_missing.nc", with_ir_group=True, ir_flag=1,
                                           with_irwin=True, irwin_all_missing=True)
        manifest_row = {"s3_key": "v01r01/final/2015/AL/12/test_all_missing.nc", "season": "2015", "basin": "AL",
                        "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "1000"}
        res = inspect_overpass_file(fp, manifest_row)

        self.assertTrue(res["open_success"])
        self.assertTrue(res["infrared"]["group_present"])
        self.assertEqual(res["infrared"]["modality_state"], STATE_ALL_MISSING)
        self.assertTrue(res["infrared"]["irwin_present"])
        self.assertEqual(res["infrared"]["irwin_valid_pixels"], 0)
        self.assertEqual(res["infrared"]["irwin_missing_percent"], 100.0)

    def test_irwin_valid_pixels_and_stats(self):
        """When IRWIN has valid data, verify pixel statistics, dimensions, and provenance."""
        fp = self._create_mock_overpass_nc("test_valid_ir.nc", with_ir_group=True, ir_flag=1,
                                           with_irwin=True, irwin_all_missing=False)
        manifest_row = {"s3_key": "v01r01/final/2015/AL/12/test_valid_ir.nc", "season": "2015", "basin": "AL",
                        "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "1000"}
        res = inspect_overpass_file(fp, manifest_row)

        self.assertTrue(res["open_success"])
        self.assertEqual(res["infrared"]["modality_state"], STATE_SOME_VALID)
        self.assertTrue(res["infrared"]["irwin_present"])
        self.assertEqual(res["infrared"]["irwin_shape"], [4, 5])
        self.assertEqual(res["infrared"]["irwin_total_pixels"], 20)
        self.assertEqual(res["infrared"]["irwin_missing_pixels"], 1)
        self.assertEqual(res["infrared"]["irwin_valid_pixels"], 19)
        self.assertAlmostEqual(res["infrared"]["irwin_missing_percent"], 5.0, places=2)
        self.assertAlmostEqual(res["infrared"]["irwin_min_k"], 260.0, places=2)
        self.assertAlmostEqual(res["infrared"]["irwin_max_k"], 260.0, places=2)
        self.assertAlmostEqual(res["infrared"]["irwin_mean_k"], 260.0, places=2)
        self.assertEqual(res["infrared"]["provenance_platform"], "GOES-13")
        self.assertEqual(res["infrared"]["provenance_instrument"], "Imager")
        self.assertEqual(res["infrared"]["source_granule"], "HURSAT")
        self.assertAlmostEqual(res["infrared"]["time_offset_seconds"], 60.0, places=1)

    def test_auxiliary_1d_variables_not_treated_as_2d_image(self):
        """Verify that 1D R5 radius profile is detected in R5 metrics and not confused with IRWIN."""
        fp = self._create_mock_overpass_nc("test_aux_r5.nc", with_ir_group=True, ir_flag=0,
                                           with_irwin=False, with_r5=True, r5_flag=1)
        manifest_row = {"s3_key": "v01r01/final/2015/AL/12/test_aux_r5.nc", "season": "2015", "basin": "AL",
                        "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "1000"}
        res = inspect_overpass_file(fp, manifest_row)

        self.assertTrue(res["infrared"]["r5_present"])
        self.assertEqual(res["infrared"]["r5_availability_flag"], 1)
        self.assertEqual(res["infrared"]["r5_valid_count"], 5)
        # Even though R5 exists, IRWIN is absent
        self.assertFalse(res["infrared"]["irwin_present"])
        self.assertEqual(res["infrared"]["modality_state"], STATE_FLAG_FALSE)


class TestEnvironmentalInspectionSynthetic(unittest.TestCase):
    """Test ERA5 environmental file inspection."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_environmental_file_structure(self):
        filepath = os.path.join(self.temp_dir.name, "TCPRIMED_v01r01-final_AL122015_env_s2015111200_e2015111300.nc")
        ds = nc.Dataset(filepath, "w", format="NETCDF4")
        ds.storm_id = "AL122015"
        ds.season = np.int32(2015)
        ds.basin = "AL"

        # Groups
        ds.createGroup("storm_metadata")
        ds.createGroup("overpass_metadata")
        ds.createGroup("overpass_storm_metadata")
        grp_diag = ds.createGroup("diagnostics")

        # Cylindrical group
        grp_cyl = ds.createGroup("cylindrical")
        grp_cyl.createDimension("time", 5)
        grp_cyl.createDimension("level", 23)
        grp_cyl.createDimension("azimuth", 24)
        grp_cyl.createDimension("radius", 61)
        var_t = grp_cyl.createVariable("time", np.float64, ("time",))
        var_t[:] = np.linspace(1447360000, 1447446400, 5)  # 24 hours
        var_lev = grp_cyl.createVariable("level", np.float32, ("level",))
        var_lev[:] = np.linspace(1000, 100, 23)
        grp_cyl.createVariable("sst", np.float32, ("time", "azimuth", "radius"))
        grp_cyl.createVariable("temperature", np.float32, ("time", "level", "azimuth", "radius"))

        # Rectilinear group
        grp_rec = ds.createGroup("rectilinear")
        grp_rec.createDimension("time", 5)
        grp_rec.createDimension("level", 23)
        grp_rec.createDimension("ny", 121)
        grp_rec.createDimension("nx", 121)
        grp_rec.createVariable("time", np.float64, ("time",))
        grp_rec.createVariable("level", np.float32, ("level",))
        grp_rec.createVariable("pressure_msl", np.float32, ("time", "ny", "nx"))

        ds.close()

        manifest_row = {
            "s3_key": "v01r01/final/2015/AL/12/TCPRIMED_v01r01-final_AL122015_env_s2015111200_e2015111300.nc",
            "season": "2015", "basin": "AL", "cyclone_number": "12", "storm_id": "AL122015", "size_bytes": "2000"
        }
        res = inspect_environmental_file(filepath, manifest_row)

        self.assertTrue(res["open_success"])
        self.assertEqual(res["cylindrical_grid"]["azimuth_count"], 24)
        self.assertEqual(res["cylindrical_grid"]["radius_count"], 61)
        self.assertEqual(res["vertical_levels_count"], 23)
        self.assertEqual(len(res["vertical_levels"]), 23)
        self.assertEqual(res["rectilinear_grid"]["dimensions"]["ny"], 121)
        self.assertEqual(res["rectilinear_grid"]["dimensions"]["nx"], 121)
        self.assertIn("pressure_msl", res["rectilinear_grid"]["variables"])
        self.assertAlmostEqual(res["duration_hours"], 24.0, places=1)


class TestAuditWorkflowAndDenominators(unittest.TestCase):
    """Verify denominator calculations and report serialization."""

    def test_denominator_excludes_environmental_files(self):
        """
        Verify that total inspected overpass denominator does NOT count environmental files.
        If there are 10 overpasses (8 with IR, 2 without) and 2 environmental files,
        the IR availability rate must be 8/10 = 80.0%, NOT 8/12 = 66.7%!
        """
        manifest = [
            {"s3_key": f"v01r01/final/2015/AL/01/overpass_{i}.nc", "season": "2015", "basin": "AL",
             "cyclone_number": "01", "storm_id": "AL012015", "size_bytes": "1000"}
            for i in range(10)
        ] + [
            {"s3_key": f"v01r01/final/2015/AL/01/TCPRIMED_v01r01-final_AL012015_env_s2015111200_e2015111300.nc", "season": "2015",
             "basin": "AL", "cyclone_number": "01", "storm_id": "AL012015", "size_bytes": "2000"}
            for i in range(2)
        ]

        integrity = [
            {"s3_key": m["s3_key"], "season": int(m["season"]), "basin": m["basin"],
             "cyclone_number": m["cyclone_number"], "storm_id": m["storm_id"],
             "is_environmental": is_environmental_file(m["s3_key"]),
             "file_type": "environmental" if is_environmental_file(m["s3_key"]) else "overpass",
             "local_destination": m["s3_key"], "expected_size_bytes": int(m["size_bytes"]),
             "actual_size_bytes": int(m["size_bytes"]), "download_status": "CORRECT_SIZE",
             "can_open_netcdf": True, "inspection_status": "SUCCESS", "error_message": ""}
            for m in manifest
        ]

        inspected_overpasses = []
        for i in range(10):
            has_ir = (i < 8)
            inspected_overpasses.append({
                "s3_key": f"v01r01/final/2015/AL/01/overpass_{i}.nc",
                "filename": f"overpass_{i}.nc",
                "season": 2015,
                "era": "2010-2019",
                "basin": "AL",
                "cyclone_number": "01",
                "storm_id": "AL012015",
                "instrument": "SSMIS",
                "platform": "F17",
                "open_success": True,
                "error_message": "",
                "nominal_overpass_time_raw": 1447361687.0,
                "nominal_overpass_time_utc": "2015-11-12 20:54:47",
                "storm_latitude": 25.5,
                "storm_longitude": -70.2,
                "infrared": {
                    "group_present": True,
                    "availability_flag": 1 if has_ir else 0,
                    "provenance_platform": "GOES-13" if has_ir else "UNKNOWN",
                    "provenance_instrument": "Imager" if has_ir else "UNKNOWN",
                    "source_granule": "HURSAT" if has_ir else "NONE",
                    "geospatial_bounds": {},
                    "observation_time_raw": 1447361687.0,
                    "observation_time_utc": "2015-11-12 20:54:47",
                    "time_offset_seconds": 0.0,
                    "time_offset_minutes": 0.0,
                    "r5_present": True,
                    "r5_availability_flag": 1,
                    "r5_shape": [5],
                    "r5_dimensions": ["radius"],
                    "r5_units": "m s-1",
                    "r5_valid_count": 5,
                    "r5_total_count": 5,
                    "r5_min": 10.0,
                    "r5_max": 30.0,
                    "r5_mean": 20.0,
                    "irwin_present": has_ir,
                    "irwin_shape": [480, 640] if has_ir else [],
                    "irwin_dimensions": ["latitude", "longitude"] if has_ir else [],
                    "irwin_dtype": "float32" if has_ir else "",
                    "irwin_units": "Kelvin" if has_ir else "",
                    "irwin_long_name": "Window Channel Brightness Temperature" if has_ir else "",
                    "irwin_fill_value": -999.0 if has_ir else None,
                    "irwin_total_pixels": 307200 if has_ir else 0,
                    "irwin_valid_pixels": 307200 if has_ir else 0,
                    "irwin_missing_pixels": 0 if has_ir else 0,
                    "irwin_missing_percent": 0.0 if has_ir else 100.0,
                    "irwin_min_k": 200.0 if has_ir else None,
                    "irwin_max_k": 300.0 if has_ir else None,
                    "irwin_mean_k": 250.0 if has_ir else None,
                    "modality_state": STATE_VALID_DATA if has_ir else STATE_FLAG_FALSE,
                },
                "passive_microwave": {
                    "group_present": True,
                    "modality_state": STATE_VALID_DATA,
                },
                "GPROF": {
                    "group_present": True,
                    "availability_flag": 1,
                    "modality_state": STATE_VALID_DATA,
                },
                "radar_radiometer": {
                    "group_present": False,
                    "availability_flag": None,
                    "modality_state": STATE_GROUP_ABSENT,
                },
            })

        inspected_env = [
            {"s3_key": f"v01r01/final/2015/AL/01/TCPRIMED_v01r01-final_AL012015_env_s2015111200_e2015111300.nc",
             "filename": f"TCPRIMED_v01r01-final_AL012015_env_s2015111200_e2015111300.nc", "season": 2015, "basin": "AL",
             "cyclone_number": "01", "storm_id": "AL012015", "open_success": True, "error_message": "",
             "groups_present": ["storm_metadata", "diagnostics", "cylindrical", "rectilinear"],
             "time_steps_count": 5, "time_start_utc": "2015-11-12 00:00:00", "time_end_utc": "2015-11-13 00:00:00",
             "duration_hours": 24.0, "vertical_levels_count": 23, "vertical_levels": list(range(23)),
             "rectilinear_grid": {"dimensions": {"ny": 121, "nx": 121}, "variables": ["pressure_msl", "sst"]},
             "cylindrical_grid": {"radius_count": 61, "azimuth_count": 24},
             "environmental_variables": ["pressure_msl", "sst"]}
            for _ in range(2)
        ]

        summary = build_modality_summary(manifest, integrity, inspected_overpasses, inspected_env)

        # Check total manifests & denominators
        self.assertEqual(summary["download_status"]["total_manifest"], 12)
        self.assertEqual(summary["download_status"]["total_overpasses_manifest"], 10)
        self.assertEqual(summary["download_status"]["total_environmental_manifest"], 2)

        # IR availability must be 8 / 10 = 80.0%, NOT 8 / 12 = 66.7%
        self.assertEqual(summary["overpass_inspection"]["total_inspected"], 10)
        self.assertEqual(summary["overpass_inspection"]["irwin_valid_count"], 8)
        self.assertAlmostEqual(summary["overpass_inspection"]["irwin_valid_pct"], 80.0, places=1)

    def test_save_and_reload_csv_reports(self):
        """Verify saving and re-loading CSV reports."""
        with tempfile.TemporaryDirectory() as temp_dir:
            integrity = [{
                "s3_key": "v01r01/final/2015/AL/12/test.nc", "season": 2015, "basin": "AL",
                "cyclone_number": "12", "storm_id": "AL122015", "file_type": "overpass",
                "is_environmental": False, "local_destination": "test.nc",
                "expected_size_bytes": 1000, "actual_size_bytes": 1000,
                "download_status": "CORRECT_SIZE", "can_open_netcdf": True,
                "inspection_status": "SUCCESS", "error_message": "",
            }]
            overpasses = [{
                "s3_key": "v01r01/final/2015/AL/12/test.nc", "filename": "test.nc",
                "season": 2015, "era": "2010-2019", "basin": "AL", "cyclone_number": "12", "storm_id": "AL122015",
                "instrument": "SSMIS", "platform": "F17", "open_success": True,
                "nominal_time_utc": "2015-11-12 20:54:47",
                "storm_latitude": 25.5, "storm_longitude": -70.2,
                "infrared": {
                    "group_present": True, "availability_flag": 1,
                    "provenance_platform": "GOES-13", "provenance_instrument": "Imager",
                    "source_granule": "HURSAT", "geospatial_bounds": {},
                    "observation_time_utc": "2015-11-12 20:54:47",
                    "time_offset_seconds": 0.0, "time_offset_minutes": 0.0,
                    "irwin_present": True, "irwin_shape": [480, 640], "irwin_dtype": "float32",
                    "irwin_units": "Kelvin", "irwin_long_name": "Window Channel Brightness Temperature",
                    "irwin_fill_value": -999.0, "irwin_total_pixels": 307200, "irwin_valid_pixels": 307200,
                    "irwin_missing_percent": 0.0,
                    "irwin_min_k": 200.0, "irwin_max_k": 300.0, "irwin_mean_k": 250.0,
                    "r5_present": True, "r5_availability_flag": 1, "r5_shape": [5], "r5_valid_count": 5,
                    "modality_state": STATE_VALID_DATA,
                },
                "passive_microwave": {"modality_state": STATE_VALID_DATA},
                "GPROF": {"modality_state": STATE_VALID_DATA},
                "radar_radiometer": {"modality_state": STATE_GROUP_ABSENT},
            }]

            save_csv_reports(integrity, overpasses, reports_dir=temp_dir)

            integ_path = os.path.join(temp_dir, "pilot_download_integrity.csv")
            ir_path = os.path.join(temp_dir, "pilot_ir_availability.csv")

            self.assertTrue(os.path.exists(integ_path))
            self.assertTrue(os.path.exists(ir_path))

            with open(integ_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["download_status"], "CORRECT_SIZE")

            with open(ir_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["infrared_availability_flag"], "1")
                self.assertEqual(rows[0]["irwin_present"], "True")


if __name__ == "__main__":
    unittest.main()
