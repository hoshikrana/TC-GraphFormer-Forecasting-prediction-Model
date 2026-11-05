"""
Modality Processors module for TC PRIMED.
Safely extracts and standardizes scientific measurements across Passive Microwave,
Infrared, GPROF, Radar, and Environmental modalities, preserving native grids and units.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import netCDF4 as nc

from .netcdf_reader import clean_scalar, get_array_with_fill, NetCDFReader


class ModalityProcessors:
    """Processes each individual TC PRIMED sensor and retrieval modality."""

    # -----------------------------------------------------------------------
    # 1. Passive Microwave Processor
    # -----------------------------------------------------------------------
    @staticmethod
    def process_passive_microwave(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Process passive microwave swaths (S1 and optional S2).
        Preserves all TB channels, native scan/pixel dimensions, and coordinates.
        """
        out: Dict[str, Any] = {
            "is_available": False,
            "swaths": {},
            "channel_names": [],
            "sensor_type": None,
        }

        pmw = NetCDFReader.get_group(ds, "passive_microwave")
        if pmw is None:
            return out

        swaths_found = {}
        all_channels = []

        for sw_name in ["S1", "S2"]:
            if sw_name not in pmw.groups:
                continue
            sw_group = pmw.groups[sw_name]

            # Discover all TB variables
            tb_vars = [v for v in sw_group.variables if v.startswith("TB_")]
            if not tb_vars:
                continue

            # Read coordinates
            lat_arr, lat_mask = NetCDFReader.read_variable_2d(sw_group, "latitude")
            lon_arr, lon_mask = NetCDFReader.read_variable_2d(sw_group, "longitude")
            x_arr, _ = NetCDFReader.read_variable_2d(sw_group, "x")
            y_arr, _ = NetCDFReader.read_variable_2d(sw_group, "y")

            # Extract TB channels
            channels_dict = {}
            for tb_name in tb_vars:
                tb_data, tb_mask = NetCDFReader.read_variable_2d(sw_group, tb_name)
                var_obj = sw_group.variables[tb_name]
                channels_dict[tb_name] = {
                    "data": tb_data,
                    "valid_mask": tb_mask,
                    "units": getattr(var_obj, "units", "K"),
                    "long_name": getattr(var_obj, "long_name", tb_name),
                    "shape": tb_data.shape if tb_data is not None else (),
                }
                all_channels.append(f"{sw_name}/{tb_name}")

            swaths_found[sw_name] = {
                "latitude": lat_arr,
                "longitude": lon_arr,
                "x": x_arr,
                "y": y_arr,
                "scan_shape": lat_arr.shape if lat_arr is not None else (),
                "channels": channels_dict,
            }

        if swaths_found:
            out["is_available"] = True
            out["swaths"] = swaths_found
            out["channel_names"] = all_channels

        return out

    # -----------------------------------------------------------------------
    # 2. Infrared Processor
    # -----------------------------------------------------------------------
    @staticmethod
    def process_infrared(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Process infrared observations and storm size metrics.
        Preserves:
          - 2D IRWIN brightness temperature image, wavelength, platform, and resolution.
          - 1D R5 storm size time series (degrees latitude along R5_time).
          - 1D 30-bin radial profile of brightness temperature with radius coordinate (km).
        """
        out: Dict[str, Any] = {
            "is_available": False,
            "availability_flag": 0,
            "provenance": "NONE",
            "nominal_resolution_km": None,
            "wavelength_descriptor": "",
            "platform": "",
            "instrument": "",
            "irwin_image": None,
            "irwin_valid_mask": None,
            "irwin_shape": (),
            "latitude": None,
            "longitude": None,
            "r5_available": False,
            "r5_values": None,
            "r5_times": None,
            "r5_units": "degrees_latitude",
            "radial_profile_tb": None,
            "radial_profile_radius_km": None,
        }

        irg = NetCDFReader.get_group(ds, "infrared")
        if irg is None:
            return out

        flag = clean_scalar(NetCDFReader.read_variable_scalar(irg, "infrared_availability_flag")) or 0
        r5_flag = clean_scalar(NetCDFReader.read_variable_scalar(irg, "R5_availability_flag")) or 0
        out["availability_flag"] = int(flag)

        # 1. 2D IRWIN Image
        if flag in (1, 2) and "IRWIN" in irg.variables:
            irw_var = irg.variables["IRWIN"]
            img, mask = NetCDFReader.read_variable_2d(irg, "IRWIN")
            if img is not None and np.any(mask):
                out["is_available"] = True
                out["irwin_image"] = img
                out["irwin_valid_mask"] = mask
                out["irwin_shape"] = img.shape
                out["wavelength_descriptor"] = str(getattr(irw_var, "long_name", "IR Window")).strip()
                out["platform"] = str(getattr(irg, "platform", "UNKNOWN")).strip()
                out["instrument"] = str(getattr(irg, "instrument", "UNKNOWN")).strip()

                if flag == 1:
                    out["provenance"] = "CIRA_STAR_McIDAS"
                    out["nominal_resolution_km"] = 4.0
                else:
                    out["provenance"] = "HURSAT_B1_ISCCP"
                    out["nominal_resolution_km"] = 8.0

                out["latitude"], _ = NetCDFReader.read_variable_2d(irg, "latitude")
                out["longitude"], _ = NetCDFReader.read_variable_2d(irg, "longitude")

        # 2. R5 Vortex Size Time-Series
        if r5_flag == 1 and "R5" in irg.variables:
            r5_arr, _ = get_array_with_fill(irg.variables["R5"])
            if r5_arr is not None and len(r5_arr) > 0:
                out["r5_available"] = True
                out["r5_values"] = r5_arr
                if "R5_time" in irg.variables:
                    out["r5_times"] = np.array(irg.variables["R5_time"][:], dtype=np.int64)

        # 3. Radial profile (30 bins, coordinate radius in km)
        if "brightness_temperature_mean_radial_profile" in irg.variables:
            rad_prof, _ = get_array_with_fill(irg.variables["brightness_temperature_mean_radial_profile"])
            out["radial_profile_tb"] = rad_prof
            if "radius" in irg.variables:
                out["radial_profile_radius_km"] = np.array(irg.variables["radius"][:], dtype=np.float32)

        return out

    # -----------------------------------------------------------------------
    # 3. GPROF Processor
    # -----------------------------------------------------------------------
    @staticmethod
    def process_gprof(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Process GPROF surface precipitation and hydrometeor profiles.
        """
        out: Dict[str, Any] = {
            "is_available": False,
            "surface_precip": None,
            "convective_precip": None,
            "rain_water_path": None,
            "ice_water_path": None,
            "latitude": None,
            "longitude": None,
            "shape": (),
        }

        gp = NetCDFReader.get_group(ds, "GPROF/S1")
        if gp is None:
            return out

        if "surfacePrecipitation" in gp.variables:
            sp, mask = NetCDFReader.read_variable_2d(gp, "surfacePrecipitation")
            if sp is not None and np.any(mask):
                out["is_available"] = True
                out["surface_precip"] = sp
                out["shape"] = sp.shape
                out["latitude"], _ = NetCDFReader.read_variable_2d(gp, "latitude")
                out["longitude"], _ = NetCDFReader.read_variable_2d(gp, "longitude")

                if "convectivePrecipitation" in gp.variables:
                    out["convective_precip"], _ = NetCDFReader.read_variable_2d(gp, "convectivePrecipitation")
                if "rainWaterPath" in gp.variables:
                    out["rain_water_path"], _ = NetCDFReader.read_variable_2d(gp, "rainWaterPath")
                if "iceWaterPath" in gp.variables:
                    out["ice_water_path"], _ = NetCDFReader.read_variable_2d(gp, "iceWaterPath")

        return out

    # -----------------------------------------------------------------------
    # 4. Radar Radiometer Processor
    # -----------------------------------------------------------------------
    @staticmethod
    def process_radar(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Process spaceborne active radar measurements (TRMM PR or GPM DPR).
        Sparsely available (~8% of overpasses).
        """
        out: Dict[str, Any] = {
            "is_available": False,
            "radar_group": None,
            "ku_reflectivity_profile": None,
            "ka_reflectivity_profile": None,
            "near_surf_precip_rate": None,
            "latitude": None,
            "longitude": None,
            "scan_shape": (),
        }

        rr = NetCDFReader.get_group(ds, "radar_radiometer")
        if rr is None:
            return out

        # Check for KuGMI or KuKaGMI
        for g_candidate in ["KuKaGMI", "KuGMI"]:
            if g_candidate in rr.groups:
                rg = rr.groups[g_candidate]
                if "nearSurfPrecipTotRate" in rg.variables or "correctedReflectFactor_Ku" in rg.variables:
                    out["is_available"] = True
                    out["radar_group"] = g_candidate
                    out["latitude"], _ = NetCDFReader.read_variable_2d(rg, "latitude")
                    out["longitude"], _ = NetCDFReader.read_variable_2d(rg, "longitude")
                    if out["latitude"] is not None:
                        out["scan_shape"] = out["latitude"].shape

                    if "nearSurfPrecipTotRate" in rg.variables:
                        out["near_surf_precip_rate"], _ = NetCDFReader.read_variable_2d(rg, "nearSurfPrecipTotRate")

                    # Ku 3D reflectivity
                    if "correctedReflectFactor_Ku" in rg.variables:
                        ku_var = rg.variables["correctedReflectFactor_Ku"]
                        ku_arr, _ = get_array_with_fill(ku_var)
                        out["ku_reflectivity_profile"] = ku_arr

                    # Ka 3D reflectivity
                    if "correctedReflectFactor_Ka" in rg.variables:
                        ka_var = rg.variables["correctedReflectFactor_Ka"]
                        ka_arr, _ = get_array_with_fill(ka_var)
                        out["ka_reflectivity_profile"] = ka_arr
                    break

        return out

    # -----------------------------------------------------------------------
    # 5. Environmental Reanalysis Processor
    # -----------------------------------------------------------------------
    @staticmethod
    def process_environmental_slice(ds_env: nc.Dataset, time_index: int) -> Dict[str, Any]:
        """
        Extract the 2D environmental reanalysis fields for a single synoptic time slice.
        """
        out: Dict[str, Any] = {
            "is_available": False,
            "time_epoch": None,
            "precipitable_water": None,
            "sst": None,
            "pressure_msl": None,
            "temperature_2m": None,
            "u_10m": None,
            "v_10m": None,
            "latitude": None,
            "longitude": None,
            "grid_shape": (),
        }

        rect = NetCDFReader.get_group(ds_env, "rectilinear")
        if rect is None or "time" not in rect.variables:
            return out

        times = rect.variables["time"][:]
        if time_index < 0 or time_index >= len(times):
            return out

        out["is_available"] = True
        out["time_epoch"] = int(times[time_index])

        # Read 2D grid coordinates for this time slice
        if "latitude" in rect.variables:
            out["latitude"] = rect.variables["latitude"][time_index, :]
        if "longitude" in rect.variables:
            out["longitude"] = rect.variables["longitude"][time_index, :]

        # Extract selected surface / integrated fields
        for field_name in ["precipitable_water", "sst", "pressure_msl", "temperature_2m", "u_10m", "v_10m"]:
            if field_name in rect.variables:
                raw_var = rect.variables[field_name]
                arr_slice = raw_var[time_index, :, :]
                if isinstance(arr_slice, np.ma.MaskedArray):
                    arr_slice = arr_slice.filled(np.nan)
                out[field_name] = arr_slice
                out["grid_shape"] = arr_slice.shape

        return out

