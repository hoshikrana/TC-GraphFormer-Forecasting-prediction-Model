"""
Storage Writer module for TC PRIMED.
Safely saves standardized per-storm NetCDF-4/HDF5 containers with atomic writes,
zlib compression, native grid preservation, and storage footprint benchmarking.
"""

import os
import json
import shutil
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import netCDF4 as nc

from .config import OutputConfig


class StorageWriter:
    """Manages writing and indexing of standardized per-storm NetCDF-4/HDF5 containers."""

    def __init__(self, output_config: Optional[OutputConfig] = None):
        self.config = output_config or OutputConfig()
        self.processed_storms: Dict[str, str] = {}
        self.raw_bytes_processed: int = 0
        self.output_bytes_written: int = 0

    def get_storm_container_path(self, season: int, basin: str, storm_id: str) -> str:
        """Determine destination path for a storm's processed container."""
        storm_dir = os.path.join(self.config.root, str(season), basin, storm_id)
        return os.path.join(storm_dir, f"{storm_id}_processed.nc")

    def is_storm_processed(self, season: int, basin: str, storm_id: str) -> bool:
        """Check if a storm container already exists and is readable."""
        cpath = self.get_storm_container_path(season, basin, storm_id)
        if not os.path.exists(cpath):
            return False
        try:
            with nc.Dataset(cpath, "r") as ds:
                if "overpasses" in ds.groups and ds.getncattr("storm_id") == storm_id:
                    return True
        except Exception:
            return False
        return False

    def write_storm_container(self,
                              storm_id: str,
                              season: int,
                              basin: str,
                              partition: str,
                              best_track: Dict[str, Any],
                              observations: List[Dict[str, Any]]) -> str:
        """
        Write a complete per-storm NetCDF-4/HDF5 container atomically.
        """
        final_path = self.get_storm_container_path(season, basin, storm_id)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        tmp_path = final_path + ".tmp"

        # Remove existing tmp if present
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        comp_level = self.config.compression_level
        do_zlib = comp_level > 0

        with nc.Dataset(tmp_path, "w", format="NETCDF4") as ds:
            # 1. Global / Root Attributes
            ds.title = f"TC PRIMED Standardized Processed Container - Storm {storm_id}"
            ds.storm_id = storm_id
            ds.season = int(season)
            ds.basin = basin
            ds.partition = partition
            ds.total_observations = len(observations)
            ds.date_created_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            ds.Conventions = "CF-1.8, TC-PRIMED-v01r01"
            ds.pipeline_version = "1.0.0"

            # 2. Sequential Best Track group (/storm_metadata)
            g_bt = ds.createGroup("storm_metadata")
            bt_times = best_track.get("times", np.array([]))
            n_bt = len(bt_times)
            if n_bt > 0:
                g_bt.createDimension("time", n_bt)
                v_t = g_bt.createVariable("time", "i8", ("time",), zlib=do_zlib, complevel=comp_level)
                v_t[:] = bt_times
                v_t.units = "seconds since 1970-01-01 00:00:00 UTC"

                if len(best_track.get("lats", [])) == n_bt:
                    v_lat = g_bt.createVariable("storm_latitude", "f4", ("time",), zlib=do_zlib, complevel=comp_level)
                    v_lat[:] = best_track["lats"]
                    v_lat.units = "degrees_north"

                if len(best_track.get("lons", [])) == n_bt:
                    v_lon = g_bt.createVariable("storm_longitude", "f4", ("time",), zlib=do_zlib, complevel=comp_level)
                    v_lon[:] = best_track["lons"]
                    v_lon.units = "degrees_east"

                if len(best_track.get("intensities", [])) == n_bt:
                    v_int = g_bt.createVariable("intensity", "f4", ("time",), zlib=do_zlib, complevel=comp_level)
                    v_int[:] = best_track["intensities"]
                    v_int.units = "kt"

                if len(best_track.get("pressures", [])) == n_bt:
                    v_prs = g_bt.createVariable("central_min_pressure", "f4", ("time",), zlib=do_zlib, complevel=comp_level)
                    v_prs[:] = best_track["pressures"]
                    v_prs.units = "hPa"

            # 3. Master /overpasses Group
            g_ops = ds.createGroup("overpasses")

            # Write each observation
            for obs in observations:
                obs_id = obs["observation_id"]
                # Clean identifier for group name (alphanumeric + underscore)
                clean_obs_name = f"obs_{obs.get('timestamp_str', 'unknown')}_{obs.get('instrument', 'unk')}"
                g_obs = g_ops.createGroup(clean_obs_name)
                g_obs.observation_id = obs_id
                g_obs.instrument = obs.get("instrument", "")
                g_obs.platform = obs.get("platform", "")
                g_obs.time_epoch = int(obs.get("timestamp_epoch") or 0)
                g_obs.time_utc = obs.get("timestamp_utc", "")
                g_obs.has_pmw = int(obs.get("has_pmw", False))
                g_obs.has_ir = int(obs.get("has_ir", False))
                g_obs.has_gprof = int(obs.get("has_gprof", False))
                g_obs.has_radar = int(obs.get("has_radar", False))
                g_obs.has_causal_env = int(obs.get("has_causal_env", False))

                # Targets subgroup
                g_tgt = g_obs.createGroup("targets")
                targets = obs.get("targets", {})
                for k, v in targets.items():
                    if v is not None and not isinstance(v, (str, bool)):
                        v_var = g_tgt.createVariable(k, "f4", ())
                        v_var.assignValue(float(v))
                    elif isinstance(v, str):
                        setattr(g_tgt, k, v)
                    elif isinstance(v, bool):
                        v_var = g_tgt.createVariable(k, "i1", ())
                        v_var.assignValue(int(v))

                # Modality: Passive Microwave
                pmw = obs.get("pmw_data", {})
                if pmw and pmw.get("is_available"):
                    g_pmw = g_obs.createGroup("passive_microwave")
                    for sw_name, sw_dict in pmw.get("swaths", {}).items():
                        g_sw = g_pmw.createGroup(sw_name)
                        lat = sw_dict.get("latitude")
                        lon = sw_dict.get("longitude")
                        if lat is not None and lon is not None:
                            dim_scan, dim_pix = lat.shape
                            dim_scan_name = f"{sw_name}_scan"
                            dim_pix_name = f"{sw_name}_pixel"
                            g_sw.createDimension(dim_scan_name, dim_scan)
                            g_sw.createDimension(dim_pix_name, dim_pix)

                            v_lat = g_sw.createVariable("latitude", "f4", (dim_scan_name, dim_pix_name), zlib=do_zlib, complevel=comp_level)
                            v_lat[:] = lat
                            v_lat.units = "degrees_north"

                            v_lon = g_sw.createVariable("longitude", "f4", (dim_scan_name, dim_pix_name), zlib=do_zlib, complevel=comp_level)
                            v_lon[:] = lon
                            v_lon.units = "degrees_east"

                            # Write TB channels
                            for ch_name, ch_dict in sw_dict.get("channels", {}).items():
                                ch_data = ch_dict.get("data")
                                if ch_data is not None and ch_data.shape == (dim_scan, dim_pix):
                                    v_ch = g_sw.createVariable(ch_name, "f4", (dim_scan_name, dim_pix_name),
                                                               fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                                    v_ch[:] = ch_data
                                    v_ch.units = ch_dict.get("units", "K")
                                    v_ch.long_name = ch_dict.get("long_name", ch_name)

                # Modality: Infrared
                ir = obs.get("ir_data", {})
                if ir and ir.get("is_available"):
                    g_ir = g_obs.createGroup("infrared")
                    g_ir.provenance = ir.get("provenance", "")
                    g_ir.nominal_resolution_km = float(ir.get("nominal_resolution_km") or 0.0)
                    g_ir.wavelength_descriptor = ir.get("wavelength_descriptor", "")
                    g_ir.platform = ir.get("platform", "")

                    ir_img = ir.get("irwin_image")
                    if ir_img is not None:
                        ny, nx = ir_img.shape
                        g_ir.createDimension("ir_y", ny)
                        g_ir.createDimension("ir_x", nx)
                        v_ir = g_ir.createVariable("IRWIN", "f4", ("ir_y", "ir_x"),
                                                   fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                        v_ir[:] = ir_img
                        v_ir.units = "K"
                        v_ir.long_name = ir.get("wavelength_descriptor", "IR Window Brightness Temperature")

                    # R5 vortex size time series (degrees latitude along R5_time)
                    if ir.get("r5_available") and ir.get("r5_values") is not None:
                        r5_vals = ir["r5_values"]
                        g_ir.createDimension("R5_time", len(r5_vals))
                        v_r5 = g_ir.createVariable("R5", "f4", ("R5_time",), zlib=do_zlib, complevel=comp_level)
                        v_r5[:] = r5_vals
                        v_r5.units = "degrees_latitude"
                        v_r5.long_name = "Radius of mean tangential winds of 5 knots at 850 hPa level"
                        if ir.get("r5_times") is not None and len(ir["r5_times"]) == len(r5_vals):
                            v_r5_t = g_ir.createVariable("R5_time", "i8", ("R5_time",), zlib=do_zlib, complevel=comp_level)
                            v_r5_t[:] = ir["r5_times"]
                            v_r5_t.units = "seconds since 1970-01-01 00:00:00 UTC"

                    # 30-bin radial profile of brightness temperature
                    if ir.get("radial_profile_tb") is not None and ir.get("radial_profile_radius_km") is not None:
                        rad_bins = len(ir["radial_profile_radius_km"])
                        g_ir.createDimension("radius", rad_bins)
                        v_rad = g_ir.createVariable("radius", "f4", ("radius",), zlib=do_zlib, complevel=comp_level)
                        v_rad[:] = ir["radial_profile_radius_km"]
                        v_rad.units = "km"
                        v_rad.long_name = "Bin centers for radial profile of mean binned infrared brightness temperature"

                        v_prof = g_ir.createVariable("brightness_temperature_mean_radial_profile", "f4", ("radius",),
                                                     fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                        v_prof[:] = ir["radial_profile_tb"]
                        v_prof.units = "K"

                # Modality: GPROF
                gp = obs.get("gprof_data", {})
                if gp and gp.get("is_available"):
                    g_gp = g_obs.createGroup("GPROF")
                    sp = gp.get("surface_precip")
                    if sp is not None:
                        ny, nx = sp.shape
                        g_gp.createDimension("gp_scan", ny)
                        g_gp.createDimension("gp_pixel", nx)
                        v_sp = g_gp.createVariable("surfacePrecipitation", "f4", ("gp_scan", "gp_pixel"),
                                                   fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                        v_sp[:] = sp
                        v_sp.units = "mm/hr"

                        # Additional GPROF hydrometeor fields if available
                        for g_field, g_name, g_unit in [
                            ("convective_precip", "convectivePrecipitation", "mm/hr"),
                            ("rain_water_path", "rainWaterPath", "kg/m^2"),
                            ("ice_water_path", "iceWaterPath", "kg/m^2"),
                        ]:
                            f_data = gp.get(g_field)
                            if f_data is not None and f_data.shape == (ny, nx):
                                v_gf = g_gp.createVariable(g_name, "f4", ("gp_scan", "gp_pixel"),
                                                          fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                                v_gf[:] = f_data
                                v_gf.units = g_unit

                # Modality: Radar
                rad = obs.get("radar_data", {})
                if rad and rad.get("is_available"):
                    g_rad = g_obs.createGroup("radar_radiometer")
                    g_rad.radar_group_name = rad.get("radar_group", "")
                    ku = rad.get("ku_reflectivity_profile")
                    if ku is not None and len(ku.shape) == 3:
                        n_range, n_scan, n_beam = ku.shape
                        g_rad.createDimension("radar_range", n_range)
                        g_rad.createDimension("radar_scan", n_scan)
                        g_rad.createDimension("radar_beam", n_beam)
                        v_ku = g_rad.createVariable("correctedReflectFactor_Ku", "f4", ("radar_range", "radar_scan", "radar_beam"),
                                                    fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                        v_ku[:] = ku
                        v_ku.units = "dBZ"

                    ka = rad.get("ka_reflectivity_profile")
                    if ka is not None and len(ka.shape) == 3:
                        n_range, n_scan, n_beam = ka.shape
                        if "radar_range" not in g_rad.dimensions:
                            g_rad.createDimension("radar_range", n_range)
                        if "radar_scan" not in g_rad.dimensions:
                            g_rad.createDimension("radar_scan", n_scan)
                        if "radar_beam" not in g_rad.dimensions:
                            g_rad.createDimension("radar_beam", n_beam)
                        v_ka = g_rad.createVariable("correctedReflectFactor_Ka", "f4", ("radar_range", "radar_scan", "radar_beam"),
                                                    fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                        v_ka[:] = ka
                        v_ka.units = "dBZ"

                # Modality: Environmental Reanalysis Slice
                env_slice = obs.get("env_slice_data", {})
                if env_slice and env_slice.get("is_available"):
                    g_env = g_obs.createGroup("environmental_synoptic")
                    g_env.time_epoch = int(env_slice.get("time_epoch") or 0)
                    g_env.time_diff_hours = float(obs.get("env_time_diff_hours") or 0.0)
                    g_env.is_causal = int(obs.get("env_is_causal", True))

                    grid_shape = env_slice.get("grid_shape")
                    ny, nx = (None, None)
                    pw = env_slice.get("precipitable_water")
                    if pw is not None:
                        ny, nx = pw.shape
                    elif grid_shape:
                        ny, nx = grid_shape

                    if ny is not None and nx is not None:
                        g_env.createDimension("env_ny", ny)
                        g_env.createDimension("env_nx", nx)

                        env_vars_to_write = [
                            ("precipitable_water", "precipitable_water", "kg/m^2"),
                            ("sst", "sst", "K"),
                            ("pressure_msl", "pressure_msl", "hPa"),
                            ("temperature_2m", "temperature_2m", "K"),
                            ("u_10m", "u_10m", "m/s"),
                            ("v_10m", "v_10m", "m/s"),
                        ]
                        for env_key, env_name, env_unit in env_vars_to_write:
                            arr = env_slice.get(env_key)
                            if arr is not None and arr.shape == (ny, nx):
                                v_e = g_env.createVariable(env_name, "f4", ("env_ny", "env_nx"),
                                                          fill_value=-9999.0, zlib=do_zlib, complevel=comp_level)
                                v_e[:] = arr
                                v_e.units = env_unit

        # Atomically rename tmp to final path
        if os.path.exists(final_path):
            os.remove(final_path)
        shutil.move(tmp_path, final_path)

        out_sz = os.path.getsize(final_path)
        self.output_bytes_written += out_sz
        self.processed_storms[storm_id] = final_path
        return final_path

    def export_storage_benchmark(self,
                                 total_pilot_raw_bytes: int,
                                 output_path: str) -> str:
        """
        Compile and save storage_benchmark.json, estimating required storage for 500 GB dataset.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        raw_gb = total_pilot_raw_bytes / (1024**3)
        proc_gb = self.output_bytes_written / (1024**3)
        comp_ratio = raw_gb / proc_gb if proc_gb > 0 else 1.0

        # Estimate full 500 GB raw deployment footprint
        target_raw_gb = 500.0
        est_processed_gb = target_raw_gb / comp_ratio if comp_ratio > 0 else target_raw_gb
        total_required_disk_gb = target_raw_gb + est_processed_gb + 20.0  # +20 GB buffer

        benchmark = {
            "pilot_benchmark": {
                "total_raw_pilot_bytes": total_pilot_raw_bytes,
                "total_raw_pilot_gb": round(raw_gb, 2),
                "total_processed_bytes": self.output_bytes_written,
                "total_processed_gb": round(proc_gb, 2),
                "compression_ratio_achieved": round(comp_ratio, 2),
                "storage_reduction_pct": round(100.0 * (1.0 - (proc_gb / raw_gb)), 2) if raw_gb > 0 else 0.0,
                "average_bytes_per_storm": round(self.output_bytes_written / len(self.processed_storms), 0) if self.processed_storms else 0,
            },
            "future_500gb_deployment_estimate": {
                "assumed_raw_archive_gb": target_raw_gb,
                "projected_processed_footprint_gb": round(est_processed_gb, 2),
                "projected_total_storage_needed_gb": round(total_required_disk_gb, 2),
                "feasibility_assessment": (
                    "A 500 GB raw dataset and its processed derivatives CANNOT fit on a single 500 GB drive "
                    f"(estimated total storage needed is ~{round(total_required_disk_gb, 1)} GB). "
                    "Recommendation: Deploy on at least a 1 TB external SSD or partition raw archive and processed outputs across separate disks."
                ),
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2)

        return output_path

