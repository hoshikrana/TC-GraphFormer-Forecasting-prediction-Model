"""
Metadata Extractor module for TC PRIMED.
Extracts overpass metadata, contemporaneous Best Track targets, and sequential Best Track tracks.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import netCDF4 as nc

from .netcdf_reader import clean_scalar, NetCDFReader


def format_epoch_seconds(s: Optional[int]) -> str:
    """Format epoch seconds as UTC ISO string."""
    if s is None or s <= 0:
        return ""
    try:
        dt = datetime.fromtimestamp(s, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, OverflowError, OSError):
        return ""


class MetadataExtractor:
    """Extracts and standardizes metadata from overpass and environmental NetCDF files."""

    @staticmethod
    def extract_overpass_metadata(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Extract variables from overpass_metadata group.
        """
        out: Dict[str, Any] = {
            "time_epoch": None,
            "time_utc": "",
            "basin": "",
            "cyclone_number": None,
            "season": None,
            "instrument": "",
            "platform": "",
            "granule_number": "",
            "coverage_fraction": None,
            "env_time_epoch": None,
            "env_time_utc": "",
        }

        g = NetCDFReader.get_group(ds, "overpass_metadata")
        if g is None:
            return out

        t = NetCDFReader.read_variable_scalar(g, "time")
        out["time_epoch"] = int(t) if t is not None else None
        out["time_utc"] = format_epoch_seconds(out["time_epoch"])

        out["basin"] = str(NetCDFReader.read_variable_scalar(g, "basin") or "").strip()
        out["cyclone_number"] = clean_scalar(NetCDFReader.read_variable_scalar(g, "cyclone_number"))
        out["season"] = clean_scalar(NetCDFReader.read_variable_scalar(g, "season"))
        out["instrument"] = str(NetCDFReader.read_variable_scalar(g, "instrument_name") or "").strip()
        out["platform"] = str(NetCDFReader.read_variable_scalar(g, "platform_name") or "").strip()
        out["granule_number"] = str(NetCDFReader.read_variable_scalar(g, "granule_number") or "").strip()
        out["coverage_fraction"] = clean_scalar(NetCDFReader.read_variable_scalar(g, "coverage_fraction"))

        env_t = NetCDFReader.read_variable_scalar(g, "env_time")
        out["env_time_epoch"] = int(env_t) if env_t is not None else None
        out["env_time_utc"] = format_epoch_seconds(out["env_time_epoch"])

        return out

    @staticmethod
    def extract_overpass_targets(ds: nc.Dataset) -> Dict[str, Any]:
        """
        Extract contemporaneous Best Track targets from overpass_storm_metadata.
        Includes current intensity, min pressure, development stage, and future delta V.
        """
        out: Dict[str, Any] = {
            "storm_latitude": None,
            "storm_longitude": None,
            "intensity_vmax_kt": None,
            "central_min_pressure_hpa": None,
            "development_stage_raw": "UNKNOWN",
            "storm_speed_mps": None,
            "storm_heading_deg": None,
            "distance_to_land_km": None,
            "intensity_change_6h_kt": None,
            "intensity_change_12h_kt": None,
            "intensity_change_18h_kt": None,
            "intensity_change_24h_kt": None,
            "is_rapid_intensification_24h": None,
        }

        g = NetCDFReader.get_group(ds, "overpass_storm_metadata")
        if g is None:
            return out

        lat = clean_scalar(NetCDFReader.read_variable_scalar(g, "storm_latitude"))
        lon = clean_scalar(NetCDFReader.read_variable_scalar(g, "storm_longitude"))
        out["storm_latitude"] = float(lat) if lat is not None else None
        out["storm_longitude"] = float(lon) if lon is not None else None

        v = clean_scalar(NetCDFReader.read_variable_scalar(g, "intensity"))
        out["intensity_vmax_kt"] = float(v) if v is not None and v > 0 else None

        p = clean_scalar(NetCDFReader.read_variable_scalar(g, "central_min_pressure"))
        out["central_min_pressure_hpa"] = float(p) if p is not None and p > 800 else None

        stg = clean_scalar(NetCDFReader.read_variable_scalar(g, "development_level"))
        out["development_stage_raw"] = str(stg).strip() if stg is not None else "UNKNOWN"

        spd = clean_scalar(NetCDFReader.read_variable_scalar(g, "storm_speed"))
        out["storm_speed_mps"] = float(spd) if spd is not None and spd >= 0 else None

        hdg = clean_scalar(NetCDFReader.read_variable_scalar(g, "storm_heading"))
        out["storm_heading_deg"] = float(hdg) if hdg is not None and 0 <= hdg <= 360 else None

        dst = clean_scalar(NetCDFReader.read_variable_scalar(g, "distance_to_land"))
        out["distance_to_land_km"] = float(dst) if dst is not None else None

        # Intensity changes: indices corresponding to periods [-24, -18, -12, -6, 0, 6, 12, 18, 24]
        # index 5 = +6h, index 6 = +12h, index 7 = +18h, index 8 = +24h
        if "intensity_change" in g.variables:
            ic_raw = g.variables["intensity_change"][:]
            if ic_raw.size >= 9:
                if not np.ma.is_masked(ic_raw[5]):
                    out["intensity_change_6h_kt"] = float(ic_raw[5])
                if not np.ma.is_masked(ic_raw[6]):
                    out["intensity_change_12h_kt"] = float(ic_raw[6])
                if not np.ma.is_masked(ic_raw[7]):
                    out["intensity_change_18h_kt"] = float(ic_raw[7])
                if not np.ma.is_masked(ic_raw[8]):
                    val_24h = float(ic_raw[8])
                    out["intensity_change_24h_kt"] = val_24h
                    out["is_rapid_intensification_24h"] = 1 if val_24h >= 30.0 else 0

        return out

    @staticmethod
    def extract_environmental_best_track(ds_env: nc.Dataset) -> Dict[str, Any]:
        """
        Extract the complete Best Track sequence from environmental file's storm_metadata group.
        This provides the sequence of (time, lat, lon, intensity, pressure) for tracking.
        """
        out: Dict[str, Any] = {
            "times": np.array([], dtype=np.int64),
            "lats": np.array([], dtype=np.float32),
            "lons": np.array([], dtype=np.float32),
            "intensities": np.array([], dtype=np.float32),
            "pressures": np.array([], dtype=np.float32),
        }

        g = NetCDFReader.get_group(ds_env, "storm_metadata")
        if g is None:
            return out

        if "time" in g.variables:
            t_raw = g.variables["time"][:]
            out["times"] = np.array(t_raw, dtype=np.int64)

        if "storm_latitude" in g.variables:
            lat_raw = g.variables["storm_latitude"][:]
            out["lats"] = np.array(lat_raw, dtype=np.float32)

        if "storm_longitude" in g.variables:
            lon_raw = g.variables["storm_longitude"][:]
            out["lons"] = np.array(lon_raw, dtype=np.float32)

        if "intensity" in g.variables:
            v_raw = g.variables["intensity"][:]
            out["intensities"] = np.array(v_raw, dtype=np.float32)

        if "central_min_pressure" in g.variables:
            p_raw = g.variables["central_min_pressure"][:]
            out["pressures"] = np.array(p_raw, dtype=np.float32)

        return out

    @staticmethod
    def query_future_track_coordinates(best_track: Dict[str, Any],
                                      obs_time_epoch: int,
                                      horizons_hours: List[int] = [6, 12, 18, 24],
                                      max_time_tolerance_seconds: int = 1800) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Query verified future coordinates from the sequential Best Track.
        Returns {f'horizon_{h}h': (lat, lon) or None}.
        If the cyclone dissipated or tracking stopped within horizon h, returns None.
        NEVER extrapolates, synthesizes, or substitutes motion vectors.
        """
        out: Dict[str, Optional[Tuple[float, float]]] = {}
        times = best_track.get("times", np.array([]))
        lats = best_track.get("lats", np.array([]))
        lons = best_track.get("lons", np.array([]))

        if len(times) == 0 or len(lats) == 0:
            for h in horizons_hours:
                out[f"future_track_{h}h"] = None
            return out

        for h in horizons_hours:
            target_t = obs_time_epoch + (h * 3600)
            diffs = np.abs(times - target_t)
            min_idx = int(np.argmin(diffs))
            if diffs[min_idx] <= max_time_tolerance_seconds:
                future_lat = float(lats[min_idx])
                future_lon = float(lons[min_idx])
                # Bounds check
                if -90 <= future_lat <= 90 and -180 <= future_lon <= 360:
                    out[f"future_track_{h}h"] = (future_lat, future_lon)
                else:
                    out[f"future_track_{h}h"] = None
            else:
                out[f"future_track_{h}h"] = None

        return out

