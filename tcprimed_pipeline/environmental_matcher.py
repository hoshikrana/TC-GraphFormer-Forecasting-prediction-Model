"""
Environmental Matcher module for TC PRIMED.
Performs strictly causal temporal alignment between satellite overpasses and ERA5 reanalysis fields,
preventing future-time leakage in forecasting datasets.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import netCDF4 as nc

from .netcdf_reader import NetCDFReader, open_netcdf


class EnvironmentalMatcher:
    """Matches satellite overpass observations with environmental reanalysis slices."""

    def __init__(self, enforce_causality: bool = True):
        self.enforce_causality = enforce_causality
        # Cache environmental times: {storm_id: (env_filepath, times_array)}
        self._storm_env_cache: Dict[str, Tuple[str, np.ndarray]] = {}

    def register_storm_environmental_file(self, storm_id: str, env_filepath: str) -> None:
        """Cache the synoptic timestamps for a storm's environmental file."""
        if storm_id in self._storm_env_cache:
            return

        with open_netcdf(env_filepath, "r") as ds:
            times = None
            for gname in ["rectilinear", "cylindrical"]:
                g = NetCDFReader.get_group(ds, gname)
                if g is not None and "time" in g.variables:
                    times = np.array(g.variables["time"][:], dtype=np.int64)
                    break
            
            if times is not None:
                self._storm_env_cache[storm_id] = (env_filepath, times)

    def match_overpass(self, storm_id: str, overpass_time_epoch: int) -> Dict[str, Any]:
        """
        Match an overpass observation time with the appropriate environmental synoptic slice.
        
        If enforce_causality is True:
          Selects the most recent PAST or contemporaneous slice (t_env <= t_overpass).
          Any future slice (t_env > t_overpass) is strictly excluded.
        If enforce_causality is False (retrospective diagnostic mode):
          Selects the absolute nearest synoptic slice, flagging if it lies in the future.
          
        Returns match diagnostics dictionary.
        """
        out: Dict[str, Any] = {
            "has_match": False,
            "matched_index": None,
            "satellite_time_epoch": overpass_time_epoch,
            "env_time_epoch": None,
            "time_diff_hours": None,
            "is_future_leakage": False,
            "causality_enforced": self.enforce_causality,
            "rejection_reason": None,
            "env_filepath": None,
        }

        if storm_id not in self._storm_env_cache:
            out["rejection_reason"] = f"No environmental file registered for storm {storm_id}"
            return out

        env_path, env_times = self._storm_env_cache[storm_id]
        out["env_filepath"] = env_path

        if len(env_times) == 0:
            out["rejection_reason"] = "Environmental file contains 0 synoptic time slices"
            return out

        diffs_sec = env_times - overpass_time_epoch
        diffs_hr = diffs_sec / 3600.0

        if self.enforce_causality:
            # Causal constraint: t_env <= t_overpass (diffs_sec <= 0)
            past_indices = np.where(diffs_sec <= 0)[0]
            if len(past_indices) == 0:
                # All environmental slices are in the future relative to this overpass
                out["is_future_leakage"] = True
                out["rejection_reason"] = "All available synoptic slices occur in the future (causal violation)"
                # Report nearest future slice info for diagnostic transparency
                nearest_future_idx = int(np.argmin(diffs_sec))
                out["env_time_epoch"] = int(env_times[nearest_future_idx])
                out["time_diff_hours"] = float(diffs_hr[nearest_future_idx])
                return out

            # Pick the most recent past slice (smallest negative diff)
            chosen_idx = int(past_indices[np.argmax(diffs_sec[past_indices])])
            out["has_match"] = True
            out["matched_index"] = chosen_idx
            out["env_time_epoch"] = int(env_times[chosen_idx])
            out["time_diff_hours"] = float(diffs_hr[chosen_idx])
            out["is_future_leakage"] = False
        else:
            # Nearest synoptic slice (diagnostic mode)
            nearest_idx = int(np.argmin(np.abs(diffs_sec)))
            out["has_match"] = True
            out["matched_index"] = nearest_idx
            out["env_time_epoch"] = int(env_times[nearest_idx])
            out["time_diff_hours"] = float(diffs_hr[nearest_idx])
            out["is_future_leakage"] = bool(diffs_sec[nearest_idx] > 0)

        return out

