"""
PyTorch Dataset for NOAA TC PRIMED v01r01-final.
Provides multi-modal spatiotemporal overpass tensors with modality masks,
channel validity masks, and multi-task forecasting targets from standardized NetCDF-4/HDF5 containers.
Supports both single-overpass and historical storm sequence workflows.
"""

import os
import math
import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..models.graph.grid_geometry import GLOBAL_GRID_GEOMETRY, GridGeometry

logger = logging.getLogger(__name__)

# Canonical physical radar height levels (16 levels in km, 0.5 to 19.5 km)
TARGET_RADAR_HEIGHTS_KM: Tuple[float, ...] = tuple(
    np.linspace(0.5, 19.5, 16, dtype=np.float32).tolist()
)

# Calibrated geospatial resampling distance thresholds by sensor footprint and resolution
DEFAULT_RESAMPLING_THRESHOLDS_KM: Dict[str, float] = {
    "ir": 15.0,     # Native IR footprint ~4 km; 15 km threshold preserves contiguous field
    "pmw": 35.0,    # Low-frequency PMW channels (19 GHz) have ~30-50 km footprint
    "gprof": 20.0,  # GPROF retrieved precipitation swath grid ~10-15 km
    "radar": 15.0,  # Spaceborne DPR/PR beam footprint ~5 km (15 km captures up to 3 cross-track beams)
    "env": 40.0,    # ERA5 0.25 deg grid spacing ~28 km at tropics
}


class SampleExclusionError(RuntimeError):
    """Raised when an observation cannot be processed under an exclude_sample policy."""
    pass

# Canonical 5-class stage mapping (0..4) with invalid stage mask for unmapped/unknown
# 0: Depression (TD, SD, MD)
# 1: Tropical/Subtropical Storm (TS, SS)
# 2: Hurricane/Typhoon (HU, TY)
# 3: Extratropical Cyclone (EX)
# 4: Disturbance / Low / Wave (DB, LO, WV)
STAGE_MAPPING_CANONICAL = {
    "TD": 0, "SD": 0, "MD": 0,
    "TS": 1, "SS": 1,
    "HU": 2, "TY": 2,
    "EX": 3,
    "DB": 4, "LO": 4, "WV": 4,
}
STAGE_MAPPING = STAGE_MAPPING_CANONICAL
NUM_STAGES = 5

# Canonical PMW 4-channel mapping across supported sensor families
# Channel 0: ~19 GHz V (18.7V, 19.35V)
# Channel 1: ~19 GHz H (18.7H, 19.35H)
# Channel 2: ~37 GHz V (36.64V, 37.0V)
# Channel 3: ~85-91 GHz V (85.5V, 89.0V)
SENSOR_PMW_CHANNEL_MAP = {
    "SSMI": {
        0: ("S1", "TB_19.35V"),
        1: ("S1", "TB_19.35H"),
        2: ("S1", "TB_37.0V"),
        3: ("S2", "TB_85.5V"),
    },
    "SSMIS": {
        0: ("S1", "TB_19.35V"),
        1: ("S1", "TB_19.35H"),
        2: ("S2", "TB_37.0V"),
        3: None,
    },
    "GMI": {
        0: ("S1", "TB_18.7V"),
        1: ("S1", "TB_18.7H"),
        2: ("S1", "TB_36.64V"),
        3: ("S1", "TB_89.0V"),
    },
    "TMI": {
        0: ("S2", "TB_19.35V"),
        1: ("S2", "TB_19.35H"),
        2: ("S2", "TB_37.0V"),
        3: None,
    },
    "AMSR2": {
        0: ("S2", "TB_18.7V"),
        1: ("S2", "TB_18.7H"),
        2: None,
        3: None,
    },
    "AMSRE": {
        0: ("S2", "TB_18.7V"),
        1: ("S2", "TB_18.7H"),
        2: None,
        3: None,
    },
    "AMSUB": {
        0: None,
        1: None,
        2: None,
        3: ("S1", "TB_89.0_0.9QV"),
    },
    "MHS": {
        0: None,
        1: None,
        2: None,
        3: ("S1", "TB_89.0V"),
    },
    "ATMS": {
        0: None,
        1: None,
        2: None,
        3: None,
    },
}


def compute_tangent_plane_displacement_km(
    lat0: float, lon0: float, lat1: float, lon1: float
) -> Tuple[float, float]:
    """
    Compute local tangent-plane displacement (dx_east_km, dy_north_km)
    from (lat0, lon0) to (lat1, lon1) with proper longitude wrapping.
    Earth radius = 6371.0 km.
    """
    R_earth = 6371.0
    # Longitude difference wrapped to [-180, 180]
    d_lon = ((lon1 - lon0 + 180.0) % 360.0) - 180.0
    d_lat = lat1 - lat0

    dx_km = d_lon * (math.pi / 180.0) * R_earth * math.cos(math.radians(lat0))
    dy_km = d_lat * (math.pi / 180.0) * R_earth
    return float(dx_km), float(dy_km)


def displacement_to_geodetic(
    lat0: float, lon0: float, dx_km: float, dy_km: float
) -> Tuple[float, float]:
    """
    Invert tangent plane displacement back to geographic (lat, lon).
    """
    R_earth = 6371.0
    d_lat_deg = (dy_km / R_earth) * (180.0 / math.pi)
    lat1 = lat0 + d_lat_deg

    cos_lat = max(1e-4, math.cos(math.radians(lat0)))
    d_lon_deg = (dx_km / (R_earth * cos_lat)) * (180.0 / math.pi)
    lon1 = (lon0 + d_lon_deg + 360.0) % 360.0
    return float(lat1), float(lon1)


class TCPRIMEDDataset(Dataset):
    """
    Random-access PyTorch Dataset for single TC PRIMED overpass samples.
    Reads directly from standardized NetCDF-4/HDF5 containers with strict
    mask-aware preprocessing and physical validation.
    """

    # Bounded LRU in-memory sample cache across dataset instances (safe for 100 GB scale)
    _GLOBAL_SAMPLE_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[Any, ...]]" = OrderedDict()
    DEFAULT_MAX_CACHE_ENTRIES: int = 2048
    DEFAULT_DISK_CACHE_DIR: str = "data/cache/resampled_64x64"
    CACHE_PREPROCESSING_VERSION: str = "v3_strict_causal_kin_cappi16_grid64"

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.geospatial_policy = "nominal_fallback"
        instance.strict_mode = False
        instance._query_pts_cache = {}
        instance._raw_radar_coord_cache = {}
        instance._grp_resample_cache = {}
        instance._sample_cache = cls._GLOBAL_SAMPLE_CACHE
        instance.max_cache_entries = cls.DEFAULT_MAX_CACHE_ENTRIES
        instance.disk_cache_dir = cls.DEFAULT_DISK_CACHE_DIR
        instance.enable_memory_cache = True
        instance.enable_disk_cache = True
        instance.processing_errors = []
        return instance

    def __init__(
        self,
        sample_index_csv: str = "reports/pipeline/sample_index.csv",
        partition: Optional[str] = "TRAIN",
        grid_size: int = 64,
        max_samples: Optional[int] = None,
        strict_mode: bool = False,
        geospatial_policy: Optional[str] = None,
        resampling_thresholds: Optional[Dict[str, float]] = None,
        transform=None,
        basin: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        raw_data_dir: Optional[str] = "data/raw/pilot",
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        disk_cache_dir: Optional[str] = DEFAULT_DISK_CACHE_DIR,
    ):
        super().__init__()
        self.sample_index_csv = sample_index_csv
        self.partition = partition.upper() if partition else None
        self.grid_size = grid_size
        self.strict_mode = strict_mode
        # Geospatial policy: "raise", "exclude_modality", "exclude_sample", or "nominal_fallback"
        if geospatial_policy is not None:
            policy = geospatial_policy.lower()
            if policy not in ("raise", "exclude_modality", "exclude_sample", "nominal_fallback"):
                raise ValueError(
                    f"Invalid geospatial_policy '{geospatial_policy}'. "
                    f"Must be one of: 'raise', 'exclude_modality', 'exclude_sample', 'nominal_fallback'."
                )
            self.geospatial_policy = policy
        elif strict_mode:
            self.geospatial_policy = "raise"
        else:
            self.geospatial_policy = "nominal_fallback"

        self.resampling_thresholds = {
            **DEFAULT_RESAMPLING_THRESHOLDS_KM,
            **(resampling_thresholds or {}),
        }
        self.transform = transform
        self.basin = basin
        self.raw_data_dir = raw_data_dir if (raw_data_dir and os.path.exists(raw_data_dir)) else None
        self._raw_radar_coord_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._query_pts_cache: Dict[Tuple[int, int, bool], np.ndarray] = {}
        self._grp_resample_cache: Dict[Tuple[Any, ...], Any] = {}
        if not hasattr(self, "_sample_cache") or self._sample_cache is None:
            self._sample_cache = self._GLOBAL_SAMPLE_CACHE
        self.max_cache_entries = max(1, int(max_cache_entries))
        self.disk_cache_dir = disk_cache_dir
        if not hasattr(self, "enable_memory_cache"):
            self.enable_memory_cache = True
        if not hasattr(self, "enable_disk_cache"):
            self.enable_disk_cache = True
        self.processing_errors: List[Dict[str, Any]] = []

        if df is not None:
            self.df = df.copy()
            if "partition" in self.df.columns and len(self.df) > 0:
                uniq_p = self.df["partition"].dropna().unique().tolist()
                if len(uniq_p) == 1:
                    self.partition = str(uniq_p[0]).upper()
        else:
            if not os.path.exists(sample_index_csv):
                raise FileNotFoundError(f"Sample index not found at: {sample_index_csv}")
            self.df = pd.read_csv(sample_index_csv)

        if self.partition and df is None:
            self.df = self.df[self.df["partition"] == self.partition].copy()

        if self.basin:
            self.df = self.df[self.df["basin"] == self.basin].copy()

        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].copy()

        self.df = self.df.reset_index(drop=True)
        print(f"[DATASET] Loaded {len(self.df)} samples for partition: {self.partition or 'ALL'} (basin: {self.basin or 'ALL'})")

    def __len__(self) -> int:
        return len(self.df)

    def _resize_to_grid_with_mask(
        self, arr: np.ndarray, target_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate 2D numpy array to canonical (1, target_h, target_w) tensor
        while preserving valid pixel mask.
        Distinguishes missing pixels (NaN or <= -9000) from valid zero measurements.
        """
        target_h, target_w = target_shape
        if arr is None or arr.size == 0 or np.all(np.isnan(arr)):
            zero_tensor = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            zero_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            return zero_tensor, zero_mask

        # Detect valid pixels
        valid = (~np.isnan(arr)) & (arr > -9000.0)
        if not np.any(valid):
            zero_tensor = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            zero_mask = torch.zeros((1, target_h, target_w), dtype=torch.float32)
            return zero_tensor, zero_mask

        mean_val = float(np.nanmean(arr[valid]))
        # Cleaned array imputes with local valid mean strictly to prevent NaN propagation in interpolation
        arr_clean = np.where(valid, arr, mean_val).astype(np.float32)
        mask_clean = valid.astype(np.float32)

        t_data = torch.from_numpy(arr_clean).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        t_mask = torch.from_numpy(mask_clean).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        data_resized = F.interpolate(t_data, size=target_shape, mode="bilinear", align_corners=False).squeeze(0)  # (1, H, W)
        mask_resized = F.interpolate(t_mask, size=target_shape, mode="nearest").squeeze(0)  # (1, H, W)

        # Zero out invalid pixels so imputed values never pollute valid tensor values
        data_clean = data_resized * mask_resized
        return data_clean, mask_resized

    def _resample_swath_with_coords(
        self,
        arr: np.ndarray,
        grp: Optional[nc.Group],
        storm_lat: Optional[float],
        storm_lon: Optional[float],
        target_shape: Tuple[int, int] = (64, 64),
        domain_km: float = 500.0,
        max_dist_km: float = 35.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Resample 2D swath array onto storm-centered reference grid [-domain_km, +domain_km].
        Uses actual UTM or geodetic coordinates when present in grp; falls back cleanly
        to nominal grid interpolation if coordinates are missing.
        Enforces distance_upper_bound=max_dist_km so pixels outside swath are strictly masked invalid (0).
        """
        if arr is None or arr.size == 0 or np.all(np.isnan(arr)):
            z_t = torch.zeros((1, target_shape[0], target_shape[1]), dtype=torch.float32)
            z_m = torch.zeros((1, target_shape[0], target_shape[1]), dtype=torch.float32)
            return z_t, z_m, "empty"

        x_2d, y_2d = None, None
        source = "nominal_fallback"
        grp_cache = getattr(self, "_grp_resample_cache", None)
        coord_cache_key = ("coords", id(grp), arr.shape, storm_lat, storm_lon) if grp is not None else None

        if grp is not None:
            if grp_cache is not None and coord_cache_key in grp_cache:
                x_2d, y_2d, source = grp_cache[coord_cache_key]
            else:
                try:
                    if "x" in grp.variables and "y" in grp.variables:
                        x_raw = np.ma.filled(grp.variables["x"][:], np.nan)
                        y_raw = np.ma.filled(grp.variables["y"][:], np.nan)
                        if x_raw.ndim == 1 and y_raw.ndim == 1:
                            if (len(y_raw), len(x_raw)) == arr.shape:
                                x_2d, y_2d = np.meshgrid(x_raw, y_raw)
                                source = "utm_storm_centered_km"
                            elif (len(x_raw), len(y_raw)) == arr.shape:
                                y_2d, x_2d = np.meshgrid(y_raw, x_raw)
                                source = "utm_storm_centered_km"
                        elif x_raw.shape == arr.shape and y_raw.shape == arr.shape:
                            x_2d, y_2d = x_raw, y_raw
                            source = "utm_storm_centered_km"
                    elif "latitude" in grp.variables and "longitude" in grp.variables and storm_lat is not None and storm_lon is not None:
                        lat_raw = np.ma.filled(grp.variables["latitude"][:], np.nan)
                        lon_raw = np.ma.filled(grp.variables["longitude"][:], np.nan)
                        if lat_raw.ndim == 1 and lon_raw.ndim == 1:
                            if (len(lat_raw), len(lon_raw)) == arr.shape:
                                lat_2d, lon_2d = np.meshgrid(lat_raw, lon_raw, indexing="ij")
                            elif (len(lon_raw), len(lat_raw)) == arr.shape:
                                lon_2d, lat_2d = np.meshgrid(lon_raw, lat_raw, indexing="ij")
                            else:
                                lat_2d, lon_2d = None, None
                            if lat_2d is not None and lon_2d is not None:
                                cos_lat = max(1e-4, math.cos(math.radians(storm_lat)))
                                d_lon = ((lon_2d - storm_lon + 180.0) % 360.0) - 180.0
                                x_2d = d_lon * cos_lat * 111.32
                                y_2d = (lat_2d - storm_lat) * 110.57
                                source = "tangent_plane_geodetic_km"
                        elif lat_raw.shape == arr.shape and lon_raw.shape == arr.shape:
                            cos_lat = max(1e-4, math.cos(math.radians(storm_lat)))
                            d_lon = ((lon_raw - storm_lon + 180.0) % 360.0) - 180.0
                            x_2d = d_lon * cos_lat * 111.32
                            y_2d = (lat_raw - storm_lat) * 110.57
                            source = "tangent_plane_geodetic_km"
                except Exception:
                    x_2d, y_2d = None, None
                    source = "nominal_fallback"
                if grp_cache is not None and coord_cache_key is not None:
                    grp_cache[coord_cache_key] = (x_2d, y_2d, source)

        if x_2d is not None and y_2d is not None and HAS_SCIPY:
            valid_pts = (~np.isnan(arr)) & (arr > -9000.0) & (~np.isnan(x_2d)) & (~np.isnan(y_2d))
            valid_pts = valid_pts & (np.abs(x_2d) <= domain_km * 2.0) & (np.abs(y_2d) <= domain_km * 2.0)

            if np.any(valid_pts):
                vals = arr[valid_pts].flatten()
                tree_cache_key = ("tree", id(grp), arr.shape, target_shape, domain_km, max_dist_km) if grp is not None else None
                cached_tree_res = grp_cache.get(tree_cache_key) if (grp_cache is not None and tree_cache_key is not None) else None

                if cached_tree_res is not None and np.array_equal(valid_pts, cached_tree_res[0]):
                    _, is_valid, matched_indices = cached_tree_res
                else:
                    coords = np.column_stack([x_2d[valid_pts].flatten(), y_2d[valid_pts].flatten()])
                    tree = cKDTree(coords)
                    # Cell-centered target coordinates using unified GridGeometry with query caching
                    cache_key = (target_shape[0], target_shape[1], False)
                    if cache_key not in self._query_pts_cache:
                        gx, gy = GLOBAL_GRID_GEOMETRY.get_coords(target_shape[0], target_shape[1], is_feature_grid=False)
                        grid_x, grid_y = np.meshgrid(gx, gy)
                        self._query_pts_cache[cache_key] = np.column_stack([grid_x.flatten(), grid_y.flatten()])
                    query_pts = self._query_pts_cache[cache_key]

                    dists, indices = tree.query(query_pts, distance_upper_bound=max_dist_km)
                    is_valid = dists <= max_dist_km
                    matched_indices = indices[is_valid]
                    if grp_cache is not None and tree_cache_key is not None:
                        grp_cache[tree_cache_key] = (valid_pts, is_valid, matched_indices)

                grid_vals = np.zeros(target_shape[0] * target_shape[1], dtype=np.float32)
                grid_mask = np.zeros(target_shape[0] * target_shape[1], dtype=np.float32)
                grid_vals[is_valid] = vals[matched_indices].astype(np.float32)
                grid_mask[is_valid] = 1.0

                t_data = torch.from_numpy(grid_vals.reshape(1, target_shape[0], target_shape[1]))
                t_mask = torch.from_numpy(grid_mask.reshape(1, target_shape[0], target_shape[1]))
                return t_data, t_mask, source

        # Policy enforcement when coordinates are missing or unregistered
        if self.strict_mode or self.geospatial_policy == "raise":
            raise ValueError(f"Strict scientific mode failure: missing coordinate metadata in group {getattr(grp, 'name', 'unnamed')}")
        elif self.geospatial_policy == "exclude_modality":
            zero_tensor = torch.zeros((1, target_shape[0], target_shape[1]), dtype=torch.float32)
            zero_mask = torch.zeros((1, target_shape[0], target_shape[1]), dtype=torch.float32)
            return zero_tensor, zero_mask, "excluded_missing_coords"
        elif self.geospatial_policy == "exclude_sample":
            raise SampleExclusionError(f"Observation excluded due to missing coordinates in {getattr(grp, 'name', 'unnamed')}")

        t_data, t_mask = self._resize_to_grid_with_mask(arr, target_shape)
        return t_data, t_mask, "nominal_fallback"

    def _resample_radar_vertical_profile_to_cappi(
        self,
        raw_ku_beam: np.ndarray,
        bin_height_km_beam: np.ndarray,
        target_heights_km: np.ndarray,
        echo_threshold_dbz: float = 12.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resamples a single ray/beam vertical reflectivity profile to 16 CAPPI height levels.
        raw_ku_beam: (n_range,) in dBZ
        bin_height_km_beam: (n_range,) in km
        target_heights_km: (16,) target levels in km
        Returns:
            norm_refl: (16,) normalized in [0, 1]
            meas_valid: (16,) 1.0 if measurement valid (-20 <= Ku <= 100), 0.0 otherwise
            echo_valid: (16,) 1.0 if echo valid (Ku >= 12.0), 0.0 otherwise
        """
        n_targets = len(target_heights_km)
        norm_refl = np.zeros(n_targets, dtype=np.float32)
        meas_valid = np.zeros(n_targets, dtype=np.float32)
        echo_valid = np.zeros(n_targets, dtype=np.float32)

        # Gate validity: non-NaN, non-sentinel, within valid measurement range [-20, 100] dBZ
        valid_gates = (
            (~np.isnan(bin_height_km_beam))
            & (~np.isnan(raw_ku_beam))
            & (raw_ku_beam > -9000.0)
            & (raw_ku_beam >= -20.0)
            & (raw_ku_beam <= 100.0)
        )
        if np.sum(valid_gates) < 2:
            return norm_refl, meas_valid, echo_valid

        h_valid = bin_height_km_beam[valid_gates]
        q_valid = raw_ku_beam[valid_gates]

        # Sort ascending by height
        sort_idx = np.argsort(h_valid)
        h_sorted = h_valid[sort_idx]
        q_sorted = q_valid[sort_idx]

        # Check for non-monotonic duplicate heights and filter if needed
        if len(h_sorted) > 1:
            diffs = np.diff(h_sorted)
            if np.any(diffs <= 0):
                unique_mask = np.concatenate([[True], diffs > 1e-4])
                h_sorted = h_sorted[unique_mask]
                q_sorted = q_sorted[unique_mask]
                if len(h_sorted) < 2:
                    return norm_refl, meas_valid, echo_valid

        # Interpolate strictly between observed bounds (no extrapolation!)
        interp_q = np.interp(target_heights_km, h_sorted, q_sorted, left=np.nan, right=np.nan)
        is_interp_valid = ~np.isnan(interp_q)

        meas_valid[is_interp_valid] = 1.0
        echo_valid[is_interp_valid & (interp_q >= echo_threshold_dbz)] = 1.0
        # Physical normalization: [-20, 100] dBZ -> [0, 1]
        norm_refl[is_interp_valid] = np.clip(
            (interp_q[is_interp_valid] - (-20.0)) / (100.0 - (-20.0)), 0.0, 1.0
        )
        return norm_refl, meas_valid, echo_valid

    def _load_radar_volume(
        self,
        radar_grp: nc.Group,
        target_grid: Tuple[int, int],
        target_depth: int = 16,
        storm_lat: Optional[float] = None,
        storm_lon: Optional[float] = None,
        domain_km: float = 500.0,
        max_dist_km: float = 15.0,
        echo_threshold_dbz: float = 12.0,
        raw_sample_id: Optional[str] = None,
        raw_data_dir: Optional[str] = None,
        return_source: bool = False,
        return_echo_mask: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, bool],
        Tuple[torch.Tensor, torch.Tensor, bool, str],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, str],
    ]:
        """
        Load 3D volumetric radar reflectivity (Ku band), interpolate vertical depth to
        16 explicitly defined physical height levels (0.5 to 19.5 km), and resample
        spatial dimensions onto the unified convolution feature grid (32x32).

        Separates measurement validity from precipitation echo classification:
          - Measurement validity: [-20.0, 100.0] dBZ (preserves valid low-reflectivity measurements)
          - Precipitation echo classification: Ku >= echo_threshold_dbz (default 12.0 dBZ MDS)
        """
        H, W = target_grid[0] // 2, target_grid[1] // 2
        empty_tensor = torch.zeros((1, target_depth, H, W), dtype=torch.float32)
        empty_mask = torch.zeros((1, target_depth, H, W), dtype=torch.float32)
        empty_echo = torch.zeros((1, target_depth, H, W), dtype=torch.float32)

        def _format_return(t, m, e, val, src):
            if return_echo_mask:
                return (t, m, e, val, src) if return_source else (t, m, e, val)
            return (t, m, val, src) if return_source else (t, m, val)

        if "correctedReflectFactor_Ku" not in radar_grp.variables:
            return _format_return(empty_tensor, empty_mask, empty_echo, False, "empty")

        raw_ku = np.ma.filled(radar_grp.variables["correctedReflectFactor_Ku"][:], np.nan)
        if raw_ku.ndim != 3:
            return _format_return(empty_tensor, empty_mask, empty_echo, False, "invalid_shape")

        # 1. Measurement Validity: [-20, 100] dBZ
        valid_meas = (~np.isnan(raw_ku)) & (raw_ku > -9000.0) & (raw_ku >= -20.0) & (raw_ku <= 100.0)
        if not np.any(valid_meas):
            return _format_return(empty_tensor, empty_mask, empty_echo, False, "no_valid_data")

        # 2. Precipitation Echo Classification: Ku >= echo_threshold_dbz (default 12.0 dBZ MDS)
        echo_class = valid_meas & (raw_ku >= echo_threshold_dbz)

        n_range, n_scan, n_beam = raw_ku.shape

        # Physical normalization: dBZ [-20, 80] -> [0, 1] with range 100 dBZ
        norm_ku = np.clip(np.where(valid_meas, raw_ku, -20.0) - (-20.0), 0.0, 100.0) / 100.0
        mask_ku = valid_meas.astype(np.float32)
        echo_ku = echo_class.astype(np.float32)

        x_2d, y_2d = None, None
        bin_height_km = None
        source = "nominal_fallback"

        # Check for physical radar gate heights in processed container
        for bh_name in ["bin_height_km", "binHeight", "bin_height"]:
            if bh_name in radar_grp.variables:
                bh_raw = np.ma.filled(radar_grp.variables[bh_name][:], np.nan)
                units = getattr(radar_grp.variables[bh_name], "units", "km")
                if "m" in units.lower() and "km" not in units.lower():
                    bh_raw = bh_raw / 1000.0  # convert meters to km
                bin_height_km = bh_raw
                break

        # Attempt coordinate extraction directly from radar group
        try:
            if "x" in radar_grp.variables and "y" in radar_grp.variables:
                x_raw = np.ma.filled(radar_grp.variables["x"][:], np.nan)
                y_raw = np.ma.filled(radar_grp.variables["y"][:], np.nan)
                if x_raw.shape == (n_scan, n_beam) and y_raw.shape == (n_scan, n_beam):
                    x_2d, y_2d = x_raw, y_raw
                    source = "radar_beam_geodetic_km"
            elif "latitude" in radar_grp.variables and "longitude" in radar_grp.variables and storm_lat is not None and storm_lon is not None:
                lat_raw = np.ma.filled(radar_grp.variables["latitude"][:], np.nan)
                lon_raw = np.ma.filled(radar_grp.variables["longitude"][:], np.nan)
                if lat_raw.shape == (n_scan, n_beam) and lon_raw.shape == (n_scan, n_beam):
                    cos_lat = max(1e-4, math.cos(math.radians(storm_lat)))
                    d_lon = ((lon_raw - storm_lon + 180.0) % 360.0) - 180.0
                    x_2d = d_lon * cos_lat * 111.32
                    y_2d = (lat_raw - storm_lat) * 110.57
                    source = "radar_beam_geodetic_km"
        except Exception:
            x_2d, y_2d = None, None
            source = "nominal_fallback"

        # If coordinates or binHeight missing from processed group, check raw NetCDF container if available
        if (x_2d is None or bin_height_km is None) and raw_sample_id is not None:
            rdir = raw_data_dir or getattr(self, "raw_data_dir", "data/raw/pilot")
            if rdir and os.path.exists(rdir):
                try:
                    cache = getattr(self, "_raw_radar_coord_cache", None)
                    if cache is not None and raw_sample_id in cache:
                        cached_entry = cache[raw_sample_id]
                        if len(cached_entry) == 3:
                            x_2d, y_2d, bin_height_km = cached_entry
                        else:
                            x_2d, y_2d = cached_entry
                        source = "radar_beam_geodetic_km"
                    else:
                        parts = raw_sample_id.replace("SMP_", "").split("_")
                        st_id = parts[0]
                        tm_str = parts[-1]
                        import glob
                        season = st_id[4:8] if len(st_id) >= 8 else ""
                        basin = st_id[:2] if len(st_id) >= 2 else ""
                        num = st_id[2:4] if len(st_id) >= 4 else ""
                        pattern = os.path.join(rdir, season, basin, num, f"*{tm_str}*.nc")
                        cand = glob.glob(pattern)
                        if not cand:
                            cand = glob.glob(os.path.join(rdir, "**", f"*{st_id}*{tm_str}*.nc"), recursive=True)
                        if cand:
                            with nc.Dataset(cand[0]) as ds_raw:
                                if "radar_radiometer" in ds_raw.groups:
                                    rr_raw = ds_raw.groups["radar_radiometer"]
                                    for sub_cand in ["KuKaGMI", "KuGMI", "KuTMI"]:
                                        if sub_cand in rr_raw.groups:
                                            sg_raw = rr_raw.groups[sub_cand]
                                            if x_2d is None and "x" in sg_raw.variables and "y" in sg_raw.variables:
                                                xr = np.ma.filled(sg_raw.variables["x"][:], np.nan)
                                                yr = np.ma.filled(sg_raw.variables["y"][:], np.nan)
                                                if xr.shape == (n_scan, n_beam) and yr.shape == (n_scan, n_beam):
                                                    x_2d, y_2d = xr, yr
                                                    source = "radar_beam_geodetic_km"
                                            if bin_height_km is None and "binHeight" in sg_raw.variables:
                                                bhr = np.ma.filled(sg_raw.variables["binHeight"][:], np.nan)
                                                units = getattr(sg_raw.variables["binHeight"], "units", "km")
                                                if "m" in units.lower() and "km" not in units.lower():
                                                    bhr = bhr / 1000.0
                                                bin_height_km = bhr
                                            if cache is not None and x_2d is not None:
                                                cache[raw_sample_id] = (x_2d, y_2d, bin_height_km)
                                            break
                except Exception:
                    pass

        target_heights_np = np.asarray(TARGET_RADAR_HEIGHTS_KM, dtype=np.float32)

        # 3. Coordinate-aware geospatial and vertical registration
        if x_2d is not None and y_2d is not None and HAS_SCIPY:
            coords = np.column_stack([x_2d.flatten(), y_2d.flatten()])
            valid_coords = (~np.isnan(coords[:, 0])) & (~np.isnan(coords[:, 1]))
            if np.any(valid_coords):
                valid_indices_original = np.where(valid_coords)[0]
                tree = cKDTree(coords[valid_coords])

                # Unified convolution feature grid target coordinates (32x32 feature receptive field centers)
                cache_key = (H, W, True)
                if cache_key not in self._query_pts_cache:
                    if H == 32 and W == 32:
                        grid_x, grid_y = GLOBAL_GRID_GEOMETRY.get_feature_mesh_32()
                    else:
                        gx, gy = GLOBAL_GRID_GEOMETRY.get_coords(H, W, is_feature_grid=True)
                        grid_x, grid_y = np.meshgrid(gx, gy)
                    self._query_pts_cache[cache_key] = np.column_stack([grid_x.flatten(), grid_y.flatten()])
                query_pts = self._query_pts_cache[cache_key]

                dists, sub_indices = tree.query(query_pts, distance_upper_bound=max_dist_km)
                is_valid_horiz = dists <= max_dist_km

                grid_data = np.zeros((target_depth, H * W), dtype=np.float32)
                grid_mask = np.zeros((target_depth, H * W), dtype=np.float32)
                grid_echo = np.zeros((target_depth, H * W), dtype=np.float32)

                if np.any(is_valid_horiz):
                    matched_sub = sub_indices[is_valid_horiz]
                    orig_flat_idx = valid_indices_original[matched_sub]
                    scan_idx, beam_idx = np.unravel_index(orig_flat_idx, (n_scan, n_beam))
                    valid_positions = np.where(is_valid_horiz)[0]

                    if bin_height_km is not None:
                        # Physical CAPPI vertical resampling along true gate heights
                        is_1d_bh = (bin_height_km.ndim == 1)
                        for col_idx, (s_i, b_i) in enumerate(zip(scan_idx, beam_idx)):
                            bh_col = bin_height_km if is_1d_bh else bin_height_km[:, s_i, b_i]
                            q_col = raw_ku[:, s_i, b_i]
                            d_col, m_col, e_col = self._resample_radar_vertical_profile_to_cappi(
                                q_col, bh_col, target_heights_np, echo_threshold_dbz=echo_threshold_dbz
                            )
                            h_pos = valid_positions[col_idx]
                            grid_data[:, h_pos] = d_col
                            grid_mask[:, h_pos] = m_col
                            grid_echo[:, h_pos] = e_col
                    else:
                        # Fallback vertical interpolation if binHeight is missing
                        if self.strict_mode or self.geospatial_policy == "raise":
                            raise ValueError(f"Strict scientific mode failure: missing radar binHeight in {getattr(radar_grp, 'name', 'unnamed')}")
                        elif self.geospatial_policy == "exclude_modality":
                            return _format_return(empty_tensor, empty_mask, empty_echo, False, "excluded_missing_height")
                        elif self.geospatial_policy == "exclude_sample":
                            raise SampleExclusionError(f"Observation excluded due to missing radar binHeight in {getattr(radar_grp, 'name', 'unnamed')}")

                        t_data = torch.from_numpy(norm_ku.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        t_mask = torch.from_numpy(mask_ku).unsqueeze(0).unsqueeze(0)
                        t_echo = torch.from_numpy(echo_ku).unsqueeze(0).unsqueeze(0)
                        t_data_v = F.interpolate(t_data, size=(target_depth, n_scan, n_beam), mode="trilinear", align_corners=False).squeeze(0).squeeze(0)
                        t_mask_v = F.interpolate(t_mask, size=(target_depth, n_scan, n_beam), mode="nearest").squeeze(0).squeeze(0)
                        t_echo_v = F.interpolate(t_echo, size=(target_depth, n_scan, n_beam), mode="nearest").squeeze(0).squeeze(0)

                        grid_data[:, is_valid_horiz] = t_data_v[:, scan_idx, beam_idx].numpy()
                        grid_mask[:, is_valid_horiz] = t_mask_v[:, scan_idx, beam_idx].numpy()
                        grid_echo[:, is_valid_horiz] = t_echo_v[:, scan_idx, beam_idx].numpy()

                data_clean = torch.from_numpy(grid_data * grid_mask).view(1, target_depth, H, W)
                mask_clean = torch.from_numpy(grid_mask).view(1, target_depth, H, W)
                echo_clean = torch.from_numpy(grid_echo).view(1, target_depth, H, W)
                is_valid = bool(mask_clean.sum() > 0)
                return _format_return(data_clean, mask_clean, echo_clean, is_valid, source)

        # 4. Geospatial policy enforcement on missing radar coordinates
        if self.strict_mode or self.geospatial_policy == "raise":
            raise ValueError(f"Strict scientific mode failure: missing radar beam coordinate metadata in group {getattr(radar_grp, 'name', 'unnamed')}")
        elif self.geospatial_policy == "exclude_modality":
            return _format_return(empty_tensor, empty_mask, empty_echo, False, "excluded_missing_coords")
        elif self.geospatial_policy == "exclude_sample":
            raise SampleExclusionError(f"Observation excluded due to missing radar beam coordinates in {getattr(radar_grp, 'name', 'unnamed')}")

        # Nominal fallback mode
        t_data = torch.from_numpy(norm_ku.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t_mask = torch.from_numpy(mask_ku).unsqueeze(0).unsqueeze(0)
        t_echo = torch.from_numpy(echo_ku).unsqueeze(0).unsqueeze(0)

        data_resized = F.interpolate(
            t_data, size=(target_depth, H, W), mode="trilinear", align_corners=False
        ).squeeze(0)
        mask_resized = F.interpolate(t_mask, size=(target_depth, H, W), mode="nearest").squeeze(0)
        echo_resized = F.interpolate(t_echo, size=(target_depth, H, W), mode="nearest").squeeze(0)

        data_clean = data_resized * mask_resized
        is_valid = bool(mask_resized.sum() > 0)
        return _format_return(data_clean, mask_resized, echo_resized, is_valid, "nominal_fallback")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        nc_path = str(row["processed_container_path"])
        group_path = str(row["container_group_path"])
        sample_id = str(row["sample_id"])
        storm_id = str(row["storm_id"])
        instrument = str(row.get("instrument", "")).strip().upper()

        if not os.path.exists(nc_path):
            alt_path = nc_path.replace("\\", "/")
            if os.path.exists(alt_path):
                nc_path = alt_path

        H, W = self.grid_size, self.grid_size

        # Tensor initializations
        ir_tensor = torch.zeros((1, H, W), dtype=torch.float32)
        pmw_tensor = torch.zeros((4, H, W), dtype=torch.float32)
        gprof_tensor = torch.zeros((4, H, W), dtype=torch.float32)
        radar_tensor = torch.zeros((1, 16, H // 2, W // 2), dtype=torch.float32)
        env_tensor = torch.zeros((6, H, W), dtype=torch.float32)

        # Pixel validity masks
        ir_pixel_mask = torch.zeros((1, H, W), dtype=torch.float32)
        pmw_pixel_mask = torch.zeros((4, H, W), dtype=torch.float32)
        gprof_pixel_mask = torch.zeros((4, H, W), dtype=torch.float32)
        radar_voxel_mask = torch.zeros((1, 16, H // 2, W // 2), dtype=torch.float32)
        radar_echo_mask = torch.zeros((1, 16, H // 2, W // 2), dtype=torch.float32)
        env_pixel_mask = torch.zeros((6, H, W), dtype=torch.float32)

        # Channel-level validity masks
        pmw_channel_mask = torch.zeros(4, dtype=torch.float32)
        gprof_channel_mask = torch.zeros(4, dtype=torch.float32)
        env_channel_mask = torch.zeros(6, dtype=torch.float32)

        has_ir = bool(row.get("has_ir", False))
        has_pmw = bool(row.get("has_pmw", False))
        has_gprof = bool(row.get("has_gprof", False))
        has_radar = bool(row.get("has_radar", False))
        has_env = bool(row.get("has_causal_env", False))

        # Geospatial provenance tracking
        ir_src = "none"
        pmw_srcs = []
        gp_srcs = []
        rad_src = "none"
        env_srcs = []

        # Targets and strictly causal issue-time velocity from container if available
        storm_lat_from_nc = None
        storm_lon_from_nc = None
        causal_u_kmh = 0.0
        causal_v_kmh = 0.0
        causal_has_kin = 0.0
        t_obs_epoch = int(row["timestamp_epoch"])

        use_mem_cache = getattr(self, "enable_memory_cache", True) and (getattr(self, "transform", None) is None)
        sample_cache = getattr(self, "_sample_cache", None)
        cache_key = None
        disk_cache_path = None
        if use_mem_cache and sample_cache is not None:
            thresh_tuple = tuple(sorted(getattr(self, "resampling_thresholds", {}).items()))
            nc_mtime_ns = 0
            nc_size_bytes = 0
            if os.path.exists(nc_path):
                try:
                    st_nc = os.stat(nc_path)
                    nc_mtime_ns = int(st_nc.st_mtime_ns)
                    nc_size_bytes = int(st_nc.st_size)
                except OSError:
                    pass
            part_label = str(row.get("partition", self.partition or "ALL")).strip().upper()
            cache_key = (
                self.CACHE_PREPROCESSING_VERSION,
                part_label,
                storm_id,
                nc_path,
                nc_mtime_ns,
                nc_size_bytes,
                group_path,
                sample_id,
                t_obs_epoch,
                instrument,
                H,
                W,
                TARGET_RADAR_HEIGHTS_KM,
                self.geospatial_policy,
                self.strict_mode,
                getattr(self, "raw_data_dir", None),
                has_ir,
                has_pmw,
                has_gprof,
                has_radar,
                has_env,
                thresh_tuple,
            )
            d_dir = getattr(self, "disk_cache_dir", None)
            if (
                getattr(self, "enable_disk_cache", True)
                and d_dir
                and "data/processed" in nc_path.replace("\\", "/")
            ):
                key_hash = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:16]
                safe_sid = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in sample_id)
                disk_cache_path = os.path.join(d_dir, f"{part_label}_{safe_sid}_{key_hash}.pt")

        loaded_from_cache = False
        if cache_key is not None and cache_key in sample_cache:
            if hasattr(sample_cache, "move_to_end"):
                sample_cache.move_to_end(cache_key)
            cached_payload = sample_cache[cache_key]
            loaded_from_cache = True
        elif disk_cache_path is not None and os.path.exists(disk_cache_path):
            try:
                cached_payload = torch.load(disk_cache_path, map_location="cpu", weights_only=False)
                if cache_key is not None and sample_cache is not None and len(cached_payload) == 29:
                    max_entries = max(1, int(getattr(self, "max_cache_entries", 2048)))
                    while len(sample_cache) >= max_entries:
                        if hasattr(sample_cache, "popitem"):
                            try:
                                sample_cache.popitem(last=False)
                            except TypeError:
                                sample_cache.pop(next(iter(sample_cache)))
                        else:
                            sample_cache.pop(next(iter(sample_cache)))
                    sample_cache[cache_key] = cached_payload
                    loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if loaded_from_cache:
            (
                ir_tensor,
                pmw_tensor,
                gprof_tensor,
                radar_tensor,
                env_tensor,
                ir_pixel_mask,
                pmw_pixel_mask,
                gprof_pixel_mask,
                radar_voxel_mask,
                radar_echo_mask,
                env_pixel_mask,
                pmw_channel_mask,
                gprof_channel_mask,
                env_channel_mask,
                has_ir,
                has_pmw,
                has_gprof,
                has_radar,
                has_env,
                ir_src,
                pmw_srcs,
                gp_srcs,
                rad_src,
                env_srcs,
                storm_lat_from_nc,
                storm_lon_from_nc,
                causal_u_kmh,
                causal_v_kmh,
                causal_has_kin,
            ) = cached_payload
        elif os.path.exists(nc_path):
            err_count_before = len(self.processing_errors)
            grp_cache = getattr(self, "_grp_resample_cache", None)
            if grp_cache is not None:
                grp_cache.clear()
            try:
                with nc.Dataset(nc_path, "r") as ds:
                    parts = [p for p in group_path.split("/") if p]
                    curr_grp = ds
                    for p in parts:
                        if p in curr_grp.groups:
                            curr_grp = curr_grp.groups[p]
                        else:
                            curr_grp = None
                            break

                    if curr_grp is not None:
                        # Extract storm latitude/longitude if present in targets
                        if "targets" in curr_grp.groups:
                            tgt_grp = curr_grp.groups["targets"]
                            if "storm_latitude" in tgt_grp.variables:
                                storm_lat_from_nc = float(tgt_grp.variables["storm_latitude"][:])
                            if "storm_longitude" in tgt_grp.variables:
                                storm_lon_from_nc = float(tgt_grp.variables["storm_longitude"][:])

                        storm_lat = storm_lat_from_nc if storm_lat_from_nc is not None else (float(row["latitude"]) if "latitude" in row and not pd.isna(row["latitude"]) else None)
                        storm_lon = storm_lon_from_nc if storm_lon_from_nc is not None else (float(row["longitude"]) if "longitude" in row and not pd.isna(row["longitude"]) else None)

                        # Strictly causal backward-difference storm velocity from fixes at time <= t_obs_epoch
                        if (
                            storm_lat is not None
                            and storm_lon is not None
                            and not math.isnan(float(storm_lat))
                            and not math.isnan(float(storm_lon))
                            and "storm_metadata" in ds.groups
                        ):
                            sm_grp = ds.groups["storm_metadata"]
                            if (
                                "time" in sm_grp.variables
                                and "storm_latitude" in sm_grp.variables
                                and "storm_longitude" in sm_grp.variables
                            ):
                                bt_times = np.asarray(sm_grp.variables["time"][:], dtype=np.int64)
                                bt_lats = np.asarray(sm_grp.variables["storm_latitude"][:], dtype=np.float64)
                                bt_lons = np.asarray(sm_grp.variables["storm_longitude"][:], dtype=np.float64)
                                # Enforce strict causal boundary: NEVER inspect any timestamp > t_obs_epoch
                                causal_idx = np.where(
                                    (bt_times <= t_obs_epoch)
                                    & (np.abs(bt_lats) <= 90.0)
                                    & (np.abs(bt_lons) <= 360.0)
                                )[0]
                                best_k = None
                                for k_idx in reversed(causal_idx):
                                    dt_h = (t_obs_epoch - int(bt_times[k_idx])) / 3600.0
                                    if 2.0 <= dt_h <= 24.0:
                                        best_k = k_idx
                                        break
                                if best_k is not None:
                                    dt_h = (t_obs_epoch - int(bt_times[best_k])) / 3600.0
                                    dx_c, dy_c = compute_tangent_plane_displacement_km(
                                        float(bt_lats[best_k]), float(bt_lons[best_k]),
                                        float(storm_lat), float(storm_lon),
                                    )
                                    causal_u_kmh = float(dx_c / dt_h)
                                    causal_v_kmh = float(dy_c / dt_h)
                                    causal_has_kin = 1.0
                                elif len(causal_idx) >= 2:
                                    k1, k0 = causal_idx[-1], causal_idx[-2]
                                    dt_h = (int(bt_times[k1]) - int(bt_times[k0])) / 3600.0
                                    if 1.0 <= dt_h <= 24.0:
                                        dx_c, dy_c = compute_tangent_plane_displacement_km(
                                            float(bt_lats[k0]), float(bt_lons[k0]),
                                            float(bt_lats[k1]), float(bt_lons[k1]),
                                        )
                                        causal_u_kmh = float(dx_c / dt_h)
                                        causal_v_kmh = float(dy_c / dt_h)
                                        causal_has_kin = 1.0

                        # 1. Infrared (IRWIN)
                        if has_ir and "infrared" in curr_grp.groups:
                            try:
                                ir_grp = curr_grp.groups["infrared"]
                                if "IRWIN" in ir_grp.variables:
                                    raw_ir = np.ma.filled(ir_grp.variables["IRWIN"][:], np.nan)
                                    # Raw valid range check: [100, 360] K before arithmetic
                                    valid_ir = (~np.isnan(raw_ir)) & (raw_ir > -9000.0) & (raw_ir > 100.0) & (raw_ir < 360.0)
                                    norm_ir = np.where(valid_ir, np.clip((raw_ir - 170.0) / 150.0, 0.0, 1.0), np.nan)
                                    ir_dist_thresh = self.resampling_thresholds.get("ir", 15.0)
                                    t_ir, m_ir, ir_src = self._resample_swath_with_coords(
                                        norm_ir, ir_grp, storm_lat, storm_lon, (H, W), max_dist_km=ir_dist_thresh
                                    )
                                    if m_ir.sum() > 0:
                                        ir_tensor = t_ir
                                        ir_pixel_mask = m_ir
                                    else:
                                        has_ir = False
                            except Exception as e_ir:
                                if self.strict_mode or self.geospatial_policy == "raise":
                                    raise
                                has_ir = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "infrared", "IRWIN", str(e_ir))

                        # 2. Passive Microwave (Deterministic Sensor-Aware Channel Selection)
                        if has_pmw and "passive_microwave" in curr_grp.groups:
                            try:
                                pmw_grp = curr_grp.groups["passive_microwave"]
                                ch_map = SENSOR_PMW_CHANNEL_MAP.get(instrument, {})
                                pmw_dist_thresh = self.resampling_thresholds.get("pmw", 35.0)

                                for ch_idx in range(4):
                                    mapping = ch_map.get(ch_idx)
                                    if mapping is not None:
                                        sw_name, var_name = mapping
                                        if sw_name in pmw_grp.groups and var_name in pmw_grp.groups[sw_name].variables:
                                            sw_grp = pmw_grp.groups[sw_name]
                                            raw_tb = np.ma.filled(sw_grp.variables[var_name][:], np.nan)
                                            # Raw valid range check: [50, 360] K before arithmetic
                                            valid_tb = (~np.isnan(raw_tb)) & (raw_tb > -9000.0) & (raw_tb > 50.0) & (raw_tb < 360.0)
                                            norm_tb = np.where(valid_tb, np.clip((raw_tb - 100.0) / 220.0, 0.0, 1.0), np.nan)
                                            t_ch, m_ch, ch_src = self._resample_swath_with_coords(
                                                norm_tb, sw_grp, storm_lat, storm_lon, (H, W), max_dist_km=pmw_dist_thresh
                                            )
                                            if m_ch.sum() > 0:
                                                pmw_tensor[ch_idx] = t_ch.squeeze(0)
                                                pmw_pixel_mask[ch_idx] = m_ch.squeeze(0)
                                                pmw_channel_mask[ch_idx] = 1.0
                                                pmw_srcs.append(ch_src)

                                if pmw_channel_mask.sum() == 0:
                                    has_pmw = False
                            except Exception as e_pmw:
                                if self.strict_mode or self.geospatial_policy == "raise":
                                    raise
                                has_pmw = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "passive_microwave", "TB", str(e_pmw))

                        # 3. GPROF Multi-Field Precipitation
                        if has_gprof and "GPROF" in curr_grp.groups:
                            try:
                                gp_grp = curr_grp.groups["GPROF"]
                                gp_dist_thresh = self.resampling_thresholds.get("gprof", 20.0)
                                gprof_fields = [
                                    ("surfacePrecipitation", 0, 5.0),
                                    ("convectivePrecipitation", 1, 5.0),
                                    ("rainWaterPath", 2, 5.0),
                                    ("iceWaterPath", 3, 5.0),
                                ]
                                for fname, fidx, norm_scale in gprof_fields:
                                    if fname in gp_grp.variables:
                                        raw_f = np.ma.filled(gp_grp.variables[fname][:], np.nan)
                                        # Detect raw invalid sentinels BEFORE any arithmetic or log1p
                                        valid_f = (~np.isnan(raw_f)) & (raw_f > -9000.0) & (raw_f >= 0.0)
                                        norm_f = np.where(valid_f, np.log1p(np.maximum(raw_f, 0.0)) / norm_scale, np.nan)
                                        t_f, m_f, f_src = self._resample_swath_with_coords(
                                            norm_f, gp_grp, storm_lat, storm_lon, (H, W), max_dist_km=gp_dist_thresh
                                        )
                                        if m_f.sum() > 0:
                                            gprof_tensor[fidx] = t_f.squeeze(0)
                                            gprof_pixel_mask[fidx] = m_f.squeeze(0)
                                            gprof_channel_mask[fidx] = 1.0
                                            gp_srcs.append(f_src)

                                if gprof_channel_mask.sum() == 0:
                                    has_gprof = False
                            except Exception as e_gp:
                                if self.strict_mode or self.geospatial_policy == "raise":
                                    raise
                                has_gprof = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "GPROF", "precipitation", str(e_gp))

                        # 4. Radar Volumetric Profile
                        if has_radar and "radar_radiometer" in curr_grp.groups:
                            try:
                                rad_grp = curr_grp.groups["radar_radiometer"]
                                rad_dist_thresh = self.resampling_thresholds.get("radar", 15.0)
                                r_tensor, r_mask, r_echo, r_valid, rad_src = self._load_radar_volume(
                                    rad_grp, (H, W), target_depth=16,
                                    storm_lat=storm_lat, storm_lon=storm_lon,
                                    max_dist_km=rad_dist_thresh,
                                    raw_sample_id=sample_id,
                                    raw_data_dir=getattr(self, "raw_data_dir", "data/raw/pilot"),
                                    echo_threshold_dbz=12.0,
                                    return_source=True,
                                    return_echo_mask=True,
                                )
                                if r_valid:
                                    radar_tensor = r_tensor
                                    radar_voxel_mask = r_mask
                                    radar_echo_mask = r_echo
                                    has_radar = True
                                else:
                                    has_radar = False
                            except Exception as e_rad:
                                if self.strict_mode or self.geospatial_policy == "raise":
                                    raise
                                has_radar = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "radar_radiometer", "Ku", str(e_rad))
                        else:
                            has_radar = False

                        # 5. Environmental Synoptic Reanalysis (6 Channels)
                        if has_env and "environmental_synoptic" in curr_grp.groups:
                            try:
                                env_grp = curr_grp.groups["environmental_synoptic"]
                                env_dist_thresh = self.resampling_thresholds.get("env", 40.0)
                                env_fields = [
                                    ("precipitable_water", 0, 80.0, 0.0),
                                    ("sst", 1, 35.0, 273.15),
                                    ("pressure_msl", 2, 50.0, 1013.25),
                                    ("temperature_2m", 3, 40.0, 273.15),
                                    ("u_10m", 4, 30.0, 0.0),
                                    ("v_10m", 5, 30.0, 0.0),
                                ]
                                for fname, fidx, fscale, foffset in env_fields:
                                    if fname in env_grp.variables:
                                        raw_e = np.ma.filled(env_grp.variables[fname][:], np.nan)
                                        # Raw invalid detection: _FillValue = -32768 or <= -9000
                                        valid_e = (~np.isnan(raw_e)) & (raw_e > -9000.0) & (raw_e != -32768)
                                        norm_e = np.where(valid_e, (raw_e - foffset) / fscale, np.nan)
                                        t_e, m_e, e_src = self._resample_swath_with_coords(
                                            norm_e, env_grp, storm_lat, storm_lon, (H, W), max_dist_km=env_dist_thresh
                                        )
                                        if m_e.sum() > 0:
                                            env_tensor[fidx] = t_e.squeeze(0)
                                            env_pixel_mask[fidx] = m_e.squeeze(0)
                                            env_channel_mask[fidx] = 1.0
                                            env_srcs.append(e_src)

                                if env_channel_mask.sum() == 0:
                                    has_env = False
                            except Exception as e_env:
                                if self.strict_mode or self.geospatial_policy == "raise":
                                    raise
                                has_env = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "environmental_synoptic", "env", str(e_env))

            except Exception as e_top:
                self._record_error(sample_id, storm_id, nc_path, group_path, "container", "open", str(e_top))
                if self.strict_mode or self.geospatial_policy == "raise":
                    raise RuntimeError(f"Strict mode failure loading container {nc_path}: {e_top}")
            finally:
                if grp_cache is not None:
                    grp_cache.clear()

            if cache_key is not None and len(self.processing_errors) == err_count_before:
                payload_to_cache = (
                    ir_tensor,
                    pmw_tensor,
                    gprof_tensor,
                    radar_tensor,
                    env_tensor,
                    ir_pixel_mask,
                    pmw_pixel_mask,
                    gprof_pixel_mask,
                    radar_voxel_mask,
                    radar_echo_mask,
                    env_pixel_mask,
                    pmw_channel_mask,
                    gprof_channel_mask,
                    env_channel_mask,
                    has_ir,
                    has_pmw,
                    has_gprof,
                    has_radar,
                    has_env,
                    ir_src,
                    list(pmw_srcs),
                    list(gp_srcs),
                    rad_src,
                    list(env_srcs),
                    storm_lat_from_nc,
                    storm_lon_from_nc,
                    causal_u_kmh,
                    causal_v_kmh,
                    causal_has_kin,
                )
                max_entries = max(1, int(getattr(self, "max_cache_entries", 2048)))
                while len(sample_cache) >= max_entries:
                    if hasattr(sample_cache, "popitem"):
                        try:
                            sample_cache.popitem(last=False)
                        except TypeError:
                            sample_cache.pop(next(iter(sample_cache)))
                    else:
                        sample_cache.pop(next(iter(sample_cache)))
                sample_cache[cache_key] = payload_to_cache
                if disk_cache_path is not None and not os.path.exists(disk_cache_path):
                    tmp_p = f"{disk_cache_path}.{os.getpid()}.{threading.get_ident()}.tmp"
                    try:
                        os.makedirs(os.path.dirname(disk_cache_path), exist_ok=True)
                        torch.save(payload_to_cache, tmp_p)
                        os.replace(tmp_p, disk_cache_path)
                    except Exception:
                        try:
                            if os.path.exists(tmp_p):
                                os.remove(tmp_p)
                        except OSError:
                            pass
        else:
            self._record_error(sample_id, storm_id, nc_path, group_path, "filesystem", "missing", "Container path does not exist")
            if self.strict_mode or self.geospatial_policy == "raise":
                raise FileNotFoundError(f"Container not found: {nc_path}")

        # Current storm coordinates
        cur_lat = row.get("storm_latitude")
        if pd.isna(cur_lat) or cur_lat is None:
            cur_lat = storm_lat_from_nc
        cur_lon = row.get("storm_longitude")
        if pd.isna(cur_lon) or cur_lon is None:
            cur_lon = storm_lon_from_nc

        has_valid_cur_pos = (cur_lat is not None) and (cur_lon is not None) and (not math.isnan(float(cur_lat))) and (not math.isnan(float(cur_lon)))

        # Strictly causal observation-time kinematic & geophysical storm state vector (10 features)
        # Uses ONLY backward-difference positions at t_prev <= t_obs (never IBTrACS centered-difference storm_speed_mps)
        raw_dland = row.get("distance_to_land_km")
        if causal_has_kin > 0.5:
            u_kmh = float(causal_u_kmh)
            v_kmh = float(causal_v_kmh)
            spd_kmh = math.hypot(u_kmh, v_kmh)
            spd_val = spd_kmh / 3.6
            if spd_kmh > 1e-4:
                sin_hdg = u_kmh / spd_kmh
                cos_hdg = v_kmh / spd_kmh
            else:
                sin_hdg, cos_hdg = 0.0, 1.0
            has_kin = 1.0
        else:
            spd_val, sin_hdg, cos_hdg, u_kmh, v_kmh, has_kin = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if has_valid_cur_pos:
            lat_rad = math.radians(float(cur_lat))
            sin_lat = math.sin(lat_rad)
            cos_lat = math.cos(lat_rad)
            hemi = 1.0 if float(cur_lat) >= 0.0 else -1.0
        else:
            sin_lat, cos_lat, hemi = 0.0, 1.0, 1.0

        if raw_dland is not None and not pd.isna(raw_dland) and float(raw_dland) >= 0.0:
            dland_norm = math.log10(1.0 + float(raw_dland)) / 3.5
        else:
            dland_norm = 0.8

        kinematic_state = torch.tensor(
            [u_kmh, v_kmh, spd_val, sin_hdg, cos_hdg, sin_lat, cos_lat, hemi, dland_norm, has_kin],
            dtype=torch.float32,
        )

        # Target 1: Current Intensity (v_max kt)
        raw_v = row.get("intensity_vmax_kt")
        if pd.isna(raw_v):
            intensity = 0.0
            intensity_mask = 0.0
        else:
            intensity = float(raw_v)
            intensity_mask = 1.0

        # Target 2: Central Minimum Pressure (p_min hPa)
        raw_p = row.get("central_min_pressure_hpa")
        if pd.isna(raw_p):
            pressure = 1010.0
            pressure_mask = 0.0
        else:
            pressure = float(raw_p)
            pressure_mask = 1.0

        # Target 3: Development Stage Classification (Canonical 5 classes)
        raw_stage = str(row.get("development_stage_raw", "UN")).strip().upper()
        if raw_stage in STAGE_MAPPING_CANONICAL:
            stage_id = STAGE_MAPPING_CANONICAL[raw_stage]
            stage_mask = 1.0
        else:
            stage_id = 0
            stage_mask = 0.0

        # Target 4: Multi-Horizon Intensity Change (+6h, +12h, +18h, +24h kt)
        int_changes = []
        int_change_masks = []
        for h in [6, 12, 18, 24]:
            val = row.get(f"intensity_change_{h}h_kt")
            if pd.isna(val):
                int_changes.append(0.0)
                int_change_masks.append(0.0)
            else:
                int_changes.append(float(val))
                int_change_masks.append(1.0)

        # Target 5: Rapid Intensification (>= 30 kt / 24h)
        ri_val = row.get("is_rapid_intensification_24h")
        if pd.isna(ri_val):
            ri_target = 0.0
            ri_mask = 0.0
        else:
            ri_target = 1.0 if float(ri_val) > 0.5 else 0.0
            ri_mask = 1.0

        # Target 6: Future Track Displacement (East/North km for +6h, +12h, +18h, +24h)
        track_displacements = []
        track_masks = []
        for h in [6, 12, 18, 24]:
            fut_lat = row.get(f"future_lat_{h}h")
            fut_lon = row.get(f"future_lon_{h}h")
            if has_valid_cur_pos and (not pd.isna(fut_lat)) and (not pd.isna(fut_lon)):
                dx, dy = compute_tangent_plane_displacement_km(
                    float(cur_lat), float(cur_lon), float(fut_lat), float(fut_lon)
                )
                track_displacements.append([dx, dy])
                track_masks.append(1.0)
            else:
                track_displacements.append([0.0, 0.0])
                track_masks.append(0.0)

        # Modality presence tensor: [has_ir, has_pmw, has_gprof, has_radar, has_env]
        modality_masks = torch.tensor(
            [
                1.0 if has_ir else 0.0,
                1.0 if has_pmw else 0.0,
                1.0 if has_gprof else 0.0,
                1.0 if has_radar else 0.0,
                1.0 if has_env else 0.0,
            ],
            dtype=torch.float32,
        )

        sample = {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "timestamp_epoch": int(row["timestamp_epoch"]),
            "timestamp_utc": str(row.get("timestamp_utc", "")),
            "instrument": instrument,
            "storm_latitude": float(cur_lat) if has_valid_cur_pos else float("nan"),
            "storm_longitude": float(cur_lon) if has_valid_cur_pos else float("nan"),
            "has_valid_cur_pos": has_valid_cur_pos,
            "modality_masks": modality_masks,
            "ir": ir_tensor,
            "pmw": pmw_tensor,
            "gprof": gprof_tensor,
            "radar": radar_tensor,
            "env": env_tensor,
            "kinematic_state": kinematic_state,
            "masks": {
                "ir_pixel": ir_pixel_mask,
                "pmw_pixel": pmw_pixel_mask,
                "gprof_pixel": gprof_pixel_mask,
                "radar_voxel": radar_voxel_mask,
                "radar_echo": radar_echo_mask,
                "env_pixel": env_pixel_mask,
                "pmw_channels": pmw_channel_mask,
                "gprof_channels": gprof_channel_mask,
                "env_channels": env_channel_mask,
            },
            "targets": {
                "intensity": torch.tensor(intensity, dtype=torch.float32),
                "intensity_mask": torch.tensor(intensity_mask, dtype=torch.float32),
                "pressure": torch.tensor(pressure, dtype=torch.float32),
                "pressure_mask": torch.tensor(pressure_mask, dtype=torch.float32),
                "stage": torch.tensor(stage_id, dtype=torch.long),
                "stage_mask": torch.tensor(stage_mask, dtype=torch.float32),
                "raw_stage": raw_stage,
                "intensity_change": torch.tensor(int_changes, dtype=torch.float32), # (4,)
                "intensity_change_mask": torch.tensor(int_change_masks, dtype=torch.float32), # (4,)
                "rapid_intensification": torch.tensor(ri_target, dtype=torch.float32),
                "ri_mask": torch.tensor(ri_mask, dtype=torch.float32),
                "track_displacement": torch.tensor(track_displacements, dtype=torch.float32), # (4, 2)
                "track_mask": torch.tensor(track_masks, dtype=torch.float32), # (4,)
            },
            "geospatial_provenance": {
                "infrared": ir_src if has_ir else "none",
                "passive_microwave": pmw_srcs[0] if has_pmw and pmw_srcs else "none",
                "gprof": gp_srcs[0] if has_gprof and gp_srcs else "none",
                "radar": rad_src if has_radar else "none",
                "environmental": env_srcs[0] if has_env and env_srcs else "none",
            },
        }
        return sample

    def _record_error(self, sample_id, storm_id, c_path, g_path, modality, var_name, err_msg):
        err_entry = {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "container_path": c_path,
            "group_path": g_path,
            "modality": modality,
            "failed_variable": var_name,
            "error": err_msg,
        }
        self.processing_errors.append(err_entry)
        if len(self.processing_errors) <= 10:
            logger.warning(f"[DATASET ERROR] {err_entry}")


class TCPRIMEDSequenceDataset(Dataset):
    """
    Storm-wise Historical Sequence Dataset for TC-GraphFormer.
    Constructs causal observation histories (t_0, ..., t_k) ending at anchor observation t_k.
    Enforces:
      - Chronological ordering by actual observation timestamps.
      - Never includes observations after anchor time t_k.
      - Strictly isolated within storm and train/val/test partition boundaries.
      - Irregular time gap preservation.
      - Anchor observation owns the multi-task future forecasting targets.
    """

    def __init__(
        self,
        sample_index_csv: str = "reports/pipeline/sample_index.csv",
        partition: Optional[str] = "TRAIN",
        grid_size: int = 64,
        max_history_len: int = 4,
        max_history_hours: float = 72.0,
        min_seq_len: int = 1,
        max_sequences: Optional[int] = None,
        strict_mode: bool = False,
        basin: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.base_dataset = TCPRIMEDDataset(
            sample_index_csv=sample_index_csv,
            partition=partition,
            grid_size=grid_size,
            strict_mode=strict_mode,
            basin=basin,
            df=df,
        )
        self.max_history_len = max_history_len
        self.max_history_hours = max_history_hours
        self.min_seq_len = min_seq_len

        # Group indices by storm_id, sorted chronologically
        df = self.base_dataset.df
        storm_groups = df.groupby("storm_id")

        self.sequence_index: List[List[int]] = []

        for storm_id, grp in storm_groups:
            # Sort chronologically by timestamp_epoch
            sorted_grp = grp.sort_values("timestamp_epoch")
            obs_indices = sorted_grp.index.tolist()
            obs_epochs = sorted_grp["timestamp_epoch"].tolist()

            for i in range(len(obs_indices)):
                anchor_epoch = obs_epochs[i]
                # Look backwards for valid history within max_history_hours
                seq = []
                for j in range(i, -1, -1):
                    dt_hours = (anchor_epoch - obs_epochs[j]) / 3600.0
                    if dt_hours <= max_history_hours and len(seq) < max_history_len:
                        seq.append(obs_indices[j])
                    else:
                        break

                # Reverse so sequence is strictly chronological [oldest ... anchor]
                seq.reverse()
                if len(seq) >= min_seq_len:
                    self.sequence_index.append(seq)

        if max_sequences is not None and max_sequences > 0:
            self.sequence_index = self.sequence_index[:max_sequences]

        print(
            f"[SEQ DATASET] Created {len(self.sequence_index)} sequence samples "
            f"(max_history_len={max_history_len}, max_window={max_history_hours}h) "
            f"for partition: {partition or 'ALL'}"
        )

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        indices = self.sequence_index[idx]
        # Return list of observation dicts in chronological order
        return [self.base_dataset[i] for i in indices]
