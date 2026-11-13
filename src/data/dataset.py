"""
PyTorch Dataset for NOAA TC PRIMED v01r01-final.
Provides multi-modal spatiotemporal overpass tensors with modality masks,
channel validity masks, and multi-task forecasting targets from standardized NetCDF-4/HDF5 containers.
Supports both single-overpass and historical storm sequence workflows.
"""

import os
import math
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        sample_index_csv: str = "reports/pipeline/sample_index.csv",
        partition: Optional[str] = "TRAIN",
        grid_size: int = 64,
        max_samples: Optional[int] = None,
        strict_mode: bool = False,
        transform=None,
    ):
        super().__init__()
        self.sample_index_csv = sample_index_csv
        self.partition = partition.upper() if partition else None
        self.grid_size = grid_size
        self.strict_mode = strict_mode
        self.transform = transform
        self.processing_errors: List[Dict[str, Any]] = []

        if not os.path.exists(sample_index_csv):
            raise FileNotFoundError(f"Sample index not found at: {sample_index_csv}")

        df = pd.read_csv(sample_index_csv)
        if self.partition:
            df = df[df["partition"] == self.partition].copy()

        if max_samples is not None and max_samples > 0:
            df = df.iloc[:max_samples].copy()

        self.df = df.reset_index(drop=True)
        print(f"[DATASET] Loaded {len(self.df)} samples for partition: {self.partition or 'ALL'}")

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

        t_data = torch.from_numpy(arr_clean).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        t_mask = torch.from_numpy(mask_clean).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

        data_resized = F.interpolate(t_data, size=target_shape, mode="bilinear", align_corners=False).squeeze(0) # (1, H, W)
        mask_resized = F.interpolate(t_mask, size=target_shape, mode="nearest").squeeze(0) # (1, H, W)

        # Zero out invalid pixels so imputed values never pollute valid tensor values
        data_clean = data_resized * mask_resized
        return data_clean, mask_resized

    def _load_radar_volume(
        self, radar_grp: nc.Group, target_grid: Tuple[int, int], target_depth: int = 16
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Load 3D volumetric radar reflectivity (Ku band), interpolate depth to target_depth,
        and spatial dimensions to (target_grid[0]//2, target_grid[1]//2).
        Returns: (radar_tensor, radar_mask, is_valid)
        """
        H, W = target_grid[0] // 2, target_grid[1] // 2
        empty_tensor = torch.zeros((1, target_depth, H, W), dtype=torch.float32)
        empty_mask = torch.zeros((1, target_depth, H, W), dtype=torch.float32)

        if "correctedReflectFactor_Ku" not in radar_grp.variables:
            return empty_tensor, empty_mask, False

        raw_ku = np.ma.filled(radar_grp.variables["correctedReflectFactor_Ku"][:], np.nan)
        # Expect shape: (range_gates, scans, beams) e.g. (88, 373, 49)
        if raw_ku.ndim != 3:
            return empty_tensor, empty_mask, False

        valid = (~np.isnan(raw_ku)) & (raw_ku > -9000.0)
        if not np.any(valid):
            return empty_tensor, empty_mask, False

        # Physical normalization: dBZ usually [0, 60] -> [0, 1]
        norm_ku = np.clip(np.where(valid, raw_ku, 0.0), 0.0, 60.0) / 60.0
        mask_ku = valid.astype(np.float32)

        # Reshape to (1, 1, D, H_in, W_in) for 3D trilinear interpolation
        t_data = torch.from_numpy(norm_ku.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t_mask = torch.from_numpy(mask_ku).unsqueeze(0).unsqueeze(0)

        data_resized = F.interpolate(
            t_data, size=(target_depth, H, W), mode="trilinear", align_corners=False
        ).squeeze(0) # (1, D, H, W)

        mask_resized = F.interpolate(
            t_mask, size=(target_depth, H, W), mode="nearest"
        ).squeeze(0) # (1, D, H, W)

        data_clean = data_resized * mask_resized
        return data_clean, mask_resized, True

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

        # Targets from container if available
        storm_lat_from_nc = None
        storm_lon_from_nc = None

        if os.path.exists(nc_path):
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

                        # 1. Infrared (IRWIN)
                        if has_ir and "infrared" in curr_grp.groups:
                            try:
                                ir_grp = curr_grp.groups["infrared"]
                                if "IRWIN" in ir_grp.variables:
                                    raw_ir = np.ma.filled(ir_grp.variables["IRWIN"][:], np.nan)
                                    # Normalize IR brightness temperature: 200-300 K -> [0, 1]
                                    norm_ir = (raw_ir - 200.0) / 100.0
                                    ir_tensor, ir_pixel_mask = self._resize_to_grid_with_mask(norm_ir, (H, W))
                                    if ir_pixel_mask.sum() == 0:
                                        has_ir = False
                            except Exception as e_ir:
                                has_ir = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "infrared", "IRWIN", str(e_ir))

                        # 2. Passive Microwave (Deterministic Sensor-Aware Channel Selection)
                        if has_pmw and "passive_microwave" in curr_grp.groups:
                            try:
                                pmw_grp = curr_grp.groups["passive_microwave"]
                                ch_map = SENSOR_PMW_CHANNEL_MAP.get(instrument, {})

                                for ch_idx in range(4):
                                    mapping = ch_map.get(ch_idx)
                                    if mapping is not None:
                                        sw_name, var_name = mapping
                                        if sw_name in pmw_grp.groups and var_name in pmw_grp.groups[sw_name].variables:
                                            raw_tb = np.ma.filled(pmw_grp.groups[sw_name].variables[var_name][:], np.nan)
                                            # Normalize TB: 150-300 K -> [0, 1]
                                            norm_tb = (raw_tb - 150.0) / 150.0
                                            t_ch, m_ch = self._resize_to_grid_with_mask(norm_tb, (H, W))
                                            if m_ch.sum() > 0:
                                                pmw_tensor[ch_idx] = t_ch.squeeze(0)
                                                pmw_pixel_mask[ch_idx] = m_ch.squeeze(0)
                                                pmw_channel_mask[ch_idx] = 1.0

                                if pmw_channel_mask.sum() == 0:
                                    has_pmw = False
                            except Exception as e_pmw:
                                has_pmw = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "passive_microwave", "TB", str(e_pmw))

                        # 3. GPROF Multi-Field Precipitation
                        if has_gprof and "GPROF" in curr_grp.groups:
                            try:
                                gp_grp = curr_grp.groups["GPROF"]
                                gprof_fields = [
                                    ("surfacePrecipitation", 0, 5.0),
                                    ("convectivePrecipitation", 1, 5.0),
                                    ("rainWaterPath", 2, 5.0),
                                    ("iceWaterPath", 3, 5.0),
                                ]
                                for fname, fidx, norm_scale in gprof_fields:
                                    if fname in gp_grp.variables:
                                        raw_f = np.ma.filled(gp_grp.variables[fname][:], np.nan)
                                        norm_f = np.log1p(np.maximum(raw_f, 0.0)) / norm_scale
                                        t_f, m_f = self._resize_to_grid_with_mask(norm_f, (H, W))
                                        if m_f.sum() > 0:
                                            gprof_tensor[fidx] = t_f.squeeze(0)
                                            gprof_pixel_mask[fidx] = m_f.squeeze(0)
                                            gprof_channel_mask[fidx] = 1.0

                                if gprof_channel_mask.sum() == 0:
                                    has_gprof = False
                            except Exception as e_gp:
                                has_gprof = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "GPROF", "precipitation", str(e_gp))

                        # 4. Radar Volumetric Profile
                        if has_radar and "radar_radiometer" in curr_grp.groups:
                            try:
                                rad_grp = curr_grp.groups["radar_radiometer"]
                                r_tensor, r_mask, r_valid = self._load_radar_volume(rad_grp, (H, W), target_depth=16)
                                if r_valid:
                                    radar_tensor = r_tensor
                                    radar_voxel_mask = r_mask
                                    has_radar = True
                                else:
                                    has_radar = False
                            except Exception as e_rad:
                                has_radar = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "radar_radiometer", "Ku", str(e_rad))
                        else:
                            has_radar = False

                        # 5. Environmental Synoptic Reanalysis (6 Channels)
                        if has_env and "environmental_synoptic" in curr_grp.groups:
                            try:
                                env_grp = curr_grp.groups["environmental_synoptic"]
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
                                        norm_e = (raw_e - foffset) / fscale
                                        t_e, m_e = self._resize_to_grid_with_mask(norm_e, (H, W))
                                        if m_e.sum() > 0:
                                            env_tensor[fidx] = t_e.squeeze(0)
                                            env_pixel_mask[fidx] = m_e.squeeze(0)
                                            env_channel_mask[fidx] = 1.0

                                if env_channel_mask.sum() == 0:
                                    has_env = False
                            except Exception as e_env:
                                has_env = False
                                self._record_error(sample_id, storm_id, nc_path, group_path, "environmental_synoptic", "env", str(e_env))

            except Exception as e_top:
                self._record_error(sample_id, storm_id, nc_path, group_path, "container", "open", str(e_top))
                if self.strict_mode:
                    raise RuntimeError(f"Strict mode failure loading container {nc_path}: {e_top}")
        else:
            self._record_error(sample_id, storm_id, nc_path, group_path, "filesystem", "missing", "Container path does not exist")
            if self.strict_mode:
                raise FileNotFoundError(f"Container not found: {nc_path}")

        # Current storm coordinates
        cur_lat = row.get("storm_latitude")
        if pd.isna(cur_lat) or cur_lat is None:
            cur_lat = storm_lat_from_nc
        cur_lon = row.get("storm_longitude")
        if pd.isna(cur_lon) or cur_lon is None:
            cur_lon = storm_lon_from_nc

        has_valid_cur_pos = (cur_lat is not None) and (cur_lon is not None) and (not math.isnan(float(cur_lat))) and (not math.isnan(float(cur_lon)))

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
            "masks": {
                "ir_pixel": ir_pixel_mask,
                "pmw_pixel": pmw_pixel_mask,
                "gprof_pixel": gprof_pixel_mask,
                "radar_voxel": radar_voxel_mask,
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
    ):
        super().__init__()
        self.base_dataset = TCPRIMEDDataset(
            sample_index_csv=sample_index_csv,
            partition=partition,
            grid_size=grid_size,
            strict_mode=strict_mode,
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
