"""
PyTorch Dataset for NOAA TC PRIMED v01r01-final.
Provides multi-modal spatiotemporal overpass tensors with modality masks
and multi-task forecasting targets from standardized NetCDF-4/HDF5 containers.
"""

import os
import math
import numpy as np
import pandas as pd
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


# Canonical 6-class stage mapping
STAGE_MAPPING = {
    "TD": 0, "SD": 0, "MD": 0,
    "TS": 1, "SS": 1,
    "HU": 2, "TY": 2,
    "EX": 3,
    "DB": 4, "LO": 4, "WV": 4,
    "UN": 5
}
NUM_STAGES = 6


class TCPRIMEDDataset(Dataset):
    """
    Random-access PyTorch Dataset for TC PRIMED overpass samples.
    Reads directly from processed NetCDF containers indexed by sample_index.csv.
    """

    def __init__(
        self,
        sample_index_csv: str = "reports/pipeline/sample_index.csv",
        partition: Optional[str] = "TRAIN",
        grid_size: int = 64,
        max_samples: Optional[int] = None,
        transform=None,
    ):
        super().__init__()
        self.sample_index_csv = sample_index_csv
        self.partition = partition.upper() if partition else None
        self.grid_size = grid_size
        self.transform = transform

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

    def _resize_to_grid(self, arr: np.ndarray, target_shape: Tuple[int, int]) -> torch.Tensor:
        """Interpolate 2D numpy array to canonical (1, target_h, target_w) tensor."""
        if arr is None or arr.size == 0 or np.isnan(arr).all():
            return torch.zeros((1, target_shape[0], target_shape[1]), dtype=torch.float32)

        # Impute NaNs with median/mean
        valid_mask = ~np.isnan(arr)
        if np.any(valid_mask):
            mean_val = float(np.mean(arr[valid_mask]))
            arr_clean = np.where(valid_mask, arr, mean_val)
        else:
            arr_clean = np.zeros_like(arr)

        t = torch.from_numpy(arr_clean.astype(np.float32)).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        t_resized = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return t_resized.squeeze(0) # (1, H, W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        nc_path = row["processed_container_path"]
        group_path = row["container_group_path"]
        
        # In case paths have Windows vs POSIX differences
        if not os.path.exists(nc_path):
            alt_path = nc_path.replace("\\", "/")
            if os.path.exists(alt_path):
                nc_path = alt_path

        # Initialize default empty tensors
        H, W = self.grid_size, self.grid_size
        ir_tensor = torch.zeros((1, H, W), dtype=torch.float32)
        pmw_tensor = torch.zeros((4, H, W), dtype=torch.float32) # [S1_low_v, S1_low_h, S2_high_v, S2_high_h]
        gprof_tensor = torch.zeros((1, H, W), dtype=torch.float32)
        radar_tensor = torch.zeros((1, 16, H // 2, W // 2), dtype=torch.float32) # (C, D, H, W)
        env_tensor = torch.zeros((1, H, W), dtype=torch.float32)

        has_ir = bool(row["has_ir"])
        has_pmw = bool(row["has_pmw"])
        has_gprof = bool(row["has_gprof"])
        has_radar = bool(row["has_radar"])
        has_env = bool(row["has_causal_env"])

        if os.path.exists(nc_path):
            try:
                with nc.Dataset(nc_path, "r") as ds:
                    # Navigate group path e.g. /overpasses/obs_...
                    parts = [p for p in group_path.split("/") if p]
                    curr_grp = ds
                    for p in parts:
                        if p in curr_grp.groups:
                            curr_grp = curr_grp.groups[p]
                        else:
                            curr_grp = None
                            break

                    if curr_grp is not None:
                        # 1. Infrared
                        if has_ir and "infrared" in curr_grp.groups:
                            ir_grp = curr_grp.groups["infrared"]
                            if "IRWIN" in ir_grp.variables:
                                raw_ir = np.ma.filled(ir_grp.variables["IRWIN"][:], np.nan)
                                # Normalize IR brightness temperature: ~200-300 K -> [0, 1]
                                norm_ir = (raw_ir - 200.0) / 100.0
                                ir_tensor = self._resize_to_grid(norm_ir, (H, W))

                        # 2. Passive Microwave
                        if has_pmw and "passive_microwave" in curr_grp.groups:
                            pmw_grp = curr_grp.groups["passive_microwave"]
                            # S1 low frequency channels
                            ch_tensors = []
                            if "S1" in pmw_grp.groups:
                                s1_grp = pmw_grp.groups["S1"]
                                tb_vars = [v for v in s1_grp.variables if v.startswith("TB_")]
                                for vname in tb_vars[:2]: # take first 2 channels (e.g. 19V, 19H)
                                    raw_tb = np.ma.filled(s1_grp.variables[vname][:], np.nan)
                                    norm_tb = (raw_tb - 150.0) / 150.0
                                    ch_tensors.append(self._resize_to_grid(norm_tb, (H, W)))
                            while len(ch_tensors) < 2:
                                ch_tensors.append(torch.zeros((1, H, W), dtype=torch.float32))

                            # S2 high frequency channels
                            if "S2" in pmw_grp.groups:
                                s2_grp = pmw_grp.groups["S2"]
                                tb_vars = [v for v in s2_grp.variables if v.startswith("TB_")]
                                for vname in tb_vars[:2]: # take first 2 channels (e.g. 85V, 85H)
                                    raw_tb = np.ma.filled(s2_grp.variables[vname][:], np.nan)
                                    norm_tb = (raw_tb - 150.0) / 150.0
                                    ch_tensors.append(self._resize_to_grid(norm_tb, (H, W)))
                            while len(ch_tensors) < 4:
                                ch_tensors.append(torch.zeros((1, H, W), dtype=torch.float32))

                            pmw_tensor = torch.cat(ch_tensors, dim=0) # (4, H, W)

                        # 3. GPROF surface precipitation
                        if has_gprof and "GPROF" in curr_grp.groups:
                            gp_grp = curr_grp.groups["GPROF"]
                            if "surfacePrecipitation" in gp_grp.variables:
                                raw_p = np.ma.filled(gp_grp.variables["surfacePrecipitation"][:], np.nan)
                                norm_p = np.log1p(np.maximum(raw_p, 0.0)) / 5.0
                                gprof_tensor = self._resize_to_grid(norm_p, (H, W))

                        # 4. Environmental Synoptic precipitable water
                        if has_env and "environmental_synoptic" in curr_grp.groups:
                            env_grp = curr_grp.groups["environmental_synoptic"]
                            if "precipitable_water" in env_grp.variables:
                                raw_pw = np.ma.filled(env_grp.variables["precipitable_water"][:], np.nan)
                                norm_pw = raw_pw / 80.0
                                env_tensor = self._resize_to_grid(norm_pw, (H, W))
            except Exception as e:
                # Fallback to zero tensors if container read issue
                pass

        # Targets extraction
        intensity = float(row.get("intensity_vmax_kt", 0.0)) if not pd.isna(row.get("intensity_vmax_kt")) else 0.0
        pressure = float(row.get("central_min_pressure_hpa", 1010.0)) if not pd.isna(row.get("central_min_pressure_hpa")) else 1010.0

        raw_stage = str(row.get("development_stage_raw", "UN")).strip().upper()
        stage_id = STAGE_MAPPING.get(raw_stage, 5)

        # Intensity changes for +6h, +12h, +18h, +24h
        int_changes = []
        int_change_masks = []
        for h in [6, 12, 18, 24]:
            col = f"intensity_change_{h}h_kt"
            val = row.get(col)
            if pd.isna(val):
                int_changes.append(0.0)
                int_change_masks.append(0.0)
            else:
                int_changes.append(float(val))
                int_change_masks.append(1.0)

        # Rapid intensification (30 kt / 24h)
        ri_val = row.get("is_rapid_intensification_24h")
        if pd.isna(ri_val):
            ri_target = 0.0
            ri_mask = 0.0
        else:
            ri_target = 1.0 if float(ri_val) > 0.5 else 0.0
            ri_mask = 1.0

        # Modality masks tensor: [has_ir, has_pmw, has_gprof, has_radar, has_env]
        modality_masks = torch.tensor(
            [1.0 if has_ir else 0.0,
             1.0 if has_pmw else 0.0,
             1.0 if has_gprof else 0.0,
             1.0 if has_radar else 0.0,
             1.0 if has_env else 0.0],
            dtype=torch.float32
        )

        sample = {
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "timestamp_epoch": int(row["timestamp_epoch"]),
            "modality_masks": modality_masks,
            "ir": ir_tensor,         # (1, H, W)
            "pmw": pmw_tensor,       # (4, H, W)
            "gprof": gprof_tensor,   # (1, H, W)
            "radar": radar_tensor,   # (1, 16, H/2, W/2)
            "env": env_tensor,       # (1, H, W)
            "targets": {
                "intensity": torch.tensor(intensity, dtype=torch.float32),
                "pressure": torch.tensor(pressure, dtype=torch.float32),
                "stage": torch.tensor(stage_id, dtype=torch.long),
                "intensity_change": torch.tensor(int_changes, dtype=torch.float32), # (4,)
                "intensity_change_mask": torch.tensor(int_change_masks, dtype=torch.float32), # (4,)
                "rapid_intensification": torch.tensor(ri_target, dtype=torch.float32),
                "ri_mask": torch.tensor(ri_mask, dtype=torch.float32),
            }
        }
        return sample

