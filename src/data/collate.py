"""
Collate functions for TC PRIMED DataLoader.
Handles batching of multimodal spatiotemporal observations with modality masks,
loss masks, and variable sequence lengths.
"""

import torch
from typing import List, Dict, Any


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard batch collation for multimodal observation samples.
    """
    sample_ids = [item["sample_id"] for item in batch]
    storm_ids = [item["storm_id"] for item in batch]
    timestamps = torch.tensor([item["timestamp_epoch"] for item in batch], dtype=torch.long)
    modality_masks = torch.stack([item["modality_masks"] for item in batch], dim=0) # (B, 5)

    ir_batch = torch.stack([item["ir"] for item in batch], dim=0)       # (B, 1, H, W)
    pmw_batch = torch.stack([item["pmw"] for item in batch], dim=0)     # (B, 4, H, W)
    gprof_batch = torch.stack([item["gprof"] for item in batch], dim=0) # (B, 1, H, W)
    radar_batch = torch.stack([item["radar"] for item in batch], dim=0) # (B, 1, D, H/2, W/2)
    env_batch = torch.stack([item["env"] for item in batch], dim=0)     # (B, 1, H, W)

    intensity_batch = torch.stack([item["targets"]["intensity"] for item in batch], dim=0)
    pressure_batch = torch.stack([item["targets"]["pressure"] for item in batch], dim=0)
    stage_batch = torch.stack([item["targets"]["stage"] for item in batch], dim=0)
    int_change_batch = torch.stack([item["targets"]["intensity_change"] for item in batch], dim=0) # (B, 4)
    int_change_mask = torch.stack([item["targets"]["intensity_change_mask"] for item in batch], dim=0) # (B, 4)
    ri_batch = torch.stack([item["targets"]["rapid_intensification"] for item in batch], dim=0)
    ri_mask_batch = torch.stack([item["targets"]["ri_mask"] for item in batch], dim=0)

    return {
        "sample_ids": sample_ids,
        "storm_ids": storm_ids,
        "timestamps": timestamps,
        "modality_masks": modality_masks, # (B, 5) -> [ir, pmw, gprof, radar, env]
        "ir": ir_batch,
        "pmw": pmw_batch,
        "gprof": gprof_batch,
        "radar": radar_batch,
        "env": env_batch,
        "targets": {
            "intensity": intensity_batch,
            "pressure": pressure_batch,
            "stage": stage_batch,
            "intensity_change": int_change_batch,
            "intensity_change_mask": int_change_mask,
            "rapid_intensification": ri_batch,
            "ri_mask": ri_mask_batch,
        }
    }

