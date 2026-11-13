"""
Collate functions for TC PRIMED DataLoader.
Handles batching of multimodal spatiotemporal observations with modality masks,
channel validity masks, target masks, and variable-length sequence padding.
"""

from typing import List, Dict, Any
import torch


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard batch collation for single-observation multimodal samples.
    """
    sample_ids = [item["sample_id"] for item in batch]
    storm_ids = [item["storm_id"] for item in batch]
    timestamps = torch.tensor([item["timestamp_epoch"] for item in batch], dtype=torch.long)
    modality_masks = torch.stack([item["modality_masks"] for item in batch], dim=0) # (B, 5)

    ir_batch = torch.stack([item["ir"] for item in batch], dim=0)       # (B, 1, H, W)
    pmw_batch = torch.stack([item["pmw"] for item in batch], dim=0)     # (B, 4, H, W)
    gprof_batch = torch.stack([item["gprof"] for item in batch], dim=0) # (B, 4, H, W)
    radar_batch = torch.stack([item["radar"] for item in batch], dim=0) # (B, 1, 16, H/2, W/2)
    env_batch = torch.stack([item["env"] for item in batch], dim=0)     # (B, 6, H, W)

    # Channel masks
    pmw_ch_mask = torch.stack([item["masks"]["pmw_channels"] for item in batch], dim=0)     # (B, 4)
    gprof_ch_mask = torch.stack([item["masks"]["gprof_channels"] for item in batch], dim=0) # (B, 4)
    env_ch_mask = torch.stack([item["masks"]["env_channels"] for item in batch], dim=0)     # (B, 6)

    # Pixel and Voxel masks
    ir_pixel_mask = torch.stack([item["masks"]["ir_pixel"] for item in batch], dim=0)       # (B, 1, H, W)
    pmw_pixel_mask = torch.stack([item["masks"]["pmw_pixel"] for item in batch], dim=0)     # (B, 4, H, W)
    gprof_pixel_mask = torch.stack([item["masks"]["gprof_pixel"] for item in batch], dim=0) # (B, 4, H, W)
    radar_voxel_mask = torch.stack([item["masks"]["radar_voxel"] for item in batch], dim=0) # (B, 1, 16, H/2, W/2)
    env_pixel_mask = torch.stack([item["masks"]["env_pixel"] for item in batch], dim=0)     # (B, 6, H, W)

    # Targets & Masks
    intensity_batch = torch.stack([item["targets"]["intensity"] for item in batch], dim=0)
    intensity_mask_batch = torch.stack([item["targets"]["intensity_mask"] for item in batch], dim=0)

    pressure_batch = torch.stack([item["targets"]["pressure"] for item in batch], dim=0)
    pressure_mask_batch = torch.stack([item["targets"]["pressure_mask"] for item in batch], dim=0)

    stage_batch = torch.stack([item["targets"]["stage"] for item in batch], dim=0)
    stage_mask_batch = torch.stack([item["targets"]["stage_mask"] for item in batch], dim=0)

    int_change_batch = torch.stack([item["targets"]["intensity_change"] for item in batch], dim=0) # (B, 4)
    int_change_mask = torch.stack([item["targets"]["intensity_change_mask"] for item in batch], dim=0) # (B, 4)

    ri_batch = torch.stack([item["targets"]["rapid_intensification"] for item in batch], dim=0)
    ri_mask_batch = torch.stack([item["targets"]["ri_mask"] for item in batch], dim=0)

    track_batch = torch.stack([item["targets"]["track_displacement"] for item in batch], dim=0) # (B, 4, 2)
    track_mask_batch = torch.stack([item["targets"]["track_mask"] for item in batch], dim=0)    # (B, 4)

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
        "channel_masks": {
            "pmw": pmw_ch_mask,
            "gprof": gprof_ch_mask,
            "env": env_ch_mask,
        },
        "pixel_masks": {
            "ir": ir_pixel_mask,
            "pmw": pmw_pixel_mask,
            "gprof": gprof_pixel_mask,
            "radar": radar_voxel_mask,
            "env": env_pixel_mask,
        },
        "targets": {
            "intensity": intensity_batch,
            "intensity_mask": intensity_mask_batch,
            "pressure": pressure_batch,
            "pressure_mask": pressure_mask_batch,
            "stage": stage_batch,
            "stage_mask": stage_mask_batch,
            "intensity_change": int_change_batch,
            "intensity_change_mask": int_change_mask,
            "rapid_intensification": ri_batch,
            "ri_mask": ri_mask_batch,
            "track_displacement": track_batch,
            "track_mask": track_mask_batch,
        },
    }


def collate_sequence_batch(batch: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Collation for historical sequence samples.
    Each item in batch is a list of chronological observations [t_0, ..., t_k].
    Pads sequences to uniform length T_max with boolean padding mask.
    The last observation in each sequence is the anchor observation possessing future prediction targets.
    """
    B = len(batch)
    seq_lengths = [len(seq) for seq in batch]
    T_max = max(seq_lengths)

    # Reference shapes from first observation
    first_obs = batch[0][0]
    H, W = first_obs["ir"].shape[-2], first_obs["ir"].shape[-1]
    D_rad = first_obs["radar"].shape[1]

    # Preallocate batch sequence tensors
    ir_seq = torch.zeros((B, T_max, 1, H, W), dtype=torch.float32)
    pmw_seq = torch.zeros((B, T_max, 4, H, W), dtype=torch.float32)
    gprof_seq = torch.zeros((B, T_max, 4, H, W), dtype=torch.float32)
    radar_seq = torch.zeros((B, T_max, 1, D_rad, H // 2, W // 2), dtype=torch.float32)
    env_seq = torch.zeros((B, T_max, 6, H, W), dtype=torch.float32)

    modality_masks_seq = torch.zeros((B, T_max, 5), dtype=torch.float32)
    pmw_ch_seq = torch.zeros((B, T_max, 4), dtype=torch.float32)
    gprof_ch_seq = torch.zeros((B, T_max, 4), dtype=torch.float32)
    env_ch_seq = torch.zeros((B, T_max, 6), dtype=torch.float32)

    ir_pixel_seq = torch.zeros((B, T_max, 1, H, W), dtype=torch.float32)
    pmw_pixel_seq = torch.zeros((B, T_max, 4, H, W), dtype=torch.float32)
    gprof_pixel_seq = torch.zeros((B, T_max, 4, H, W), dtype=torch.float32)
    radar_voxel_seq = torch.zeros((B, T_max, 1, D_rad, H // 2, W // 2), dtype=torch.float32)
    env_pixel_seq = torch.zeros((B, T_max, 6, H, W), dtype=torch.float32)

    delta_times_seq = torch.zeros((B, T_max), dtype=torch.float32)
    seq_pad_mask = torch.ones((B, T_max), dtype=torch.bool) # True = padded/forbidden

    anchor_samples = []

    for b in range(B):
        seq = batch[b]
        L = len(seq)
        t0 = seq[0]["timestamp_epoch"]

        for t in range(L):
            obs = seq[t]
            ir_seq[b, t] = obs["ir"]
            pmw_seq[b, t] = obs["pmw"]
            gprof_seq[b, t] = obs["gprof"]
            radar_seq[b, t] = obs["radar"]
            env_seq[b, t] = obs["env"]

            modality_masks_seq[b, t] = obs["modality_masks"]
            pmw_ch_seq[b, t] = obs["masks"]["pmw_channels"]
            gprof_ch_seq[b, t] = obs["masks"]["gprof_channels"]
            env_ch_seq[b, t] = obs["masks"]["env_channels"]

            ir_pixel_seq[b, t] = obs["masks"]["ir_pixel"]
            pmw_pixel_seq[b, t] = obs["masks"]["pmw_pixel"]
            gprof_pixel_seq[b, t] = obs["masks"]["gprof_pixel"]
            radar_voxel_seq[b, t] = obs["masks"]["radar_voxel"]
            env_pixel_seq[b, t] = obs["masks"]["env_pixel"]

            # Time gap in hours relative to sequence start
            delta_times_seq[b, t] = (obs["timestamp_epoch"] - t0) / 3600.0
            seq_pad_mask[b, t] = False # valid timestep

        # The anchor is the last observation in the sequence
        anchor_samples.append(seq[-1])

    # Collate targets using anchor observations
    anchor_collate = collate_multimodal_batch(anchor_samples)

    return {
        "sample_ids": anchor_collate["sample_ids"],
        "storm_ids": anchor_collate["storm_ids"],
        "timestamps": anchor_collate["timestamps"],
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.long),
        "seq_pad_mask": seq_pad_mask, # (B, T_max) bool
        "delta_times_hours": delta_times_seq, # (B, T_max)
        "modality_masks": modality_masks_seq, # (B, T_max, 5)
        "ir": ir_seq,                         # (B, T_max, 1, H, W)
        "pmw": pmw_seq,                       # (B, T_max, 4, H, W)
        "gprof": gprof_seq,                   # (B, T_max, 4, H, W)
        "radar": radar_seq,                   # (B, T_max, 1, 16, H/2, W/2)
        "env": env_seq,                       # (B, T_max, 6, H, W)
        "channel_masks": {
            "pmw": pmw_ch_seq,
            "gprof": gprof_ch_seq,
            "env": env_ch_seq,
        },
        "pixel_masks": {
            "ir": ir_pixel_seq,
            "pmw": pmw_pixel_seq,
            "gprof": gprof_pixel_seq,
            "radar": radar_voxel_seq,
            "env": env_pixel_seq,
        },
        "targets": anchor_collate["targets"],
    }
