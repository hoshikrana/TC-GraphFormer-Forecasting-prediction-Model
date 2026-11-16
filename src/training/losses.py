"""
Multi-Task Loss Module for TC-GraphFormer.
Computes strictly mask-aware multi-task losses with valid-target denominators:
- Current Wind Speed (kt)
- Central Minimum Pressure (hPa)
- Development Stage Classification (Canonical 5 classes)
- Multi-Horizon Intensity Changes (+6h, +12h, +18h, +24h kt)
- Rapid Intensification (24h binary flag with TRAIN-fitted pos_weight)
- Track Displacements (Delta x, Delta y in km across +6h, +12h, +18h, +24h)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Combines regression, classification, horizon changes, and geodesic track losses
    with independent validity masks.
    """

    def __init__(
        self,
        weight_intensity: float = 1.0,
        weight_pressure: float = 0.5,
        weight_stage: float = 0.5,
        weight_intensity_change: float = 1.0,
        weight_ri: float = 2.0,
        weight_track: float = 1.0,
        ri_pos_weight: float = 33.53, # Fitted strictly on TRAIN partition (1207 neg / 36 pos)
    ):
        super().__init__()
        self.w_v = weight_intensity
        self.w_p = weight_pressure
        self.w_s = weight_stage
        self.w_dv = weight_intensity_change
        self.w_ri = weight_ri
        self.w_track = weight_track

        self.register_buffer("ri_weight", torch.tensor([ri_pos_weight], dtype=torch.float32))

    @property
    def task_weights(self) -> Dict[str, float]:
        return {
            "intensity": self.w_v,
            "pressure": self.w_p,
            "stage": self.w_s,
            "intensity_change": self.w_dv,
            "ri": self.w_ri,
            "track": self.w_track,
        }

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        group_target_counts: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds: output of TCGraphFormer
        targets: target dict from batch
        group_target_counts: optional dictionary of total valid targets across an accumulation group
                            for exact sample-weighted gradient accumulation.
        Returns:
            total_loss: torch.Tensor
            loss_components: Dict[str, float]
        """
        # 1. Current Intensity Loss (Huber, masked)
        v_pred = preds["intensity"]
        v_true = targets["intensity"]
        v_mask = targets.get("intensity_mask", torch.ones_like(v_true))
        diff_v = F.smooth_l1_loss(v_pred, v_true, reduction="none", beta=1.0)
        unreduced_v = (diff_v * v_mask).sum()
        batch_valid_v = v_mask.sum()
        n_v = group_target_counts.get("intensity") if (group_target_counts and "intensity" in group_target_counts) else batch_valid_v.clamp(min=1.0)
        loss_v = unreduced_v / max(1.0, float(n_v))

        # 2. Central Pressure Loss (Huber, scaled down by 10 to balance gradient magnitudes, masked)
        p_pred = preds["pressure"] / 10.0
        p_true = targets["pressure"] / 10.0
        p_mask = targets.get("pressure_mask", torch.ones_like(p_true))
        diff_p = F.smooth_l1_loss(p_pred, p_true, reduction="none", beta=1.0)
        unreduced_p = (diff_p * p_mask).sum()
        batch_valid_p = p_mask.sum()
        n_p = group_target_counts.get("pressure") if (group_target_counts and "pressure" in group_target_counts) else batch_valid_p.clamp(min=1.0)
        loss_p = unreduced_p / max(1.0, float(n_p))

        # 3. Development Stage Classification Loss (Cross Entropy, masked for valid stages)
        stage_logits = preds["stage_logits"] # (B, 5) or (B, 6)
        s_true = targets["stage"]
        s_mask = targets.get("stage_mask", torch.ones_like(s_true, dtype=torch.float32))
        ce_s = F.cross_entropy(stage_logits, s_true, reduction="none")
        unreduced_s = (ce_s * s_mask).sum()
        batch_valid_s = s_mask.sum()
        n_s = group_target_counts.get("stage") if (group_target_counts and "stage" in group_target_counts) else batch_valid_s.clamp(min=1.0)
        loss_s = unreduced_s / max(1.0, float(n_s))

        # 4. Multi-Horizon Intensity Change Loss (Masked Smooth L1)
        dv_pred = preds["intensity_change"] # (B, 4)
        dv_true = targets["intensity_change"] # (B, 4)
        dv_mask = targets["intensity_change_mask"] # (B, 4)
        diff_dv = F.smooth_l1_loss(dv_pred, dv_true, reduction="none", beta=1.0)
        unreduced_dv = (diff_dv * dv_mask).sum()
        batch_valid_dv = dv_mask.sum()
        n_dv = group_target_counts.get("intensity_change") if (group_target_counts and "intensity_change" in group_target_counts) else batch_valid_dv.clamp(min=1.0)
        loss_dv = unreduced_dv / max(1.0, float(n_dv))

        # 5. Rapid Intensification Loss (Masked BCE with Logits)
        ri_pred = preds["ri_logits"] # (B,)
        ri_true = targets["rapid_intensification"] # (B,)
        ri_mask = targets["ri_mask"] # (B,)
        ri_bce = F.binary_cross_entropy_with_logits(
            ri_pred, ri_true, pos_weight=self.ri_weight, reduction="none"
        )
        unreduced_ri = (ri_bce * ri_mask).sum()
        batch_valid_ri = ri_mask.sum()
        n_ri = group_target_counts.get("rapid_intensification") if (group_target_counts and "rapid_intensification" in group_target_counts) else batch_valid_ri.clamp(min=1.0)
        loss_ri = unreduced_ri / max(1.0, float(n_ri))

        # 6. Future Track Displacement Loss (Masked Smooth L1 on [dx, dy] in km scaled by 100km)
        if "track_delta" in preds and "track_displacement" in targets:
            track_pred = preds["track_delta"].view(-1, 4, 2) / 100.0 # (B, 4, 2)
            track_true = targets["track_displacement"] / 100.0       # (B, 4, 2)
            track_mask = targets.get("track_mask")
            if track_mask is None:
                track_mask = torch.ones_like(track_pred[:, :, :1])
                valid_tr_timesteps = float(track_mask.sum().item())
                valid_tr_components = valid_tr_timesteps * 2.0
            elif track_mask.dim() == 2:
                if track_mask.size(1) != 4:
                    raise ValueError(f"Unsupported track_mask shape {track_mask.shape}; expected (B, 4)")
                track_mask = track_mask.unsqueeze(-1)  # (B, 4, 1)
                valid_tr_timesteps = float(track_mask.sum().item())
                valid_tr_components = valid_tr_timesteps * 2.0
            elif track_mask.dim() == 3:
                if track_mask.size(1) != 4:
                    raise ValueError(f"Unsupported track_mask shape {track_mask.shape}; expected (B, 4, 1) or (B, 4, 2)")
                if track_mask.size(2) == 1:
                    valid_tr_timesteps = float(track_mask.sum().item())
                    valid_tr_components = valid_tr_timesteps * 2.0
                elif track_mask.size(2) == 2:
                    valid_tr_components = float(track_mask.sum().item())
                    valid_tr_timesteps = valid_tr_components / 2.0
                else:
                    raise ValueError(f"Unsupported track_mask last dimension {track_mask.size(2)}; expected 1 or 2")
            else:
                raise ValueError(f"Unsupported track_mask dim {track_mask.dim()}; expected 2 or 3")

            diff_tr = F.smooth_l1_loss(track_pred, track_true, reduction="none", beta=1.0)
            unreduced_tr = (diff_tr * track_mask).sum()
            n_tr = group_target_counts.get("track_displacement") if (group_target_counts and "track_displacement" in group_target_counts) else max(1.0, valid_tr_components)
            loss_track = unreduced_tr / max(1.0, float(n_tr))
            valid_tr_count = valid_tr_components
        else:
            loss_track = torch.tensor(0.0, device=preds["intensity"].device)
            unreduced_tr = torch.tensor(0.0, device=preds["intensity"].device)
            batch_valid_tr = torch.tensor(0.0, device=preds["intensity"].device)
            valid_tr_timesteps = 0.0
            valid_tr_components = 0.0
            valid_tr_count = 0.0
            n_tr = 0.0

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.w_v * loss_v +
            self.w_p * loss_p +
            self.w_s * loss_s +
            self.w_dv * loss_dv +
            self.w_ri * loss_ri +
            self.w_track * loss_track
        )

        def _to_float(val):
            return float(val.item()) if hasattr(val, "item") else float(val)

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_intensity": float(loss_v.detach().item()),
            "loss_pressure": float(loss_p.detach().item()),
            "loss_stage": float(loss_s.detach().item()),
            "loss_intensity_change": float(loss_dv.detach().item()),
            "loss_ri": float(loss_ri.detach().item()),
            "loss_track": float(loss_track.detach().item()),
            "unreduced_loss_intensity": float(unreduced_v.detach().item()),
            "unreduced_loss_pressure": float(unreduced_p.detach().item()),
            "unreduced_loss_stage": float(unreduced_s.detach().item()),
            "unreduced_loss_intensity_change": float(unreduced_dv.detach().item()),
            "unreduced_loss_ri": float(unreduced_ri.detach().item()),
            "unreduced_loss_track": float(unreduced_tr.detach().item()),
            "valid_intensity_count": float(batch_valid_v.detach().item()),
            "valid_pressure_count": float(batch_valid_p.detach().item()),
            "valid_stage_count": float(batch_valid_s.detach().item()),
            "valid_intensity_change_count": float(batch_valid_dv.detach().item()),
            "valid_ri_count": float(batch_valid_ri.detach().item()),
            "valid_track_components_count": valid_tr_components,
            "valid_track_timesteps_count": valid_tr_timesteps,
            "valid_track_count": valid_tr_components,
            "valid_v_count": float(batch_valid_v.detach().item()),
            "valid_p_count": float(batch_valid_p.detach().item()),
            "valid_s_count": float(batch_valid_s.detach().item()),
            "valid_dv_count": float(batch_valid_dv.detach().item()),
            "valid_tr_count": valid_tr_timesteps,
        }
        return total_loss, loss_dict
