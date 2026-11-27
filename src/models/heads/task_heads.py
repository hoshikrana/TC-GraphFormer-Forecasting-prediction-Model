"""
Multi-Task Prediction Heads for TC-GraphFormer.
Decodes shared latent spatiotemporal representation into:
- Current Wind Speed (kt)
- Central Minimum Pressure (hPa)
- Development Stage (5 canonical classes: 0=TD/SD/MD, 1=TS/SS, 2=HU/TY, 3=EX, 4=DB/LO/WV)
- Intensity Changes (+6h, +12h, +18h, +24h)
- Rapid Intensification (24h binary flag)
- Track Displacements (Delta x_east, Delta y_north in km across horizons)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class TaskPredictionHeads(nn.Module):
    """
    Multi-task prediction heads operating on shared latent representation.
    """

    def __init__(self, d_model: int = 128, num_stages: int = 5, head_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages
        self.head_dropout = float(head_dropout)
        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError(f"head_dropout must be in [0.0, 1.0), got {head_dropout}")
        self.head_drop = nn.Dropout(p=self.head_dropout)

        # 1. Current Intensity Head (Wind Speed in knots)
        self.head_intensity = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 2. Central Minimum Pressure Head (hPa)
        self.head_pressure = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 3. Development Stage Classification Head (5 canonical classes: TD, TS, HU/TY, EX, DB/LO/WV)
        self.head_stage = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_stages),
        )

        # 4. Multi-Horizon Intensity Change Head (+6h, +12h, +18h, +24h kt)
        self.head_intensity_change = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

        # 5. Rapid Intensification Binary Classification Head (>= 30 kt / 24h)
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 6. Track Displacement Head (Delta lat, Delta lon for 4 horizons = 8 values)
        self.head_track = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )

        self._init_physical_priors()

    def _init_physical_priors(self) -> None:
        """
        Initialize output layer biases to TRAIN partition physical climatological priors
        (wind speed mean = 38.19 kt, central pressure mean = 997.25 hPa, multi-horizon
        intensity change means, RI base-rate log-odds = -3.5124, and mean westward/poleward
        track displacements in km) to prevent Huber loss gradient saturation and probability
        miscalibration.
        """
        with torch.no_grad():
            self.head_intensity[-1].bias.fill_(38.19)
            self.head_pressure[-1].bias.fill_(997.25)
            self.head_intensity_change[-1].bias.copy_(
                torch.tensor([0.231, 0.575, 1.075, 1.596], dtype=torch.float32)
            )
            self.head_ri[-1].bias.fill_(-3.5124)
            self.head_track[-1].bias.copy_(
                torch.tensor(
                    [-25.35, 27.92, -40.77, 60.58, -62.82, 88.80, -79.29, 114.98],
                    dtype=torch.float32,
                )
            )

    def _apply_head(self, head: nn.Sequential, latent: torch.Tensor) -> torch.Tensor:
        """
        Apply 2-layer task head with optional intermediate dropout after GELU activation:
        Linear(d_model, 64) -> GELU -> Dropout(p=head_dropout) -> Linear(64, out_dim).
        """
        h = head[0](latent)
        h = head[1](h)
        if self.head_dropout > 0.0:
            h = self.head_drop(h)
        return head[2](h)

    def forward(
        self,
        latent: torch.Tensor,
        kinematic_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        latent: (B, d_model)
        kinematic_state: optional (B, 10) causal observation-time translation & geophysical state
        Returns dictionary of predictions:
            intensity: (B,)
            pressure: (B,)
            stage_logits: (B, 5)
            intensity_change: (B, 4)
            ri_logits: (B,)
            track_delta: (B, 8)
        """
        raw_v = self._apply_head(self.head_intensity, latent).squeeze(-1)   # (B,)
        v_pred = 38.19 + 12.0 * (raw_v - 38.19)

        raw_p = self._apply_head(self.head_pressure, latent).squeeze(-1)    # (B,)
        p_pred = 997.25 + 12.0 * (raw_p - 997.25)

        stage_logits = self._apply_head(self.head_stage, latent)            # (B, num_stages)

        raw_dv = self._apply_head(self.head_intensity_change, latent)       # (B, 4)
        dv_prior = torch.tensor(
            [0.231, 0.575, 1.075, 1.596], device=latent.device, dtype=raw_dv.dtype
        )
        dv_pred = dv_prior + 5.0 * (raw_dv - dv_prior)

        ri_logit = self._apply_head(self.head_ri, latent).squeeze(-1)       # (B,)

        raw_tr = self._apply_head(self.head_track, latent)                  # (B, 8)
        clim_tr = torch.tensor(
            [-25.35, 27.92, -40.77, 60.58, -62.82, 88.80, -79.29, 114.98],
            device=latent.device,
            dtype=raw_tr.dtype,
        )
        tr_residual = 20.0 * (raw_tr - clim_tr)

        if kinematic_state is not None and kinematic_state.dim() == 2 and kinematic_state.size(-1) >= 10:
            kin = kinematic_state.to(device=latent.device, dtype=raw_tr.dtype)
            u_kmh = kin[:, 0:1]
            v_kmh = kin[:, 1:2]
            has_kin = (kin[:, 9:10] > 0.5).to(raw_tr.dtype)

            # Strictly causal TRAIN-fitted damped kinematic translation prior across +6h, +12h, +18h, +24h
            dx_6 = 0.970 * (u_kmh * 6.0) + 5.10
            dy_6 = 0.849 * (v_kmh * 6.0) + 4.47
            dx_12 = 0.938 * (u_kmh * 12.0) + 14.14
            dy_12 = 0.785 * (v_kmh * 12.0) + 16.01
            dx_18 = 0.888 * (u_kmh * 18.0) + 15.96
            dy_18 = 0.761 * (v_kmh * 18.0) + 27.49
            dx_24 = 0.868 * (u_kmh * 24.0) + 28.74
            dy_24 = 0.749 * (v_kmh * 24.0) + 35.16

            kin_disp = torch.cat([dx_6, dy_6, dx_12, dy_12, dx_18, dy_18, dx_24, dy_24], dim=-1)
            base_tr = has_kin * kin_disp + (1.0 - has_kin) * clim_tr
            track_delta = base_tr + tr_residual
        else:
            track_delta = clim_tr + tr_residual

        return {
            "intensity": v_pred,
            "pressure": p_pred,
            "stage_logits": stage_logits,
            "intensity_change": dv_pred,
            "ri_logits": ri_logit,
            "track_delta": track_delta,
        }

