"""
TC-GraphFormer: Multimodal Spatiotemporal Graph-Transformer for Tropical Cyclone Forecasting.
Primary architecture unifying:
- 5 Modality-Specific Encoders (IR, PMW, GPROF, Radar, Environmental)
- Mask-Aware Cross-Modal Attention Fusion
- Sparse Spatial Graph Message Passing Network
- Continuous-Time Causal Temporal Transformer
- Multi-Task Prediction Heads
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from .encoders.ir_encoder import IREncoder
from .encoders.pmw_encoder import PMWEncoder
from .encoders.gprof_encoder import GPROFEncoder
from .encoders.radar_encoder import RadarEncoder
from .encoders.env_encoder import EnvEncoder
from .fusion.cross_modal_fusion import CrossModalFusion
from .graph.spatial_graph import SpatialGraphEncoder, PolarSectorPooling
from .temporal.temporal_transformer import CausalTemporalTransformer
from .heads.task_heads import TaskPredictionHeads


class TCGraphFormer(nn.Module):
    """
    Complete TC-GraphFormer Model Architecture.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_fusion_heads: int = 4,
        num_fusion_layers: int = 2,
        num_graph_layers: int = 2,
        num_temporal_layers: int = 2,
        num_stages: int = 5,
        min_sector_coverage: float = 0.05,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        modality_dropout_prob: float = 0.0,
        sector_token_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.min_sector_coverage = min_sector_coverage
        self.head_dropout = float(head_dropout)
        self.modality_dropout_prob = float(modality_dropout_prob)
        self.sector_token_dropout = float(sector_token_dropout)
        self.drop_path_rate = float(drop_path_rate)
        if not (0.0 <= self.modality_dropout_prob < 1.0):
            raise ValueError(f"modality_dropout_prob must be in [0.0, 1.0), got {modality_dropout_prob}")
        if not (0.0 <= self.sector_token_dropout < 1.0):
            raise ValueError(f"sector_token_dropout must be in [0.0, 1.0), got {sector_token_dropout}")
        if not (0.0 <= self.drop_path_rate < 1.0):
            raise ValueError(f"drop_path_rate must be in [0.0, 1.0), got {drop_path_rate}")

        # Linearly schedule stochastic depth (DropPath) from 0 -> drop_path_rate across
        # the 2 fusion layers + 2 graph layers + 2 temporal layers (6 total layers)
        total_dp_layers = num_fusion_layers + num_graph_layers + num_temporal_layers
        if self.drop_path_rate > 0.0 and total_dp_layers > 1:
            all_dp_rates = [
                self.drop_path_rate * float(i) / float(total_dp_layers - 1)
                for i in range(total_dp_layers)
            ]
        else:
            all_dp_rates = [0.0] * total_dp_layers
        self.fusion_drop_path_rates = all_dp_rates[:num_fusion_layers]
        self.graph_drop_path_rates = all_dp_rates[num_fusion_layers:num_fusion_layers + num_graph_layers]
        self.temporal_drop_path_rates = all_dp_rates[num_fusion_layers + num_graph_layers:]

        # 1. Modality Encoders (extract 2D spatial feature representations)
        self.ir_encoder = IREncoder(in_channels=1, d_model=d_model)
        self.pmw_encoder = PMWEncoder(in_channels=4, d_model=d_model)
        self.gprof_encoder = GPROFEncoder(in_channels=4, d_model=d_model)
        self.radar_encoder = RadarEncoder(in_channels=1, d_model=d_model)
        self.env_encoder = EnvEncoder(in_channels=6, d_model=d_model)

        # 2. Polar Sector Pooling: Connects 2D spatial feature maps and validity masks
        # to 16 physical storm-centered polar sectors (4 concentric rings x 4 azimuthal quadrants)
        self.polar_pool = PolarSectorPooling(
            in_channels=d_model, d_model=d_model, min_sector_coverage=min_sector_coverage, bias=False
        )

        # Connect polar pooling module to all modality encoders
        self.ir_encoder.polar_pool = self.polar_pool
        self.pmw_encoder.polar_pool = self.polar_pool
        self.gprof_encoder.polar_pool = self.polar_pool
        self.radar_encoder.polar_pool = self.polar_pool
        self.env_encoder.polar_pool = self.polar_pool

        # 3. Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            d_model=d_model,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
            num_tokens_per_mod=16,
            sector_token_dropout=self.sector_token_dropout,
            drop_path_rates=self.fusion_drop_path_rates,
        )

        # 4. Spatial Polar Node Aggregator: Project 80 multi-modal tokens -> 16 spatial graph nodes
        self.spatial_aggregator = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 5. Sparse Spatial Polar Graph Encoder (Topological message passing over storm circulation)
        self.spatial_graph = SpatialGraphEncoder(
            d_model=d_model,
            num_spatial_nodes=16,
            num_layers=num_graph_layers,
            drop_path_rates=self.graph_drop_path_rates,
        )

        # 6. Causal Continuous-Time Temporal Transformer
        self.temporal_transformer = CausalTemporalTransformer(
            d_model=d_model,
            num_heads=num_fusion_heads,
            num_layers=num_temporal_layers,
            drop_path_rates=self.temporal_drop_path_rates,
        )

        # 7. Multi-Task Heads
        self.task_heads = TaskPredictionHeads(
            d_model=d_model, num_stages=num_stages, head_dropout=self.head_dropout
        )

    def _apply_whole_modality_dropout(self, modality_masks: torch.Tensor) -> torch.Tensor:
        """
        Train-only whole-modality dropout:
        - Each non-IR modality (indices 1..4: PMW, GPROF, Radar, Env) is dropped independently
          per sample with probability `self.modality_dropout_prob` by setting its modality mask to 0.0.
        - Never drops the IR modality (index 0).
        - Never drops all modalities: if a sample had >= 1 valid modality before dropout (e.g. when IR
          was absent in the data) and dropout would zero all remaining modalities, restores one
          randomly chosen originally-present modality for that sample.
        """
        if not self.training or self.modality_dropout_prob <= 0.0:
            return modality_masks

        orig_present = modality_masks > 0.5  # (..., 5)
        keep_non_ir = torch.rand(modality_masks[..., 1:].shape, device=modality_masks.device) >= self.modality_dropout_prob
        out_masks = modality_masks.clone()
        out_masks[..., 1:] = out_masks[..., 1:] * keep_non_ir.to(out_masks.dtype)

        # Guard against all-modality-dropped edge case on samples that had at least 1 valid modality
        had_any = orig_present.any(dim=-1)
        now_any = (out_masks > 0.5).any(dim=-1)
        all_dropped = had_any & (~now_any)
        if all_dropped.any():
            # Restore original mask for any sample where all available modalities were dropped
            out_masks[all_dropped] = modality_masks[all_dropped]
            # If a restored sample had >1 non-IR modality, keep at least its first available modality
            flat_orig = orig_present[all_dropped]  # (K, 5)
            first_valid_idx = flat_orig.float().argmax(dim=-1, keepdim=True)  # (K, 1)
            one_hot_restore = torch.zeros_like(out_masks[all_dropped])
            one_hot_restore.scatter_(-1, first_valid_idx, 1.0)
            out_masks[all_dropped] = one_hot_restore

        return out_masks

    def extract_overpass_features(
        self,
        ir: torch.Tensor,
        pmw: torch.Tensor,
        gprof: torch.Tensor,
        radar: torch.Tensor,
        env: torch.Tensor,
        modality_masks: torch.Tensor,
        channel_masks: Dict[str, torch.Tensor] = None,
        pixel_masks: Dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract spatial polar graph representation for an observation batch.
        Returns: (B, d_model)
        """
        B = modality_masks.size(0)
        modality_masks = self._apply_whole_modality_dropout(modality_masks)

        # Unpack masks: [ir, pmw, gprof, radar, env]
        m_ir = modality_masks[:, 0]
        m_pmw = modality_masks[:, 1]
        m_gprof = modality_masks[:, 2]
        m_radar = modality_masks[:, 3]
        m_env = modality_masks[:, 4]

        ch_pmw = channel_masks.get("pmw") if channel_masks else None
        ch_gprof = channel_masks.get("gprof") if channel_masks else None
        ch_env = channel_masks.get("env") if channel_masks else None

        pix_ir = pixel_masks.get("ir") if pixel_masks else None
        pix_pmw = pixel_masks.get("pmw") if pixel_masks else None
        pix_gprof = pixel_masks.get("gprof") if pixel_masks else None
        pix_radar = pixel_masks.get("radar") if pixel_masks else None
        pix_env = pixel_masks.get("env") if pixel_masks else None

        # 1. Modality Spatial 2D Feature Extraction
        feat_ir = self.ir_encoder(ir, m_ir, pixel_mask=pix_ir, return_spatial_map=True)
        feat_pmw = self.pmw_encoder(pmw, m_pmw, channel_mask=ch_pmw, pixel_mask=pix_pmw, return_spatial_map=True)
        feat_gprof = self.gprof_encoder(gprof, m_gprof, channel_mask=ch_gprof, pixel_mask=pix_gprof, return_spatial_map=True)
        feat_radar = self.radar_encoder(radar, m_radar, voxel_mask=pix_radar, return_spatial_map=True)
        feat_env = self.env_encoder(env, m_env, channel_mask=ch_env, pixel_mask=pix_env, return_spatial_map=True)

        # 2. Polar Sector Pooling: Explicitly invoked in TCGraphFormer
        # Aggregates 2D feature maps into 16 physical storm-centered polar sectors (4 concentric rings x 4 azimuthal quadrants)
        t_ir, val_ir = self.polar_pool(feat_ir, pixel_mask=pix_ir)
        t_pmw, val_pmw = self.polar_pool(feat_pmw, pixel_mask=pix_pmw)
        t_gprof, val_gprof = self.polar_pool(feat_gprof, pixel_mask=pix_gprof)

        pix_radar_2d = None
        if pix_radar is not None:
            if pix_radar.dim() == 5:
                pix_radar_2d = (pix_radar.mean(dim=(1, 2)) > 0.0).float().unsqueeze(1)
            elif pix_radar.dim() == 4:
                pix_radar_2d = pix_radar
            else:
                pix_radar_2d = pix_radar.unsqueeze(1)
        t_radar, val_radar = self.polar_pool(feat_radar, pixel_mask=pix_radar_2d)

        t_env, val_env = self.polar_pool(feat_env, pixel_mask=pix_env)

        # Gate tokens and validity by modality presence (suppress linear projection bias for unobserved tokens)
        m_ir_exp = m_ir.view(B, 1, 1)
        m_pmw_exp = m_pmw.view(B, 1, 1)
        m_gp_exp = m_gprof.view(B, 1, 1)
        m_rad_exp = m_radar.view(B, 1, 1)
        m_env_exp = m_env.view(B, 1, 1)

        val_ir = val_ir * m_ir_exp
        t_ir = t_ir * val_ir
        val_ir = val_ir.squeeze(-1)

        val_pmw = val_pmw * m_pmw_exp
        t_pmw = t_pmw * val_pmw
        val_pmw = val_pmw.squeeze(-1)

        val_gprof = val_gprof * m_gp_exp
        t_gprof = t_gprof * val_gprof
        val_gprof = val_gprof.squeeze(-1)

        val_radar = val_radar * m_rad_exp
        t_radar = t_radar * val_radar
        val_radar = val_radar.squeeze(-1)

        val_env = val_env * m_env_exp
        t_env = t_env * val_env
        val_env = val_env.squeeze(-1)

        # 3. Mask-Aware Cross-Modal Fusion across 80 polar tokens (5 modalities x 16 polar sectors)
        # Passes per-token validity masks as attention padding masks
        fused = self.fusion(
            [t_ir, t_pmw, t_gprof, t_radar, t_env],
            modality_masks=modality_masks,
            token_masks=[val_ir, val_pmw, val_gprof, val_radar, val_env],
        )

        # 4. Reshape and aggregate 5 modalities per polar sector into 16 polar graph nodes:
        # (B, 5, 16, d_model) -> (B, 16, 5 * d_model) -> (B, 16, d_model)
        # Node 0 represents Ring 0 (Eye) Quadrant 0 (NE)
        fused_reshaped = fused.view(B, 5, 16, self.d_model).permute(0, 2, 1, 3).reshape(B, 16, 5 * self.d_model)
        spatial_nodes = self.spatial_aggregator(fused_reshaped)  # (B, 16, d_model)

        # Construct spatial node validity: sector p is valid if ANY modality has valid observation in sector p
        node_validity = (
            (val_ir > 0.5) | (val_pmw > 0.5) | (val_gprof > 0.5) | (val_radar > 0.5) | (val_env > 0.5)
        ).float().unsqueeze(-1)  # (B, 16, 1)

        # Suppress aggregator linear layer & LayerNorm bias revival for entirely unobserved sectors
        spatial_nodes = spatial_nodes * node_validity

        # 5. Spatial Polar Graph Message Passing with node-validity gating
        graph_latent = self.spatial_graph(spatial_nodes, node_validity=node_validity)  # (B, d_model)
        return graph_latent

    def _encode_kinematic_context(self, kinematic_state: torch.Tensor, target_dim: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Deterministic multi-frequency Fourier physical encoding of causal observation-time
        kinematic and geophysical state (speed, heading, Coriolis latitude, hemisphere, distance to land).
        Introduces zero additional state_dict keys so checkpoint compatibility remains 100% strict.
        """
        kin = kinematic_state.to(dtype=torch.float32)
        # Extract 8 normalized physical components: (..., 8)
        f_kin = torch.stack([
            kin[..., 0] / 30.0,   # u_trans (30 km/h scale)
            kin[..., 1] / 30.0,   # v_trans (30 km/h scale)
            kin[..., 2] / 10.0,   # translation speed (10 m/s scale)
            kin[..., 3],          # sin(heading)
            kin[..., 4],          # cos(heading)
            kin[..., 5],          # sin(lat) -> Coriolis parameter f / 2*Omega
            kin[..., 7],          # hemisphere sign (+1 NH, -1 SH)
            kin[..., 8],          # log10(1 + dist_to_land_km) / 3.5
        ], dim=-1)
        freqs = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0], device=kin.device, dtype=torch.float32)
        angles = f_kin.unsqueeze(-1) * freqs  # (..., 8, 8)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(*kin.shape[:-1], 128)
        if target_dim < 128:
            emb = emb[..., :target_dim]
        elif target_dim > 128:
            reps = (target_dim + 127) // 128
            emb = emb.repeat(*([1] * (emb.dim() - 1)), reps)[..., :target_dim]
        return (0.15 * emb).to(dtype=dtype)

    def forward(
        self,
        ir: torch.Tensor,
        pmw: torch.Tensor,
        gprof: torch.Tensor,
        radar: torch.Tensor,
        env: torch.Tensor,
        modality_masks: torch.Tensor,
        delta_times_hours: torch.Tensor = None,
        seq_pad_mask: torch.Tensor = None,
        channel_masks: Dict[str, torch.Tensor] = None,
        pixel_masks: Dict[str, torch.Tensor] = None,
        timestamps: torch.Tensor = None,
        kinematic_state: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Multimodal forward pass supporting both single observations and historical sequences.
        Single observation: ir has shape (B, C, H, W) -> sequence length T=1
        Historical sequence: ir has shape (B, T, C, H, W)
        """
        if ir.dim() == 5:
            # Sequence mode: (B, T, C, H, W)
            B, T = ir.size(0), ir.size(1)

            # Flatten across batch and time for efficient spatial encoding
            ir_flat = ir.reshape(B * T, *ir.shape[2:])
            pmw_flat = pmw.reshape(B * T, *pmw.shape[2:])
            gprof_flat = gprof.reshape(B * T, *gprof.shape[2:])
            radar_flat = radar.reshape(B * T, *radar.shape[2:])
            env_flat = env.reshape(B * T, *env.shape[2:])
            masks_flat = modality_masks.reshape(B * T, *modality_masks.shape[2:])

            ch_flat = None
            if channel_masks:
                ch_flat = {}
                for k, v in channel_masks.items():
                    if v is not None and v.dim() >= 2:
                        ch_flat[k] = v.reshape(B * T, *v.shape[2:])

            pix_flat = None
            if pixel_masks:
                pix_flat = {}
                for k, v in pixel_masks.items():
                    if v is not None and v.dim() >= 3:
                        pix_flat[k] = v.reshape(B * T, *v.shape[2:])

            # Extract spatial graph latent for each timestep: (B * T, d_model)
            latent_flat = self.extract_overpass_features(
                ir_flat, pmw_flat, gprof_flat, radar_flat, env_flat, masks_flat,
                channel_masks=ch_flat, pixel_masks=pix_flat
            )
            seq_latent = latent_flat.view(B, T, self.d_model) # (B, T, d_model)

            anchor_kin = None
            if kinematic_state is not None:
                if kinematic_state.dim() == 3 and kinematic_state.size(1) == T:
                    kin_emb = self._encode_kinematic_context(kinematic_state, self.d_model, seq_latent.dtype)
                    has_any_mod = (modality_masks.sum(dim=-1, keepdim=True) > 0.5).to(seq_latent.dtype)
                    seq_latent = seq_latent + kin_emb * has_any_mod
                elif kinematic_state.dim() == 2:
                    anchor_kin = kinematic_state

            # Delta times in hours
            if delta_times_hours is None:
                delta_times_hours = torch.zeros((B, T), device=ir.device, dtype=torch.float32)

            # Continuous-time causal temporal transformer
            temporal_out = self.temporal_transformer(seq_latent, delta_times_hours, mask=seq_pad_mask) # (B, T, d_model)

            # Anchor selection: select representation corresponding to anchor observation
            # (the last valid, non-padded historical observation in each sequence)
            anchor_latents = []
            anchor_kins = []
            for b in range(B):
                if seq_pad_mask is not None:
                    valid_idx = torch.where(~seq_pad_mask[b])[0]
                    last_idx = valid_idx[-1].item() if len(valid_idx) > 0 else 0
                else:
                    last_idx = T - 1
                anchor_latents.append(temporal_out[b, last_idx, :])
                if kinematic_state is not None and kinematic_state.dim() == 3:
                    anchor_kins.append(kinematic_state[b, last_idx, :])

            final_latent = torch.stack(anchor_latents, dim=0) # (B, d_model)
            if anchor_kins:
                anchor_kin = torch.stack(anchor_kins, dim=0)

            # Multi-task predictions for anchor forecast
            predictions = self.task_heads(final_latent, kinematic_state=anchor_kin)
            predictions["latent"] = final_latent
            predictions["temporal_sequence"] = temporal_out
            return predictions

        else:
            # Single-observation mode: (B, C, H, W)
            B = modality_masks.size(0)
            latent = self.extract_overpass_features(
                ir, pmw, gprof, radar, env, modality_masks,
                channel_masks=channel_masks, pixel_masks=pixel_masks
            )

            anchor_kin = kinematic_state if (kinematic_state is not None and kinematic_state.dim() == 2) else None
            if anchor_kin is not None:
                kin_emb = self._encode_kinematic_context(anchor_kin, self.d_model, latent.dtype)
                has_any_mod = (modality_masks.sum(dim=-1, keepdim=True) > 0.5).to(latent.dtype)
                latent = latent + kin_emb * has_any_mod

            # Pass through temporal model as T=1
            delta_t_hours = torch.zeros((B, 1), device=latent.device, dtype=torch.float32)
            seq_latent = latent.unsqueeze(1) # (B, 1, d_model)
            temporal_out = self.temporal_transformer(seq_latent, delta_t_hours).squeeze(1) # (B, d_model)

            predictions = self.task_heads(temporal_out, kinematic_state=anchor_kin)
            predictions["latent"] = temporal_out
            return predictions

