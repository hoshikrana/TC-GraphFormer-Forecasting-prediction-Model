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
from .graph.spatial_graph import SpatialGraphEncoder
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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Modality Encoders
        self.ir_encoder = IREncoder(in_channels=1, d_model=d_model)
        self.pmw_encoder = PMWEncoder(in_channels=4, d_model=d_model)
        self.gprof_encoder = GPROFEncoder(in_channels=4, d_model=d_model)
        self.radar_encoder = RadarEncoder(in_channels=1, d_model=d_model)
        self.env_encoder = EnvEncoder(in_channels=6, d_model=d_model)

        # 2. Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            d_model=d_model,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
            num_tokens_per_mod=16,
        )

        # 3. Spatial Aggregator: Project 80 multi-modal tokens -> 16 spatial graph nodes
        self.spatial_aggregator = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 4. Sparse Spatial Graph Encoder
        self.spatial_graph = SpatialGraphEncoder(
            d_model=d_model,
            num_spatial_nodes=16,
            num_layers=num_graph_layers,
        )

        # 5. Causal Continuous-Time Temporal Transformer
        self.temporal_transformer = CausalTemporalTransformer(
            d_model=d_model,
            num_heads=num_fusion_heads,
            num_layers=num_temporal_layers,
        )

        # 6. Multi-Task Heads
        self.task_heads = TaskPredictionHeads(d_model=d_model, num_stages=num_stages)

    def extract_overpass_features(
        self,
        ir: torch.Tensor,
        pmw: torch.Tensor,
        gprof: torch.Tensor,
        radar: torch.Tensor,
        env: torch.Tensor,
        modality_masks: torch.Tensor,
        channel_masks: Dict[str, torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract spatial graph representation for an observation batch.
        Returns: (B, d_model)
        """
        B = modality_masks.size(0)

        # Unpack masks: [ir, pmw, gprof, radar, env]
        m_ir = modality_masks[:, 0]
        m_pmw = modality_masks[:, 1]
        m_gprof = modality_masks[:, 2]
        m_radar = modality_masks[:, 3]
        m_env = modality_masks[:, 4]

        ch_pmw = channel_masks.get("pmw") if channel_masks else None
        ch_gprof = channel_masks.get("gprof") if channel_masks else None
        ch_env = channel_masks.get("env") if channel_masks else None

        # 1. Modality Token Encoders (each produces B, 16, d_model)
        t_ir = self.ir_encoder(ir, m_ir)
        t_pmw = self.pmw_encoder(pmw, m_pmw, channel_mask=ch_pmw)
        t_gprof = self.gprof_encoder(gprof, m_gprof, channel_mask=ch_gprof)
        t_radar = self.radar_encoder(radar, m_radar)
        t_env = self.env_encoder(env, m_env, channel_mask=ch_env)

        # 2. Mask-Aware Cross-Modal Fusion
        # fused_tokens: (B, 80, d_model)
        fused = self.fusion([t_ir, t_pmw, t_gprof, t_radar, t_env], modality_masks)

        # 3. Reshape and project to 16 spatial nodes: (B, 16, 5 * d_model) -> (B, 16, d_model)
        # 80 tokens = 5 modalities x 16 patches
        fused_reshaped = fused.view(B, 5, 16, self.d_model).permute(0, 2, 1, 3).reshape(B, 16, 5 * self.d_model)
        spatial_nodes = self.spatial_aggregator(fused_reshaped) # (B, 16, d_model)

        # 4. Spatial Graph Message Passing
        graph_latent = self.spatial_graph(spatial_nodes) # (B, d_model)
        return graph_latent

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
        timestamps: torch.Tensor = None,
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

            # Extract spatial graph latent for each timestep: (B * T, d_model)
            latent_flat = self.extract_overpass_features(
                ir_flat, pmw_flat, gprof_flat, radar_flat, env_flat, masks_flat, channel_masks=ch_flat
            )
            seq_latent = latent_flat.view(B, T, self.d_model) # (B, T, d_model)

            # Delta times in hours
            if delta_times_hours is None:
                delta_times_hours = torch.zeros((B, T), device=ir.device, dtype=torch.float32)

            # Continuous-time causal temporal transformer
            temporal_out = self.temporal_transformer(seq_latent, delta_times_hours, mask=seq_pad_mask) # (B, T, d_model)

            # Anchor selection: select representation corresponding to anchor observation
            # (the last valid, non-padded historical observation in each sequence)
            anchor_latents = []
            for b in range(B):
                if seq_pad_mask is not None:
                    valid_idx = torch.where(~seq_pad_mask[b])[0]
                    last_idx = valid_idx[-1].item() if len(valid_idx) > 0 else 0
                else:
                    last_idx = T - 1
                anchor_latents.append(temporal_out[b, last_idx, :])

            final_latent = torch.stack(anchor_latents, dim=0) # (B, d_model)

            # Multi-task predictions for anchor forecast
            predictions = self.task_heads(final_latent)
            predictions["latent"] = final_latent
            predictions["temporal_sequence"] = temporal_out
            return predictions

        else:
            # Single-observation mode: (B, C, H, W)
            B = modality_masks.size(0)
            latent = self.extract_overpass_features(
                ir, pmw, gprof, radar, env, modality_masks, channel_masks=channel_masks
            )

            # Pass through temporal model as T=1
            delta_t_hours = torch.zeros((B, 1), device=latent.device, dtype=torch.float32)
            seq_latent = latent.unsqueeze(1) # (B, 1, d_model)
            temporal_out = self.temporal_transformer(seq_latent, delta_t_hours).squeeze(1) # (B, d_model)

            predictions = self.task_heads(temporal_out)
            predictions["latent"] = temporal_out
            return predictions

