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
        num_stages: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Modality Encoders
        self.ir_encoder = IREncoder(in_channels=1, d_model=d_model)
        self.pmw_encoder = PMWEncoder(in_channels=4, d_model=d_model)
        self.gprof_encoder = GPROFEncoder(in_channels=1, d_model=d_model)
        self.radar_encoder = RadarEncoder(in_channels=1, d_model=d_model)
        self.env_encoder = EnvEncoder(in_channels=1, d_model=d_model)

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

        # 1. Modality Token Encoders (each produces B, 16, d_model)
        t_ir = self.ir_encoder(ir, m_ir)
        t_pmw = self.pmw_encoder(pmw, m_pmw)
        t_gprof = self.gprof_encoder(gprof, m_gprof)
        t_radar = self.radar_encoder(radar, m_radar)
        t_env = self.env_encoder(env, m_env)

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
        timestamps: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-step or batched multi-modal forward pass.
        """
        B = modality_masks.size(0)

        # Extract graph representation for current observations: (B, d_model)
        latent = self.extract_overpass_features(ir, pmw, gprof, radar, env, modality_masks)

        # Pass through temporal model (if sequence or delta t available)
        # For batch of single overpasses, delta_t is 0
        delta_t_hours = torch.zeros((B, 1), device=latent.device, dtype=torch.float32)
        seq_latent = latent.unsqueeze(1) # (B, 1, d_model)
        temporal_out = self.temporal_transformer(seq_latent, delta_t_hours).squeeze(1) # (B, d_model)

        # Predict multi-task outputs
        predictions = self.task_heads(temporal_out)
        predictions["latent"] = temporal_out
        return predictions

