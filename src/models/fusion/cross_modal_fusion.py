"""
Mask-Aware Cross-Modal Fusion Transformer for TC-GraphFormer.
Fuses heterogeneous observations (IR, PMW, GPROF, Radar, Environmental)
using masked multi-head attention with modality-specific type embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class CrossModalFusion(nn.Module):
    """
    Fuses tokens from available satellite and environmental modalities.
    Missing modalities are strictly masked out from key/value attention.
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, num_layers: int = 2, num_tokens_per_mod: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_tokens_per_mod = num_tokens_per_mod
        self.num_modalities = 5 # IR, PMW, GPROF, Radar, Env

        # Modality type embeddings (5 modalities)
        self.modality_embed = nn.Embedding(5, d_model)
        
        # Spatial position embeddings (16 tokens per modality)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_tokens_per_mod, d_model))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Transformer Encoder layers for cross-modal interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_tokens: List[torch.Tensor], modality_masks: torch.Tensor) -> torch.Tensor:
        """
        modality_tokens: list of 5 tensors, each (B, 16, d_model)
        modality_masks: (B, 5) where 1.0 = present, 0.0 = absent
        Returns:
            fused_tokens: (B, 5 * 16, d_model)
        """
        B = modality_masks.size(0)
        embedded_tokens = []
        key_padding_mask_list = []

        for m_idx, tokens in enumerate(modality_tokens):
            # Add spatial position embedding and modality embedding
            mod_emb = self.modality_embed(torch.tensor(m_idx, device=tokens.device)).view(1, 1, self.d_model)
            t = tokens + self.spatial_pos_embed + mod_emb
            embedded_tokens.append(t)

            # Mask: True means IGNORED / PADDED in PyTorch nn.MultiheadAttention
            is_present = modality_masks[:, m_idx] > 0.5 # (B,)
            is_absent = ~is_present # (B,)
            # Repeat mask across the 16 tokens of this modality
            pad_mask = is_absent.unsqueeze(1).repeat(1, self.num_tokens_per_mod) # (B, 16)
            key_padding_mask_list.append(pad_mask)

        # Concatenate along sequence length dimension
        seq = torch.cat(embedded_tokens, dim=1) # (B, 80, d_model)
        key_padding_mask = torch.cat(key_padding_mask_list, dim=1) # (B, 80)

        # If an entire sample has no modalities present (edge case), unmask first token to prevent NaN
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked, 0] = False

        # Apply Cross-Modal Transformer
        fused = self.transformer(seq, src_key_padding_mask=key_padding_mask)
        fused = self.norm(fused)
        return fused

