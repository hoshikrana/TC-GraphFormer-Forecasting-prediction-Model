"""
Mask-Aware Cross-Modal Fusion Transformer for TC-GraphFormer.
Fuses heterogeneous observations (IR, PMW, GPROF, Radar, Environmental)
using masked multi-head attention with modality-specific type embeddings,
optional train-only sector-token dropout, and optional stochastic depth (DropPath).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic depth (DropPath) per sample along batch dimension 0.
    Strictly a no-op when drop_prob <= 0.0 or not training.
    """
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Module wrapper for stochastic depth (DropPath) with zero learnable parameters."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class CrossModalFusion(nn.Module):
    """
    Fuses tokens from available satellite and environmental modalities.
    Missing modalities are strictly masked out from key/value attention.
    Supports optional train-only sector-token dropout (`sector_token_dropout`)
    and per-layer stochastic depth (`drop_path_rates`).
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_tokens_per_mod: int = 16,
        sector_token_dropout: float = 0.0,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_tokens_per_mod = num_tokens_per_mod
        self.num_modalities = 5  # IR, PMW, GPROF, Radar, Env
        self.sector_token_dropout = float(sector_token_dropout)
        if not (0.0 <= self.sector_token_dropout < 1.0):
            raise ValueError(f"sector_token_dropout must be in [0.0, 1.0), got {sector_token_dropout}")

        if drop_path_rates is None:
            self.drop_path_rates = [0.0] * num_layers
        else:
            if len(drop_path_rates) != num_layers:
                raise ValueError(f"Expected {num_layers} drop_path_rates, got {len(drop_path_rates)}")
            self.drop_path_rates = [float(r) for r in drop_path_rates]

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def _forward_layer_with_drop_path(
        self,
        layer: nn.TransformerEncoderLayer,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        drop_prob: float,
    ) -> torch.Tensor:
        if drop_prob <= 0.0 or not self.training:
            return layer(x, src_key_padding_mask=src_key_padding_mask)
        # Pre-LN TransformerEncoderLayer with DropPath on both residual branches
        sa_out = layer._sa_block(layer.norm1(x), None, src_key_padding_mask)
        x = x + drop_path(sa_out, drop_prob, self.training)
        ff_out = layer._ff_block(layer.norm2(x))
        x = x + drop_path(ff_out, drop_prob, self.training)
        return x

    def forward(
        self,
        modality_tokens: List[torch.Tensor],
        modality_masks: torch.Tensor,
        token_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        modality_tokens: list of 5 tensors, each (B, 16, d_model)
        modality_masks: (B, 5) where 1.0 = present, 0.0 = absent
        token_masks: optional list of 5 tensors, each (B, 16) where 1.0 = valid, 0.0 = invalid
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
            is_present = modality_masks[:, m_idx] > 0.5  # (B,)
            if token_masks is not None and m_idx < len(token_masks) and token_masks[m_idx] is not None:
                # Per-token validity: token is attended strictly if modality is present AND token is valid
                tok_valid = is_present.unsqueeze(1) & (token_masks[m_idx] > 0.5)  # (B, 16)
                pad_mask = ~tok_valid  # (B, 16)
            else:
                is_absent = ~is_present  # (B,)
                pad_mask = is_absent.unsqueeze(1).repeat(1, self.num_tokens_per_mod)  # (B, 16)
            key_padding_mask_list.append(pad_mask)

        # Concatenate along sequence length dimension
        seq = torch.cat(embedded_tokens, dim=1)  # (B, 80, d_model)
        key_padding_mask = torch.cat(key_padding_mask_list, dim=1)  # (B, 80)

        # Optional train-only sector-token dropout across the 80 tokens via key_padding_mask
        if self.training and self.sector_token_dropout > 0.0:
            drop_mask = torch.rand(key_padding_mask.shape, device=seq.device) < self.sector_token_dropout
            candidate_mask = key_padding_mask | drop_mask
            # Guarantee we never mask out all valid tokens of a sample that had at least 1 valid token
            had_valid = ~key_padding_mask.all(dim=1)
            now_all_masked = candidate_mask.all(dim=1)
            revert_rows = had_valid & now_all_masked
            if revert_rows.any():
                candidate_mask = candidate_mask.clone()
                candidate_mask[revert_rows] = key_padding_mask[revert_rows]
            key_padding_mask = candidate_mask

        # If an entire sample has no valid tokens (edge case), unmask first token to prevent NaN
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        # Apply Cross-Modal Transformer (with optional stochastic depth per layer)
        if any(r > 0.0 for r in self.drop_path_rates) and self.training:
            fused = seq
            for layer_idx, layer in enumerate(self.transformer.layers):
                dp_rate = self.drop_path_rates[layer_idx] if layer_idx < len(self.drop_path_rates) else 0.0
                fused = self._forward_layer_with_drop_path(layer, fused, key_padding_mask, dp_rate)
            if self.transformer.norm is not None:
                fused = self.transformer.norm(fused)
        else:
            fused = self.transformer(seq, src_key_padding_mask=key_padding_mask)

        fused = self.norm(fused)

        # Strictly zero out invalid/padded tokens so unobserved sectors never leak into spatial_aggregator
        valid_tok_mask = (~key_padding_mask).unsqueeze(-1).float()  # (B, 80, 1)
        if all_masked.any():
            valid_tok_mask[all_masked] = 0.0
        fused = fused * valid_tok_mask

        return fused
