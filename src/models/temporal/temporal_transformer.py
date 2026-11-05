"""
Causal Continuous-Time Temporal Transformer for TC-GraphFormer.
Processes irregular satellite overpass sequences with continuous time-delta embeddings
and strict causal attention masking (t_j <= t_i).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousTimeEmbedding(nn.Module):
    """
    Fourier continuous-time positional encoding for irregular overpass gaps (hours).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1)))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_t_hours: torch.Tensor) -> torch.Tensor:
        """
        delta_t_hours: (B, T) or (B, T, 1)
        Returns: (B, T, d_model)
        """
        if delta_t_hours.dim() == 2:
            delta_t_hours = delta_t_hours.unsqueeze(-1) # (B, T, 1)

        angles = delta_t_hours * self.freqs.view(1, 1, -1) # (B, T, half_dim)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # (B, T, d_model)
        return self.proj(emb)


class CausalTemporalTransformer(nn.Module):
    """
    Applies multi-layer causal continuous-time self-attention over storm overpass history.
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, num_layers: int = 2, max_seq_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.time_embed = ContinuousTimeEmbedding(d_model)

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

    def forward(
        self,
        features: torch.Tensor,
        delta_times_hours: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        features: (B, T, d_model)
        delta_times_hours: (B, T) relative to sequence start or previous overpass
        mask: (B, T) boolean padding mask (True where padded / invalid)
        Returns:
            causal_features: (B, T, d_model)
        """
        B, T, D = features.shape

        # Add continuous time encoding
        time_pe = self.time_embed(delta_times_hours) # (B, T, d_model)
        h = features + time_pe

        # Causal attention mask: lower triangular matrix (T, T)
        # In PyTorch nn.TransformerEncoder, True means masked / forbidden
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=features.device), diagonal=1)

        out = self.transformer(h, mask=causal_mask, src_key_padding_mask=mask)
        return self.norm(out)

