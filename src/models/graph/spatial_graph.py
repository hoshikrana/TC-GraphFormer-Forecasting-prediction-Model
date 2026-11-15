"""
Sparse Spatial Polar Graph Encoder for TC-GraphFormer.
Models asymmetric eyewall convection, cyclonic vortex flow, and environmental shear tilt
using Message Passing Neural Networks (MPNN) over cyclone-centric polar topological regions.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid_geometry import (
    GridGeometry,
    GLOBAL_GRID_GEOMETRY,
    CANONICAL_DOMAIN_RADIUS_KM,
    CANONICAL_RING_BOUNDS_KM,
    CANONICAL_RING_CENTERS_KM,
    CANONICAL_SECTOR_AREAS_KM2,
    CANONICAL_SECTOR_ANGLES_DEG,
)


class GraphMessagePassingLayer(nn.Module):
    """
    Message Passing Layer with directional edge attributes, residual connection,
    and strict node-validity masking.
    m_{ij} = MLP([h_i, h_j, e_{ij}]) * v_i
    h_i' = LayerNorm(h_i + Aggregate(m_{ij})) * v_i
    """

    def __init__(self, d_model: int, edge_dim: int = 8):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(d_model * 2 + edge_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        nodes: (B, num_nodes, d_model)
        edge_index: (2, num_edges)
        edge_attr: (num_edges, edge_dim)
        node_validity: (B, num_nodes, 1) or None
        Returns: updated nodes (B, num_nodes, d_model)
        """
        B, N, D = nodes.shape
        src_idx, dst_idx = edge_index[0], edge_index[1]  # each shape (E,)
        E = src_idx.size(0)

        # Gate input nodes by validity
        if node_validity is not None:
            if node_validity.dim() == 2:
                node_validity = node_validity.unsqueeze(-1)
            nodes = nodes * node_validity

        # Gather source and destination node features: (B, E, D)
        h_src = nodes[:, src_idx, :]
        h_dst = nodes[:, dst_idx, :]
        e_attr = edge_attr.unsqueeze(0).repeat(B, 1, 1)  # (B, E, edge_dim)

        # Compute messages: (B, E, D)
        msg_in = torch.cat([h_src, h_dst, e_attr], dim=-1)
        messages = self.msg_mlp(msg_in)

        # Gate messages by source validity: invalid source cannot broadcast
        if node_validity is not None:
            v_src = node_validity[:, src_idx, :]  # (B, E, 1)
            messages = messages * v_src

        # Aggregate messages into destination nodes (mean pooling over valid sources)
        agg = torch.zeros_like(nodes)  # (B, N, D)
        agg = agg.index_add(1, dst_idx, messages)

        if node_validity is not None:
            # Count valid incoming messages per destination node
            v_src_weights = node_validity[:, src_idx, 0]  # (B, E)
            deg = torch.zeros((B, N), device=nodes.device, dtype=torch.float32)
            deg = deg.index_add(1, dst_idx, v_src_weights)
            deg = deg.clamp(min=1.0).unsqueeze(-1)  # (B, N, 1)
            agg = agg / deg
        else:
            deg = torch.zeros(N, device=nodes.device).index_add(
                0, dst_idx, torch.ones(E, device=nodes.device)
            ).clamp(min=1.0)
            agg = agg / deg.view(1, N, 1)

        # Update node states with residual
        updated = self.update_mlp(torch.cat([nodes, agg], dim=-1))
        out = self.norm(nodes + updated)

        # Gate destination node states: invalid node remains strictly zero
        if node_validity is not None:
            out = out * node_validity

        return out


class PolarSectorPooling(nn.Module):
    """
    Pools a 2D spatial feature map (B, C, H, W) or 3D volume into 16 polar sector tokens
    spanning 4 concentric radial rings and 4 azimuthal quadrants with strict mask awareness.
    
    Enforces a configurable minimum physical sector coverage threshold:
      coverage = valid_observed_area_in_sector / total_geometric_area_of_sector
      token_valid = (coverage >= min_sector_coverage)
    Invalid tokens are zeroed out before and after projection to eliminate linear bias revival.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        domain_radius_km: float = CANONICAL_DOMAIN_RADIUS_KM,
        ring_bounds: Tuple[float, ...] = CANONICAL_RING_BOUNDS_KM,
        sector_angles_deg: Tuple[float, ...] = CANONICAL_SECTOR_ANGLES_DEG,
        min_sector_coverage: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.domain_radius_km = domain_radius_km
        self.ring_bounds = ring_bounds
        self.sector_angles_deg = sector_angles_deg
        self.min_sector_coverage = float(min_sector_coverage)
        self.grid_geom = GridGeometry(domain_radius_km=domain_radius_km)
        self.proj = nn.Linear(in_channels, d_model, bias=bias)

    def _get_region_masks(self, H: int, W: int, device: torch.device, is_feature_grid: bool = True) -> torch.Tensor:
        return self.grid_geom.get_sector_masks_torch(H, W, device, is_feature_grid=is_feature_grid)

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)  (typically 32x32 feature map)
        pixel_mask: (B, 1, H_orig, W_orig) or (B, C, H_orig, W_orig) or None
        Returns:
            tokens: (B, 16, d_model)  -- strictly zeroed for invalid sectors
            token_validity: (B, 16, 1) -- 1.0 for valid, 0.0 for invalid
        """
        B, C, H, W = x.shape
        is_feature_grid = (H == 32 and W == 32)
        region_masks = self._get_region_masks(H, W, x.device, is_feature_grid=is_feature_grid)  # (16, H, W)

        if pixel_mask is not None:
            if pixel_mask.dim() == 2:
                pm = pixel_mask.unsqueeze(0).unsqueeze(0).float()
            elif pixel_mask.dim() == 3:
                pm = pixel_mask.unsqueeze(1).float()
            elif pixel_mask.dim() == 4:
                if pixel_mask.size(1) > 1:
                    pm = (pixel_mask.mean(dim=1, keepdim=True) > 0.0).float()
                else:
                    pm = pixel_mask.float()
            elif pixel_mask.dim() == 5:
                pm = (pixel_mask.mean(dim=(1, 2), keepdim=True) > 0.0).float().squeeze(2)
            else:
                pm = pixel_mask.float()

            H_orig, W_orig = pm.shape[-2:]
            # Compute fractional physical area coverage at native mask resolution
            # Native cell area (km^2)
            dx_orig = (2.0 * self.domain_radius_km) / float(W_orig)
            dy_orig = (2.0 * self.domain_radius_km) / float(H_orig)
            cell_area_km2 = dx_orig * dy_orig

            orig_reg_masks = self._get_region_masks(H_orig, W_orig, x.device, is_feature_grid=False)  # (16, H_orig, W_orig)
            # Valid observed area per sector: (B, 16, 1)
            valid_pixels_per_sector = (orig_reg_masks.unsqueeze(0) * pm).sum(dim=(-2, -1)).unsqueeze(-1)
            valid_area_km2 = valid_pixels_per_sector * cell_area_km2

            # Geometric total area of each sector (km^2)
            sector_areas = torch.tensor(
                [self.grid_geom.get_sector_area_km2(p) for p in range(16)],
                dtype=torch.float32,
                device=x.device,
            ).view(1, 16, 1)

            coverage_ratio = valid_area_km2 / (sector_areas + 1e-6)  # (B, 16, 1)
            token_validity = (coverage_ratio >= self.min_sector_coverage).float()

            # Downsample pixel mask to feature resolution for feature weighted averaging
            if (H_orig, W_orig) != (H, W):
                pm_down = (F.interpolate(pm, size=(H, W), mode="nearest") > 0.0).float()
            else:
                pm_down = pm

            # Enforce pixel-level invariance
            x = x * pm_down
            weights = region_masks.unsqueeze(0).unsqueeze(2) * pm_down.unsqueeze(1)  # (B, 16, 1, H, W)
        else:
            # Unmasked input: all sectors are geometrically valid with 100% coverage
            token_validity = torch.ones((B, 16, 1), dtype=torch.float32, device=x.device)
            weights = region_masks.unsqueeze(0).unsqueeze(2).expand(B, 16, 1, H, W)

        weight_sum = weights.sum(dim=(-2, -1))  # (B, 16, 1)
        weighted_x = x.unsqueeze(1) * weights
        token_features = weighted_x.sum(dim=(-2, -1)) / (weight_sum + 1e-6)

        # Zero out invalid sector feature vectors before projection
        token_features = token_features * token_validity  # (B, 16, C)

        # Linear projection with post-projection zeroing to eliminate bias revival
        tokens = self.proj(token_features) * token_validity  # (B, 16, d_model)
        return tokens, token_validity


class SpatialGraphEncoder(nn.Module):
    """
    Constructs a true topological storm-centered polar cyclone graph
    with 16 nodes partitioning the storm circulation into 4 concentric radial rings
    and 4 azimuthal quadrants.
    
    Concentric rings:
      - Ring 0: Inner Core / Eye (0 to 75 km, nominal r_0 = 37.5 km)
      - Ring 1: Eyewall / Inner Rainband (75 to 175 km, nominal r_1 = 125.0 km)
      - Ring 2: Outer Rainband (175 to 300 km, nominal r_2 = 237.5 km)
      - Ring 3: Synoptic Outflow / Environment (300 to 500 km, nominal r_3 = 400.0 km)
      
    Azimuthal quadrants (Mathematical Polar Angle):
      - Sector 0: NE (45 deg, pi/4)
      - Sector 1: NW (135 deg, 3pi/4)
      - Sector 2: SW (225 deg, 5pi/4)
      - Sector 3: SE (315 deg, 7pi/4)
      
    8-Dimensional Real Geometric Edge Attributes:
      [dr, dtheta, sin(dtheta), cos(dtheta), r_u / R_max, r_v / R_max, |dr|, d_uv / R_max]
      where R_max = 500 km.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_spatial_nodes: int = 16,
        num_layers: int = 2,
        domain_radius_km: float = CANONICAL_DOMAIN_RADIUS_KM,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_spatial_nodes
        self.edge_dim = 8
        self.domain_radius_km = domain_radius_km

        radii_km = list(CANONICAL_RING_CENTERS_KM)
        angles_deg = list(CANONICAL_SECTOR_ANGLES_DEG)
        angles_rad = [math.radians(a) for a in angles_deg]

        node_coords = []
        node_polar = []
        for r in radii_km:
            for th in angles_rad:
                x_km = r * math.cos(th)
                y_km = r * math.sin(th)
                node_coords.append((x_km, y_km))
                node_polar.append((r, th))

        self.register_buffer("node_coords_km", torch.tensor(node_coords, dtype=torch.float32))
        self.register_buffer("node_polar_coords", torch.tensor(node_polar, dtype=torch.float32))

        # Build true polar graph topology
        edges = set()
        # 1. Azimuthal loops within each ring
        for i in range(4):
            for j in range(4):
                u = i * 4 + j
                v_ccw = i * 4 + ((j + 1) % 4)
                v_cw = i * 4 + ((j - 1) % 4)
                edges.add((u, v_ccw))
                edges.add((u, v_cw))

        # 2. Radial connections between adjacent rings in the same quadrant
        for j in range(4):
            for i in range(3):
                u = i * 4 + j
                v = (i + 1) * 4 + j
                edges.add((u, v))
                edges.add((v, u))

        # 3. Distance threshold d_uv <= 220 km (connects adjacent diagonals and nearby sectors)
        for u in range(16):
            xu, yu = node_coords[u]
            for v in range(16):
                if u == v:
                    continue
                xv, yv = node_coords[v]
                d = math.sqrt((xv - xu) ** 2 + (yv - yu) ** 2)
                if d <= 220.0:
                    edges.add((u, v))

        edge_list = sorted(list(edges))
        edge_attrs = []
        R_max = max(1.0, domain_radius_km)

        for u, v in edge_list:
            xu, yu = node_coords[u]
            xv, yv = node_coords[v]
            ru, thu = node_polar[u]
            rv, thv = node_polar[v]

            dr = (rv - ru) / R_max
            dtheta = (thv - thu + math.pi) % (2.0 * math.pi) - math.pi
            d_uv = math.sqrt((xv - xu) ** 2 + (yv - yu) ** 2) / R_max

            edge_attrs.append([
                dr,
                dtheta,
                math.sin(dtheta),
                math.cos(dtheta),
                ru / R_max,
                rv / R_max,
                abs(dr),
                d_uv,
            ])

        self.register_buffer("edge_index", torch.tensor(edge_list, dtype=torch.long).t().contiguous())
        self.register_buffer("edge_attr", torch.tensor(edge_attrs, dtype=torch.float32))

        # Graph message passing layers
        self.layers = nn.ModuleList([
            GraphMessagePassingLayer(d_model=d_model, edge_dim=self.edge_dim)
            for _ in range(num_layers)
        ])

        # Graph pooling into single graph representation vector
        self.graph_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        node_validity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        spatial_tokens: (B, 16, d_model)
        node_validity: (B, 16, 1) or (B, 16) or None
        Returns:
            graph_feature: (B, d_model) -- strictly zeroed and finite for all-invalid graphs
        """
        B, N, D = spatial_tokens.shape
        h = spatial_tokens

        if node_validity is not None:
            if node_validity.dim() == 2:
                node_validity = node_validity.unsqueeze(-1)
            h = h * node_validity

        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_attr, node_validity=node_validity)

        # Attention pooling over graph nodes with masked validity
        scores = self.graph_pool(h)  # (B, 16, 1)

        if node_validity is not None:
            # Check for all-invalid samples to prevent NaNs from softmax over -inf
            valid_counts = node_validity.sum(dim=1)  # (B, 1)
            all_invalid = (valid_counts < 0.5)        # (B, 1)

            # Mask out invalid nodes with large negative value
            scores = scores.masked_fill(~(node_validity > 0.5), -1e9)
            attn_weights = F.softmax(scores, dim=1) * node_validity  # (B, 16, 1)
            attn_norm = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attn_weights = attn_weights / attn_norm

            graph_embed = torch.sum(h * attn_weights, dim=1)  # (B, d_model)
            # Explicitly zero out all-invalid representations to prevent artificial bias output
            graph_embed = torch.where(all_invalid, torch.zeros_like(graph_embed), graph_embed)
        else:
            attn_weights = F.softmax(scores, dim=1)
            graph_embed = torch.sum(h * attn_weights, dim=1)

        return graph_embed


# Alias for backward compatibility
SpatialPolarGraph = SpatialGraphEncoder
