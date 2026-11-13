"""
Sparse Spatial Polar Graph Encoder for TC-GraphFormer.
Models asymmetric eyewall convection, cyclonic vortex flow, and environmental shear tilt
using Message Passing Neural Networks (MPNN) over cyclone-centric polar topological regions.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessagePassingLayer(nn.Module):
    """
    Message Passing Layer with directional edge attributes and residual connection.
    m_{ij} = MLP([h_i, h_j, e_{ij}])
    h_i' = LayerNorm(h_i + Aggregate(m_{ij}))
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

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        nodes: (B, num_nodes, d_model)
        edge_index: (2, num_edges)
        edge_attr: (num_edges, edge_dim)
        Returns: updated nodes (B, num_nodes, d_model)
        """
        B, N, D = nodes.shape
        src_idx, dst_idx = edge_index[0], edge_index[1]  # each shape (E,)
        E = src_idx.size(0)

        # Gather source and destination node features: (B, E, D)
        h_src = nodes[:, src_idx, :]
        h_dst = nodes[:, dst_idx, :]
        e_attr = edge_attr.unsqueeze(0).repeat(B, 1, 1)  # (B, E, edge_dim)

        # Compute messages: (B, E, D)
        msg_in = torch.cat([h_src, h_dst, e_attr], dim=-1)
        messages = self.msg_mlp(msg_in)

        # Aggregate messages into destination nodes (mean pooling)
        agg = torch.zeros_like(nodes)  # (B, N, D)
        agg = agg.index_add(1, dst_idx, messages)
        # Normalize by degree
        deg = torch.zeros(N, device=nodes.device).index_add(
            0, dst_idx, torch.ones(E, device=nodes.device)
        ).clamp(min=1.0)
        agg = agg / deg.view(1, N, 1)

        # Update node states with residual
        updated = self.update_mlp(torch.cat([nodes, agg], dim=-1))
        return self.norm(nodes + updated)


class PolarSectorPooling(nn.Module):
    """
    Pools a 2D spatial feature map (B, C, H, W) or 3D volume into 16 polar sector tokens
    spanning 4 concentric radial rings and 4 azimuthal quadrants with strict mask awareness.
    Invalid pixels are excluded from the sector mean.
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        domain_radius_km: float = 500.0,
        ring_bounds: Tuple[float, ...] = (0.0, 75.0, 175.0, 300.0, 500.0),
        sector_angles_deg: Tuple[float, ...] = (45.0, 135.0, 225.0, 315.0),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.domain_radius_km = domain_radius_km
        self.ring_bounds = ring_bounds
        self.sector_angles_deg = sector_angles_deg
        self.proj = nn.Linear(in_channels, d_model)
        self._cached_masks: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _get_region_masks(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key in self._cached_masks:
            return self._cached_masks[key]

        # Storm-centered coordinates: Y is North positive, X is East positive
        y = torch.linspace(self.domain_radius_km, -self.domain_radius_km, H, device=device)
        x = torch.linspace(-self.domain_radius_km, self.domain_radius_km, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)
        theta = (torch.atan2(yy, xx) + 2.0 * math.pi) % (2.0 * math.pi)

        # 4 azimuthal sectors:
        # Sector 0: NE [0, pi/2), Sector 1: NW [pi/2, pi), Sector 2: SW [pi, 3pi/2), Sector 3: SE [3pi/2, 2pi)
        sector_bounds = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]

        masks = torch.zeros((16, H, W), dtype=torch.float32, device=device)
        for i in range(4):
            r_m = (r >= self.ring_bounds[i]) & (r < self.ring_bounds[i + 1])
            for j in range(4):
                th_m = (theta >= sector_bounds[j]) & (theta < sector_bounds[j + 1])
                masks[i * 4 + j] = (r_m & th_m).float()

        # Handle very small grids by assigning nearest grid cell if empty
        for p in range(16):
            if masks[p].sum() == 0.0:
                ri = 0.5 * (self.ring_bounds[p // 4] + self.ring_bounds[p // 4 + 1])
                thi = math.radians(self.sector_angles_deg[p % 4])
                xi, yi = ri * math.cos(thi), ri * math.sin(thi)
                dist_grid = (xx - xi) ** 2 + (yy - yi) ** 2
                min_idx = torch.argmin(dist_grid)
                masks[p].view(-1)[min_idx] = 1.0

        self._cached_masks[key] = masks
        return masks

    def forward(
        self,
        x: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)
        pixel_mask: (B, 1, H, W) or (B, C, H, W) or None
        Returns:
            tokens: (B, 16, d_model)
            token_validity: (B, 16, 1)
        """
        B, C, H, W = x.shape
        region_masks = self._get_region_masks(H, W, x.device)

        if pixel_mask is not None:
            if pixel_mask.dim() == 4 and pixel_mask.size(1) > 1:
                pm = (pixel_mask.mean(dim=1, keepdim=True) > 0.0).float()
            else:
                pm = pixel_mask.float()
            # Enforce pixel-level invariance
            x = x * pm
            weights = region_masks.unsqueeze(0).unsqueeze(2) * pm.unsqueeze(1)  # (B, 16, 1, H, W)
        else:
            weights = region_masks.unsqueeze(0).unsqueeze(2).expand(B, 16, 1, H, W)

        weight_sum = weights.sum(dim=(-2, -1))  # (B, 16, 1)
        weighted_x = x.unsqueeze(1) * weights
        token_features = weighted_x.sum(dim=(-2, -1)) / (weight_sum + 1e-6)

        token_validity = (weight_sum > 0.0).float()
        token_features = token_features * token_validity  # (B, 16, C)

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
      
    Azimuthal quadrants:
      - Quadrant 0: NE (45 deg, pi/4)
      - Quadrant 1: NW (135 deg, 3pi/4)
      - Quadrant 2: SW (225 deg, 5pi/4)
      - Quadrant 3: SE (315 deg, 7pi/4)
      
    Directed edges:
      - Azimuthal loops within each ring: j -> (j +/- 1) mod 4 (cyclonic / anticyclonic flow)
      - Radial connections between adjacent rings: i -> i +/- 1 (inflow / outflow)
      - Distance-based connectivity: pairs with physical distance d_uv <= 220 km
      
    8-Dimensional Real Geometric Edge Attributes:
      [dr, dtheta, sin(dtheta), cos(dtheta), r_u / R_max, r_v / R_max, |dr|, d_uv / R_max]
      where R_max = 500 km.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_spatial_nodes: int = 16,
        num_layers: int = 2,
        domain_radius_km: float = 500.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_spatial_nodes
        self.edge_dim = 8
        self.domain_radius_km = domain_radius_km

        # 4 radial rings (centers in km) and 4 azimuthal quadrants (radians)
        radii_km = [37.5, 125.0, 237.5, 400.0]
        angles_deg = [45.0, 135.0, 225.0, 315.0]
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

        # Graph layers
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

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        spatial_tokens: (B, 16, d_model)
        Returns:
            graph_feature: (B, d_model)
        """
        h = spatial_tokens
        for layer in self.layers:
            h = layer(h, self.edge_index, self.edge_attr)

        # Attention pooling over graph nodes
        attn_weights = F.softmax(self.graph_pool(h), dim=1)  # (B, 16, 1)
        graph_embed = torch.sum(h * attn_weights, dim=1)     # (B, d_model)
        return graph_embed


# Alias for backward compatibility
SpatialPolarGraph = SpatialGraphEncoder
