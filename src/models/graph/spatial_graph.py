"""
Sparse Spatial Graph Encoder for TC-GraphFormer.
Models asymmetric eyewall convection, cyclonic vortex flow, and environmental shear tilt
using Message Passing Neural Networks (MPNN) over cyclone-centric topological regions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphMessagePassingLayer(nn.Module):
    """
    Message Passing Layer with directional edge attributes and residual connection.
    m_{ij} = MLP([h_i, h_j, e_{ij}])
    h_i' = LayerNorm(h_i + Aggregate(m_{ij}))
    """

    def __init__(self, d_model: int, edge_dim: int = 16):
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
        src_idx, dst_idx = edge_index[0], edge_index[1] # each shape (E,)
        E = src_idx.size(0)

        # Gather source and destination node features: (B, E, D)
        h_src = nodes[:, src_idx, :]
        h_dst = nodes[:, dst_idx, :]
        e_attr = edge_attr.unsqueeze(0).repeat(B, 1, 1) # (B, E, edge_dim)

        # Compute messages: (B, E, D)
        msg_in = torch.cat([h_src, h_dst, e_attr], dim=-1)
        messages = self.msg_mlp(msg_in)

        # Aggregate messages into destination nodes (mean pooling)
        # Using index_add for efficient batch scatter
        agg = torch.zeros_like(nodes) # (B, N, D)
        # Reshape for scatter_add: flatten batch dimension
        agg = agg.index_add(1, dst_idx, messages)
        # Normalize by degree
        deg = torch.zeros(N, device=nodes.device).index_add(0, dst_idx, torch.ones(E, device=nodes.device)).clamp(min=1.0)
        agg = agg / deg.view(1, N, 1)

        # Update node states with residual
        updated = self.update_mlp(torch.cat([nodes, agg], dim=-1))
        return self.norm(nodes + updated)


class SpatialGraphEncoder(nn.Module):
    """
    Constructs a topological cyclone graph (Core, Eyewall, Rainbands, Quadrants)
    and executes message passing across spatial nodes.
    """

    def __init__(self, d_model: int = 128, num_spatial_nodes: int = 16, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_nodes = num_spatial_nodes
        self.edge_dim = 16

        # Build cyclone-centric spatial graph topology (4x4 grid patches)
        edge_list = []
        edge_attrs = []
        grid_dim = int(math.isqrt(num_spatial_nodes)) # 4

        # Center of the 4x4 grid is (1.5, 1.5)
        cx, cy = (grid_dim - 1) / 2.0, (grid_dim - 1) / 2.0

        for i in range(grid_dim):
            for j in range(grid_dim):
                u = i * grid_dim + j
                r_u = math.sqrt((i - cx)**2 + (j - cy)**2)
                theta_u = math.atan2(j - cy, i - cx)

                # Connect to 4-way neighbors and cyclonic neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_dim and 0 <= nj < grid_dim:
                        v = ni * grid_dim + nj
                        r_v = math.sqrt((ni - cx)**2 + (nj - cy)**2)
                        theta_v = math.atan2(nj - cy, ni - cx)

                        edge_list.append((u, v))
                        dr = r_v - r_u
                        dtheta = theta_v - theta_u
                        # Encode relative radial and azimuthal distance
                        edge_attrs.append([
                            dr, dtheta, math.sin(dtheta), math.cos(dtheta),
                            r_u, r_v, abs(dr), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        attn_weights = F.softmax(self.graph_pool(h), dim=1) # (B, 16, 1)
        graph_embed = torch.sum(h * attn_weights, dim=1)    # (B, d_model)
        return graph_embed

