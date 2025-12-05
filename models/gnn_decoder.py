"""
GNN-based Decoder for reconstructing flow fields from latent representations.
Implements Equation 10 from the reference: Up-sampling via GNN.
"""

import torch
import torch.nn as nn
from .gnn_layer import GNNLayer
try:
    from data.interpolation import interpolate_to_coordinates
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.interpolation import interpolate_to_coordinates


class GNNDecoder(nn.Module):
    """
    Graph Neural Network Decoder following Equation 10.
    
    Process:
    1. Interpolation: U0 = I_{X1}^{X2}(Z) from reduced to full coordinates
    2. mlp0 + lnm0: Transform node features U0 -> U1
    3. mlp1 + lnm1: Transform edge features E0 -> E1
    4. GNN layers: G2 = gnn_Ng(G1)
    5. mlp2: Final output U = [mlp2(u^2_1), ..., mlp2(u^2_{|X1|})]
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 3,  # u, v, p
        num_layers: int = 3,
        use_edge_features: bool = True,
        use_position_encoding: bool = True
    ):
        super(GNNDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        self.use_position_encoding = use_position_encoding
        
        # Edge feature dimension (if used)
        if use_edge_features:
            self.edge_dim = 2  # Relative displacement or distance
        else:
            self.edge_dim = 0
        
        # mlp0 + lnm0: Initial node feature transformation (after interpolation)
        self.mlp0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lnm0 = nn.LayerNorm(hidden_dim)
        
        # mlp1 + lnm1: Initial edge feature transformation (if use_edge_features)
        if use_edge_features:
            self.mlp1 = nn.Sequential(
                nn.Linear(self.edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.edge_dim)
            )
            self.lnm1 = nn.LayerNorm(self.edge_dim)
        else:
            self.mlp1 = None
            self.lnm1 = None
        
        # GNN layers: gnn_Ng
        self.gnn_layers = nn.ModuleList([
            GNNLayer(
                node_dim=hidden_dim,
                edge_dim=self.edge_dim if use_edge_features else None,
                hidden_dim=hidden_dim
            )
            for _ in range(num_layers)
        ])
        
        # mlp2: Final output projection (no LayerNorm as per Equation 10)
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Latent projection to hidden dimension (for interpolation)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Position projection (optional, for conditioning on node positions)
        if use_position_encoding:
            self.pos_proj = nn.Linear(2, hidden_dim)
        else:
            self.pos_proj = None
    
    def _compute_edge_features(self, node_positions, edge_index):
        """Compute edge features from node positions."""
        row, col = edge_index
        edge_attr = node_positions[col] - node_positions[row]  # Relative displacement
        return edge_attr
    
    def forward(self, z, edge_index, node_positions=None, num_nodes=None):
        """
        Forward pass through the decoder following Equation 10.
        
        Args:
            z: Latent representation [batch_size, latent_dim] or [1, latent_dim]
            edge_index: Graph connectivity [2, num_edges]
            node_positions: Node coordinates X1 [num_nodes, 2] (required for interpolation)
            num_nodes: Number of nodes in the graph (if z is batch-level)
        
        Returns:
            x_recon: Reconstructed flow field [num_nodes, output_dim]
        """
        # Determine number of nodes
        if num_nodes is None:
            if node_positions is not None:
                num_nodes = node_positions.size(0)
            else:
                num_nodes = edge_index.max().item() + 1
        
        # Step 1: Interpolation from reduced coordinates to full coordinates
        # U0 = I_{X1}^{X2}(Z)
        if z.dim() == 2 and z.size(0) == 1:
            z_expanded = z.squeeze(0)  # [latent_dim]
        elif z.dim() == 2:
            z_expanded = z[0]  # [latent_dim]
        else:
            z_expanded = z  # [latent_dim]
        
        # Project latent to hidden dimension for interpolation
        z_hidden = self.latent_proj(z_expanded.unsqueeze(0))
        z_hidden = z_hidden.squeeze(0)  # [hidden_dim]
        
        if node_positions is not None:
            # Create reduced coordinate set X2 (from encoder)
            # For now, assume we have a single reduced coordinate or use mean
            # In practice, this would come from the encoder's reduced coordinates
            num_reduced = 1  # Single global latent vector
            x2 = node_positions.mean(dim=0, keepdim=True)  # [1, 2] - mean coordinate
            x1 = node_positions  # [num_nodes, 2]
            
            # Expand z_hidden to match number of reduced coordinates
            z_features_reduced = z_hidden.unsqueeze(0)  # [1, hidden_dim]
            
            # Interpolate from reduced coordinates to full coordinates
            u0 = interpolate_to_coordinates(x2, x1, z_features_reduced)  # [num_nodes, hidden_dim]
        else:
            # Fallback: broadcast to all nodes
            u0 = z_hidden.unsqueeze(0).expand(num_nodes, -1)  # [num_nodes, hidden_dim]
        
        # Optionally condition on node positions
        if node_positions is not None and self.pos_proj is not None:
            pos_features = self.pos_proj(node_positions)  # [num_nodes, hidden_dim]
            u0 = u0 + pos_features
        
        # Step 2: mlp0 + lnm0: Transform node features
        # U1 = [lnm0 o mlp0(u^0_1), ..., lnm0 o mlp0(u^0_{|X1|})]
        u1 = self.mlp0(u0)  # [num_nodes, hidden_dim]
        u1 = self.lnm0(u1)
        
        # Step 3: mlp1 + lnm1: Transform edge features (if used)
        if self.use_edge_features:
            if node_positions is not None:
                e0 = self._compute_edge_features(node_positions, edge_index)
            else:
                # Fallback: use zero edge features
                num_edges = edge_index.size(1)
                e0 = torch.zeros(num_edges, self.edge_dim, device=z.device, dtype=z.dtype)
            
            # E1 = {e^1_{i,j} | e^1_{i,j} = lnm1 o mlp1(e^0_{i,j})}
            e1 = self.mlp1(e0)  # [num_edges, edge_dim]
            e1 = self.lnm1(e1)
        else:
            e1 = None
        
        # Step 4: GNN layers: G2 = gnn_Ng(G1)
        # G1 = (U1, A, E1)
        u2 = u1
        e2 = e1
        for gnn_layer in self.gnn_layers:
            u2, e2 = gnn_layer(u2, edge_index, e2)
        
        # Step 5: mlp2: Final output
        # U = [mlp2(u^2_1), ..., mlp2(u^2_{|X1|})]
        x_recon = self.mlp2(u2)  # [num_nodes, output_dim]
        
        return x_recon
