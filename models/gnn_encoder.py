"""
GNN-based Encoder for processing unstructured mesh data.
Implements Equation 9 from the reference: Dimension reduction via GNN.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from .gnn_layer import GNNLayer
try:
    from data.interpolation import interpolate_to_coordinates
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.interpolation import interpolate_to_coordinates


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder following Equation 9.
    
    Process:
    1. mlp0 + lnm0: Transform initial node features U0 -> U1
    2. mlp1 + lnm1: Transform initial edge features E0 -> E1
    3. GNN layers: G2 = gnn_Ng(G1)
    4. mlp2 + lnm2: Transform node features U2 -> U3
    5. Interpolation: Z = I_{X2}^{X1}(U3) to reduced coordinates
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # u, v, p (velocity components and pressure)
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        use_edge_features: bool = True,
        aggregation: str = 'mean',  # 'mean', 'max', 'sum', 'attention'
        reduction_factor: float = 0.5  # Factor for coordinate reduction |X2|/|X1|
    ):
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        self.aggregation = aggregation
        self.reduction_factor = reduction_factor
        
        # Edge feature dimension (if used)
        if use_edge_features:
            self.edge_dim = 2  # Relative displacement or distance
        else:
            self.edge_dim = 0
        
        # mlp0 + lnm0: Initial node feature transformation
        self.mlp0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        
        # mlp2 + lnm2: Final node feature transformation
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lnm2 = nn.LayerNorm(hidden_dim)
        
        # Attention-based aggregation (if selected)
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output projection to latent space (after interpolation)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def _compute_edge_features(self, node_positions, edge_index):
        """Compute edge features from node positions."""
        row, col = edge_index
        edge_attr = node_positions[col] - node_positions[row]  # Relative displacement
        return edge_attr
    
    def forward(self, x, edge_index, node_positions=None, batch=None):
        """
        Forward pass through the encoder following Equation 9.
        
        Args:
            x: Node features U0 [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            node_positions: Node coordinates X1 [num_nodes, 2] (optional, for interpolation)
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            z: Latent representation [batch_size, latent_dim] or [num_nodes, latent_dim]
        """
        # Step 1: mlp0 + lnm0: Transform initial node features
        # U1 = [lnm0 o mlp0(u^0_1), ..., lnm0 o mlp0(u^0_{|X1|})]
        u1 = self.mlp0(x)  # [num_nodes, hidden_dim]
        u1 = self.lnm0(u1)
        
        # Step 2: mlp1 + lnm1: Transform initial edge features (if used)
        if self.use_edge_features:
            if node_positions is not None:
                e0 = self._compute_edge_features(node_positions, edge_index)
            else:
                # Fallback: use zero edge features
                num_edges = edge_index.size(1)
                e0 = torch.zeros(num_edges, self.edge_dim, device=x.device, dtype=x.dtype)
            
            # E1 = {e^1_{i,j} | e^1_{i,j} = lnm1 o mlp1(e^0_{i,j})}
            e1 = self.mlp1(e0)  # [num_edges, edge_dim]
            e1 = self.lnm1(e1)
        else:
            e1 = None
        
        # Step 3: GNN layers: G2 = gnn_Ng(G1)
        # G1 = (U1, A, E1)
        u2 = u1
        e2 = e1
        for gnn_layer in self.gnn_layers:
            u2, e2 = gnn_layer(u2, edge_index, e2)
        
        # Step 4: mlp2 + lnm2: Transform node features
        # U3 = [lnm2 o mlp2(u^2_1), ..., lnm2 o mlp2(u^2_{|X1|})]
        u3 = self.mlp2(u2)  # [num_nodes, hidden_dim]
        u3 = self.lnm2(u3)
        
        # Step 5: Interpolation to reduced coordinates (if node_positions provided)
        # Z = I_{X2}^{X1}(U3)
        if node_positions is not None and self.reduction_factor < 1.0:
            # Create reduced coordinate set X2
            num_nodes = node_positions.size(0)
            num_reduced = max(1, int(num_nodes * self.reduction_factor))
            
            # Sample or use subset of coordinates for X2
            if num_reduced < num_nodes:
                # Use k-means or simple sampling for reduced coordinates
                indices = torch.linspace(0, num_nodes - 1, num_reduced, dtype=torch.long, device=node_positions.device)
                x2 = node_positions[indices]
                x1 = node_positions
                
                # Interpolate U3 to reduced coordinates
                z_features = interpolate_to_coordinates(x1, x2, u3)  # [num_reduced, hidden_dim]
            else:
                z_features = u3  # No reduction
        else:
            # No interpolation: use global pooling
            z_features = u3
        
        # Global aggregation (if batch provided or no interpolation)
        if batch is not None:
            if self.aggregation == 'mean':
                z = global_mean_pool(z_features, batch)
            elif self.aggregation == 'max':
                z = global_max_pool(z_features, batch)
            elif self.aggregation == 'sum':
                z = global_mean_pool(z_features, batch) * (batch.max().item() + 1)
            elif self.aggregation == 'attention':
                att_weights = self.attention(z_features)
                att_weights = torch.softmax(att_weights, dim=0)
                z = global_mean_pool(z_features * att_weights, batch)
            else:
                z = global_mean_pool(z_features, batch)
        else:
            # If no batch, use mean pooling or return node-level features
            if z_features.dim() == 2 and z_features.size(0) > 1:
                if self.aggregation == 'mean':
                    z = z_features.mean(dim=0, keepdim=True)
                elif self.aggregation == 'max':
                    z = z_features.max(dim=0, keepdim=True)[0]
                else:
                    z = z_features.mean(dim=0, keepdim=True)
            else:
                z = z_features
        
        # Project to latent space
        z = self.output_proj(z)  # [batch_size, latent_dim] or [1, latent_dim]
        
        return z

