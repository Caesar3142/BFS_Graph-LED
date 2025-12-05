"""
GNN-based Decoder for reconstructing flow fields from latent representations.
Maps latent vectors back to flow field data on unstructured meshes.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops


class EdgeConv(MessagePassing):
    """Edge convolution layer for learning edge features."""
    
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(edge_features)


class GNNDecoder(nn.Module):
    """
    Graph Neural Network Decoder for unstructured meshes.
    Reconstructs flow field data from latent representations.
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
        
        # Latent projection to initial node features
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        if use_edge_features:
            for _ in range(num_layers):
                self.gnn_layers.append(EdgeConv(hidden_dim, hidden_dim))
        else:
            for _ in range(num_layers):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Position projection (optional, for conditioning on node positions)
        if use_position_encoding:
            self.pos_proj = nn.Linear(2, hidden_dim)
        else:
            self.pos_proj = None
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, z, edge_index, node_positions=None, num_nodes=None):
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent representation [batch_size, latent_dim] or [1, latent_dim]
            edge_index: Graph connectivity [2, num_edges]
            node_positions: Optional node positions for conditioning [num_nodes, 2]
            num_nodes: Number of nodes in the graph (if z is batch-level)
        
        Returns:
            x_recon: Reconstructed flow field [num_nodes, output_dim]
        """
        # Determine number of nodes
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        
        # Expand latent vector to all nodes
        if z.dim() == 2 and z.size(0) == 1:
            # Single graph case: broadcast to all nodes
            x = self.latent_proj(z).expand(num_nodes, -1)
        elif z.dim() == 2:
            # Batch case: need to handle per-graph expansion
            # For now, assume single graph per batch
            x = self.latent_proj(z[0:1]).expand(num_nodes, -1)
        else:
            x = self.latent_proj(z)
        
        # Optionally condition on node positions
        if node_positions is not None and self.pos_proj is not None:
            pos_features = self.pos_proj(node_positions)
            x = x + pos_features
        
        # Apply GNN layers
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.use_edge_features:
                x_new = gnn_layer(x, edge_index)
            else:
                x_new = gnn_layer(x, edge_index)
            
            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new
            
            x = bn(x)
            x = self.activation(x)
        
        # Output projection
        x_recon = self.output_proj(x)
        
        return x_recon

