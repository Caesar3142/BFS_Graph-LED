"""
GNN-based Encoder for processing unstructured mesh data.
Converts flow field data on meshes to latent representations.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import add_self_loops, degree


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


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder for unstructured meshes.
    Processes flow field data on variable-size meshes and produces latent representations.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # u, v, p (velocity components and pressure)
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 3,
        use_edge_features: bool = True,
        aggregation: str = 'mean'  # 'mean', 'max', 'sum', 'attention'
    ):
        super(GNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        self.aggregation = aggregation
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        if use_edge_features:
            self.gnn_layers.append(EdgeConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(EdgeConv(hidden_dim, hidden_dim))
        else:
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention-based aggregation (if selected)
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output projection to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] (optional)
        
        Returns:
            z: Latent representation [batch_size, latent_dim] or [num_nodes, latent_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
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
        
        # Global aggregation
        if batch is not None:
            if self.aggregation == 'mean':
                z = global_mean_pool(x, batch)
            elif self.aggregation == 'max':
                z = global_max_pool(x, batch)
            elif self.aggregation == 'sum':
                z = global_mean_pool(x, batch) * (batch.max().item() + 1)  # Approximate sum
            elif self.aggregation == 'attention':
                # Attention-weighted aggregation
                att_weights = self.attention(x)  # [num_nodes, 1]
                att_weights = torch.softmax(att_weights, dim=0)
                z = global_mean_pool(x * att_weights, batch)
            else:
                z = global_mean_pool(x, batch)
        else:
            # If no batch, return node-level features or mean pool
            if self.aggregation == 'mean':
                z = x.mean(dim=0, keepdim=True)
            elif self.aggregation == 'max':
                z = x.max(dim=0, keepdim=True)[0]
            else:
                z = x.mean(dim=0, keepdim=True)
        
        # Project to latent space
        z = self.output_proj(z)
        
        return z

