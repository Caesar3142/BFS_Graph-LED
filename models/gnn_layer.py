"""
GNN Layer implementation following Equation 7 from the reference.
Implements edge update then node update with Layer Normalization.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, scatter


class GNNLayer(MessagePassing):
    """
    GNN Layer as per Equation 7:
    - Step 1: Edge feature update: e_{i,j}^2 = e_{i,j}^1 + lnm_e o mlp_e([u_i^1, u_j^1, e_{i,j}^1])
    - Step 2: Node feature update: u_i^2 = u_i^1 + lnm_n o mlp_n([u_i^1, (1/|N(i)|) * sum(e_{i,j}^2)])
    """
    
    def __init__(self, node_dim, edge_dim=None, hidden_dim=None):
        """
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features (if None, no edge features)
            hidden_dim: Hidden dimension for MLPs (defaults to node_dim)
        """
        super(GNNLayer, self).__init__(aggr='mean')  # Mean aggregation as per reference
        
        if hidden_dim is None:
            hidden_dim = node_dim
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim if edge_dim is not None else 0
        self.hidden_dim = hidden_dim
        self.use_edge_features = edge_dim is not None and edge_dim > 0
        
        # Edge update MLP: mlp_e
        if self.use_edge_features:
            # Input: [u_i, u_j, e_{i,j}]
            edge_input_dim = 2 * node_dim + edge_dim
            self.mlp_e = nn.Sequential(
                nn.Linear(edge_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, edge_dim)
            )
            # Edge Layer Normalization: lnm_e
            self.lnm_e = nn.LayerNorm(edge_dim)
        else:
            self.mlp_e = None
            self.lnm_e = None
        
        # Node update MLP: mlp_n
        # Input: [u_i, mean(e_{i,j}^2 for j in N(i))]
        node_input_dim = node_dim + (edge_dim if self.use_edge_features else 0)
        self.mlp_n = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        # Node Layer Normalization: lnm_n
        self.lnm_n = nn.LayerNorm(node_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass through GNN layer.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
        
        Returns:
            x_out: Updated node features [num_nodes, node_dim]
            edge_attr_out: Updated edge features [num_edges, edge_dim] (if use_edge_features)
        """
        # Step 1: Update edge features
        if self.use_edge_features and edge_attr is not None:
            edge_attr_out = self._update_edges(x, edge_index, edge_attr)
        else:
            edge_attr_out = None
        
        # Step 2: Update node features
        x_out = self._update_nodes(x, edge_index, edge_attr_out)
        
        return x_out, edge_attr_out
    
    def _update_edges(self, x, edge_index, edge_attr):
        """
        Step 1: Update edge features.
        e_{i,j}^2 = e_{i,j}^1 + lnm_e o mlp_e([u_i^1, u_j^1, e_{i,j}^1])
        """
        row, col = edge_index
        x_i = x[row]  # [num_edges, node_dim]
        x_j = x[col]  # [num_edges, node_dim]
        
        # Concatenate [u_i, u_j, e_{i,j}]
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)  # [num_edges, 2*node_dim + edge_dim]
        
        # Apply MLP and LayerNorm with residual
        edge_update = self.mlp_e(edge_input)  # [num_edges, edge_dim]
        edge_update = self.lnm_e(edge_update)
        edge_attr_out = edge_attr + edge_update  # Residual connection
        
        return edge_attr_out
    
    def _update_nodes(self, x, edge_index, edge_attr_out):
        """
        Step 2: Update node features.
        u_i^2 = u_i^1 + lnm_n o mlp_n([u_i^1, (1/|N(i)|) * sum(e_{i,j}^2)])
        """
        # Aggregate edge features: mean of neighbors' edge features
        if self.use_edge_features and edge_attr_out is not None:
            # Aggregate edge features per node using scatter
            # edge_index[0] contains the source nodes (i), edge_index[1] contains target nodes (j)
            # We want to aggregate edges where node i is the source
            row, col = edge_index
            num_nodes = x.size(0)
            
            # Aggregate edge features: mean of all edges connected to each node
            # For each node i, aggregate all edges e_{i,j} where j is a neighbor
            aggregated_edges = scatter(
                edge_attr_out, 
                row, 
                dim=0, 
                dim_size=num_nodes, 
                reduce='mean'
            )  # [num_nodes, edge_dim]
            
            # Concatenate [u_i, mean(e_{i,j}^2)]
            node_input = torch.cat([x, aggregated_edges], dim=1)  # [num_nodes, node_dim + edge_dim]
        else:
            # If no edge features, use only node features
            node_input = x  # [num_nodes, node_dim]
        
        # Apply MLP and LayerNorm with residual
        node_update = self.mlp_n(node_input)  # [num_nodes, node_dim]
        node_update = self.lnm_n(node_update)
        x_out = x + node_update  # Residual connection
        
        return x_out
