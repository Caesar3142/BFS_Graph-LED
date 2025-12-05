"""
Graph-LED: Graph-based Learning of Effective Dynamics
Main model implementation combining GNN encoder/decoder with temporal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GNNLayer(MessagePassing):
    """Graph Neural Network layer with edge features."""
    
    def __init__(self, in_dim, out_dim, use_edge_features=True, aggregation='mean'):
        super(GNNLayer, self).__init__(aggr=aggregation)
        self.use_edge_features = use_edge_features
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Edge feature transformation (if used)
        if use_edge_features:
            self.edge_mlp = nn.Sequential(
                nn.Linear(1, out_dim),  # Edge distance -> feature
                nn.ReLU()
            )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges] or [num_edges, 1] (optional)
        """
        # Ensure edge_attr is 2D if provided
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)  # [num_edges, 1]
        
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, num_nodes=x.size(0)
        )
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        """Compute messages between nodes."""
        msg = x_j
        
        if self.use_edge_features and edge_attr is not None:
            # edge_attr should be [num_edges, 1] after add_self_loops
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_feat = self.edge_mlp(edge_attr)
            msg = msg + edge_feat
        
        return msg
    
    def update(self, aggr_out, x):
        """Update node features."""
        return self.node_mlp(aggr_out)


class GNNEncoder(nn.Module):
    """GNN-based encoder for mesh data."""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3, 
                 use_edge_features=True, aggregation='mean'):
        super(GNNEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.layers.append(
                GNNLayer(in_dim, hidden_dim, use_edge_features, aggregation)
            )
        
        # Latent projection
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [batch_size, num_nodes, input_dim] or [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges] or [num_edges, 1]
        """
        # Handle batch dimension
        if x.dim() == 3:
            batch_size, num_nodes, input_dim = x.shape
            x = x.reshape(-1, input_dim)  # [batch_size * num_nodes, input_dim]
            # Expand edge_index for batch
            edge_index_list = []
            edge_attr_list = []
            for b in range(batch_size):
                offset = b * num_nodes
                edge_index_b = edge_index + offset
                edge_index_list.append(edge_index_b)
                if edge_attr is not None:
                    edge_attr_list.append(edge_attr)
            
            edge_index = torch.cat(edge_index_list, dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat(edge_attr_list, dim=0)
        
        # Project input
        x = self.input_proj(x)
        
        # Apply GNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Project to latent space
        z = self.latent_proj(x)
        
        # Reshape back if batched
        if z.dim() == 2 and hasattr(self, '_batch_size'):
            z = z.reshape(self._batch_size, -1, z.size(-1))
        
        return z


class GNNDecoder(nn.Module):
    """GNN-based decoder."""
    
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3,
                 use_edge_features=True, aggregation='mean'):
        super(GNNDecoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Latent projection
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # GNN layers
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.layers.append(
                GNNLayer(in_dim, hidden_dim, use_edge_features, aggregation)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z, edge_index, edge_attr=None):
        """
        Args:
            z: Latent features [batch_size, num_nodes, latent_dim] or [num_nodes, latent_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges] or [num_edges, 1]
        """
        # Handle batch dimension
        if z.dim() == 3:
            batch_size, num_nodes, latent_dim = z.shape
            z = z.reshape(-1, latent_dim)
            # Expand edge_index for batch
            edge_index_list = []
            edge_attr_list = []
            for b in range(batch_size):
                offset = b * num_nodes
                edge_index_b = edge_index + offset
                edge_index_list.append(edge_index_b)
                if edge_attr is not None:
                    edge_attr_list.append(edge_attr)
            
            edge_index = torch.cat(edge_index_list, dim=1)
            if edge_attr is not None:
                edge_attr = torch.cat(edge_attr_list, dim=0)
        
        # Project latent
        x = self.latent_proj(z)
        
        # Apply GNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Project to output
        out = self.output_proj(x)
        
        # Reshape back if batched
        if out.dim() == 2 and hasattr(self, '_batch_size'):
            out = out.reshape(self._batch_size, -1, out.size(-1))
        
        return out


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for autoregressive prediction."""
    
    def __init__(self, latent_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(TemporalAttention, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, z_sequence):
        """
        Args:
            z_sequence: Latent sequence [batch_size, sequence_length, num_nodes, latent_dim]
        """
        batch_size, seq_len, num_nodes, latent_dim = z_sequence.shape
        
        # Reshape: [batch_size * num_nodes, sequence_length, latent_dim]
        z_flat = z_sequence.reshape(batch_size * num_nodes, seq_len, latent_dim)
        
        # Apply temporal attention
        z_attended = self.transformer(z_flat)
        
        # Reshape back: [batch_size, sequence_length, num_nodes, latent_dim]
        z_attended = z_attended.reshape(batch_size, seq_len, num_nodes, latent_dim)
        
        return z_attended


class GraphLED(nn.Module):
    """Main Graph-LED model combining GNN encoder/decoder with temporal attention."""
    
    def __init__(
        self,
        input_dim=3,
        output_dim=3,
        hidden_dim=64,
        latent_dim=32,
        num_gnn_layers=3,
        num_attention_heads=4,
        num_attention_layers=2,
        use_edge_features=True,
        dropout=0.1,
        aggregation='mean'
    ):
        super(GraphLED, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # GNN Encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_gnn_layers,
            use_edge_features=use_edge_features,
            aggregation=aggregation
        )
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(
            latent_dim=latent_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # GNN Decoder
        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers,
            use_edge_features=use_edge_features,
            aggregation=aggregation
        )
    
    def forward(self, input_sequence, edge_index, edge_attr=None):
        """
        Args:
            input_sequence: Input sequence [batch_size, sequence_length, num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge attributes [num_edges] or [num_edges, 1] (optional)
        
        Returns:
            predictions: [batch_size, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = input_sequence.shape
        
        # Encode each timestep
        z_sequence = []
        for t in range(seq_len):
            x_t = input_sequence[:, t, :, :]  # [batch_size, num_nodes, input_dim]
            # Flatten for encoder
            x_t_flat = x_t.reshape(-1, input_dim)  # [batch_size * num_nodes, input_dim]
            
            # Expand edge_index for batch
            # edge_index should be [2, num_edges]
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError(f"Expected edge_index shape [2, num_edges], got {edge_index.shape}")
            
            edge_index_batch = []
            edge_attr_batch = []
            for b in range(batch_size):
                offset = b * num_nodes
                # Add offset to both rows of edge_index
                edge_index_b = edge_index.clone()
                edge_index_b[0, :] += offset
                edge_index_b[1, :] += offset
                edge_index_batch.append(edge_index_b)
                if edge_attr is not None:
                    edge_attr_batch.append(edge_attr)
            
            edge_index_expanded = torch.cat(edge_index_batch, dim=1)  # [2, batch_size * num_edges]
            edge_attr_expanded = torch.cat(edge_attr_batch, dim=0) if edge_attr is not None else None  # [batch_size * num_edges] or [batch_size * num_edges, 1]
            
            # Encode
            z_t = self.encoder(x_t_flat, edge_index_expanded, edge_attr_expanded)
            z_t = z_t.reshape(batch_size, num_nodes, self.latent_dim)
            z_sequence.append(z_t)
        
        # Stack: [batch_size, sequence_length, num_nodes, latent_dim]
        z_sequence = torch.stack(z_sequence, dim=1)
        
        # Apply temporal attention
        z_attended = self.temporal_attention(z_sequence)
        
        # Use last timestep for prediction
        z_last = z_attended[:, -1, :, :]  # [batch_size, num_nodes, latent_dim]
        
        # Prediction head
        z_pred = self.prediction_head(z_last.reshape(-1, self.latent_dim))
        z_pred = z_pred.reshape(batch_size, num_nodes, self.latent_dim)
        
        # Decode
        z_pred_flat = z_pred.reshape(-1, self.latent_dim)
        edge_index_batch = []
        edge_attr_batch = []
        for b in range(batch_size):
            offset = b * num_nodes
            edge_index_b = edge_index.clone()
            edge_index_b[0, :] += offset
            edge_index_b[1, :] += offset
            edge_index_batch.append(edge_index_b)
            if edge_attr is not None:
                edge_attr_batch.append(edge_attr)
        
        edge_index_expanded = torch.cat(edge_index_batch, dim=1)
        edge_attr_expanded = torch.cat(edge_attr_batch, dim=0) if edge_attr is not None else None
        
        predictions = self.decoder(z_pred_flat, edge_index_expanded, edge_attr_expanded)
        predictions = predictions.reshape(batch_size, num_nodes, self.output_dim)
        
        return predictions
