"""
Graph-LED: Main model combining GNN encoder/decoder with temporal attention.
"""

import torch
import torch.nn as nn
from .gnn_encoder import GNNEncoder
from .gnn_decoder import GNNDecoder
from .temporal_model import TemporalAttentionModel


class GraphLED(nn.Module):
    """
    Graph-based Learning of Effective Dynamics (Graph-LED).
    
    Combines:
    - GNN encoder: Maps flow fields to latent space
    - Temporal attention model: Learns temporal dynamics in latent space
    - GNN decoder: Maps latent representations back to flow fields
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # u, v, p
        hidden_dim: int = 64,
        latent_dim: int = 32,
        output_dim: int = 3,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        use_edge_features: bool = True,
        dropout: float = 0.1,
        aggregation: str = 'mean'
    ):
        super(GraphLED, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # GNN Encoder
        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_gnn_layers,
            use_edge_features=use_edge_features,
            aggregation=aggregation
        )
        
        # Temporal Attention Model
        self.temporal_model = TemporalAttentionModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            num_layers=num_attention_layers,
            dropout=dropout
        )
        
        # GNN Decoder
        self.decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_gnn_layers,
            use_edge_features=use_edge_features,
            use_position_encoding=True
        )
    
    def encode(self, x, edge_index, batch=None):
        """Encode flow field to latent representation."""
        return self.encoder(x, edge_index, batch)
    
    def decode(self, z, edge_index, node_positions=None, num_nodes=None):
        """Decode latent representation to flow field."""
        return self.decoder(z, edge_index, node_positions, num_nodes)
    
    def forward(
        self,
        x_sequence,
        edge_index,
        node_positions=None,
        batch=None,
        num_steps=1,
        return_latent=False
    ):
        """
        Forward pass through Graph-LED.
        
        Args:
            x_sequence: Sequence of flow fields [batch_size, seq_len, num_nodes, input_dim]
                       or list of [num_nodes, input_dim] tensors
            edge_index: Graph connectivity [2, num_edges]
            node_positions: Node positions [num_nodes, 2] (optional)
            batch: Batch vector [num_nodes] (optional)
            num_steps: Number of future steps to predict
            return_latent: Whether to return latent representations
        
        Returns:
            predictions: Predicted flow fields [batch_size, num_steps, num_nodes, output_dim]
            latent_seq: Latent sequence (if return_latent=True)
        """
        # Handle different input formats
        if isinstance(x_sequence, list):
            # List of graphs
            seq_len = len(x_sequence)
            batch_size = 1
            num_nodes = x_sequence[0].shape[0]
            x_seq_tensor = torch.stack(x_sequence, dim=0).unsqueeze(0)  # [1, seq_len, num_nodes, input_dim]
        else:
            batch_size, seq_len, num_nodes, input_dim = x_sequence.shape
            x_seq_tensor = x_sequence
        
        # Encode sequence to latent space
        z_sequence = []
        for t in range(seq_len):
            x_t = x_seq_tensor[:, t, :, :].squeeze(0)  # [num_nodes, input_dim] or [batch_size, num_nodes, input_dim]
            if x_t.dim() == 3:
                # Batch case - need to handle per-graph encoding
                z_t_list = []
                for b in range(batch_size):
                    z_t = self.encode(x_t[b], edge_index, batch)
                    z_t_list.append(z_t)
                z_t = torch.stack(z_t_list, dim=0)
            else:
                z_t = self.encode(x_t, edge_index, batch)
            
            z_sequence.append(z_t)
        
        # Stack latent representations
        if len(z_sequence) == 0:
            raise ValueError("Empty z_sequence")
        
        # Check shape of first element
        first_z = z_sequence[0]
        if first_z.dim() == 1:
            # Single graph case: [latent_dim] -> stack to [1, seq_len, latent_dim]
            z_seq = torch.stack(z_sequence, dim=0).unsqueeze(0)  # [1, seq_len, latent_dim]
        elif first_z.dim() == 2:
            # Batch case: [batch_size, latent_dim] -> [batch_size, seq_len, latent_dim]
            z_seq = torch.stack(z_sequence, dim=1)  # [batch_size, seq_len, latent_dim]
        elif first_z.dim() == 3:
            # Already batched: [batch_size, 1, latent_dim] -> [batch_size, seq_len, latent_dim]
            z_seq = torch.cat(z_sequence, dim=1)  # Concatenate along sequence dimension
        else:
            raise ValueError(f"Unexpected z shape: {first_z.shape}")
        
        # Ensure z_seq has 3 dimensions: [batch, seq_len, latent_dim]
        if z_seq.dim() == 2:
            z_seq = z_seq.unsqueeze(0)  # Add batch dimension
        
        # Predict future latent states
        if num_steps == 1:
            z_pred, _ = self.temporal_model(z_seq)
            z_pred = z_pred.unsqueeze(1)  # [batch_size, 1, latent_dim]
        else:
            z_pred = self.temporal_model.autoregressive_forward(z_seq, num_steps)
            # z_pred: [batch_size, num_steps, latent_dim]
        
        # Decode predicted latent states to flow fields
        predictions = []
        for step in range(num_steps):
            z_pred_step = z_pred[:, step, :]  # [batch_size, latent_dim]
            
            if z_pred_step.dim() == 1:
                z_pred_step = z_pred_step.unsqueeze(0)
            
            # Decode each graph in batch
            x_pred_list = []
            for b in range(batch_size):
                z_b = z_pred_step[b:b+1]  # [1, latent_dim]
                x_pred = self.decode(z_b, edge_index, node_positions, num_nodes)
                x_pred_list.append(x_pred)
            
            if len(x_pred_list) == 1:
                predictions.append(x_pred_list[0])
            else:
                predictions.append(torch.stack(x_pred_list, dim=0))
        
        # Stack predictions
        if isinstance(predictions[0], torch.Tensor) and predictions[0].dim() == 2:
            pred_tensor = torch.stack(predictions, dim=0).unsqueeze(0)  # [1, num_steps, num_nodes, output_dim]
        else:
            pred_tensor = torch.stack(predictions, dim=1)  # [batch_size, num_steps, num_nodes, output_dim]
        
        if return_latent:
            return pred_tensor, z_seq, z_pred
        return pred_tensor

