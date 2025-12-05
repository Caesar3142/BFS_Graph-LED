"""
Attention-based Autoregressive Temporal Model.
Learns temporal dependencies in the latent space.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalAttentionModel(nn.Module):
    """
    Autoregressive temporal model with self-attention following Equation 14.
    Implements sliding window mechanism: S_j = {Z_0, ..., Z_j} if j < N_sw - 1,
    S_j = {Z_{j+2-N_sw}, ..., Z_j} if j >= N_sw - 1.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        sliding_window: int = None  # N_sw from Equation 14
    ):
        super(TemporalAttentionModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sliding_window = sliding_window  # N_sw: sliding window length
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_sequence, mask=None):
        """
        Forward pass through the temporal model.
        
        Args:
            z_sequence: Sequence of latent representations [batch_size, seq_len, latent_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            z_pred: Predicted next latent state [batch_size, latent_dim]
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, _ = z_sequence.shape
        
        # Input projection
        x = self.input_proj(z_sequence)  # [batch_size, seq_len, hidden_dim]
        
        # Transpose for positional encoding (seq_len first)
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # Apply attention layers
        attention_weights_list = []
        for i, (attn, ffn, ln1, ln2) in enumerate(
            zip(self.attention_layers, self.ffn_layers, 
                self.layer_norms1, self.layer_norms2)
        ):
            # Self-attention with residual connection
            residual = x
            x_attn, attn_weights = attn(x, x, x, mask)
            x = ln1(x + self.dropout(x_attn))
            attention_weights_list.append(attn_weights)
            
            # Feed-forward with residual connection
            residual = x
            x = ln2(x + ffn(x))
        
        # Use the last time step for prediction
        x_last = x[:, -1, :]  # [batch_size, hidden_dim]
        
        # Output projection
        z_pred = self.output_proj(x_last)  # [batch_size, latent_dim]
        
        return z_pred, attention_weights_list
    
    def autoregressive_forward(self, z_sequence, num_steps=1):
        """
        Autoregressive prediction with sliding window following Equation 14.
        
        Args:
            z_sequence: Initial sequence [batch_size, seq_len, latent_dim]
            num_steps: Number of future steps to predict
        
        Returns:
            predictions: Predicted sequences [batch_size, num_steps, latent_dim]
        """
        predictions = []
        batch_size, seq_len, latent_dim = z_sequence.shape
        
        # Initialize sequence S_0 = {Z_0}
        current_sequence = z_sequence.clone()  # [batch_size, seq_len, latent_dim]
        
        # Determine sliding window length N_sw
        if self.sliding_window is not None:
            n_sw = self.sliding_window
        else:
            n_sw = seq_len  # Use full sequence if not specified
        
        for j in range(num_steps):
            # Get current sequence S_j following Equation 14
            if j < n_sw - 1:
                # S_j = {Z_0, ..., Z_j} if j < N_sw - 1
                s_j = current_sequence[:, :j+1, :]  # [batch_size, j+1, latent_dim]
            else:
                # S_j = {Z_{j+2-N_sw}, ..., Z_j} if j >= N_sw - 1
                start_idx = j + 2 - n_sw
                s_j = current_sequence[:, start_idx:j+1, :]  # [batch_size, N_sw, latent_dim]
            
            # Predict Z_{j+1} = mhat_N_h^(S_j)(Z_j)
            # Use last element Z_j as query
            z_j = s_j[:, -1:, :]  # [batch_size, 1, latent_dim]
            
            # Forward pass with sliding window sequence
            z_pred, _ = self.forward(s_j)  # [batch_size, latent_dim]
            predictions.append(z_pred)
            
            # Update sequence: append Z_{j+1} and maintain window
            z_pred_expanded = z_pred.unsqueeze(1)  # [batch_size, 1, latent_dim]
            
            if current_sequence.size(1) < n_sw:
                # Still building up to window size
                current_sequence = torch.cat([current_sequence, z_pred_expanded], dim=1)
            else:
                # Sliding window: remove oldest, add newest
                current_sequence = torch.cat([current_sequence[:, 1:, :], z_pred_expanded], dim=1)
        
        return torch.stack(predictions, dim=1)  # [batch_size, num_steps, latent_dim]

