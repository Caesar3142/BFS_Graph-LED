"""
Example usage script for Graph-LED.
This demonstrates how to use the model for training and inference.
"""

import torch
import numpy as np
from models import GraphLED
from data import create_dataloader
from utils import set_seed, compute_metrics

# Set random seed for reproducibility
set_seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Example 1: Create model
print("\n" + "="*50)
print("Example 1: Creating Graph-LED Model")
print("="*50)

model = GraphLED(
    input_dim=3,  # u, v, p
    hidden_dim=64,
    latent_dim=32,
    output_dim=3,
    num_gnn_layers=3,
    num_attention_heads=4,
    num_attention_layers=2,
    use_edge_features=True,
    dropout=0.1
).to(device)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Example 2: Generate synthetic data (for demonstration)
print("\n" + "="*50)
print("Example 2: Generating Synthetic Data")
print("="*50)

# Create a simple mesh
num_nodes = 100
node_positions = np.random.rand(num_nodes, 2) * 10

# Generate synthetic flow fields
num_timesteps = 50
flow_fields = np.random.randn(num_timesteps, num_nodes, 3) * 0.5

print(f"Generated mesh with {num_nodes} nodes")
print(f"Generated {num_timesteps} timesteps of flow data")

# Example 3: Create dataloader
print("\n" + "="*50)
print("Example 3: Creating DataLoader")
print("="*50)

train_loader, val_loader, test_loader, dataset_stats = create_dataloader(
    node_positions=node_positions,
    flow_fields=flow_fields,
    sequence_length=10,
    prediction_horizon=1,
    batch_size=2,
    shuffle=True,
    normalize=True,
    k_neighbors=8
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Example 4: Forward pass
print("\n" + "="*50)
print("Example 4: Forward Pass")
print("="*50)

model.eval()
with torch.no_grad():
    # Get a batch
    batch = next(iter(train_loader))
    input_sequence = batch['input_sequence'].to(device)
    target_sequence = batch['target_sequence'].to(device)
    edge_index = batch['edge_index'].to(device)
    node_positions = batch['node_positions'].to(device)
    
    print(f"Input sequence shape: {input_sequence.shape}")
    print(f"Target sequence shape: {target_sequence.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Node positions shape: {node_positions.shape}")
    
    # Forward pass
    predictions = model(
        input_sequence,
        edge_index,
        node_positions=node_positions,
        num_steps=target_sequence.shape[1]
    )
    
    print(f"Predictions shape: {predictions.shape}")
    
    # Compute metrics
    metrics = compute_metrics(predictions, target_sequence)
    print(f"\nMetrics:")
    print(f"  MSE: {metrics['mse']:.6e}")
    print(f"  RMSE: {metrics['rmse']:.6e}")
    print(f"  MAE: {metrics['mae']:.6e}")

# Example 5: Training step
print("\n" + "="*50)
print("Example 5: Training Step")
print("="*50)

model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch = next(iter(train_loader))
input_sequence = batch['input_sequence'].to(device)
target_sequence = batch['target_sequence'].to(device)
edge_index = batch['edge_index'].to(device)
node_positions = batch['node_positions'].to(device)

optimizer.zero_grad()
predictions = model(
    input_sequence,
    edge_index,
    node_positions=node_positions,
    num_steps=target_sequence.shape[1]
)
loss = criterion(predictions, target_sequence)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.6f}")

print("\n" + "="*50)
print("Example usage completed!")
print("="*50)
print("\nTo train on real data:")
print("1. Prepare your data in the format: node_positions [N, 2] and flow_fields [T, N, 3]")
print("2. Save as .npz file with keys 'node_positions' and 'flow_fields'")
print("3. Update config file with data path")
print("4. Run: python train.py --config configs/cylinder.yaml")

