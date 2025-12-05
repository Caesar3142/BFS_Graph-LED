"""
Training script for Graph-LED model.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from models import GraphLED
from data import create_dataloader
from utils import compute_metrics, save_checkpoint, set_seed


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        # Move data to device
        input_sequence = batch['input_sequence'].to(device)  # [batch, seq_len, nodes, features]
        target_sequence = batch['target_sequence'].to(device)  # [batch, pred_horizon, nodes, features]
        edge_index = batch['edge_index'].to(device)
        if edge_index.dim() == 3:
            edge_index = edge_index[0]  # Take first if batched
        node_positions = batch['node_positions'].to(device)
        if node_positions.dim() == 3:
            node_positions = node_positions[0]  # Take first if batched
        
        # Forward pass
        optimizer.zero_grad()
        
        # Reshape for model input (model expects [batch, seq_len, nodes, features])
        batch_size, seq_len, num_nodes, num_features = input_sequence.shape
        
        # Predict
        predictions = model(
            input_sequence,
            edge_index,
            node_positions=node_positions,
            num_steps=target_sequence.shape[1]
        )  # [batch, pred_horizon, nodes, features]
        
        # Compute loss
        loss = criterion(predictions, target_sequence)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            edge_index = batch['edge_index'].to(device)
            if edge_index.dim() == 3:
                edge_index = edge_index[0]  # Take first if batched
            node_positions = batch['node_positions'].to(device)
            if node_positions.dim() == 3:
                node_positions = node_positions[0]  # Take first if batched
            
            # Predict
            predictions = model(
                input_sequence,
                edge_index,
                node_positions=node_positions,
                num_steps=target_sequence.shape[1]
            )
            
            # Compute loss
            loss = criterion(predictions, target_sequence)
            total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu())
            all_targets.append(target_sequence.cpu())
            num_batches += 1
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Graph-LED model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, help='Path to data file (overrides config)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print('Loading data...')
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config['data']['path']
    
    # Load flow field data
    # Expected format: numpy file with 'node_positions' and 'flow_fields' keys
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        # If it's a dict-like array
        data = data.item()
    
    node_positions = data['node_positions']
    flow_fields = data['flow_fields']
    
    print(f'Data shape: {flow_fields.shape}')
    print(f'Number of nodes: {node_positions.shape[0]}')
    
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset_stats = create_dataloader(
        node_positions=node_positions,
        flow_fields=flow_fields,
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        normalize=config['data'].get('normalize', True),
        k_neighbors=config['data'].get('k_neighbors', 8),
        graph_type=config['data'].get('graph_type', 'knn')
    )
    
    # Create model
    model = GraphLED(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        output_dim=config['model']['output_dim'],
        num_gnn_layers=config['model']['num_gnn_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_attention_layers=config['model']['num_attention_layers'],
        use_edge_features=config['model'].get('use_edge_features', True),
        dropout=config['model'].get('dropout', 0.1),
        aggregation=config['model'].get('aggregation', 'mean')
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training'].get('weight_decay', 1e-5))
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['training'].get('patience', 10)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = config['training'].get('early_stop_patience', None)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Train Loss: {train_loss:.6f}')
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f'Val Loss: {val_metrics["loss"]:.6f}')
        print(f'Val RMSE: {val_metrics["rmse"]:.6f}')
        print(f'Val MAE: {val_metrics["mae"]:.6f}')
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics['loss'],
            val_metrics,
            checkpoint_path,
            is_best=is_best
        )
        
        if is_best:
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_path)
            print(f'Saved best model to {best_path}')
        
        # Early stopping
        if early_stop_patience and epochs_without_improvement >= early_stop_patience:
            print(f'\nEarly stopping: No improvement for {early_stop_patience} epochs')
            print(f'Best validation loss: {best_val_loss:.6f}')
            break
    
    print('\nTraining completed!')


if __name__ == '__main__':
    main()

