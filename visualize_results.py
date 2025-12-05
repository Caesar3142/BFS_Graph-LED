"""
Visualize training results and predictions.
"""

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from models import GraphLED
from data import create_dataloader
from utils import visualize_flow_field, visualize_prediction_comparison, set_seed, interpolate_to_grid


def visualize_data_overview(data_path, output_dir='results/visualizations'):
    """Visualize overview of the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load(data_path, allow_pickle=True)
    node_positions = data['node_positions']
    flow_fields = data['flow_fields']
    
    print(f"Dataset shape: {flow_fields.shape}")
    print(f"Node positions: {node_positions.shape}")
    
    # Visualize mesh
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        c='lightblue',
        s=1,
        alpha=0.6
    )
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Mesh Overview', fontsize=14)
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_mesh_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/01_mesh_overview.png")
    
    # Visualize first and last timestep for each field using contours
    num_timesteps = flow_fields.shape[0]
    num_features = flow_fields.shape[2]
    feature_names = ['u-velocity', 'v-velocity', 'pressure']
    colormaps = ['viridis', 'coolwarm', 'plasma']
    
    for feature_idx in range(min(num_features, 3)):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, t in enumerate([0, num_timesteps - 1]):
            values = flow_fields[t, :, feature_idx]
            
            # Interpolate to regular grid for smooth ParaView-like contours
            X_grid, Y_grid, Z_grid = interpolate_to_grid(
                node_positions, values, grid_resolution=300
            )
            
            # Use regular contourf for smooth appearance
            contour = axes[idx].contourf(
                X_grid, Y_grid, Z_grid,
                levels=50,
                cmap=colormaps[feature_idx],
                vmin=flow_fields[:, :, feature_idx].min(),
                vmax=flow_fields[:, :, feature_idx].max(),
                extend='both'
            )
            # Add subtle contour lines
            axes[idx].contour(
                X_grid, Y_grid, Z_grid,
                levels=20,
                colors='black',
                alpha=0.15,
                linewidths=0.3
            )
            plt.colorbar(contour, ax=axes[idx], label=feature_names[feature_idx])
            axes[idx].set_title(f'{feature_names[feature_idx]} at t={t} (timestep {t})', fontsize=12)
            axes[idx].set_xlabel('X', fontsize=10)
            axes[idx].set_ylabel('Y', fontsize=10)
            axes[idx].axis('equal')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_{feature_names[feature_idx].replace("-", "_")}_evolution.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/02_{feature_names[feature_idx].replace("-", "_")}_evolution.png")
    
    # Time series at a few selected points
    num_sample_points = 5
    sample_indices = np.linspace(0, len(node_positions) - 1, num_sample_points, dtype=int)
    
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 4 * num_features))
    if num_features == 1:
        axes = [axes]
    
    for feature_idx in range(num_features):
        for point_idx in sample_indices:
            time_series = flow_fields[:, point_idx, feature_idx]
            axes[feature_idx].plot(time_series, label=f'Point {point_idx}', alpha=0.7, linewidth=1.5)
        
        axes[feature_idx].set_xlabel('Timestep', fontsize=10)
        axes[feature_idx].set_ylabel(feature_names[feature_idx], fontsize=10)
        axes[feature_idx].set_title(f'{feature_names[feature_idx]} Time Series at Sample Points', fontsize=12)
        axes[feature_idx].legend(fontsize=8)
        axes[feature_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/03_time_series.png")


def visualize_predictions(config_path, checkpoint_path, output_dir='results/visualizations', num_samples=5):
    """Visualize model predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    data_path = config['data']['path']
    data = np.load(data_path, allow_pickle=True)
    node_positions = data['node_positions']
    flow_fields = data['flow_fields']
    
    # Create dataloader
    train_loader, val_loader, test_loader, dataset_stats = create_dataloader(
        node_positions=node_positions,
        flow_fields=flow_fields,
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=1,
        shuffle=False,
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
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith('.pt'):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f'Loaded model from {checkpoint_path}')
    else:
        print('Warning: No checkpoint found, using untrained model')
    
    model.eval()
    
    # Get samples from test set
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
            
            input_sequence = batch['input_sequence'].to(device)
            target_sequence = batch['target_sequence'].to(device)
            edge_index = batch['edge_index'].to(device)
            if edge_index.dim() == 3:
                edge_index = edge_index[0]
            node_positions_tensor = batch['node_positions'].to(device)
            if node_positions_tensor.dim() == 3:
                node_positions_tensor = node_positions_tensor[0]
            
            # Predict
            predictions = model(
                input_sequence,
                edge_index,
                node_positions=node_positions_tensor,
                num_steps=target_sequence.shape[1]
            )
            
            # Convert to numpy
            pred_np = predictions[0].cpu().numpy()
            target_np = target_sequence[0].cpu().numpy()
            node_pos_np = node_positions_tensor.cpu().numpy()
            
            # Denormalize if needed
            if config['data'].get('normalize', True):
                # Get normalization stats from dataset
                flow_mean = dataset_stats['flow_mean']
                flow_std = dataset_stats['flow_std']
                pred_np = pred_np * flow_std + flow_mean
                target_np = target_np * flow_std + flow_mean
            
            # Visualize each feature
            feature_names = ['u-velocity', 'v-velocity', 'pressure']
            for feature_idx in range(min(pred_np.shape[2], 3)):
                for timestep in range(min(pred_np.shape[0], 3)):
                    save_path = f'{output_dir}/prediction_sample{sample_count}_feature{feature_idx}_timestep{timestep}.png'
                    visualize_prediction_comparison(
                        pred_np[timestep:timestep+1],
                        target_np[timestep:timestep+1],
                        node_pos_np,
                        timestep=0,
                        feature_idx=feature_idx,
                        save_path=save_path
                    )
                    print(f"Saved: {save_path}")
            
            sample_count += 1
    
    print(f"\nVisualized {sample_count} samples")


def main():
    parser = argparse.ArgumentParser(description='Visualize Graph-LED results')
    parser.add_argument('--config', type=str, default='configs/backward_step.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--data-only', action='store_true',
                       help='Only visualize data, not predictions')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of prediction samples to visualize')
    
    args = parser.parse_args()
    
    # Load config to get data path
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['path']
    
    print("="*60)
    print("Visualizing Graph-LED Results")
    print("="*60)
    
    # Visualize data overview
    print("\n1. Visualizing data overview...")
    visualize_data_overview(data_path, args.output_dir)
    
    # Visualize predictions if checkpoint provided
    if not args.data_only:
        if args.checkpoint:
            print("\n2. Visualizing predictions...")
            visualize_predictions(args.config, args.checkpoint, args.output_dir, args.num_samples)
        else:
            print("\n2. Skipping predictions (no checkpoint provided)")
            print("   Use --checkpoint to visualize model predictions")
    
    print("\n" + "="*60)
    print(f"Visualizations saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

