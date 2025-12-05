"""
Utility functions for training and evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics for flow field predictions.
    
    Args:
        predictions: Predicted flow fields [batch, timesteps, nodes, features]
        targets: Target flow fields [batch, timesteps, nodes, features]
    
    Returns:
        Dictionary of metrics
    """
    predictions = torch.as_tensor(predictions)
    targets = torch.as_tensor(targets)
    
    # Mean Squared Error (MSE)
    mse = torch.mean((predictions - targets) ** 2).item()
    
    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
    
    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Relative Error
    relative_error = torch.mean(
        torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
    ).item()
    
    # Per-feature errors
    per_feature_mse = torch.mean((predictions - targets) ** 2, dim=(0, 1, 2)).numpy()
    per_feature_mae = torch.mean(torch.abs(predictions - targets), dim=(0, 1, 2)).numpy()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'relative_error': relative_error,
        'per_feature_mse': per_feature_mse,
        'per_feature_mae': per_feature_mae
    }
    
    return metrics


def interpolate_to_grid(node_positions, values, grid_resolution=600):
    """
    Interpolate unstructured mesh data to regular grid for smooth contour plots.
    
    Args:
        node_positions: [N, 2] array of node positions
        values: [N] array of values at nodes
        grid_resolution: Resolution of the grid (grid_resolution x grid_resolution)
    
    Returns:
        X_grid, Y_grid: Meshgrid arrays
        Z_grid: Interpolated values on grid
    """
    from scipy.interpolate import griddata
    
    # Get domain bounds
    x_min, x_max = node_positions[:, 0].min(), node_positions[:, 0].max()
    y_min, y_max = node_positions[:, 1].min(), node_positions[:, 1].max()
    
    # Create regular grid
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Interpolate using cubic interpolation for smoothness
    # Try cubic first, fall back to linear if cubic fails
    try:
        Z_grid = griddata(
            node_positions,
            values,
            (X_grid, Y_grid),
            method='cubic',
            fill_value=np.nan  # Fill outside domain with NaN
        )
    except:
        # Fallback to linear if cubic fails
        Z_grid = griddata(
            node_positions,
            values,
            (X_grid, Y_grid),
            method='linear',
            fill_value=np.nan
        )
    
    return X_grid, Y_grid, Z_grid


def visualize_flow_field(
    flow_field,
    node_positions,
    feature_idx=0,
    title='Flow Field',
    save_path=None,
    vmin=None,
    vmax=None,
        use_contour=True,
        grid_resolution=600
):
    """
    Visualize flow field on unstructured mesh with smooth ParaView-like contours.
    
    Args:
        flow_field: Flow field values [num_nodes, num_features] or [num_nodes]
        node_positions: Node positions [num_nodes, 2]
        feature_idx: Index of feature to visualize (if flow_field has multiple features)
        title: Plot title
        save_path: Path to save figure (optional)
        vmin, vmax: Value limits for colormap
        use_contour: If True, use smooth contour plots; if False, use scatter
        grid_resolution: Resolution for grid interpolation (higher = smoother)
    """
    if isinstance(flow_field, torch.Tensor):
        flow_field = flow_field.detach().cpu().numpy()
    if isinstance(node_positions, torch.Tensor):
        node_positions = node_positions.detach().cpu().numpy()
    
    if flow_field.ndim > 1:
        values = flow_field[:, feature_idx]
    else:
        values = flow_field
    
    plt.figure(figsize=(10, 8))
    
    if use_contour:
        # Interpolate to regular grid for smooth contours
        X_grid, Y_grid, Z_grid = interpolate_to_grid(
            node_positions, values, grid_resolution=grid_resolution
        )
        
        # Use regular contourf for smooth ParaView-like appearance
        contour = plt.contourf(
            X_grid, Y_grid, Z_grid,
            levels=50,  # More levels for smoother appearance
            cmap='viridis',
            vmin=vmin if vmin is not None else np.nanmin(values),
            vmax=vmax if vmax is not None else np.nanmax(values),
            extend='both'
        )
        # Add contour lines
        plt.contour(
            X_grid, Y_grid, Z_grid,
            levels=20,
            colors='black',
            alpha=0.15,
            linewidths=0.3
        )
        plt.colorbar(contour, label='Flow Value')
    else:
        scatter = plt.scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=values,
            cmap='viridis',
            s=20,
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(scatter, label='Flow Value')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def visualize_prediction_comparison(
    predictions,
    targets,
    node_positions,
    timestep=0,
    feature_idx=0,
    save_path=None,
        grid_resolution=600
):
    """
    Visualize comparison between predictions and targets using smooth continuous contour plots.
    
    Args:
        predictions: Predicted flow fields [timesteps, nodes, features] or [nodes, features]
        targets: Target flow fields [timesteps, nodes, features] or [nodes, features]
        node_positions: Node positions [num_nodes, 2]
        timestep: Timestep to visualize
        feature_idx: Feature index to visualize
        save_path: Path to save figure (optional)
        grid_resolution: Resolution for grid interpolation
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(node_positions, torch.Tensor):
        node_positions = node_positions.detach().cpu().numpy()
    
    # Extract specific timestep and feature
    if predictions.ndim == 3:
        pred_values = predictions[timestep, :, feature_idx]
        target_values = targets[timestep, :, feature_idx]
    else:
        pred_values = predictions[:, feature_idx]
        target_values = targets[:, feature_idx]
    
    # Compute error
    error = np.abs(pred_values - target_values)
    
    # Determine colormap and value range
    vmin = min(pred_values.min(), target_values.min())
    vmax = max(pred_values.max(), target_values.max())
    cmap = 'viridis'
    
    # Calculate domain aspect ratio for proper figure sizing
    x_range = node_positions[:, 0].max() - node_positions[:, 0].min()
    y_range = node_positions[:, 1].max() - node_positions[:, 1].min()
    aspect_ratio = x_range / y_range if y_range > 0 else 1.0
    
    # Calculate figure size based on aspect ratio (maintain reasonable size)
    base_height = 5.0
    base_width = 6.0  # Width per subplot
    total_width = base_width * 3  # 3 subplots
    total_height = base_height
    
    # Adjust height to match aspect ratio if needed (but keep reasonable limits)
    if aspect_ratio > 2.0:
        # Very wide domain - keep height reasonable
        total_height = base_height
    elif aspect_ratio < 0.5:
        # Very tall domain - increase height
        total_height = base_height * (1.0 / aspect_ratio)
        total_height = min(total_height, 10.0)  # Cap at reasonable max
    
    # Interpolate all fields to grid for continuous visualization (high resolution for smoothness)
    X_pred, Y_pred, Z_pred = interpolate_to_grid(node_positions, pred_values, grid_resolution)
    X_target, Y_target, Z_target = interpolate_to_grid(node_positions, target_values, grid_resolution)
    X_error, Y_error, Z_error = interpolate_to_grid(node_positions, error, grid_resolution)
    
    # Create subplots with proper aspect ratio
    fig, axes = plt.subplots(1, 3, figsize=(total_width, total_height))
    
    # Prediction - use more levels for smoother appearance
    contour1 = axes[0].contourf(
        X_pred, Y_pred, Z_pred,
        levels=100,  # Increased from 50 for smoother appearance
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend='both'
    )
    axes[0].contour(
        X_pred, Y_pred, Z_pred,
        levels=30,  # Increased from 20
        colors='black',
        alpha=0.1,  # Reduced opacity for smoother look
        linewidths=0.2
    )
    plt.colorbar(contour1, ax=axes[0], label='Value', fraction=0.046, pad=0.04)
    axes[0].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X', fontsize=10)
    axes[0].set_ylabel('Y', fontsize=10)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True, alpha=0.3)
    
    # Target - use more levels for smoother appearance
    contour2 = axes[1].contourf(
        X_target, Y_target, Z_target,
        levels=100,  # Increased from 50 for smoother appearance
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extend='both'
    )
    axes[1].contour(
        X_target, Y_target, Z_target,
        levels=30,  # Increased from 20
        colors='black',
        alpha=0.1,  # Reduced opacity for smoother look
        linewidths=0.2
    )
    plt.colorbar(contour2, ax=axes[1], label='Value', fraction=0.046, pad=0.04)
    axes[1].set_title('Target', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X', fontsize=10)
    axes[1].set_ylabel('Y', fontsize=10)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].grid(True, alpha=0.3)
    
    # Error - use more levels for smoother appearance
    contour3 = axes[2].contourf(
        X_error, Y_error, Z_error,
        levels=100,  # Increased from 50 for smoother appearance
        cmap='Reds',
        extend='both'
    )
    axes[2].contour(
        X_error, Y_error, Z_error,
        levels=30,  # Increased from 20
        colors='black',
        alpha=0.1,  # Reduced opacity for smoother look
        linewidths=0.2
    )
    plt.colorbar(contour3, ax=axes[2], label='Error', fraction=0.046, pad=0.04)
    axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X', fontsize=10)
    axes[2].set_ylabel('Y', fontsize=10)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    
    plt.close()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    metrics,
    filepath,
    is_best=False
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

