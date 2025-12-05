"""
Interpolation utilities for coordinate assignment and up-sampling/down-sampling.
Implements nearest-neighbor interpolation as per Section 2.3.2 of the reference.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def nearest_neighbor_interpolation(
    source_coords, 
    target_coords, 
    source_features
):
    """
    Nearest-neighbor interpolation: I_{X2}^{X1}(U)
    
    Maps features from source coordinates X1 to target coordinates X2
    using nearest-neighbor interpolation.
    
    Args:
        source_coords: Source coordinates [N1, dim] (numpy or torch)
        target_coords: Target coordinates [N2, dim] (numpy or torch)
        source_features: Source features [N1, feature_dim] or [feature_dim, N1] (numpy or torch)
    
    Returns:
        target_features: Interpolated features [N2, feature_dim] or [feature_dim, N2]
    """
    # Convert to numpy if needed
    is_torch = isinstance(source_features, torch.Tensor)
    
    if is_torch:
        source_coords_np = source_coords.detach().cpu().numpy()
        target_coords_np = target_coords.detach().cpu().numpy()
        source_features_np = source_features.detach().cpu().numpy()
        device = source_features.device
        dtype = source_features.dtype
    else:
        source_coords_np = np.asarray(source_coords)
        target_coords_np = np.asarray(target_coords)
        source_features_np = np.asarray(source_features)
    
    # Handle feature dimension: [N1, feature_dim] or [feature_dim, N1]
    if source_features_np.shape[0] == source_coords_np.shape[0]:
        # [N1, feature_dim]
        feature_dim = source_features_np.shape[1]
        transpose = False
    else:
        # [feature_dim, N1]
        source_features_np = source_features_np.T
        feature_dim = source_features_np.shape[1]
        transpose = True
    
    # Build k-d tree for nearest neighbor search
    tree = cKDTree(source_coords_np)
    
    # Find nearest neighbors for target coordinates
    _, indices = tree.query(target_coords_np, k=1)
    
    # Interpolate features (nearest neighbor)
    target_features_np = source_features_np[indices]  # [N2, feature_dim]
    
    # Transpose back if needed
    if transpose:
        target_features_np = target_features_np.T
    
    # Convert back to torch if needed
    if is_torch:
        target_features = torch.from_numpy(target_features_np).to(device=device, dtype=dtype)
    else:
        target_features = target_features_np
    
    return target_features


def interpolate_to_coordinates(source_coords, target_coords, source_features):
    """
    Wrapper function for interpolation operation I_{X2}^{X1}(U).
    
    This is the main interface for coordinate assignment and interpolation
    as described in Section 2.3.2.
    
    Args:
        source_coords: Source coordinates X1 [N1, dim]
        target_coords: Target coordinates X2 [N2, dim]
        source_features: Source features U [N1, feature_dim]
    
    Returns:
        target_features: Interpolated features [N2, feature_dim]
    """
    return nearest_neighbor_interpolation(source_coords, target_coords, source_features)
