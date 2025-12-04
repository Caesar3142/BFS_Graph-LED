"""
Utilities for processing unstructured mesh data and building graphs.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def build_graph_from_mesh(
    node_positions,
    k_neighbors=8,
    max_distance=None,
    self_loops=False
):
    """
    Build graph connectivity from mesh node positions.
    
    Args:
        node_positions: Array of shape [num_nodes, 2] or [num_nodes, 3]
        k_neighbors: Number of nearest neighbors to connect
        max_distance: Maximum distance for edge connections (optional)
        self_loops: Whether to include self-loops
    
    Returns:
        edge_index: Tensor of shape [2, num_edges]
        edge_attr: Optional edge attributes (distances)
    """
    node_positions = np.asarray(node_positions)
    num_nodes = node_positions.shape[0]
    
    # Build k-d tree for efficient nearest neighbor search
    tree = cKDTree(node_positions)
    
    # Find k nearest neighbors for each node
    if max_distance is not None:
        distances, indices = tree.query(
            node_positions, 
            k=min(k_neighbors + 1, num_nodes),
            distance_upper_bound=max_distance
        )
    else:
        distances, indices = tree.query(
            node_positions, 
            k=min(k_neighbors + 1, num_nodes)
        )
    
    # Build edge list
    edges = []
    edge_distances = []
    
    for i in range(num_nodes):
        neighbors = indices[i]
        dists = distances[i]
        
        # Filter out invalid neighbors (infinite distance)
        valid_mask = np.isfinite(dists)
        neighbors = neighbors[valid_mask]
        dists = dists[valid_mask]
        
        # Remove self-connections if not desired
        if not self_loops:
            mask = neighbors != i
            neighbors = neighbors[mask]
            dists = dists[mask]
        
        # Add edges
        for j, neighbor in enumerate(neighbors):
            if neighbor < num_nodes:  # Safety check
                edges.append([i, neighbor])
                edge_distances.append(dists[j])
    
    # Convert to edge_index format [2, num_edges]
    if len(edges) == 0:
        # Fallback: connect to nearest neighbor only
        for i in range(num_nodes):
            if i < num_nodes - 1:
                edges.append([i, i + 1])
                edge_distances.append(np.linalg.norm(node_positions[i] - node_positions[i + 1]))
    
    edge_index = np.array(edges).T
    edge_attr = np.array(edge_distances)
    
    # Remove duplicate edges (undirected graph)
    edge_index, unique_indices = np.unique(edge_index, axis=1, return_index=True)
    edge_attr = edge_attr[unique_indices]
    
    # Convert to torch tensors
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_attr).float()
    
    return edge_index, edge_attr


def build_delaunay_graph(node_positions):
    """
    Build graph using Delaunay triangulation (for 2D meshes).
    
    Args:
        node_positions: Array of shape [num_nodes, 2]
    
    Returns:
        edge_index: Tensor of shape [2, num_edges]
    """
    from scipy.spatial import Delaunay
    
    node_positions = np.asarray(node_positions)
    tri = Delaunay(node_positions)
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([simplex[i], simplex[j]]))
            edges.add(edge)
    
    edge_list = list(edges)
    edge_index = np.array(edge_list).T
    
    return torch.from_numpy(edge_index).long()


def normalize_mesh(node_positions):
    """Normalize mesh coordinates to [0, 1] range."""
    node_positions = np.asarray(node_positions)
    min_coords = node_positions.min(axis=0)
    max_coords = node_positions.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1  # Avoid division by zero
    
    normalized = (node_positions - min_coords) / range_coords
    return normalized, min_coords, range_coords


def load_mesh_data(filepath, format='numpy'):
    """
    Load mesh data from file.
    Supports various formats (numpy, hdf5, etc.)
    """
    if format == 'numpy':
        data = np.load(filepath, allow_pickle=True)
        return data
    elif format == 'hdf5':
        import h5py
        with h5py.File(filepath, 'r') as f:
            data = {
                'positions': f['positions'][:],
                'flow_fields': f['flow_fields'][:] if 'flow_fields' in f else None
            }
        return data
    else:
        raise ValueError(f"Unsupported format: {format}")


def extract_boundary_nodes(node_positions, boundary_threshold=0.01):
    """
    Identify boundary nodes based on proximity to domain boundaries.
    
    Args:
        node_positions: Array of shape [num_nodes, 2]
        boundary_threshold: Distance threshold for boundary detection
    
    Returns:
        boundary_mask: Boolean array indicating boundary nodes
    """
    node_positions = np.asarray(node_positions)
    min_coords = node_positions.min(axis=0)
    max_coords = node_positions.max(axis=0)
    
    boundary_mask = np.zeros(len(node_positions), dtype=bool)
    for dim in range(node_positions.shape[1]):
        boundary_mask |= (node_positions[:, dim] <= min_coords[dim] + boundary_threshold)
        boundary_mask |= (node_positions[:, dim] >= max_coords[dim] - boundary_threshold)
    
    return boundary_mask

