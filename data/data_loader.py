"""
Data loading and preprocessing for fluid flow datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .mesh_utils import build_graph_from_mesh, normalize_mesh


class FluidFlowDataset(Dataset):
    """
    Dataset for fluid flow simulation data on unstructured meshes.
    
    Expected data format:
    - node_positions: [num_nodes, 2] or [num_nodes, 3]
    - flow_fields: [num_timesteps, num_nodes, num_features]
      where num_features typically includes u, v, p (velocity components and pressure)
    """
    
    def __init__(
        self,
        node_positions,
        flow_fields,
        sequence_length=10,
        prediction_horizon=1,
        normalize=True,
        k_neighbors=8,
        graph_type='knn'
    ):
        """
        Initialize dataset.
        
        Args:
            node_positions: Node coordinates [num_nodes, 2] or [num_nodes, 3]
            flow_fields: Flow field data [num_timesteps, num_nodes, num_features]
            sequence_length: Length of input sequences
            prediction_horizon: Number of future steps to predict
            normalize: Whether to normalize flow fields
            k_neighbors: Number of neighbors for graph construction
            graph_type: 'knn' or 'delaunay'
        """
        self.node_positions = np.asarray(node_positions)
        self.flow_fields = np.asarray(flow_fields)
        
        self.num_timesteps, self.num_nodes, self.num_features = self.flow_fields.shape
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.k_neighbors = k_neighbors
        self.graph_type = graph_type
        
        # Normalize node positions
        self.node_positions_norm, self.pos_min, self.pos_range = normalize_mesh(
            self.node_positions
        )
        
        # Normalize flow fields
        if normalize:
            self.flow_mean = self.flow_fields.mean(axis=(0, 1), keepdims=True)
            self.flow_std = self.flow_fields.std(axis=(0, 1), keepdims=True)
            self.flow_std[self.flow_std == 0] = 1  # Avoid division by zero
            self.flow_fields_norm = (self.flow_fields - self.flow_mean) / self.flow_std
        else:
            self.flow_mean = np.zeros((1, 1, self.num_features))
            self.flow_std = np.ones((1, 1, self.num_features))
            self.flow_fields_norm = self.flow_fields
        
        # Build graph
        if graph_type == 'knn':
            from .mesh_utils import build_graph_from_mesh
            self.edge_index, self.edge_attr = build_graph_from_mesh(
                self.node_positions_norm,
                k_neighbors=k_neighbors
            )
        elif graph_type == 'delaunay':
            from .mesh_utils import build_delaunay_graph
            self.edge_index = build_delaunay_graph(self.node_positions_norm)
            self.edge_attr = None
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")
        
        # Compute valid sequence indices
        self.valid_indices = []
        for i in range(self.num_timesteps - sequence_length - prediction_horizon + 1):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Extract input sequence
        input_sequence = self.flow_fields_norm[
            start_idx:start_idx + self.sequence_length
        ]  # [sequence_length, num_nodes, num_features]
        
        # Extract target sequence
        target_sequence = self.flow_fields_norm[
            start_idx + self.sequence_length:
            start_idx + self.sequence_length + self.prediction_horizon
        ]  # [prediction_horizon, num_nodes, num_features]
        
        # Convert to torch tensors
        input_sequence = torch.from_numpy(input_sequence).float()
        target_sequence = torch.from_numpy(target_sequence).float()
        node_positions = torch.from_numpy(self.node_positions_norm).float()
        
        return {
            'input_sequence': input_sequence,
            'target_sequence': target_sequence,
            'node_positions': node_positions,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr if self.edge_attr is not None else None
        }
    
    def denormalize(self, flow_fields):
        """Denormalize flow fields back to original scale."""
        if isinstance(flow_fields, torch.Tensor):
            flow_fields = flow_fields.numpy()
        
        return flow_fields * self.flow_std + self.flow_mean


def create_dataloader(
    node_positions,
    flow_fields,
    sequence_length=10,
    prediction_horizon=1,
    batch_size=1,
    shuffle=True,
    normalize=True,
    k_neighbors=8,
    graph_type='knn',
    train_split=0.8,
    val_split=0.1
):
    """
    Create train/val/test dataloaders from flow field data.
    
    Returns:
        train_loader, val_loader, test_loader, dataset_stats
    """
    dataset = FluidFlowDataset(
        node_positions=node_positions,
        flow_fields=flow_fields,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        normalize=normalize,
        k_neighbors=k_neighbors,
        graph_type=graph_type
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    dataset_stats = {
        'flow_mean': dataset.flow_mean,
        'flow_std': dataset.flow_std,
        'num_nodes': dataset.num_nodes,
        'num_features': dataset.num_features,
        'edge_index': dataset.edge_index,
        'node_positions': dataset.node_positions_norm
    }
    
    return train_loader, val_loader, test_loader, dataset_stats

