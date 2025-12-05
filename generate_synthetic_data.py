"""
Script to generate synthetic flow field data for testing.
This creates example data in the expected format for training.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def generate_cylinder_mesh(nx=50, ny=50, radius=0.5, center=(0, 0)):
    """Generate mesh around a cylinder."""
    x_min, x_max = -2, 8
    y_min, y_max = -2, 2
    
    # Create grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Remove points inside cylinder
    cx, cy = center
    mask = (X - cx)**2 + (Y - cy)**2 > radius**2
    
    node_positions = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    
    return node_positions


def generate_backward_step_mesh(nx=50, ny=50, step_height=0.5):
    """Generate mesh for backward-facing step."""
    x_min, x_max = -1, 5
    y_min, y_max = -1, 1
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Remove points in step region
    mask = ~((X < 0) & (Y < step_height))
    
    node_positions = np.column_stack([X[mask].ravel(), Y[mask].ravel()])
    
    return node_positions


def generate_flow_field(node_positions, timestep, problem='cylinder', Re=100):
    """
    Generate synthetic flow field data.
    This is a simplified example - in practice, you would use actual CFD simulation data.
    """
    num_nodes = node_positions.shape[0]
    flow_fields = np.zeros((num_nodes, 3))  # u, v, p
    
    if problem == 'cylinder':
        # Simplified flow past cylinder
        x, y = node_positions[:, 0], node_positions[:, 1]
        
        # Distance from cylinder center
        cx, cy = 0, 0
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Add time-dependent wake
        omega = 0.1 * timestep
        wake = np.sin(omega) * np.exp(-(x - 2) / 2)
        
        # Velocity field (simplified)
        flow_fields[:, 0] = 1.0 - 0.5 * np.exp(-r) + 0.3 * wake  # u
        flow_fields[:, 1] = 0.2 * np.sin(omega) * np.exp(-r)  # v
        flow_fields[:, 2] = 0.5 * (1 - np.exp(-r))  # p
        
    elif problem == 'backward_step':
        # Simplified backward-facing step flow
        x, y = node_positions[:, 0], node_positions[:, 1]
        
        omega = 0.1 * timestep
        
        # Recirculation zone
        recirc = np.exp(-((x - 1)**2 + (y + 0.25)**2) / 0.5)
        
        flow_fields[:, 0] = 1.0 - 0.3 * recirc + 0.2 * np.sin(omega)  # u
        flow_fields[:, 1] = 0.3 * recirc * np.sin(omega)  # v
        flow_fields[:, 2] = 0.3 * recirc  # p
    
    return flow_fields


def generate_dataset(problem='cylinder', num_timesteps=200, Re=100):
    """Generate complete dataset."""
    print(f'Generating {problem} dataset...')
    
    # Generate mesh
    if problem == 'cylinder':
        node_positions = generate_cylinder_mesh(nx=60, ny=40)
    elif problem == 'backward_step':
        node_positions = generate_backward_step_mesh(nx=60, ny=40)
    else:
        raise ValueError(f"Unknown problem: {problem}")
    
    print(f'Generated mesh with {node_positions.shape[0]} nodes')
    
    # Generate flow fields
    flow_fields = []
    for t in range(num_timesteps):
        flow_field = generate_flow_field(node_positions, t, problem, Re)
        flow_fields.append(flow_field)
    
    flow_fields = np.array(flow_fields)  # [num_timesteps, num_nodes, num_features]
    
    print(f'Generated {num_timesteps} timesteps of flow data')
    print(f'Flow field shape: {flow_fields.shape}')
    
    # Save dataset
    output_path = f'data/{problem}_flow.npz'
    np.savez(
        output_path,
        node_positions=node_positions,
        flow_fields=flow_fields
    )
    
    print(f'Saved dataset to {output_path}')
    
    # Visualize first and last timestep
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, t in enumerate([0, num_timesteps - 1]):
        scatter = axes[idx].scatter(
            node_positions[:, 0],
            node_positions[:, 1],
            c=flow_fields[t, :, 0],  # u-velocity
            cmap='viridis',
            s=10
        )
        plt.colorbar(scatter, ax=axes[idx])
        axes[idx].set_title(f'u-velocity at t={t}')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        axes[idx].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'data/{problem}_visualization.png', dpi=150)
    print(f'Saved visualization to data/{problem}_visualization.png')
    plt.close()
    
    return node_positions, flow_fields


if __name__ == '__main__':
    import os
    import sys
    
    os.makedirs('data', exist_ok=True)
    
    # Focus on backward-facing step
    problem = 'backward_step'
    if len(sys.argv) > 1:
        problem = sys.argv[1]
    
    print(f'Generating {problem} dataset...')
    generate_dataset(problem, num_timesteps=200, Re=100)
    
    print(f'\nDone! Dataset saved to data/{problem}_flow.npz')
    print('You can now train the model with:')
    print(f'  python train.py --config configs/{problem}.yaml')

