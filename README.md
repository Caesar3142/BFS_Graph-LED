# Graph-LED: Graph-based Learning of Effective Dynamics

This repository implements the Graph-LED framework for learning effective dynamics from fluid flow simulations using Graph Neural Networks (GNNs) and attention-based autoregressive models.

**Focus: Backward-Facing Step Flow Problem**

## Overview

Graph-LED combines:
- **GNN-based dimensionality reduction** for variable-size unstructured meshes
- **Autoregressive temporal attention model** for learning temporal dependencies

## Features

- Handles unstructured meshes with complex geometries
- Supports variable-size meshes
- **Focused on backward-facing step flow** with recirculation zones and reattachment dynamics

## Installation

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
# venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you encounter "externally-managed-environment" error on macOS, use a virtual environment as shown above. If you prefer not to use a virtual environment, you can use:
```bash
python3 -m pip install --user -r requirements.txt
```

### 3. Activate virtual environment before running scripts

Always activate the virtual environment before running training or evaluation:
```bash
source venv/bin/activate  # On macOS/Linux
```

## Quick Start (Backward-Facing Step)

### 1. Generate or Convert Data

**Synthetic data (for testing):**
```bash
source venv/bin/activate
python -c "from generate_synthetic_data import generate_dataset; generate_dataset('backward_step', num_timesteps=200, Re=100)"
```

**OpenFOAM data:**
```bash
python convert_openfoam_data.py /path/to/backward_step_case --output data/backward_step_flow.npz
```

### 2. Train Model

```bash
python train.py --config configs/backward_step.yaml
```

### 3. Evaluate

```bash
python evaluate.py --config configs/backward_step.yaml --checkpoint checkpoints/best_model.pt --visualize
```

**📖 For detailed backward-facing step guide, see [BACKWARD_STEP_GUIDE.md](BACKWARD_STEP_GUIDE.md)**

### Quick Start Script

For convenience, use the automated script:

```bash
chmod +x quick_start_backward_step.sh
./quick_start_backward_step.sh
```

This interactive script guides you through:
1. Generating synthetic data
2. Converting OpenFOAM data
3. Training the model
4. Evaluating the model
5. Running the full workflow

### Using OpenFOAM Data

**📖 For detailed instructions, see [OPENFOAM_GUIDE.md](OPENFOAM_GUIDE.md)**

The code expects data in numpy format with:
- `node_positions`: Array of shape `[num_nodes, 2]` or `[num_nodes, 3]` (x, y, z coordinates)
- `flow_fields`: Array of shape `[num_timesteps, num_nodes, num_features]` (typically u, v, p)

#### Method 1: Direct OpenFOAM Conversion (Recommended)

Convert your OpenFOAM case directly:

```bash
python convert_openfoam_data.py /path/to/openfoam/case --output data/my_case.npz --fields U p
```

**OpenFOAM Case Structure:**
```
case/
├── constant/
│   └── polyMesh/
│       └── points          # Mesh point coordinates
└── [time directories]/     # e.g., 0, 0.1, 0.2, ...
    ├── U                   # Velocity field
    └── p                   # Pressure field
```

**Example:**
```bash
# Convert cylinder case
python convert_openfoam_data.py /path/to/cylinder_case --output data/cylinder_flow.npz

# Convert with specific time directories
python convert_openfoam_data.py /path/to/case --output data/case.npz --time-dirs 0 0.1 0.2 0.3

# Convert with custom fields
python convert_openfoam_data.py /path/to/case --output data/case.npz --fields U p T
```

#### Method 2: Via VTK Format

If you prefer to export OpenFOAM data to VTK first:

1. Export from OpenFOAM:
   ```bash
   paraFoam -case /path/to/case
   # Export to VTK in ParaView
   ```

2. Convert VTK files:
   ```bash
   python convert_openfoam_vtk.py /path/to/vtk/files --output data/case.npz
   ```

**Note:** For VTK conversion, you'll need to install VTK:
```bash
pip install vtk
```

#### Method 3: Manual Conversion

If you have your own data processing pipeline, ensure the output format is:

```python
import numpy as np

# Save your data
np.savez(
    'data/my_case.npz',
    node_positions=points,      # [N, 2] or [N, 3]
    flow_fields=flow_fields     # [T, N, F] where F is typically 3 (u, v, p)
)
```

Then update your config file:
```yaml
data:
  path: "data/my_case.npz"
  # ... rest of config
```

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── gnn_encoder.py      # GNN-based encoder for mesh data
│   ├── gnn_decoder.py       # GNN-based decoder
│   ├── temporal_model.py    # Attention-based autoregressive model
│   └── graph_led.py         # Main Graph-LED model
├── data/
│   ├── __init__.py
│   ├── mesh_utils.py        # Mesh processing utilities
│   └── data_loader.py       # Data loading and preprocessing
├── configs/
│   └── backward_step.yaml    # Configuration for backward-facing step
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── convert_openfoam_data.py # OpenFOAM to numpy converter
├── convert_openfoam_vtk.py  # VTK to numpy converter
├── generate_synthetic_data.py # Synthetic data generator
├── quick_start_backward_step.sh # Quick start script
├── BACKWARD_STEP_GUIDE.md   # Detailed guide for backward-facing step
├── OPENFOAM_GUIDE.md        # OpenFOAM conversion guide
└── utils.py                 # Utility functions
```

