# OpenFOAM Data Conversion Guide

This guide explains how to convert OpenFOAM simulation data for use with Graph-LED.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Convert your OpenFOAM case
python convert_openfoam_data.py /path/to/your/openfoam/case --output data/my_case.npz
```

## OpenFOAM Case Structure

Your OpenFOAM case should have the following structure:

```
case_directory/
├── constant/
│   └── polyMesh/
│       ├── points          # REQUIRED: Mesh point coordinates
│       ├── faces
│       ├── owner
│       └── neighbour
└── [time directories]/      # e.g., 0, 0.1, 0.2, 0.5, 1.0, ...
    ├── U                   # Velocity field (vector)
    └── p                   # Pressure field (scalar)
```

## Requirements

1. **Mesh file**: `constant/polyMesh/points` must exist
2. **Field files**: At least one time directory with field files (U, p, etc.)
3. **File format**: ASCII format (binary format support is limited)

## Converting Binary OpenFOAM Files

If your OpenFOAM files are in binary format, convert them to ASCII first:

```bash
# Using OpenFOAM utilities
foamFormatConvert -case /path/to/case

# Or convert specific files
foamFormatConvert -case /path/to/case -file constant/polyMesh/points
foamFormatConvert -case /path/to/case -file 0/U
foamFormatConvert -case /path/to/case -file 0/p
```

## Usage Examples

### Basic Conversion

```bash
python convert_openfoam_data.py /path/to/case --output data/cylinder.npz
```

### Specify Fields

```bash
# Extract only velocity and pressure (default)
python convert_openfoam_data.py /path/to/case --output data/case.npz --fields U p

# Extract velocity, pressure, and temperature
python convert_openfoam_data.py /path/to/case --output data/case.npz --fields U p T
```

### Select Specific Time Steps

```bash
# Convert only specific time directories
python convert_openfoam_data.py /path/to/case \
    --output data/case.npz \
    --time-dirs 0 0.1 0.2 0.3 0.4 0.5
```

### 2D vs 3D Cases

The converter automatically detects 2D cases:
- If all z-coordinates are constant (within tolerance), it uses only x-y coordinates
- For 2D cases, it extracts u, v, p (instead of u, v, w, p)

## Output Format

The converted data is saved as a numpy `.npz` file with:

- `node_positions`: `[num_nodes, 2]` or `[num_nodes, 3]` - Point coordinates
- `flow_fields`: `[num_timesteps, num_nodes, num_features]` - Flow field data
  - For 2D: `[T, N, 3]` where features are [u, v, p]
  - For 3D: `[T, N, 4]` where features are [u, v, w, p]
- `time_directories`: List of time directory names that were processed

## Using the Converted Data

After conversion, update your config file:

```yaml
data:
  path: "data/my_case.npz"  # Path to converted data
  sequence_length: 10
  prediction_horizon: 1
  # ... rest of config
```

Then train:

```bash
python train.py --config configs/cylinder.yaml
```

## Troubleshooting

### Error: "Points file not found"

- Check that `constant/polyMesh/points` exists in your case directory
- Ensure you're pointing to the case root directory (not a subdirectory)

### Error: "No time directories found"

- Check that time directories exist (e.g., `0`, `0.1`, etc.)
- Time directories should contain field files (U, p, etc.)
- You can manually specify time directories with `--time-dirs`

### Error: "Field dimension mismatch"

- This occurs when different fields have different numbers of values
- Common causes:
  - Mixing cell-centered and point-centered fields
  - Incomplete field files
  - Mesh changes between time steps

### Binary Format Issues

If you get errors about binary format:
1. Convert files to ASCII (see above)
2. Or use the VTK conversion method instead

## Alternative: VTK Conversion

If you prefer to export OpenFOAM data to VTK format first:

1. **Export from ParaView:**
   - Open case in ParaView: `paraFoam -case /path/to/case`
   - Export to VTK format (File → Save Data)

2. **Convert VTK files:**
   ```bash
   pip install vtk  # If not already installed
   python convert_openfoam_vtk.py /path/to/vtk/files --output data/case.npz
   ```

## Data Requirements for Training

For best results, ensure:

1. **Sufficient timesteps**: At least 50-100 timesteps recommended
2. **Consistent mesh**: Mesh should not change between time steps
3. **Complete fields**: All required fields (U, p) should be present at all timesteps
4. **Time resolution**: Time steps should be evenly spaced (or at least consistent)

## Example Workflow

```bash
# 1. Convert OpenFOAM case
python convert_openfoam_data.py /path/to/cylinder_case --output data/cylinder.npz

# 2. Update config file
# Edit configs/cylinder.yaml and set:
#   data:
#     path: "data/cylinder.npz"

# 3. Train model
python train.py --config configs/cylinder.yaml

# 4. Evaluate
python evaluate.py --config configs/cylinder.yaml \
    --checkpoint checkpoints/best_model.pt \
    --visualize
```

## Notes

- The converter reads OpenFOAM ASCII format files directly
- It handles both 2D and 3D cases automatically
- Field names are case-sensitive (U, p, not u, P)
- The converter validates data consistency across timesteps

