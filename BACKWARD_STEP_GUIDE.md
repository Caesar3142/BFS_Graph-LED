# Backward-Facing Step: Complete Guide

This guide focuses specifically on the backward-facing step flow problem using Graph-LED.

## Problem Description

The backward-facing step is a classic benchmark problem in fluid dynamics featuring:
- Sudden expansion geometry
- Recirculation zones
- Reattachment points
- Complex flow separation and reattachment dynamics

## Quick Start

### 1. Generate Synthetic Data (for testing)

```bash
source venv/bin/activate
python -c "from generate_synthetic_data import generate_dataset; generate_dataset('backward_step', num_timesteps=200, Re=100)"
```

### 2. Convert OpenFOAM Data (for real simulations)

```bash
python convert_openfoam_data.py /path/to/backward_step_case --output data/backward_step_flow.npz
```

### 3. Train Model

```bash
python train.py --config configs/backward_step.yaml
```

### 4. Evaluate

```bash
python evaluate.py --config configs/backward_step.yaml --checkpoint checkpoints/best_model.pt --visualize
```

## Configuration

The backward-facing step configuration is in `configs/backward_step.yaml`:

```yaml
data:
  path: "data/backward_step_flow.npz"
  sequence_length: 10
  prediction_horizon: 1

model:
  input_dim: 3   # u, v, p
  output_dim: 3
  hidden_dim: 64
  latent_dim: 32

problem:
  name: "backward_step"
  reynolds_number: 100
```

## Expected Data Format

Your data file (`backward_step_flow.npz`) should contain:

- `node_positions`: `[num_nodes, 2]` - x, y coordinates
- `flow_fields`: `[num_timesteps, num_nodes, 3]` - u, v, p fields

## Geometry

Typical backward-facing step geometry:
- Step height: 0.5 (normalized)
- Domain: x ∈ [-1, 5], y ∈ [-1, 1]
- Step located at x = 0, y < step_height

## Key Features to Capture

The model should learn:
1. **Recirculation zone**: Behind the step
2. **Reattachment point**: Where flow reattaches downstream
3. **Wake dynamics**: Time-dependent flow structures
4. **Pressure recovery**: Downstream of the step

## Training Tips

1. **Sequence Length**: Start with 10-20 timesteps
2. **Batch Size**: Adjust based on mesh size (typically 2-8)
3. **Learning Rate**: 0.001 works well, reduce if unstable
4. **Epochs**: 50-100 epochs usually sufficient

## Evaluation Metrics

Key metrics to monitor:
- **MSE/RMSE**: Overall prediction accuracy
- **MAE**: Mean absolute error
- **Per-feature errors**: Check u, v, p separately
- **Visual inspection**: Recirculation zone accuracy

## Troubleshooting

### Poor predictions in recirculation zone
- Increase `hidden_dim` or `latent_dim`
- Increase `num_gnn_layers`
- Use `aggregation: "attention"` in config

### Training instability
- Reduce learning rate
- Increase batch size
- Add gradient clipping (already in code)

### Memory issues
- Reduce batch size
- Reduce sequence length
- Use fewer GNN layers

## Example Workflow

```bash
# 1. Setup
source venv/bin/activate

# 2. Prepare data (choose one):
# Option A: Generate synthetic data
python -c "from generate_synthetic_data import generate_dataset; generate_dataset('backward_step', num_timesteps=200, Re=100)"

# Option B: Convert OpenFOAM data
python convert_openfoam_data.py /path/to/backward_step_case --output data/backward_step_flow.npz

# 3. Train
python train.py --config configs/backward_step.yaml

# 4. Evaluate
python evaluate.py --config configs/backward_step.yaml \
    --checkpoint checkpoints/best_model.pt \
    --visualize

# 5. Check results
ls results/  # View predictions and metrics
```

## Next Steps

- Experiment with different Reynolds numbers
- Try different model architectures
- Analyze attention weights to understand learned dynamics
- Compare with CFD reference solutions

