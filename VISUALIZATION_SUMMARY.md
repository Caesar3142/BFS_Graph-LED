# Visualization Summary

## Generated Visualizations

The following visualizations have been created for the pitzDaily backward-facing step flow data:

### 1. Comprehensive Overview (`00_comprehensive_overview.png`)
- Mesh overview showing all 12,225 nodes
- U-velocity evolution at timesteps 0, 50, and 99
- V-velocity evolution at the same timesteps
- Shows the overall flow structure and evolution

### 2. Mesh Overview (`01_mesh_overview.png`)
- Detailed view of the unstructured mesh
- Shows the geometry of the backward-facing step domain

### 3. Field Evolution
- **U-velocity** (`02_u_velocity_evolution.png`): Evolution of x-component velocity
- **V-velocity** (`02_v_velocity_evolution.png`): Evolution of y-component velocity  
- **Pressure** (`02_pressure_evolution.png` and `04_pressure_evolution.png`): Pressure field evolution

### 4. Time Series (`03_time_series.png`)
- Time series plots at selected sample points
- Shows temporal evolution of u, v, and p at different locations

## Data Statistics

- **Number of nodes**: 12,225
- **Number of timesteps**: 100
- **Features**: u-velocity, v-velocity, pressure (3 features)
- **Domain**: Backward-facing step geometry

## Viewing Visualizations

All visualizations are saved in `results/visualizations/`

To view them:
```bash
# On macOS
open results/visualizations/*.png

# On Linux
xdg-open results/visualizations/*.png
```

## Next Steps

To visualize model predictions (after training):
```bash
python visualize_results.py --config configs/backward_step.yaml --checkpoint checkpoints/best_model.pt
```

Or use the evaluation script:
```bash
python evaluate.py --config configs/backward_step.yaml --checkpoint checkpoints/best_model.pt --visualize
```

