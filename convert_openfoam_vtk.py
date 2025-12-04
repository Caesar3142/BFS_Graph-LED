"""
Alternative: Convert OpenFOAM data via VTK format.

If you have OpenFOAM data exported to VTK format (using paraFoam or similar),
this script can convert VTK files to Graph-LED format.

Usage:
1. Export OpenFOAM data to VTK:
   paraFoam -case <case_path>
   # Then export to VTK format

2. Or use this script to read VTK files directly
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Tuple
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("Warning: VTK not available. Install with: pip install vtk")


def read_vtk_file(vtk_path: str) -> Tuple[np.ndarray, dict]:
    """
    Read VTK file and extract point coordinates and field data.
    
    Args:
        vtk_path: Path to VTK file
    
    Returns:
        points: Array of shape [num_points, 3]
        fields: Dictionary of field arrays
    """
    if not VTK_AVAILABLE:
        raise ImportError("VTK is required. Install with: pip install vtk")
    
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    
    data = reader.GetOutput()
    
    # Get points
    points = vtk_to_numpy(data.GetPoints().GetData())
    
    # Get field data
    fields = {}
    point_data = data.GetPointData()
    
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        array_data = vtk_to_numpy(point_data.GetArray(i))
        fields[array_name] = array_data
    
    return points, fields


def convert_vtk_sequence(
    vtk_dir: str,
    output_path: str,
    field_names: list = ['U', 'p'],
    pattern: str = '*.vtk'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sequence of VTK files to Graph-LED format.
    
    Args:
        vtk_dir: Directory containing VTK files
        output_path: Output path for converted data
        field_names: Names of fields to extract
        pattern: File pattern to match VTK files
    
    Returns:
        node_positions: Array of shape [num_nodes, 3]
        flow_fields: Array of shape [num_timesteps, num_nodes, num_features]
    """
    vtk_dir = Path(vtk_dir)
    vtk_files = sorted(vtk_dir.glob(pattern))
    
    if not vtk_files:
        raise ValueError(f"No VTK files found in {vtk_dir} matching pattern {pattern}")
    
    print(f"Found {len(vtk_files)} VTK files")
    
    # Read first file to get mesh structure
    print("Reading first VTK file to get mesh...")
    points, fields = read_vtk_file(str(vtk_files[0]))
    print(f"  Mesh has {len(points)} points")
    
    # Check available fields
    available_fields = list(fields.keys())
    print(f"  Available fields: {available_fields}")
    
    # Determine which fields to use
    fields_to_use = []
    for field_name in field_names:
        # Try to find matching field (case-insensitive)
        found = False
        for avail_field in available_fields:
            if field_name.lower() in avail_field.lower():
                fields_to_use.append(avail_field)
                found = True
                break
        if not found:
            print(f"  Warning: Field {field_name} not found, skipping")
    
    if not fields_to_use:
        raise ValueError("No valid fields found")
    
    print(f"  Using fields: {fields_to_use}")
    
    # Read all time steps
    flow_fields_list = []
    for vtk_file in vtk_files:
        _, fields = read_vtk_file(str(vtk_file))
        
        # Extract and combine fields
        field_data = []
        for field_name in fields_to_use:
            field = fields[field_name]
            if field.ndim == 1:
                field = field.reshape(-1, 1)
            field_data.append(field)
        
        combined_field = np.concatenate(field_data, axis=1)
        flow_fields_list.append(combined_field)
    
    flow_fields = np.array(flow_fields_list)
    
    # Handle 2D case
    if points.shape[1] == 3:
        z_range = points[:, 2].max() - points[:, 2].min()
        if z_range < 1e-6:
            points = points[:, :2]
            print("  Detected 2D case, using x-y coordinates only")
    
    print(f"\nConverted data:")
    print(f"  Node positions: {points.shape}")
    print(f"  Flow fields: {flow_fields.shape}")
    
    # Save
    np.savez(output_path, node_positions=points, flow_fields=flow_fields)
    print(f"\nSaved to {output_path}")
    
    return points, flow_fields


def main():
    parser = argparse.ArgumentParser(
        description='Convert VTK files to Graph-LED format'
    )
    parser.add_argument(
        'vtk_dir',
        type=str,
        help='Directory containing VTK files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/vtk_data.npz',
        help='Output path (default: data/vtk_data.npz)'
    )
    parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        default=['U', 'p'],
        help='Field names to extract'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.vtk',
        help='File pattern for VTK files (default: *.vtk)'
    )
    
    args = parser.parse_args()
    
    if not VTK_AVAILABLE:
        print("Error: VTK is not installed.")
        print("Install with: pip install vtk")
        return 1
    
    try:
        convert_vtk_sequence(
            vtk_dir=args.vtk_dir,
            output_path=args.output,
            field_names=args.fields,
            pattern=args.pattern
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

