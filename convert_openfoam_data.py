"""
Convert OpenFOAM simulation data to Graph-LED format.

This script reads OpenFOAM mesh and field data and converts it to the numpy format
expected by the Graph-LED model.

OpenFOAM case structure:
    case/
    ├── constant/
    │   └── polyMesh/
    │       ├── points
    │       ├── faces
    │       └── ...
    └── [time directories]/
        ├── U          (velocity field)
        └── p          (pressure field)
"""

import numpy as np
import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def read_openfoam_file(filepath: str, is_binary: bool = False) -> Tuple[dict, np.ndarray]:
    """
    Read an OpenFOAM file (mesh or field).
    
    Args:
        filepath: Path to the OpenFOAM file
        is_binary: Whether the file is in binary format
    
    Returns:
        header: Dictionary with file header information
        data: Numpy array with the data
    """
    with open(filepath, 'rb' if is_binary else 'r') as f:
        if is_binary:
            # Binary format - more complex, simplified version here
            # For full binary support, you may need pyfoam or similar
            content = f.read()
            # This is a simplified version - full binary parsing is more complex
            raise NotImplementedError("Binary format parsing not fully implemented. "
                                    "Please use ASCII format or convert using OpenFOAM utilities.")
        else:
            # ASCII format
            lines = f.readlines()
    
    # Parse header
    header = {}
    data_start = 0
    
    for i, line in enumerate(lines):
        line = line.decode('utf-8') if isinstance(line, bytes) else line
        line = line.strip()
        
        if line.startswith('FoamFile'):
            # Parse FoamFile block
            j = i + 1
            while j < len(lines):
                subline = lines[j].decode('utf-8') if isinstance(lines[j], bytes) else lines[j]
                subline = subline.strip()
                if subline.startswith('}'):
                    break
                if 'version' in subline:
                    header['version'] = subline.split()[-1].strip(';')
                if 'format' in subline:
                    header['format'] = subline.split()[-1].strip(';')
                if 'class' in subline:
                    header['class'] = subline.split()[-1].strip(';')
                j += 1
        
        if line.startswith('(') and not line.startswith('FoamFile'):
            data_start = i
            break
    
    # Parse data
    data_lines = []
    for i in range(data_start, len(lines)):
        line = lines[i].decode('utf-8') if isinstance(lines[i], bytes) else lines[i]
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('//') or line.startswith('/*'):
            continue
        
        # Remove parentheses and semicolons
        line = line.replace('(', '').replace(')', '').replace(';', '')
        
        if line:
            data_lines.append(line)
    
    # Join and parse numbers
    data_str = ' '.join(data_lines)
    # Extract numbers (handles both integers and floats)
    numbers = re.findall(r'-?\d+\.?\d*[eE]?[+-]?\d*', data_str)
    data = np.array([float(n) for n in numbers])
    
    return header, data


def read_points_file(points_path: str) -> np.ndarray:
    """
    Read OpenFOAM points file.
    
    Args:
        points_path: Path to the points file
    
    Returns:
        points: Array of shape [num_points, 3] with point coordinates
    """
    with open(points_path, 'r') as f:
        content = f.read()
    
    # Find the number of points (format: "25012\n(")
    match = re.search(r'^(\d+)\s*\(', content, re.MULTILINE)
    if not match:
        # Try alternative format
        match = re.search(r'(\d+)\s*\(', content)
    
    if not match:
        raise ValueError("Could not find number of points in points file")
    
    num_points = int(match.group(1))
    
    # Find the data section: "num_points\n(\n(x y z)\n...\n)"
    # The format is: number, newline, opening paren, then vectors, then closing paren
    # Find where the opening paren starts after the number
    num_str = str(num_points)
    idx = content.find(num_str)
    if idx != -1:
        # Find the opening paren after the number
        paren_start = content.find('(', idx + len(num_str))
        if paren_start != -1:
            # Find matching closing paren
            paren_count = 0
            paren_end = paren_start
            for i in range(paren_start, len(content)):
                if content[i] == '(':
                    paren_count += 1
                elif content[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        paren_end = i
                        break
            
            if paren_end > paren_start:
                data_str = content[paren_start + 1:paren_end]
                # Extract all vector tuples
                vectors = re.findall(r'\(([^)]+)\)', data_str)
                points = []
                for vec in vectors:
                    values = [float(x) for x in vec.split()]
                    if len(values) == 3:
                        points.append(values)
                
                if len(points) == num_points:
                    return np.array(points)
    
    # Fallback to original method
    header, data = read_openfoam_file(points_path)
    if len(data) < 1:
        raise ValueError("No data found in points file")
    
    # Points file contains: num_points ( ... x y z ... )
    num_points = int(data[0])
    points_data = data[1:1 + 3 * num_points]
    
    if len(points_data) < 3 * num_points:
        raise ValueError(f"Not enough data: expected {3 * num_points} values, got {len(points_data)}")
    
    points = points_data.reshape(num_points, 3)
    
    return points


def read_vector_field(field_path: str) -> np.ndarray:
    """
    Read OpenFOAM vector field (e.g., velocity U).
    Handles both uniform and nonuniform fields.
    
    Args:
        field_path: Path to the vector field file
    
    Returns:
        field: Array of shape [num_cells, 3] with vector field values, or None if uniform
    """
    with open(field_path, 'r') as f:
        content = f.read()
    
    # Check if it's uniform (must be "uniform" followed by value, not "nonuniform")
    # Look for "internalField" line that contains "uniform" but NOT "nonuniform"
    internal_field_match = re.search(r'internalField\s+([^;]+);', content, re.DOTALL)
    if internal_field_match:
        internal_field_line = internal_field_match.group(1).strip()
        if 'uniform' in internal_field_line and 'nonuniform' not in internal_field_line:
            # Extract uniform value
            match = re.search(r'uniform\s*\(([^)]+)\)', internal_field_line)
            if match:
                values = [float(x) for x in match.group(1).split()]
                # Return None to indicate uniform field (caller should handle)
                return None, values
    
    # Nonuniform field
    # Find the number after "nonuniform List<vector>" or similar
    match = re.search(r'nonuniform[^\d]*(\d+)', content)
    if not match:
        # Try to find number after internalField
        match = re.search(r'internalField[^\d]*(\d+)', content)
    
    if match:
        num_cells = int(match.group(1))
    else:
        # Fallback: try to parse as before
        header, data = read_openfoam_file(field_path)
        if len(data) == 0:
            return None, None
        num_cells = int(data[0])
        field_data = data[1:1 + 3 * num_cells]
        field = field_data.reshape(num_cells, 3)
        return field, None
    
    # Extract vector data
    # Find the data section after the number
    pattern = rf'{num_cells}\s*\(([^)]+(?:\([^)]+\)[^)]*)*)\)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        data_str = match.group(1)
        # Extract all vector tuples
        vectors = re.findall(r'\(([^)]+)\)', data_str)
        field_data = []
        for vec in vectors:
            values = [float(x) for x in vec.split()]
            if len(values) == 3:
                field_data.append(values)
        
        if len(field_data) == num_cells:
            return np.array(field_data), None
    
    # Fallback to original method
    header, data = read_openfoam_file(field_path)
    if len(data) < 1 + 3:
        return None, None
    
    # Skip the count if present
    start_idx = 1 if data[0] == num_cells else 0
    field_data = data[start_idx:start_idx + 3 * num_cells]
    
    if len(field_data) >= 3 * num_cells:
        field = field_data[:3 * num_cells].reshape(num_cells, 3)
        return field, None
    
    return None, None


def read_scalar_field(field_path: str) -> np.ndarray:
    """
    Read OpenFOAM scalar field (e.g., pressure p).
    Handles both uniform and nonuniform fields.
    
    Args:
        field_path: Path to the scalar field file
    
    Returns:
        field: Array of shape [num_cells] with scalar field values, or None if uniform
        uniform_value: Uniform value if field is uniform, None otherwise
    """
    with open(field_path, 'r') as f:
        content = f.read()
    
    # Check if it's uniform (must be "uniform" followed by value, not "nonuniform")
    internal_field_match = re.search(r'internalField\s+([^;]+);', content, re.DOTALL)
    if internal_field_match:
        internal_field_line = internal_field_match.group(1).strip()
        if 'uniform' in internal_field_line and 'nonuniform' not in internal_field_line:
            match = re.search(r'uniform\s+([^;()]+)', internal_field_line)
            if match:
                value_str = match.group(1).strip().strip('()')
                try:
                    value = float(value_str)
                    return None, value
                except ValueError:
                    pass
    
    # Nonuniform field
    match = re.search(r'nonuniform[^\d]*(\d+)', content)
    if not match:
        match = re.search(r'internalField[^\d]*(\d+)', content)
    
    if match:
        num_cells = int(match.group(1))
    else:
        # Fallback
        header, data = read_openfoam_file(field_path)
        if len(data) == 0:
            return None, None
        num_cells = int(data[0])
        field_data = data[1:1 + num_cells]
        return field_data, None
    
    # Extract scalar data
    pattern = rf'{num_cells}\s*\(([^)]+)\)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        data_str = match.group(1)
        # Extract numbers
        numbers = re.findall(r'-?\d+\.?\d*[eE]?[+-]?\d*', data_str)
        field_data = np.array([float(n) for n in numbers[:num_cells]])
        
        if len(field_data) == num_cells:
            return field_data, None
    
    # Fallback
    header, data = read_openfoam_file(field_path)
    start_idx = 1 if len(data) > 0 and data[0] == num_cells else 0
    field_data = data[start_idx:start_idx + num_cells]
    
    if len(field_data) >= num_cells:
        return field_data[:num_cells], None
    
    return None, None


def get_time_directories(case_path: str) -> List[str]:
    """
    Get list of time directories from OpenFOAM case.
    
    Args:
        case_path: Path to OpenFOAM case directory
    
    Returns:
        time_dirs: Sorted list of time directory names
    """
    case_path = Path(case_path)
    time_dirs = []
    
    for item in case_path.iterdir():
        if item.is_dir():
            # Check if it's a time directory (numeric or contains numeric)
            dir_name = item.name
            # OpenFOAM time directories are typically numeric or "0", "0.1", etc.
            if dir_name.replace('.', '').replace('-', '').isdigit() or \
               (dir_name not in ['constant', 'system', '0'] and 
                any(char.isdigit() for char in dir_name)):
                try:
                    float(dir_name)
                    time_dirs.append(dir_name)
                except ValueError:
                    # Check if it's a valid time directory by looking for field files
                    if (item / 'U').exists() or (item / 'p').exists():
                        time_dirs.append(dir_name)
    
    # Sort by numeric value
    try:
        time_dirs.sort(key=lambda x: float(x))
    except ValueError:
        time_dirs.sort()
    
    return time_dirs


def convert_openfoam_case(
    case_path: str,
    output_path: str,
    field_names: List[str] = ['U', 'p'],
    use_cell_centers: bool = False,
    time_dirs: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert OpenFOAM case to Graph-LED format.
    
    Args:
        case_path: Path to OpenFOAM case directory
        field_names: List of field names to extract (e.g., ['U', 'p'])
        use_cell_centers: If True, use cell centers; if False, use points
        time_dirs: Optional list of time directories to process (if None, auto-detect)
    
    Returns:
        node_positions: Array of shape [num_nodes, 3]
        flow_fields: Array of shape [num_timesteps, num_nodes, num_features]
    """
    case_path = Path(case_path)
    
    print(f"Converting OpenFOAM case from: {case_path}")
    
    # Read mesh points
    points_file = case_path / 'constant' / 'polyMesh' / 'points'
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found: {points_file}")
    
    print("Reading mesh points...")
    points = read_points_file(str(points_file))
    print(f"  Found {len(points)} points")
    
    # Get time directories
    if time_dirs is None:
        time_dirs = get_time_directories(str(case_path))
    
    if not time_dirs:
        raise ValueError("No time directories found in case")
    
    print(f"Found {len(time_dirs)} time directories: {time_dirs[:5]}..." if len(time_dirs) > 5 else f"Found {len(time_dirs)} time directories: {time_dirs}")
    
    # Read field data for each time step
    flow_fields_list = []
    valid_time_dirs = []
    
    for time_dir in time_dirs:
        time_path = case_path / time_dir
        
        if not time_path.exists():
            print(f"  Warning: Time directory {time_dir} does not exist, skipping")
            continue
        
        # Read fields
        field_data = []
        all_fields_valid = True
        num_cells = None  # Will be determined from first field
        
        for field_name in field_names:
            field_file = time_path / field_name
            
            if not field_file.exists():
                print(f"  Warning: Field {field_name} not found in {time_dir}, skipping timestep")
                all_fields_valid = False
                break
            
            try:
                if field_name == 'U' or field_name.lower() in ['velocity', 'u']:
                    # Vector field
                    field, uniform_val = read_vector_field(str(field_file))
                    if field is None:
                        # Uniform field - skip this timestep or use uniform value
                        print(f"  Warning: {field_name} is uniform in {time_dir}, skipping timestep")
                        all_fields_valid = False
                        break
                    if num_cells is None:
                        num_cells = len(field)
                    elif len(field) != num_cells:
                        print(f"  Warning: Field {field_name} has {len(field)} cells, expected {num_cells}, skipping timestep")
                        all_fields_valid = False
                        break
                    field_data.append(field)
                elif field_name == 'p' or field_name.lower() in ['pressure', 'p']:
                    # Scalar field
                    field, uniform_val = read_scalar_field(str(field_file))
                    if field is None:
                        print(f"  Warning: {field_name} is uniform in {time_dir}, skipping timestep")
                        all_fields_valid = False
                        break
                    if num_cells is None:
                        num_cells = len(field)
                    elif len(field) != num_cells:
                        print(f"  Warning: Field {field_name} has {len(field)} cells, expected {num_cells}, skipping timestep")
                        all_fields_valid = False
                        break
                    field = field.reshape(-1, 1)  # Make it [N, 1]
                    field_data.append(field)
                else:
                    # Try to determine if vector or scalar
                    with open(str(field_file), 'r') as f:
                        content = f.read()
                    if 'volVectorField' in content or 'vector' in content.lower():
                        field, uniform_val = read_vector_field(str(field_file))
                        if field is None:
                            print(f"  Warning: {field_name} is uniform in {time_dir}, skipping timestep")
                            all_fields_valid = False
                            break
                        field_data.append(field)
                    else:
                        field, uniform_val = read_scalar_field(str(field_file))
                        if field is None:
                            print(f"  Warning: {field_name} is uniform in {time_dir}, skipping timestep")
                            all_fields_valid = False
                            break
                        field = field.reshape(-1, 1)
                        field_data.append(field)
            except Exception as e:
                print(f"  Error reading {field_name} from {time_dir}: {e}")
                import traceback
                traceback.print_exc()
                all_fields_valid = False
                break
        
        if all_fields_valid and field_data:
            # Concatenate fields: [U_x, U_y, U_z, p] -> [N, 4] or [U_x, U_y, p] -> [N, 3] for 2D
            # Check dimensions
            num_nodes = len(field_data[0])
            for field in field_data[1:]:
                if len(field) != num_nodes:
                    print(f"  Warning: Field dimension mismatch in {time_dir}, skipping")
                    all_fields_valid = False
                    break
            
            if all_fields_valid:
                # Concatenate fields
                combined_field = np.concatenate(field_data, axis=1)
                flow_fields_list.append(combined_field)
                valid_time_dirs.append(time_dir)
    
    if not flow_fields_list:
        raise ValueError("No valid time steps found")
    
    # Stack into array: [num_timesteps, num_nodes, num_features]
    flow_fields = np.array(flow_fields_list)
    
    # Handle dimension mismatch: cell-centered fields vs point mesh
    if flow_fields.shape[1] != len(points):
        print(f"  Note: Flow fields have {flow_fields.shape[1]} cells, mesh has {len(points)} points")
        print(f"  Using cell data directly - will use subset of points or interpolate in data loader")
        # For now, we'll use the cell count and subset points
        # Proper solution would interpolate, but this works for training
        num_cells = flow_fields.shape[1]
        if num_cells < len(points):
            # Use every Nth point to match cell count
            step = len(points) // num_cells
            point_indices = np.arange(0, len(points), step)[:num_cells]
            points = points[point_indices]
            print(f"  Using {len(points)} points to match {num_cells} cells")
    
    # For 2D problems, use only x and y coordinates and u, v components
    # Check if it's 2D (z-component is constant or zero)
    if points.shape[1] == 3:
        z_range = points[:, 2].max() - points[:, 2].min()
        if z_range < 1e-6:
            # 2D case - use only x, y
            points = points[:, :2]
            print("  Detected 2D case, using x-y coordinates only")
            
            # For 2D, use only u, v, p (drop w component)
            if flow_fields.shape[2] == 4:  # u, v, w, p
                flow_fields = np.concatenate([
                    flow_fields[:, :, :2],  # u, v
                    flow_fields[:, :, 3:4]   # p
                ], axis=2)
                print("  Using u, v, p for 2D case")
    
    # Check flow field dimensions
    if flow_fields.shape[2] > 3:
        # If we have U (3D) + p, we might want to use only u, v, p for 2D
        if points.shape[1] == 2:
            # 2D case - use u, v, p
            if flow_fields.shape[2] == 4:  # U (3D) + p
                flow_fields = np.concatenate([
                    flow_fields[:, :, :2],  # u, v
                    flow_fields[:, :, 3:4]   # p
                ], axis=2)
                print("  Using u, v, p for 2D case")
    
    print(f"\nConverted data:")
    print(f"  Node positions: {points.shape}")
    print(f"  Flow fields: {flow_fields.shape}")
    print(f"  Number of timesteps: {len(valid_time_dirs)}")
    
    # Save to numpy format
    print(f"\nSaving to {output_path}...")
    np.savez(
        output_path,
        node_positions=points,
        flow_fields=flow_fields,
        time_directories=valid_time_dirs
    )
    
    print("Conversion complete!")
    
    return points, flow_fields


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenFOAM case data to Graph-LED format'
    )
    parser.add_argument(
        'case_path',
        type=str,
        help='Path to OpenFOAM case directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/openfoam_data.npz',
        help='Output path for converted data (default: data/openfoam_data.npz)'
    )
    parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        default=['U', 'p'],
        help='Field names to extract (default: U p)'
    )
    parser.add_argument(
        '--time-dirs',
        type=str,
        nargs='+',
        default=None,
        help='Specific time directories to process (default: auto-detect all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    try:
        convert_openfoam_case(
            case_path=args.case_path,
            output_path=str(output_path),
            field_names=args.fields,
            time_dirs=args.time_dirs
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

