"""
3D Tracking Error Analysis

This script provides a comprehensive framework for analyzing the accuracy of a 3D
tracking system against a known ground truth. It loads 3D coordinate data from a
JSON file, compares these tracked points against a generated reference grid,
and computes detailed error metrics.

The analysis includes:
- Calculating Euclidean distance errors between tracked points and the nearest grid points.
- Differentiating between matched and unmatched points based on a distance threshold.
- Computing statistical measures of the error (mean, median, std dev, RMSE) for
  both overall distance and individual axes (X, Y, Z).
- Generating a detailed text-based report summarizing the findings.
- Creating a multi-panel visualization that includes a top-down 2D view and a 3D
  perspective of the tracking errors, which is saved in both PNG and PDF formats.

The script is configurable via command-line arguments for parameters such as
grid dimensions, spacing, and input file paths.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.colors import Normalize
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualization'))
from reference_grid_visualizer import generate_grid

def load_3d_coordinates(json_path: str) -> list:
    """
    Loads 3D coordinate data from a specified JSON file.

    Args:
        json_path (str): The file path to the 3D coordinates JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary contains a
              timestamp, capture name, and the detected 3D points.
              Returns an empty list if loading fails.
              
    Raises:
        FileNotFoundError: If the specified json_path does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading 3D coordinates: {e}")
        return []

def generate_reference_grid(center_x: float = 0, center_y: float = 0, grid_size: int = 7, spacing: float = 0.5) -> np.ndarray:
    """
    Generates a 2D reference grid and projects it to 3D space with Z=0.

    This function utilizes the 'grid_generator' module to create the base grid.
    Note: Two specific points, [1, -1.5, 0] and [1.5, -1.5, 0], are hardcoded for
    exclusion, likely corresponding to known invalid positions in an
    experimental setup.

    Args:
        center_x (float): The x-coordinate for the center of the grid.
        center_y (float): The y-coordinate for the center of the grid.
        grid_size (int): The number of points along each dimension of the grid.
        spacing (float): The distance between adjacent points in the grid.

    Returns:
        np.ndarray: A NumPy array of shape (n, 3) containing the XYZ coordinates
                    of the reference grid points.
    """
    X, Y = generate_grid(center_x, center_y, grid_size, spacing)
    
    # Flatten grid coordinates into a list of [X, Y, Z] points, assuming Z=0.
    grid_points = [[X[i, j], Y[i, j], 0] for i in range(grid_size) for j in range(grid_size)]

    # Exclude specific points known to be invalid for this analysis.
    filtered_points = [
        point for point in grid_points
        if point not in [[1, -1.5, 0], [1.5, -1.5, 0]]
    ]

    return np.array(filtered_points)

def find_nearest_grid_point(tracked_point: np.ndarray, grid_points: np.ndarray) -> tuple:
    """
    Finds the closest reference grid point to a given tracked 3D point.

    The search is performed in the XY plane, ignoring the Z-axis, to find the
    closest horizontal correspondence.

    Args:
        tracked_point (np.ndarray): A 1D array representing the [X, Y, Z]
                                    coordinates of the tracked point.
        grid_points (np.ndarray): A 2D array of reference grid points.

    Returns:
        tuple: A tuple containing:
               - np.ndarray: The nearest grid point [X, Y, Z].
               - float: The Euclidean distance in the XY plane.
               - int: The index of the nearest grid point in the input array.
    """
    # Compute Euclidean distance in the XY plane only.
    distances = np.linalg.norm(grid_points[:, :2] - tracked_point[:2], axis=1)
    nearest_idx = np.argmin(distances)
    
    return grid_points[nearest_idx], distances[nearest_idx], nearest_idx

def analyze_tracking_error(tracked_points_data: list, grid_points: np.ndarray, max_distance: float = 0.5) -> tuple:
    """
    Analyzes and quantifies the error between tracked points and the reference grid.

    This function iterates through all tracked points, finds their nearest grid
    correspondence, and classifies them as "matched" or "unmatched" based on the
    `max_distance` threshold. It then computes detailed error statistics.

    Args:
        tracked_points_data (list): The list of tracked points data from the JSON file.
        grid_points (np.ndarray): The array of reference grid points.
        max_distance (float): The maximum distance in meters for a tracked point
                              to be considered a valid match to a grid point.

    Returns:
        tuple: A tuple containing:
               - list: A list of dictionaries for each matched pair.
               - list: A list of Euclidean distance errors for matched pairs.
               - dict: A dictionary of comprehensive error statistics.
               - list: A list of dictionaries for each unmatched point.
    """
    matched_pairs = []
    errors = []
    unmatched_points = []
    
    # Aggregate all tracked points from all captures into a single list.
    all_tracked_points = []
    capture_indices = []
    for capture_idx, capture in enumerate(tracked_points_data):
        for point in capture['points']:
            all_tracked_points.append(point)
            capture_indices.append(capture_idx)
    
    if not all_tracked_points:
        return [], [], {}, []

    all_tracked_points = np.array(all_tracked_points)
    
    # Associate each tracked point with its nearest grid point.
    for i, point in enumerate(all_tracked_points):
        nearest_grid_point, distance, _ = find_nearest_grid_point(point, grid_points)
        
        # Classify as a match if the distance is within the defined threshold.
        if distance <= max_distance:
            error_vector = point - nearest_grid_point
            
            matched_pairs.append({
                'tracked_point': point,
                'grid_point': nearest_grid_point,
                'distance': distance,
                'error_x': error_vector[0],
                'error_y': error_vector[1],
                'error_z': error_vector[2],
                'capture_name': tracked_points_data[capture_indices[i]]['capture_name']
            })
            errors.append(distance)
        else:
            unmatched_points.append({
                'tracked_point': point,
                'nearest_distance': distance,
                'capture_name': tracked_points_data[capture_indices[i]]['capture_name']
            })
    
    # Compute error statistics if any matches were found.
    if errors:
        errors_x = np.array([pair['error_x'] for pair in matched_pairs])
        errors_y = np.array([pair['error_y'] for pair in matched_pairs])
        errors_z = np.array([pair['error_z'] for pair in matched_pairs])

        error_stats = {
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'num_matched': len(errors),
            'num_unmatched': len(unmatched_points),
            'matching_rate': len(errors) / len(all_tracked_points) * 100,
            'mean_error_x': np.mean(errors_x),
            'mean_error_y': np.mean(errors_y),
            'mean_error_z': np.mean(errors_z),
            'std_error_x': np.std(errors_x),
            'std_error_y': np.std(errors_y),
            'std_error_z': np.std(errors_z),
            'rmse_x': np.sqrt(np.mean(errors_x**2)),
            'rmse_y': np.sqrt(np.mean(errors_y**2)),
            'rmse_z': np.sqrt(np.mean(errors_z**2))
        }
    else:
        # Default stats for the case of no matches.
        error_stats = {key: 0 for key in [
            'min_error', 'max_error', 'mean_error', 'median_error', 'std_error',
            'num_matched', 'matching_rate', 'mean_error_x', 'mean_error_y',
            'mean_error_z', 'std_error_x', 'std_error_y', 'std_error_z',
            'rmse_x', 'rmse_y', 'rmse_z'
        ]}
        error_stats['num_unmatched'] = len(unmatched_points)

    return matched_pairs, errors, error_stats, unmatched_points

def visualize_tracking_error(matched_pairs: list, unmatched_points: list, grid_points: np.ndarray, error_stats: dict, output_dir: str = ".", show_plot: bool = True):
    """
    Generates and displays a comprehensive visualization of tracking errors.

    The visualization consists of two subplots:
    1. A top-down 2D plot showing the XY errors.
    2. A 3D plot showing errors in XYZ space.

    The plot indicates reference grid points, matched tracked points (color-coded
    by error magnitude), unmatched points, and lines connecting matched pairs.

    Args:
        matched_pairs (list): A list of matched point data.
        unmatched_points (list): A list of unmatched point data.
        grid_points (np.ndarray): The array of reference grid points.
        error_stats (dict): The dictionary of calculated error statistics.
        output_dir (str): Directory to save output files (default: current directory).
    """
    fig = plt.figure(figsize=(10, 10))
    ax_2d = fig.add_subplot(211)
    ax_3d = fig.add_subplot(212, projection='3d')

    # Plot reference grid points.
    ax_2d.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', marker='o', s=50, label='Grid Points')
    ax_3d.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c='blue', marker='o', s=50, label='Grid Points')

    # Add reference grid lines for clarity in the 2D view.
    for x in sorted(set(grid_points[:, 0])):
        ax_2d.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
    for y in sorted(set(grid_points[:, 1])):
        ax_2d.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        
    # Plot matched points and error vectors if they exist.
    if matched_pairs:
        distances = [pair['distance'] for pair in matched_pairs]
        tracked_points = np.array([pair['tracked_point'] for pair in matched_pairs])

        # Use a colormap to represent error magnitude.
        scatter_3d = ax_3d.scatter(
            tracked_points[:, 0], tracked_points[:, 1], tracked_points[:, 2], 
            c=distances, cmap='viridis_r', s=80, alpha=0.8,
            marker='x', label='Tracked Points (matched)'
        )
        ax_2d.scatter(
            tracked_points[:, 0], tracked_points[:, 1], 
            c=distances, cmap='viridis_r', s=80, alpha=0.8,
            marker='x'
        )
        
        # Draw lines connecting tracked points to their grid matches.
        for pair in matched_pairs:
            tracked, grid = pair['tracked_point'], pair['grid_point']
            ax_2d.plot([tracked[0], grid[0]], [tracked[1], grid[1]], 'r-', alpha=0.3)
            ax_3d.plot([tracked[0], grid[0]], [tracked[1], grid[1]], [tracked[2], grid[2]], 'r-', alpha=0.3)
        
        cbar = fig.colorbar(scatter_3d, ax=ax_3d, pad=0.1)
        cbar.set_label('Error (meters)')
    
    # Plot unmatched points if they exist.
    if unmatched_points:
        unmatched_array = np.array([p['tracked_point'] for p in unmatched_points])
        label = f'Unmatched Points ({len(unmatched_points)})'
        ax_2d.scatter(unmatched_array[:, 0], unmatched_array[:, 1], c='red', marker='x', s=80, alpha=0.5, label=label)
        ax_3d.scatter(unmatched_array[:, 0], unmatched_array[:, 1], unmatched_array[:, 2], c='red', marker='x', s=80, alpha=0.5, label=label)
    
    # Configure plot aesthetics and labels.
    ax_2d.set_xlabel('X (meters)'); ax_2d.set_ylabel('Y (meters)')
    ax_2d.set_title('Top-Down View of Tracking Errors'); ax_2d.axis('equal'); ax_2d.legend()
    
    ax_3d.set_xlabel('X (meters)'); ax_3d.set_ylabel('Y (meters)'); ax_3d.set_zlabel('Z (meters)')
    ax_3d.set_title('3D View of Tracking Errors'); ax_3d.set_zlim(-1, 1); ax_3d.legend()
    
    plt.suptitle('3D Tracking Error Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure to multiple formats
    pdf_path = os.path.join(output_dir, 'tracking_error_analysis.pdf')
    png_path = os.path.join(output_dir, 'tracking_error_analysis.png')
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    print(f"Visualization plots saved: {pdf_path} and {png_path}")
    plt.show()

def generate_error_report(matched_pairs: list, error_stats: dict, output_file: str = None):
    """
    Generates a formatted text report of the error analysis.

    The report includes overall statistics, coordinate-wise statistics, and a
    detailed breakdown of errors for each matched point, grouped by capture name.

    Args:
        matched_pairs (list): A list of matched point data.
        error_stats (dict): The dictionary of calculated error statistics.
        output_file (str, optional): The path to save the report file. If None,
                                     the report is only printed to the console.
    """
    report = [
        "=" * 80,
        "3D TRACKING ERROR ANALYSIS REPORT",
        "=" * 80,
        "",
        "OVERALL ERROR STATISTICS:",
        f"  Mean Error: {error_stats['mean_error']:.3f} meters",
        f"  Median Error: {error_stats['median_error']:.3f} meters",
        f"  Standard Deviation: {error_stats['std_error']:.3f} meters",
        f"  Min/Max Error: {error_stats['min_error']:.3f}m / {error_stats['max_error']:.3f}m",
        f"  Matched / Unmatched Points: {error_stats['num_matched']} / {error_stats['num_unmatched']}",
        f"  Matching Rate: {error_stats['matching_rate']:.1f}%",
        "",
        "COORDINATE-WISE ERROR STATISTICS (Mean | Std Dev | RMSE):",
        f"  X-axis: {error_stats['mean_error_x']:<+7.3f}m | {error_stats['std_error_x']:.3f}m | {error_stats['rmse_x']:.3f}m",
        f"  Y-axis: {error_stats['mean_error_y']:<+7.3f}m | {error_stats['std_error_y']:.3f}m | {error_stats['rmse_y']:.3f}m",
        f"  Z-axis: {error_stats['mean_error_z']:<+7.3f}m | {error_stats['std_error_z']:.3f}m | {error_stats['rmse_z']:.3f}m",
        "",
        "DETAILED ERRORS BY CAPTURE:",
        "-" * 80,
        f"{'Capture Name':<25} | {'Total(m)':<8} | {'Err X(m)':<9} | {'Err Y(m)':<9} | {'Err Z(m)':<9} | {'Grid Point':<18} | {'Tracked Point'}",
        "-" * 80,
    ]
    
    # Group errors by the capture name for organized reporting.
    captures = {}
    for pair in matched_pairs:
        captures.setdefault(pair['capture_name'], []).append(pair)
    
    for capture_name, pairs in sorted(captures.items()):
        for pair in sorted(pairs, key=lambda p: p['distance'], reverse=True):
            grid_str = f"({pair['grid_point'][0]:.2f}, {pair['grid_point'][1]:.2f})"
            tracked_str = f"({pair['tracked_point'][0]:.2f}, {pair['tracked_point'][1]:.2f}, {pair['tracked_point'][2]:.2f})"
            report.append(
                f"{capture_name:<25} | {pair['distance']:<8.3f} | "
                f"{pair['error_x']:<+9.3f} | {pair['error_y']:<+9.3f} | {pair['error_z']:<+9.3f} | "
                f"{grid_str:<18} | {tracked_str}"
            )
    
    report_str = "\n".join(report)
    print(report_str)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_str)
            print(f"\nError report saved to {output_file}")
        except IOError as e:
            print(f"Error saving report to file: {e}")

def main():
    """
    Main execution entry point for the 3D tracking error analysis script.

    Parses command-line arguments, orchestrates the loading of data, grid
    generation, error analysis, and generation of reports and visualizations.
    """
    parser = argparse.ArgumentParser(
        description="Analyze errors between tracked 3D points and a reference grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--coordinates", 
                        default="experiments/paper/ground_truth/people_standing_grid_3d_coordinates.json",
                        help="Path to the JSON file with 3D coordinates.")
    parser.add_argument("--center-x", type=float, default=0,
                        help="X-coordinate of the grid center.")
    parser.add_argument("--center-y", type=float, default=0,
                        help="Y-coordinate of the grid center.")
    parser.add_argument("--grid-size", type=int, default=7,
                        help="Number of points per side for the square grid.")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Spacing between grid points in meters.")
    parser.add_argument("--max-distance", type=float, default=0.5,
                        help="Maximum distance for a point to be considered a match.")
    parser.add_argument("--report", default="tracking_error_report.txt",
                        help="Path to the output error report file.")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save output files (plots and reports).")
    args = parser.parse_args()
    
    # Step 1: Load tracked points data.
    tracked_points_data = load_3d_coordinates(args.coordinates)
    if not tracked_points_data:
        print("No tracked points data found. Exiting.")
        return
    
    # Step 2: Generate the reference ground-truth grid.
    print(f"Generating {args.grid_size}x{args.grid_size} reference grid with {args.spacing}m spacing.")
    grid_points = generate_reference_grid(args.center_x, args.center_y, args.grid_size, args.spacing)
    
    # Step 3: Perform the core error analysis.
    print(f"Analyzing tracking errors with a max matching distance of {args.max_distance}m.")
    matched_pairs, errors, error_stats, unmatched_points = analyze_tracking_error(
        tracked_points_data, grid_points, args.max_distance
    )
    
    # Step 4: Generate and save a text report.
    report_path = args.report if os.path.isabs(args.report) else os.path.join(args.output_dir, args.report)
    generate_error_report(matched_pairs, error_stats, report_path)
    
    # Step 5: Create and save visualizations.
    if matched_pairs or unmatched_points:
        visualize_tracking_error(matched_pairs, unmatched_points, grid_points, error_stats, args.output_dir)
    else:
        print("No matched or unmatched points to visualize.")

if __name__ == "__main__":
    main()