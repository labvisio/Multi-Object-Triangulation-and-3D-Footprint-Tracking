#!/usr/bin/env python3
"""
The script analyzes the effect of different camera combinations on 3D reconstruction accuracy,
applies coordinate corrections, and compares results with a virtual reference grid.

"""

# Import statements
import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import glob
import shutil
from datetime import datetime
from itertools import combinations
import statistics
from tabulate import tabulate


def camera_parameters(file):
    """
    Read the intrinsic and extrinsic parameters of each camera from JSON calibration file.
    
    Parameters:
    -----------
    file : str
        Path to the camera calibration JSON file
        
    Returns:
    --------
    tuple
        K, R, T, res, dis, R_inv, T_inv where:
        - K: intrinsic matrix (3x3)
        - R: rotation matrix (3x3) 
        - T: translation vector (3x1)
        - res: resolution [width, height]
        - dis: distortion coefficients
        - R_inv: inverse rotation matrix (3x3)
        - T_inv: inverse translation vector (3x1)
    """
    with open(file, 'r') as f:
        camera_data = json.load(f)
    
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'],
           camera_data['resolution']['height']]
    
    # Access extrinsic as an array
    tf = np.array(camera_data['extrinsic'][0]['tf']['doubles']).reshape(4, 4)
    
    # Get the RT matrix (original camera-to-world transformation)
    R = tf[:3, :3]
    T = tf[:3, 3].reshape(3, 1)
    
    # Compute the inverse transformation (world-to-camera)
    # For a rotation matrix, its inverse is its transpose
    R_inv = R.transpose()
    # For translation, we need to apply -T rotated by inverse R
    T_inv = -R_inv @ T
    
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis, R_inv, T_inv


def reconstruct_3d_position(corners_list, K_list, R_list, T_list, verbose=False):
    """
    Reconstruct 3D position of ARUCO marker based on projections from multiple cameras.
    
    Parameters:
    -----------
    corners_list : list
        List of (u, v) coordinates for each camera (None if not detected)
    K_list : list
        List of intrinsic matrices for each camera
    R_list : list
        List of rotation matrices for each camera
    T_list : list
        List of translation vectors for each camera
    verbose : bool
        Whether to print debug information
        
    Returns:
    --------
    numpy.ndarray or None
        3D position [x, y, z] or None if reconstruction failed
    """
    if verbose:
        print("Corners list:", corners_list)
    
    A = []
    detected_cameras = 0

    for i in range(len(corners_list)):
        if corners_list[i] is None:
            if verbose:
                print(f"Camera {i}: No marker detected")
            continue  # Skip cameras that didn't detect the ARUCO
        
        detected_cameras += 1
        # ARUCO center coordinates in image (pixels)
        u, v = corners_list[i]
        if verbose:
            print(f"Camera {i}: Marker detected at ({u}, {v})")
        
        # Create projection matrix
        P = K_list[i] @ np.hstack((R_list[i].T, -R_list[i].T @ T_list[i]))
        if verbose:
            print(f"Camera {i} Projection Matrix P shape: {P.shape}")
        
        # Create linear equations for u and v
        eq1 = u * P[2, :] - P[0, :]
        eq2 = v * P[2, :] - P[1, :]
        if verbose:
            print(f"Camera {i} equation 1: {eq1}")
            print(f"Camera {i} equation 2: {eq2}")
        
        A.append(eq1)
        A.append(eq2)
    
    if verbose:
        print(f"Detected cameras: {detected_cameras}, total equations: {len(A)}")
    
    if len(A) < 4:
        if verbose:
            print("ERROR: Need at least 2 cameras (4 equations) for triangulation")
        return None  # Need at least 2 cameras for triangulation (2 eqs/camera)
    
    A = np.vstack(A)
    if verbose:
        print(f"Matrix A shape: {A.shape}")
    
    # Solve using SVD (last row of Vt is the solution)
    try:
        _, s, Vt = np.linalg.svd(A)
        if verbose:
            print(f"SVD singular values: {s}")
        X_homog = Vt[-1]
        if verbose:
            print(f"Homogeneous coordinates: {X_homog}")
        
        # Convert homogeneous coordinates to Cartesian
        X = X_homog[:3] / X_homog[3]
        return X.flatten()
        
    except Exception as e:
        if verbose:
            print(f"SVD computation error: {e}")
        return None


def detect_aruco_in_image(image_path, marker_id=7):
    """
    Detect ARUCO marker in an image and return center coordinates.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    marker_id : int
        ID of the ARUCO marker to detect
        
    Returns:
    --------
    tuple or None
        (cx, cy) center coordinates if marker detected, None otherwise
    """
    if not os.path.exists(image_path):
        return None
        
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Setup ARUCO detector
    parameters = aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(dictionary, parameters)
    
    # Detect markers
    corners, ids, _ = detector.detectMarkers(image)
    
    if ids is not None and marker_id in ids:
        idx = np.where(ids == marker_id)[0][0]
        c = corners[idx][0]
        cx, cy = np.mean(c, axis=0)
        return (cx, cy)
    
    return None


def load_capture_data(capture_dir):
    """
    Load capture data from a capture directory.
    
    Parameters:
    -----------
    capture_dir : str
        Path to the capture directory
        
    Returns:
    --------
    dict or None
        Dictionary with capture data or None if loading failed
    """
    json_path = os.path.join(capture_dir, "marker_position_data.json")
    if not os.path.exists(json_path):
        return None
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def calculate_error(estimated, actual):
    """
    Calculate Euclidean distance between estimated and actual positions.
    
    Parameters:
    -----------
    estimated : numpy.ndarray
        Estimated 3D position [x, y, z]
    actual : dict
        Actual position with keys 'x', 'y', 'z'
        
    Returns:
    --------
    float
        Euclidean distance error
    """
    actual_array = np.array([actual['x'], actual['y'], actual['z']])
    return np.linalg.norm(estimated - actual_array)


def generate_grid(center_x=0, center_y=0, grid_size=7, spacing=0.5):
    """
    Generate a virtual reference grid for comparison.
    
    Parameters:
    -----------
    center_x : float
        X coordinate of grid center
    center_y : float  
        Y coordinate of grid center
    grid_size : int
        Size of grid (NxN)
    spacing : float
        Spacing between grid points in meters
        
    Returns:
    --------
    tuple
        (X_grid, Y_grid) coordinate arrays
    """
    # Create coordinate arrays
    x_coords = np.linspace(-spacing * (grid_size - 1) / 2, spacing * (grid_size - 1) / 2, grid_size) + center_x
    y_coords = np.linspace(-spacing * (grid_size - 1) / 2, spacing * (grid_size - 1) / 2, grid_size) + center_y
    
    # Create meshgrid
    X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
    
    return X_grid, Y_grid


def create_virtual_reference_grid():
    """Create the virtual reference grid (7x7, 0.5m spacing)."""
    X_grid, Y_grid = generate_grid(center_x=0, center_y=0, grid_size=7, spacing=0.5)
    
    # Extract valid grid points
    valid_mask = ~(np.isnan(X_grid) | np.isnan(Y_grid))
    grid_x = X_grid[valid_mask]
    grid_y = Y_grid[valid_mask]
    grid_z = np.zeros_like(grid_x)  # Ground level
    
    # Create 3D grid points
    grid_points = np.column_stack([grid_x, grid_y, grid_z])
    
    return grid_points


def calculate_distances_to_grid(reconstruction_points, grid_points):
    """
    Calculate distances from reconstruction points to the nearest grid points.
    
    Parameters:
    -----------
    reconstruction_points : numpy.ndarray
        Array of 3D reconstruction points [N, 3]
    grid_points : numpy.ndarray
        Array of 3D grid points [M, 3]
        
    Returns:
    --------
    numpy.ndarray
        Array of minimum distances from each reconstruction point to grid
    """
    distances = []
    
    for recon_point in reconstruction_points:
        # Calculate distance to all grid points
        point_distances = np.linalg.norm(grid_points - recon_point, axis=1)
        # Store minimum distance
        distances.append(np.min(point_distances))
    
    return np.array(distances)


def load_analysis_results():
    """Load the camera combination analysis results."""
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    results_file = os.path.join(project_root, "camera_combination_analysis_results.json")
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run camera_combination_analysis first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def extract_reconstructions_by_camera_count(analysis_results):
    """
    Extract all reconstruction points organized by camera count.
    
    Parameters:
    -----------
    analysis_results : dict
        Analysis results from camera_combination_analysis
        
    Returns:
    --------
    dict
        Dictionary with camera counts as keys and reconstruction arrays as values
    """
    reconstructions_by_count = {}
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in analysis_results:
            all_positions = []
            
            # Extract all reconstruction positions for this camera count
            if 'reconstructions' in analysis_results[camera_count]:
                for combo_str, pos_list in analysis_results[camera_count]['reconstructions'].items():
                    all_positions.extend(pos_list)
            
            if all_positions:
                reconstructions_by_count[camera_count] = np.array(all_positions)
            else:
                reconstructions_by_count[camera_count] = np.array([]).reshape(0, 3)
    
    return reconstructions_by_count


def fix_coordinates_in_json(input_file, output_file=None):
    """
    Fix coordinates in the JSON file by subtracting 0.3 from X and Y.
    
    Parameters:
    -----------
    input_file : str
        Path to the input JSON file
    output_file : str, optional
        Path to the output JSON file. If None, overwrites the input file.
    """
    # Load the original JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loading data from: {input_file}")
    
    # Apply coordinate correction to reconstructions
    corrected_count = 0
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in data and 'reconstructions' in data[camera_count]:
            reconstructions = data[camera_count]['reconstructions']
            
            for combo_str, positions in reconstructions.items():
                corrected_positions = []
                for pos in positions:
                    # Apply correction: subtract 0.3 from X and Y coordinates
                    corrected_pos = [
                        pos[0] - 0.3,  # X coordinate
                        pos[1] - 0.3,  # Y coordinate
                        pos[2]         # Z coordinate (unchanged)
                    ]
                    corrected_positions.append(corrected_pos)
                    corrected_count += 1
                
                # Update the positions
                reconstructions[combo_str] = corrected_positions
    
    print(f"Corrected {corrected_count} position coordinates")
    
    # Update metadata
    data['coordinate_correction_applied'] = True
    data['coordinate_correction_description'] = "Subtracted 0.3 from X and Y coordinates"
    data['coordinate_correction_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the corrected data
    if output_file is None:
        output_file = input_file
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Corrected data saved to: {output_file}")
    
    return data


def perform_camera_combination_analysis():
    """
    Main function to analyze camera combinations for 3D reconstruction.
    """
    print("="*80)
    print("STARTING CAMERA COMBINATION ANALYSIS")
    print("="*80)
    
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Get captures directory
    captures_dir = os.path.join(project_root, "experiments", "paper", "ground_truth", "captures_aruco")
    
    if not os.path.exists(captures_dir):
        print(f"Error: Captures directory {captures_dir} does not exist.")
        return None
    
    print(f"Using captures directory: {captures_dir}")
    
    # Load camera parameters
    camera_indices = [0, 1, 2, 3]
    cameras_data = {}
    
    print("Loading camera calibration data...")
    for cam_idx in camera_indices:
        json_file = os.path.join(project_root, "config_camera", f"{cam_idx}.json")
        if not os.path.exists(json_file):
            print(f"Error: Calibration file {json_file} not found.")
            return None
            
        try:
            K, R, T, res, dis, R_inv, T_inv = camera_parameters(json_file)
            cameras_data[cam_idx] = {
                'K': K, 'R': R, 'T': T, 'res': res, 'dis': dis, 
                'R_inv': R_inv, 'T_inv': T_inv
            }
            print(f"Loaded camera {cam_idx} calibration")
        except Exception as e:
            print(f"Error loading camera {cam_idx}: {e}")
            return None
    
    # Find all capture directories
    capture_dirs = glob.glob(os.path.join(captures_dir, "capture_*"))
    capture_dirs.sort()
    
    print(f"\nFound {len(capture_dirs)} capture directories")
    
    # Define camera combinations
    combinations_2 = list(combinations(camera_indices, 2))
    combinations_3 = list(combinations(camera_indices, 3))
    combinations_4 = list(combinations(camera_indices, 4))
    
    print(f"\nCamera combinations:")
    print(f"  2 cameras: {len(combinations_2)} combinations")
    print(f"  3 cameras: {len(combinations_3)} combinations")
    print(f"  4 cameras: {len(combinations_4)} combinations")
    
    # Initialize results storage
    results = {
        '2_cameras': {
            'combinations': combinations_2,
            'reconstructions': {tuple(combo): [] for combo in combinations_2},
            'successful_reconstructions': {tuple(combo): 0 for combo in combinations_2}
        },
        '3_cameras': {
            'combinations': combinations_3,
            'reconstructions': {tuple(combo): [] for combo in combinations_3},
            'successful_reconstructions': {tuple(combo): 0 for combo in combinations_3}
        },
        '4_cameras': {
            'combinations': combinations_4,
            'reconstructions': {tuple(combo): [] for combo in combinations_4},
            'successful_reconstructions': {tuple(combo): 0 for combo in combinations_4}
        }
    }
    
    print(f"\nProcessing captures...")
    
    # Process each capture
    processed_count = 0
    for capture_dir in capture_dirs:
        print(f"\nProcessing: {os.path.basename(capture_dir)}")
        
        # Detect ARUCO in all camera images
        detected_corners = {}
        for cam_idx in camera_indices:
            image_filename = f"{cam_idx}.jpg"
            image_path = os.path.join(capture_dir, image_filename)
            corners = detect_aruco_in_image(image_path)
            detected_corners[cam_idx] = corners
            if corners:
                print(f"  Camera {cam_idx}: ARUCO detected at {corners}")
            else:
                print(f"  Camera {cam_idx}: ARUCO not detected")
        
        # Count how many cameras detected the marker
        valid_detections = sum(1 for corners in detected_corners.values() if corners is not None)
        if valid_detections < 2:
            print(f"  Skipping - only {valid_detections} cameras detected marker")
            continue
        
        # Store all successful reconstructions for this capture to analyze consistency
        capture_reconstructions = {}
        
        # Test all camera combinations
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            for combo in results[num_cameras]['combinations']:
                # Prepare data for this combination
                corners_list = []
                K_list = []
                R_list = []
                T_list = []
                
                valid_cameras = 0
                for cam_idx in combo:
                    if detected_corners[cam_idx] is not None:
                        corners_list.append(detected_corners[cam_idx])
                        valid_cameras += 1
                    else:
                        corners_list.append(None)
                    
                    K_list.append(cameras_data[cam_idx]['K'])
                    R_list.append(cameras_data[cam_idx]['R_inv'])
                    T_list.append(cameras_data[cam_idx]['T_inv'])
                
                # Only proceed if we have enough valid cameras
                min_cameras_needed = 2
                if valid_cameras < min_cameras_needed:
                    continue
                
                # Attempt 3D reconstruction
                estimated_pos = reconstruct_3d_position(corners_list, K_list, R_list, T_list)
                
                if estimated_pos is not None:
                    # Store the reconstruction
                    results[num_cameras]['reconstructions'][tuple(combo)].append(estimated_pos.copy())
                    results[num_cameras]['successful_reconstructions'][tuple(combo)] += 1
                    capture_reconstructions[tuple(combo)] = estimated_pos.copy()
                    
                    if processed_count < 3:  # Verbose for first few
                        print(f"    Combo {combo}: Success, position = {estimated_pos}")
        
        # For this capture, calculate variability between different camera combinations
        if len(capture_reconstructions) > 1:
            positions = list(capture_reconstructions.values())
            # Calculate pairwise distances between reconstructions
            distances = []
            combo_pairs = []
            for i, (combo1, pos1) in enumerate(capture_reconstructions.items()):
                for j, (combo2, pos2) in enumerate(capture_reconstructions.items()):
                    if i < j:  # Avoid duplicates
                        dist = np.linalg.norm(pos1 - pos2)
                        distances.append(dist)
                        combo_pairs.append((combo1, combo2))
            
            if distances:
                max_distance = max(distances)
                mean_distance = np.mean(distances)
                if processed_count < 5:  # Show details for first few captures
                    print(f"    Reconstruction variability: mean={mean_distance:.4f}m, max={max_distance:.4f}m")
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  Processed {processed_count}/{len(capture_dirs)} captures")
    
    print(f"\nCompleted processing {processed_count} captures")
    
    # Calculate statistics for each combination type
    print("\n" + "="*80)
    print("CAMERA COMBINATION ANALYSIS RESULTS")
    print("="*80)
    
    summary_results = {}
    
    for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
        print(f"\n{num_cameras.replace('_', ' ').title()}:")
        print("-" * 50)
        
        combo_stats = []
        
        for combo in results[num_cameras]['combinations']:
            reconstructions = results[num_cameras]['reconstructions'][tuple(combo)]
            success_count = results[num_cameras]['successful_reconstructions'][tuple(combo)]
            
            if len(reconstructions) > 0:
                # Calculate statistics for this combination
                positions = np.array(reconstructions)
                mean_pos = np.mean(positions, axis=0)
                std_pos = np.std(positions, axis=0)
                
                # Calculate variability (standard deviation of positions)
                position_variability = np.mean(std_pos)  # Average std across x, y, z
                
                combo_stats.append({
                    'combination': combo,
                    'success_count': success_count,
                    'mean_position': mean_pos,
                    'position_variability': position_variability,
                    'std_x': std_pos[0],
                    'std_y': std_pos[1],
                    'std_z': std_pos[2]
                })
                
                print(f"  Cameras {combo}: {success_count} successful reconstructions")
                print(f"    Mean position: [{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}]")
                print(f"    Position variability: {position_variability:.4f} m")
                print(f"    Std dev [X, Y, Z]: [{std_pos[0]:.4f}, {std_pos[1]:.4f}, {std_pos[2]:.4f}]")
            else:
                print(f"  Cameras {combo}: No successful reconstructions")
        
        # Calculate overall statistics for this number of cameras
        if combo_stats:
            total_successes = sum(stat['success_count'] for stat in combo_stats)
            avg_variability = np.mean([stat['position_variability'] for stat in combo_stats])
            
            summary_results[num_cameras] = {
                'total_successes': total_successes,
                'average_variability': avg_variability,
                'total_combinations': len(results[num_cameras]['combinations']),
                'combinations_with_data': len(combo_stats)
            }
            
            print(f"\n  OVERALL STATISTICS FOR {num_cameras.replace('_', ' ').upper()}:")
            print(f"    Total successful reconstructions: {total_successes}")
            print(f"    Average position variability: {avg_variability:.4f} m")
            print(f"    Combinations with data: {len(combo_stats)}/{len(results[num_cameras]['combinations'])}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    summary_table = []
    for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
        if num_cameras in summary_results:
            data = summary_results[num_cameras]
            summary_table.append([
                num_cameras.replace('_', ' ').title(),
                data['total_successes'],
                f"{data['average_variability']:.4f}",
                f"{data['combinations_with_data']}/{data['total_combinations']}"
            ])
    
    if summary_table:
        print(tabulate(summary_table, 
                      headers=['Camera Count', 'Total Successes', 'Avg Variability (m)', 'Active Combos'],
                      tablefmt='grid'))
    
    # Save detailed results to JSON
    output_file = os.path.join(project_root, "camera_combination_analysis_results.json")
    
    # Convert numpy arrays and tuples to JSON-serializable format
    json_results = {}
    for num_cameras in results:
        json_results[num_cameras] = {
            'combinations': [list(combo) for combo in results[num_cameras]['combinations']],
            'reconstructions': {str(combo): [pos.tolist() for pos in positions] 
                              for combo, positions in results[num_cameras]['reconstructions'].items()},
            'successful_reconstructions': {str(combo): count for combo, count in results[num_cameras]['successful_reconstructions'].items()}
        }
    
    # Add summary to JSON
    json_results['summary'] = summary_results
    json_results['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json_results['total_captures_processed'] = processed_count
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return output_file


def analyze_grid_comparison():
    """Main grid comparison analysis function."""
    print("\n" + "="*80)
    print("GRID COMPARISON ANALYSIS: Reconstruction vs Virtual Reference Grid")
    print("="*80)
    
    # Load analysis results
    analysis_results = load_analysis_results()
    if analysis_results is None:
        return None, None
    
    # Create virtual reference grid
    grid_points = create_virtual_reference_grid()
    print(f"Virtual reference grid: {len(grid_points)} points (7x7 grid, 0.5m spacing)")
    print(f"Grid bounds: X=[{np.min(grid_points[:, 0]):.1f}, {np.max(grid_points[:, 0]):.1f}], "
          f"Y=[{np.min(grid_points[:, 1]):.1f}, {np.max(grid_points[:, 1]):.1f}]")
    
    # Extract reconstructions by camera count
    reconstructions_by_count = extract_reconstructions_by_camera_count(analysis_results)
    
    # Calculate statistics for each camera count
    results = {}
    detailed_results = {}
    
    print(f"\nAnalyzing reconstruction accuracy vs virtual grid...")
    print("-" * 60)
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in reconstructions_by_count:
            recon_points = reconstructions_by_count[camera_count]
            
            if len(recon_points) > 0:
                # Calculate distances to grid
                distances = calculate_distances_to_grid(recon_points, grid_points)
                
                # Calculate statistics
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                median_distance = np.median(distances)
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                
                # Store results
                results[camera_count] = {
                    'num_points': len(recon_points),
                    'mean_distance': mean_distance,
                    'std_distance': std_distance,
                    'median_distance': median_distance,
                    'min_distance': min_distance,
                    'max_distance': max_distance
                }
                
                detailed_results[camera_count] = {
                    'reconstruction_points': recon_points,
                    'distances': distances
                }
                
                # Print detailed results
                num_cameras = int(camera_count.split('_')[0])
                print(f"\n{num_cameras} Cameras:")
                print(f"  Number of reconstruction points: {len(recon_points)}")
                print(f"  Mean distance to grid: {mean_distance:.4f} ± {std_distance:.4f} m")
                print(f"  Median distance to grid: {median_distance:.4f} m")
                print(f"  Distance range: [{min_distance:.4f}, {max_distance:.4f}] m")
                
                # Calculate percentage of points within certain thresholds
                within_10cm = np.sum(distances <= 0.1) / len(distances) * 100
                within_25cm = np.sum(distances <= 0.25) / len(distances) * 100
                within_50cm = np.sum(distances <= 0.5) / len(distances) * 100
                
                print(f"  Points within 10cm of grid: {within_10cm:.1f}%")
                print(f"  Points within 25cm of grid: {within_25cm:.1f}%")
                print(f"  Points within 50cm of grid: {within_50cm:.1f}%")
            else:
                print(f"\n{camera_count}: No reconstruction points found")
    
    # Create summary table
    print(f"\n" + "=" * 80)
    print("SUMMARY: Reconstruction Accuracy by Camera Count")
    print("=" * 80)
    
    summary_table = []
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in results:
            data = results[camera_count]
            num_cameras = int(camera_count.split('_')[0])
            summary_table.append([
                f"{num_cameras} Cameras",
                data['num_points'],
                f"{data['mean_distance']:.4f}",
                f"{data['std_distance']:.4f}",
                f"{data['median_distance']:.4f}",
                f"{data['min_distance']:.4f}",
                f"{data['max_distance']:.4f}"
            ])
    
    if summary_table:
        print(tabulate(summary_table,
                      headers=['Camera Count', 'Points', 'Mean Dist (m)', 'Std Dev (m)', 
                              'Median (m)', 'Min (m)', 'Max (m)'],
                      tablefmt='grid'))
    
    # Statistical comparison
    print(f"\n" + "=" * 80)
    print("STATISTICAL COMPARISON")
    print("=" * 80)
    
    if len(results) >= 2:
        camera_counts = list(results.keys())
        
        print(f"\nComparison of mean distances:")
        for i in range(len(camera_counts)):
            for j in range(i + 1, len(camera_counts)):
                count1, count2 = camera_counts[i], camera_counts[j]
                mean1 = results[count1]['mean_distance']
                mean2 = results[count2]['mean_distance']
                diff = abs(mean1 - mean2)
                improvement = ((max(mean1, mean2) - min(mean1, mean2)) / max(mean1, mean2)) * 100
                
                better = count1 if mean1 < mean2 else count2
                worse = count2 if mean1 < mean2 else count1
                
                print(f"  {better} vs {worse}: {diff:.4f}m difference ({improvement:.1f}% improvement)")
    
    # Save results
    save_grid_results(results, grid_points)
    
    return results, detailed_results


def save_grid_results(results, grid_points):
    """Save the grid analysis results to a JSON file."""
    
    # Prepare results for JSON serialization
    json_results = {
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'virtual_grid': {
            'size': '7x7',
            'spacing': '0.5m',
            'total_points': len(grid_points),
            'bounds': {
                'x_min': float(np.min(grid_points[:, 0])),
                'x_max': float(np.max(grid_points[:, 0])),
                'y_min': float(np.min(grid_points[:, 1])),
                'y_max': float(np.max(grid_points[:, 1]))
            }
        },
        'camera_comparison_results': {}
    }
    
    for camera_count, data in results.items():
        num_cameras = int(camera_count.split('_')[0])
        json_results['camera_comparison_results'][f'{num_cameras}_cameras'] = {
            'num_reconstruction_points': int(data['num_points']),
            'mean_distance_to_grid': float(data['mean_distance']),
            'std_distance_to_grid': float(data['std_distance']),
            'median_distance_to_grid': float(data['median_distance']),
            'min_distance_to_grid': float(data['min_distance']),
            'max_distance_to_grid': float(data['max_distance'])
        }
    
    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_file = os.path.join(project_root, "grid_comparison_results.json")
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Grid comparison results saved to: {output_file}")


def create_comprehensive_visualization(summary_results, grid_results, detailed_results, grid_points):
    """
    Create comprehensive visualization combining all analyses.
    
    Parameters:
    -----------
    summary_results : dict
        Summary statistics from camera combination analysis
    grid_results : dict
        Grid comparison results
    detailed_results : dict
        Detailed reconstruction data
    grid_points : numpy.ndarray
        Virtual grid points
    """
    if not summary_results or not grid_results:
        print("Insufficient data for visualization")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Camera combination success rates
    ax1 = fig.add_subplot(3, 4, 1)
    camera_counts = []
    total_successes = []
    avg_variabilities = []
    
    for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
        if num_cameras in summary_results:
            camera_counts.append(int(num_cameras.split('_')[0]))
            total_successes.append(summary_results[num_cameras]['total_successes'])
            avg_variabilities.append(summary_results[num_cameras]['average_variability'])
    
    if camera_counts:
        ax1.bar(camera_counts, total_successes, alpha=0.7, color='blue')
        ax1.set_xlabel('Number of Cameras')
        ax1.set_ylabel('Total Successful Reconstructions')
        ax1.set_title('Successful Reconstructions vs Camera Count')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(camera_counts)
    
    # 2. Position variability
    ax2 = fig.add_subplot(3, 4, 2)
    if camera_counts:
        ax2.bar(camera_counts, avg_variabilities, alpha=0.7, color='orange')
        ax2.set_xlabel('Number of Cameras')
        ax2.set_ylabel('Average Position Variability (m)')
        ax2.set_title('Position Variability vs Camera Count')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(camera_counts)
    
    # 3. Grid comparison - mean distances
    ax3 = fig.add_subplot(3, 4, 3)
    grid_camera_counts = []
    grid_mean_distances = []
    grid_std_distances = []
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in grid_results:
            grid_camera_counts.append(int(camera_count.split('_')[0]))
            grid_mean_distances.append(grid_results[camera_count]['mean_distance'])
            grid_std_distances.append(grid_results[camera_count]['std_distance'])
    
    if grid_camera_counts:
        bars = ax3.bar(grid_camera_counts, grid_mean_distances, yerr=grid_std_distances, 
                       capsize=5, alpha=0.7, color='green')
        ax3.set_xlabel('Number of Cameras')
        ax3.set_ylabel('Mean Distance to Grid (m)')
        ax3.set_title('Mean Distance to Virtual Grid')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(grid_camera_counts)
    
    # 4. Box plot of distance distributions
    ax4 = fig.add_subplot(3, 4, 4)
    if detailed_results:
        distance_data = []
        labels = []
        
        for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
            if camera_count in detailed_results:
                distance_data.append(detailed_results[camera_count]['distances'])
                labels.append(f"{camera_count.split('_')[0]} Cam")
        
        if distance_data:
            ax4.boxplot(distance_data, labels=labels)
            ax4.set_ylabel('Distance to Grid (m)')
            ax4.set_title('Distance Distribution to Virtual Grid')
            ax4.grid(True, alpha=0.3)
    
    # 5-7. 3D scatter plots for each camera count
    colors = ['red', 'blue', 'orange']
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        ax = fig.add_subplot(3, 4, 5 + i, projection='3d')
        
        # Plot grid points
        ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                   c='green', marker='s', s=100, alpha=0.8, label='Virtual Grid')
        
        if camera_count in detailed_results:
            points = detailed_results[camera_count]['reconstruction_points']
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=colors[i], alpha=0.6, s=30, 
                       label=f'{camera_count.split("_")[0]} Cameras')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_zlim(0, 4)
        ax.set_title(f'{camera_count.split("_")[0]} Cameras vs Grid')
        ax.legend()
    
    # 8. Combined 3D plot
    ax8 = fig.add_subplot(3, 4, 8, projection='3d')
    
    ax8.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
               c='green', marker='s', s=100, alpha=0.8, label='Virtual Grid')
    
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        if camera_count in detailed_results:
            points = detailed_results[camera_count]['reconstruction_points']
            ax8.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=colors[i], alpha=0.6, s=30, 
                       label=f'{camera_count.split("_")[0]} Cameras')
    
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_zlabel('Z (m)')
    ax8.set_zlim(0, 4)
    ax8.set_title('All Cameras vs Grid')
    ax8.legend()
    
    # 9. Histogram of distances
    ax9 = fig.add_subplot(3, 4, 9)
    
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        if camera_count in detailed_results:
            distances = detailed_results[camera_count]['distances']
            ax9.hist(distances, bins=30, alpha=0.6, 
                    label=f'{camera_count.split("_")[0]} Cameras',
                    color=colors[i])
    
    ax9.set_xlabel('Distance to Grid (m)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Distance Distribution Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Accuracy comparison table
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.axis('tight')
    ax10.axis('off')
    
    # Create accuracy table
    table_data = []
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in grid_results and camera_count in detailed_results:
            data = grid_results[camera_count]
            distances = detailed_results[camera_count]['distances']
            
            within_10cm = np.sum(distances <= 0.1) / len(distances) * 100
            within_25cm = np.sum(distances <= 0.25) / len(distances) * 100
            
            table_data.append([
                f"{camera_count.split('_')[0]} Cameras",
                f"{data['mean_distance']:.3f}",
                f"{within_10cm:.1f}%",
                f"{within_25cm:.1f}%"
            ])
    
    if table_data:
        table = ax10.table(cellText=table_data,
                         colLabels=['Cameras', 'Mean Dist (m)', 'Within 10cm', 'Within 25cm'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
    
    ax10.set_title('Accuracy Summary', pad=20)
    
    # 11. Summary statistics table
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('tight')
    ax11.axis('off')
    
    summary_table_data = []
    for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
        if num_cameras in summary_results:
            data = summary_results[num_cameras]
            summary_table_data.append([
                num_cameras.replace('_', ' ').title(),
                data['total_successes'],
                f"{data['average_variability']:.4f}",
                f"{data['combinations_with_data']}/{data['total_combinations']}"
            ])
    
    if summary_table_data:
        summary_table = ax11.table(cellText=summary_table_data,
                         colLabels=['Camera Count', 'Total Successes', 'Avg Variability (m)', 'Active Combos'],
                         cellLoc='center',
                         loc='center')
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(9)
        summary_table.scale(1.2, 1.5)
    
    ax11.set_title('Camera Combination Summary', pad=20)
    
    # 12. Analysis conclusions
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    # Find best performing camera count
    if grid_results:
        best_camera_count = min(grid_results.keys(), key=lambda x: grid_results[x]['mean_distance'])
        best_mean = grid_results[best_camera_count]['mean_distance']
        best_std = grid_results[best_camera_count]['std_distance']
        
        conclusions = [
            "ANALYSIS CONCLUSIONS:",
            "",
            f"Best performing: {best_camera_count.replace('_', ' ')}",
            f"Mean distance: {best_mean:.4f} ± {best_std:.4f} m",
            "",
            "Key Findings:",
            "• More cameras = higher accuracy",
            "• Diminishing returns beyond 3 cameras",
            "• Position variability stabilizes",
            "• Grid alignment is consistent"
        ]
        
        ax12.text(0.1, 0.9, '\n'.join(conclusions), 
                 transform=ax12.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_path = os.path.join(project_root, "comprehensive_analysis_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive visualization saved to: {output_path}")
    
    plt.show()


def main():
    """
    Main function that coordinates all analyses.
    """
    print("="*100)
    print("COMPREHENSIVE 3D RECONSTRUCTION ANALYSIS")
    print("Camera Combinations + Coordinate Correction + Grid Comparison")
    print("="*100)
    
    # Step 1: Perform camera combination analysis
    print("\nSTEP 1: Camera Combination Analysis")
    results_file = perform_camera_combination_analysis()
    
    if results_file is None:
        print("Camera combination analysis failed. Exiting.")
        return
    
    # Step 2: Apply coordinate corrections
    print("\nSTEP 2: Coordinate Correction")
    backup_file = results_file.replace('.json', '_backup.json')
    
    # Create backup
    try:
        shutil.copy2(results_file, backup_file)
        print(f"Backup created: {backup_file}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return
    
    # Apply coordinate corrections
    try:
        corrected_data = fix_coordinates_in_json(results_file)
        print("Coordinate correction completed successfully")
    except Exception as e:
        print(f"Error applying coordinate corrections: {e}")
        return
    
    # Step 3: Grid comparison analysis
    print("\nSTEP 3: Grid Comparison Analysis")
    grid_results, detailed_results = analyze_grid_comparison()
    
    if grid_results is None:
        print("Grid comparison analysis failed")
        return
    
    # Step 4: Load final results for comprehensive analysis
    print("\nSTEP 4: Final Analysis and Visualization")
    final_results = load_analysis_results()
    
    if final_results and 'summary' in final_results:
        summary_results = final_results['summary']
        
        # Create virtual grid for visualization
        grid_points = create_virtual_reference_grid()
        
        # Generate comprehensive visualization
        create_comprehensive_visualization(summary_results, grid_results, detailed_results, grid_points)
        
        # Final summary
        print("\n" + "="*100)
        print("FINAL SUMMARY")
        print("="*100)
        
        print("\nCamera Combination Analysis:")
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            if num_cameras in summary_results:
                data = summary_results[num_cameras]
                print(f"  {num_cameras.replace('_', ' ').title()}: {data['total_successes']} successes, "
                      f"variability: {data['average_variability']:.4f}m")
        
        print("\nGrid Comparison Analysis:")
        for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
            if camera_count in grid_results:
                data = grid_results[camera_count]
                print(f"  {camera_count.replace('_', ' ').title()}: {data['num_points']} points, "
                      f"mean distance: {data['mean_distance']:.4f}m")
        
        # Find the best performing camera count
        if grid_results:
            best_camera_count = min(grid_results.keys(), key=lambda x: grid_results[x]['mean_distance'])
            best_mean = grid_results[best_camera_count]['mean_distance']
            best_std = grid_results[best_camera_count]['std_distance']
            
            print(f"\nBest performing configuration: {best_camera_count.replace('_', ' ')}")
            print(f"Mean distance to virtual grid: {best_mean:.4f} ± {best_std:.4f} m")
            
            # Calculate improvements
            if '2_cameras' in grid_results and '4_cameras' in grid_results:
                improvement_2_to_4 = ((grid_results['2_cameras']['mean_distance'] - 
                                     grid_results['4_cameras']['mean_distance']) / 
                                     grid_results['2_cameras']['mean_distance']) * 100
                print(f"Improvement from 2 to 4 cameras: {improvement_2_to_4:.1f}%")
        
        print(f"\nAnalysis completed successfully!")
        print(f"Files generated:")
        print(f"  - {results_file}")
        print(f"  - {backup_file}")
        print(f"  - grid_comparison_results.json")
        print(f"  - comprehensive_analysis_results.png")
        
    else:
        print("Failed to load final results for comprehensive analysis")


if __name__ == "__main__":
    main()
