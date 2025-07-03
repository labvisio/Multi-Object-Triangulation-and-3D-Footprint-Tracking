#!/usr/bin/env python3
"""
Grid and Camera Pose Visualization Tool

This script generates a 2D grid of points on the Z=0 plane and visualizes it in
a 3D plot. It can also load camera calibration data to overlay the 3D position
and orientation (pose) of multiple cameras in the same plot.

The primary purpose is to create a visual representation of a ground-truth setup,
which is useful for validating camera calibration, testing 3D reconstruction
algorithms, or planning data capture sessions in a known environment.

The script is configurable via command-line arguments for grid properties (center,
size, spacing) and camera information (indices, path to calibration files).

Usage:
    python grid_generator.py [OPTIONS]

Example:
    # Generate a 5x5 grid with 0.4m spacing and plot cameras 0 and 2
    python grid_generator.py --size 5 --spacing 0.4 --cameras 0 2
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json

def generate_grid(center_x=0.3, center_y=0.3, grid_size=7, spacing=0.5):
    """
    Generates the X and Y coordinates for a 2D grid.

    The grid is centered at a specific point with a given size and spacing.

    Args:
        center_x (float): The x-coordinate of the grid's center.
        center_y (float): The y-coordinate of the grid's center.
        grid_size (int): The number of points along each dimension (e.g., 7 for a 7x7 grid).
        spacing (float): The distance between adjacent grid points in meters.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two 2D NumPy arrays (X, Y)
                                       representing the grid coordinates.
    """
    # Calculate the total span of the grid to determine the start and end points
    offset = (grid_size - 1) / 2 * spacing
    x_range = np.linspace(center_x - offset, center_x + offset, grid_size)
    y_range = np.linspace(center_y - offset, center_y + offset, grid_size)
    
    # Create a meshgrid from the x and y ranges
    X, Y = np.meshgrid(x_range, y_range)
    
    return X, Y

def camera_parameters(file_path):
    """
    Reads and parses camera calibration parameters from a JSON file.

    This function extracts intrinsic parameters (K matrix, distortion),
    extrinsic parameters (rotation R and translation T), and image resolution.
    It computes both camera-to-world (R, T) and world-to-camera (R_inv, T_inv)
    transformations.

    Args:
        file_path (str): Path to the camera calibration JSON file.

    Returns:
        tuple: A tuple containing (K, R, T, resolution, distortion, R_inv, T_inv).
    """
    with open(file_path) as f:
        camera_data = json.load(f)

    # Intrinsic camera matrix K
    K = np.array(camera_data['intrinsic']['doubles']).reshape(3, 3)
    res = [camera_data['resolution']['width'], camera_data['resolution']['height']]
    
    # Extrinsic parameters (camera-to-world transformation)
    tf = np.array(camera_data['extrinsic'][0]['tf']['doubles']).reshape(4, 4)
    R = tf[:3, :3]  # Rotation part
    T = tf[:3, 3].reshape(3, 1)  # Translation part
    
    # The inverse transformation (world-to-camera) is required for projecting world points
    # into the camera's view.
    R_inv = R.transpose()  # For a rotation matrix, the inverse is its transpose.
    T_inv = -R_inv @ T     # The inverse translation is -R_inv * T.
    
    # Distortion coefficients
    dis = np.array(camera_data['distortion']['doubles'])
    return K, R, T, res, dis, R_inv, T_inv

def plot_camera_axes(R, T, ax, scale=0.5, label=''):
    """
    Plots the 3D coordinate frame (axes) of a camera in a 3D plot.

    Args:
        R (np.ndarray): The rotation matrix (3x3) representing the camera's orientation
                        in the world (camera-to-world). Each column is a basis vector.
        T (np.ndarray): The translation vector (3x1) representing the camera's position
                        in the world.
        ax: The Matplotlib 3D Axes object to plot on.
        scale (float): The length of the rendered axes arrows.
        label (str): A text label to display at the camera's origin.
    """
    # The camera's origin is its translation vector T
    origin = T.flatten()
    
    # The columns of the camera-to-world rotation matrix represent the camera's
    # local x, y, and z axes in world coordinates.
    x_axis, y_axis, z_axis = R[:, 0], R[:, 1], R[:, 2]
    
    # Plot the axes using 3D quiver plots (arrows)
    # X-axis in Red
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', length=scale, normalize=True)
    # Y-axis in Green
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', length=scale, normalize=True)
    # Z-axis in Blue
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', length=scale, normalize=True)
    
    # Add a text label identifying the camera
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)

def plot_grid(X, Y, spacing=0.5, cameras_data=None, camera_indices_to_plot=None):
    """
    Creates a 3D plot of the grid points and optional camera poses.

    Args:
        X (np.ndarray): A 2D array of the grid's x-coordinates.
        Y (np.ndarray): A 2D array of the grid's y-coordinates.
        spacing (float): The grid spacing, used for the plot title.
        cameras_data (dict, optional): A dictionary with calibration data for all cameras.
        camera_indices_to_plot (list, optional): A list of camera indices to visualize.
    """
    grid_size = X.shape[0]
    center_x = X[grid_size//2, grid_size//2]
    center_y = Y[grid_size//2, grid_size//2]

    # Initialize a large figure for better visualization
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the grid points on the Z=0 plane
    ax.scatter(X, Y, np.zeros_like(X), color='blue', s=50, label='Grid Points (Z=0)')
    
    # Plot a marker for the world origin (0,0,0) for reference
    ax.scatter(0, 0, 0, color='k', s=100, marker='x', label='World Origin')
    
    # If camera data is provided, plot the axes for each specified camera
    if cameras_data and camera_indices_to_plot:
        for cam_idx in camera_indices_to_plot:
            if cam_idx in cameras_data:
                # Note: We use R and T (camera-to-world) to plot the camera's pose
                # in the world coordinate system.
                plot_camera_axes(
                    cameras_data[cam_idx]['R'], 
                    cameras_data[cam_idx]['T'], 
                    ax, scale=0.5, 
                    label=f'Camera {cam_idx}'
                )
            else:
                print(f"Warning: Calibration data for camera {cam_idx} not found.")

    # Set plot labels and title
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_zlabel('Z (meters)', fontsize=12)
    ax.set_title(f'{grid_size}x{grid_size} Grid (Z=0) & Camera Poses\nSpacing: {spacing}m, Center: ({center_x:.1f}, {center_y:.1f})', fontsize=14)
    
    # Automatically determine axis limits to fit all plotted elements
    all_x, all_y, all_z = list(X.flatten()), list(Y.flatten()), [0]
    if cameras_data and camera_indices_to_plot:
        for cam_idx in camera_indices_to_plot:
            if cam_idx in cameras_data:
                T_cam = cameras_data[cam_idx]['T'].flatten()
                all_x.append(T_cam[0])
                all_y.append(T_cam[1])
                all_z.append(T_cam[2])
    
    # Set fixed, sensible axis limits for consistent viewing
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 3)

    # Set axis ticks for better readability
    ax.set_xticks(np.arange(-3, 3.5, 0.5))
    ax.set_yticks(np.arange(-3, 3.5, 0.5))
    ax.set_zticks(np.arange(0, 3.5, 0.5))
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.legend()
    plt.tight_layout()

def main():
    """
    Main function to parse arguments, generate data, and create the plot.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate and plot a 3D grid of points with camera poses.')
    parser.add_argument('--center-x', type=float, default=0.3, help='X-coordinate of the grid center (default: 0.3)')
    parser.add_argument('--center-y', type=float, default=0.3, help='Y-coordinate of the grid center (default: 0.3)')
    parser.add_argument('--size', type=int, default=7, help='Grid size (e.g., 7 for a 7x7 grid, default: 7)')
    parser.add_argument('--spacing', type=float, default=0.5, help='Spacing between points in meters (default: 0.5)')
    parser.add_argument('--cameras', type=int, nargs='+', default=[0, 1, 2, 3], help='List of camera indices to plot (e.g., 0 1 2 3)')
    parser.add_argument('--calib-path', type=str, default='calibrations', help='Path to the camera calibration JSON files directory')
    args = parser.parse_args()
    
    X, Y = generate_grid(args.center_x, args.center_y, args.size, args.spacing)

    # Load calibration data for the specified cameras
    cameras_data = {}
    if args.cameras:
        print(f"Loading calibration data for cameras: {args.cameras}")
        for cam_idx in args.cameras:
            json_file = f"{args.calib_path}/{cam_idx}.json"
            try:
                # Unpack all calibration data
                K, R, T, res, dis, R_inv, T_inv = camera_parameters(json_file)
                cameras_data[cam_idx] = {'K': K, 'R': R, 'T': T, 'res': res, 'dis': dis, 'R_inv': R_inv, 'T_inv': T_inv}
                print(f"-> Successfully loaded calibration for camera {cam_idx}")
            except FileNotFoundError:
                print(f"-> Error: Calibration file {json_file} not found for camera {cam_idx}.")
            except Exception as e:
                print(f"-> Error loading calibration for camera {cam_idx}: {e}")
    
    # Generate the 3D plot
    plot_grid(X, Y, args.spacing, cameras_data, args.cameras)
    
    # Print summary information to the console
    print(f"\nGenerated a {args.size}x{args.size} grid centered at ({args.center_x}, {args.center_y}) with {args.spacing}m spacing.")
    print("\nGrid Coordinates (X, Y):")
    for i in range(args.size):
        for j in range(args.size):
            print(f"  Point ({i},{j}): ({X[i, j]:.2f}, {Y[i, j]:.2f})")
    
    # Save the figure with high resolution for publications, then display it
    print("\nSaving plot to 'grid_plot_3d.png'...")
    plt.savefig("grid_plot_3d.png", dpi=300, bbox_inches='tight')
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()