"""
Trajectory Alignment and Error Analysis

This script aligns a robot's odometry trajectory with a ground-truth
world trajectory. It performs the following steps:
1.  Loads a ground-truth path (from a JSON file) and an odometry path (from a CSV).
2.  Fits a circle to each trajectory to determine their respective centers and radii.
3.  Calculates a scale factor from the ratio of the radii to correct the odometry data.
4.  Aligns the trajectories by translation (using circle centers) and scale.
5.  Checks and corrects for opposite path directions (e.g., clockwise vs. counter-clockwise).
6.  Optimizes the rotational alignment by minimizing the distance between the two paths.
7.  Calculates key error metrics (Mean Error, RMSE, Standard Deviation).
8.  Generates a publication-quality plot comparing the final aligned trajectories.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize_scalar

# --- Constants ---
WORLD_DATA_PATH = 'robis_circle_3D_plot.json'
ODOMETRY_DATA_PATH = 'odometry_data.csv'
PLOT_OUTPUT_PATH = "odom_x_triangulated.eps"

def fit_circle(points):
    """
    Fits a circle to a set of 2D points using the least squares method.

    This function determines the optimal center (xc, yc) and radius (R) for a
    circle that best represents the given cloud of points.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2) where N is the
                             number of points and each row is an [x, y] coordinate.

    Returns:
        tuple[np.ndarray, float]: A tuple containing:
            - np.ndarray: The center of the circle as a [xc, yc] array.
            - float: The mean radius of the circle.
    """
    # Define a function to calculate the distance of each point from a candidate center.
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    # Define the error function for the least squares optimization.
    # The error is the deviation of each point's radius from the mean radius.
    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Use the mean of the points as an initial guess for the center.
    center_estimate = np.mean(points, axis=0)
    
    # Perform the least squares optimization to find the best-fit center.
    center_fit = least_squares(f_2, center_estimate)
    
    # Extract the optimized center coordinates.
    xc, yc = center_fit.x
    
    # Calculate the final radii and the mean radius.
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    
    return np.array([xc, yc]), R

def get_path_direction(points):
    """
    Determines the direction (clockwise/counter-clockwise) of a 2D path.

    This function uses the Shoelace formula (also known as the surveyor's
    formula). A positive result typically indicates a counter-clockwise path,
    while a negative result indicates a clockwise path.

    Args:
        points (np.ndarray): An array of shape (N, 2) representing the path vertices.

    Returns:
        float: The signed area of the polygon. The sign indicates the direction.
    """
    x = points[:, 0]
    y = points[:, 1]
    # Shoelace formula: sum over edges of (x_i * y_{i+1}) - (x_{i+1} * y_i)
    return np.sum((x[:-1] * y[1:]) - (x[1:] * y[:-1]))

def rotation_error_func(angle, path_to_rotate, ref_path):
    """
    Calculates the alignment error between two paths for a given rotation angle.

    The error is computed by rotating `path_to_rotate` by the given `angle` and
    then summing the minimum squared distances from a subset of points on
    `ref_path` to the rotated path.

    Args:
        angle (float): The rotation angle in radians.
        path_to_rotate (np.ndarray): The centered path to be rotated.
        ref_path (np.ndarray): The centered reference path.

    Returns:
        float: The total alignment error for the given angle.
    """
    # Define the 2D rotation matrix.
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                           [np.sin(angle), np.cos(angle)]])
    
    # Apply the rotation to the path.
    rotated_path = path_to_rotate @ rot_matrix.T
    
    total_error = 0
    # Subsample the reference path for efficiency.
    # For each reference point, find the closest point on the rotated path.
    for i in range(0, len(ref_path), 10):
        # Calculate the squared Euclidean distance from the ref point to all points in the rotated path.
        distances = np.sum((rotated_path - ref_path[i])**2, axis=1)
        # Add the minimum distance to the total error.
        total_error += np.min(distances)
        
    return total_error

def main():
    """
    Main function to execute the trajectory alignment and analysis workflow.
    """
    # --- 1. Load Data ---
    try:
        with open(WORLD_DATA_PATH, 'r') as f:
            world_data_raw = json.load(f)
        # Extract the first [x, y] position from each valid entry.
        mtx_world = np.array([p['points'][0]['position'][:2] for p in world_data_raw if p['points']])
    except FileNotFoundError:
        print(f"Error: File '{WORLD_DATA_PATH}' not found.")
        return

    try:
        df_odom = pd.read_csv(ODOMETRY_DATA_PATH)
        mtx_odom = df_odom[['x_odom', 'y_odom']].values
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading '{ODOMETRY_DATA_PATH}': {e}")
        return

    # --- 2. Fit Circles and Determine Scale Factor ---
    center_world, R_world = fit_circle(mtx_world)
    center_odom, R_odom = fit_circle(mtx_odom)
    
    # The scale factor corrects for discrepancies between odometry units and world units.
    scale_factor = R_world / R_odom

    # --- 3. Scale, Center, and Align Path Direction ---
    # First, center the odometry data around its origin (0,0).
    mtx_odom_scaled = mtx_odom - center_odom
    # Then, apply the scale factor.
    mtx_odom_scaled *= scale_factor
    
    # Check if paths have opposite directions (e.g., one clockwise, one counter-clockwise).
    dir_world = get_path_direction(mtx_world)
    dir_odom = get_path_direction(mtx_odom_scaled)
    if np.sign(dir_world) != np.sign(dir_odom):
        print("WARNING: Opposite path directions detected. Inverting odometry path.")
        # Flipping the point order reverses the path direction.
        mtx_odom_scaled = np.flipud(mtx_odom_scaled)

    # --- 4. Optimize Rotational Alignment ---
    print("Optimizing rotation for best alignment...")
    # Center the world path for the rotation optimization function.
    world_centered = mtx_world - center_world
    
    # Find the angle that minimizes the error between the two centered paths.
    result = minimize_scalar(
        rotation_error_func, 
        args=(mtx_odom_scaled, world_centered), 
        bounds=(0, 2 * np.pi), 
        method='bounded'
    )
    optimal_angle = result.x
    print(f"Optimal rotation found: {np.degrees(optimal_angle):.1f} degrees.")

    # --- 5. Apply Final Transformation ---
    # Create the final rotation matrix.
    final_rot_matrix = np.array([[np.cos(optimal_angle), -np.sin(optimal_angle)], 
                                 [np.sin(optimal_angle), np.cos(optimal_angle)]])
    # Apply the rotation and then translate the path back to the world center.
    mtx_odom_aligned = (mtx_odom_scaled @ final_rot_matrix.T) + center_world

    # --- 6. Calculate Alignment Error ---
    # For each point in the ground-truth path, find its closest corresponding point
    # in the fully aligned odometry path.
    mtx_odom_corresponding = np.zeros_like(mtx_world)
    for i, world_point in enumerate(mtx_world):
        distances_to_odom_points = np.sum((mtx_odom_aligned - world_point)**2, axis=1)
        closest_odom_idx = np.argmin(distances_to_odom_points)
        mtx_odom_corresponding[i] = mtx_odom_aligned[closest_odom_idx]

    # Calculate error metrics based on the distances between corresponding points.
    errors = np.sqrt(np.sum((mtx_world - mtx_odom_corresponding)**2, axis=1))
    mean_error = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    std_error = np.std(errors)

    print("\n--- Trajectory Alignment Error Analysis ---")
    print(f"Mean Error:       {mean_error:.3f} m ({mean_error*100:.1f} cm)")
    print(f"RMSE:             {rmse:.3f} m ({rmse*100:.1f} cm)")
    print(f"Standard Deviation: {std_error:.3f} m ({std_error*100:.1f} cm)")

    # --- 7. Plotting ---
    # Set plot style for a professional, publication-ready look (IEEE style).
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif',
        'axes.linewidth': 0.8, 'grid.linewidth': 0.5,
        'lines.linewidth': 1.5, 'patch.linewidth': 0.5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.minor.size': 2, 'ytick.minor.size': 2,
    })

    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot ground-truth trajectory.
    ax.plot(mtx_world[:, 0], mtx_world[:, 1], '-', color='blue', alpha=0.9, 
            label='Triangulated Trajectory', linewidth=2)
    
    # Plot the final, fully aligned odometry trajectory.
    ax.plot(mtx_odom_aligned[:, 0], mtx_odom_aligned[:, 1], '--', color='red', alpha=0.9, 
            label='Robot Odometry', linewidth=2)

    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9, loc='center', frameon=True, fancybox=False, 
              shadow=False, framealpha=1.0, edgecolor='black')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure and display it.
    plt.savefig(PLOT_OUTPUT_PATH, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    main()