# -*- coding: utf-8 -*-
"""
This script performs a comparative analysis of robot trajectory data from two sources:
1.  External cameras providing a "ground truth" world coordinate path.
2.  The robot's internal odometry.

The analysis is divided into two main parts:
1.  Hybrid Analysis: This part optimally aligns the odometry trajectory to the
    camera trajectory by scaling, translating, and rotating it. It then calculates
    the direct point-to-point error between the two aligned paths.
2.  Fixed Radius Analysis: This part assumes the robot was intended to follow a
    perfect circle of a known, fixed radius. It calculates the optimal center
    for such a circle for the camera data and then measures the Root Mean Square
    Error (RMSE) of both the camera and odometry trajectories with respect to this
    ideal circle.

The script loads data from a JSON file (for camera positions) and a CSV file
(for odometry), performs the calculations for both analyses, prints the results
to the console, and generates a comparative visualization plot saved as a PNG file.

Dependencies:
- pandas
- numpy
- matplotlib
- scipy
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import least_squares, minimize_scalar

# --- FUNCTION DEFINITIONS ---

def fit_circle(points):
    """
    Fits a circle to a given set of 2D points using the least-squares method.

    This function determines the optimal center (xc, yc) and radius (R) for a
    circle that best fits the input points. It minimizes the variance of the
    distances from the points to the circle's center.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 2), where N is the
                             number of points and each row is an [x, y] coordinate.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A NumPy array with the [xc, yc] coordinates of the fitted circle's center.
            - float: The mean radius (R) of the fitted circle.
    """
    # Define a helper function to calculate the radial distance of each point from a candidate center (xc, yc)
    def calc_R(xc, yc):
        """Calculates the distance of each point to the center (xc, yc)."""
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    # Define the objective function for the least_squares optimizer.
    # It calculates the difference between each point's radial distance and the mean of all radial distances.
    # The optimizer will try to make these differences zero.
    def f_2(c):
        """Calculates the residuals (the difference between each radius and the mean radius)."""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    # Provide an initial guess for the center, which is the mean of all points.
    center_estimate = np.mean(points, axis=0)
    
    # Run the least-squares optimization to find the best-fit center.
    center_fit = least_squares(f_2, center_estimate)
    
    # Extract the optimized center coordinates.
    xc, yc = center_fit.x
    
    # Calculate the final radii and the mean radius from the optimized center.
    Ri = calc_R(xc, yc)
    R = Ri.mean()
    
    return np.array([xc, yc]), R

def fit_circle_fixed_radius(points, R_fixed):
    """
    Finds the optimal center for a circle with a fixed radius that best fits a
    set of 2D points.

    Args:
        points (np.ndarray): An array of shape (N, 2) representing the [x, y] points.
        R_fixed (float): The predetermined, fixed radius of the circle.

    Returns:
        np.ndarray: The [xc, yc] coordinates of the optimal center.
    """
    # Define a helper function to calculate the radial distance of each point from a candidate center (xc, yc).
    def calc_R(xc, yc):
        """Calculates the distance of each point to the center (xc, yc)."""
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    # The objective function minimizes the difference between each point's
    # distance to the center and the fixed radius.
    def f_fixed_radius(c):
        """Calculates residuals against the fixed radius."""
        Ri = calc_R(*c)
        return Ri - R_fixed

    # Use the mean of the points as an initial guess for the center.
    center_estimate = np.mean(points, axis=0)
    
    # Run least-squares optimization to find the best center.
    center_fit = least_squares(f_fixed_radius, center_estimate)
    
    return center_fit.x

def get_path_direction(points):
    """
    Calculates the signed area of the polygon formed by the points to determine
    the path's direction (clockwise or counter-clockwise).

    Uses the Shoelace formula. A positive result indicates a counter-clockwise
    path, while a negative result indicates a clockwise path.

    Args:
        points (np.ndarray): An array of shape (N, 2) with the ordered [x, y] points of the path.

    Returns:
        float: The signed area. Positive for CCW, negative for CW.
    """
    x = points[:, 0]
    y = points[:, 1]
    # Apply the Shoelace formula: 0.5 * sum(x_i * y_{i+1} - x_{i+1} * y_i)
    return np.sum((x[:-1] * y[1:]) - (x[1:] * y[:-1]))

def rotation_error_func(angle, path_to_rotate, ref_path, center_world):
    """
    Calculates the alignment error between a rotated path and a reference path.

    This function is used as the objective function for the rotation optimization.
    It rotates `path_to_rotate` by a given `angle` and then computes the sum of
    minimum squared distances to a subset of points on the `ref_path`.

    Args:
        angle (float): The rotation angle in radians to test.
        path_to_rotate (np.ndarray): The path to be rotated (e.g., scaled odometry).
        ref_path (np.ndarray): The reference path (e.g., centered world data).
        center_world (np.ndarray): The original center of the world data, not used here but
                                   kept for compatibility with older function signatures.

    Returns:
        float: The total alignment error for the given angle. A lower value means better alignment.
    """
    # Create the 2D rotation matrix for the given angle.
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    # Apply the rotation to the path.
    rotated_path = path_to_rotate @ rot_matrix.T
    
    total_error = 0
    # Iterate through a subset of the reference path to calculate error efficiently.
    # Using a step of 10 reduces computation time.
    for i in range(0, len(ref_path), 10):
        # For each reference point, find the closest point in the rotated path.
        distances = np.sum((rotated_path - ref_path[i])**2, axis=1)
        total_error += np.min(distances)
        
    return total_error

# --- DATA LOADING ---

# Load world data (camera positions) from a JSON file.
# Try multiple common file paths to increase flexibility.
try:
    with open('robis_circle_3D_plot.json', 'r') as f:
        world_data_raw = json.load(f)
    # Extract the 2D (x, y) coordinates from the raw data structure.
    mtx_world = np.array([p['points'][0]['position'][:2] for p in world_data_raw if p['points']])
except FileNotFoundError:
    try:
        # Fallback to an alternative path.
        with open('experiments/paper/robis/robis_circle_3D_plot.json', 'r') as f:
            world_data_raw = json.load(f)
        mtx_world = np.array([p['points'][0]['position'][:2] for p in world_data_raw if p['points']])
    except FileNotFoundError:
        print("Error: JSON file for world data not found in expected locations.")
        exit() # Exit if data is essential and not found.

# Load odometry data from a CSV file.
try:
    df_odom = pd.read_csv('odometry_data.csv')
    # Extract 'x_odom' and 'y_odom' columns into a NumPy array.
    mtx_odom = df_odom[['x_odom', 'y_odom']].values
except (FileNotFoundError, KeyError) as e:
    print(f"Error loading 'odometry_data.csv': {e}")
    exit() # Exit on error.

print("=== FUSED ANALYSIS: HYBRID ALIGNMENT + FIXED RADIUS COMPARISON ===\n")

# --- PART 1: HYBRID ANALYSIS (OPTIMAL ALIGNMENT) ---
print("--- Part 1: Hybrid Analysis (Optimized Alignment) ---")

# Fit circles to both the world (camera) and odometry trajectories to find their centers and radii.
center_world, R_world = fit_circle(mtx_world)
center_odom, R_odom = fit_circle(mtx_odom)

# Calculate the scale factor to resize the odometry circle to match the world circle.
scale_factor = R_world / R_odom

print(f"World circle - Center: ({center_world[0]:.3f}, {center_world[1]:.3f}), Radius: {R_world:.3f}m")
print(f"Odom circle - Center: ({center_odom[0]:.3f}, {center_odom[1]:.3f}), Radius: {R_odom:.3f}m")
print(f"Scale factor: {scale_factor:.3f}")

# Center the odometry data by subtracting its calculated center.
mtx_odom_scaled = mtx_odom.copy() - center_odom
# Scale the centered odometry data using the calculated scale factor.
mtx_odom_scaled *= scale_factor

# Check if the paths have the same direction (both clockwise or both counter-clockwise).
dir_world = get_path_direction(mtx_world)
dir_odom = get_path_direction(mtx_odom_scaled)
if np.sign(dir_world) != np.sign(dir_odom):
    print("WARNING: Opposite path directions detected. Inverting odometry path.")
    # If directions differ, flip the odometry point order to match.
    mtx_odom_scaled = np.flipud(mtx_odom_scaled)

# Optimize the rotation to align the scaled odometry path with the world path.
print("Optimizing rotation for best alignment...")
# Center the world data for the rotation optimization process.
world_centered = mtx_world - center_world
# Use minimize_scalar to find the angle that minimizes the rotation_error_func.
result = minimize_scalar(rotation_error_func, args=(mtx_odom_scaled, world_centered, center_world), bounds=(0, 2*np.pi), method='bounded')
optimal_angle = result.x
print(f"Optimal rotation found: {np.degrees(optimal_angle):.1f} degrees.")

# Apply the final transformation: rotate the scaled odometry data and then translate it to the world center.
final_rot_matrix = np.array([[np.cos(optimal_angle), -np.sin(optimal_angle)], [np.sin(optimal_angle), np.cos(optimal_angle)]])
mtx_odom_aligned = (mtx_odom_scaled @ final_rot_matrix.T) + center_world

# --- Calculate Hybrid Alignment Errors ---
# For each point in the world path, find its closest corresponding point in the aligned odometry path.
mtx_odom_corresponding = np.zeros_like(mtx_world)
for i, world_point in enumerate(mtx_world):
    # Calculate squared Euclidean distances to all aligned odometry points.
    distances_to_odom_points = np.sum((mtx_odom_aligned - world_point)**2, axis=1)
    # Find the index of the closest odometry point.
    closest_odom_idx = np.argmin(distances_to_odom_points)
    mtx_odom_corresponding[i] = mtx_odom_aligned[closest_odom_idx]

# Calculate the final error as the Euclidean distance between each world point and its corresponding odometry point.
hybrid_errors = np.sqrt(np.sum((mtx_world - mtx_odom_corresponding)**2, axis=1))
hybrid_mean_error = np.mean(hybrid_errors)
hybrid_max_error = np.max(hybrid_errors)

# Print the results of the hybrid analysis.
print(f"Hybrid Analysis Results:")
print(f"  Mean Error: {hybrid_mean_error:.3f} m ({hybrid_mean_error*100:.1f} cm)")
print(f"  Max Error: {hybrid_max_error:.3f} m ({hybrid_max_error*100:.1f} cm)")


# --- PART 2: FIXED RADIUS ANALYSIS ---
print("\n--- Part 2: Fixed Radius Circle Analysis ---")

# Define the ideal or reference radius the robot was supposed to follow.
R_IDEAL = 1.5
print(f"Reference circle radius: {R_IDEAL:.2f} m")

# Find the optimal center for the world data, assuming it should form a circle with the ideal radius.
center_ideal = fit_circle_fixed_radius(mtx_world, R_IDEAL)
print(f"Optimal center for R={R_IDEAL}m: ({center_ideal[0]:.3f}, {center_ideal[1]:.3f})")

# For this analysis, simply translate the fully aligned odometry path from Part 1
# so its center matches the new `center_ideal`.
mtx_odom_simple_aligned = mtx_odom_aligned - center_world + center_ideal

# --- Calculate RMSE for Fixed Radius Analysis ---
def calc_R(points, center):
    """Helper function to calculate radial distances of points from a center."""
    return np.sqrt(np.sum((points - center)**2, axis=1))

# Calculate the residuals (errors) for the world path against the ideal circle.
residuals_world = calc_R(mtx_world, center_ideal) - R_IDEAL
# Calculate the Root Mean Square Error.
rmse_world = np.sqrt(np.mean(residuals_world**2))

# Calculate the residuals and RMSE for the odometry path against the ideal circle.
residuals_odom = calc_R(mtx_odom_simple_aligned, center_ideal) - R_IDEAL
rmse_odom = np.sqrt(np.mean(residuals_odom**2))

# Print the results of the fixed radius analysis.
print(f"Fixed Radius Analysis Results:")
print(f"  World RMSE vs R={R_IDEAL}m: {rmse_world:.3f} m ({rmse_world*100:.1f} cm)")
print(f"  Odom RMSE vs R={R_IDEAL}m: {rmse_odom:.3f} m ({rmse_odom*100:.1f} cm)")


# --- VISUALIZATION ---
print("\n--- Generating Combined Visualization ---")

# Generate points for the ideal reference circle for plotting.
theta = np.linspace(0, 2 * np.pi, 200)
x_circ_ideal = center_ideal[0] + R_IDEAL * np.cos(theta)
y_circ_ideal = center_ideal[1] + R_IDEAL * np.sin(theta)

# Create a figure with two subplots side-by-side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# --- Plot 1: Hybrid Analysis Visualization ---
ax1.set_title(f'Hybrid Analysis\nMean Error: {hybrid_mean_error*100:.1f} cm', fontsize=14)
ax1.plot(mtx_world[:, 0], mtx_world[:, 1], 'o', color='dodgerblue', markersize=4, alpha=0.7, label='Cameras')
ax1.plot(mtx_odom_aligned[:, 0], mtx_odom_aligned[:, 1], 'o', color='limegreen', markersize=4, alpha=0.7, label='Odometry (Optimally Aligned)')

# Draw red lines connecting corresponding camera and odometry points to visualize the error.
# Plot only a subset of lines for clarity.
step = len(mtx_world) // 25
for i in range(0, len(mtx_world), step):
    ax1.plot([mtx_world[i, 0], mtx_odom_corresponding[i, 0]],
            [mtx_world[i, 1], mtx_odom_corresponding[i, 1]],
            color='red', linestyle='-', linewidth=0.8)

# Add a dummy plot for the error line legend entry.
ax1.plot([], [], color='red', linestyle='-', linewidth=0.8, label='Point-to-Point Error')
ax1.set_xlabel('X Position (meters)')
ax1.set_ylabel('Y Position (meters)')
ax1.set_aspect('equal', adjustable='box') # Ensure X and Y scales are equal.
ax1.legend()
ax1.grid(True)

# --- Plot 2: Fixed Radius Analysis Visualization ---
ax2.set_title(f'Fixed Radius Analysis\nReference Circle: R={R_IDEAL:.1f}m', fontsize=14)
ax2.plot(mtx_world[:, 0], mtx_world[:, 1], 'o', markersize=4, 
         label=f'Cameras (RMSE: {rmse_world*100:.1f} cm)', color='dodgerblue', alpha=0.8)
ax2.plot(mtx_odom_simple_aligned[:, 0], mtx_odom_simple_aligned[:, 1], 'o', markersize=4, 
         label=f'Odometry (RMSE: {rmse_odom*100:.1f} cm)', color='limegreen', alpha=0.8)
ax2.plot(x_circ_ideal, y_circ_ideal, '-', linewidth=2.5, 
         label=f'Reference Circle (R={R_IDEAL:.1f}m)', color='darkorange')

ax2.set_xlabel('X Position (meters)')
ax2.set_ylabel('Y Position (meters)')
ax2.set_aspect('equal', adjustable='box')
ax2.legend()
ax2.grid(True)

# Adjust layout, save the figure, and display it.
plt.tight_layout()
plt.savefig("fused_analysis_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# --- FINAL SUMMARY ---
print(f"\n=== SUMMARY ===")
print(f"Hybrid Analysis (Optimized Alignment):")
print(f"  - This method finds the best possible fit between the two trajectories.")
print(f"  - Mean error between corresponding points: {hybrid_mean_error*100:.1f} cm")
print(f"  - Maximum error: {hybrid_max_error*100:.1f} cm")
print(f"\nFixed Radius Analysis (vs {R_IDEAL}m circle):")
print(f"  - This method evaluates how well each trajectory conforms to an ideal circular path.")
print(f"  - Camera trajectory RMSE: {rmse_world*100:.1f} cm")
print(f"  - Odometry trajectory RMSE: {rmse_odom*100:.1f} cm")
print(f"\nVisualization saved as: fused_analysis_comparison.png")