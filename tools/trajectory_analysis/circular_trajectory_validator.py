# -*- coding: utf-8 -*-
"""
Analysis of Robot Trajectories Against a Fixed-Radius Reference Circle.

This script performs a comparative analysis of two robot trajectory datasets:
1.  A "world" trajectory derived from external cameras (via 3D reconstruction).
2.  An internal "odometry" trajectory from the robot's own sensors.

The core of the analysis is to evaluate how well each trajectory conforms to an
idealized circular path with a fixed, known radius (R=1.5m).

The process is as follows:
1.  Loads the world and odometry trajectory data from their respective files.
2.  Determines the optimal center for a 1.5m radius circle that best fits the
    world (camera) data. This circle becomes the "ground truth" reference.
3.  Calculates the best-fit center for the odometry data (allowing its radius
    to be variable).
4.  Aligns the odometry trajectory by translating its center to match the
    ideal reference center found in the previous step.
5.  Calculates and prints error metrics (RMSE, MAE, Standard Deviation) for both
    the world trajectory and the aligned odometry trajectory against the
    1.5m reference circle.
6.  Generates and saves publication-style plot visualizing
    both trajectories and the ideal reference circle.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import json

# --- Helper and Error Functions for Circle Fitting ---

def calc_R(x_data, y_data, xc, yc):
    """
    Calculates the radial distance of each point (x, y) from a center (xc, yc).

    Args:
        x_data (np.ndarray): Array of x-coordinates.
        y_data (np.ndarray): Array of y-coordinates.
        xc (float): x-coordinate of the circle's center.
        yc (float): y-coordinate of the circle's center.

    Returns:
        np.ndarray: An array of radial distances for each point.
    """
    return np.sqrt((x_data - xc)**2 + (y_data - yc)**2)

def f_fixed_radius(c, x_data, y_data, R_fixed):
    """
    Error function for fitting a circle with a FIXED radius.

    This function is used by the optimizer to find the best center 'c' for a
    circle of a known radius 'R_fixed'. It returns the residuals, which are the
    differences between each point's distance to the center and the fixed radius.

    Args:
        c (list or tuple): The center coordinates [xc, yc] being optimized.
        x_data (np.ndarray): Array of x-coordinates of the data points.
        y_data (np.ndarray): Array of y-coordinates of the data points.
        R_fixed (float): The fixed radius of the circle.

    Returns:
        np.ndarray: The array of residuals.
    """
    Ri = calc_R(x_data, y_data, *c)
    return Ri - R_fixed

def f_variable_radius(c, x_data, y_data):
    """
    Error function for fitting a circle with a VARIABLE radius.

    This function is used to find the best-fit circle when the radius is unknown.
    It calculates the residuals as the difference between each point's radial
    distance and the mean of all radial distances. The optimizer minimizes these
    residuals, effectively minimizing the variance of the radii.

    Args:
        c (list or tuple): The center coordinates [xc, yc] being optimized.
        x_data (np.ndarray): Array of x-coordinates of the data points.
        y_data (np.ndarray): Array of y-coordinates of the data points.

    Returns:
        np.ndarray: The array of residuals from the mean radius.
    """
    Ri = calc_R(x_data, y_data, *c)
    return Ri - Ri.mean()

# --- Data Loading ---

# Load the "world" trajectory data from camera reconstructions
try:
    with open('experiments/paper/robis/robis_circle_3D_plot.json', 'r') as f:
        world_data_raw = json.load(f)
    # Extract x and y coordinates from the JSON structure
    x_world = np.array([p['points'][0]['position'][0] for p in world_data_raw if p['points']])
    y_world = np.array([p['points'][0]['position'][1] for p in world_data_raw if p['points']])
except FileNotFoundError:
    print("Error: File 'robis_circle_3D_plot.json' not found. Please ensure the path is correct.")
    exit()

# Load the robot's odometry data
try:
    df_odom = pd.read_csv('odometry_data.csv')
    x_odom = df_odom['x_odom'].values
    y_odom = df_odom['y_odom'].values
except FileNotFoundError:
    print("Error: File 'odometry_data.csv' not found.")
    exit()

# --- Analysis and Trajectory Alignment ---

# Define the ideal radius for the reference circle (ground truth).
R_IDEAL = 1.5

# Step 1: Find the optimal center for the world data, assuming it should follow the ideal radius.
# We use the mean of the data points as an initial guess for the center.
center_estimate_world = np.mean(x_world), np.mean(y_world)
center_fit_ideal = least_squares(f_fixed_radius, center_estimate_world, args=(x_world, y_world, R_IDEAL))
xc_ideal, yc_ideal = center_fit_ideal.x

# Step 2: Find the best-fit center for the raw odometry data (with a variable radius).
# This tells us where the odometry "thinks" its center of rotation is.
center_estimate_odom = np.mean(x_odom), np.mean(y_odom)
center_fit_odom = least_squares(f_variable_radius, center_estimate_odom, args=(x_odom, y_odom))
xc_odom, yc_odom = center_fit_odom.x

# Step 3: Align the odometry trajectory to the ideal reference circle's center.
# This is a simple translation of the entire odometry path.
x_odom_aligned = x_odom - xc_odom + xc_ideal
y_odom_aligned = y_odom - yc_odom + yc_ideal

# --- Error Calculation ---

# Calculate error metrics for the WORLD trajectory against the ideal circle.
residuos_world = calc_R(x_world, y_world, xc_ideal, yc_ideal) - R_IDEAL
rmse_world = np.sqrt(np.mean(residuos_world**2)) # Root Mean Square Error
mae_world = np.mean(np.abs(residuos_world))     # Mean Absolute Error
std_error_world = np.std(residuos_world)       # Standard Deviation of the error

# Calculate error metrics for the ALIGNED ODOMETRY trajectory against the ideal circle.
residuos_odom = calc_R(x_odom_aligned, y_odom_aligned, xc_ideal, yc_ideal) - R_IDEAL
rmse_odom = np.sqrt(np.mean(residuos_odom**2))
mae_odom = np.mean(np.abs(residuos_odom))
std_error_odom = np.std(residuos_odom)

# --- Print Results ---

print("--- Comparative Analysis vs. Reference Circle (Fixed Radius = 1.5m) ---")
print(f"\nReference Circle:")
print(f"  Fixed Radius: R={R_IDEAL:.2f} m")
print(f"  Optimal Center based on Camera Data: (xc={xc_ideal:.2f}, yc={yc_ideal:.2f}) m")
print("\nAnalysis of Camera Trajectory (World):")
print(f"  RMSE vs. 1.5m Circle: {rmse_world:.3f} m ({rmse_world*100:.1f} cm)")
print(f"  MAE vs. 1.5m Circle:  {mae_world:.3f} m ({mae_world*100:.1f} cm)")
print(f"  Std. Dev. of Error:   {std_error_world:.3f} m ({std_error_world*100:.1f} cm)")
print("\nAnalysis of Robot Odometry (Aligned):")
print(f"  RMSE vs. 1.5m Circle: {rmse_odom:.3f} m ({rmse_odom*100:.1f} cm)")
print(f"  MAE vs. 1.5m Circle:  {mae_odom:.3f} m ({mae_odom*100:.1f} cm)")
print(f"  Std. Dev. of Error:   {std_error_odom:.3f} m ({std_error_odom*100:.1f} cm)")
print("-----------------------------------------------------------------")

# --- Visualization ---

# Generate points for plotting the ideal reference circle.
theta = np.linspace(0, 2 * np.pi, 200)
x_circ_ideal = xc_ideal + R_IDEAL * np.cos(theta)
y_circ_ideal = yc_ideal + R_IDEAL * np.sin(theta)

# Configure Matplotlib for high-quality, IEEE-style publication plots.
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})

# Create the plot figure and axes.
# A square figure size is often good for plots with equal aspect ratios.
fig, ax = plt.subplots(figsize=(5, 5))

# Plot the ideal reference circle (black, dashed).
ax.plot(x_circ_ideal, y_circ_ideal, '--', linewidth=2, label=f'Reference Circle (R={R_IDEAL}m)', color='black')

# Plot the world trajectory from camera data (blue, solid).
ax.plot(x_world, y_world, '-', linewidth=2, color="blue", 
        label=f'Triangulated Trajectory (RMSE: {rmse_world*100:.1f} cm)', alpha=0.8) 

# Plot the aligned odometry trajectory (red, solid).
ax.plot(x_odom_aligned, y_odom_aligned, '-', linewidth=2, color="red",
        label=f'Robot Odometry (RMSE: {rmse_odom*100:.1f} cm)', alpha=0.8)

# Apply final formatting to the plot.
ax.set_xlabel('X Position (m)', fontsize=10)
ax.set_ylabel('Y Position (m)', fontsize=10)
ax.set_aspect('equal', adjustable='box') # Ensures the circle looks like a circle.
ax.legend(fontsize=9, loc='best', frameon=True, fancybox=False, edgecolor='black')
ax.grid(True, linestyle='-', alpha=0.3)

# Use a tight layout to prevent labels from being cut off.
plt.tight_layout()

# Save the figure in multiple formats for different uses (PNG for viewing, EPS/PDF for papers).
plt.savefig("reference_circle_comparison.png", dpi=400, bbox_inches='tight', pad_inches=0.05)
plt.savefig("reference_circle_comparison.eps", format='eps', dpi=400, bbox_inches='tight', pad_inches=0.05)
print("\nPlot saved as 'reference_circle_comparison.png' and '.eps'")

# Display the plot.
plt.show()