# -*- coding: utf-8 -*-
"""
Ground Truth Position Comparison and Analysis Tool

This script loads, analyzes, and visualizes 3D position data from two distinct
experimental datasets:
1.  ArUco marker data, containing both ground truth (actual) and estimated positions.
2.  Data from triangulated positions of people standing on a grid.

The script provides three main functionalities:
-   `print_statistics()`: Computes and prints summary statistics (min, max, mean,
    and standard deviation) for each coordinate (X, Y, Z) across all datasets.
-   `create_comparison_plot()`: Generates publication-quality 2D and 3D scatter
    plots to visually compare the spatial distribution of the different data sources.
-   `create_detailed_analysis()`: Creates a multi-panel plot with histograms for
    each coordinate and a 2D coverage area comparison using covariance ellipses.

The output includes console printouts and saved plot files in both PDF and PNG formats,
styled according to common IEEE publication standards.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# Configure Matplotlib for high-quality, IEEE-style publication plots.
# A serif font is used as a common substitute for Times New Roman.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.0

def load_data():
    """
    Loads position data from the ArUco and people ground truth JSON files.

    Returns:
        tuple[dict, dict]: A tuple containing the loaded data from the ArUco
                           JSON file and the people JSON file.
    """
    # Load ArUco ground truth and estimated position data
    with open('experiments/ground_truth/aruco_ground_grid_3d_coordinates.json', 'r') as f:
        aruco_data = json.load(f)
    
    # Load triangulated positions of people
    with open('experiments/ground_truth/people_standing_grid_3d_coordinates.json', 'r') as f:
        people_data = json.load(f)
    
    return aruco_data, people_data

def extract_positions(aruco_data, people_data):
    """
    Extracts and structures 3D position data into NumPy arrays.

    This function parses the raw dictionary data loaded from the JSON files and
    organizes the coordinates into separate, easy-to-use NumPy arrays.

    Args:
        aruco_data (dict): The loaded data from the ArUco JSON file.
        people_data (dict): The loaded data from the people JSON file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three
            NumPy arrays: ArUco ground truth positions, ArUco estimated
            positions, and people's triangulated positions.
    """
    # Extract ArUco ground truth positions (the actual reference points)
    aruco_ground_truth = [
        [comp['actual_position']['x'], comp['actual_position']['y'], comp['actual_position']['z']]
        for comp in aruco_data['position_comparisons']
    ]
    
    # Extract ArUco estimated positions (the algorithm's output)
    aruco_estimated = [
        [comp['estimated_position']['x'], comp['estimated_position']['y'], comp['estimated_position']['z']]
        for comp in aruco_data['position_comparisons']
    ]
    
    # Extract people's triangulated positions
    people_positions = [point for capture in people_data for point in capture['points']]
    
    return np.array(aruco_ground_truth), np.array(aruco_estimated), np.array(people_positions)

def create_comparison_plot():
    """
    Generates and saves 2D and 3D scatter plots comparing the position data.

    This function creates two separate figures:
    1.  A 2D top-down (X-Y plane) view.
    2.  A 3D scatter plot.

    Both plots compare the ground truth, ArUco estimated, and people's triangulated
    positions. The plots are saved in both PDF and PNG formats.
    """
    aruco_data, people_data = load_data()
    aruco_gt, aruco_est, people_pos = extract_positions(aruco_data, people_data)
    
    # --- Create 2D Figure (Top-Down View) ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    
    # Plot each dataset with distinct markers and colors
    ax1.scatter(aruco_gt[:, 0], aruco_gt[:, 1], 
               marker='s', s=50, c='green', alpha=0.8, 
               label='Ground Truth', edgecolors='darkgreen', linewidth=0.5)
    ax1.scatter(aruco_est[:, 0], aruco_est[:, 1], 
               marker='o', s=40, c='blue', alpha=0.7, 
               label='ArUco Estimated', edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(people_pos[:, 0], people_pos[:, 1], 
               marker='^', s=40, c='red', alpha=0.7, 
               label='Person Triangulated', edgecolors='darkred', linewidth=0.5)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, frameon=True, fancybox=False, shadow=False)
    ax1.set_aspect('equal', adjustable='box') # Ensures X and Y axes have the same scale
    
    plt.tight_layout()
    plt.savefig('experiments/ground_truth/position_comparison_2d_ieee.pdf', 
                dpi=400, bbox_inches='tight', format='pdf')
    plt.savefig('experiments/ground_truth/position_comparison_2d_ieee.png', 
                dpi=400, bbox_inches='tight', format='png')
    plt.show()
    
    # --- Create 3D Figure ---
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    ax2.scatter(aruco_gt[:, 0], aruco_gt[:, 1], aruco_gt[:, 2],
               marker='s', s=50, c='green', alpha=0.8, 
               label='Ground Truth', edgecolors='darkgreen', linewidth=0.5)
    ax2.scatter(aruco_est[:, 0], aruco_est[:, 1], aruco_est[:, 2],
               marker='o', s=40, c='blue', alpha=0.7, 
               label='ArUco Estimated', edgecolors='darkblue', linewidth=0.5)
    ax2.scatter(people_pos[:, 0], people_pos[:, 1], people_pos[:, 2],
               marker='^', s=40, c='red', alpha=0.7, 
               label='Person Triangulated', edgecolors='darkred', linewidth=0.5)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_zlabel('Z (m)', fontsize=12)
    ax2.set_zlim(0, 2)  # Set Z-axis limits for better visualization
    ax2.legend(fontsize=12, frameon=True, fancybox=False, shadow=False)
    
    plt.tight_layout()
    plt.savefig('experiments/ground_truth/position_comparison_3d_ieee.pdf', 
                dpi=400, bbox_inches='tight', format='pdf')
    plt.savefig('experiments/ground_truth/position_comparison_3d_ieee.png', 
                dpi=400, bbox_inches='tight', format='png')
    plt.show()

def create_detailed_analysis():
    """
    Generates a multi-panel plot for detailed statistical analysis.

    This function creates a 2x2 figure containing:
    - Histograms of X, Y, and Z coordinate distributions for ArUco and people data.
    - A 2D scatter plot showing the coverage area of both datasets, visualized
      with 1-standard-deviation covariance ellipses.
    """
    aruco_data, people_data = load_data()
    aruco_gt, _, people_pos = extract_positions(aruco_data, people_data) # We only need GT and people here
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # --- Histograms for Coordinate Distributions ---
    # X-coordinate distribution
    axes[0, 0].hist(aruco_gt[:, 0], bins=15, alpha=0.7, label='ArUco GT', color='blue', edgecolor='darkblue')
    axes[0, 0].hist(people_pos[:, 0], bins=15, alpha=0.7, label='People', color='red', edgecolor='darkred')
    axes[0, 0].set_xlabel('X Position (m)'); axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('X-Coordinate Distribution'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # Y-coordinate distribution
    axes[0, 1].hist(aruco_gt[:, 1], bins=15, alpha=0.7, label='ArUco GT', color='blue', edgecolor='darkblue')
    axes[0, 1].hist(people_pos[:, 1], bins=15, alpha=0.7, label='People', color='red', edgecolor='darkred')
    axes[0, 1].set_xlabel('Y Position (m)'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Y-Coordinate Distribution'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    # Z-coordinate distribution
    axes[1, 0].hist(aruco_gt[:, 2], bins=15, alpha=0.7, label='ArUco GT', color='blue', edgecolor='darkblue')
    axes[1, 0].hist(people_pos[:, 2], bins=15, alpha=0.7, label='People', color='red', edgecolor='darkred')
    axes[1, 0].set_xlabel('Z Position (m)'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Z-Coordinate Distribution'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    # --- Coverage Area Comparison with Covariance Ellipses ---
    ax_cov = axes[1, 1]
    ax_cov.scatter(aruco_gt[:, 0], aruco_gt[:, 1], marker='o', s=20, c='blue', alpha=0.6, label='ArUco GT')
    ax_cov.scatter(people_pos[:, 0], people_pos[:, 1], marker='^', s=20, c='red', alpha=0.6, label='People')
    
    # Calculate and plot covariance ellipse for ArUco data
    # Covariance matrix describes the shape and orientation of the data spread
    aruco_cov = np.cov(aruco_gt[:, :2], rowvar=False)
    eigenvals_a, eigenvecs_a = np.linalg.eigh(aruco_cov)
    angle_a = np.degrees(np.arctan2(*eigenvecs_a[:, 1]))
    ellipse_a = Ellipse(xy=np.mean(aruco_gt[:, :2], axis=0),
                       width=2 * np.sqrt(eigenvals_a[0]), height=2 * np.sqrt(eigenvals_a[1]),
                       angle=angle_a, facecolor='none', edgecolor='blue', linewidth=1.5, label='ArUco 1σ Covariance')

    # Calculate and plot covariance ellipse for people data
    people_cov = np.cov(people_pos[:, :2], rowvar=False)
    eigenvals_p, eigenvecs_p = np.linalg.eigh(people_cov)
    angle_p = np.degrees(np.arctan2(*eigenvecs_p[:, 1]))
    ellipse_p = Ellipse(xy=np.mean(people_pos[:, :2], axis=0),
                       width=2 * np.sqrt(eigenvals_p[0]), height=2 * np.sqrt(eigenvals_p[1]),
                       angle=angle_p, facecolor='none', edgecolor='red', linewidth=1.5, label='People 1σ Covariance')

    ax_cov.add_patch(ellipse_a)
    ax_cov.add_patch(ellipse_p)
    ax_cov.set_xlabel('X Position (m)'); ax_cov.set_ylabel('Y Position (m)')
    ax_cov.set_title('Coverage Area Comparison'); ax_cov.legend(); ax_cov.grid(True, alpha=0.3)
    ax_cov.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('experiments/ground_truth/detailed_comparison_ieee.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/ground_truth/detailed_comparison_ieee.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics():
    """
    Calculates and prints a statistical summary of the position data to the console.

    The summary includes the number of points and the min, max, mean (μ), and
    standard deviation (σ) for the X, Y, and Z coordinates of each dataset.
    """
    aruco_data, people_data = load_data()
    aruco_gt, aruco_est, people_pos = extract_positions(aruco_data, people_data)
    
    print("=== Position Comparison Statistics ===")
    print(f"Total ArUco Ground Truth Points: {len(aruco_gt)}")
    print(f"Total ArUco Estimated Points:  {len(aruco_est)}")
    print(f"Total People Standing Points:  {len(people_pos)}\n")
    
    print("ArUco Ground Truth Statistics:")
    print(f"  X-range: [{aruco_gt[:, 0].min():.3f}, {aruco_gt[:, 0].max():.3f}] m,  μ={aruco_gt[:, 0].mean():.3f}, σ={aruco_gt[:, 0].std():.3f}")
    print(f"  Y-range: [{aruco_gt[:, 1].min():.3f}, {aruco_gt[:, 1].max():.3f}] m,  μ={aruco_gt[:, 1].mean():.3f}, σ={aruco_gt[:, 1].std():.3f}")
    print(f"  Z-range: [{aruco_gt[:, 2].min():.3f}, {aruco_gt[:, 2].max():.3f}] m,  μ={aruco_gt[:, 2].mean():.3f}, σ={aruco_gt[:, 2].std():.3f}\n")
    
    print("ArUco Estimated Statistics:")
    print(f"  X-range: [{aruco_est[:, 0].min():.3f}, {aruco_est[:, 0].max():.3f}] m,  μ={aruco_est[:, 0].mean():.3f}, σ={aruco_est[:, 0].std():.3f}")
    print(f"  Y-range: [{aruco_est[:, 1].min():.3f}, {aruco_est[:, 1].max():.3f}] m,  μ={aruco_est[:, 1].mean():.3f}, σ={aruco_est[:, 1].std():.3f}")
    print(f"  Z-range: [{aruco_est[:, 2].min():.3f}, {aruco_est[:, 2].max():.3f}] m,  μ={aruco_est[:, 2].mean():.3f}, σ={aruco_est[:, 2].std():.3f}\n")
    
    print("People Standing Statistics:")
    print(f"  X-range: [{people_pos[:, 0].min():.3f}, {people_pos[:, 0].max():.3f}] m,  μ={people_pos[:, 0].mean():.3f}, σ={people_pos[:, 0].std():.3f}")
    print(f"  Y-range: [{people_pos[:, 1].min():.3f}, {people_pos[:, 1].max():.3f}] m,  μ={people_pos[:, 1].mean():.3f}, σ={people_pos[:, 1].std():.3f}")
    print(f"  Z-range: [{people_pos[:, 2].min():.3f}, {people_pos[:, 2].max():.3f}] m,  μ={people_pos[:, 2].mean():.3f}, σ={people_pos[:, 2].std():.3f}")

if __name__ == "__main__":
    # The main execution block.
    # It first prints the statistical summary, then creates the primary comparison plots.
    print_statistics()
    create_comparison_plot()
    
    # The detailed analysis function is commented out by default.
    # Uncomment the line below to generate the additional statistical plots.
    # create_detailed_analysis()