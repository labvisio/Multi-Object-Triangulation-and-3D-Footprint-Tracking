# -*- coding: utf-8 -*-
"""
Comparative Analysis of Robot Trajectories: Odometry vs. Camera Reconstruction.

This script loads, processes, and visualizes two different trajectory datasets for a robot:
1.  Internal odometry data recorded from the robot's sensors (from a CSV file).
2.  Externally captured trajectory data from a camera-based 3D reconstruction system
    (from a JSON file).

The script performs the following steps:
- Loads both datasets.
- Applies a manual coordinate transformation to the odometry data to align its
  coordinate frame with the world frame used by the camera system.
- Generates a 2x2 multi-panel plot to visually compare the trajectories:
    - Plot 1: An overlay of both trajectories.
    - Plot 2: The odometry trajectory in isolation with start/end points.
    - Plot 3: The camera reconstruction trajectory in isolation with start/end points.
    - Plot 4: An analysis of the Euclidean distance (error) between corresponding
      points of the two trajectories over time.
- Prints a final statistical report summarizing the datasets and the calculated error metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# --- 1. DATA LOADING AND PREPARATION ---

# Load robot odometry data from a CSV file
try:
    df_odometry = pd.read_csv('rosbags/rosbag2_2025_06_11-15_09_30_csv/odrive_odom.csv')
except FileNotFoundError:
    print("Error: Odometry file 'odrive_odom.csv' not found. Please check the path.")
    exit()

# Load camera-based reconstruction data from a JSON file
try:
    with open('experiments/paper/robis/robis_circle_3D_plot.json', 'r') as f:
        camera_reconstruction_data = json.load(f)
except FileNotFoundError:
    print("Error: Camera reconstruction file 'robis_circle_3D_plot.json' not found.")
    exit()


# --- 2. DATA PROCESSING AND TRANSFORMATION ---

# Manually transform the robot's local odometry coordinates to the world frame.
# NOTE: This transformation (swapping X/Y and applying an offset) is specific
# to this experimental setup. It aligns the robot's starting orientation and
# position with the world coordinate system defined by the camera setup.
odom_x_world = df_odometry['pose.pose.position.y'] - 1.5
odom_y_world = df_odometry['pose.pose.position.x']

# Extract the X and Y positions from the camera reconstruction data.
cam_x_world = []
cam_y_world = []
for frame_data in camera_reconstruction_data:
    # Check if the 'points' key exists and is not empty
    if 'points' in frame_data and len(frame_data['points']) > 0:
        # Assume the first point detected in each frame is the target
        point = frame_data['points'][0]
        cam_x_world.append(point['position'][0])
        cam_y_world.append(point['position'][1])

# --- 3. VISUALIZATION: TRAJECTORY COMPARISON PLOT ---

plt.figure(figsize=(16, 12))

# Subplot 1: Direct comparison of the two trajectories
plt.subplot(2, 2, 1)
plt.plot(odom_x_world, odom_y_world,
         linestyle='-', marker='.', markersize=3, label='Robot Odometry', color='blue', alpha=0.7)

if cam_x_world and cam_y_world:
    plt.plot(cam_x_world, cam_y_world,
             linestyle='-', marker='o', markersize=3, label='Camera Reconstruction', color='red', alpha=0.7)

plt.title('Trajectory Comparison: Odometry vs. Camera')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.grid(True)
plt.axis('equal')  # Ensure aspect ratio is 1:1, so circles are not distorted
plt.legend()

# Subplot 2: Odometry trajectory only, with start and end points
plt.subplot(2, 2, 2)
plt.plot(odom_x_world, odom_y_world,
         linestyle='-', marker='.', markersize=3, label='Odometry', color='blue')
plt.plot(odom_x_world.iloc[0], odom_y_world.iloc[0],
         'go', markersize=8, label='Start')
plt.plot(odom_x_world.iloc[-1], odom_y_world.iloc[-1],
         'ro', markersize=8, label='End')
plt.title('Robot Odometry Trajectory')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.grid(True)
plt.axis('equal')
plt.legend()

# Subplot 3: Camera reconstruction trajectory only, with start and end points
plt.subplot(2, 2, 3)
if cam_x_world and cam_y_world:
    plt.plot(cam_x_world, cam_y_world,
             linestyle='-', marker='o', markersize=3, label='Camera', color='red')
    plt.plot(cam_x_world[0], cam_y_world[0], 'go', markersize=8, label='Start')
    plt.plot(cam_x_world[-1], cam_y_world[-1], 'ro', markersize=8, label='End')
    plt.title('Camera Reconstruction Trajectory')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
else:
    # Display a message if camera data is unavailable
    plt.text(0.5, 0.5, 'Camera data not available',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Camera Reconstruction Trajectory')

# Subplot 4: Error analysis (if both datasets are available)
plt.subplot(2, 2, 4)
if cam_x_world and cam_y_world:
    # For a direct comparison, truncate both datasets to the length of the shorter one.
    # This assumes the data streams start at roughly the same time.
    min_len = min(len(odom_x_world), len(cam_x_world))
    
    if min_len > 1:
        # Slice the datasets to the same length for point-wise comparison
        odom_x_truncated = odom_x_world.iloc[:min_len]
        odom_y_truncated = odom_y_world.iloc[:min_len]
        cam_x_truncated = cam_x_world[:min_len]
        cam_y_truncated = cam_y_world[:min_len]
        
        # Calculate the Euclidean distance between corresponding points
        distances = np.sqrt((np.array(odom_x_truncated) - np.array(cam_x_truncated))**2 +
                            (np.array(odom_y_truncated) - np.array(cam_y_truncated))**2)
        
        plt.plot(range(len(distances)), distances, 'g-', linewidth=2)
        plt.title(f'Error Between Trajectories\nMean Error: {np.mean(distances):.4f}m')
        plt.xlabel('Sample Index')
        plt.ylabel('Euclidean Distance (meters)')
        plt.grid(True)
        
        # Add a text box with key error statistics
        stats_text = (f'Max Error: {np.max(distances):.4f}m\n'
                      f'Min Error: {np.min(distances):.4f}m\n'
                      f'Std Dev: {np.std(distances):.4f}m')
        plt.text(0.02, 0.98, stats_text,
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'Insufficient data for analysis',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Error Analysis')
else:
    plt.text(0.5, 0.5, 'Camera data not available\nfor error analysis',
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.title('Error Analysis')

plt.suptitle('Comparative Analysis of Robot Trajectories', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()

# --- 4. STATISTICAL REPORT ---
print("\n" + "="*50)
print("   TRAJECTORY COMPARISON REPORT")
print("="*50)
print(f"Odometry data points: {len(df_odometry)}")
print(f"Camera data points:   {len(cam_x_world)}")

if cam_x_world and cam_y_world:
    print("\n--- Trajectory Extents ---")
    print(f"Odometry Extent:")
    print(f"  X: from {odom_x_world.min():.4f} to {odom_x_world.max():.4f} m")
    print(f"  Y: from {odom_y_world.min():.4f} to {odom_y_world.max():.4f} m")

    print(f"\nCamera Extent:")
    print(f"  X: from {min(cam_x_world):.4f} to {max(cam_x_world):.4f} m")
    print(f"  Y: from {min(cam_y_world):.4f} to {max(cam_y_world):.4f} m")
    
    # Recalculate distances for the final report
    if len(cam_x_world) > 1:
        min_len = min(len(odom_x_world), len(cam_x_world))
        odom_x_comp = odom_x_world.iloc[:min_len]
        odom_y_comp = odom_y_world.iloc[:min_len]
        cam_x_comp = cam_x_world[:min_len]
        cam_y_comp = cam_y_world[:min_len]
        
        distances = np.sqrt((np.array(odom_x_comp) - np.array(cam_x_comp))**2 +
                            (np.array(odom_y_comp) - np.array(cam_y_comp))**2)
        
        print("\n--- ERROR STATISTICS ---")
        print(f"Mean Error:      {np.mean(distances):.4f} m")
        print(f"Max Error:       {np.max(distances):.4f} m")
        print(f"Min Error:       {np.min(distances):.4f} m")
        print(f"Std. Deviation:  {np.std(distances):.4f} m")
        print(f"RMS Error:       {np.sqrt(np.mean(distances**2)):.4f} m")
        print("="*50)
else:
    print("\nComparative error analysis could not be performed due to missing camera data.")