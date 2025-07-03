# -*- coding: utf-8 -*-
"""
Robot Trajectory Visualization from Odometry Data.

This script reads robot odometry data from a specified CSV file and generates a 2D
top-down plot of its trajectory. It visualizes the path taken by the robot by
plotting its X and Y positions.

Key Features:
- Reads odometry data using the pandas library.
- Plots the trajectory using matplotlib.
- Clearly marks the start (green circle) and end (red circle) points of the path.
- Enhances the plot with a title, axis labels, grid, and an equal aspect ratio
  to prevent distortion.
- Saves the final plot as a PNG image file.
- Includes error handling for common issues like a missing file or incorrect column names.
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Define the name of the input CSV file containing the odometry data.
csv_filename = 'rosbags/rosbag2_2025_06_11-15_09_30_csv/odrive_odom.csv'
output_image_filename = 'robot_trajectory_plot.png'

try:
    # --- Data Loading ---
    # Attempt to read the CSV file into a pandas DataFrame.
    df = pd.read_csv(csv_filename)
    print("Successfully read the CSV file!")

    # --- Plot Creation ---
    # Create a figure with a defined size for better visualization.
    plt.figure(figsize=(10, 8))

    # Plot the primary trajectory using X and Y position columns.
    # 'linestyle' creates the connecting line, and 'marker' shows individual points.
    plt.plot(df['pose.pose.position.x'], df['pose.pose.position.y'],
             linestyle='-', marker='.', markersize=3, label='Trajectory')

    # Mark the start and end points for better context.
    # .iloc[0] accesses the first row (start point).
    # 'go' stands for green ('g') circle ('o').
    plt.plot(df['pose.pose.position.x'].iloc[0], df['pose.pose.position.y'].iloc[0],
             'go', markersize=10, label='Start')

    # .iloc[-1] accesses the last row (end point).
    # 'ro' stands for red ('r') circle ('o').
    plt.plot(df['pose.pose.position.x'].iloc[-1], df['pose.pose.position.y'].iloc[-1],
             'ro', markersize=10, label='End')

    # --- Plot Enhancements ---
    plt.title('Robot Trajectory (Top-Down View)')
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.grid(True)  # Add a grid for easier reading of coordinates.
    plt.axis('equal')  # Ensure the scaling on X and Y axes is identical to prevent path distortion.
    plt.legend()  # Display the legend (e.g., "Trajectory", "Start", "End").
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping.

    # --- Output ---
    # Save the plot to an image file.
    plt.savefig(output_image_filename)
    # Display the plot on the screen.
    plt.show()
    print(f"Plot successfully saved as '{output_image_filename}'")

# --- Error Handling ---
except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found. Please ensure it is in the correct path.")
except KeyError as e:
    print(f"Error: The column {e} was not found in the file. Please verify the CSV column names.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")