# -*- coding: utf-8 -*-
"""
3D Trajectory Visualization Tool from JSON Data.

This script loads 3D trajectory data for multiple objects from a JSON file and
creates sophisticated visualizations. It supports both animated and static 3D plots.

A key feature is its ability to recognize and utilize object class information
(e.g., 'person', 'chair') to apply distinct styling (colors and markers) for
clearer visualization.

The script provides a flexible command-line interface to:
- Specify the input and output file paths.
- Choose between generating an animated video or a static PNG image.
- Filter the visualization to show only specific object classes.
- Save the output without displaying it interactively.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse

def load_trajectory_data(json_file):
    """
    Loads trajectory data from a specified JSON file.

    Args:
        json_file (str): The path to the input JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a frame.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_trajectories(data):
    """
    Parses raw trajectory data into structured NumPy arrays for plotting.

    This function processes a list of frame data to identify all unique object IDs
    and organizes their positions, timestamps, and classes into aligned NumPy arrays.
    It handles missing data by using np.nan as a placeholder.

    Args:
        data (list): The raw data loaded from the JSON file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: `trajectories` (frames, num_objects, 3) for XYZ positions.
            - list: `timestamps` for each frame.
            - list: `frame_numbers` for each frame.
            - np.ndarray: `classes` (num_objects,) for the class ID of each object.
            - list: `id_list` containing the unique object IDs in sorted order.
    """
    frames = len(data)
    
    # First pass: find all unique object IDs to determine array dimensions
    unique_ids = set()
    for frame_data in data:
        for point in frame_data.get('points', []):
            if isinstance(point, dict) and 'id' in point:
                unique_ids.add(point['id'])
    
    # Create a sorted list and a mapping from ID to array index for consistency
    id_list = sorted(list(unique_ids))
    id_to_idx = {obj_id: idx for idx, obj_id in enumerate(id_list)}
    num_objects = len(id_list)
    
    # Initialize arrays with appropriate dimensions. Use np.nan for missing points.
    trajectories = np.full((frames, num_objects, 3), np.nan)
    classes = np.full(num_objects, -1, dtype=int)  # Default class is -1 (unknown)
    timestamps = []
    frame_numbers = []
    
    # Second pass: populate the arrays with data
    for i, frame_data in enumerate(data):
        timestamps.append(frame_data.get('timestamp', f"Frame {i}"))
        frame_numbers.append(frame_data.get('frame', i))
        
        for point in frame_data.get('points', []):
            if isinstance(point, dict) and 'id' in point and 'position' in point:
                obj_id = point['id']
                if obj_id in id_to_idx:
                    idx = id_to_idx[obj_id]
                    
                    # Store position data
                    trajectories[i, idx] = point['position']
                    
                    # Store class ID, but only if it hasn't been set yet
                    if 'class' in point and classes[idx] == -1:
                        classes[idx] = point['class']
    
    return trajectories, timestamps, frame_numbers, classes, id_list

def get_class_name(class_id):
    """
    Converts a numeric class ID to a human-readable name.

    Args:
        class_id (int): The class ID (e.g., from COCO dataset).

    Returns:
        str: The corresponding class name or a default string.
    """
    # Extend this dictionary with more COCO class mappings as needed
    class_names = {
        0: 'person',
        56: 'chair',
    }
    return class_names.get(class_id, f"class_{class_id}")

def plot_trajectories(trajectories, timestamps, frame_numbers, classes, id_list, output_file=None):
    """
    Creates and optionally saves a 3D animated plot of the trajectories.

    Args:
        trajectories (np.ndarray): Array of XYZ positions.
        timestamps (list): List of timestamps for each frame.
        frame_numbers (list): List of frame numbers.
        classes (np.ndarray): Array of class IDs for each object.
        id_list (list): List of unique object IDs.
        output_file (str, optional): Path to save the animation as an MP4 file.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define color and marker styles based on object class
    class_colors = {0: 'tab:blue', 56: 'tab:red', -1: 'tab:gray'}
    class_markers = {0: 'o', 56: 's', -1: 'x'}
    
    # Determine axis limits dynamically with a 10% buffer
    valid_points = trajectories[~np.isnan(trajectories)]
    ax_min, ax_max = (np.min(valid_points) - 0.5, np.max(valid_points) + 0.5) if valid_points.size > 0 else (-2, 2)
    
    # Initialize plot elements (lines for trails, points for current position)
    lines, points = [], []
    class_handles = {}  # To create a clean legend with one entry per class
    
    for i in range(trajectories.shape[1]):
        obj_id, class_id = id_list[i], int(classes[i])
        color = class_colors.get(class_id, f"C{i % 10}")
        marker = class_markers.get(class_id, 'o')
        label = f"{get_class_name(class_id)} ID:{obj_id}"
        
        line, = ax.plot([], [], [], '.-', color=color, alpha=0.7, linewidth=2)
        point, = ax.plot([], [], [], marker=marker, markersize=10, color=color, label=label)
        lines.append(line); points.append(point)
        
        if class_id not in class_handles:
            class_handles[class_id] = point
    
    # Setup plot aesthetics
    title = ax.set_title('')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim(ax_min, ax_max); ax.set_ylim(ax_min, ax_max); ax.set_zlim(ax_min, ax_max)
    ax.grid(True)
    ax.legend(handles=list(class_handles.values()), labels=[h.get_label() for h in class_handles.values()], loc='upper right')
    
    def init():
        """Initializes the animation plot, clearing all lines and points."""
        for line, point in zip(lines, points):
            line.set_data_3d([], [], [])
            point.set_data_3d([], [], [])
        return lines + points + [title]
    
    def animate(frame):
        """Updates the plot for each animation frame."""
        title.set_text(f'Frame: {frame_numbers[frame]}, Time: {timestamps[frame]}')
        
        for i, (line, point) in enumerate(zip(lines, points)):
            # Update trajectory trail (all points up to the current frame)
            x, y, z = trajectories[:frame+1, i].T
            valid = ~np.isnan(x)
            line.set_data_3d(x[valid], y[valid], z[valid])
            
            # Update current position marker
            current_pos = trajectories[frame, i]
            if not np.isnan(current_pos).any():
                point.set_data_3d([current_pos[0]], [current_pos[1]], [current_pos[2]])
            else:
                point.set_data_3d([], [], [])
        return lines + points + [title]
    
    # Create the animation object
    ani = FuncAnimation(fig, animate, frames=len(trajectories), init_func=init, interval=100, blit=True)
    
    plt.tight_layout()
    
    # Save to file if an output path is provided
    if output_file:
        print(f"Saving animation to {output_file}...")
        ani.save(output_file, writer='ffmpeg', dpi=150)
        print("Animation saved successfully!")
    else:
        plt.show()

def create_static_plot(trajectories, classes, id_list, output_file=None):
    """
    Creates a static 3D plot showing the complete trajectories.

    Args:
        trajectories (np.ndarray): Array of XYZ positions.
        classes (np.ndarray): Array of class IDs for each object.
        id_list (list): List of unique object IDs.
        output_file (str, optional): Path to save the static plot as a PNG image.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    class_colors = {0: 'tab:blue', 56: 'tab:red', -1: 'tab:gray'}
    class_markers = {0: 'o', 56: 's', -1: 'x'}
    
    for i in range(trajectories.shape[1]):
        obj_id, class_id = id_list[i], int(classes[i])
        color = class_colors.get(class_id, f"C{i % 10}")
        
        x, y, z = trajectories[:, i].T
        valid = ~np.isnan(x)
        if np.any(valid):
            x_valid, y_valid, z_valid = x[valid], y[valid], z[valid]
            # Plot the full trajectory line
            ax.plot(x_valid, y_valid, z_valid, '-', color=color, alpha=0.7, linewidth=2)
            # Mark start (circle) and end (square) points
            ax.scatter(x_valid[0], y_valid[0], z_valid[0], s=100, color=color, marker='o', label=f"Start {get_class_name(class_id)} ID:{obj_id}")
            ax.scatter(x_valid[-1], y_valid[-1], z_valid[-1], s=100, color=color, marker='s', label=f"End {get_class_name(class_id)} ID:{obj_id}")
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.grid(True); ax.set_title('Static 3D Trajectories')
    
    # Handle duplicate legend entries to create a clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to {output_file}")
    plt.show()

def main():
    """Main function to parse arguments and run the visualization."""
    parser = argparse.ArgumentParser(description='Plot 3D trajectories from JSON data with class-based styling.')
    parser.add_argument('--file', '-f', type=str, help='Path to the JSON trajectory file.')
    parser.add_argument('--output', '-o', type=str, help='Output path for the animation video (e.g., animation.mp4).')
    parser.add_argument('--static', '-s', action='store_true', help='Generate a static plot instead of an animation.')
    parser.add_argument('--class-filter', '-c', type=int, nargs='+', help='Filter by class IDs (e.g., --class-filter 0 56).')
    args = parser.parse_args()
    
    # Automatically find a default JSON file if none is provided
    if not args.file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_files = [
            os.path.join(script_dir, 'teste0.json'),
            os.path.join(script_dir, 'output.json'),
            os.path.join(script_dir, 'rastro_pessoa.json'),
            os.path.join(script_dir, 'experiments/three_chairs/teste_tracking_kalman.json')
        ]
        for potential_file in potential_files:
            if os.path.isfile(potential_file):
                args.file = potential_file
                break
        else:
            print("Error: Could not find a default trajectory file. Please specify one using --file.")
            return

    print(f"Loading data from {args.file}")
    data = load_trajectory_data(args.file)
    trajectories, timestamps, frame_numbers, classes, id_list = extract_trajectories(data)
    
    # Apply class filtering if specified by the user
    if args.class_filter:
        class_mask = np.isin(classes, args.class_filter)
        if not np.any(class_mask):
            print(f"No trajectories found for specified classes: {args.class_filter}")
            return
        trajectories = trajectories[:, class_mask, :]
        classes = classes[class_mask]
        id_list = [id_val for i, id_val in enumerate(id_list) if class_mask[i]]
        print(f"Filtered to show only classes: {[get_class_name(c) for c in args.class_filter]}")
    
    # Decide which plotting function to call based on arguments
    if args.static:
        static_output_path = args.output.replace('.mp4', '_static.png') if args.output else 'static_plot.png'
        create_static_plot(trajectories, classes, id_list, static_output_path)
    else:
        plot_trajectories(trajectories, timestamps, frame_numbers, classes, id_list, args.output)

if __name__ == "__main__":
    main()