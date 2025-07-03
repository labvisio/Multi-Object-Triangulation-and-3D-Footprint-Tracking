"""
Plot Tracking Data from JSON Files

This script loads tracking data from JSON files and visualizes it using the same
plotting style as the main tracking system, without needing to run the full pipeline.
"""

import argparse
import json
import logging
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Set global font size to 12
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.0

# Import necessary modules from the tracking system
sys.path.append('source')
from ploting_utils import Utils
from config import CLASS_NAMES
from visualization_utils import visualize_camera_positions
from load_fundamental_matrices import FundamentalMatrices

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set matplotlib style for white background
PLOT_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set1.colors)

def load_json_data(json_file):
    """
    Load tracking data from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        list: List of frame data dictionaries
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} frames from {json_file}")
        return data
    except Exception as e:
        logging.error(f"Error loading {json_file}: {str(e)}")
        return []

def extract_frame_data(frame_data):
    """
    Extract 3D positions, IDs, and class information from a frame data dictionary.
    
    Args:
        frame_data (dict): Single frame data from JSON
        
    Returns:
        tuple: (positions, ids, class_ids, frame_number)
    """
    positions = []
    ids = []
    class_ids = []
    
    frame_number = frame_data.get('frame', 0)
    
    for point_data in frame_data.get('points', []):
        if 'position' in point_data and 'id' in point_data:
            positions.append(point_data['position'])
            ids.append(point_data['id'])
            class_ids.append(point_data.get('class', 0))  # Default to class 0 if not specified
    
    return positions, ids, class_ids, frame_number

def setup_3d_axis(ax):
    """Configure 3D axis properties for consistent visualization."""
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([0, 4]) # Match plot_position_comparison.py
    
    # Set axis labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

def process_dataset_for_plotting(data, utils):
    """
    Process a dataset and extract trajectories, colors, and classes.
    
    Args:
        data (list): List of frame data
        utils: Utils object for color generation
        
    Returns:
        tuple: (trajectories, colors, classes) dictionaries
    """
    trajectories = {}
    colors = {}
    classes = {}
    
    # Process all frames
    for frame_data in data:
        positions, ids, class_ids, frame_num = extract_frame_data(frame_data)
        
        for pos, track_id, class_id in zip(positions, ids, class_ids):
            # Initialize trajectory if new track
            if track_id not in trajectories:
                trajectories[track_id] = []
                # Generate consistent color for this track
                color_rgb = utils.id_to_rgb_color(track_id)
                colors[track_id] = color_rgb
                classes[track_id] = class_id
            
            # Add position to trajectory
            trajectories[track_id].append(pos)
            # Update class (in case it changes)
            classes[track_id] = class_id
    
    return trajectories, colors, classes

def plot_comparison(data1, data2, label1, label2, show_trajectories=True):
    """
    Plot two datasets together in the same 3D space with ID-based colors and class-based markers.
    
    Args:
        data1, data2 (list): List of frame data
        label1, label2 (str): Labels for the datasets
        show_trajectories (bool): Whether to show trajectory lines
    """
    utils = Utils()
    
    # Set up the figure with single 3D plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    
    # Configure axis
    setup_3d_axis(ax)
    
    # Try to load camera positions
    try:
        camera_configs = [
            "config_camera/0.json",
            "config_camera/1.json", 
            "config_camera/2.json",
            "config_camera/3.json",
        ]
        if all(os.path.exists(config) for config in camera_configs):
            matrices = FundamentalMatrices()
            P_all = matrices.projection_matrices_all(camera_configs)
            # visualize_camera_positions(ax, P_all)
    except Exception as e:
        logging.warning(f"Could not load camera positions: {str(e)}")
    
    # Process both datasets and collect all trajectories
    all_trajectories = {}
    all_colors = {}
    all_classes = {}
    all_datasets = {}  # Track which dataset each track belongs to
    
    # Process first dataset
    trajectories1, colors1, classes1 = process_dataset_for_plotting(data1, utils)
    for track_id in trajectories1:
        all_trajectories[track_id] = trajectories1[track_id]
        all_colors[track_id] = colors1[track_id]
        all_classes[track_id] = classes1[track_id]
        all_datasets[track_id] = label1
    
    # Process second dataset - keep original IDs but use colors based on original ID
    trajectories2, colors2, classes2 = process_dataset_for_plotting(data2, utils)
    id_offset = 1000  # Offset for second dataset to avoid conflicts in plotting
    for track_id in trajectories2:
        new_id = track_id + id_offset
        all_trajectories[new_id] = trajectories2[track_id]
        # Use color based on ORIGINAL ID (before offset) for consistency
        original_color = utils.id_to_rgb_color(track_id)
        all_colors[new_id] = original_color
        all_classes[new_id] = classes2[track_id]
        all_datasets[new_id] = label2
    
    # Define class-specific markers
    class_markers = {
        0: 'o',    # pessoa
        3: '^',    # robo
        56: 's'    # cadeira
    }
    
    # Define class names for legend (same as single dataset plot)
    class_names = {
        0: 'Person',
        3: 'Robot', 
        56: 'Chair'
    }
    
    legend_elements = []
    
    # Plot all trajectories
    for track_id, trajectory in all_trajectories.items():
        if not trajectory:
            continue
            
        trajectory = np.array(trajectory)
        color_rgb = all_colors[track_id]
        color_rgb_norm = utils.normalize_rgb_color(color_rgb)
        class_id = all_classes[track_id]
        dataset_label = all_datasets[track_id]
        
        # Get marker for this class
        marker = class_markers.get(int(class_id), 'o')  # Default to 'o' if class not found
        
        # Plot trajectory line
        if show_trajectories and len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   c=color_rgb_norm, alpha=0.8, linewidth=2.5, zorder=2)  # Lower zorder
        
        # Plot final position
        final_pos = trajectory[-1]
        ax.scatter(final_pos[0], final_pos[1], final_pos[2],
                  c=[color_rgb_norm], s=150, marker=marker,
                  edgecolors='black', linewidths=1.5, alpha=0.8, zorder=10)  # Keep markers on top
        
        # Plot vertical line to ground
        ax.plot([final_pos[0], final_pos[0]], [final_pos[1], final_pos[1]], 
               [0, final_pos[2]], '--', color=color_rgb_norm, alpha=0.5, 
               linewidth=1, zorder=4)
        
        # Determine original ID for labeling
        original_id = track_id if dataset_label == label1 else track_id - id_offset
        
        # Create legend entry (same format as single dataset)
        class_name = class_names.get(int(class_id), f"Class {class_id}")
        legend_label = f"ID {original_id}: {class_name}"
        
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                    markerfacecolor=color_rgb_norm, markersize=10,
                                    markeredgecolor='black', markeredgewidth=1.5,
                                    label=legend_label))
    
    # Add legend (same styling as single dataset)
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements, loc='upper left', 
            bbox_to_anchor=(0.01, 0.99), fontsize=11,
            frameon=True, fancybox=False, shadow=False
        )
        # legend.get_frame().set_facecolor('white')
        # legend.get_frame().set_alpha(0.9)
    
    # Remove title to match single dataset behavior
    # Set title
    
    # Adjust layout
    plt.tight_layout()
    return fig

def plot_single_dataset(data, label, show_trajectories=True):
    """
    Plot a single dataset using the exact same style as comparison plots.
    
    Args:
        data (list): List of frame data
        label (str): Label for the dataset
        show_trajectories (bool): Whether to show trajectory lines
    """
    utils = Utils()
    
    # Set up the figure exactly like comparison plots
    fig = plt.figure(figsize=(7, 5))  # Match plot_comparison
    ax_3d = fig.add_subplot(111, projection="3d")
    
    # Use the same setup_3d_axis function as comparison plots
    setup_3d_axis(ax_3d)
    
    # Try to load camera positions exactly like comparison plots
    try:
        camera_configs = [
            "config_camera/0.json",
            "config_camera/1.json", 
            "config_camera/2.json",
            "config_camera/3.json",
        ]
        if all(os.path.exists(config) for config in camera_configs):
            matrices = FundamentalMatrices()
            P_all = matrices.projection_matrices_all(camera_configs)
            # visualize_camera_positions(ax_3d, P_all)
    except Exception as e:
        logging.warning(f"Could not load camera positions: {str(e)}")
    
    # Plot all data at once with same styling as comparison plots
    trajectories = {}
    track_colors = {}
    track_classes = {}  # Store class info for each track
    
    # Collect all trajectories and class information
    for frame_data in data:
        positions, ids, class_ids, frame_num = extract_frame_data(frame_data)
        for pos, track_id, class_id in zip(positions, ids, class_ids):
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append(pos)
            
            if track_id not in track_colors:
                color_rgb = utils.id_to_rgb_color(track_id)
                track_colors[track_id] = color_rgb
            
            # Store class info for this track
            track_classes[track_id] = class_id
    
    # Define class-specific markers (same as comparison plot)
    class_markers = {
        0: 'o',    # pessoa
        3: '^',    # robo
        56: 's'    # cadeira
    }
    
    # Define class names for legend (same as comparison plot)
    class_names = {
        0: 'Person',
        3: 'Robot', 
        56: 'Chair'
    }
    
    # Collect legend information
    legend_elements = []
    
    # Plot each object with same styling as comparison plots
    for track_id, trajectory in trajectories.items():
        if not trajectory:
            continue
            
        trajectory = np.array(trajectory)
        color_rgb = track_colors[track_id]
        color_rgb_norm = utils.normalize_rgb_color(color_rgb)
        
        # Get class info and determine marker
        class_id = track_classes.get(track_id, 0)
        
        # Use class-specific markers (same as comparison plot)
        marker = class_markers.get(int(class_id), 'o')  # Default to 'o' if class not found
        
        # Plot trajectory
        if show_trajectories and len(trajectory) > 1:
            ax_3d.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                      c=color_rgb_norm, alpha=0.8, linewidth=2.5, zorder=2)  # Lower zorder
        
        # Plot final position with class-specific marker
        final_pos = trajectory[-1]
        ax_3d.scatter(final_pos[0], final_pos[1], final_pos[2],
                     c=[color_rgb_norm], s=150, marker=marker,
                     edgecolors='black', linewidths=1.5, alpha=0.8, zorder=10)  # Keep markers on top
        
        # Add vertical line
        ax_3d.plot([final_pos[0], final_pos[0]], [final_pos[1], final_pos[1]], 
                  [0, final_pos[2]], '--', color=color_rgb_norm, alpha=0.5, 
                  linewidth=1, zorder=4)
        
        # Create legend entry with class info (same as comparison plot)
        class_name = class_names.get(int(class_id), f"Class {class_id}")
        legend_label = f"ID {track_id}: {class_name}"
        
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                    markerfacecolor=color_rgb_norm, markersize=10,
                                    markeredgecolor='black', markeredgewidth=1.5,
                                    label=legend_label))
    
    # Add legend with same styling as comparison plots
    if legend_elements:
        legend = ax_3d.legend(
            handles=legend_elements, loc='upper left', 
            bbox_to_anchor=(0.01, 0.99), fontsize=11,
            frameon=True, fancybox=False, shadow=False
        )
        # legend.get_frame().set_facecolor('white')
        # legend.get_frame().set_alpha(0.9)
        # legend.get_frame().set_edgecolor('lightgray')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to handle command line arguments and plotting."""
    parser = argparse.ArgumentParser(description="Plot tracking data from JSON files")
    parser.add_argument("json_files", nargs='+', help="One or more JSON files to plot")
    parser.add_argument("--labels", nargs='+', help="Labels for the datasets")
    parser.add_argument("--no-trajectories", action="store_true", help="Don't show trajectory lines")
    parser.add_argument("--output", type=str, help="Save plot to file instead of displaying")
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.json_files) < 1:
        print("Error: Please provide at least 1 JSON file")
        return
    
    # Load data
    datasets = []
    for json_file in args.json_files:
        if not os.path.exists(json_file):
            print(f"Error: File {json_file} not found")
            return
        data = load_json_data(json_file)
        if not data:
            print(f"Error: No data loaded from {json_file}")
            return
        datasets.append(data)
    
    # Set labels
    if args.labels:
        if len(args.labels) != len(args.json_files):
            print("Error: Number of labels must match number of JSON files")
            return
        labels = args.labels
    else:
        labels = [os.path.basename(f).replace('.json', '') for f in args.json_files]
    
    show_trajectories = not args.no_trajectories
    
    # Plot data
    if len(datasets) == 2:
        # Comparison mode for exactly 2 datasets
        fig = plot_comparison(datasets[0], datasets[1], labels[0], labels[1], show_trajectories)
    elif len(datasets) == 1:
        # Single dataset plot
        fig = plot_single_dataset(datasets[0], labels[0], show_trajectories)
    else:
        # Error for other cases
        print("Error: This script supports plotting 1 or 2 JSON files at a time.")
        return
    
    # Handle output
    if args.output:
        # Remove extension if provided to create base filename
        base_filename = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        
        # Save both PNG and PDF at 400 DPI with extra padding for 3D plots
        plt.savefig(f"{base_filename}.pdf", dpi=400, bbox_inches='tight', 
                   format='pdf')
        plt.savefig(f"{base_filename}.png", dpi=400, bbox_inches='tight', 
                   format='png')
        print(f"Plot saved to {base_filename}.pdf and {base_filename}.png")
        plt.close()
    else:
        plt.show()
        input("Press Enter to exit...")
        plt.close()

if __name__ == "__main__":
    main()
