"""
Visualization Utilities Module

This module provides functions for 3D visualization in the tracking system,
including plotting camera positions, ground planes, and bounding boxes.
"""

import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib import patheffects
import logging
from config import CLASS_NAMES


def plot_camera_axes(R, T, ax, scale=0.5, label=''):
    """
    Plots the camera reference frame axes.
    - R: rotation matrix (3x3), where each column represents an axis (x, y, z)
    - T: translation vector (3x1), camera position in the environment
    - ax: Axes3D object where plotting will occur
    - scale: length of the arrows for visualization
    - label: label to identify the camera
    """
    # Convert T to a 1D array
    origin = T.flatten()
    
    # Camera axis vectors in global coordinates
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Plot axes using ax.quiver
    ax.quiver(origin[0], origin[1], origin[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', length=scale, normalize=True)
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', length=scale, normalize=True)
    
    # Add a label at the camera position
    ax.text(origin[0], origin[1], origin[2], label, fontsize=10)

def visualize_camera_positions(ax, P_all):
    """
    Visualize camera positions in the 3D space.
    
    Args:
        ax (matplotlib.axes.Axes): 3D axis object
        P_all (dict): Dictionary of projection matrices for each camera
    """
    # Load camera parameters for visualization
    camera_configs = [
        "config_camera/0.json",
        "config_camera/1.json",
        "config_camera/2.json",
        "config_camera/3.json",
    ]
    
    # Function to extract camera parameters from config file
    def camera_parameters(file):
        camera_data = json.load(open(file))
        # Get extrinsic parameters (R, T)
        extrinsic_matrix = np.array(camera_data['extrinsic'][0]['tf']['doubles']).reshape(4, 4)
        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3].reshape(3, 1)
        
        # For visualization, we need the inverse transformation
        # For a rotation matrix, its inverse is its transpose
        R_inv = R.transpose()
        # For translation, we need to apply -T rotated by inverse R
        T_inv = -R_inv @ T
        
        return R_inv, T_inv
    
    # Plot each camera
    for cam_idx, config_file in enumerate(camera_configs):
        if os.path.exists(config_file):
            try:
                R_inv, T_inv = camera_parameters(config_file)
                plot_camera_axes(R_inv, T_inv, ax, scale=0.5, label=f'C{cam_idx}')
            except Exception as e:
                logging.warning(f"Error loading camera {cam_idx}: {str(e)}")

def draw_bbox(frame, bbox, class_name, object_id, color, reference_point="bottom_center"):
    """
    Draw a stylish bounding box with enhanced visual appearance.
    
    Args:
        frame (np.ndarray): Frame to draw on
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]
        class_name (str): Class name of the detected object
        object_id (int): ID of the tracked object
        color (tuple): RGB color tuple for the bounding box
        reference_point (str): Reference point used for triangulation
                             ("bottom_center", "center", "top_center", "feet")
    
    Returns:
        np.ndarray: Frame with the bounding box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Use larger font for better visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Increased from 0.6
    font_thickness = 2
    
    # Create ID and class text
    id_text = f"ID: {int(object_id)}"
    class_text = f"{class_name}"
    
    # Calculate text sizes
    id_size = cv2.getTextSize(id_text, font, font_scale, font_thickness)[0]
    class_size = cv2.getTextSize(class_text, font, font_scale, font_thickness)[0]
    
    # Box parameters
    box_thickness = 2
    
    # Draw the main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    
    # Draw corner accents for more style
    corner_length = min((x2-x1)//4, (y2-y1)//4, 30)  # Length of corner lines, max 30px
    # Top-left
    cv2.line(frame, (x1, y1), (x1+corner_length, y1), (255, 255, 255), box_thickness+1)
    cv2.line(frame, (x1, y1), (x1, y1+corner_length), (255, 255, 255), box_thickness+1)
    # Top-right
    cv2.line(frame, (x2, y1), (x2-corner_length, y1), (255, 255, 255), box_thickness+1)
    cv2.line(frame, (x2, y1), (x2, y1+corner_length), (255, 255, 255), box_thickness+1)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1+corner_length, y2), (255, 255, 255), box_thickness+1)
    cv2.line(frame, (x1, y2), (x1, y2-corner_length), (255, 255, 255), box_thickness+1)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2-corner_length, y2), (255, 255, 255), box_thickness+1)
    cv2.line(frame, (x2, y2), (x2, y2-corner_length), (255, 255, 255), box_thickness+1)
    
    # Draw text ABOVE the bounding box for ID
    padding = 5
    bg_height = id_size[1] + 2 * padding
    
    # Calculate text position - centered above the box
    text_x = x1 + (x2 - x1 - id_size[0]) // 2
    text_y = y1 - padding - 5  # Position above the box
    
    # Ensure text stays within frame bounds
    if text_y < id_size[1] + padding:
        text_y = y1 + id_size[1] + padding  # If too close to top, place below top edge
    
    # Draw semi-transparent background for ID text
    cv2.rectangle(
        frame,
        (text_x - padding, text_y - id_size[1] - padding),
        (text_x + id_size[0] + padding, text_y + padding),
        color,
        -1
    )
    
    # Draw ID text with white color for contrast
    cv2.putText(
        frame, 
        id_text, 
        (text_x, text_y), 
        font, 
        font_scale, 
        (255, 255, 255), 
        font_thickness, 
        cv2.LINE_AA
    )
    
    # Draw text BELOW the bounding box for class name
    # Calculate text position - centered below the box
    text_x = x1 + (x2 - x1 - class_size[0]) // 2
    text_y = y2 + class_size[1] + padding + 5  # Position below the box
    
    # Draw semi-transparent background for class text
    cv2.rectangle(
        frame,
        (text_x - padding, text_y - class_size[1] - padding),
        (text_x + class_size[0] + padding, text_y + padding),
        color,
        -1
    )
    
    # Draw class text with white color for contrast
    cv2.putText(
        frame, 
        class_text, 
        (text_x, text_y), 
        font, 
        font_scale, 
        (255, 255, 255), 
        font_thickness, 
        cv2.LINE_AA
    )
    

    # Determine reference point coordinates based on the selected reference point type
    if reference_point == "center":
        ref_x, ref_y = (x1 + x2) // 2, (y1 + y2) // 2
    elif reference_point == "top_center":
        ref_x, ref_y = (x1 + x2) // 2, y1
    elif reference_point == "feet":
        ref_x, ref_y = (x1 + x2) // 2, y2 - int(0.05 * (y2 - y1))
    else:  # Default to bottom_center
        ref_x, ref_y = (x1 + x2) // 2, y2
    
    # Draw the reference point as a circle with a cross in the middle
    marker_size = 10
    marker_color = (0, 255, 255)  # Cyan color for visibility
    thickness = 2
    
    # Draw circle
    cv2.circle(frame, (ref_x, ref_y), marker_size, marker_color, thickness)
    
    # Draw cross
    cv2.line(frame, (ref_x - marker_size, ref_y), (ref_x + marker_size, ref_y), marker_color, thickness)
    cv2.line(frame, (ref_x, ref_y - marker_size), (ref_x, ref_y + marker_size), marker_color, thickness)
    
    return frame