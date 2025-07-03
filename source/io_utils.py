"""
I/O Utility Functions Module

This module provides functions for data input/output operations,
including saving 3D coordinate data to JSON files.
"""

import json
from datetime import datetime
import os
import logging
import numpy as np

def save_3d_coordinates(frame, point_3d_list, output_file):
    """
    Save the 3D coordinates with capture name to a JSON file.
    
    Args:
        frame (str): Name of the capture folder
        point_3d_list (list): List of 3D points
        output_file (str): Path to output JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    points_list = [point.tolist() if isinstance(point, np.ndarray) else point for point in point_3d_list]
    
    # Create data entry for this capture
    capture_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "frame": frame,
        "points": points_list
    }
    
    # Load existing data if file exists
    data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Error reading {output_file}, creating new file")
    
    # Add new data and save
    data.append(capture_data)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved 3D coordinates for {frame} to {output_file}")

def save_3d_coordinates_with_ids(frame, point_3d_list, track_ids, output_file, class_ids=None):
    """
    Save the 3D coordinates with their tracking IDs and class information to a JSON file.
    
    Args:
        frame (int): Current frame number
        point_3d_list (list): List of 3D points
        track_ids (list): List of tracking IDs corresponding to each point
        output_file (str): Path to output JSON file
        class_ids (list, optional): List of class IDs corresponding to each point
    """
    # Convert numpy arrays to lists for JSON serialization and pair with IDs and classes
    points_with_ids = []
    for i, point in enumerate(point_3d_list):
        track_id = track_ids[i] if i < len(track_ids) else i
        class_id = class_ids[i] if class_ids and i < len(class_ids) else None
        
        if isinstance(point, np.ndarray):
            point_data = {
                "position": point.tolist(),
                "id": int(track_id)
            }
            # Add class information if available
            if class_id is not None:
                point_data["class"] = int(class_id)
        else:
            point_data = {
                "position": point,
                "id": int(track_id)
            }
            # Add class information if available
            if class_id is not None:
                point_data["class"] = int(class_id)
                
        points_with_ids.append(point_data)
    
    # Create data entry for this frame
    capture_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "frame": frame,
        "points": points_with_ids
    }
    
    # Load existing data if file exists
    data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.warning(f"Error reading {output_file}, creating new file")
    
    # Add new data and save
    data.append(capture_data)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved 3D coordinates with tracking IDs and class info for frame {frame} to {output_file}")