"""
Bounding Box Utility Functions Module

This module provides utility functions for processing and analyzing bounding boxes,
including subdivision and centroid calculations for object tracking and analysis.
"""

import numpy as np


def divide_bbox(bbox):
    """
    Divides a bounding box into 6 vertical subdivisions and returns their centroids.
    
    This function is useful for creating reference points along the vertical axis
    of detected objects, which can be used for pose estimation or height analysis.
    
    Args:
        bbox (tuple): Bounding box coordinates in format (x_min, y_min, x_max, y_max)
    
    Returns:
        list: List of 6 centroid points, where each point is [x, y] representing
              the center point of each vertical subdivision
    """
    x_min, y_min, x_max, y_max = bbox
    subdivision_height = (y_max - y_min) / 5  # Divide height into 5 equal parts for 6 points
    sub_centroids = []

    # Calculate centroids for each subdivision
    for i in range(6):
        sub_centroids.append([(x_min + x_max) / 2, y_min + subdivision_height * i])

    return sub_centroids


def get_centroid(bbox):
    """
    Calculates the centroid(s) of one or multiple bounding boxes.
    
    Args:
        bbox (np.ndarray or list): Single bounding box or list of bounding boxes
                                  in format (x_min, y_min, x_max, y_max)
    
    Returns:
        np.ndarray: Array of centroids where each centroid is [x, y]
                   representing the center point of each bounding box
    """
    return np.array([[(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2] for bb in bbox])

