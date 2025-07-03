"""
Epipolar Geometry Utility Functions Module

This module provides utility functions for epipolar geometry calculations,
including computation of epipolar lines and distance measurements between
points and lines for multi-view geometry applications.
"""

import cv2
import numpy as np

def calculate_lines(F, points):
    """
    Calculate epipolar lines for given points using the fundamental matrix.
    
    Args:
        F (np.ndarray): 3x3 Fundamental matrix
        points (np.ndarray): Array of points for which to compute epipolar lines
    
    Returns:
        np.ndarray: Array of epipolar lines in the format [a, b, c] where ax + by + c = 0
    """
    lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, F)
    return lines.reshape(-1, 3)

def dist_p_l(line, centroid):
    """
    Calculate the distance between a point and a line.
    
    Args:
        line (tuple): Line coefficients [a, b, c] of the line equation ax + by + c = 0
        centroid (tuple): Point coordinates (x, y) to calculate distance from
    
    Returns:
        float: Perpendicular distance from the point to the line
    """
    a, b, c = line
    x, y = centroid
    return abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)

def cross_distance(bbox1, bbox2, line1, line2):
    """
    Calculate the normalized cross-distance between two bounding boxes and their epipolar lines.
    
    This function computes a normalized measure of how well two detections satisfy
    epipolar constraints by calculating their distances to respective epipolar lines.
    
    Args:
        bbox1 (tuple): First bounding box coordinates (x1, y1, x2, y2)
        bbox2 (tuple): Second bounding box coordinates (x1, y1, x2, y2)
        line1 (tuple): First epipolar line coefficients [a, b, c]
        line2 (tuple): Second epipolar line coefficients [a, b, c]
    
    Returns:
        float: Normalized sum of distances from each bbox centroid to its corresponding epipolar line
    """
    # Extract bbox dimensions for normalization
    x1, y1, x2, y2 = bbox2
    width_1 = x2 - x1
    height_1 = y2 - y1

    x3, y3, x4, y4 = bbox1
    width_2 = x4 - x3
    height_2 = y4 - y3

    # Calculate normalized distances to epipolar lines
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    d_cross = dist_p_l([a1, b1, c1], [(x1 + x2) / 2, (y1 + y2) / 2]) / abs(
        width_1 + height_1
    ) + dist_p_l([a2, b2, c2], [(x3 + x4) / 2, (y3 + y4) / 2]) / abs(
        width_2 + height_2
    )
    return d_cross


