"""
Detection Matching Module

This module provides functionality for matching object detections across multiple camera views
using epipolar geometry. It manages fundamental matrices and projection matrices for all
camera pairs and implements matching logic based on epipolar constraints.
"""

import logging
import numpy as np
import cv2
from ploting_utils import Utils
from load_fundamental_matrices import FundamentalMatrices
from epipolar_utils import calculate_lines, cross_distance


class Matcher:
    """
    A class to match object detections across multiple camera views.
    
    This class handles the matching of detections between different camera views using
    epipolar geometry constraints. It maintains the fundamental and projection matrices
    for all camera pairs and provides methods for visualization and matching.
    
    Attributes:
        F_all (dict): Dictionary of fundamental matrices for each camera pair
        P_all (dict): Dictionary of projection matrices for each camera
        lines_1_2 (np.ndarray): Epipolar lines from camera 1 to camera 2
        lines_2_1 (np.ndarray): Epipolar lines from camera 2 to camera 1
        colors (dict): Color mapping for visualization
        previous_matches (dict): Cache of previously successful matches
    """

    def __init__(self, distance_threshold, drift_threshold):
        """
        Initialize the Matcher with camera configuration files.
        
        Loads fundamental and projection matrices from camera configuration files.
        Currently supports a fixed 4-camera setup.
        """
        logging.info("Initializing Matcher")
        
        # Load camera configurations
        camera_configs = [
            "config_camera/0.json",
            "config_camera/1.json",
            "config_camera/2.json",
            "config_camera/3.json",
        ]
        
        # Initialize fundamental and projection matrices
        matrices = FundamentalMatrices()
        self.F_all = matrices.fundamental_matrices_all(camera_configs)
        self.P_all = matrices.projection_matrices_all(camera_configs)

        # Initialize storage for epipolar lines
        self.lines_1_2 = None
        self.lines_2_1 = None
        self.colors = {}
        
        # For tracking match consistency across frames
        self.previous_matches = {}
        self.match_consistency_score = 0.1  # Weight for considering previous matches
        self.distance_threshold = distance_threshold
        self.drift_threshold = drift_threshold
        

    def plot_lines(self, img1, img2):
        """
        Plots epipolar lines on image pairs for visualization.
        
        Args:
            img1 (np.ndarray): First camera image
            img2 (np.ndarray): Second camera image
            
        Returns:
            tuple: Pair of images with epipolar lines drawn
        """
        # Draw lines on second image
        for r in self.lines_1_2:
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [img2.shape[1], -(r[2] + r[0] * img2.shape[1]) / r[1]])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Draw lines on first image
        for r2 in self.lines_2_1:
            x0_2, y0_2 = map(int, [0, -r2[2] / r2[1]])
            x1_2, y1_2 = map(int, [img1.shape[1], -(r2[2] + r2[0] * img1.shape[1]) / r2[1]])
            img1 = cv2.line(img1, (x0_2, y0_2), (x1_2, y1_2), (0, 255, 0), 2)

        return img1, img2

    def get_match_key(self, cam1, id1, cam2, id2):
        """
        Create a unique key for a camera-detection pair match.
        
        Args:
            cam1 (int): First camera ID
            id1 (float/int): First detection ID
            cam2 (int): Second camera ID
            id2 (float/int): Second detection ID
            
        Returns:
            str: Unique match key
        """
        # Ensure the camera order is consistent (lower cam ID first)
        if cam1 > cam2:
            return f"{cam2}_{id2}_{cam1}_{id1}"
        return f"{cam1}_{id1}_{cam2}_{id2}"
        
    def match_detections(self, detections, cams):
        """
        Match detections between two camera views using epipolar geometry.
        
        This method implements a sophisticated matching algorithm that:
        1. Computes epipolar lines between camera pairs
        2. Calculates cross-distances between detections
        3. Filters matches based on distance threshold and uniqueness
        4. Handles ambiguous matches using a drift threshold
        5. Uses temporal consistency from previous matches
        
        Args:
            detections (list): List of ObjectDetection instances
            cams (list): List of two camera indices to match between
            
        Returns:
            list: List of matched detection pairs that satisfy the epipolar constraints
        """
        # Initialize epipolar lines
        self.lines_1_2 = []
        self.lines_2_1 = []

        # Get fundamental matrices for the camera pair
        try:
            F_1_2 = self.F_all[cams[0]][cams[1]]
            F_2_1 = self.F_all[cams[1]][cams[0]]
        except KeyError:
            logging.warning(f"No fundamental matrix for camera pair {cams[0]},{cams[1]}")
            return []

        # Filter detections by camera
        detections_cam_1 = [det for det in detections if det.cam == cams[0]]
        detections_cam_2 = [det for det in detections if det.cam == cams[1]]

        if len(detections_cam_1) < 1 or len(detections_cam_2) < 1:
            return []

        # Extract centroids and calculate epipolar lines
        centroids_cam_1 = np.array([det.centroid for det in detections_cam_1])
        centroids_cam_2 = np.array([det.centroid for det in detections_cam_2])
        
        self.lines_1_2 = calculate_lines(F_1_2, centroids_cam_1)
        self.lines_2_1 = calculate_lines(F_2_1, centroids_cam_2)

        # Find potential matches based on epipolar geometry
        maybe_matches = []
        match_key = f"{cams[0]}_{cams[1]}"
        
        for i, det_cam_1 in enumerate(detections_cam_1):
            for j, det_cam_2 in enumerate(detections_cam_2):
                # Only consider same-class detections
                if det_cam_1.name == det_cam_2.name:
                    # Calculate cross distance
                    d_cross = cross_distance(
                        det_cam_1.bbox, det_cam_2.bbox, self.lines_1_2[i], self.lines_2_1[j]
                    )
                    
                    # Check for temporal consistency (was there a match between these objects before?)
                    consistency_bonus = 0
                    match_identifier = self.get_match_key(det_cam_1.cam, det_cam_1.id, 
                                                         det_cam_2.cam, det_cam_2.id)
                    
                    if match_identifier in self.previous_matches:
                        consistency_bonus = self.match_consistency_score
                    
                    # Adjusted score considers both geometry and temporal consistency
                    adjusted_score = d_cross - consistency_bonus
                    
                    # Store with both original distance and adjusted score
                    maybe_matches.append([det_cam_1, det_cam_2, d_cross, adjusted_score])

        # Sort matches by adjusted score and filter based on threshold
        sorted_maybe_matches = sorted(maybe_matches, key=lambda x: x[3])
        filtered_list = []

        # First pass: filter based on distance threshold
        for maybe_match in sorted_maybe_matches:
            if maybe_match[2] < self.distance_threshold:  # Slightly more permissive
                # Check if any existing match conflicts with this one
                conflicts = False
                for f in filtered_list:
                    if maybe_match[0].id == f[0].id or maybe_match[1].id == f[1].id:
                        conflicts = True
                        # If this match is better than the existing one, replace it
                        if maybe_match[3] < f[3]:
                            filtered_list.remove(f)
                            filtered_list.append(maybe_match)
                        break
                
                # If no conflicts, add this match
                if not conflicts:
                    filtered_list.append(maybe_match)

        # Second pass: handle ambiguous matches
        final_list = filtered_list.copy()

        for i, filtered_match in enumerate(filtered_list):
            for maybe_match in sorted_maybe_matches:
                # Check for conflicting matches (same object in different pairs)
                if ((maybe_match[0].id != filtered_match[0].id and 
                     maybe_match[1].id == filtered_match[1].id) or
                    (maybe_match[0].id == filtered_match[0].id and 
                     maybe_match[1].id != filtered_match[1].id)):
                    
                    # If the scores are too close, neither match is reliable
                    if abs(maybe_match[2] - filtered_match[2]) < self.drift_threshold:
                        if filtered_match in final_list:
                            final_list.remove(filtered_match)
                
        # Update the previous matches cache with successful matches
        for match in final_list:
            det_cam_1, det_cam_2, d_cross, _ = match
            match_identifier = self.get_match_key(det_cam_1.cam, det_cam_1.id, 
                                              det_cam_2.cam, det_cam_2.id)
            self.previous_matches[match_identifier] = True
            
            # Limit size of previous_matches cache to prevent memory issues
            if len(self.previous_matches) > 1000:
                self.previous_matches.clear()

        # Log successful matches
        for match in final_list:
            det_cam_1, det_cam_2, d_cross, _ = match
            logging.info(f"\nMatch found between cameras {cams[0]} and {cams[1]}:")
            logging.info(f"Camera {cams[0]} Detection - ID: {det_cam_1.id}, "
                        f"Class: {int(det_cam_1.name)}, "
                        f"Centroid: ({det_cam_1.centroid[0]:.2f}, {det_cam_1.centroid[1]:.2f})")
            logging.info(f"Camera {cams[1]} Detection - ID: {det_cam_2.id}, "
                        f"Class: {int(det_cam_2.name)}, "
                        f"Centroid: ({det_cam_2.centroid[0]:.2f}, {det_cam_2.centroid[1]:.2f})")
            logging.info(f"Cross Distance: {d_cross:.4f}")

        # Return final matches but drop the adjusted score from the output
        return [[m[0], m[1], m[2]] for m in final_list]
