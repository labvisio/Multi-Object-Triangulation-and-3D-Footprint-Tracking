"""
Object Detection Data Structure Module

This module defines the ObjectDetection class that encapsulates all relevant information
about a detected object across multiple camera views, including its position, identification,
and appearance characteristics.
"""

class ObjectDetection:
    """
    A class to represent a detected object in a camera view.
    
    This class stores all relevant information about a detected object, including its
    position (bounding box and centroids), identification (camera ID and object ID),
    and classification information.
    
    Attributes:
        cam (int): Camera identifier number
        id (int): Unique identifier for the detected object
        bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
        frame (numpy.ndarray): Reference to the frame where the object was detected
        centroids_list (list): List of centroid points from bbox subdivisions
        centroid (tuple): Main centroid point (x, y) of the bounding box
        name (str): Classification or category name of the detected object
        color (tuple, optional): RGB color assigned for visualization purposes
    """

    def __init__(self, cam, id, bbox, frame, centroids_list, centroid, name):
        """
        Initialize an ObjectDetection instance.
        
        Args:
            cam (int): Camera identifier number
            id (int): Unique identifier for the detected object
            bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max)
            frame (numpy.ndarray): Reference to the frame where object was detected
            centroids_list (list): List of centroid points from bbox subdivisions
            centroid (tuple): Main centroid point (x, y) of the bounding box
            name (str): Classification or category name of the detected object
        """
        self.cam = cam
        self.id = id
        self.bbox = bbox
        self.frame = frame
        self.centroids_list = centroids_list
        self.centroid = centroid
        self.name = name
        self.color = None  # Initially None, can be set later for visualization

    def __str__(self):
        """
        String representation of the ObjectDetection instance.
        
        Returns:
            str: Formatted string with essential object information
        """
        return f"ObjectDetection(cam={self.cam}, id={self.id}, single_centroid={self.centroid})"
