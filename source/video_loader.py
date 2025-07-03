"""
Video and Image Loading and Synchronization Module

This module provides functionality for loading and synchronizing multiple video streams
or image sequences simultaneously. It handles capture operations and ensures synchronized
frame retrieval across all cameras.
"""

import cv2
import logging
import os
import glob
import re


class VideoLoader:
    """
    A class to handle multiple synchronized video streams.
    
    This class manages multiple video captures simultaneously, providing synchronized
    access to frames from all cameras. It ensures that frame retrieval is properly
    coordinated across all video sources.
    
    Attributes:
        sources_list (list): List of video source paths
        video_captures (list): List of OpenCV VideoCapture objects
    """

    def __init__(self, sources_list: list):
        """
        Initialize VideoLoader with multiple video sources.
        
        Args:
            sources_list (list): List of paths to video files or camera indices
        
        Raises:
            RuntimeError: If any video source fails to open
        """
        self.sources_list = sources_list
        self.video_captures = []
        
        # Initialize video captures for each source
        for source in self.sources_list:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source}")
            self.video_captures.append(cap)

        logging.info(f"VideoLoader initialized with {len(sources_list)} sources")

    def get_frames(self):
        """
        Retrieve synchronized frames from all video sources.
        
        Returns:
            list: List of frames from all cameras, in the same order as sources_list.
                 Returns None for any failed frame reads.
        """
        frames = []
        for idx, video_capture in enumerate(self.video_captures):
            ret, frame = video_capture.read()
            if not ret:
                logging.warning(f"Failed to read frame from source {self.sources_list[idx]}")
                frames.append(None)
            else:
                frames.append(frame)
        return frames

    def get_number_of_frames(self):
        """
        Get the minimum number of frames across all video sources.
        
        This ensures synchronization by using the shortest video length
        when videos have different durations.
        
        Returns:
            int: Minimum number of frames across all video sources
        """
        num_frames = []
        for idx, video_capture in enumerate(self.video_captures):
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Source {self.sources_list[idx]}: {frame_count} frames")
            num_frames.append(frame_count)
        return min(num_frames)

    def release(self):
        """
        Release all video captures and free resources.
        
        Should be called when done with video processing to properly
        clean up resources.
        """
        for video_capture in self.video_captures:
            video_capture.release()
        logging.info("Released all video captures")


class ImageLoader:
    """
    A class to handle multiple synchronized image sequences from directories.
    
    This class manages loading images from folders organized by capture sets,
    providing synchronized access to frames from all cameras.
    
    Attributes:
        base_dir (str): Base directory containing image captures
        capture_folders (list): List of capture folder paths sorted by timestamp
        current_capture_idx (int): Index of the current capture being processed
        sources_list (list): List of camera indices
    """

    def __init__(self, base_dir, camera_indices=None):
        """
        Initialize ImageLoader with a directory of image captures.
        
        Args:
            base_dir (str): Directory containing capture folders
            camera_indices (list, optional): List of camera indices to look for
                                           Default is [0, 1, 2, 3]
        
        Raises:
            RuntimeError: If base directory doesn't exist or no captures found
        """
        self.base_dir = base_dir
        self.sources_list = camera_indices or [0, 1, 2, 3]
        self.current_capture_idx = 0
        
        if not os.path.exists(base_dir):
            raise RuntimeError(f"Image directory does not exist: {base_dir}")
            
        # Get all capture folders sorted by timestamp (assumes format capture_YYYYMMDD_HHMMSS)
        self.capture_folders = sorted(
            glob.glob(os.path.join(base_dir, "capture_*")),
            key=lambda x: os.path.basename(x).split("_")[1:]
        )
        
        if not self.capture_folders:
            raise RuntimeError(f"No capture folders found in {base_dir}")
            
        logging.info(f"ImageLoader initialized with {len(self.capture_folders)} capture sets")
        logging.info(f"First capture: {os.path.basename(self.capture_folders[0])}")

    def get_frames(self):
        """
        Retrieve synchronized frames from all cameras for the current capture.
        
        Returns:
            list: List of frames from all cameras.
                 Returns None for any missing image files.
        """
        if self.current_capture_idx >= len(self.capture_folders):
            return [None] * len(self.sources_list)
            
        current_folder = self.capture_folders[self.current_capture_idx]
        frames = []
        
        for cam_idx in self.sources_list:
            img_path = os.path.join(current_folder, f"{cam_idx}.jpg")
            if os.path.exists(img_path):
                frame = cv2.imread(img_path)
                frames.append(frame)
            else:
                logging.warning(f"No image for camera {cam_idx} in {current_folder}")
                frames.append(None)
                
        self.current_capture_idx += 1
        return frames

    def get_number_of_frames(self):
        """
        Get the number of capture sets available.
        
        Returns:
            int: Number of capture folders
        """
        return len(self.capture_folders)

    def release(self):
        """
        Release any resources (no-op for images, included for API compatibility).
        """
        logging.info("ImageLoader released")