#!/usr/bin/env python3
"""
YOLO Tracking Demo for Four Camera System

This script demonstrates YOLO object tracking on four synchronized camera feeds.
It processes the video feeds, applies YOLO tracking with the same styling as the main
tracking system, and saves annotated output videos for each camera.

Usage:
    python demo_yolo_tracking.py --input_dir videos --output_dir output_videos

Dependencies:
- OpenCV, NumPy, Ultralytics YOLO
- Custom modules from four_view_tracker
"""

import argparse
import logging
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Add the four_view_tracker directory to the path
sys.path.append('source')

from config import YOLO_MODEL, CLASS_NAMES, CONFIDENCE
from ploting_utils import Utils
from visualization_utils import draw_bbox
from bbox_utils import get_centroid
    
# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class YOLOTrackingDemo:
    """
    Simplified YOLO tracking demo for four camera system.
    """
    
    def __init__(self, input_dir, output_dir, yolo_model, confidence, class_list):
        """
        Initialize the YOLO tracking demo.
        
        Args:
            input_dir (str): Directory containing input video files
            output_dir (str): Directory to save output videos
            yolo_model (str): Path to YOLO model file
            confidence (float): Confidence threshold for detections
            class_list (list): List of class IDs to track
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.confidence = confidence
        self.class_list = class_list
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize utils for color management
        self.utils = Utils()
        
        # Find video files (expecting cam0.mp4, cam1.mp4, cam2.mp4, cam3.mp4)
        self.video_files = self._find_video_files()
        
        # Initialize YOLO models for each camera
        self.yolo_models = [YOLO(yolo_model) for _ in range(len(self.video_files))]
        
        # Initialize video captures and writers
        self.video_captures = []
        self.video_writers = []
        self.frame_count = None
        
        self._initialize_videos()
        
    def _find_video_files(self):
        """Find and sort video files in the input directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        # Look for files matching cam*.extension pattern
        for i in range(4):  # Expecting 4 cameras
            found = False
            for ext in video_extensions:
                cam_file = os.path.join(self.input_dir, f"cam{i}{ext}")
                if os.path.exists(cam_file):
                    video_files.append(cam_file)
                    found = True
                    break
            
            if not found:
                logging.warning(f"Video file for camera {i} not found")
        
        if not video_files:
            raise ValueError(f"No video files found in {self.input_dir}")
        
        logging.info(f"Found {len(video_files)} video files: {[os.path.basename(f) for f in video_files]}")
        return video_files
    
    def _initialize_videos(self):
        """Initialize video captures and writers."""
        for i, video_file in enumerate(self.video_files):
            # Initialize video capture
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_file}")
            
            self.video_captures.append(cap)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.frame_count is None:
                self.frame_count = frame_count
            else:
                self.frame_count = min(self.frame_count, frame_count)
            
            # Initialize video writer
            output_filename = os.path.join(self.output_dir, f"cam{i}_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create output video: {output_filename}")
            
            self.video_writers.append(writer)
            
            logging.info(f"Camera {i}: {width}x{height} @ {fps}fps, {frame_count} frames")
    
    def process_videos(self):
        """Process all video feeds and apply YOLO tracking."""
        logging.info(f"Processing {self.frame_count} frames...")
        
        for frame_idx in range(self.frame_count):
            frames = []
            
            # Read frames from all cameras
            for cap in self.video_captures:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Failed to read frame {frame_idx}")
                    break
                frames.append(frame)
            
            if len(frames) != len(self.video_captures):
                logging.warning(f"Incomplete frame set at frame {frame_idx}, stopping")
                break
            
            # Process each camera feed
            for cam_idx, (frame, yolo_model, writer) in enumerate(zip(frames, self.yolo_models, self.video_writers)):
                annotated_frame = self._process_frame(frame, yolo_model, cam_idx)
                writer.write(annotated_frame)
            
            # Log progress
            if frame_idx % 50 == 0:
                progress = (frame_idx / self.frame_count) * 100
                logging.info(f"Progress: {frame_idx}/{self.frame_count} ({progress:.1f}%)")
        
        logging.info("Processing completed!")
    
    def _process_frame(self, frame, yolo_model, cam_idx):
        """
        Process a single frame with YOLO tracking and annotation.
        
        Args:
            frame (np.ndarray): Input frame
            yolo_model (YOLO): YOLO model instance
            cam_idx (int): Camera index
        
        Returns:
            np.ndarray: Annotated frame
        """
        # Run YOLO tracking
        results = yolo_model.track(
            frame,
            persist=True,
            classes=self.class_list,
            conf=self.confidence,
            verbose=False,
            show=False,
            tracker="four_view_tracker/yolo_conf.yaml"
        )
        
        result = results[0]
        annotated_frame = frame.copy()
        
        # Process detections if any exist
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
            bboxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            # Draw bounding boxes with consistent styling
            for track_id, bbox, class_id in zip(ids, bboxes, class_ids):
                # Get consistent color for this track ID
                color_rgb = self.utils.id_to_rgb_color(int(track_id))
                
                # Get class name
                class_name = CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}")
                
                # Draw the styled bounding box (same as main_plot.py)
                annotated_frame = draw_bbox(
                    annotated_frame,
                    bbox,
                    class_name,
                    int(track_id),
                    color_rgb,
                    reference_point="bottom_center"
                )
        
        # Add camera label
        self._add_camera_label(annotated_frame, cam_idx)
        
        return annotated_frame
    
    def _add_camera_label(self, frame, cam_idx):
        """Add camera label to the frame."""
        # Add semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add camera label text
        cv2.putText(
            frame, 
            f"Camera {cam_idx}", 
            (20, 45), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
    
    def cleanup(self):
        """Release all video captures and writers."""
        for cap in self.video_captures:
            cap.release()
        
        for writer in self.video_writers:
            writer.release()
        
        logging.info("Cleanup completed")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="YOLO Tracking Demo for Four Camera System")
    parser.add_argument("--input_dir", type=str, default="videos/good_videos/two_men_lemniscata", 
                       help="Directory containing input video files (default: /)")
    parser.add_argument("--output_dir", type=str, default="output_videos", 
                       help="Directory to save output videos (default: output_videos)")
    parser.add_argument("--yolo_model", type=str, default=YOLO_MODEL, 
                       help="Path to YOLO model file")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE, 
                       help="Confidence threshold for YOLO detection")
    parser.add_argument("--class_list", type=int, nargs='+', default=[0], 
                       help="List of classes to track (e.g., 0 1 2) (default: 0 for person)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory not found: {args.input_dir}")
    
    logging.info("=" * 60)
    logging.info("YOLO Tracking Demo for Four Camera System")
    logging.info("=" * 60)
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"YOLO model: {args.yolo_model}")
    logging.info(f"Confidence threshold: {args.confidence}")
    logging.info(f"Classes to track: {args.class_list}")
    
    try:
        # Initialize demo
        demo = YOLOTrackingDemo(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            yolo_model=args.yolo_model,
            confidence=args.confidence,
            class_list=args.class_list
        )
        
        # Process videos
        demo.process_videos()
        
        # Cleanup
        demo.cleanup()
        
        logging.info("=" * 60)
        logging.info("Demo completed successfully!")
        logging.info(f"Output videos saved in: {args.output_dir}")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
