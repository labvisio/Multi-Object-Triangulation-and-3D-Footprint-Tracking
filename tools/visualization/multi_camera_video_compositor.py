"""Video Mosaic Creator for YOLO Tracking Output.

This script processes tracked videos from a multi-camera setup (specifically 
four cameras) and combines them into a single mosaic video. The output styling,
including borders, layout, and frame counters, is designed to precisely replicate 
the visual output of a main tracking system.

Usage:
    python create_video_mosaic.py --input_dir input_videos --output_video mosaic_output.mp4

Dependencies:
- OpenCV-Python
- NumPy
"""

import argparse
import logging
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Configure logging for clear and informative output.
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class VideoMosaicCreator:
    """Creates a mosaic video from multiple camera outputs.

    This class handles finding input video files, processing them frame by frame,
    and compiling them into a single 2x2 mosaic video file with specific
    styling.

    Attributes:
        input_dir (str): Directory containing source video files.
        output_video (str): Path to save the final mosaic video.
        target_fps (int): Desired frames per second for the output video.
    """
    
    def __init__(self, input_dir: str, output_video: str, target_fps: int = 10):
        """Initializes the VideoMosaicCreator.

        Args:
            input_dir (str): Directory containing input tracked videos.
            output_video (str): Path for the output mosaic video.
            target_fps (int): Frame rate for the output video.
        """
        self.input_dir = input_dir
        self.output_video = output_video
        self.target_fps = target_fps
        
        # --- Video and Frame Properties ---
        self.video_files: List[str] = self._find_video_files()
        self.video_captures: List[cv2.VideoCapture] = []
        self.frame_count: Optional[int] = None
        
        # --- Mosaic Layout Constants ---
        # These dimensions are hardcoded to match the specific styling of the
        # primary tracking system's output.
        self.frame_width: int = 540
        self.frame_height: int = 360
        self.border_width: int = 5
        
        # Calculate final mosaic dimensions: a 2x2 grid with borders.
        # Width: (Frame Width * 2) + Vertical Border
        # Height: (Frame Height * 2) + Horizontal Border
        self.mosaic_width: int = self.frame_width * 2 + self.border_width
        self.mosaic_height: int = self.frame_height * 2 + self.border_width
        
        self.video_writer: Optional[cv2.VideoWriter] = None
        
        # Initialize video capture objects and determine the minimum frame count
        # to ensure all videos are processed to the same length.
        self._initialize_videos()
        
    def _find_video_files(self) -> List[str]:
        """Finds and sorts the four tracked video files."""
        video_files = []
        
        # Expects 4 cameras with a specific naming convention: 'cam{i}_tracked.mp4'.
        for i in range(4):
            tracked_file = os.path.join(self.input_dir, f"cam{i}_tracked.mp4")
            if not os.path.exists(tracked_file):
                raise FileNotFoundError(f"Required video file not found: {tracked_file}")
            video_files.append(tracked_file)
        
        logging.info(f"Found {len(video_files)} tracked video files.")
        return video_files
    
    def _initialize_videos(self) -> None:
        """Initializes video captures and determines the minimum frame count."""
        min_frame_count = float('inf')
        
        for i, video_file in enumerate(self.video_files):
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_file}")
            
            self.video_captures.append(cap)
            
            # Extract video properties for logging and validation.
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            min_frame_count = min(min_frame_count, frame_count)
            
            logging.info(
                f"Video {i} ('{os.path.basename(video_file)}'): "
                f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"@ {int(cap.get(cv2.CAP_PROP_FPS))} FPS, {frame_count} frames"
            )
        
        # Set the master frame count to the smallest count among all videos
        # to prevent errors when one video ends before others.
        self.frame_count = min_frame_count
    
    def _create_mosaic_frame(self, frames: List[np.ndarray], frame_number: int) -> np.ndarray:
        """Creates a single mosaic frame from four individual camera frames.

        Args:
            frames (List[np.ndarray]): A list of 4 frames (as NumPy arrays).
            frame_number (int): The current frame number to display.
            
        Returns:
            np.ndarray: The final composed mosaic frame.
        """
        # Resize all frames to the standard dimension.
        processed_frames = [
            cv2.resize(frame, (self.frame_width, self.frame_height)) for frame in frames
        ]

        # Create white borders for the mosaic grid.
        v_border = np.full((self.frame_height, self.border_width, 3), 255, dtype=np.uint8)
        h_border = np.full((self.border_width, self.mosaic_width, 3), 255, dtype=np.uint8)
        
        # Assemble the 2x2 grid using NumPy stacking.
        # Top row: [Frame 0 | Vertical Border | Frame 1]
        top_row = np.hstack((processed_frames[0], v_border, processed_frames[1]))
        # Bottom row: [Frame 2 | Vertical Border | Frame 3]
        bottom_row = np.hstack((processed_frames[2], v_border, processed_frames[3]))
        # Full mosaic: [Top Row / Horizontal Border / Bottom Row]
        full_mosaic = np.vstack((top_row, h_border, bottom_row))
        
        # --- Add Frame Counter Overlay ---
        # This styling is designed to match the main system's output.
        text = f"Frame: {frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        
        # Position the text in the bottom-right corner.
        text_origin = (self.mosaic_width - 190, self.mosaic_height - 15)
        box_top_left = (self.mosaic_width - 200, self.mosaic_height - 50)
        box_bottom_right = (self.mosaic_width, self.mosaic_height)
        
        # Draw a black rectangle as a background for the text.
        cv2.rectangle(full_mosaic, box_top_left, box_bottom_right, (0, 0, 0), -1)
        # Draw the white text.
        cv2.putText(full_mosaic, text, text_origin, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return full_mosaic
    
    def create_mosaic_video(self) -> None:
        """Reads frames from all videos, assembles them, and writes the output video."""
        if self.frame_count is None:
            logging.error("Frame count not determined. Cannot create video.")
            return

        logging.info(f"Creating mosaic video with {self.frame_count} frames...")
        
        for frame_idx in range(self.frame_count):
            frames = []
            
            # Read one frame from each video capture.
            for cap in self.video_captures:
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"Could not read frame {frame_idx}. The video source may have ended unexpectedly.")
                    break
                frames.append(frame)
            
            # If a frame could not be read from any camera, stop processing.
            if len(frames) != len(self.video_captures):
                logging.error(f"Incomplete frame set at index {frame_idx}. Halting video creation.")
                break
            
            # Create the combined mosaic frame.
            mosaic_frame = self._create_mosaic_frame(frames, frame_idx)
            
            # Initialize the VideoWriter on the first frame, once we know the exact
            # dimensions of the mosaic.
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
                output_dims = (mosaic_frame.shape[1], mosaic_frame.shape[0])
                self.video_writer = cv2.VideoWriter(
                    self.output_video, fourcc, self.target_fps, output_dims
                )
                
                if not self.video_writer.isOpened():
                    raise RuntimeError(f"Failed to initialize VideoWriter for: {self.output_video}")
                
                logging.info(f"Output video initialized: {output_dims[0]}x{output_dims[1]} @ {self.target_fps} FPS")
            
            self.video_writer.write(mosaic_frame)
            
            # Log progress periodically.
            if frame_idx > 0 and frame_idx % 50 == 0:
                progress = (frame_idx / self.frame_count) * 100
                logging.info(f"Progress: {frame_idx}/{self.frame_count} ({progress:.1f}%)")
        
        logging.info("Mosaic video processing complete.")
    
    def cleanup(self) -> None:
        """Releases all video capture and writer resources."""
        logging.info("Cleaning up resources...")
        for cap in self.video_captures:
            cap.release()
        
        if self.video_writer is not None:
            self.video_writer.release()
        
        logging.info("Cleanup complete.")

def main() -> None:
    """Parses command-line arguments and runs the mosaic creation process."""
    parser = argparse.ArgumentParser(description="Video Mosaic Creator for YOLO Tracking Output")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="output_videos", 
        help="Directory containing tracked video files (default: 'output_videos')."
    )
    parser.add_argument(
        "--output_video", 
        type=str, 
        default="mosaic_output.mp4", 
        help="Path for the output mosaic video (default: 'mosaic_output.mp4')."
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10, 
        help="Frame rate for the output video (default: 10)."
    )
    
    args = parser.parse_args()
    
    # Input validation.
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1) # Exit with an error code.
    
    logging.info("=" * 60)
    logging.info("Starting Video Mosaic Creator")
    logging.info(f"Input Directory: {args.input_dir}")
    logging.info(f"Output Video: {args.output_video}")
    logging.info(f"Target FPS: {args.fps}")
    logging.info("=" * 60)
    
    creator = None
    try:
        creator = VideoMosaicCreator(
            input_dir=args.input_dir,
            output_video=args.output_video,
            target_fps=args.fps
        )
        creator.create_mosaic_video()
        logging.info("=" * 60)
        logging.info("Mosaic video created successfully!")
        logging.info(f"Output saved to: {args.output_video}")
        logging.info("=" * 60)
        
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Ensure cleanup runs even if errors occur.
        if creator:
            creator.cleanup()

if __name__ == "__main__":
    main()