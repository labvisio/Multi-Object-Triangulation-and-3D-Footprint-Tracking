"""
Detection Combination Analysis for Multi-Camera YOLO Tracking

This script analyzes the effect of different camera combinations on object detection 
and 3D triangulation accuracy using the YOLO tracking pipeline. It processes image 
captures and evaluates performance with 2, 3, and 4 camera combinations.

Date: 2025
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from itertools import combinations
import statistics
from tabulate import tabulate
import logging
import glob

# Add the source directory to the path
# Get the project root directory (two levels up from this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
source = os.path.join(project_root, 'source')
sys.path.append(source)

from config import YOLO_MODEL, CLASS_NAMES, CONFIDENCE
from ploting_utils import Utils
from video_loader import ImageLoader
from tracker import Tracker
from triangulation import triangulate_ransac
from matcher import Matcher
from bbox_utils import get_centroid, divide_bbox
from detection import ObjectDetection

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during analysis

class DetectionCombinationAnalysis:
    """
    Analyzes object detection and triangulation performance across different camera combinations.
    """
    
    def __init__(self, image_path, confidence=0.6, class_list=[0], reference_point="bottom_center"):
        """
        Initialize the analysis.
        
        Args:
            image_path (str): Path to image captures directory
            confidence (float): YOLO confidence threshold
            class_list (list): List of class IDs to detect
            reference_point (str): Reference point for triangulation
        """
        self.image_path = image_path
        self.confidence = confidence
        self.class_list = class_list
        self.reference_point = reference_point
        
        # Camera setup
        self.camera_indices = [0, 1, 2, 3]
        
        # Initialize components
        self.utils = Utils()
        self.image_loader = ImageLoader(image_path, camera_indices=self.camera_indices)
        self.matcher = Matcher(distance_threshold=0.2, drift_threshold=0.4)
        
        # Results storage
        self.results = {}
        self._initialize_results()
        
    def _initialize_results(self):
        """Initialize results storage structure."""
        # Define camera combinations
        combinations_2 = list(combinations(self.camera_indices, 2))
        combinations_3 = list(combinations(self.camera_indices, 3))
        combinations_4 = list(combinations(self.camera_indices, 4))
        
        self.results = {
            '2_cameras': {
                'combinations': combinations_2,
                'detections': {tuple(combo): [] for combo in combinations_2},
                'triangulations': {tuple(combo): [] for combo in combinations_2},
                'successful_triangulations': {tuple(combo): 0 for combo in combinations_2},
                'detection_counts': {tuple(combo): [] for combo in combinations_2},
                'matching_success': {tuple(combo): [] for combo in combinations_2}
            },
            '3_cameras': {
                'combinations': combinations_3,
                'detections': {tuple(combo): [] for combo in combinations_3},
                'triangulations': {tuple(combo): [] for combo in combinations_3},
                'successful_triangulations': {tuple(combo): 0 for combo in combinations_3},
                'detection_counts': {tuple(combo): [] for combo in combinations_3},
                'matching_success': {tuple(combo): [] for combo in combinations_3}
            },
            '4_cameras': {
                'combinations': combinations_4,
                'detections': {tuple(combo): [] for combo in combinations_4},
                'triangulations': {tuple(combo): [] for combo in combinations_4},
                'successful_triangulations': {tuple(combo): 0 for combo in combinations_4},
                'detection_counts': {tuple(combo): [] for combo in combinations_4},
                'matching_success': {tuple(combo): [] for combo in combinations_4}
            }
        }
    
    def _create_tracker_for_cameras(self, camera_subset):
        """Create a tracker instance for a specific subset of cameras."""
        cam_numbers = list(camera_subset)
        return Tracker([YOLO_MODEL for _ in range(len(cam_numbers))], 
                      cam_numbers, self.class_list, self.confidence)
    
    def _filter_detections_by_cameras(self, detections, camera_subset):
        """Filter detections to only include specified cameras."""
        return [det for det in detections if det.cam in camera_subset]
    
    def _perform_detection_and_matching(self, frames, camera_subset):
        """
        Perform detection and matching for a specific camera subset.
        
        Args:
            frames (list): List of frames from all cameras
            camera_subset (tuple): Subset of camera indices to use
            
        Returns:
            tuple: (detections, matches, triangulated_points)
        """
        import networkx as nx
        
        # Filter frames to only include the camera subset
        subset_frames = [frames[i] if i in camera_subset else None for i in range(len(frames))]
        
        # Create tracker for this subset
        tracker = self._create_tracker_for_cameras(camera_subset)
        
        # Detect and track
        tracker.detect_and_track([frames[i] for i in camera_subset])
        detections = tracker.get_detections()
        
        if not detections:
            return [], [], []
        
        # Build matching graph
        graph = nx.Graph()
        
        # Add nodes for each detection
        for d in detections:
            graph.add_node(
                f"cam{d.cam}id{int(d.id)}",
                bbox=d.bbox,
                id=int(d.id),
                centroid=d.centroid,
                name=d.name,
            )
        
        # Match detections between camera pairs
        matches = []
        camera_list = list(camera_subset)
        for k in range(len(camera_list)):
            for j in range(k+1, len(camera_list)):
                cam_matches = self.matcher.match_detections(detections, [camera_list[k], camera_list[j]])
                matches.extend(cam_matches)
                
                # Add edges to graph
                for match in cam_matches:
                    n1 = f"cam{camera_list[k]}id{int(match[0].id)}"
                    n2 = f"cam{camera_list[j]}id{int(match[1].id)}"
                    if n1 in graph.nodes and n2 in graph.nodes:
                        graph.add_edge(n1, n2)
        
        # Perform triangulation on connected components
        triangulated_points = []
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            if len(subgraph.nodes) >= 2:  # Need at least 2 cameras
                ids = sorted(subgraph.nodes)
                d2_points = []
                proj_matrices = []
                
                for node in ids:
                    cam = int(node.split("cam")[1].split("id")[0])
                    bbox = subgraph.nodes[node]["bbox"]
                    
                    # Get projection matrix for this camera
                    P_cam = self.matcher.P_all[cam]
                    
                    # Calculate reference point based on configuration
                    if self.reference_point == "center":
                        point_2d = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    elif self.reference_point == "top_center":
                        point_2d = ((bbox[0] + bbox[2]) / 2, bbox[1])
                    elif self.reference_point == "feet":
                        feet_y = bbox[3] - 0.2 * (bbox[3] - bbox[1])
                        point_2d = ((bbox[0] + bbox[2]) / 2, feet_y)
                    else:  # bottom_center
                        point_2d = ((bbox[0] + bbox[2]) / 2, bbox[3])
                    
                    d2_points.append(point_2d)
                    proj_matrices.append(P_cam)
                
                if len(d2_points) >= 2:
                    try:
                        point_3d, _ = triangulate_ransac(proj_matrices, d2_points)
                        triangulated_points.append(point_3d)
                    except:
                        pass  # Failed triangulation
        
        return detections, matches, triangulated_points
    
    def analyze_capture(self, capture_idx):
        """
        Analyze a single capture with all camera combinations.
        
        Args:
            capture_idx (int): Index of the capture to analyze
            
        Returns:
            dict: Analysis results for this capture
        """
        # Set the current capture
        self.image_loader.current_capture_idx = capture_idx
        frames = self.image_loader.get_frames()
        
        if any(frame is None for frame in frames):
            return None
        
        capture_results = {}
        
        # Test all camera combinations
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            capture_results[num_cameras] = {}
            
            for combo in self.results[num_cameras]['combinations']:
                detections, matches, triangulated_points = self._perform_detection_and_matching(
                    frames, combo
                )
                
                # Store results
                detection_count = len(detections)
                triangulation_count = len(triangulated_points)
                matching_success_rate = len(matches) / max(len(detections), 1)
                
                capture_results[num_cameras][combo] = {
                    'detection_count': detection_count,
                    'triangulation_count': triangulation_count,
                    'matching_success_rate': matching_success_rate,
                    'triangulated_points': triangulated_points
                }
                
                # Update overall results
                self.results[num_cameras]['detection_counts'][combo].append(detection_count)
                self.results[num_cameras]['triangulations'][combo].extend(triangulated_points)
                if triangulation_count > 0:
                    self.results[num_cameras]['successful_triangulations'][combo] += 1
                self.results[num_cameras]['matching_success'][combo].append(matching_success_rate)
        
        return capture_results
    
    def run_analysis(self, max_captures=None):
        """
        Run the complete analysis on all captures.
        
        Args:
            max_captures (int, optional): Maximum number of captures to process
        """
        total_captures = self.image_loader.get_number_of_frames()
        if max_captures:
            total_captures = min(total_captures, max_captures)
        
        print(f"Starting analysis of {total_captures} captures...")
        print(f"Camera combinations:")
        print(f"  2 cameras: {len(self.results['2_cameras']['combinations'])} combinations")
        print(f"  3 cameras: {len(self.results['3_cameras']['combinations'])} combinations")
        print(f"  4 cameras: {len(self.results['4_cameras']['combinations'])} combinations")
        
        processed_count = 0
        for i in range(total_captures):
            print(f"\nProcessing capture {i+1}/{total_captures}...")
            
            result = self.analyze_capture(i)
            if result is not None:
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Successfully processed {processed_count} captures")
            else:
                print(f"Skipped capture {i+1} (missing frames)")
        
        print(f"\nCompleted analysis of {processed_count} captures")
    
    def calculate_statistics(self):
        """Calculate summary statistics from the analysis results."""
        summary = {}
        
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            combo_stats = []
            total_detections = 0
            total_triangulations = 0
            total_successful_captures = 0
            
            for combo in self.results[num_cameras]['combinations']:
                detection_counts = self.results[num_cameras]['detection_counts'][combo]
                triangulations = self.results[num_cameras]['triangulations'][combo]
                successful_triangulations = self.results[num_cameras]['successful_triangulations'][combo]
                matching_success = self.results[num_cameras]['matching_success'][combo]
                
                # Calculate statistics for this combination
                avg_detections = np.mean(detection_counts) if detection_counts else 0
                avg_triangulations = len(triangulations) / len(detection_counts) if detection_counts else 0
                avg_matching_success = np.mean(matching_success) if matching_success else 0
                success_rate = successful_triangulations / len(detection_counts) if detection_counts else 0
                
                # Calculate triangulation consistency (variability)
                if len(triangulations) > 1:
                    positions = np.array(triangulations)
                    if len(positions.shape) == 2 and positions.shape[0] > 1:
                        variability = np.std(positions, axis=0).mean()
                    else:
                        variability = 0
                else:
                    variability = 0
                
                combo_stats.append({
                    'combination': combo,
                    'avg_detections_per_capture': avg_detections,
                    'avg_triangulations_per_capture': avg_triangulations,
                    'triangulation_success_rate': success_rate,
                    'avg_matching_success_rate': avg_matching_success,
                    'position_variability': variability,
                    'total_triangulations': len(triangulations),
                    'captures_processed': len(detection_counts)
                })
                
                total_detections += sum(detection_counts)
                total_triangulations += len(triangulations)
                total_successful_captures += successful_triangulations
            
            # Overall statistics for this camera count
            summary[num_cameras] = {
                'combination_stats': combo_stats,
                'total_detections': total_detections,
                'total_triangulations': total_triangulations,
                'total_successful_captures': total_successful_captures,
                'average_detections_per_combo': total_detections / len(combo_stats) if combo_stats else 0,
                'average_triangulations_per_combo': total_triangulations / len(combo_stats) if combo_stats else 0,
                'average_success_rate': total_successful_captures / (len(combo_stats) * len(detection_counts)) if combo_stats and detection_counts else 0
            }
        
        return summary
    
    def print_results(self, summary):
        """Print formatted analysis results."""
        print("\n" + "="*80)
        print("DETECTION AND TRIANGULATION COMBINATION ANALYSIS RESULTS")
        print("="*80)
        
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            print(f"\n{num_cameras.replace('_', ' ').title()}:")
            print("-" * 50)
            
            # Create table for this camera count
            table_data = []
            for stat in summary[num_cameras]['combination_stats']:
                table_data.append([
                    str(stat['combination']),
                    f"{stat['avg_detections_per_capture']:.2f}",
                    f"{stat['avg_triangulations_per_capture']:.2f}",
                    f"{stat['triangulation_success_rate']:.2%}",
                    f"{stat['avg_matching_success_rate']:.2%}",
                    f"{stat['position_variability']:.4f}",
                    f"{stat['total_triangulations']}"
                ])
            
            if table_data:
                headers = ["Cameras", "Avg Det/Cap", "Avg Tri/Cap", "Success Rate", 
                          "Match Rate", "Pos Var", "Total Tri"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            
            # Overall statistics
            print(f"\nOverall Statistics:")
            print(f"  Total detections: {summary[num_cameras]['total_detections']}")
            print(f"  Total triangulations: {summary[num_cameras]['total_triangulations']}")
            print(f"  Average success rate: {summary[num_cameras]['average_success_rate']:.2%}")
        
        # Summary comparison
        print("\n" + "="*80)
        print("SUMMARY COMPARISON")
        print("="*80)
        
        summary_table = []
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            camera_count = num_cameras.split('_')[0]
            summary_table.append([
                camera_count,
                summary[num_cameras]['total_detections'],
                summary[num_cameras]['total_triangulations'],
                f"{summary[num_cameras]['average_success_rate']:.2%}",
                f"{summary[num_cameras]['average_detections_per_combo']:.1f}",
                f"{summary[num_cameras]['average_triangulations_per_combo']:.1f}"
            ])
        
        if summary_table:
            headers = ["Cameras", "Total Det", "Total Tri", "Avg Success", "Avg Det/Combo", "Avg Tri/Combo"]
            print(tabulate(summary_table, headers=headers, tablefmt="grid"))
    
    def save_results(self, summary, output_file):
        """Save results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for num_cameras in self.results:
            json_results[num_cameras] = {
                'combinations': [list(combo) for combo in self.results[num_cameras]['combinations']],
                'triangulations': {},
                'successful_triangulations': {},
                'detection_counts': {},
                'matching_success': {}
            }
            
            for combo in self.results[num_cameras]['combinations']:
                combo_key = str(combo)
                triangulations = self.results[num_cameras]['triangulations'][combo]
                json_results[num_cameras]['triangulations'][combo_key] = [
                    point.tolist() if isinstance(point, np.ndarray) else point 
                    for point in triangulations
                ]
                json_results[num_cameras]['successful_triangulations'][combo_key] = \
                    self.results[num_cameras]['successful_triangulations'][combo]
                json_results[num_cameras]['detection_counts'][combo_key] = \
                    self.results[num_cameras]['detection_counts'][combo]
                json_results[num_cameras]['matching_success'][combo_key] = \
                    self.results[num_cameras]['matching_success'][combo]
        
        # Add summary and metadata
        json_results['summary'] = summary
        json_results['analysis_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json_results['parameters'] = {
            'confidence_threshold': self.confidence,
            'class_list': self.class_list,
            'reference_point': self.reference_point,
            'image_path': self.image_path
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nDetailed results saved to: {output_file}")
    
    def create_visualization(self, summary, output_path):
        """Create visualization of the analysis results."""
        # Prepare data for plotting
        camera_counts = []
        total_detections = []
        total_triangulations = []
        success_rates = []
        
        for num_cameras in ['2_cameras', '3_cameras', '4_cameras']:
            camera_count = int(num_cameras.split('_')[0])
            camera_counts.append(camera_count)
            total_detections.append(summary[num_cameras]['total_detections'])
            total_triangulations.append(summary[num_cameras]['total_triangulations'])
            success_rates.append(summary[num_cameras]['average_success_rate'])
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total detections
        ax1.bar(camera_counts, total_detections, alpha=0.7, color='blue')
        ax1.set_xlabel('Number of Cameras')
        ax1.set_ylabel('Total Detections')
        ax1.set_title('Total Detections vs Number of Cameras')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(camera_counts)
        
        # Plot total triangulations
        ax2.bar(camera_counts, total_triangulations, alpha=0.7, color='green')
        ax2.set_xlabel('Number of Cameras')
        ax2.set_ylabel('Total Triangulations')
        ax2.set_title('Total Triangulations vs Number of Cameras')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(camera_counts)
        
        # Plot success rates
        success_rates_percent = [rate * 100 for rate in success_rates]
        ax3.bar(camera_counts, success_rates_percent, alpha=0.7, color='orange')
        ax3.set_xlabel('Number of Cameras')
        ax3.set_ylabel('Average Success Rate (%)')
        ax3.set_title('Triangulation Success Rate vs Number of Cameras')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(camera_counts)
        
        # Plot efficiency (triangulations per detection)
        efficiency = [tri/det if det > 0 else 0 for tri, det in zip(total_triangulations, total_detections)]
        ax4.bar(camera_counts, efficiency, alpha=0.7, color='red')
        ax4.set_xlabel('Number of Cameras')
        ax4.set_ylabel('Triangulations per Detection')
        ax4.set_title('Triangulation Efficiency vs Number of Cameras')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(camera_counts)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.show()


def main():
    """Main function to run the detection combination analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detection Combination Analysis")
    parser.add_argument("--image_path", type=str, 
                       help="Path to image captures directory")
    parser.add_argument("--confidence", type=float, default=0.6,
                       help="YOLO confidence threshold (default: 0.6)")
    parser.add_argument("--class_list", type=int, nargs='+', default=[0],
                       help="List of classes to detect (default: [0] for person)")
    parser.add_argument("--reference_point", type=str, default="bottom_center",
                       choices=["bottom_center", "center", "top_center", "feet"],
                       help="Reference point for triangulation (default: bottom_center)")
    parser.add_argument("--max_captures", type=int,
                       help="Maximum number of captures to process")
    parser.add_argument("--output_file", type=str, default="detection_combination_analysis_results.json",
                       help="Output JSON file (default: detection_combination_analysis_results.json)")
    
    args = parser.parse_args()
    
    # Ensure output file uses absolute path to project root
    if not os.path.isabs(args.output_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.output_file = os.path.join(project_root, args.output_file)
    
    # Use default path if not provided
    if not args.image_path:
        # Get the project root directory (two levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        args.image_path = os.path.join(project_root, "captures_clean")
    
    # Validate input directory
    if not os.path.exists(args.image_path):
        print(f"Error: Image directory not found: {args.image_path}")
        print("Available directories:")
        # Get the project root directory (two levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        for item in os.listdir(project_root):
            item_path = os.path.join(project_root, item)
            if os.path.isdir(item_path) and ("capture" in item.lower() or "experiment" in item.lower()):
                print(f"  {item}")
        return
    
    print("="*80)
    print("DETECTION AND TRIANGULATION COMBINATION ANALYSIS")
    print("="*80)
    print(f"Image path: {args.image_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Classes to detect: {args.class_list}")
    print(f"Reference point: {args.reference_point}")
    if args.max_captures:
        print(f"Max captures: {args.max_captures}")
    
    try:
        # Initialize analysis
        analysis = DetectionCombinationAnalysis(
            image_path=args.image_path,
            confidence=args.confidence,
            class_list=args.class_list,
            reference_point=args.reference_point
        )
        
        # Run analysis
        analysis.run_analysis(max_captures=args.max_captures)
        
        # Calculate and display results
        summary = analysis.calculate_statistics()
        analysis.print_results(summary)
        
        # Save results
        analysis.save_results(summary, args.output_file)
        
        # Create visualization
        plot_path = args.output_file.replace('.json', '_plot.png')
        # Ensure plot path is absolute
        if not os.path.isabs(plot_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            plot_path = os.path.join(project_root, plot_path)
        analysis.create_visualization(summary, plot_path)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
