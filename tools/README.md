# Tools Directory

This directory contains analysis and utility tools for the Multi-Object Triangulation and 3D Footprint Tracking system. The tools are organized into functional categories for data processing, validation, analysis, and visualization.

## Data Processing

### rosbag_odometry_extractor.py
Extracts and visualizes odometry data from ROS 2 bag files. Reads messages from `/odrive/odom`, `/amcl_pose`, and trigger topics, normalizes timestamps, and generates trajectory plots. Outputs processed data to CSV format for further analysis.

## Reconstruction Validation

### grid_tracking_accuracy_analyzer.py
Analyzes 3D tracking accuracy against a known reference grid. Calculates Euclidean distance errors, statistical metrics (RMSE, MAE, standard deviation), and generates error visualization plots. Provides comprehensive accuracy assessment reports.

### multi_camera_combination_aruco_analyzer.py
Evaluates 3D reconstruction accuracy across different camera combinations using ArUco markers. Tests 2, 3, and 4 camera setups, applies coordinate corrections, and compares results against virtual reference grids. Generates statistical analysis reports.

## Trajectory Analysis

### circular_trajectory_validator.py
Validates robot trajectories against fixed-radius reference circles. Compares world trajectory (from cameras) and odometry data against idealized circular paths. Calculates optimal circle centers and error metrics for trajectory conformance analysis.

### odometry_camera_fusion_analyzer.py
Performs comparative analysis between camera-based and odometry trajectories. Implements hybrid analysis with optimal alignment (scaling, translation, rotation) and fixed-radius analysis for circular path validation. Outputs statistical error metrics and visualization plots.

### odometry_time_preprocessor.py
Preprocesses raw odometry CSV files by adding formatted timestamp columns. Converts Unix timestamps to human-readable formats, calculates relative time measurements, and enhances data readability for time-based analysis.

### robot_trajectory_comparator.py
Compares robot trajectories from odometry sensors and camera reconstruction systems. Applies coordinate transformations, generates multi-panel comparison plots, and calculates point-to-point error metrics between trajectory datasets.

### trajectory_alignment_comparator.py
Aligns odometry trajectories with ground-truth world trajectories using circle fitting algorithms. Performs scale correction, translation alignment, direction correction, and rotational optimization. Generates aligned trajectory comparisons and error analysis.

## Visualization

### ground_truth_position_plotter.py
Visualizes and analyzes ground truth position data from ArUco markers and people detection experiments. Generates statistical summaries, 2D/3D scatter plots, histogram analysis, and covariance ellipse comparisons for spatial distribution assessment.

### multi_camera_video_compositor.py
Creates mosaic videos from multiple camera tracking outputs. Combines four camera feeds into a single 2x2 grid layout with styling that matches the main tracking system output. Includes frame counters and border formatting.

### multi_object_trajectory_plotter.py
Generates 3D trajectory visualizations for multiple tracked objects from JSON data. Supports both animated and static plotting modes, applies class-based styling (colors/markers), and provides filtering options for specific object types.

### plot_from_json.py
Visualizes tracking data from JSON files using the same plotting style as the main tracking system. Loads trajectory data and creates plots without requiring the full tracking pipeline execution.

### reference_grid_visualizer.py
Generates 2D reference grids on the Z=0 plane and visualizes them in 3D plots. Can overlay camera calibration data to show camera positions and orientations. Useful for validation and planning of data capture sessions.

### ros2_odometry_plotter.py
Creates 2D trajectory plots from robot odometry CSV data. Visualizes robot path with start/end point markers, grid overlay, and proper aspect ratio. Includes error handling for file and column validation.

## YOLO Detection

### yolo_camera_combination_analyzer.py
Analyzes object detection and 3D triangulation performance across different camera combinations. Evaluates 2, 3, and 4 camera setups using the YOLO tracking pipeline and generates performance metrics for each configuration.

### yolo_grid_accuracy_validator.py
Validates YOLO triangulation accuracy against virtual reference grids. Compares detection results from different camera combinations with a 7x7 grid (0.5m spacing) and calculates distance statistics and error metrics.

### yolo_multi_camera_demo.py
Demonstrates YOLO object tracking on synchronized four-camera feeds. Processes video streams, applies tracking with consistent styling, and generates annotated output videos for each camera view.

## Usage

Each tool can be executed independently with appropriate input data. Most tools support command-line arguments for configuration. Refer to individual file headers for specific usage instructions and parameter details.

## Dependencies

- OpenCV-Python
- NumPy
- Matplotlib
- Pandas
- SciPy
- ROS 2 (for rosbag tools)
- Ultralytics YOLO
- Custom modules from the main tracking system