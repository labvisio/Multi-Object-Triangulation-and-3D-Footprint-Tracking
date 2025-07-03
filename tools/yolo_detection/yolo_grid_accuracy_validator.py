"""
Detection Grid Comparison Analysis: YOLO Triangulation Accuracy vs Virtual Reference Grid

This script compares triangulation points from different camera combinations 
(2, 3, and 4 cameras) using YOLO detection data with a virtual reference grid 
(7x7, 0.5m spacing) and calculates mean distances and standard deviations for each camera count.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tabulate import tabulate


def generate_virtual_grid(center_x=0, center_y=0, grid_size=7, spacing=0.5):
    """
    Generate a virtual reference grid.
    
    Parameters:
    -----------
    center_x : float
        X coordinate of grid center
    center_y : float
        Y coordinate of grid center
    grid_size : int
        Size of the grid (grid_size x grid_size)
    spacing : float
        Spacing between grid points in meters
        
    Returns:
    --------
    tuple
        (X_grid, Y_grid) numpy arrays with grid coordinates
    """
    # Calculate grid bounds
    half_size = (grid_size - 1) * spacing / 2
    
    # Create grid coordinates
    x = np.linspace(center_x - half_size, center_x + half_size, grid_size)
    y = np.linspace(center_y - half_size, center_y + half_size, grid_size)
    
    X_grid, Y_grid = np.meshgrid(x, y)
    
    return X_grid, Y_grid


def load_detection_analysis_results():
    """Load the detection combination analysis results."""
    results_file = "detection_combination_analysis_results.json"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run yolo_camera_combination_analyzer.py first.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def create_virtual_reference_grid():
    """Create the virtual reference grid (7x7, 0.5m spacing)."""
    X_grid, Y_grid = generate_virtual_grid(center_x=0, center_y=0, grid_size=7, spacing=0.5)
    
    # Extract valid grid points
    valid_mask = ~(np.isnan(X_grid) | np.isnan(Y_grid))
    grid_x = X_grid[valid_mask]
    grid_y = Y_grid[valid_mask]
    grid_z = np.zeros_like(grid_x)  # Ground level
    
    # Create 3D grid points
    grid_points = np.column_stack([grid_x, grid_y, grid_z])
    
    return grid_points


def calculate_distances_to_grid(triangulation_points, grid_points):
    """
    Calculate distances from triangulation points to the nearest grid points.
    
    Parameters:
    -----------
    triangulation_points : numpy.ndarray
        Array of 3D triangulation points [N, 3]
    grid_points : numpy.ndarray
        Array of 3D grid points [M, 3]
        
    Returns:
    --------
    numpy.ndarray
        Array of minimum distances from each triangulation point to grid
    """
    distances = []
    
    for tri_point in triangulation_points:
        # Calculate distance to all grid points
        point_distances = np.linalg.norm(grid_points - tri_point, axis=1)
        # Store minimum distance
        distances.append(np.min(point_distances))
    
    return np.array(distances)


def extract_triangulations_by_camera_count(analysis_results):
    """
    Extract all triangulation points organized by camera count.
    
    Parameters:
    -----------
    analysis_results : dict
        Analysis results from yolo_camera_combination_analyzer.py
        
    Returns:
    --------
    dict
        Dictionary with camera counts as keys and triangulation arrays as values
    """
    triangulations_by_count = {}
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in analysis_results:
            all_triangulations = []
            
            # Extract all triangulation points for this camera count
            for combo_str, tri_list in analysis_results[camera_count]['triangulations'].items():
                all_triangulations.extend(tri_list)
            
            if all_triangulations:
                triangulations_by_count[camera_count] = np.array(all_triangulations)
            else:
                triangulations_by_count[camera_count] = np.array([]).reshape(0, 3)
    
    return triangulations_by_count


def calculate_position_statistics(triangulations_by_count):
    """
    Calculate position variability statistics for each camera count.
    
    Parameters:
    -----------
    triangulations_by_count : dict
        Dictionary with triangulation points by camera count
        
    Returns:
    --------
    dict
        Statistics for each camera count
    """
    stats = {}
    
    for camera_count, triangulations in triangulations_by_count.items():
        if len(triangulations) > 1:
            # Calculate centroid
            centroid = np.mean(triangulations, axis=0)
            
            # Calculate distances from centroid
            distances_from_centroid = np.linalg.norm(triangulations - centroid, axis=1)
            
            # Calculate statistics
            stats[camera_count] = {
                'centroid': centroid,
                'mean_distance_from_centroid': np.mean(distances_from_centroid),
                'std_distance_from_centroid': np.std(distances_from_centroid),
                'position_variance': np.var(triangulations, axis=0),
                'position_std': np.std(triangulations, axis=0),
                'spatial_spread': np.max(triangulations, axis=0) - np.min(triangulations, axis=0)
            }
        else:
            stats[camera_count] = None
    
    return stats


def analyze_detection_grid_comparison():
    """Main analysis function."""
    print("Detection Grid Comparison Analysis: YOLO Triangulation vs Virtual Reference Grid")
    print("=" * 90)
    
    # Load detection analysis results
    analysis_results = load_detection_analysis_results()
    if analysis_results is None:
        return
    
    # Create virtual reference grid
    grid_points = create_virtual_reference_grid()
    print(f"Virtual reference grid: {len(grid_points)} points (7x7 grid, 0.5m spacing)")
    print(f"Grid bounds: X=[{np.min(grid_points[:, 0]):.1f}, {np.max(grid_points[:, 0]):.1f}], "
          f"Y=[{np.min(grid_points[:, 1]):.1f}, {np.max(grid_points[:, 1]):.1f}]")
    
    # Extract triangulations by camera count
    triangulations_by_count = extract_triangulations_by_camera_count(analysis_results)
    
    # Calculate position statistics
    position_stats = calculate_position_statistics(triangulations_by_count)
    
    # Calculate grid comparison statistics
    results = {}
    detailed_results = {}
    
    print(f"\nAnalyzing YOLO triangulation accuracy vs virtual grid...")
    print("-" * 70)
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in triangulations_by_count:
            tri_points = triangulations_by_count[camera_count]
            
            if len(tri_points) > 0:
                # Calculate distances to grid
                distances = calculate_distances_to_grid(tri_points, grid_points)
                
                # Calculate statistics
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                median_distance = np.median(distances)
                min_distance = np.min(distances)
                max_distance = np.max(distances)
                
                # Store results
                results[camera_count] = {
                    'num_points': len(tri_points),
                    'mean_distance': mean_distance,
                    'std_distance': std_distance,
                    'median_distance': median_distance,
                    'min_distance': min_distance,
                    'max_distance': max_distance
                }
                
                detailed_results[camera_count] = {
                    'triangulation_points': tri_points,
                    'distances': distances
                }
                
                # Print detailed results
                num_cameras = int(camera_count.split('_')[0])
                print(f"\n{num_cameras} Cameras:")
                print(f"  Number of triangulation points: {len(tri_points)}")
                print(f"  Mean distance to grid: {mean_distance:.4f} ± {std_distance:.4f} m")
                print(f"  Median distance to grid: {median_distance:.4f} m")
                print(f"  Distance range: [{min_distance:.4f}, {max_distance:.4f}] m")
                
                # Calculate percentage of points within certain thresholds
                within_10cm = np.sum(distances <= 0.1) / len(distances) * 100
                within_25cm = np.sum(distances <= 0.25) / len(distances) * 100
                within_50cm = np.sum(distances <= 0.5) / len(distances) * 100
                within_1m = np.sum(distances <= 1.0) / len(distances) * 100
                
                print(f"  Points within 10cm of grid: {within_10cm:.1f}%")
                print(f"  Points within 25cm of grid: {within_25cm:.1f}%")
                print(f"  Points within 50cm of grid: {within_50cm:.1f}%")
                print(f"  Points within 1m of grid: {within_1m:.1f}%")
                
                # Position spread analysis
                if position_stats[camera_count] is not None:
                    pos_stats = position_stats[camera_count]
                    print(f"  Position spread (X,Y,Z): [{pos_stats['spatial_spread'][0]:.3f}, "
                          f"{pos_stats['spatial_spread'][1]:.3f}, {pos_stats['spatial_spread'][2]:.3f}] m")
                    print(f"  Position std dev (X,Y,Z): [{pos_stats['position_std'][0]:.3f}, "
                          f"{pos_stats['position_std'][1]:.3f}, {pos_stats['position_std'][2]:.3f}] m")
            else:
                print(f"\n{camera_count}: No triangulation points found")
    
    # Create summary table
    print(f"\n" + "=" * 84)
    print("SUMMARY: Triangulation Accuracy by Camera Count")
    print("=" * 84)
    
    summary_table = []
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in results:
            data = results[camera_count]
            num_cameras = int(camera_count.split('_')[0])
            
            summary_table.append([
                f"{num_cameras} Cameras",
                data['num_points'],
                data['mean_distance'],
                data['std_distance'],
                data['median_distance'],
                data['min_distance'],
                data['max_distance']
            ])
    
    if summary_table:
        print(tabulate(summary_table,
                      headers=['Camera Count', 'Points', 'Mean Dist (m)', 'Std Dev (m)', 
                              'Median (m)', 'Min (m)', 'Max (m)'],
                      tablefmt='grid',
                      floatfmt='.4f'))
    
    # Statistical comparison
    print(f"\n" + "=" * 90)
    print("STATISTICAL COMPARISON")
    print("=" * 90)
    
    if len(results) >= 2:
        camera_counts = list(results.keys())
        
        print(f"\nComparison of mean distances:")
        for i in range(len(camera_counts)):
            for j in range(i + 1, len(camera_counts)):
                count1, count2 = camera_counts[i], camera_counts[j]
                mean1 = results[count1]['mean_distance']
                mean2 = results[count2]['mean_distance']
                diff = abs(mean1 - mean2)
                improvement = ((max(mean1, mean2) - min(mean1, mean2)) / max(mean1, mean2)) * 100
                
                better = count1 if mean1 < mean2 else count2
                worse = count2 if mean1 < mean2 else count1
                
                print(f"  {better} vs {worse}: {diff:.4f}m difference ({improvement:.1f}% improvement)")
    
    # Comparison with detection parameters
    if 'parameters' in analysis_results:
        params = analysis_results['parameters']
        print(f"\nAnalysis Parameters:")
        print(f"  Confidence threshold: {params.get('confidence_threshold', 'N/A')}")
        print(f"  Reference point: {params.get('reference_point', 'N/A')}")
        print(f"  Classes detected: {params.get('class_list', 'N/A')}")
    
    # Create visualizations
    create_detection_grid_plots(results, detailed_results, grid_points, position_stats)
    
    # Save results
    save_detection_grid_results(results, grid_points, analysis_results)
    
    return results, detailed_results


def create_detection_grid_plots(results, detailed_results, grid_points, position_stats):
    """Create comprehensive visualization plots for the detection grid comparison analysis."""
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Colors for different camera counts
    colors = ['red', 'blue', 'green']
    camera_labels = ['2 Cameras', '3 Cameras', '4 Cameras']
    
    # 1. Bar plot of mean distances with error bars
    ax1 = fig.add_subplot(3, 4, 1)
    camera_counts = []
    mean_distances = []
    std_distances = []
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in results:
            camera_counts.append(int(camera_count.split('_')[0]))
            mean_distances.append(results[camera_count]['mean_distance'])
            std_distances.append(results[camera_count]['std_distance'])
    
    bars = ax1.bar(camera_counts, mean_distances, yerr=std_distances, 
                   capsize=5, alpha=0.7, color=colors[:len(camera_counts)])
    ax1.set_xlabel('Number of Cameras')
    ax1.set_ylabel('Mean Distance to Grid (m)')
    ax1.set_title('Mean Distance to Virtual Grid\n(YOLO Detection Triangulation)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(camera_counts)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, mean_distances, std_distances):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                f'{mean_val:.3f}±{std_val:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Box plot of distance distributions
    ax2 = fig.add_subplot(3, 4, 2)
    distance_data = []
    labels = []
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in detailed_results:
            distance_data.append(detailed_results[camera_count]['distances'])
            labels.append(f"{camera_count.split('_')[0]} Cam")
    
    if distance_data:
        bp = ax2.boxplot(distance_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(distance_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_ylabel('Distance to Grid (m)')
        ax2.set_title('Distance Distribution to Virtual Grid')
        ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy comparison bar chart
    ax3 = fig.add_subplot(3, 4, 3)
    
    accuracy_data = {
        'Within 25cm': [],
        'Within 50cm': [],
        'Within 1m': []
    }
    
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in detailed_results:
            distances = detailed_results[camera_count]['distances']
            accuracy_data['Within 25cm'].append(np.sum(distances <= 0.25) / len(distances) * 100)
            accuracy_data['Within 50cm'].append(np.sum(distances <= 0.5) / len(distances) * 100)
            accuracy_data['Within 1m'].append(np.sum(distances <= 1.0) / len(distances) * 100)
    
    x_pos = np.arange(len(camera_counts))
    width = 0.25
    
    for i, (threshold, values) in enumerate(accuracy_data.items()):
        ax3.bar(x_pos + i*width, values, width, label=threshold, alpha=0.8)
    
    ax3.set_xlabel('Number of Cameras')
    ax3.set_ylabel('Percentage of Points (%)')
    ax3.set_title('Accuracy Thresholds')
    ax3.set_xticks(x_pos + width)
    ax3.set_xticklabels(camera_counts)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of triangulations comparison
    ax4 = fig.add_subplot(3, 4, 4)
    num_triangulations = [results[f'{c}_cameras']['num_points'] for c in camera_counts 
                         if f'{c}_cameras' in results]
    
    bars = ax4.bar(camera_counts, num_triangulations, alpha=0.7, color=colors[:len(camera_counts)])
    ax4.set_xlabel('Number of Cameras')
    ax4.set_ylabel('Number of Triangulations')
    ax4.set_title('Total Triangulations per Camera Count')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(camera_counts)
    
    # Add value labels
    for bar, count in zip(bars, num_triangulations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontsize=10)
    
    # 5-7. 3D scatter plots for each camera count
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        ax = fig.add_subplot(3, 4, 5+i, projection='3d')
        
        # Plot grid points
        ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                  c='green', marker='s', s=80, alpha=0.8, label='Virtual Grid')
        
        if camera_count in detailed_results:
            points = detailed_results[camera_count]['triangulation_points']
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c=colors[i], alpha=0.6, s=20, label=camera_labels[i])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_zlim(-0.5, 3)
        ax.set_title(f'{camera_labels[i]} vs Grid')
        ax.legend()
    
    # 8. Combined 3D plot
    ax8 = fig.add_subplot(3, 4, 8, projection='3d')
    
    ax8.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
               c='green', marker='s', s=100, alpha=0.8, label='Virtual Grid')
    
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        if camera_count in detailed_results:
            points = detailed_results[camera_count]['triangulation_points']
            ax8.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c=colors[i], alpha=0.6, s=30, label=camera_labels[i])
    
    ax8.set_xlabel('X (m)')
    ax8.set_ylabel('Y (m)')
    ax8.set_zlabel('Z (m)')
    ax8.set_zlim(-0.5, 3)
    ax8.set_title('All Camera Counts vs Grid')
    ax8.legend()
    
    # 9. Histogram of distances
    ax9 = fig.add_subplot(3, 4, 9)
    
    for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
        if camera_count in detailed_results:
            distances = detailed_results[camera_count]['distances']
            ax9.hist(distances, bins=30, alpha=0.6, 
                    label=camera_labels[i], color=colors[i], density=True)
    
    ax9.set_xlabel('Distance to Grid (m)')
    ax9.set_ylabel('Density')
    ax9.set_title('Distance Distribution Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Position spread analysis
    ax10 = fig.add_subplot(3, 4, 10)
    
    if position_stats:
        spread_data = {'X': [], 'Y': [], 'Z': []}
        for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
            if camera_count in position_stats and position_stats[camera_count] is not None:
                spread = position_stats[camera_count]['spatial_spread']
                spread_data['X'].append(spread[0])
                spread_data['Y'].append(spread[1])
                spread_data['Z'].append(spread[2])
        
        x_pos = np.arange(len(camera_counts))
        width = 0.25
        
        for i, (axis, values) in enumerate(spread_data.items()):
            if values:
                ax10.bar(x_pos + i*width, values, width, label=f'{axis} Spread', alpha=0.8)
        
        ax10.set_xlabel('Number of Cameras')
        ax10.set_ylabel('Spatial Spread (m)')
        ax10.set_title('Position Spread by Axis')
        ax10.set_xticks(x_pos + width)
        ax10.set_xticklabels(camera_counts)
        ax10.legend()
        ax10.grid(True, alpha=0.3)
    
    # 11. Summary statistics table
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('tight')
    ax11.axis('off')
    
    # Create summary table
    table_data = []
    for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
        if camera_count in results:
            data = results[camera_count]
            distances = detailed_results[camera_count]['distances']
            
            within_25cm = np.sum(distances <= 0.25) / len(distances) * 100
            within_50cm = np.sum(distances <= 0.5) / len(distances) * 100
            
            table_data.append([
                f"{camera_count.split('_')[0]} Cam",
                f"{data['num_points']}",
                f"{data['mean_distance']:.3f}",
                f"{within_25cm:.1f}%",
                f"{within_50cm:.1f}%"
            ])
    
    if table_data:
        table = ax11.table(cellText=table_data,
                         colLabels=['Cameras', 'Points', 'Mean Dist', '≤25cm', '≤50cm'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
    
    ax11.set_title('Detection Grid Comparison Summary', pad=20)
    
    # 12. Performance comparison radar chart
    ax12 = fig.add_subplot(3, 4, 12, projection='polar')
    
    # Metrics for radar chart (normalized to 0-1)
    if len(results) >= 2:
        metrics = ['Accuracy (≤25cm)', 'Precision (low std)', 'Coverage (points)', 'Consistency']
        
        # Normalize metrics for comparison
        max_accuracy = max([np.sum(detailed_results[cc]['distances'] <= 0.25) / len(detailed_results[cc]['distances']) 
                           for cc in results.keys()])
        min_std = min([results[cc]['std_distance'] for cc in results.keys()])
        max_std = max([results[cc]['std_distance'] for cc in results.keys()])
        max_points = max([results[cc]['num_points'] for cc in results.keys()])
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, camera_count in enumerate(['2_cameras', '3_cameras', '4_cameras']):
            if camera_count in results:
                accuracy = np.sum(detailed_results[camera_count]['distances'] <= 0.25) / len(detailed_results[camera_count]['distances'])
                precision = 1 - (results[camera_count]['std_distance'] - min_std) / (max_std - min_std) if max_std > min_std else 1
                coverage = results[camera_count]['num_points'] / max_points
                consistency = precision  # Using precision as a proxy for consistency
                
                values = [accuracy / max_accuracy, precision, coverage, consistency]
                values += values[:1]  # Complete the circle
                
                ax12.plot(angles, values, 'o-', linewidth=2, label=camera_labels[i], color=colors[i])
                ax12.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax12.set_xticks(angles[:-1])
        ax12.set_xticklabels(metrics)
        ax12.set_ylim(0, 1)
        ax12.set_title('Performance Comparison\n(Normalized Metrics)', pad=20)
        ax12.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "detection_grid_comparison_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive visualization saved to: {output_path}")
    
    plt.show()


def save_detection_grid_results(results, grid_points, analysis_results):
    """Save the detection grid analysis results to a JSON file."""
    
    # Prepare results for JSON serialization
    json_results = {
        'analysis_timestamp': '2025-06-30 20:00:00',
        'analysis_type': 'YOLO Detection Grid Comparison',
        'virtual_grid': {
            'size': '7x7',
            'spacing': '0.5m',
            'total_points': len(grid_points),
            'bounds': {
                'x_min': float(np.min(grid_points[:, 0])),
                'x_max': float(np.max(grid_points[:, 0])),
                'y_min': float(np.min(grid_points[:, 1])),
                'y_max': float(np.max(grid_points[:, 1]))
            }
        },
        'detection_parameters': analysis_results.get('parameters', {}),
        'camera_comparison_results': {}
    }
    
    for camera_count, data in results.items():
        num_cameras = int(camera_count.split('_')[0])
        json_results['camera_comparison_results'][f'{num_cameras}_cameras'] = {
            'num_triangulation_points': int(data['num_points']),
            'mean_distance_to_grid': float(data['mean_distance']),
            'std_distance_to_grid': float(data['std_distance']),
            'median_distance_to_grid': float(data['median_distance']),
            'min_distance_to_grid': float(data['min_distance']),
            'max_distance_to_grid': float(data['max_distance'])
        }
    
    # Add comparison summary
    if len(results) >= 2:
        best_camera_count = min(results.keys(), key=lambda x: results[x]['mean_distance'])
        worst_camera_count = max(results.keys(), key=lambda x: results[x]['mean_distance'])
        
        json_results['comparison_summary'] = {
            'best_performing': best_camera_count,
            'worst_performing': worst_camera_count,
            'best_mean_distance': float(results[best_camera_count]['mean_distance']),
            'worst_mean_distance': float(results[worst_camera_count]['mean_distance']),
            'improvement_percentage': float(
                ((results[worst_camera_count]['mean_distance'] - results[best_camera_count]['mean_distance']) / 
                 results[worst_camera_count]['mean_distance']) * 100
            )
        }
    
    # Save to file
    output_file = "detection_grid_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function."""
    results, detailed_results = analyze_detection_grid_comparison()
    
    if results:
        print(f"\n" + "=" * 90)
        print("CONCLUSION")
        print("=" * 90)
        
        # Find the best and worst performing camera counts
        best_camera_count = min(results.keys(), key=lambda x: results[x]['mean_distance'])
        worst_camera_count = max(results.keys(), key=lambda x: results[x]['mean_distance'])
        
        best_mean = results[best_camera_count]['mean_distance']
        best_std = results[best_camera_count]['std_distance']
        worst_mean = results[worst_camera_count]['mean_distance']
        
        print(f"\nBest performing configuration: {best_camera_count.replace('_', ' ')}")
        print(f"Mean distance to virtual grid: {best_mean:.4f} ± {best_std:.4f} m")
        
        print(f"\nWorst performing configuration: {worst_camera_count.replace('_', ' ')}")
        print(f"Mean distance to virtual grid: {worst_mean:.4f} m")
        
        # Calculate improvement
        improvement = ((worst_mean - best_mean) / worst_mean) * 100
        print(f"\nImprovement from worst to best: {improvement:.1f}%")
        
        # Analysis insights
        print(f"\nKey Insights:")
        for camera_count in ['2_cameras', '3_cameras', '4_cameras']:
            if camera_count in results:
                data = results[camera_count]
                within_50cm = np.sum(detailed_results[camera_count]['distances'] <= 0.5) / len(detailed_results[camera_count]['distances']) * 100
                print(f"  {camera_count.replace('_', ' ')}: {data['num_points']} triangulations, "
                      f"{within_50cm:.1f}% within 50cm of grid")


if __name__ == "__main__":
    main()
