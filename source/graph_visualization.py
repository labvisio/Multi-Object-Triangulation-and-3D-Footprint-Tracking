"""
Graph Visualization Module

This module provides functions for visualizing correspondence graphs
in multi-camera tracking applications using NetworkX.
"""

import networkx as nx
import matplotlib.pyplot as plt
from config import CLASS_NAMES

def visualize_graph(graph, ax, frame_number, pos, node_color_map):
    """
    Visualizes the detection graph showing connections between cameras and tracked objects
    with enhanced styling and information display.
    
    Args:
        graph (nx.Graph): NetworkX graph containing detection information
        ax (matplotlib.axes.Axes): Matplotlib axis for drawing
        frame_number (int): Current frame number for title
        pos (dict): Node positions for visualization
        node_color_map (dict): Mapping of nodes to their display colors
    """
    ax.clear()
    
    # Set a clean background with light grid
    ax.set_facecolor('#f8f8f8')
    ax.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
    
    # Check if graph has nodes before visualization
    if not graph.nodes():
        ax.set_title(f"Frame {frame_number} (No detections)", 
                   fontsize=14, fontweight='bold', pad=10, color='#555555')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.text(0, 0, "No objects detected across views", 
               ha='center', va='center', fontsize=12, color='#888888',
               bbox={'facecolor': 'white', 'alpha': 0.8, 'boxstyle': 'round,pad=0.5'})
        return
    
    # Group nodes by camera for better visualization
    camera_groups = {}
    for node in graph.nodes():
        cam = int(node.split("cam")[1].split("id")[0])
        if cam not in camera_groups:
            camera_groups[cam] = []
        camera_groups[cam].append(node)
    
    # Draw edges with enhanced styling
    # Use curved edges with alpha transparency for better visibility
    nx.draw_networkx_edges(
        graph, pos, ax=ax, 
        alpha=0.6, 
        width=2.0,
        edge_color='#444444',
        connectionstyle='arc3,rad=0.15',  # Curved edges
        style='solid'
    )
    
    # Draw nodes with colors from the 3D point mapping
    for cam, nodes in camera_groups.items():
        # Get node colors for this camera group
        node_colors = [node_color_map.get(node, (0.5, 0.5, 0.5)) for node in nodes]
        
        # Draw nodes with a black edge for better visibility
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=nodes,
            node_size=700,
            ax=ax,
            node_color=node_colors,
            edgecolors='black',
            linewidths=2.0,
            alpha=0.89
        )
    
    # Draw node labels with better styling above nodes
    # Extract node information for custom labels
    custom_labels = {}
    label_pos = {}  # Custom positions for labels (above nodes)
    
    for node in graph.nodes():
        cam = int(node.split("cam")[1].split("id")[0])
        obj_id = int(node.split("id")[1])
        name = int(graph.nodes[node]['name'])
        
        # Create custom label text
        if graph.nodes[node].get('name') is not None:
            class_name = f"{CLASS_NAMES.get(int(name), f'Class {int(name)}')}"
            custom_labels[node] = f"Cam {cam} - ID:{obj_id}\n{class_name}"
        else:
            custom_labels[node] = f"Cam {cam} - ID:{obj_id}"
        
        # Calculate position above the node (y-offset by 0.3 units)
        label_pos[node] = (pos[node][0], pos[node][1] + 0.3)
    
    # Draw labels with custom positioning
    for node, label_text in custom_labels.items():
        ax.text(
            label_pos[node][0],
            label_pos[node][1],
            label_text,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=9,
            fontweight='bold',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'black',
                'boxstyle': 'round,pad=0.3',
                'alpha': 0.9
            },
            zorder=100  # Ensure labels appear on top
        )
    
    # Auto-adjust axis limits with padding
    x_vals = [pos[node][0] for node in graph.nodes()]
    y_vals = [pos[node][1] for node in graph.nodes()]
    
    if x_vals and y_vals:  # Check if lists are not empty
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # Add padding (20% on each side)
        x_padding = 0.2 * (x_max - x_min) if x_max > x_min else 2
        y_padding = 0.2 * (y_max - y_min) if y_max > y_min else 2
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    
    # Set title and axis labels
    ax.set_title(
        f"Correspondence Graph - Frame {frame_number}",
        fontsize=14,
        fontweight='bold',
        pad=10,
        color='#333333'
    )
    
    # Add explanation text
    ax.text(
        0.5, 0.01, 
        f"Nodes: {len(graph.nodes)} detections | Connections: {len(graph.edges)} matches",
        ha='center',
        transform=ax.transAxes,
        fontsize=10,
        bbox={
            'facecolor': 'white', 
            'alpha': 0.7, 
            'pad': 0,
            'edgecolor': '#dddddd'
        }
    )
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(1)