import matplotlib.pyplot as plt  
import pandas as pd              
import rclpy                     # ROS 2 client library for Python
from rclpy.serialization import deserialize_message # Function to deserialize ROS messages
import argparse                  # Command-line argument parsing

# ROS 2 message types
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped

# rosbag2_py for reading bag files
from rosbag2_py import SequentialReader
from rosbag2_py import StorageOptions, ConverterOptions

class OdometryPlotter:
    """
    A utility class to read, process, and visualize odometry data from a ROS 2 bag file.

    This class opens a specified rosbag, reads messages from relevant topics,
    extracts positional data, plots the resulting trajectories, and saves
    the primary odometry data to a CSV file.

    Attributes:
        reader (SequentialReader): The rosbag reader instance.
        topic_type_map (dict): A mapping of topic names to their ROS message types.
        positions_x_odom (list): Stores the X-coordinates from the '/odrive/odom' topic.
        positions_y_odom (list): Stores the Y-coordinates from the '/odrive/odom' topic.
        timestamps_odom (list): Stores the timestamps from the '/odrive/odom' topic.
        positions_x_amcl (list): Stores the X-coordinates from the '/amcl_pose' topic.
        positions_y_amcl (list): Stores the Y-coordinates from the '/amcl_pose' topic.
        timestamps_amcl (list): Stores the timestamps from the '/amcl_pose' topic.
        t0 (int): The initial timestamp (in nanoseconds) used for normalization.
    """
    def __init__(self, bag_file_uri='odom_bag', output_csv='odometry_data.csv', show_plot=True):
        """
        Initializes the OdometryPlotter, sets up the bag reader, and runs the processing pipeline.

        Args:
            bag_file_uri (str): The path to the input rosbag directory.
            output_csv (str): The filename for the output CSV file.
            show_plot (bool): Whether to display the trajectory plot.
        """
        # --- Setup Bag Reader ---
        self.reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_file_uri, storage_id='sqlite3')
        converter_options = ConverterOptions('')
        self.reader.open(storage_options, converter_options)

        # --- Topic and Message Type Mapping ---
        # Define the topics of interest and their corresponding message types for deserialization.
        self.topic_type_map = {
            '/odrive/odom': Odometry,
            '/amcl_pose': PoseWithCovarianceStamped,
            '/teste': String, # A trigger topic to set the initial time
        }

        # --- Data Storage Initialization ---
        self.positions_x_odom = []
        self.positions_y_odom = []
        self.timestamps_odom = []

        self.positions_x_amcl = []
        self.positions_y_amcl = []
        self.timestamps_amcl = []

        # --- Timestamp Normalization ---
        self.tick = False # Flag to check if the initial time has been set
        self.t0 = 0       # Initial timestamp for normalization

        # --- Configuration ---
        self.show_plot = show_plot

        # --- Execution Workflow ---
        # The entire process is executed upon instantiation.
        self.process_messages()
        rclpy.shutdown()
        if self.show_plot:
            self.plot_trajectories()
        self.save_to_csv(output_csv)

    def process_messages(self):
        """
        Iterates through the rosbag, deserializes messages, and dispatches them for processing.
        """
        while self.reader.has_next():
            # Read the next message from the bag
            topic, serialized_msg, time_stamp = self.reader.read_next()

            if topic in self.topic_type_map:
                # Get the expected message type for the current topic
                msg_type = self.topic_type_map[topic]
                # Convert the raw serialized data into a ROS message object
                msg = deserialize_message(serialized_msg, msg_type)
                # Process the deserialized message
                self.handle_message(msg, topic, time_stamp)
            else:
                # This could be logged to a file in a production environment
                print(f"Warning: Skipping unknown topic: {topic}")

    def handle_message(self, msg, topic, time_stamp):
        """
        Processes a deserialized message based on its type and topic.

        Args:
            msg (object): The deserialized ROS message.
            topic (str): The topic from which the message originated.
            time_stamp (int): The message timestamp in nanoseconds.
        """
        # Handle Odometry messages
        if isinstance(msg, Odometry):
            position = msg.pose.pose.position
            print(f"[{topic}] Position: x={position.x:.3f}, y={position.y:.3f} TS: {self.normalize_time(self.t0, time_stamp):.3f}s")
            
            if topic == '/odrive/odom':
                self.positions_x_odom.append(position.x)
                self.positions_y_odom.append(position.y)
                self.timestamps_odom.append(time_stamp)

        # Handle PoseWithCovarianceStamped messages (typically from AMCL)
        elif isinstance(msg, PoseWithCovarianceStamped):
            position = msg.pose.pose.position
            # Example of selectively printing for debugging
            # print(f"[{topic}] Position: x={position.x:.3f}, y={position.y:.3f} TS: {self.normalize_time(self.t0, time_stamp):.3f}s")

            if topic == '/amcl_pose':
                self.positions_x_amcl.append(position.x)
                self.positions_y_amcl.append(position.y)
                self.timestamps_amcl.append(time_stamp)
        
        # Handle String messages, used here as a time normalization trigger
        elif isinstance(msg, String):
            # Set the initial time `t0` on the first received message from this topic.
            if not self.tick:
                self.t0 = time_stamp
                self.tick = True
            print(f"[{topic}] \t Message: '{msg.data}'  TS: {self.normalize_time(self.t0, time_stamp):.3f}s")

        else:
            print(f"Warning: Unhandled message type on topic {topic}: {type(msg)}")
    
    def normalize_time(self, t0, current_time):
        """
        Calculates the elapsed time in seconds from a start time.

        Args:
            t0 (int): The initial time in nanoseconds.
            current_time (int): The current time in nanoseconds.

        Returns:
            float: The elapsed time in seconds.
        """
        # Convert nanoseconds to seconds for readability
        return (current_time - t0) * 1e-9

    def plot_trajectories(self):
        """
        Generates and displays a 2D plot of the robot's trajectories.
        """
        plt.figure(figsize=(10, 8))

        # Plot the primary odometry path if data exists
        if self.positions_x_odom and self.positions_y_odom:
            plt.plot(self.positions_x_odom, self.positions_y_odom, marker='o', markersize=3, linestyle='-', label='Odometry (/odrive/odom)')

        # Plot the AMCL pose path if data exists
        if self.positions_x_amcl and self.positions_y_amcl:
            plt.plot(self.positions_x_amcl, self.positions_y_amcl, marker='x', markersize=3, linestyle='--', label='Localization (/amcl_pose)')

        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title('Robot Trajectory Comparison')
        plt.legend()
        plt.grid(True)
        # Ensure aspect ratio is equal to prevent path distortion
        plt.axis('equal')
        plt.show()
    
    def save_to_csv(self, filename):
        """
        Saves the processed odometry data to a CSV file using pandas.

        Note: This implementation only saves the '/odrive/odom' data. It could be
        extended to handle multiple data sources.

        Args:
            filename (str): The path for the output CSV file.
        """
        # Check if there is data to save
        if not self.timestamps_odom:
            print("No odometry data to save.")
            return

        # Create a dictionary to hold the data
        data = {
            'timestamp_sec': [self.normalize_time(self.t0, ts) for ts in self.timestamps_odom],
            'x_odom': self.positions_x_odom,
            'y_odom': self.positions_y_odom,
        }
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file without the index column
        df.to_csv(filename, index=False)
        print(f"Odometry data successfully saved to '{filename}'")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract and visualize odometry data from ROS 2 bag files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rosbag_odometry_extractor.py
  python rosbag_odometry_extractor.py --input my_bag --output my_data.csv
  python rosbag_odometry_extractor.py --input /path/to/bag --output /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='odom_bag',
        help='Path to the input ROS 2 bag file directory (default: odom_bag)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='odometry_data.csv',
        help='Path for the output CSV file (default: odometry_data.csv)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting and only extract data to CSV'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Display configuration
    print("ROS 2 Bag Odometry Extractor")
    print("=" * 40)
    print(f"Input bag file: {args.input}")
    print(f"Output CSV file: {args.output}")
    print(f"Show plots: {'No' if args.no_plot else 'Yes'}")
    print("=" * 40)
    
    # Initialize the ROS 2 client library. This is necessary for using any rclpy functionality.
    rclpy.init()
    
    try:
        # Instantiate the plotter, which triggers the entire read-process-plot-save workflow.
        odometry_plotter = OdometryPlotter(
            bag_file_uri=args.input, 
            output_csv=args.output,
            show_plot=not args.no_plot
        )
        print("\nExtraction completed successfully!")
    except Exception as e:
        print(f"\nError during extraction: {e}")
        print("Please check that the bag file exists and is accessible.")
    finally:
        rclpy.shutdown()