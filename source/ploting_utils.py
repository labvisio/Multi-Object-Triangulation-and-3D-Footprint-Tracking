import colorsys
import random
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import logging


class Utils:
    """
    Utility class for handling color management and plot visualization.
    Provides methods for color conversion, plot updates, and figure creation.
    """

    def __init__(self):
        """
        Initialize the Utils class with an empty color dictionary.
        """
        logging.info("--------- Utils initialized ---------")
        self.COLORS = {}  # Dictionary to store ID-color mappings

    def id_to_rgb_color(self, id: int) -> tuple[int, int, int]:
        """
        Generates a unique RGB color for a given ID. If the ID is already registered,
        returns the associated color.

        Args:
            id (int): A unique identifier for color generation.

        Returns:
            tuple[int, int, int]: RGB color normalized in range (0-255).

        Note:
            Colors are generated using HSL color space for better consistency
            and stored in the COLORS dictionary for reuse.
        """
        if id not in self.COLORS:
            random.seed(int(id))
            hue = random.uniform(0, 1)  # Random hue between 0 and 1
            saturation = 0.8  # Fixed saturation for consistent color vibrancy
            luminance = 0.6  # Fixed luminance for consistent brightness

            # Convert HLS to RGB (0-255 range)
            r, g, b = [
                int(x * 255) for x in colorsys.hls_to_rgb(hue, luminance, saturation)
            ]
            self.COLORS[id] = (r, g, b)
        return self.COLORS[id]

    def fig_to_image(self, fig):
        """
        Converts a Matplotlib figure to an OpenCV-compatible image.

        Args:
            fig: Matplotlib figure object

        Returns:
            numpy.ndarray: Image array in RGB format (height, width, 3)
        """
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        w, h = canvas.get_width_height()
        return buf.reshape(h, w, 3)

    def normalize_rgb_color(
        self, color: tuple[int, int, int]
    ) -> tuple[float, float, float]:
        """
        Converts RGB color from (0-255) to (0-1) format for Matplotlib.

        Args:
            color (tuple[int, int, int]): RGB color in 0-255 range

        Returns:
            tuple[float, float, float]: Normalized RGB color in 0-1 range
        """
        return tuple(channel / 255.0 for channel in color)

    def update_3d_plot(self, keypoint_list, ids, ax_3d):
        """
        Updates a 3D scatter plot with keypoints for multiple objects.

        Args:
            keypoint_list: List of 3D keypoints for each object
            ids: List of object IDs
            ax_3d: Matplotlib 3D axis object
        """
        ax_3d.clear()
        print("Keypoints list:  ", keypoint_list)
        for keypoints_3d, obj_id in zip(keypoint_list, ids):
            color_rgb = self.normalize_rgb_color(self.id_to_rgb_color(obj_id))
            ax_3d.scatter(
                keypoints_3d[0],
                keypoints_3d[1],
                keypoints_3d[2],
                c=[color_rgb],
                s=50,
                label=f"ID: {obj_id}",
            )

        # Set plot parameters
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_xlim([-4, 4])
        ax_3d.set_ylim([-4, 4])
        ax_3d.set_zlim([0, 4])
        ax_3d.legend(loc="upper left")  # Mover legenda para canto superior esquerdo

    def rgb_to_bgr(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
        """
        Converts RGB color to BGR format for OpenCV compatibility.

        Args:
            color (tuple[int, int, int]): RGB color tuple

        Returns:
            tuple[int, int, int]: BGR color tuple
        """
        return color[2], color[1], color[0]

    def update_2d_plot(self, keypoint_list, ids, ax_2d):
        """
        Updates a 2D scatter plot with keypoints for multiple objects.

        Args:
            keypoint_list: List of keypoints for each object
            ids: List of object IDs
            ax_2d: Matplotlib 2D axis object
        """
        ax_2d.clear()
        for keypoints_3d, obj_id in zip(keypoint_list, ids):
            color_rgb = self.normalize_rgb_color(self.id_to_rgb_color(obj_id))
            ax_2d.scatter(
                keypoints_3d[0],
                keypoints_3d[1],
                c=[color_rgb],
                label=f"ID: {obj_id}",
            )

        # Set plot parameters
        ax_2d.set_xlabel("X")
        ax_2d.set_ylabel("Y")
        ax_2d.set_xlim([-4, 4])
        ax_2d.set_ylim([-4, 4])
        ax_2d.legend(loc="upper right")

    def create_plt_figure(self):
        """
        Creates a new figure with 3D and 2D subplots.

        Returns:
            tuple: (figure object, 3D axis object, 2D axis object)
        """
        fig = plt.figure(figsize=(10, 5))
        ax_3d = fig.add_subplot(121, projection="3d")
        ax_2d = fig.add_subplot(122)
        return fig, ax_3d, ax_2d