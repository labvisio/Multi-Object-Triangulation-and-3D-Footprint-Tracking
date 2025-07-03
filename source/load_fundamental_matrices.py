import json

import numpy as np


class FundamentalMatrices:
    """
    A class to handle camera calibration and compute the fundamental matrices
    between multiple cameras.

    Methods:
        _load_camera_parameters(calibration):
            Loads intrinsic and extrinsic camera parameters from a calibration file.

        _calculate_fundamental_matrix(K1, K2, TF1, TF2):
            Computes the fundamental matrix between two cameras given their parameters.

        fundamental_matrices_all(camera_files):
            Computes and returns the fundamental matrices for all pairs of cameras.
    """

    def load_intrinsics_and_distortion(self, config_file):
        data = json.load(open(config_file))

        # Extrair matriz intrínseca K
        K = np.array(data["intrinsic"]["doubles"]).reshape(3, 3)

        # Extrair vetor de distorção
        dist = np.array(data["distortion"]["doubles"])

        # Extrair parâmetros extrínsecos (R, T)
        extrinsic_data = data["extrinsic"][0]["tf"]["doubles"]
        extrinsic_matrix = np.array(extrinsic_data).reshape(4, 4)
        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3]

        return K, dist, R, T

    def _load_camera_parameters(self, calibration: str):
        """
        Loads camera intrinsic and extrinsic parameters from a JSON calibration file.

        Args:
            calibration (str): The path to the camera calibration file in JSON format.

        Returns:
            tuple: A tuple containing the transformation matrix (TF), rotation matrix (R),
            translation vector (T), and intrinsic matrix (K) for the camera.
        """
        # Load camera parameters from the calibration file (JSON format)
        camera_data = json.load(open(calibration))

        # Intrinsic matrix (3x3)
        K = np.array(camera_data["intrinsic"]["doubles"]).reshape(3, 3)

        # Image resolution (width, height)
        res = [camera_data["resolution"]["width"], camera_data["resolution"]["height"]]

        # Transformation matrix (4x4) containing rotation and translation
        tf = np.array(camera_data["extrinsic"][0]["tf"]["doubles"]).reshape(4, 4)

        # Rotation matrix (3x3)
        R = tf[:3, :3]

        # Translation vector (3x1)
        T = tf[:3, 3].reshape(3, 1)

        # Distortion coefficients are ignored here but could be added if needed
        # dis = np.array(camera_data["distortion"]["doubles"]) # Not used in this code

        return tf, R, T, K

    def _calculate_fundamental_matrix(
        self, K1: np.ndarray, K2: np.ndarray, TF1: np.ndarray, TF2: np.ndarray
    ):
        """
        Computes the fundamental matrix between two cameras.

        Args:
            K1 (np.ndarray): Intrinsic matrix of the first camera.
            K2 (np.ndarray): Intrinsic matrix of the second camera.
            TF1 (np.ndarray): Transformation matrix (4x4) of the first camera.
            TF2 (np.ndarray): Transformation matrix (4x4) of the second camera.

        Returns:
            np.ndarray: The fundamental matrix (3x3) between the two cameras.
        """
        # Compute the relative transformation matrix between the two cameras
        tF_2_1 = TF2 @ np.linalg.inv(TF1)

        # Extract the rotation and translation between the two cameras
        R = tF_2_1[:3, :3]
        T = tF_2_1[:3, 3]

        # Compute the skew-symmetric matrix of the translation vector
        t_hat = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])

        # Compute the essential matrix: E = T_hat * R
        essential_matrix = t_hat @ R

        # Compute the fundamental matrix: F = inv(K2)^T * E * inv(K1)
        F = (np.linalg.inv(K2).T) @ essential_matrix @ (np.linalg.inv(K1))

        return F

    def _calculate_projection_matrix(self, K: np.ndarray, T: np.ndarray):
        """
        Computes the projection matrix for a camera given its intrinsic and extrinsic parameters.

        Args:
            K (np.ndarray): Intrinsic matrix of the camera.
            T (np.ndarray): Translation vector of the camera.

        Returns:
            np.ndarray: The projection matrix (3x4) of the camera.
        """
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

        return K @ (P @ T)

    def projection_matrices_all(self, camera_files: list):
        """
        Calculates the projection matrices for all cameras.

        Args:
            camera_files (list): A list of file paths to camera calibration files (JSON format).

        Returns:
            dict: A dictionary where the keys are camera indices, and the values
            are the corresponding projection matrices for each camera.
        """
        # Dictionary to store the projection matrices for all cameras
        P_all = {}

        # Loop over all cameras
        for C in range(len(camera_files)):
            # Load the camera parameters for the camera (C)
            tf, R, T, K = self._load_camera_parameters(camera_files[C])

            # Calculate the projection matrix for the camera
            P = self._calculate_projection_matrix(K, tf)

            # Store the projection matrix in the dictionary
            P_all[C] = P

        return P_all

    def fundamental_matrices_all(self, camera_files: list):
        """
        Calculates the fundamental matrices between all pairs of cameras.

        Args:
            camera_files (list): A list of file paths to camera calibration files (JSON format).

        Returns:
            dict: A nested dictionary where the keys are camera indices, and the values
            are the corresponding fundamental matrices between each pair of cameras.
        """
        # Dictionary to store the fundamental matrices for all camera pairs
        F_all = {}

        # Loop over all camera pairs
        for C_s in range(len(camera_files)):
            for C_d in range(len(camera_files)):
                if (
                    C_s == C_d
                ):  # Skip pairs where the source and destination cameras are the same
                    continue

                # Load the camera parameters for the source camera (C_s) and destination camera (C_d)
                tf1, R1, T1, K1 = self._load_camera_parameters(camera_files[C_s])
                tf2, R2, T2, K2 = self._load_camera_parameters(camera_files[C_d])

                # Calculate the fundamental matrix between the source and destination cameras
                F = self._calculate_fundamental_matrix(K1, K2, tf1, tf2)

                # Store the fundamental matrix in a nested dictionary
                if C_s not in F_all:
                    F_all[C_s] = {}
                F_all[C_s][C_d] = F

        return F_all
