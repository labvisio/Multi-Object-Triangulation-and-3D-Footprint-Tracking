import random

import numpy as np


def euclidean_to_homogeneous(points):
    """
    Converts a set of points from Euclidean coordinates to homogeneous coordinates.
    
    Homogeneous coordinates are useful in computer vision and graphics as they allow
    for translation and projection transformations using matrix multiplication.
    
    Parameters:
    -----------
    points : ndarray
        A 2D numpy array of shape (n_points, n_dimensions), where each row represents a point.
    
    Returns:
    --------
    ndarray
        A 2D numpy array of the same shape as input, with an additional dimension for homogeneous coordinates.
    """
    return np.hstack([points, np.ones((len(points), 1))])

def homogeneous_to_euclidean(points):
    """
    Converts a set of points from homogeneous coordinates to Euclidean coordinates.
    
    This function divides each element of the homogeneous coordinate vector by its last element 
    to convert it back to Euclidean space.
    
    Parameters:
    -----------
    points : ndarray
        A 2D numpy array of shape (n_points, n_dimensions + 1), where each row is a point in homogeneous coordinates.
    
    Returns:
    --------
    ndarray
        A 2D numpy array of points in Euclidean space.
    """
    return (points.T[:-1] / points.T[-1]).T

def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """
    Projects 3D points onto a 2D image plane using a given projection matrix without distortion.
    
    This function converts the input points into homogeneous coordinates, applies the projection matrix,
    and optionally converts them back into Euclidean coordinates (2D).
    
    Parameters:
    -----------
    proj_matrix : ndarray
        A 3x4 projection matrix that maps 3D points to 2D points.
    points_3d : ndarray
        A 2D numpy array of shape (n_points, 3) representing 3D points.
    convert_back_to_euclidean : bool, optional
        Whether to convert the resulting homogeneous points back to Euclidean coordinates (default is True).
    
    Returns:
    --------
    ndarray
        A 2D numpy array representing the projected 2D points.
    """
    result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
    if convert_back_to_euclidean:
        result = homogeneous_to_euclidean(result)
    return result

def triangulate_point_from_multiple_views_linear(proj_matricies, points):
    """
    Performs linear triangulation to estimate the 3D point from multiple views.
    
    Given 2D projections of a point in multiple views, and the corresponding projection matrices, 
    this function estimates the 3D point using the Direct Linear Transformation (DLT) algorithm.
    
    Parameters:
    -----------
    proj_matricies : list of ndarray
        A list of projection matrices (3x4), one for each view/camera.
    points : list of ndarray
        A list of 2D points (in image coordinates) corresponding to the 3D point, one for each view.
    
    Returns:
    --------
    ndarray
        The estimated 3D point in Euclidean coordinates.
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    
    # Constructing the linear system from the projection matrices and 2D points
    for j, _ in enumerate(proj_matricies):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    # Solve the system using Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]
    
    # Convert the solution from homogeneous to Euclidean coordinates
    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d

def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    """
    Calculates the reprojection error matrix for given 3D keypoints and their 2D projections.
    
    Reprojection error measures how well the 3D point reprojects back onto the image plane.
    This function computes the difference between actual 2D points and the 2D points projected 
    from the 3D points.
    
    Parameters:
    -----------
    keypoints_3d : ndarray
        A 2D numpy array of shape (n_points, 3) representing 3D points.
    keypoints_2d_list : list of ndarray
        A list of 2D points, one array per camera view, representing the observed 2D keypoints.
    proj_matricies : list of ndarray
        A list of projection matrices (3x4) for each camera view.
    
    Returns:
    --------
    ndarray
        A 2D numpy array containing the reprojection error for each point in each view.
    """
    reprojection_error_matrix = []
    
    # Project each 3D point onto the corresponding 2D image and calculate reprojection error
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(
            proj_matrix, keypoints_3d
        )
        reprojection_error = (
            1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        )
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T

def triangulate_ransac(proj_matrices, points_2d, n_iters=100, reprojection_error_epsilon=50):
    assert len(proj_matrices) == len(points_2d)
    assert len(points_2d) >= 2

    proj_matrices = np.array(proj_matrices)
    points_2d = np.array(points_2d)

    n_views = len(points_2d)
    view_set = list(range(n_views))  # Ensure view_set is a list
    inlier_set = set()

    # Debugging: check what view_set is
    # print("View Set:", view_set)

    for i in range(n_iters):
        sampled_views = sorted(random.sample(view_set, 2))  # Ensure random.sample works with a list

        keypoint_3d_in_base_camera = triangulate_point_from_multiple_views_linear(
            proj_matrices[sampled_views], points_2d[sampled_views]
        )

        reprojection_error_vector = calc_reprojection_error_matrix(
            np.array([keypoint_3d_in_base_camera]), points_2d, proj_matrices
        )[0]

        new_inlier_set = set(sampled_views)
        for view in view_set:
            current_reprojection_error = reprojection_error_vector[view]
            if current_reprojection_error < reprojection_error_epsilon:
                new_inlier_set.add(view)

        if len(new_inlier_set) > len(inlier_set):
            inlier_set = new_inlier_set

    if len(inlier_set) == 0:
        inlier_set = set(view_set)

    inlier_list = np.array(sorted(inlier_set))
    inlier_proj_matrices = proj_matrices[inlier_list]
    inlier_points = points_2d[inlier_list]

    keypoint_3d_in_base_camera = triangulate_point_from_multiple_views_linear(
        inlier_proj_matrices, inlier_points
    )

    reprojection_error_vector = calc_reprojection_error_matrix(
        np.array([keypoint_3d_in_base_camera]), inlier_points, inlier_proj_matrices
    )[0]
    reprojection_error_mean = np.mean(reprojection_error_vector)

    return keypoint_3d_in_base_camera, inlier_list

