import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from teaserpp_python import _teaserpp as tpp
import copy
import json
import os
import glob
import time
import pyrealsense2 as rs 
from dataclasses import dataclass


def enforce_upright_pose_y_up(T):
    """
    Align model's local +Y (stud direction) with world +Y (up).
    Handles arbitrary tilt (not just upside-down).
    """
    T = np.array(T, copy=True)
    R = T[:3, :3]

    # current up vector in world coords
    up_local = R[:, 1]
    world_up = np.array([0, 1, 0], dtype=np.float64)

    # if already mostly aligned, skip
    cos_angle = np.dot(up_local, world_up) / (np.linalg.norm(up_local) + 1e-8)
    if abs(cos_angle) > 0.999:
        return T  # already upright

    # rotation axis = cross product between current up and desired up
    axis = np.cross(up_local, world_up)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # up_local is opposite to world_up → rotate 180° around X
        axis = np.array([1, 0, 0])
        angle = np.pi
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # Rodrigues' rotation formula to get correction rotation
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R_correction = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

    # Apply correction
    R_fixed = R_correction @ R
    T[:3, :3] = R_fixed
    return T

def uniform_downsample_farthest_point(pcd, target_points=500):
    """
    Farthest Point Sampling (FPS) for uniform downsampling.
    Good spatial distribution, moderate computation time.
    """
    points = np.asarray(pcd.points)
    idx = np.random.choice(len(points), size=target_points, replace=False)
    return pcd.select_by_index(idx)



# Modified preprocessing function
def preprocess_point_cloud_uniform(pcd, target_points=500, calc_fpfh=True):
    """
    Preprocess point cloud with uniform downsampling to exactly target_points.
    
    Methods:
    - 'random': Fast random sampling
    - 'grid': Grid-based uniform sampling
    - 'kmeans': K-means clustering (most uniform, slowest)
    - 'farthest_point': Farthest Point Sampling (good balance)
    - 'adaptive_voxel': Adaptive voxel size (fastest, approximate)
    """
    if len(pcd.points) == 0:
        print("Empty point cloud!")
        return None, None
    pcd_down = uniform_downsample_farthest_point(pcd, target_points)
    
    # Ensure we have enough points for feature computation
    if len(pcd_down.points) < 10:
        print("Too few points after downsampling!")
        return None, None
    if calc_fpfh:
        # Estimate normals with adaptive parameters
        nn_param = min(30, len(pcd_down.points) // 2)  # Adaptive neighbor count
        radius = 0.05  # You might want to adjust this based on your data scale
        
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn_param)
        )
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2.5, max_nn=100)
        )
    else:
        fpfh = None
    

    return pcd_down, fpfh


def run_teaser(source, target, correspondences, noise_bound_m=0.001):
    if len(correspondences) < 3:
        return np.eye(4), [], [], []

    src_corr = np.array([source.points[i] for i, _ in correspondences]).T  # 3xN
    tgt_corr = np.array([target.points[j] for _, j in correspondences]).T  # 3xN

    params = tpp.RobustRegistrationSolver.Params()
    params.cbar2 = 1
    params.noise_bound = float(noise_bound_m)
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = tpp.RotationEstimationAlgorithm.GNC_TLS
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12

    solver = tpp.RobustRegistrationSolver(params)
    solver.solve(src_corr, tgt_corr)
    sol = solver.getSolution()

    R = np.asarray(sol.rotation)
    t = np.asarray(sol.translation).reshape((3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T

def chamfer_distance(src, dst):
    d1 = src.compute_point_cloud_distance(dst)
    d2 = dst.compute_point_cloud_distance(src)
    # symmetric mean
    return float(np.mean(d1) + np.mean(d2))


def cloud_resolution(pcd, k=8):
    pts = np.asarray(pcd.points)
    if len(pts) < 2:
        return 0.005
    kd = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i, p in enumerate(pts[::max(1, len(pts)//5000)]):  # subsample for speed
        _, idx, dd = kd.search_knn_vector_3d(p, min(k+1, len(pts)))
        # skip self (dd[0] == 0)
        if len(dd) > 1:
            dists.extend(np.sqrt(dd[1:]))
    return float(np.median(dists)) if dists else 0.005


def get_correspondences(pcd1_down, pcd2_down, fpfh1, fpfh2, distance_threshold=0.1):
    # Try different distance thresholds if needed
    for dist_thresh in [distance_threshold, distance_threshold*2, distance_threshold*0.5]:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd1_down, pcd2_down, fpfh1, fpfh2,
            mutual_filter=False,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,  # Reduced from 4
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
        )
        
        if len(result.correspondence_set) >= 3:  # Minimum for registration
            return result.correspondence_set

    return result.correspondence_set


def camera_eye_lookat_up_from_H(H):
    """
    Convert model→camera H to eye/target/up in model/world coords.
    - eye: camera center
    - target: eye + forward vector
    - up: image up
    """
    R = H[:3, :3]
    t = H[:3, 3]

    # camera center in world
    eye = -R.T @ t

    # camera axes in world
    forward = R.T @ np.array([0, 0, 1.0])   # camera +Z
    up      = R.T @ np.array([0, -1.0, 0.0])  # screen up

    target = eye + forward
    up = up / (np.linalg.norm(up) + 1e-12)

    
    return eye, target, up




def project_points(points_3d, K, T_m2c):
    """Project 3D points into image pixels."""
    pts_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Nx4
    pts_cam = (T_m2c @ pts_h.T).T[:, :3]  # Nx3 in camera frame
    
    # Filter only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    
    # Project
    uv = (K @ (pts_cam.T)).T  # Nx3
    uv = uv[:, :2] / uv[:, 2:3]
    return uv.astype(int)
