import numpy as np
import open3d as o3d
import cv2
import json
from teaserpp_python import _teaserpp as tpp
import copy
import time
import glob
import os
from dataclasses import dataclass

@dataclass
class TemplateMetrics:
    template_idx: int
    num_correspondences: int
    num_inliers: int
    num_s_inliers: int
    num_t_inliers: int



def load_camera_intrinsics(scene_camera_path, frame_id, image_width, image_height):
    """
    Returns Open3D PinholeCameraIntrinsic object from BlenderProc camera data.
    """
    if isinstance(frame_id, int):
        frame_id = f"{frame_id}"

    with open(scene_camera_path, "r") as f:
        cam_data = json.load(f)

    if frame_id not in cam_data:
        raise ValueError(f"Frame ID {frame_id} not found in scene_camera.json")

    cam_K = cam_data[frame_id]["cam_K"]
    fx, fy = cam_K[0], cam_K[4]
    cx, cy = cam_K[2], cam_K[5]

    depth_scale = cam_data[frame_id]["depth_scale"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_width,
        height=image_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    return intrinsic, depth_scale

def get_pointcloud(depth_path, rgb_path, scene_camera_path, mask):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color_raw = cv2.imread(rgb_path)
    #mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    binary_mask = (mask == 255).astype(np.uint8)
    mask_pixels = np.sum(binary_mask)

    if mask_pixels == 0:
        print("WARNING: No pixels selected by mask!")
        return None
    
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    
    h, w = depth_raw.shape
    intrinsic, depth_scale = load_camera_intrinsics(scene_camera_path, 0, w, h)
    depth_raw = depth_raw * depth_scale
    
    masked_depth = np.where(binary_mask, depth_raw, 0.0)
    masked_color = np.where(binary_mask[:, :, None], color_raw, 0)

    valid_depth_mask = (masked_depth > 0.01) & (masked_depth < 10.0)  
    masked_depth = np.where(valid_depth_mask, masked_depth, 0.0)
    masked_color = np.where(valid_depth_mask[:, :, None], masked_color, 0)
    
    depth_o3d = o3d.geometry.Image(masked_depth)
    color_o3d = o3d.geometry.Image(masked_color)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Transform coordinate system
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    # Remove outliers
    if len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd


def uniform_downsample_farthest_point(pcd, target_points=2000):
    """
    Farthest Point Sampling (FPS) for uniform downsampling.
    Good spatial distribution, moderate computation time.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    print(f"Number of points: {n_points}, target: {target_points}")

    target_points = int(target_points)
    if n_points <= target_points:
        return pcd
    
    # Initialize with random point
    selected_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)
    
    for i in range(1, target_points):
        # Update distances to nearest selected point
        last_point = points[selected_indices[-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select point with maximum distance to any selected point
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
        
        # Set distance of selected point to 0
        distances[next_idx] = 0
    
    return pcd.select_by_index(selected_indices)



# Modified preprocessing function
def preprocess_point_cloud_uniform(pcd, target_points=500):
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
        
    print(f"Original points: {len(pcd.points)}")

    pcd_down = uniform_downsample_farthest_point(pcd, target_points)
    
    print(f"Downsampled points: {len(pcd_down.points)}")
    
    # Ensure we have enough points for feature computation
    if len(pcd_down.points) < 10:
        print("Too few points after downsampling!")
        return None, None
    
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
    
    print(f"FPFH feature dimension: {fpfh.dimension()}")
    return pcd_down, fpfh

def get_correspondences(pcd1_down, pcd2_down, fpfh1, fpfh2, distance_threshold=0.1):
    # Try different distance thresholds if needed
    for dist_thresh in [distance_threshold, distance_threshold*2, distance_threshold*0.5]:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd1_down, pcd2_down, fpfh1, fpfh2,
            mutual_filter=True,
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

def run_teaser(source, target, correspondences):
    if len(correspondences) < 3:
        return np.eye(4), []
    
    src_corr = np.array([source.points[i] for i, _ in correspondences]).T
    tgt_corr = np.array([target.points[j] for _, j in correspondences]).T
    
    params = tpp.RobustRegistrationSolver.Params()
    params.cbar2 = 1
    params.noise_bound = 0.001
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = tpp.RotationEstimationAlgorithm.GNC_TLS
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12
    
    solver = tpp.RobustRegistrationSolver(params)
    solver.solve(src_corr, tgt_corr)
    sol = solver.getSolution()
    
    R_inliers = solver.getRotationInliers()
    s_inliers = solver.getScaleInliers()
    t_inliers = solver.getTranslationInliers()
    R = np.array(sol.rotation)
    t = np.array(sol.translation).reshape((3, 1))
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    
    return T, R_inliers, s_inliers, t_inliers

def find_best_template_teaser(scene_pcd, template_pcds, target_points=500):
    scene_down, scene_fpfh = preprocess_point_cloud_uniform(scene_pcd, target_points)
    
    best_inliers = -1
    best_idx = -1
    best_transform = np.eye(4)
    
    all_metrics = []  # store metrics for all templates
    
    for idx, template_pcd in enumerate(template_pcds):
        template_down, template_fpfh = preprocess_point_cloud_uniform(template_pcd, target_points)
        if template_down is None:
            continue
        
        correspondences = get_correspondences(template_down, scene_down, template_fpfh, scene_fpfh)
        T, R_inliers, s_inliers, t_inliers = run_teaser(template_down, scene_down, correspondences)
        
        R_inlier_count = len(R_inliers) if R_inliers is not None else 0
        s_inlier_count = len(s_inliers) if R_inliers is not None else 0
        t_inlier_count = len(t_inliers) if R_inliers is not None else 0
        
        metrics = TemplateMetrics(
            template_idx=idx,
            num_correspondences=len(correspondences),
            num_inliers=R_inlier_count,
            num_s_inliers = s_inlier_count,
            num_t_inliers = t_inlier_count
        )
        all_metrics.append(metrics)
        
        if idx == 8:#R_inlier_count > best_inliers:
            best_inliers = R_inlier_count
            best_idx = idx
            best_transform = T
    
    return best_idx, best_transform, best_inliers, all_metrics

def try_manual_alignment(cad_pcd, scene_pcd, scale_ratio):
    """Try manual initial alignment based on centroids, scale, and PCA rotation"""
    print("\n=== TRYING MANUAL ALIGNMENT ===")

    cad_aligned = copy.deepcopy(cad_pcd)

    # 1. Center both point clouds
    cad_center = cad_aligned.get_center()
    scene_center = scene_pcd.get_center()
    cad_aligned.translate(-cad_center)
    cad_aligned.translate(scene_center)

    # 2. Scale
    if abs(scale_ratio - 1.0) > 0.1:
        print(f"Applying scale factor: {scale_ratio:.3f}")
        cad_aligned.scale(scale_ratio, center=scene_center)

 
    
    return cad_aligned

def center_pointcloud(pcd):
    """Center pointcloud at origin"""
    centered = pcd.translate(-pcd.get_center())
    return centered