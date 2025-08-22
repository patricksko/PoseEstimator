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


def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 

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

    
    # Remove outliers
    if len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    
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




import copy
def visualize_correspondences(src_down, dst_down, correspondences, offset=(0.1,0,0)):
    """
    Visualize correspondences between two point clouds in Open3D.
    """
    src_temp = copy.deepcopy(src_down)
    dst_temp = copy.deepcopy(dst_down)

    src_temp = src_temp.translate((0, 0, 0))
    dst_temp = dst_temp.translate(offset)  # Shift for visibility

    # Create lines connecting corresponding points
    lines = []
    colors = []
    for (src_idx, dst_idx) in correspondences:
        lines.append([src_idx, len(src_down.points) + dst_idx])
        colors.append([0, 0, 1])  # blue lines

    # Merge both clouds into a single point cloud for LineSet indexing
    merged_points = np.vstack((np.asarray(src_temp.points), np.asarray(dst_temp.points)))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(merged_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    src_temp.paint_uniform_color([1, 0, 0])
    dst_temp.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([src_temp, dst_temp, line_set])



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
def chamfer_distance(src, dst):
    """Compute symmetric Chamfer distance between two Open3D clouds"""
    src_pts = np.asarray(src.points)
    dst_pts = np.asarray(dst.points)
    src_kd = o3d.geometry.KDTreeFlann(dst)
    dst_kd = o3d.geometry.KDTreeFlann(src)

    d_src = []
    for p in src_pts:
        _, _, d = src_kd.search_knn_vector_3d(p, 1)
        d_src.append(np.sqrt(d[0]))
    d_dst = []
    for p in dst_pts:
        _, _, d = dst_kd.search_knn_vector_3d(p, 1)
        d_dst.append(np.sqrt(d[0]))
    return np.mean(d_src) + np.mean(d_dst)

def is_y_up_camera(H, tol_deg=20):
    """
    Check if model's +Y axis points upwards relative to camera frame.
    In camera convention: +Y is down, so "up" = [0, -1, 0].
    """
    cam_up = [0,1,0]
    R = H[:3, :3]
    R_inv = np.linalg.inv(R)
    model_y_cam = R_inv[:, 1] / (np.linalg.norm(R_inv[:, 1]))
    cosang = float(np.clip(np.dot(model_y_cam, cam_up), -1.0, 1.0))
    angle = np.degrees(np.arccos(cosang))
    print("="*100)
    return angle <= tol_deg, angle

def find_best_template_teaser(dst_cloud, src_clouds, target_points=100):
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, target_points)
    
    best_score = float("inf")
    best_idx = -1
    best_transform = np.eye(4)
    all_metrics = []
    tol_deg = 20
    
    for idx, src_cloud in enumerate(src_clouds):
        # o3d.visualization.draw_geometries([
        #     src_cloud.paint_uniform_color([0, 1, 1]),
        # ], window_name="Template")
        src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, target_points)
        if src_down is None:
            continue
        
        correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh)
        H, R_inliers, s_inliers, t_inliers = run_teaser(src_down, dst_down, correspondences)
        
        # # # --- NEW: ICP refinement ---
        # icp_result = o3d.pipelines.registration.registration_icp(
        #     src_cloud, dst_cloud, 0.01, H,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # )
        # refined_transform = icp_result.transformation
        
        src_aligned = copy.deepcopy(src_cloud).transform(H)
        score = chamfer_distance(src_aligned, dst_cloud)
        # Check Y-up physics constraint
        #y_ok, angle = is_y_up_camera(H, tol_deg=tol_deg)
        metrics = {
            "template_idx": idx,
            "num_corr": len(correspondences),
            "num_inliers": len(R_inliers or []),
            # "icp_fitness": icp_result.fitness,
            # "icp_rmse": icp_result.inlier_rmse,
            "chamfer": score
        }
        all_metrics.append(metrics)
        
        if score < best_score:  
            best_score = score
            best_idx = idx
            best_transform = H
    
    return best_idx, best_transform, best_score, all_metrics


def crop_pointcloud_fov(pcd, fov_deg=60.0, look_dir=np.array([0, 0, 1])):
    """
    Crops a point cloud to simulate a camera FoV.
    - pcd: Open3D PointCloud
    - fov_deg: field of view in degrees
    - look_dir: viewing direction vector (camera looks along this)
    """
    points = np.asarray(pcd.points)
    look_dir = look_dir / np.linalg.norm(look_dir)
    fov_rad = np.deg2rad(fov_deg)

    # Compute angles between look_dir and each point vector
    norms = np.linalg.norm(points, axis=1)
    norms[norms == 0] = 1e-8
    dirs = points / norms[:, None]
    cos_angles = np.dot(dirs, look_dir)
    mask = cos_angles > np.cos(fov_rad / 2)

    return pcd.select_by_index(np.where(mask)[0])

def crop_pointcloud_fov_hpr(pcd, camera_position=np.array([0, 0, -1]),
                            camera_lookat=np.array([0, 0, 0]),
                            fov_deg=60.0):
    """
    Simulates a camera FoV with hidden point removal.
    Returns only points visible from the given camera position and direction.
    
    Parameters:
    - pcd: Open3D PointCloud
    - camera_position: 3D position of the camera
    - camera_lookat: target point the camera is looking at
    - fov_deg: field of view (degrees)
    """
    # View direction
    cam_dir = camera_lookat - camera_position
    cam_dir /= np.linalg.norm(cam_dir)

    # First, remove points outside the angular FoV
    points = np.asarray(pcd.points)
    vecs = points - camera_position
    vecs_norm = np.linalg.norm(vecs, axis=1)
    vecs_norm[vecs_norm == 0] = 1e-8
    dirs = vecs / vecs_norm[:, None]

    cos_angles = np.dot(dirs, cam_dir)
    mask_fov = cos_angles > np.cos(np.deg2rad(fov_deg) / 2)
    pcd_fov = pcd.select_by_index(np.where(mask_fov)[0])
    o3d.visualization.draw_geometries([
        pcd_fov.paint_uniform_color([0, 1, 1]),
        #dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="FOV")

    # Now apply Hidden Point Removal to simulate occlusion
    radius = np.linalg.norm(np.asarray(pcd_fov.points) - camera_position, axis=1).max() * 3
    _, pt_map = pcd_fov.hidden_point_removal(camera_position, radius)
    pcd_visible = pcd_fov.select_by_index(pt_map)
    o3d.visualization.draw_geometries([
            pcd_visible.paint_uniform_color([0, 1, 1]),
            #dst_cloud.paint_uniform_color([0, 1, 0])
        ], window_name="Visible")
    return pcd_visible


def apply_transform_and_noise(pcd, R, t, noise_sigma=0.0):
    """
    Applies rigid transform and optional Gaussian noise to a point cloud.
    - R: 3x3 rotation matrix
    - t: 3-element translation vector
    - noise_sigma: standard deviation of Gaussian noise
    """
    transformed = copy.deepcopy(pcd)
    pts = np.asarray(transformed.points)
    pts = (R @ pts.T).T + t
    if noise_sigma > 0:
        pts += np.random.normal(scale=noise_sigma, size=pts.shape)
    transformed.points = o3d.utility.Vector3dVector(pts)
    return transformed


def sample_pointcloud_with_noise(mesh, num_points=5000, jitter_sigma=0.0):
    """
    Samples a point cloud from a mesh with optional jitter noise per point.
    - mesh: Open3D TriangleMesh
    - num_points: number of points to sample
    - jitter_sigma: standard deviation for Gaussian noise
    """
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    if jitter_sigma > 0:
        pts = np.asarray(pcd.points)
        pts += np.random.normal(scale=jitter_sigma, size=pts.shape)
        pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def pyrender_to_open3d(pose_pyrender):
    """
    Convert a 4x4 camera pose from pyrender (X-right, Y-forward, Z-up)
    to Open3D coordinates (X-right, Y-up, Z-forward).
    """
    # Rotation to swap Y and Z axes
    R_swap = np.array([[-1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])
    
    T_new = np.eye(4)
    T_new[:3, :3] = R_swap @ pose_pyrender[:3, :3]
    T_new[:3, 3] = R_swap @ pose_pyrender[:3, 3]
    return T_new