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

def get_angular_error(R_exp, R_est):
    """Calculate angular error in radians"""
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))


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

    R_inliers = list(solver.getRotationInliers() or [])
    s_inliers = list(solver.getScaleInliers() or [])
    t_inliers = list(solver.getTranslationInliers() or [])

    R = np.asarray(sol.rotation)
    t = np.asarray(sol.translation).reshape((3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T, R_inliers, s_inliers, t_inliers


def chamfer_distance(src, dst):
    d1 = src.compute_point_cloud_distance(dst)
    d2 = dst.compute_point_cloud_distance(src)
    # symmetric mean
    return float(np.mean(d1) + np.mean(d2))



def centroid_of(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3)
    return pts.mean(axis=0)

def pca_axes(pcd: o3d.geometry.PointCloud):
    """
    Returns:
      R (3x3): columns are principal directions (X=col0, Y=col1, Z=col2)
      s (3,) : singular values (sqrt eigenvalues) proportional to std along each axis
    Columns sorted by decreasing variance. Ensures right-handed basis (det=+1).
    """
    P = np.asarray(pcd.points)
    C = P.mean(axis=0, keepdims=True)
    X = P - C
    # covariance
    cov = (X.T @ X) / max(len(P) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)           # eigenvalues ascending
    order = vals.argsort()[::-1]               # descending
    vals = vals[order]
    R = vecs[:, order]
    # right-handedness: flip third axis if needed
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    s = np.sqrt(np.maximum(vals, 0.0))
    return R, s



def initial_align_centroid_pca(src: o3d.geometry.PointCloud,
                               dst: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Build a rigid transform T0 that:
      - aligns src centroid to dst centroid
      - aligns src PCA axes to dst PCA axes (with sign fixes)
    """
    c_s = centroid_of(src)
    c_d = centroid_of(dst)

    R_s, _ = pca_axes(src)
    R_d, _ = pca_axes(dst)

    R_s_adj = R_s.copy()
    for i in range(3):
        if np.dot(R_s[:, i], R_d[:, i]) < 0:
            R_s_adj[:, i] *= -1.0
    # recompute right-handed
    if np.linalg.det(R_s_adj) < 0:
        R_s_adj[:, 2] *= -1.0

    # rotation mapping src->dst
    R0 = R_d @ R_s_adj.T
    # translation to map centroids
    t0 = c_d - R0 @ c_s

    T0 = np.eye(4)
    T0[:3, :3] = R0
    T0[:3, 3] = t0
    return T0


def find_best_template_teaser(dst_cloud, src_clouds, target_points=100):
    # Pre-align destination to its own PCA frame (not required, for stability we leave dst as-is)
    # Downsample dst once
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, target_points)
    if not dst_down.has_normals():
        dst_down.estimate_normals()
    res = cloud_resolution(dst_down)
    noise_bound = 1.5 * res           # adaptive, often far better than a fixed 1mm
    match_max_dist = 4.0 * res        # cap for geometric plausibility

    best = dict(idx=-1, T=np.eye(4), score=np.inf)
    all_metrics = []

    for idx, src_cloud in enumerate(src_clouds):
        # 0) Initial centroid + PCA alignment (src->dst)
        T0 = initial_align_centroid_pca(src_cloud, dst_cloud)
        src0 = copy.deepcopy(src_cloud).transform(T0)

        # 1) Downsample & FPFH *after* coarse alignment
        src_down, src_fpfh = preprocess_point_cloud_uniform(src0, target_points)
        if src_down is None:
            continue
        if not src_down.has_normals():
            src_down.estimate_normals()

        # 2) putative correspondences
        correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=match_max_dist)
        if len(correspondences) < 20:
            print("Not enough correspondences")
            all_metrics.append({"template_idx": idx, "num_corr": len(correspondences),
                                "num_inliers": 0, "inlier_ratio": 0.0,
                                "geom": float("inf"), "score": float("inf"),
                                "note": "few_corr"})
            continue
        # 3) TEASER++ (no init required, but the pre-align helps the correspondences)
        H, R_inliers, _, _ = run_teaser(src_down, dst_down, correspondences, noise_bound_m=noise_bound)
        inlier_ratio = (len(R_inliers) / max(1, len(correspondences)))
        # 4) Score with Chamfer on full clouds (apply full transform: H ∘ T0)
        T_full = H @ T0
        src_aligned = copy.deepcopy(src_cloud).transform(T_full)
        alpha = 1
    

        geom_err = chamfer_distance(src_aligned, dst_cloud)

        # Combined score
        score = alpha * geom_err

        all_metrics.append({
            "template_idx": idx,
            "num_corr": int(len(correspondences)),
            "num_inliers": int(len(R_inliers)),
            "inlier_ratio": float(inlier_ratio),
            "geom": float(geom_err),
            "score": float(score)
        })

        print(f"[{idx}] corr={len(correspondences):3d}, inl={len(R_inliers):3d} "
              f"({inlier_ratio*100:4.1f}%), geom={geom_err:.5f}, score={score:.5f}")


        if score < best["score"]:
            best.update(idx=idx, T=T_full, score=score)

    return best["idx"], best["T"], best["score"], all_metrics



