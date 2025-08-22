import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from teaserpp_python import _teaserpp as tpp
import copy



def uniform_downsample_farthest_point(pcd, target_points=2000):
    """
    Farthest Point Sampling (FPS) for uniform downsampling.
    Good spatial distribution, moderate computation time.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    #print(f"Number of points: {n_points}, target: {target_points}")

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
        
    #print(f"Original points: {len(pcd.points)}")

    pcd_down = uniform_downsample_farthest_point(pcd, target_points)
    
    #print(f"Downsampled points: {len(pcd_down.points)}")
    
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
    
    #print(f"FPFH feature dimension: {fpfh.dimension()}")
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
    
    # R_inliers = solver.getRotationInliers()
    # s_inliers = solver.getScaleInliers()
    # t_inliers = solver.getTranslationInliers()
    R = np.array(sol.rotation)
    t = np.array(sol.translation).reshape((3, 1))
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    
    return T#, R_inliers, s_inliers, t_inliers
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



def detect_mask(model, img_bgr, class_id=0, conf=0.8, imgsz=512):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    results = model(source=img_bgr, imgsz=imgsz, conf=conf, device="0", save=False, verbose=False)
    # if results[0].boxes is None or len(results[0].boxes) == 0:
    #     print("No bounding boxes detected. Exiting...")
    #     exit()  # or return if inside a function
    for r in results:
        if not hasattr(r, "masks") or r.masks is None:
            continue
        for seg, cls in zip(r.masks.xy, r.boxes.cls):
            
            if int(cls) != class_id: 
                continue
            poly = np.array(seg, dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
            return mask    # first match
    return mask

def rs_get_intrinsics(profile):
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    K = np.array([[intr.fx, 0,         intr.ppx],
                  [0,        intr.fy,   intr.ppy],
                  [0,        0,         1       ]], dtype=np.float32)
    return intr, K

def o3d_intrinsics_from_rs(intr):
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )

def find_best_template_teaser(dst_cloud, src_clouds, target_points=100):
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, target_points)
    dst_down, ind = dst_down.remove_statistical_outlier(
                    nb_neighbors=20,      # number of neighbors to analyze
                    std_ratio=2.0         # threshold: smaller = more aggressive
                )
    
    best_score = float("inf")
    best_idx = -1
    best_transform = np.eye(4)
    all_metrics = []
    
    for idx, src_cloud in enumerate(src_clouds):
        src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, target_points)
        if src_down is None:
            continue
        
        correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh)
        H = run_teaser(src_down, dst_down, correspondences)
        
        # # --- NEW: ICP refinement ---
        # icp_result = o3d.pipelines.registration.registration_icp(
        #     src_cloud, dst_cloud, 0.01, H,
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
        # )
        # refined_transform = icp_result.transformation
        
        src_aligned = copy.deepcopy(src_cloud).transform(H)
        score = chamfer_distance(src_aligned, dst_cloud)
        
        # metrics = {
        #     "template_idx": idx,
        #     "num_corr": len(correspondences),
        #     # "num_inliers": len(R_inliers or []),
        #     # "icp_fitness": icp_result.fitness,
        #     # "icp_rmse": icp_result.inlier_rmse,
        #     "chamfer": score
        # }
        # all_metrics.append(metrics)
        
        if score < best_score:  
            best_score = score
            best_idx = idx
            best_transform = H
    
    return best_idx, best_transform, dst_down, dst_fpfh





def fx_from_fov(fov_deg, width):
    return 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)


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


def crop_pointcloud_fov_hpr(pcd, camera_position=np.array([0, 0, -1]),
                           camera_lookat=np.array([0, 0, 0]),
                           fov_deg=60.0, radius_scale=100.0):
    """
    FOV + Hidden Point Removal (robust radius).
    radius_scale: increase if you still see over-pruning (typical 20–200).
    """
    # View direction
    cam_dir = camera_lookat - camera_position
    cam_dir /= (np.linalg.norm(cam_dir) + 1e-12)

    # Angular FoV prefilter
    pts = np.asarray(pcd.points)
    vecs = pts - camera_position
    dist = np.linalg.norm(vecs, axis=1)
    safe_dist = np.maximum(dist, 1e-8)
    dirs = vecs / safe_dist[:, None]
    cos_angles = np.dot(dirs, cam_dir)
    mask_fov = cos_angles > np.cos(np.deg2rad(fov_deg) / 2.0)
    idx_fov = np.where(mask_fov)[0]
    if idx_fov.size == 0:
        return o3d.geometry.PointCloud()  # nothing in FoV

    pcd_fov = pcd.select_by_index(idx_fov)

    # --- Robust radius choice ---
    # Use the maximum distance in the *full* cloud as a baseline to avoid local underestimation
    max_dist_full = (np.linalg.norm(pts - camera_position, axis=1).max()
                     if len(pts) else 1.0)
    # Make radius big (HPR becomes conservative as radius grows)
    radius = max(1e-3, radius_scale * max_dist_full)

    # HPR can still be sensitive if the camera is inside the convex hull.
    # Keep the camera comfortably away from the object relative to its size.
    _, visible_idx = pcd_fov.hidden_point_removal(camera_position, radius)
    return pcd_fov.select_by_index(visible_idx)