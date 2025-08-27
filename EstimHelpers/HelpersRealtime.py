import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from teaserpp_python import _teaserpp as tpp
import copy
import json
import os
import glob


def uniform_downsample_farthest_point(pcd, target_points=500):
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

def find_best_template_teaser(dst_cloud, src_clouds, target_points=100):
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, target_points)
    if not dst_down.has_normals():
        dst_down.estimate_normals()
    res = cloud_resolution(dst_down)
    noise_bound = 1.5 * res           # adaptive, often far better than a fixed 1mm
    match_max_dist = 4.0 * res        # cap for geometric plausibility

    best = dict(idx=-1, T=np.eye(4), score=np.inf)
    
    for idx, src_cloud in enumerate(src_clouds):
        T0 = initial_align_centroid_pca(src_cloud, dst_cloud)
        src0 = copy.deepcopy(src_cloud).transform(T0)

        # 1) Downsample & FPFH *after* coarse alignment
        src_down, src_fpfh = preprocess_point_cloud_uniform(src0, target_points)
        if src_down is None:
            continue
        if not src_down.has_normals():
            src_down.estimate_normals()
        correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=match_max_dist)
        if len(correspondences) < 5:
            print("Not enough correspondences")
            continue
        H = run_teaser(src_down, dst_down, correspondences, noise_bound_m=noise_bound)
        icp_result = o3d.pipelines.registration.registration_icp(
            src_down, dst_down, 0.01, H,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        refined_transform = icp_result.transformation  # refined

        T_full = refined_transform @ T0
        src_aligned = copy.deepcopy(src_cloud).transform(T_full)
        alpha = 1
    

        geom_err = chamfer_distance(src_aligned, dst_cloud)

        # Combined score
        score = alpha * geom_err

        if score < best["score"]:
            best.update(idx=idx, T=T_full, score=score)

    return best["idx"], best["T"]

def load_scene_camera(cam_json_path):
    with open(cam_json_path, "r") as f: data = json.load(f)
    # returns dict: frame_id -> (K (3x3), depth_scale, width, height)
    cams = {}
    for k, v in data.items():
        Klist = v["cam_K"] if "cam_K" in v else v["K"]     # be tolerant
        K = np.array(Klist, dtype=np.float32).reshape(3,3)
        depth_scale = float(v.get("depth_scale", 0.1))
        width  = int(v.get("width",  640))
        height = int(v.get("height", 480))
        cams[k] = (K, depth_scale, width, height)
    return cams

def list_bop_frames(seq_dir):
    rgb_dir   = os.path.join(seq_dir, "rgb")
    depth_dir = os.path.join(seq_dir, "depth")
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
    # ids are filenames without extension (usually 000000, 000001, …)
    frames = []
    for rgb_path in rgb_files:
        fid = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = os.path.join(depth_dir, f"{fid}.png")
        if os.path.exists(depth_path):
            frames.append((fid, rgb_path, depth_path))
    return frames

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
    cam_K_np = np.array(cam_K).reshape(3,3)
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
    return intrinsic, depth_scale,cam_K_np
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


def get_pointcloud(depth_raw, color_raw, scene_camera_parent, mask):    
    
    binary_mask = (mask == 255).astype(np.uint8)
    mask_pixels = np.sum(binary_mask)
    scene_camera_path = scene_camera_parent + "/scene_camera.json"
    if mask_pixels == 0:
        print("WARNING: No pixels selected by mask!")
        return None
    
    #color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    
    h, w = depth_raw.shape
    intrinsic, depth_scale, K = load_camera_intrinsics(scene_camera_path, 0, w, h)
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
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.9)
    
    return pcd, K, intrinsic

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
