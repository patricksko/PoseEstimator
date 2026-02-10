import numpy as np
import open3d as o3d
import teaserpp_python #import _teaserpp as tpp
import cv2
from scipy.spatial import cKDTree


def enforce_upright_pose_y_up(T):
    """
    Align model's local +Y axis with world -Y (up direction).
    Allows up to ±10° tolerance; otherwise rotates 90° around X repeatedly.
    """
    T = np.array(T, copy=True)
    R = T[:3, :3]

    world_up = np.array([0, -1, 0], dtype=np.float64)
    cos_tolerance = np.cos(np.deg2rad(30))  # ≈ 0.9848

    for _ in range(4):
        print("="*20)
        print("\n")
        print(R)
        print("="*20)
        print("\n")
        up_local = R[:, 1]
        cos_angle = np.dot(up_local, world_up)/(np.linalg.norm(up_local)*np.linalg.norm(world_up))
        print(cos_angle)
        if cos_angle >= cos_tolerance:
            print("="*20)
            print("\n")
            print("Gewählt:")
            T[:3, :3] = R
            print(R)
            return T

        # Rotate 90° about X
        Rz = np.array([[0, -1, 0],
                       [1, 0, 0],
                       [0, 0, 1]], dtype=np.float64)
        R = R @ Rz

    T[:3, :3] = R
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
def preprocess_point_cloud_uniform(pcd, target_points=500, calc_fpfh=False):
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

def nn_residuals(src_aligned, dst_cloud):
    src_pts = np.asarray(src_aligned.points)
    dst_pts = np.asarray(dst_cloud.points)
    tree = cKDTree(dst_pts)
    dists, _ = tree.query(src_pts, k=1, workers=-1)
    print("hahahahahahhahahahaha")
    return dists

def voxel_coverage(points, voxel_size):
    voxels = np.floor(points / voxel_size).astype(np.int32)
    return len(np.unique(voxels, axis=0))

def alignment_score(src_aligned, src_down, dst_down, voxel_size):
    dists = nn_residuals(src_aligned, dst_down)
    if dists is None:
        return np.inf

    med = np.median(dists)
    p90 = np.percentile(dists, 90)

    cov_aligned = voxel_coverage(
        np.asarray(src_aligned.points), voxel_size
    )
    cov_full = voxel_coverage(
        np.asarray(src_down.points), voxel_size
    )
    cov_norm = cov_aligned / max(cov_full, 1)

    # final score (lower is better)
    score = med + 0.3 * p90 + 0.5 * (1.0 - cov_norm)
    return score

def run_teaser(source, target, voxel_size):
    # Compute FPFH using voxel_size (NOT noise_bound)
    noise_bound = voxel_size# * 1.5

    dst_feats = extract_fpfh(target, voxel_size)
    src_feats = extract_fpfh(source, voxel_size)

    noise_bound = voxel_size * 1.5
    correspondences = get_correspondences(source, target, src_feats, dst_feats, distance_threshold=noise_bound*1.5)

    src_corr = np.array([source.points[i] for i, _ in correspondences]).T  # 3xN
    dst_corr = np.array([target.points[j] for _, j in correspondences]).T  # 3xN

    num_corrs = dst_corr.shape[1]
    points = np.concatenate((dst_corr.T,src_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([target,source,line_set])
    points = np.concatenate((dst_corr.T,src_corr.T),axis=0)
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.noise_bound = float(noise_bound)
    params.estimate_scaling = False  # IMPORTANT
    params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    params.rotation_estimation_algorithm = teaserpp_python.RotationEstimationAlgorithm.GNC_TLS

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(src_corr, dst_corr)
    sol = solver.getSolution()

    T = np.eye(4)
    T[:3, :3] = np.asarray(sol.rotation)
    T[:3, 3]  = np.asarray(sol.translation).reshape(3)
    return T

    




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

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size
  nn_param = min(30, len(pcd.points) // 2)  # Adaptive neighbor count
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=nn_param))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return fpfh

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


def draw_model_projection_with_axes(color, cad_points, K, T_m2c, axis_length=0.05):
    """
    Projects the CAD model points and its coordinate frame onto the color image.

    Args:
        color (np.ndarray): BGR color image to draw on.
        cad_points (np.ndarray): Nx3 CAD model points.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        T_m2c (np.ndarray): 4x4 transformation (model → camera).
        axis_length (float, optional): Length of each coordinate axis in meters.
    """
    # ---- Project CAD points ----
    uv = project_points(cad_points, K, T_m2c)
    for (u, v) in uv:
        if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
            cv2.circle(color, (u, v), 1, (0, 0, 255), -1)  # red points

    # ---- Draw coordinate frame ----
    origin = np.array([[0, 0, 0]])
    x_axis = np.array([[axis_length, 0, 0]])
    y_axis = np.array([[0, axis_length, 0]])
    z_axis = np.array([[0, 0, axis_length]])

    axes_pts = np.vstack([origin, x_axis, y_axis, z_axis])
    axes_uv = project_points(axes_pts, K, T_m2c)
    o = tuple(axes_uv[0])

    # Draw X (red), Y (green), Z (blue)
    cv2.line(color, o, tuple(axes_uv[1]), (0, 0, 255), 2)  # X-axis
    cv2.line(color, o, tuple(axes_uv[2]), (0, 255, 0), 2)  # Y-axis
    cv2.line(color, o, tuple(axes_uv[3]), (255, 0, 0), 2)  # Z-axis
