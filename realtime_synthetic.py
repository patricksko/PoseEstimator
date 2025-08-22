#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import open3d as o3d
from teaserpp_python import _teaserpp as tpp
#from EstimHelpers.HelpersRealtime import *
import glob
import copy
import json


# -------------------- CONFIG: ----------------------------
BOP_SEQ_DIR   = "./datageneration/Blenderproc/output_blenderproc_video/bop_data/Legoblock_full/train_pbr/000000"  # where BlenderProc wrote the frames
TEMPLATES_DIR = "./data/lego_views"                          # your 11/26 template PLYs
CAD_MODEL_PLY = "./data/obj_000001.ply"                      # full CAD (for projection)
YOLO_WEIGHTS  = "./data/best.pt"
CLASS_ID      = 0
TRACK_EVERY   = 1   # run TEASER update every N frames
TARGET_PTS    = 300  # feature downsample size



def uniform_downsample_farthest_point(pcd, target_points=TARGET_PTS):
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
def preprocess_point_cloud_uniform(pcd, target_points=TARGET_PTS):
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


def dense_all_to_all_correspondences(n_src, n_dst, max_pairs=2000, seed=0):
    """
    Build a big pool of putative correspondences WITHOUT feature matching.
    We hypothesize (src_i, dst_j) for many i,j pairs and randomly cap to max_pairs.

    Returns:
        corr_2xM (np.ndarray int, shape (2, M)): 
           [[src_idx0, src_idx1, ...],
            [dst_idx0, dst_idx1, ...]]
    """
    
    rng = np.random.default_rng(seed)
    # All index pairs (cartesian product) -> potentially huge
    ii, jj = np.meshgrid(np.arange(n_src), np.arange(n_dst), indexing="ij")
    pairs = np.stack([ii.ravel(), jj.ravel()], axis=1)

    if pairs.shape[0] > max_pairs:
        sel = rng.choice(pairs.shape[0], size=max_pairs, replace=False)
        pairs = pairs[sel]
    return pairs.T.astype(np.int32)


def run_teaser(source, target, corr_2xM):
    if corr_2xM is None or corr_2xM.size < 6:  # need >= 3 pairs
        return np.eye(4)
    print(corr_2xM.size)
    src_idx = corr_2xM[0]
    dst_idx = corr_2xM[1]
    
    src_pts = np.asarray(source.points)[src_idx].T  # 3xM
    dst_pts = np.asarray(target.points)[dst_idx].T  # 3xM

    params = tpp.RobustRegistrationSolver.Params()
    params.cbar2 = 1
    params.noise_bound = 0.01     # <-- tune to your scale (meters). Larger tolerates more outliers.
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = tpp.RotationEstimationAlgorithm.GNC_TLS
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12

    solver = tpp.RobustRegistrationSolver(params)
    solver.solve(src_pts, dst_pts)
    sol = solver.getSolution()

    R = np.array(sol.rotation)
    t = np.array(sol.translation).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T
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
    dst_down, _ = preprocess_point_cloud_uniform(dst_cloud, target_points)
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
        print("aaaaaaa")
        corr = dense_all_to_all_correspondences(
            n_src=len(src_down.points),
            n_dst=len(dst_down.points),
            max_pairs=1000,   # keep this manageable; raise if TARGET_PTS is small
            seed=idx           # any changing seed is fine
        )
        print("bbbbbbb")
        H = run_teaser(src_down, dst_down, corr)
        print("cccccccc")
        
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
    
    return best_idx, best_transform, dst_down

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

def main():
    frames = list_bop_frames(BOP_SEQ_DIR)
    if not frames:
        raise RuntimeError(f"No frames found in {BOP_SEQ_DIR}")
    cam_json = os.path.join(BOP_SEQ_DIR, "scene_camera.json")
    _ = load_scene_camera(cam_json)  # optional sanity

    # Load templates + CAD
    template_paths = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.ply")))
    if not template_paths:
        raise RuntimeError(f"No PLYs in {TEMPLATES_DIR}")
    templates = [o3d.io.read_point_cloud(p) for p in template_paths]
    cad_model = o3d.io.read_point_cloud(CAD_MODEL_PLY)
    cad_points = np.asarray(cad_model.points)

    mesh = o3d.io.read_triangle_mesh(CAD_MODEL_PLY)
    mesh.compute_vertex_normals()

    yolo = YOLO(YOLO_WEIGHTS)

    initialized = False
    T_m2c = np.eye(4)

    # Offscreen renderer (lazy-init once we know width/height)
    renderer = None
    scene = None
    render_w = render_h = None

    idx = 0
    while idx < len(frames):
        print("=" * 20, "loop", "=" * 20, "idx =", idx)
        fid, rgb_path, depth_path = frames[idx]
        print(f"reading frame {fid}")

        color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # uint16

        # YOLO mask
        mask = detect_mask(yolo, color, class_id=CLASS_ID, conf=0.7, imgsz=640)

        # Scene point cloud (uses correct per-frame intrinsics)
        dst_cloud, K, intr_o3d = get_pointcloud(depth_raw, color, BOP_SEQ_DIR, mask)
        if len(dst_cloud.points) == 0:
            cv2.imshow("Synthetic Tracking", color)
            if cv2.waitKey(1) == 27:
                break
            continue

        if not initialized:
            print("[INFO] Initializing with templates…")
            best_idx, refined_transform, dst_down = find_best_template_teaser(
                dst_cloud, templates, target_points=TARGET_PTS
            )
            cand_cad = copy.deepcopy(templates[best_idx])
            cand_cad.transform(refined_transform)
            o3d.visualization.draw_geometries([
                dst_cloud.paint_uniform_color([0, 1, 0]),
                cand_cad.paint_uniform_color([0, 1, 1])
            ], window_name="Init registration (ENTER=accept, anything else=retry)")
            nb = input('Accept? [y/N]: ')
            if nb.lower() == 'y':
                print("[INFO] Initial pose accepted.")
                initialized = True
                T_m2c = refined_transform.copy()
                # prepare renderer once we know dims
                render_w, render_h = intr_o3d.width, intr_o3d.height
                renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
                scene = renderer.scene
                scene.set_background([1, 1, 1, 1])
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                scene.add_geometry("mesh", mesh, mat)
                idx = 1  # start tracking on next frame
                continue
            else:
                print("[INFO] Initial pose rejected. Retrying from start…")
                idx = 0
                continue

        # ensure renderer matches this frame's intr dims (rarely changes)
        if (render_w != intr_o3d.width) or (render_h != intr_o3d.height):
            render_w, render_h = intr_o3d.width, intr_o3d.height
            renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
            scene = renderer.scene
            scene.set_background([1, 1, 1, 1])
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            scene.add_geometry("mesh", mesh, mat)

        idx += 1
        start = time.time()

        # Projection (intrinsics) for this frame
        near, far = 0.01, 2.0
        scene.camera.set_projection(K, near, far, render_w, render_h)

        # Extrinsics from current T_m2c
        eye, target, up = camera_eye_lookat_up_from_H(T_m2c)
        scene.camera.look_at(target, eye, up)

        # Render synthetic view of CAD at T_m2c
        color_img = renderer.render_to_image()
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)

        # Backproject to get "src" cloud in camera coords
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
        )
        src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)

        prev_down, prev_fpfh = preprocess_point_cloud_uniform(src, TARGET_PTS)

        if idx % TRACK_EVERY == 0:
            dst_down, _ = preprocess_point_cloud_uniform(dst_cloud, TARGET_PTS)
            corr = dense_all_to_all_correspondences(
                n_src=len(prev_down.points),
                n_dst=len(dst_down.points),
                max_pairs=2000,   # keep this manageable; raise if TARGET_PTS is small
                seed=idx           # any changing seed is fine
            )

            delta = run_teaser(prev_down, dst_down, corr)

            icp_result = o3d.pipelines.registration.registration_icp(
                prev_down, dst_cloud, 0.01, delta,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            delta = icp_result.transformation  # refined
            T_m2c = delta@T_m2c
            print("="*20, "T_m2c", "="*20)
            print(T_m2c)
            # (optional) visualize alignment quality
            # o3d.visualization.draw_geometries([
            #     src.transform(delta).paint_uniform_color([0, 1, 0]),
            #     dst_cloud.paint_uniform_color([1, 0, 0]),
            # ])
            
        end = time.time()
        print(end-start)
        uv = project_points(cad_points, K, T_m2c)
        for (u, v) in uv:
            if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                cv2.circle(color, (u, v), 1, (0, 0, 255), -1)

        cv2.imshow("Live Tracking", color)
        if cv2.waitKey(1) & 0xFF == 27:
            break
       

if __name__ == "__main__":
    main()
