#!/usr/bin/env python3
import argparse, os, numpy as np, cv2, open3d as o3d, yaml
# from tf_transformations import euler_matrix
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pose_estimator.EstimHelpers.HelpersRealtime import *
from pose_estimator.EstimHelpers.RealSenseClass import RealSenseCamera
from pose_estimator.EstimHelpers.PoseEstimator import PoseEstimator
from pose_estimator.EstimHelpers.Detector import Detector
from pose_estimator.EstimHelpers.template_creation import render_templates
import glob
import copy
from colorama import Fore, Style

######## Global Variables  #########
WEIGHTS = "./data/best.pt"
PLY_PATH = "./data/seibersdorf_views/"
CAD_PATH = "./data/_Daten_Seibersdorf_Patrick/ConcreteBlock.ply"


def print_cloud_info(name, cloud):
    if isinstance(cloud, o3d.geometry.PointCloud):
        pts = np.asarray(cloud.points)
    elif isinstance(cloud, o3d.geometry.TriangleMesh):
        pts = np.asarray(cloud.vertices)
    else:
        raise ValueError("Unsupported geometry type")

    if pts.shape[0] == 0:
        print(f"{name}: EMPTY")
        return

    bbox = cloud.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()  # size along x,y,z
    print(f"{name}:")
    print(f"  Num points: {pts.shape[0]}")
    print(f"  Center: {bbox.get_center()}")
    print(f"  Extents: {extents} (x,y,z)")
    print(f"  Diagonal length: {np.linalg.norm(extents)}")

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

def load_calib(path):
    with open(path, "r") as f: c = yaml.safe_load(f)
    K = np.asarray(c["K"], dtype=float).reshape(3,3)
    D = np.asarray(c.get("D", []), dtype=float).reshape(-1)
    if "T" in c:
        T = np.asarray(c["T"], dtype=float).reshape(4,4)
    else:
        assert "xyz" in c and "rpy" in c, "calib.yaml must have T or (xyz+rpy)"
        # T = euler_matrix(*c["rpy"]); T[:3,3] = np.array(c["xyz"], dtype=float)
        r = R.from_euler("xyz", c["rpy"], degrees=False)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = np.array(c["xyz"], dtype=float)
    return K, D, T

def project_count(pts, R, t, K, D, W, H):
    Xc = (R @ pts.T).T + t
    z = Xc[:,2]
    front = z > 0.1
    if not np.any(front): return 0, front, None, None
    Xf = Xc[front]
    uv, _ = cv2.projectPoints(Xf.astype(np.float32),
                              np.zeros((3,1)), np.zeros((3,1)),
                              K, D if D.size in (4,5,8) else None)
    uv = uv.reshape(-1,2).astype(np.int32)
    in_img = (uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    return int(in_img.sum()), front, uv, in_img

def project_and_colorize(image_path, cloud_path, calib_path, save_path=None, max_points=250000):
    K, D, T = load_calib(calib_path)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise SystemExit(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    img_arr = np.asarray(img_bgr)

    height = img_bgr.shape[0]
    width  = img_bgr.shape[1]

    intr = o3d.camera.PinholeCameraIntrinsic(
    width,
    height,
    K[0, 0],  # fx
    K[1, 1],  # fy
    K[0, 2],  # cx
    K[1, 2],  # cy
)
    estimator = PoseEstimator(CAD_PATH, PLY_PATH, intr, K, 500) # call the constructor of PoseEstimator for 6d-Pose
    detector = Detector(WEIGHTS)
    mesh = o3d.io.read_triangle_mesh(CAD_PATH)

    # Ensure normals exist (optional, helps for visualization)
    mesh.compute_vertex_normals()
    

    # Sample N points uniformly on the surface
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    print_cloud_info("CAD source", pcd)

    cad_points = np.asarray(pcd.points)
    print("Sampled CAD points:", cad_points.shape)
    
    
    detections = detector.detect_mask(img)

    print(detections)
    if len(detections) == 0:
        exit()
    mask = detections[0]["mask"]
    
    
    # Cloud
    pcd = o3d.io.read_point_cloud(cloud_path)
    if len(pcd.points) == 0: raise SystemExit(f"No points in cloud: {cloud_path}")
    pts = np.asarray(pcd.points, dtype=np.float64)
    if max_points and pts.shape[0] > max_points:
        pts = pts[np.random.choice(pts.shape[0], max_points, replace=False)]

    # Calib
    

    # Build candidates
    T_inv = np.linalg.inv(T)
    Rinv, tinv = T_inv[:3,:3].astype(np.float64), T_inv[:3,3].astype(np.float64)
   
    n_in, front, uv, in_img = project_count(pts, Rinv, tinv, K, D, W, H)
    print(f"[inverse] front-facing: {int(front.sum())}  in-image: {n_in}")
        
    if n_in == 0:
        raise SystemExit("No projected points landed inside the image with any transform.\n"
                         "Likely the provided T is for the opposite direction or a different frame.\n"
                         "Hint: if T came from URDF parent→child (LiDAR←Camera), you usually need its inverse + optical fix.")


    # Final color sampling
    idx_front = np.where(front)[0]
    idx_inimg = idx_front[in_img]
    uv_in = uv[in_img]

    mask_bool = mask.astype(bool)

    # select only projected points inside the mask
    inside_mask = mask_bool[uv_in[:,1], uv_in[:,0]]
    idx_inmask = idx_inimg[inside_mask]
    uv_inmask  = uv_in[inside_mask]

    # get colors + corresponding 3D points
    colors = img[uv_inmask[:,1], uv_inmask[:,0], :].astype(np.float32) / 255.0
    pts_col = pts[idx_inmask]

    pcd_col = o3d.geometry.PointCloud()
    pcd_col.points = o3d.utility.Vector3dVector(pts_col)
    pcd_col.colors = o3d.utility.Vector3dVector(colors)
    pcd_clean, _ = pcd_col.remove_statistical_outlier(
        nb_neighbors=30,
        std_ratio=1.0
    )
    
    print_cloud_info("Scene target", pcd_clean)
    o3d.visualization.draw([pcd_clean], bg_color=(0, 0, 0, 1))

    dst_cloud = copy.deepcopy(pcd_clean)
    if dst_cloud is None or len(dst_cloud.points) == 0:
            print("Failed to generate scene point cloud!")
            exit(1)
    

    H, src_down = estimator.find_best_template_teaser(dst_cloud)
    o3d.visualization.draw_geometries([
        src_down.paint_uniform_color([1, 0, 0]),
        dst_cloud.paint_uniform_color([0, 1, 1])
    ], window_name="Initial Alignment before TEASER")

    src_cloud = copy.deepcopy(src_down)
    src_cloud.transform(H)
    print(H)
        
    # 7. Visualization
    print("\n7. Visualization...")
  
    o3d.visualization.draw_geometries([
        src_cloud.paint_uniform_color([1, 0, 0]),
        dst_cloud.paint_uniform_color([0, 1, 1])
    ], window_name="ICP Refined Registration")

    T_m2c = np.linalg.inv(T)@H
    
    uv = project_points(cad_points, K, T_m2c)
   
    for (u, v) in uv:
        if 0 <= u < img_arr.shape[1] and 0 <= v < img_arr.shape[0]:
            cv2.circle(img_arr, (u, v), 1, (0, 0, 255), -1)  # red dots

    cv2.imshow("Live Tracking", img_arr)
    print("Press ESC to close window...")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
if __name__ == "__main__":
    project_and_colorize(
        image_path="./data/_Daten_Seibersdorf_Patrick/exported_rgb_pcl/1765974786.360612.png",
        cloud_path="./data/_Daten_Seibersdorf_Patrick/exported_rgb_pcl/1765974786.360612.ply",
        calib_path="./data/_Daten_Seibersdorf_Patrick/calib_zed2i_to_seyond.yaml",
        save_path="./data/_Daten_Seibersdorf_Patrick/colored_cloud.ply",
        max_points=20000000
    )
