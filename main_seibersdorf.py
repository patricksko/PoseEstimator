#!/usr/bin/env python3
import argparse, os, numpy as np, cv2, open3d as o3d, yaml
# from tf_transformations import euler_matrix
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from EstimHelpers.detection_utils import detect_mask
from EstimHelpers.registration_utils import find_best_template_teaser
import glob
import copy
from colorama import Fore, Style

######## Global Variables  #########
WEIGHTS = "./data/best.pt"
PLY_PATH = "./data/seibersdorf_views/"
CAD_PATH = "./data/block_seibersdorf.ply"


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
    
    # Read CAD Model for comparision
    cad_model = o3d.io.read_point_cloud(CAD_PATH)
    cad_points = np.asarray(cad_model.points)

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None: raise SystemExit(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
    img_arr = np.asarray(img)
    mask = detect_mask(WEIGHTS, img)   # Detect mask of Object with YOLO Network

    # Cloud
    pcd = o3d.io.read_point_cloud(cloud_path)
    if len(pcd.points) == 0: raise SystemExit(f"No points in cloud: {cloud_path}")
    pts = np.asarray(pcd.points, dtype=np.float64)
    if max_points and pts.shape[0] > max_points:
        pts = pts[np.random.choice(pts.shape[0], max_points, replace=False)]

    # Calib
    K, D, T = load_calib(calib_path)

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
    o3d.visualization.draw([pcd_col], bg_color=(0, 0, 0, 1))

    # Read Pointcloud Templates and append them to list (Source)
    ply_files = sorted(glob.glob(os.path.join(PLY_PATH, "*.ply")))  
    src_clouds = []
    for ply_file in ply_files:
        src = o3d.io.read_point_cloud(ply_file)
        src_clouds.append(src)
        print(f"Loaded: {ply_file} with {len(src.points)} points")

    dst_cloud = copy.deepcopy(pcd_col)
    if dst_cloud is None or len(dst_cloud.points) == 0:
            print("Failed to generate scene point cloud!")
            exit(1)
    
    o3d.visualization.draw_geometries([
        src_clouds[4].paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")
    exit()
    best_idx, H, best_inliers, all_metrics = find_best_template_teaser(dst_cloud, src_clouds, target_points=100)
    for m in all_metrics:
        print(f"Template {m['template_idx']}: Chamfer = {m['score']:.6f}")
    print(best_idx)
    # Apply final transformation
    if best_idx >= 0:
        src_cloud = copy.deepcopy(src_clouds[best_idx])
        src_cloud.transform(H)
    print(H)
        
    # 7. Visualization
    print("\n7. Visualization...")
  
    o3d.visualization.draw_geometries([
        src_cloud.paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")


    uv = project_points(cad_points, K, H)
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
        image_path="./data/seibersdorf/export_images_01/rgb_1752488483145676170.png",
        cloud_path="./data/seibersdorf/export_pointclouds_01/cloud_1752488483100407156.ply",
        calib_path="./data/seibersdorf/calib.yaml",
        save_path="./data/seibersdorf/colored_cloud.ply",
        max_points=200000
    )
