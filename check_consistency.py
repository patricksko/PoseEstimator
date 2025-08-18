import open3d as o3d
import numpy as np
import json
import copy
from pathlib import Path
from EstimHelpers.detection_utils import detect_mask
from EstimHelpers.registration_utils import run_teaser, get_pointcloud, visualize_correspondences, preprocess_point_cloud_uniform, get_correspondences, pyrender_to_open3d

def load_gt_pose(scene_gt_path, frame_id=0, obj_idx=0, unit='mm'):
    """Load homogeneous transform T_m2c from scene_gt.json for given frame and object."""
    with open(scene_gt_path, "r") as f:
        gt_data = json.load(f)

    key = str(frame_id)
    gt_obj = gt_data[key][obj_idx]

    R = np.array(gt_obj["cam_R_m2c"]).reshape(3, 3)
    t = np.array(gt_obj["cam_t_m2c"]).reshape(3)

    if unit == 'mm':
        t = t / 1000.0  # convert mm → m

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def apply_transform(pcd, T):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(T)
    return pcd_copy

if __name__ == "__main__":
    weights = "./data/best.pt"
    rgb_path = "./data/000000.jpg"
    depth_path = "./data/000000.png"
    scene_camera_path = "./data/scene_camera.json"
    gt_data = "./data/scene_gt.json"
    
    # Read JSON file
    with open(gt_data, "r") as f:
        data = json.load(f)

    # Take the first image/frame entry
    first_key = sorted(data.keys())[1]  # e.g., "0"
    first_obj = data[first_key][0]      # first object in that frame

    # Extract rotation and translation
    R_list = first_obj["cam_R_m2c"]
    t_list = first_obj["cam_t_m2c"]

    # Convert to numpy arrays
    R = np.array(R_list).reshape(3, 3)
    t = np.array(t_list).reshape(3, 1)

    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R
    T_m2c[:3, 3] = t[:, 0]
    mask = detect_mask(weights, rgb_path)
    
    # #Read CAD Path
    cad_path = "./data/obj_000001.ply"
    # 1. Load model point cloud
    src = o3d.io.read_point_cloud(cad_path)
    T_cv2ogl = np.array([[1, 0, 0, 0],
                     [0,-1, 0, 0],
                     [0, 0,-1, 0],
                     [0, 0, 0, 1]])
    # src = src.transform(np.linalg.inv(T_cv2ogl))
    # 2. Load scene point cloud (already reconstructed from depth)
    dst = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)

    # 3. Load GT pose for frame 0
    T_m2c = load_gt_pose(gt_data, frame_id=0, obj_idx=0, unit="m")
    # 4. Transform model into camera frame
    src_gt = apply_transform(src, T_m2c)
    
    # 5. Visualize overlay
    src_gt.paint_uniform_color([1, 0, 0])  # red model at GT pose
    dst.paint_uniform_color([0, 1, 0])     # green scene cloud
    o3d.visualization.draw_geometries([src_gt, dst],
                                      window_name="GT Overlay (Red=model, Green=scene)")
