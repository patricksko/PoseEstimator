import numpy as np
import open3d as o3d
import json
from teaserpp_python import _teaserpp as tpp
import copy
import time
import glob
import os
from EstimHelpers.registration_utils import find_best_template_teaser, get_pointcloud, preprocess_point_cloud_uniform
from EstimHelpers.detection_utils import detect_mask
import cv2 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    weights = "./data/best.pt"
    rgb_path = "./data/000000.jpg"
    depth_path = "./data/000000.png"
    scene_camera_path = "./data/scene_camera.json"

    mask = detect_mask(weights, rgb_path)
    
    gt_data = "./data/scene_gt.json"
    #Read CAD Path
    cad_path = "./data/lego_views/"
    
    ply_files = sorted(glob.glob(os.path.join(cad_path, "*.ply")))

    pointclouds = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(ply_file)
        pointclouds.append(pcd)
        print(f"Loaded: {ply_file} with {len(pcd.points)} points")
    

    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)


    # o3d.visualization.draw_geometries([
    #     scene_pcd.paint_uniform_color([1, 0, 0]),
    #     pointclouds[8].paint_uniform_color([0, 1, 0])
    # ])
    
    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("Failed to generate scene point cloud!")
        exit(1)
    
    start_time = time.time()
    best_idx, best_transform, best_inliers, all_metrics = find_best_template_teaser(scene_pcd, pointclouds, target_points=100)
    
    # Apply final transformation
    if best_idx >= 0:
        final_template = copy.deepcopy(pointclouds[best_idx])
        final_template.transform(best_transform)
        
    scale_ratio = 1
    cad_pcd = final_template
    end_time=time.time()
    coord_transform = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

    # Object pose in standard camera coordinates
    standard_camera_pose = np.linalg.inv(coord_transform) @ best_transform @ coord_transform
    print(end_time-start_time)
    print("----------------------------")
    #cad_aligned = try_manual_alignment(cad_pcd, scene_pcd, scale_ratio)
    o3d.visualization.draw_geometries([scene_pcd.paint_uniform_color([1, 0, 0]),
                                        cad_pcd.paint_uniform_color([0, 1, 0])])
    # Read JSON file
    with open(gt_data, "r") as f:
        data = json.load(f)

    # Take the first image/frame entry
    first_key = sorted(data.keys())[0]  # e.g., "0"
    first_obj = data[first_key][0]      # first object in that frame

    # Extract rotation and translation
    R_list = first_obj["cam_R_m2c"]
    t_list = first_obj["cam_t_m2c"]

    # Convert to numpy arrays
    R = np.array(R_list).reshape(3, 3)
    t = np.array(t_list).reshape(3, 1)

    R_est = best_transform[:3, :3]
    t_est = best_transform[:3, 3:]

    print("Groundtruth Rotation")
    print(R)
    print("Estimated Rotation")
    print(R_est)

    print("Groundtruth Translation")
    print(t)
    print("Estimated Translation")
    print(t_est)


    np.savetxt("finaltransform.txt", best_transform, fmt="%.6f")


    
        

 