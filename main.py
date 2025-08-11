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
    weights = "/home/skoumal/dev/ObjectDetection/detection/best.pt"
    rgb_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000000.jpg"
    depth_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"

    mask = detect_mask(weights, rgb_path)
    
    #Read CAD Path
    cad_path = "/home/skoumal/dev/TEASER-plusplus/build/python/lego_views/"

    # Read all ply files
    ply_files = sorted(glob.glob(os.path.join(cad_path, "*.ply")))

    pointclouds = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(ply_file)
        pointclouds.append(pcd)
        print(f"Loaded: {ply_file} with {len(pcd.points)} points")
    
    # for pcd in pointclouds:
    #     o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])

    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)
    #scene_pcd_centered = scene_pcd.translate(-scene_pcd.get_center())


    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("Failed to generate scene point cloud!")
        exit(1)
    
    start_time = time.time()
    best_idx, best_transform, best_inliers, all_metrics = find_best_template_teaser(scene_pcd, pointclouds, target_points=100)
  



    # template_ids = [m.template_idx for m in all_metrics]
    # correspondences = [m.num_correspondences for m in all_metrics]
    # inliers = [m.num_inliers for m in all_metrics]
    # s_inliers = [m.num_s_inliers for m in all_metrics]
    # t_inliers = [m.num_t_inliers for m in all_metrics]

    # plt.figure(figsize=(10,5))
    # plt.plot(template_ids, correspondences, label="Correspondences", marker='o')
    # plt.plot(template_ids, inliers, label="R Inliers", marker='x')
    # plt.plot(template_ids, s_inliers, label="s Inliers", marker='x')
    # plt.plot(template_ids, t_inliers, label="t Inliers", marker='x')
    # plt.xlabel("Template Index")
    # plt.ylabel("Count")
    # plt.title("TEASER++ Template Matching Metrics")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # exit()

    # Apply final transformation
    if best_idx >= 0:
        final_template = copy.deepcopy(pointclouds[best_idx])
        final_template.transform(best_transform)
        
    scale_ratio = 1
    cad_pcd = final_template
    end_time=time.time()
    print(end_time-start_time)
    print("----------------------------")
    #cad_aligned = try_manual_alignment(cad_pcd, scene_pcd, scale_ratio)
    o3d.visualization.draw_geometries([scene_pcd.paint_uniform_color([1, 0, 0]),
                                        cad_pcd.paint_uniform_color([0, 1, 0])])
    
    cam_pose_path = "/home/skoumal/dev/ObjectDetection/registration/TEASER-plusplus/my_examples/lego_views/pose_08.npy"  # example for frame 5

    # Load the camera pose matrix
    cam_pose = np.load(cam_pose_path)

    # Compute object pose in world coordinates
    object_pose_world = cam_pose @ best_transform

    # Extract translation (position)
    position = object_pose_world[:3, 3]

    # Extract rotation matrix
    rotation = object_pose_world[:3, :3]

    # Optionally convert rotation matrix to Euler angles or quaternion
    from scipy.spatial.transform import Rotation as R

    r = R.from_matrix(rotation)
    euler_angles = r.as_euler('xyz', degrees=True)
    quaternion = r.as_quat()  # x,y,z,w

    print("Object position (world):", position)
    print("Object rotation (Euler angles xyz degrees):", euler_angles)
    print("Object rotation (quaternion x,y,z,w):", quaternion)
    
        

 