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
    rgb_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000001.jpg"
    depth_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000001.png"
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
    
    cam_pose_path = "/home/skoumal/dev/ObjectDetection/registration/TEASER-plusplus/my_examples/lego_views/pose_08.npy"  

    # Load the camera pose matrix
    cam_pose = np.load(cam_pose_path)
    cam_rotation = cam_pose[:3, :3]
    cam_translation = cam_pose[:3, 3]
    
    rotation_matrix = best_transform[:3, :3]
    translation_vector = best_transform[:3, 3]

    # If you have the template pose, you might need to compose transformations
    if cam_pose is not None:
        # Convert template pose to transformation matrix
        template_rotation = cam_rotation.reshape(3, 3)
        template_translation = cam_translation.reshape(3, 1)
        
        template_transform = np.eye(4)
        template_transform[:3, :3] = template_rotation
        template_transform[:3, 3] = template_translation.flatten()
        
        # Compose: final_pose = teaser_transform @ template_transform
        final_transform = best_transform @ template_transform
        
        rotation_matrix = final_transform[:3, :3]
        translation_vector = final_transform[:3, 3]
    
    np.savetxt("finaltransform.txt", best_transform, fmt="%.6f")


    
        

 