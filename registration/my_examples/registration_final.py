import numpy as np
import open3d as o3d
import cv2
import json
from teaserpp_python import _teaserpp as tpp
import copy
import time
import glob
import os
from EstimHelpers.registration_utils import find_best_template_teaser, get_pointcloud, preprocess_point_cloud_uniform


if __name__ == "__main__":
    # Load images
    depth_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000005.png"
    rgb_path   = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000005.jpg"
    mask_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000005_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"
    
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

    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask_path=mask_path)
    #scene_pcd_centered = scene_pcd.translate(-scene_pcd.get_center())


    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("Failed to generate scene point cloud!")
        exit(1)
    
    start_time = time.time()
    best_idx, best_transform, inliers = find_best_template_teaser(scene_pcd, pointclouds, target_points=200)
    print(f"Best match: Template {best_idx} with {inliers} rotation inliers")

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
    exit()
    print("\n=== TRYING TEASER++ WITH ALIGNED CAD ===")
    best_correspondences = []
    best_voxel_size = None
    
    voxel_size = 0.001
    print(f"\nTrying voxel size: {voxel_size}")
    
    cad_down, cad_fpfh = preprocess_point_cloud_uniform(cad_aligned, 500)
    scene_down, scene_fpfh = preprocess_point_cloud_uniform(scene_pcd, 500)
    
    if cad_down is None or scene_down is None:
        exit()
    start_time = time.time() 
    correspondences = get_correspondences(cad_down, scene_down, cad_fpfh, scene_fpfh, 
                                        distance_threshold=voxel_size*10)
    
    if len(correspondences) > len(best_correspondences):
        best_correspondences = correspondences
        best_voxel_size = voxel_size
   
    if len(best_correspondences) >= 3:
        print(f"\nBest result: {len(best_correspondences)} correspondences with voxel size {best_voxel_size}")
        
        T = run_teaser(cad_down, scene_down, best_correspondences)
        
        # Apply transformation and visualize
        cad_final = copy.deepcopy(cad_aligned)
        cad_final.transform(T)
        cad_final_vis = cad_final.paint_uniform_color([1, 0, 0])
        end_time = time.time()
        
        scene_vis = copy.deepcopy(scene_pcd)
        scene_vis = scene_vis.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([cad_final_vis, scene_vis], 
                                        window_name="TEASER++ Result")
        

        # Duration
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.4f} seconds")
    else:
        print(f"\nTEASER++ failed. Best attempt: {len(best_correspondences)} correspondences")
        exit()
        