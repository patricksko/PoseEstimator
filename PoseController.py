import numpy as np
import open3d as o3d
from EstimHelpers.registration_utils import find_best_template_teaser, get_pointcloud, preprocess_point_cloud_uniform


if __name__ == "__main__":
    # Load images
    rgb_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000000.jpg"
    depth_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000000.png"
    mask_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000000_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"
    
    #Read CAD Path
    cad = "/home/skoumal/dev/BlenderProc/LegoBlock_variant.ply"

  
    pcd = o3d.io.read_point_cloud(cad)
    
    # for pcd in pointclouds:
    #     o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])

    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask_path)

    o3d.visualization.draw_geometries([scene_pcd.paint_uniform_color([1, 0, 0]),
                                        pcd.paint_uniform_color([0, 1, 0])])
    
    final_transform_loaded = np.loadtxt("finaltransform.txt")

    # Check shape & values
    if final_transform_loaded.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transformation, got {final_transform_loaded.shape}")

    print("Loaded Transformation Matrix:")
    print(final_transform_loaded)

    # Apply transformation to CAD point cloud
    pcd.transform(final_transform_loaded)

    # Visualize transformed CAD with scene point cloud
    o3d.visualization.draw_geometries([
        scene_pcd.paint_uniform_color([1, 0, 0]),  # Scene = Red
        pcd.paint_uniform_color([0, 1, 0])         # CAD (transformed) = Green
    ])