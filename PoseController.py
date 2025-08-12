import numpy as np
import open3d as o3d
from EstimHelpers.registration_utils import find_best_template_teaser, get_pointcloud
import cv2

if __name__ == "__main__":
    # Load images
    rgb_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000000.jpg"
    depth_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000000.png"
    mask_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000000_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"

    # Read CAD
    cad = "/home/skoumal/dev/BlenderProc/LegoBlock_variant.ply"
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)
    
    print("=== TESTING DIFFERENT TRANSFORMATIONS ===")
    
    # Test 1: Direct TEASER++ output
    try:
        transform1 = np.loadtxt("best_transform_debug.txt")
        pcd1 = o3d.io.read_point_cloud(cad)
        pcd1.transform(transform1)
        
        scene_center = scene_pcd.get_center()
        pcd1_center = pcd1.get_center()
        error1 = np.linalg.norm(scene_center - pcd1_center)
        print(f"Transform 1 (direct): Center error = {error1:.6f}")
        
        print("Visualizing Transform 1...")
        o3d.visualization.draw_geometries([
            scene_pcd.paint_uniform_color([1, 0, 0]),    # Red = scene
            pcd1.paint_uniform_color([0, 1, 0])          # Green = CAD
        ])
    except FileNotFoundError:
        print("best_transform_debug.txt not found!")
    
    # Test 2: Standard camera coordinates
    try:
        transform2 = np.loadtxt("standard_camera_transform.txt")
        pcd2 = o3d.io.read_point_cloud(cad)
        pcd2.transform(transform2)
        
        pcd2_center = pcd2.get_center()
        error2 = np.linalg.norm(scene_center - pcd2_center)
        print(f"Transform 2 (standard): Center error = {error2:.6f}")
        
        print("Visualizing Transform 2...")
        o3d.visualization.draw_geometries([
            scene_pcd.paint_uniform_color([1, 0, 0]),    # Red = scene
            pcd2.paint_uniform_color([0, 0, 1])          # Blue = CAD
        ])
    except FileNotFoundError:
        print("standard_camera_transform.txt not found!")
    
    # Test 3: Original finaltransform.txt
    try:
        transform3 = np.loadtxt("finaltransform.txt")
        pcd3 = o3d.io.read_point_cloud(cad)
        pcd3.transform(transform3)
        
        pcd3_center = pcd3.get_center()
        error3 = np.linalg.norm(scene_center - pcd3_center)
        print(f"Transform 3 (original): Center error = {error3:.6f}")
        
        print("Visualizing Transform 3...")
        o3d.visualization.draw_geometries([
            scene_pcd.paint_uniform_color([1, 0, 0]),    # Red = scene
            pcd3.paint_uniform_color([1, 1, 0])          # Yellow = CAD
        ])
    except FileNotFoundError:
        print("finaltransform.txt not found!")
    
    # Test 4: Check if we need to use the same CAD file that was used for template matching
    print("\n=== CHECKING CAD vs TEMPLATE ALIGNMENT ===")
    print("Make sure you're using the SAME CAD file that corresponds to the best template!")
    print("Your best template index will be printed in the debug script.")
    
    # Additional debugging
    print(f"\n=== SCENE PROPERTIES ===")
    print(f"Scene points: {len(scene_pcd.points)}")
    print(f"Scene center: {scene_pcd.get_center()}")
    print(f"Scene bounds: {scene_pcd.get_axis_aligned_bounding_box()}")
    
    original_cad = o3d.io.read_point_cloud(cad)
    print(f"\n=== ORIGINAL CAD PROPERTIES ===")
    print(f"CAD points: {len(original_cad.points)}")
    print(f"CAD center: {original_cad.get_center()}")
    print(f"CAD bounds: {original_cad.get_axis_aligned_bounding_box()}")
