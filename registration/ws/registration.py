import numpy as np
import open3d as o3d
import copy
import time
from colorama import Fore, Style
from EstimHelpers.registration_utils import (
    run_teaser, get_correspondences, preprocess_point_cloud_uniform,
    crop_pointcloud_fov_hpr, apply_transform_and_noise, 
    sample_pointcloud_with_noise, visualize_correspondences
)

# Configuration
VOXEL_SIZE = 0.05
NOISE_BOUND = VOXEL_SIZE

def timer_print(start_time, label):
    """Clean timer output"""
    elapsed = time.time() - start_time
    print(Fore.GREEN + f"  {label}: {elapsed:.3f}s" + Style.RESET_ALL)
    return elapsed

def get_angular_error(R_exp, R_est):
    """Calculate angular error in radians"""
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))

def create_test_data():
    """Create test point clouds with camera projections"""
    # Ground truth transformation
    theta_x, theta_z = np.deg2rad(5), np.deg2rad(10)
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = Rz @ Rx
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.1, -0.05, 0.0]
    
    # Load CAD model and create point clouds
    cad_mesh = o3d.io.read_triangle_mesh("./LegoBlock_variant.ply")
    cad_mesh.compute_vertex_normals()
    
    src_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=500, jitter_sigma=0.0)
    dst_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=500, jitter_sigma=0.0005)
    
    # Apply camera FoV + HPR
    camera_pos_src = np.array([0.3, 0.2, 2.5])
    camera_target = np.array([0, 0, 0])
    
    src_cloud = crop_pointcloud_fov_hpr(src_mesh_sample, camera_pos_src, camera_target, fov_deg=60.0)
    dst_cloud = crop_pointcloud_fov_hpr(dst_mesh_sample, camera_pos_src, camera_target, fov_deg=60.0)
    
    # Apply transformation & noise to target
    dst_cloud = apply_transform_and_noise(dst_cloud, R, T[:3, 3], noise_sigma=0.0002)
    
    return src_cloud, dst_cloud, T

def print_results(T_gt, T_est, method_name):
    """Print registration errors"""
    rot_error_rad = get_angular_error(T_gt[:3, :3], T_est[:3, :3])
    rot_error_deg = np.rad2deg(rot_error_rad)
    trans_error = np.linalg.norm(T_gt[:3, 3] - T_est[:3, 3])
    
    print(f"{method_name} Results:")
    print(f"  Rotation error: {rot_error_deg:.2f}° ({rot_error_rad:.4f} rad)")
    print(f"  Translation error: {trans_error:.4f}m")
    print(f"  Expected: {T_gt[:3, 3]}")
    print(f"  Estimated: {T_est[:3, 3]}")

def main():
    """Main registration pipeline"""
    print("Point Cloud Registration Pipeline")
    print("=" * 50)
    
    # 1. Create test data
    print("\n1. Creating test data...")
    start = time.time()
    src_cloud, dst_cloud, T_gt = create_test_data()
    timer_print(start, "Data creation")
    
    print(f"  Source: {len(src_cloud.points)} points")
    print(f"  Target: {len(dst_cloud.points)} points")
    
    # 2. Preprocessing
    print("\n2. Preprocessing...")
    start = time.time()
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 2000)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 2000)
    timer_print(start, "Downsampling & features")
    
    print(f"  Downsampled to {len(src_down.points)} and {len(dst_down.points)} points")
    
    # 3. Find correspondences
    print("\n3. Finding correspondences...")
    start = time.time()
    correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=0.5)
    timer_print(start, "Correspondence search")
    
    print(f"  Found {len(correspondences)} correspondences")
    
    if len(correspondences) < 3:
        print("  ERROR: Too few correspondences!")
        return
    
    # 4. TEASER++ registration
    print("\n4. TEASER++ registration...")
    start = time.time()
    H, R_inliers, s_inliers, t_inliers = run_teaser(src_down, dst_down, correspondences)
    timer_print(start, "TEASER++")
    
    num_inliers = len(R_inliers) if R_inliers is not None else 0
    inlier_ratio = num_inliers / len(correspondences)
    quality = "Good" if inlier_ratio > 0.7 else "Moderate" if inlier_ratio > 0.4 else "Poor"
    
    print(f"  Inliers: {num_inliers}/{len(correspondences)} ({inlier_ratio:.2f}) - {quality} fit")
    
    # 5. ICP refinement
    print("\n5. ICP refinement...")
    start = time.time()
    icp_result = o3d.pipelines.registration.registration_icp(
        src_cloud, dst_cloud, NOISE_BOUND, H,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10)
    )
    T_icp = icp_result.transformation
    timer_print(start, "ICP")
    
    # 6. Results
    print("\n6. Results:")
    print("=" * 30)
    print_results(T_gt, H, "TEASER++")
    print()
    print_results(T_gt, T_icp, "ICP")
    
    # 7. Visualization
    print("\n7. Visualization...")
    
    # Original clouds
    print("  Showing original clouds (Red: Source, Green: Target)")
    o3d.visualization.draw_geometries([
        src_cloud.paint_uniform_color([1, 0, 0]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="Original Point Clouds")
    
    # Correspondences
    if len(correspondences) > 0:
        print("  Showing correspondences")
        visualize_correspondences(src_down, dst_down, correspondences)
    
    # TEASER++ result
    print("  Showing TEASER++ result (Blue: Aligned, Green: Target)")
    src_teaser = copy.deepcopy(src_cloud)
    src_teaser.transform(H)
    o3d.visualization.draw_geometries([
        src_teaser.paint_uniform_color([0, 0, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="TEASER++ Registration")
    
    # ICP result
    print("  Showing ICP result (Cyan: Aligned, Green: Target)")
    src_icp = copy.deepcopy(src_cloud)
    src_icp.transform(T_icp)
    o3d.visualization.draw_geometries([
        src_icp.paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")
    
    print(f"\n{Fore.GREEN}Registration pipeline completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()