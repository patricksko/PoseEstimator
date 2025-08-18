import numpy as np
import open3d as o3d
from teaserpp_python import _teaserpp as tpp
import copy
import time
from EstimHelpers.registration_utils import run_teaser, get_pointcloud, visualize_correspondences, preprocess_point_cloud_uniform, get_correspondences, pyrender_to_open3d
from EstimHelpers.detection_utils import detect_mask
from colorama import Fore, Style
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


VOXEL_SIZE = 0.05
NOISE_BOUND = VOXEL_SIZE


def get_angular_error(R_exp, R_est):
    """Calculate angular error in radians"""
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)))

def timer_print(start_time, label):
    """Clean timer output"""
    elapsed = time.time() - start_time
    print(Fore.GREEN + f"  {label}: {elapsed:.3f}s" + Style.RESET_ALL)
    return elapsed

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

def plot_frame(ax, R, origin=[0,0,0], label='Frame', length=1.0):
    """Plot a 3D coordinate frame given a rotation matrix R and origin."""
    origin = np.array(origin)
    x_axis = origin + R[:,0] * length
    y_axis = origin + R[:,1] * length
    z_axis = origin + R[:,2] * length
    
    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r', label=f'{label}-X')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g', label=f'{label}-Y')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b', label=f'{label}-Z')
if __name__ == "__main__":

    #Read CAD Path
    cad_path = "/home/skoumal/dev/TEASER-plusplus/build/python/lego_views/"
    
    weights = "./data/best.pt"
    rgb_path = "./data/000000.jpg"
    depth_path = "./data/000000.png"
    scene_camera_path = "./data/scene_camera.json"

    mask = detect_mask(weights, rgb_path)
    
    gt_data = "./data/scene_gt.json"
    # #Read CAD Path
    cad_path = "./data/lego_views/pointcloud_14_edge.ply"
    src_cloud = o3d.io.read_point_cloud(cad_path)
    dst_cloud = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)
    if dst_cloud is None or len(dst_cloud.points) == 0:
            print("Failed to generate scene point cloud!")
            exit(1)
    
    # 1. Preprocessing
    print("\n 1. Preprocessing")
    start = time.time()
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 50)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 50)
    timer_print(start, "Downsampling & features")
    
    print(f"  Downsampled to {len(src_down.points)} and {len(dst_down.points)} points")
    
    # 2. Find correspondences
    correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=0.5)
    timer_print(start, "Correspondence search")
    print(f"  Found {len(correspondences)} correspondences")
    
    if len(correspondences) < 3:
        print("  ERROR: Too few correspondences!")
        exit(1)
    
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

    # 7. Visualization
    print("\n7. Visualization...")
    
    # Original clouds
    # print("  Showing original clouds (Red: Source, Green: Target)")
    # o3d.visualization.draw_geometries([
    #     src_cloud.paint_uniform_color([1, 0, 0]),
    #     dst_cloud.paint_uniform_color([0, 1, 0])
    # ], window_name="Original Point Clouds")
    
    # Correspondences
    # if len(correspondences) > 0:
    #     print("  Showing correspondences")
    #     visualize_correspondences(src_down, dst_down, correspondences)
    
    # TEASER++ result
    print("  Showing TEASER++ result (Blue: Aligned, Green: Target)")
    src_teaser = copy.deepcopy(src_cloud)
    src_teaser.transform(H)
    # o3d.visualization.draw_geometries([
    #     src_teaser.paint_uniform_color([0, 0, 1]),
    #     dst_cloud.paint_uniform_color([0, 1, 0])
    # ], window_name="TEASER++ Registration")
    
    # ICP result
    print("  Showing ICP result (Cyan: Aligned, Green: Target)")
    src_icp = copy.deepcopy(src_cloud)
    src_icp.transform(T_icp)
    o3d.visualization.draw_geometries([
        src_icp.paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")
    
    print(f"\n{Fore.GREEN}Registration pipeline completed!{Style.RESET_ALL}")

    
    T_m2c_est = T_icp
    R_m2c_est = np.array(T_m2c_est[:3, :3]).reshape(3, 3)
    t_m2c_est = np.array(T_m2c_est[:3, 3]).reshape(3) * 1000.0  # <-- mm → m

    T_m2c_est = np.eye(4)
    T_m2c_est[:3, :3] = R_m2c_est
    T_m2c_est[:3, 3] = t_m2c_est
    R_m2c_est = T_m2c_est[:3, :3]
    gt_data = "./data/scene_gt.json"
    
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

    T_m2c = np.eye(4)
    T_m2c[:3, :3] = R
    T_m2c[:3, 3] = t[:, 0]  # flatten column vector into row

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotation Matrices in 3D')

    # Plot the frames
    plot_frame(ax, R, origin=[0,0,0], label='R')
    plot_frame(ax, R_m2c_est, origin=[0,0,0.5], label='R_m2c')
    plt.show()

    print("Homogeneous Transformation:\n", T_m2c)
    print("Estimated: ", T_m2c_est)
    print("Difference = ", get_angular_error(R, R_m2c_est))


        

 