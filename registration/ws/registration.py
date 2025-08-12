import numpy as np
import open3d as o3d
from EstimHelpers.registration_utils import run_teaser, get_correspondences, get_pointcloud, preprocess_point_cloud_uniform
import copy
import time
from colorama import Fore, Style

VOXEL_SIZE = 0.05
VISUALIZE = True
NOISE_BOUND = VOXEL_SIZE


def tic():
    return time.time()

def toc(start_time, label=""):
    print(Fore.GREEN + f"Elapsed time for {label}: {time.time() - start_time:.3f} seconds" + Style.RESET_ALL)


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

if __name__ == "__main__":
    # Load images
    # cad_path = "./registration/ws/LegoBlock_variant.ply"
    cad_path = "./LegoBlock_variant.ply"
    
    theta_x = np.deg2rad(5)
    theta_z = np.deg2rad(10)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    R = Rz @ Rx  # rotate X then Z

    T = np.eye(4)
    T[:3, :3] = R
    # T[:3, 3] = [-0.115576939, -0.0387705398, 0.114874890]
    T[:3, 3] = [0.1, -0.05, 0.0]  # Example translation
    

    from EstimHelpers.registration_utils import (
        preprocess_point_cloud_uniform,
        crop_pointcloud_fov_hpr,
        apply_transform_and_noise,
        sample_pointcloud_with_noise,
        visualize_correspondences
    )

    # Load CAD model as mesh
    cad_mesh = o3d.io.read_triangle_mesh(cad_path)
    cad_mesh.compute_vertex_normals()

    # Simulate different sampling for source & target
    src_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=500, jitter_sigma=0.0)
    dst_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=500, jitter_sigma=0.0005)

    print("nr of voxel points:", len(src_mesh_sample.points))
    print("nr of voxel points:", len(dst_mesh_sample.points))
    # dst_mesh_sample = src_mesh_sample

    # Apply FoV + HPR separately
    camera_pos_src = np.array([0.3, 0.2, -0.5])
    camera_target_src = np.array([0, 0, 0])
    src_cloud = crop_pointcloud_fov_hpr(
        src_mesh_sample,
        camera_position=camera_pos_src,
        camera_lookat=camera_target_src,
        fov_deg=60.0
    )

    # camera_pos_dst = np.array([-0.2, 0.1, -0.6])
    camera_pos_dst = camera_pos_src
    camera_target_dst = np.array([0, 0, 0])
    dst_cloud = crop_pointcloud_fov_hpr(
        dst_mesh_sample,
        camera_position=camera_pos_dst,
        camera_lookat=camera_target_dst,
        fov_deg=60.0
    )

    # Apply transformation & noise to destination
    dst_cloud = apply_transform_and_noise(dst_cloud, R, T[:3, 3], noise_sigma=0.0002)


    # FPFH Correspondences
    # Downsample & compute features
    t1 = tic()
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 30)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 30)
    toc(t1, "downsample")

    t2 = tic()
    correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=0.5)
    toc(t2, "get correspondences")

    print("Number of correspondences found:", len(correspondences))



    o3d.visualization.draw_geometries([src_cloud.paint_uniform_color([1, 0, 0]),
                                        dst_cloud.paint_uniform_color([0, 1, 0])])
    


    # Example usage:
    visualize_correspondences(src_down, dst_down, correspondences)


    t3 = tic()
    H, R_inliers, s_inliers, t_inliers = run_teaser(src_down, dst_down, correspondences)
    toc(t3, "run_teaser")

    final_template = copy.deepcopy(src_cloud)
    final_template.transform(H)


    # Visualize the registration results
    A_pcd = copy.deepcopy(src_cloud)
    A_pcd_T_teaser = copy.deepcopy(src_cloud).transform(H)
    B_pcd = copy.deepcopy(dst_cloud)
    o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])

    # local refinement using ICP
    t4 = tic()
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, H,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    toc(t4, "icp")
    T_icp = icp_sol.transformation

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd).transform(T_icp)
    o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd])

    print(Fore.RED + "H: " + str(H) + Style.RESET_ALL)
    print(Fore.RED + "T_icp: " + str(T_icp) + Style.RESET_ALL)

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")


    print("TEASER++ Inliers:")
    print(f"  Rotation inliers: {len(R_inliers) if R_inliers is not None else 0}")
    print(f"  Scale inliers: {len(s_inliers) if s_inliers is not None else 0}")
    print(f"  Translation inliers: {len(t_inliers) if t_inliers is not None else 0}")

    num_correspondences = len(correspondences)
    num_rot_inliers = len(R_inliers) if R_inliers is not None else 0

    if num_correspondences > 0:
        inlier_ratio = num_rot_inliers / num_correspondences
    else:
        inlier_ratio = 0
    print("Correspondences: ", len(correspondences))
    print("Rotation inliers: ", num_rot_inliers)
    print(f"Inlier ratio (rotation): {inlier_ratio:.2f}")

    # Optional thresholds for quality check
    if inlier_ratio > 0.7:
        print("Good fit")
    elif inlier_ratio > 0.4:
        print("Moderate fit")
    else:
        print("Poor fit")



    # print("Expected rotation: ")
    # print(T[:3, :3])
    # print("Estimated rotation: ")
    # print(H[:3, :3])
    print("Error (rad): ")
    print(get_angular_error(T[:3,:3], H[:3, :3]))

    print("Expected translation: ")
    print(T[:3, 3])
    print("Estimated translation: ")
    print(H[:3, 3])
    print("Error (m): ")
    print(np.linalg.norm(T[:3, 3] - H[:3, 3]))

    print("--------------------")
    print("Error (rad): ")
    print(get_angular_error(T[:3,:3], T_icp[:3, :3]))

    print("Expected translation: ")
    print(T[:3, 3])
    print("Estimated translation: ")
    print(T_icp[:3, 3])
    print("Error (m): ")
    print(np.linalg.norm(T[:3, 3] - T_icp[:3, 3]))

    # o3d.visualization.draw_geometries([dst_cloud.paint_uniform_color([1, 0, 0]),
    #                                     final_template.paint_uniform_color([0, 1, 0])])
    