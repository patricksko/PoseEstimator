import numpy as np
import open3d as o3d
from EstimHelpers.registration_utils import run_teaser, get_correspondences, get_pointcloud, preprocess_point_cloud_uniform
import copy


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));

if __name__ == "__main__":
    # Load images
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
    T[:3, 3] = [-0.115576939, -0.0387705398, 0.114874890]
    
    # src_cloud = o3d.io.read_point_cloud(cad_path)
    # src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 300)
    # dst_cloud = copy.deepcopy(src_cloud)
    # dst_cloud.transform(T)
    # dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 300)


    from EstimHelpers.registration_utils import (
        preprocess_point_cloud_uniform,
        crop_pointcloud_fov_hpr,
        apply_transform_and_noise,
        sample_pointcloud_with_noise
    )

    # Load CAD model as mesh
    cad_mesh = o3d.io.read_triangle_mesh(cad_path)
    cad_mesh.compute_vertex_normals()

    # Simulate different sampling for source & target
    src_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=5000, jitter_sigma=0.0005)
    dst_mesh_sample = sample_pointcloud_with_noise(cad_mesh, num_points=5000, jitter_sigma=0.0005)

    # Apply FoV + HPR separately
    camera_pos_src = np.array([0.3, 0.2, -0.5])
    camera_target_src = np.array([0, 0, 0])
    src_cloud = crop_pointcloud_fov_hpr(
        src_mesh_sample,
        camera_position=camera_pos_src,
        camera_lookat=camera_target_src,
        fov_deg=60.0
    )

    camera_pos_dst = np.array([-0.2, 0.1, -0.6])
    camera_target_dst = np.array([0, 0, 0])
    dst_cloud = crop_pointcloud_fov_hpr(
        dst_mesh_sample,
        camera_position=camera_pos_dst,
        camera_lookat=camera_target_dst,
        fov_deg=60.0
    )

    # Apply transformation & noise to destination
    dst_cloud = apply_transform_and_noise(dst_cloud, R, T[:3, 3], noise_sigma=0.0002)

    # Downsample & compute features
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 300)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 300)






    o3d.visualization.draw_geometries([src_cloud.paint_uniform_color([1, 0, 0]),
                                        dst_cloud.paint_uniform_color([0, 1, 0])])
    correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=0.5)
    H, R_inliers, s_inliers, t_inliers = run_teaser(src_down, dst_down, correspondences)

    final_template = copy.deepcopy(src_cloud)
    final_template.transform(H)

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")

    print("Expected rotation: ")
    print(T[:3, :3])
    print("Estimated rotation: ")
    print(H[:3, :3])
    print("Error (rad): ")
    print(get_angular_error(T[:3,:3], H[:3, :3]))

    print("Expected translation: ")
    print(T[:3, 3])
    print("Estimated translation: ")
    print(H[:3, 3])
    print("Error (m): ")
    print(np.linalg.norm(T[:3, 3] - H[:3, 3]))

    o3d.visualization.draw_geometries([dst_cloud.paint_uniform_color([1, 0, 0]),
                                        final_template.paint_uniform_color([0, 1, 0])])
    