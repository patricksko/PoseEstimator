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
    rgb_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000000.jpg"
    depth_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000000.png"
    mask_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000000_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"
    
    T = np.array(
        [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
         [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
         [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
         [0, 0, 0, 1]])
    

    src_cloud = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask_path)
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 300)
    dst_cloud = copy.deepcopy(src_cloud)
    dst_cloud.transform(T)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 300)

    o3d.visualization.draw_geometries([src_cloud.paint_uniform_color([1, 0, 0]),
                                        dst_cloud.paint_uniform_color([0, 1, 0])])
    correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh)
    H, R_inliers, s_inliers, t_inliers = run_teaser(src_down, dst_down, correspondences)

    final_template = copy.deepcopy(src_cloud)
    final_template.transform(T)

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


    