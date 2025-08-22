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
import os


VOXEL_SIZE = 0.05
NOISE_BOUND = VOXEL_SIZE


def camera_eye_lookat_up_from_H(H):
    """
    Convert model→camera H to eye/target/up in model/world coords.
    - eye: camera center
    - target: eye + forward vector
    - up: image up
    """
    R = H[:3, :3]
    t = H[:3, 3]

    # camera center in world
    eye = -R.T @ t

    # camera axes in world
    forward = R.T @ np.array([0, 0, 1.0])   # camera +Z
    up      = R.T @ np.array([0, -1, 0])  # screen up

    target = eye + forward
    up = up / (np.linalg.norm(up) + 1e-12)
    return eye, target, up


def rs_get_intrinsics():
    """Get intrinsics from a running RealSense pipeline profile."""
    # color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    # intr = color_stream.get_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        640,
        480,
        1077.8, 1078.2,
        323.8, 279.7
    )


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
def get_homogenous(gt_data):
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
    return T_m2c

if __name__ == "__main__":
    
    weights = "./data/best.pt"
    rgb_path = "./data/000000.jpg"
    depth_path = "./data/000000.png"
    scene_camera_path = "./data/scene_camera.json"

    mask = detect_mask(weights, rgb_path)
    intr_o3d = rs_get_intrinsics()
    
    gt_data = "./data/scene_gt.json"
    # #Read CAD Path
    cad_path = "./data/obj_000001.ply"
    if not os.path.exists(cad_path):
        raise FileNotFoundError(cad_path)

    mesh = o3d.io.read_triangle_mesh(cad_path)
    mesh.compute_vertex_normals()

    H_mm = get_homogenous(gt_data)
    H_gt = H_mm.copy()
    H_gt[:3,3] *= 0.001   # mm → m

    dst_cloud = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask=mask)
    if dst_cloud is None or len(dst_cloud.points) == 0:
            print("Failed to generate scene point cloud!")
            exit(1)
    w, h = intr_o3d.width, intr_o3d.height
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # white

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    # camera intrinsics (from RealSense, or use synthetic FOV)
    #fx, fy, cx, cy = intr_o3d.get_focal_length()[0], intr_o3d.get_focal_length()[1], intr_o3d.get_principal_point()[0], intr_o3d.get_principal_point()[1]
    near, far = 0.01, 2.0
    K = intr_o3d.intrinsic_matrix
    scene.camera.set_projection(K, near, far, w, h)

    # ------------------------
    # Extrinsics
    # ------------------------
    eye, target, up = camera_eye_lookat_up_from_H(H_gt)
    scene.camera.look_at(target, eye, up)

    # ------------------------
    # Render
    # ------------------------
    output_dir = "./../../data/lego_homogenous"
    os.makedirs(output_dir, exist_ok=True)

    color_img = renderer.render_to_image()
    depth_img = renderer.render_to_depth_image(z_in_view_space=True)

    # rgb_path   = os.path.join(output_dir, f"rgb.png")
    # o3d.io.write_image(rgb_path, color_img)

    # Backproject depth to point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
    )
    src_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)

    
    # 1. Preprocessing
    print("\n 1. Preprocessing")
    start = time.time()
    src_down, src_fpfh = preprocess_point_cloud_uniform(src_cloud, 100)
    dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 100)
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
    T_icp = T_icp@H_gt
    o3d.visualization.draw_geometries([
        src_icp.paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")
    
    print(f"\n{Fore.GREEN}Registration pipeline completed!{Style.RESET_ALL}")

    
    T_est = T_icp
    gt_data = "./data/scene_gt.json"
    
    


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
    plot_frame(ax, H_mm[:3,:3], origin=[0,0,0], label='R')
    plot_frame(ax, T_est[:3,:3], origin=[0,0,0.5], label='R_m2c')
    plt.show()

    print("Homogeneous Transformation:\n", H_mm)
    print("Estimated: ", T_est)
    print("Difference = ", get_angular_error(H_gt[:3,:3], T_est[:3,:3]))


        

 