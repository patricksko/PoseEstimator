import numpy as np
import open3d as o3d
import json
from teaserpp_python import _teaserpp as tpp
import copy
import time
import glob
import os
from EstimHelpers.registration_utils import find_best_template_teaser, get_pointcloud,get_angular_error
from EstimHelpers.detection_utils import detect_mask
import cv2 
import matplotlib.pyplot as plt

######## Global Variables  #########
WEIGHTS = "./data/best.pt"
RGB_PATH = "./data/000000.jpg"
DEPTH_PATH = "./data/000000.png"
SCENE_CAMERA_PATH = "./data/scene_camera.json"
CAD_PATH = "./data/lego_views/"
GT_DATA = "./data/scene_gt.json"
PLY = "./data/obj_000001.ply"

def project_points(points_3d, K, T_m2c):
    """Project 3D points into image pixels."""
    pts_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Nx4
    pts_cam = (T_m2c @ pts_h.T).T[:, :3]  # Nx3 in camera frame
    
    # Filter only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    
    # Project
    uv = (K @ (pts_cam.T)).T  # Nx3
    uv = uv[:, :2] / uv[:, 2:3]
    return uv.astype(int)


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
    mask = detect_mask(WEIGHTS, RGB_PATH)   # Detect mask of Object with YOLO Network
    rgb_img = cv2.imread(RGB_PATH)
    color = np.asarray(rgb_img)
    # Read Pointcloud Templates and append them to list (Source)
    ply_files = sorted(glob.glob(os.path.join(CAD_PATH, "*.ply")))  
    src_clouds = []
    for ply_file in ply_files:
        src = o3d.io.read_point_cloud(ply_file)
        src_clouds.append(src)
        print(f"Loaded: {ply_file} with {len(src.points)} points")
    
    # Create Pointcloud of Destination
    dst_cloud, K = get_pointcloud(DEPTH_PATH, RGB_PATH, SCENE_CAMERA_PATH, mask=mask)
    if dst_cloud is None or len(dst_cloud.points) == 0:
            print("Failed to generate scene point cloud!")
            exit(1)

    K = np.asarray(K).reshape(3,3)
    best_idx, H, best_inliers, all_metrics = find_best_template_teaser(dst_cloud, src_clouds, target_points=200)
    for m in all_metrics:
        print(f"Template {m['template_idx']}: Chamfer = {m['score']:.6f}")
    print(best_idx)
    # Apply final transformation
    if best_idx >= 0:
        src_cloud = copy.deepcopy(src_clouds[best_idx])
        src_cloud.transform(H)
        
    # 7. Visualization
    print("\n7. Visualization...")
  
    o3d.visualization.draw_geometries([
        src_cloud.paint_uniform_color([0, 1, 1]),
        dst_cloud.paint_uniform_color([0, 1, 0])
    ], window_name="ICP Refined Registration")
    

    
    T_m2c_est = H
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
    plot_frame(ax, R_m2c_est, origin=[0,0,2], label='R_m2c')
    plt.show()

    print("Homogeneous Transformation:\n", T_m2c)
    print("Estimated: ", T_m2c_est)
    print("Difference = ", get_angular_error(R, R_m2c_est))
    
    # Read CAD Model for comparision
    cad_model = o3d.io.read_point_cloud(PLY)
    cad_points = np.asarray(cad_model.points)

    uv = project_points(cad_points, K, T_m2c_est)
    for (u, v) in uv:
        if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
            cv2.circle(color, (u, v), 1, (0, 0, 255), -1)  # red dots

    cv2.imshow("Live Tracking", color)
    cv2.waitKey(0)

    # Then close the window
    cv2.destroyAllWindows()


        

 



    
        

 