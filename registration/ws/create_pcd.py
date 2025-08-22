import os
import glob
import numpy as np
import cv2
import open3d as o3d

def create_pointcloud(rgb_path, depth_path, depth_scale=1.0, is_depth_npy=True):
    # Load RGB image
    color_raw = cv2.imread(rgb_path)
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)

    # Load depth image
    if is_depth_npy:
        depth = np.load(depth_path).astype(np.float32)
        depth = depth
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth /= depth_scale

    color_o3d = o3d.geometry.Image(color_raw)
    depth_o3d = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    width, height = 640, 480
    fx, fy = 500.0, 500.0
    cx, cy = 323.7872, 279.6921
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # pcd.transform([[1, 0, 0, 0],
    #                [0, -1, 0, 0],
    #                [0, 0, -1, 0],
    #                [0, 0, 0, 1]])

    return pcd


# Example usage
folder = "../../data/lego_views"

rgb_files = sorted(glob.glob(os.path.join(folder, "rgb_*.png")))
depth_files = sorted(glob.glob(os.path.join(folder, "depth_*.npy")))

# Check matching count
assert len(rgb_files) == len(depth_files), "Mismatch in number of RGB and depth files"


for rgb_path, depth_path in zip(rgb_files, depth_files):
    filename = os.path.splitext(os.path.basename(rgb_path))[0].replace("rgb_", "pcd_")
    pcd = create_pointcloud(rgb_path, depth_path)
    o3d.io.write_point_cloud(os.path.join(folder, f"{filename}.ply"), pcd)
    print(f"Saved point cloud: {filename}.ply with {len(pcd.points)} points")


# Visualize