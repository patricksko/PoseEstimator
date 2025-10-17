import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor, as_completed

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("❌ No Intel RealSense device connected.")
        
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # filters
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_fill = rs.hole_filling_filter()

        self.color = None
        self.depth = None
        self.intr = None

    def get_rgbd(self):
        frameset = self.align.process(self.pipe.wait_for_frames())
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # apply filters
        frame = self.spatial.process(depth_frame)
        frame = self.temporal.process(frame)
        frame = self.hole_fill.process(frame)

        self.depth = np.asanyarray(frame.get_data())
        self.color = np.asanyarray(color_frame.get_data())
        return self.color

    def rs_get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intr = color_stream.get_intrinsics()
        K = np.array([[self.intr.fx, 0, self.intr.ppx],
                      [0, self.intr.fy, self.intr.ppy],
                      [0, 0, 1]], dtype=np.float32)
        return self.intr, K

    def stop(self):
        self.pipe.stop()

    
    def get_pcd_from_rgbd(self, mask):
        
        depth_m = self.depth.astype(np.float32) * self.depth_scale # mind depth scale of realsense
        depth_m[mask == 0] = 0.0 # create a mask of the depth image for pointcloud

        depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32)) # create open3d depth image for later processing
        color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)) # create open3d color image for later processing
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
        self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
    ))
        dst_cloud, _ = dst_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        return dst_cloud
    
    # def get_pcd_from_rgbd(self, masks):
    #     if isinstance(masks, list):
    #         masks = np.stack(masks, axis = 0)

    #     base_depth = self.depth.astype(np.float32) * self.depth_scale # mind depth scale of realsense

    #     color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)) # create open3d color image for later processing
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         self.intr.width,
    #         self.intr.height,
    #         self.intr.fx,
    #         self.intr.fy,
    #         self.intr.ppx,
    #         self.intr.ppy
    #     )
    #     dst_clouds = []
    #     for mask in masks:
    #         depth_m = np.where(mask > 0, base_depth, 0.0)
    #         depth_o3d = o3d.geometry.Image(depth_m)
    #         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #             color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    #         )
    #         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    #         pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    #         dst_clouds.append(pcd)
    #     return dst_clouds
    # def get_pcd_from_rgbd_mf(self, masks):
    #     if isinstance(masks, list):
    #         masks = np.stack(masks, axis = 0)

    #     color_rgb = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
    #     base_depth = self.depth.astype(np.float32) * self.depth_scale

    #     color_o3d = o3d.geometry.Image(cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)) # create open3d color image for later processing
    #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #         self.intr.width,
    #         self.intr.height,
    #         self.intr.fx,
    #         self.intr.fy,
    #         self.intr.ppx,
    #         self.intr.ppy
    #     )
    #     def process_mask(mask):
    #         """Process a single mask → point cloud."""
    #         depth_m = np.where(mask > 0, base_depth, 0.0)
    #         depth_o3d = o3d.geometry.Image(depth_m)
    #         color_o3d = o3d.geometry.Image(color_rgb)
    #         rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #             color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
    #         )
    #         pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    #         pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    #         return pcd

    #     dst_clouds = []
    #     with ThreadPoolExecutor(max_workers=8) as executor:
    #         futures = [executor.submit(process_mask, m) for m in masks]
    #         for f in as_completed(futures):
    #             dst_clouds.append(f.result())

    #     return dst_clouds