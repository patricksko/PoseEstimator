#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from EstimHelpers.HelpersRealtime import *
import glob

def check_delta(delta, max_trans=0.1):
    """
    Check if delta transformation is reasonable.
    max_trans: maximum allowed translation [m]
    max_angle_deg: maximum allowed rotation [deg]
    """
    # Extract translation
    t = delta[:3, 3]
    trans_norm = np.linalg.norm(t)



    return trans_norm <= max_trans, trans_norm
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

# ---------- RealSense + YOLO live demo ----------
def main():
    # Configure RealSense pipeline
    weights_path="./data/best.pt"
    cad_path = "./data/lego_views/"
    cad_model = o3d.io.read_point_cloud("./data/obj_000001.ply")
    cad_points = np.asarray(cad_model.points)

    #Preload Templates for Template matching
    ply_files = sorted(glob.glob(os.path.join(cad_path, "*.ply")))
    if not ply_files:
            raise RuntimeError(f"No PLYs in {cad_path}")
    src_clouds = []
    for ply_file in ply_files:
        src = o3d.io.read_point_cloud(ply_file)
        src_clouds.append(src)
        print(f"Loaded: {ply_file} with {len(src.points)} points")
    
    # Check if realsense is connected
    ctx = rs.context()
    if len(ctx.devices) == 0:
        raise RuntimeError("❌ No Intel RealSense device connected.")
    
    # Configure realsense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    intr, K = rs_get_intrinsics(profile=profile) # get intrinsic info and intrinsic camera matrix

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() # get depth scale of realsense
   
    # Preload YOLO model once
    yolo_model = YOLO(weights_path)
    
    prev_down = None 
    frame_counter = 0
    initialized = False # Variable for initial orientation
    frame_id = 0

    try:
        while True:
            if not initialized:
                # align frames and read depth and rgb data
                frames = align.process(pipeline.wait_for_frames())
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert to numpy
                depth = np.asanyarray(depth_frame.get_data())  # uint16 depth in sensor units
                color = np.asanyarray(color_frame.get_data())  # BGR
                
                while frame_counter!=10:
                    mask = detect_mask(yolo_model, color)
                    if mask is None:
                        frame_counter = 0
                    
                    frames = align.process(pipeline.wait_for_frames())
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    color = np.asanyarray(color_frame.get_data())
                    depth = np.asanyarray(depth_frame.get_data())
                    frame_counter+=1
                
                depth_m = depth.astype(np.float32) * depth_scale
                depth_m[mask == 0] = 0.0 

                # # Visuals
                # overlay = color.copy()
                # overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)

                # # Optionally visualize masked depth as heatmap
                # masked_depth = np.where(mask > 0, depth, 0)
                # # Scale depth for viz (in meters -> normalize to 0..255)
                # depth_m = masked_depth.astype(np.float32) * depth_scale
                # d_norm = np.clip((depth_m / np.nanmax(depth_m[depth_m > 0]) if np.any(depth_m > 0) else depth_m), 0, 1)
                # depth_viz = (d_norm * 255).astype(np.uint8)
                # depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)

                # # Stack and show
                # top = np.hstack([color, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
                # bottom = np.hstack([overlay, depth_viz])
                # vis = np.vstack([top, bottom])

                # cv2.imshow("RealSense + YOLO (color | mask)  /  (overlay | masked depth)", vis)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                
                # Create RGBD (Open3D expects RGB, depth in meters here because depth_scale=1.0)
                depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
                )
                dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
                
                # First guess with templates
                best_idx, H_init, dst_down, dst_fpfh = find_best_template_teaser(dst_cloud, src_clouds, target_points=50)
                print(H_init)
                # Candidate CAD for tracking going forward
                cand_cad = copy.deepcopy(src_clouds[best_idx])
                cand_cad.transform(H_init)

                # Show candidate guess
                o3d.visualization.draw_geometries([
                    cand_cad.paint_uniform_color([0, 1, 1]),
                    dst_cloud.paint_uniform_color([0, 1, 0])
                ], window_name="Init registration (ENTER=accept, SPACE=reject)")

                print("[INFO] Press Y/y to accept this guess. Or something else to try again")

                nb = input('Enter your input:')
                if nb.lower() == 'y':  
                    print("[INFO] Initial pose accepted.")
                    del src_clouds
                    src_clouds = None
                    initialized = True
                    T_m2c = H_init.copy()
                    continue
                else:
                    print("[INFO] Initial pose rejected. Retrying...")
                    frame_counter = 0
                    continue
            print(T_m2c)

            cam = camera_eye_lookat_up_from_H(T_m2c)
            src = crop_pointcloud_fov_hpr(
                    cad_model,
                    camera_position=cam["eye"],
                    camera_lookat=cam["target"],
                    fov_deg=60.0
                )
            prev_down, prev_fpfh = preprocess_point_cloud_uniform(src, 50)
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth = np.asanyarray(depth_frame.get_data())  # uint16 depth in sensor units
            color = np.asanyarray(color_frame.get_data())  # BGR
            frame_id += 1
            if frame_id % 3 == 0:
                start = time.time()
                mask = detect_mask(yolo_model, color)
                depth_m = depth.astype(np.float32) * depth_scale
                depth_m[mask == 0] = 0.0  

                # Create RGBD (Open3D expects RGB, depth in meters here because depth_scale=1.0)
                depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
                )
                dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
                dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 50)
                corr = get_correspondences(prev_down, dst_down, prev_fpfh, dst_fpfh)
                T_m2c = run_teaser(prev_down, dst_down, corr)

                # ok, trans_norm = check_delta(delta, max_trans=0.5)
                # if not ok:
                #     print(f"[WARN] Rejecting delta. Translation={trans_norm:.3f} m")
                #     delta = np.eye(4)  # ignore update
                end = time.time()
                print("="*50)
                print(end-start)
                #T_m2c = delta @ T_m2c
                
                
                # o3d.visualization.draw_geometries([
                #     dst_down.paint_uniform_color([0, 1, 1]),
                #     prev_down.paint_uniform_color([0, 1, 0])
                # ], window_name="Init registration (ENTER=accept, SPACE=reject)")
            uv = project_points(cad_points, K, T_m2c)
            for (u, v) in uv:
                if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                    cv2.circle(color, (u, v), 1, (0, 0, 255), -1)  # red dots

            cv2.imshow("Live Tracking", color)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
