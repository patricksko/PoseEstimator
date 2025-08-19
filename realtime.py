#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from EstimHelpers.HelpersRealtime import *
import glob

# ---------- RealSense + YOLO live demo ----------
def main(weights_path, class_id=0, device="0", conf=0.7, imgsz=640):
    # Configure RealSense pipeline
    cad_path = "./data/lego_views/"

    ply_files = sorted(glob.glob(os.path.join(cad_path, "*.ply")))

    src_clouds = []
    for ply_file in ply_files:
        src = o3d.io.read_point_cloud(ply_file)
        src_clouds.append(src)
        print(f"Loaded: {ply_file} with {len(src.points)} points")

    pipeline = rs.pipeline()
    config = rs.config()
    # Ask for color + depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    profile = pipeline.start(config)
    intr, K = rs_get_intrinsics(profile=profile)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    
    # Preload YOLO model once
    yolo_model = YOLO(weights_path)

    t0 = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth = np.asanyarray(depth_frame.get_data())  # uint16 depth in sensor units
            color = np.asanyarray(color_frame.get_data())  # BGR

            mask = detect_mask(yolo_model, color)
            # Convert depth to meters and apply mask
            depth_m = depth.astype(np.float32) * depth_scale
            depth_m[mask == 0] = 0.0

            depth_o3d = o3d.geometry.Image(depth_m)
            color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
)
            # Generate point cloud
            dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
            best_idx, H, best_inliers, all_metrics = find_best_template_teaser(dst_cloud, src_clouds, target_points=200)
            print("="*50)
            for m in all_metrics:
                print(f"Template {m['template_idx']}: Chamfer = {m['chamfer']:.6f}")

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
            exit()
            # Stack and show
            top = np.hstack([color, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            vis = top

            fps = 1.0 / (time.time() - t0)
            t0 = time.time()
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("RealSense + YOLO (color | mask)  /  (overlay | masked depth)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change weights path as needed; class_id=0 by default
    main(weights_path="./data/best.pt", class_id=0, device="0", conf=0.7, imgsz=640)
