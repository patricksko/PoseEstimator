import numpy as np                        # fundamental package for scientific computing
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from EstimHelpers.TheRetector import TheRetector
from ultralytics import YOLO
import cv2

WEIGHTS_PATH = "./data/best.pt"
PCD_PATH = "./data/lego_views/"
CAD_PATH = "./data/obj_000001.ply"
TARGET_PTS = 100
TRACK_EVERY = 1

def main():
    # Setup:
    # Check if realsense is connected
    ctx = rs.context()
    if len(ctx.devices) == 0:
        raise RuntimeError("❌ No Intel RealSense device connected.")
    
    # Configure Realsense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # enable depth stream
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # enable color stream
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()  # get depth scale of realsense
    
    # Filters of Realsense
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_fill = rs.hole_filling_filter()

    # Preload YOLO model once
    yolo_model = YOLO(WEIGHTS_PATH)
    detector = TheRetector(model=yolo_model, profile=profile, class_id=0, conf=0.8)

    intr, K = detector.rs_get_intrinsics()  # get intrinsic info and intrinsic camera matrix

    try:
        while True:
            # Read Camera frames (color and depth) and align them
            frameset = align.process(pipe.wait_for_frames())
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Preprocess the images
            frame = depth_frame
            frame = spatial.process(frame)
            frame = temporal.process(frame)
            frame = hole_fill.process(frame)

            # Convert to numpy
            depth = np.asanyarray(frame.get_data())  # uint16 depth in sensor units
            color = np.asanyarray(color_frame.get_data())  # BGR

            # Detection
            masks = detector.detect_masks(color)

            # Overlay
            overlay_img = detector.overlay_masks(color, masks)

            # Show result
            cv2.imshow("Detections", overlay_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        pipe.stop()

if __name__ == "__main__":
    main()
