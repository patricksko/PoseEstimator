#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ---------- YOLO mask helpers (keep your API + ndarray version) ----------
def detect_mask(weights_path, image_path, class_id=0, device="0", conf=0.7, imgsz=640):
    model = YOLO(weights_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return _detect_mask_core(model, img, class_id, device, conf, imgsz)

def detect_mask_from_array(weights_path, img_bgr, class_id=0, device="0", conf=0.7, imgsz=640):
    model = YOLO(weights_path)
    return _detect_mask_core(model, img_bgr, class_id, device, conf, imgsz)

def _detect_mask_core(model, img_bgr, class_id, device, conf, imgsz):
    h, w = img_bgr.shape[:2]
    results = model(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        verbose=False
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        # segmentation masks available?
        if not hasattr(r, "masks") or r.masks is None:
            continue
        for seg, cls in zip(r.masks.xy, r.boxes.cls):
            if int(cls) != class_id:
                continue
            polygon = np.array(seg, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            # return first match (your original behavior)
            return mask
    return mask

# ---------- RealSense + YOLO live demo ----------
def main(weights_path, class_id=0, device="0", conf=0.9, imgsz=640):
    # Configure RealSense pipeline
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
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # meters per unit

    print(f"[INFO] RealSense started. Depth scale = {depth_scale:.6f} m/unit")
    print("[INFO] Press 'q' to quit.")

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

            # Run YOLO segmentation on the color frame (ndarray path)
            results = yolo_model(
                source=color,
                imgsz=imgsz,
                conf=conf,
                device=device,
                save=False,
                verbose=False
            )

            mask = np.zeros(color.shape[:2], dtype=np.uint8)
            for r in results:
                if not hasattr(r, "masks") or r.masks is None:
                    continue
                for seg, cls in zip(r.masks.xy, r.boxes.cls):
                    if int(cls) != class_id:
                        continue
                    polygon = np.array(seg, dtype=np.int32)
                    cv2.fillPoly(mask, [polygon], 255)
                    break  # first matching instance
                if mask.any():
                    break

            # Visuals
            overlay = color.copy()
            overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)

            # Optionally visualize masked depth as heatmap
            masked_depth = np.where(mask > 0, depth, 0)
            # Scale depth for viz (in meters -> normalize to 0..255)
            depth_m = masked_depth.astype(np.float32) * depth_scale
            d_norm = np.clip((depth_m / np.nanmax(depth_m[depth_m > 0]) if np.any(depth_m > 0) else depth_m), 0, 1)
            depth_viz = (d_norm * 255).astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)

            # Stack and show
            top = np.hstack([color, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            bottom = np.hstack([overlay, depth_viz])
            vis = np.vstack([top, bottom])

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
