import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO  # YOLOv8 library

# ----------------- Load YOLO model -----------------
MODEL_PATH = "./data/best.pt"  # Change to your .pt file
model = YOLO(MODEL_PATH)

# ----------------- Setup RealSense pipeline -----------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# ----------------- Main loop -----------------
try:
    while True:
        # Get frames from camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO inference directly on the frame
        results = model(frame, conf=0.8)  # conf is confidence threshold

        # Draw results on the frame
        annotated_frame = results[0].plot()  # YOLO provides annotated frame automatically

        # Show the frame
        cv2.imshow("YOLO Live Detection", annotated_frame)

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()