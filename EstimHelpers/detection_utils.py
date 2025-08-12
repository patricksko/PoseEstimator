import cv2
import os
import numpy as np
from ultralytics import YOLO

def detect_mask(weights_path, image_path, class_id=0):
    # Load model
    model = YOLO(weights_path)

    # Load image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    height, width = img.shape[:2]

    # Predict
    results = model(
        source=img,
        imgsz=640,
        conf=0.7,
        device="0",
        save=False,
        verbose=False
    )
    
    # Create an empty binary mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process results (masks if available)
    for r in results:
        if not hasattr(r, "masks") or r.masks is None:
            continue
        
        for seg, cls in zip(r.masks.xy, r.boxes.cls):
            if int(cls) != class_id:
                continue
            polygon = np.array(seg, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
            return mask  # immediately return after first match

    return mask
