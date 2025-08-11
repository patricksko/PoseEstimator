import cv2
import os
from ultralytics import YOLO

def generate_segmentation_polygons_yolo_format(weights_path, image_path, output_folder, class_id=0):
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
    )

