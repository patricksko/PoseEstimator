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
        conf=0.8,
        device="0",
        save=True,
    )

if __name__ == "__main__":
    weights = "/home/skoumal/dev/ObjectDetection/data/best.pt"
    img_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output/output_blenderproc/bop_data/Legoblock_full/train_pbr/002014/rgb/000000.jpg"
    out_dir = "/home/skoumal/dev/ObjectDetection/data/"

    generate_segmentation_polygons_yolo_format(weights, img_path, out_dir)
