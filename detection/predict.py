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
        save=False,
        show=True,
    )
    

if __name__ == "__main__":
    weights = "./data/best.pt"
    img_path = "./data/_Daten_Seibersdorf_Patrick/exported_rgb_pcl/1765975160.708315.png"
    out_dir = "/home/skoumal/dev/ObjectDetection/data/"

    generate_segmentation_polygons_yolo_format(weights, img_path, out_dir)
