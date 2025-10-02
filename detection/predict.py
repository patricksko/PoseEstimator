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

    # result = results[0]

    # if result.masks is not None and result.masks.xy is not None:
    #     polygons = result.masks.xy  # list of np arrays with polygon points

    #     os.makedirs(output_folder, exist_ok=True)
    #     txt_path = os.path.join(output_folder, "segmentation_polygons.txt")

    #     with open(txt_path, "w") as f:
    #         for poly in polygons:
    #             # Normalize points by image size
    #             normalized_points = []
    #             for x, y in poly.tolist():
    #                 xn = x / width
    #                 yn = y / height
    #                 normalized_points.extend([xn, yn])

    #             # Format: class_id x1 y1 x2 y2 ...
    #             line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
    #             f.write(line + "\n")

    #     print(f"Saved segmentation polygons in YOLO format to {txt_path}")
    # else:
    #     print("No segmentation polygons found in prediction.")

if __name__ == "__main__":
    weights = "/home/skoumal/dev/ObjectDetection/data/best.pt"
    img_path = "/home/skoumal/dev/ObjectDetection/data/scan2.png"
    out_dir = "/home/skoumal/dev/ObjectDetection/data/"

    generate_segmentation_polygons_yolo_format(weights, img_path, out_dir)
