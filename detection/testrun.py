import cv2
import numpy as np

def draw_yolo_polygons(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = image.shape[:2]

    # Read label file
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # invalid line

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [(int(x * w), int(y * h)) for x, y in zip(coords[::2], coords[1::2])]

        # Convert to contour format
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw filled polygon with border
        cv2.polylines(image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(image, [contour], color=(0, 255, 0, 50))  # Optional filled area

    # Show the result
    cv2.imshow("Polygon Overlay", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
draw_yolo_polygons("/home/skoumal/dev/ObjectDetection/data/train_tuutuuut/images/000004_000009.jpg", "/home/skoumal/dev/ObjectDetection/data/train_tuutuuut/labels/000004_000009.txt")
