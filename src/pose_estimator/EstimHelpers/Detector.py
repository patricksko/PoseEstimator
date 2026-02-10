from ultralytics import YOLO# --- GPU SUPPORT ---
import numpy as np
import cv2

class Detector:
    def __init__(self, yolo_weights: str):
        self.yolo = YOLO(yolo_weights)

    def detect_mask(self ,img_bgr, class_id=0, conf=0.7):
        """
        Detects and extracts the segmentation mask of target objects in an RGB image
        using the YOLO segmentation model.

        The method runs the YOLO model on the input image, filters detections by the
        specified `class_id` and confidence threshold, and converts the corresponding
        segmentation polygon into a binary mask image. The mask highlights pixels
        belonging to the detected object class.

        Args:
            img_bgr (np.ndarray): Input RGB image in BGR format, shape (H, W, 3).
            class_id (int, optional): The target YOLO class ID to detect (e.g., 0 for 'object A'). 
                Defaults to 0.
            conf (float, optional): Minimum confidence threshold for YOLO detections to be accepted. 
                Defaults to 0.8.

        Returns:
            np.ndarray:
                A 2D binary mask (uint8) of shape (H, W) where:
                - Pixel value 255 indicates object presence.
                - Pixel value 0 indicates background.
                If no object of the given class is found, an all-zero mask is returned.
        """
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        results = self.yolo(source=img_bgr, conf=conf, device=0, save=False, show=False, verbose=False)

        detections = []

        for r in results:
            if not hasattr(r, "masks") or r.masks is None:
                continue
            for poly, cls, cf, bbox in zip(
                    r.masks.xy,
                    r.boxes.cls,
                    r.boxes.conf,
                    r.boxes.xyxy
            ):
                poly_np = np.array(poly, dtype=np.int32)

                # Create binary mask
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_np], 255)

                detections.append({
                    "mask": mask,
                    "class_id": int(cls),
                    # "bbox": bbox.cpu().numpy().tolist()
                })

        return detections
    
