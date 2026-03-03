from ultralytics import YOLO# --- GPU SUPPORT ---
from ultralytics import SAM
import numpy as np
import cv2

class Detector:
    def __init__(self, yolo_weights: str, sam_weights: str):
        self.yolo = YOLO(yolo_weights)
        self.sam = SAM(sam_weights)

    def detect_mask(self ,img_bgr, conf=0.7):
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
        results = results[0]
        detections = []
        
        if results.masks is None:
            return None
        for poly, cls, data, bbox in zip(
                results.masks.xy,
                results.boxes.cls,
                results.masks.data,
                results.boxes.xyxy
        ):
            result_sam = self.sam(img_bgr, bboxes=bbox.cpu().numpy().tolist(), device=0, save=False, verbose=False)
            if result_sam[0].masks is None:
                continue
            

            # Create binary mask
            mask = result_sam[0].masks.data.cpu()#np.zeros((h, w), dtype=np.uint8)
            mask = mask.squeeze(0)
            # cv2.fillPoly(mask, [poly_np], 255)

            detections.append({
                "mask": mask,
                "class_id": int(cls),
                # "bbox": bbox.cpu().numpy().tolist()
            })

        return detections
    
