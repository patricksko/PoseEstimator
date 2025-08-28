import cv2
import numpy as np
import pyrealsense2 as rs

class TheRetector:
    def __init__(self, model, profile, class_id=0, conf=0.8, imgsz=512, device="0"):
        self.model = model
        self.class_id = class_id
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.profile = profile

    def detect_masks(self, img_bgr):
        """Run YOLO and return a list of masks (one per detected object)."""
        h, w = img_bgr.shape[:2]
        masks = []

        results = self.model(
            source=img_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            save=False,
            verbose=False
        )

        for r in results:
            if not hasattr(r, "masks") or r.masks is None:
                continue
            for seg, cls in zip(r.masks.xy, r.boxes.cls):
                if int(cls) != self.class_id:
                    continue
                poly = np.array(seg, dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)
                masks.append(mask)

        return masks

    def overlay_masks(self, img_bgr, masks, alpha=0.5):
        """Overlay each mask in a different random color."""
        overlay = img_bgr.copy()

        # Pre-generate distinct colors for each mask
        rng = np.random.default_rng(42)  # reproducible random colors
        colors = rng.integers(0, 255, size=(len(masks), 3))

        for mask, color in zip(masks, colors):
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[mask > 0] = color.tolist()
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        return overlay
    
    def rs_get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array([[intr.fx, 0,         intr.ppx],
                    [0,        intr.fy,   intr.ppy],
                    [0,        0,         1       ]], dtype=np.float32)
        return intr, K
