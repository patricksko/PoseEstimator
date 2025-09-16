import cv2
import numpy as np
import pyrealsense2 as rs

class TheRetector:
    def __init__(self, model, profile, class_id=0, conf=0.8, device="0"):
        self.model = model
        self.class_id = class_id
        self.conf = conf
        self.device = device
        self.profile = profile
        self.prev_centroids = []     # list of (u,v) from previous frame
        self.color_map = {}          # maps object ID -> color
        self.next_color_id = 0   

    def detect_masks(self, img_bgr):
        """Run YOLO and return a list of masks (one per detected object)."""
        h, w = img_bgr.shape[:2]
        masks = []
        centroids = []

        results = self.model(
            source=img_bgr,
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
                # compute centroid
                M = cv2.moments(mask)
                if M["m00"] > 0:
                    cX = int(M["m10"] / (M["m00"]+ 1e-5))
                    cY = int(M["m01"] / (M["m00"]+ 1e-5))
                else:
                    cX, cY = 0, 0
                centroids.append((cX, cY))

        return masks, centroids

    def overlay_masks_consistent(self, img_bgr, masks, centroids, alpha=0.5):
        """Overlay each mask with a consistent color across frames"""
        overlay = img_bgr.copy()
        new_centroids = []

        for mask, centroid in zip(masks, centroids):
            # find nearest previous centroid
            assigned_id = None
            if self.prev_centroids:
                dists = [np.linalg.norm(np.array(centroid)-np.array(pc)) for pc in self.prev_centroids]
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < 50:  # threshold in pixels
                    assigned_id = min_idx

            # if new object, assign new ID
            if assigned_id is None:
                assigned_id = self.next_color_id
                self.next_color_id += 1

            # store centroid for next frame
            new_centroids.append(centroid)

            # assign color
            if assigned_id not in self.color_map:
                rng = np.random.default_rng(assigned_id)
                self.color_map[assigned_id] = rng.integers(0, 255, size=(3,))
            color = self.color_map[assigned_id]

            # apply mask overlay
            colored_mask = np.zeros_like(img_bgr)
            colored_mask[mask > 0] = color.tolist()
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)

        self.prev_centroids = new_centroids
        return overlay
    
    def rs_get_intrinsics(self):
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        K = np.array([[intr.fx, 0,         intr.ppx],
                    [0,        intr.fy,   intr.ppy],
                    [0,        0,         1       ]], dtype=np.float32)
        return intr, K
