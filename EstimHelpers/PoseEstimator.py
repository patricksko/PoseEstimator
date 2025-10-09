import os
import glob
from ultralytics import YOLO
from EstimHelpers.template_creation import render_lego_views
from EstimHelpers.HelpersRealtime import preprocess_point_cloud_uniform, cloud_resolution, get_correspondences, run_teaser, chamfer_distance, camera_eye_lookat_up_from_H
import copy
import numpy as np
import cv2
import open3d as o3d

class PoseEstimator:
    def __init__(self, yolo_weights, cad_path, pcd_path, intr, K, target_points=100):
        self.yolo = YOLO(yolo_weights)

        self.mesh = o3d.io.read_triangle_mesh(cad_path)
        self.mesh.compute_vertex_normals()

        self.target_points = target_points
        self.templates = self.load_templates(pcd_path, cad_path)
        self.K = K
        self.intr = intr

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.intr.width, self.intr.height)
        self.scene = self.renderer.scene
        self.scene.set_background([1, 1, 1, 1])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.add_geometry("mesh", self.mesh, mat)

    def load_templates(self, pcd_path, cad_path):
        ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))
        if not ply_files:
            render_lego_views(mesh_path=cad_path, output_dir=pcd_path)
            ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))
        src_clouds = []
        for ply_file in ply_files:
            src = o3d.io.read_point_cloud(ply_file)
            src_clouds.append(src)
            print(f"Loaded: {ply_file} with {len(src.points)} points")
        return src_clouds
        
    def detect_mask(self, img_bgr, class_id=0, conf=0.8, imgsz=512):
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        results = self.yolo(source=img_bgr, imgsz=imgsz, conf=conf, device="0", save=False, verbose=False)
        # if results[0].boxes is None or len(results[0].boxes) == 0:
        #     print("No bounding boxes detected. Exiting...")
        #     exit()  # or return if inside a function
        for r in results:
            if not hasattr(r, "masks") or r.masks is None:
                continue
            for seg, cls in zip(r.masks.xy, r.boxes.cls):
                
                if int(cls) != class_id: 
                    continue
                poly = np.array(seg, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 255)
                return mask    # first match
        return mask
    
    def find_best_template_teaser(self, dst_cloud):
        dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, 400)
        if not dst_down.has_normals():
            dst_down.estimate_normals()
        res = cloud_resolution(dst_down)
        noise_bound = 1.5 * res           # adaptive, often far better than a fixed 1mm
        match_max_dist = 4.0 * res        # cap for geometric plausibility

        best = dict(idx=-1, T=np.eye(4), score=np.inf)
        
        for idx, src_cloud in enumerate(self.templates):
            src0 = copy.deepcopy(src_cloud)

            # 1) Downsample & FPFH *after* coarse alignment
            src_down, src_fpfh = preprocess_point_cloud_uniform(src0, 400)
            if src_down is None:
                continue
            if not src_down.has_normals():
                src_down.estimate_normals()
            correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=match_max_dist)
            if len(correspondences) < 5:
                print("Not enough correspondences")
                continue
            H = run_teaser(src_down, dst_down, correspondences, noise_bound_m=noise_bound)
            icp_result = o3d.pipelines.registration.registration_icp(
                src_down, dst_down, 0.01, H,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            refined_transform = icp_result.transformation  # refined

            T_full = refined_transform
            src_aligned = copy.deepcopy(src_cloud).transform(T_full)
            alpha = 1
        

            geom_err = chamfer_distance(src_aligned, dst_cloud)

            # Combined score
            score = alpha * geom_err

            if score < best["score"]:
                best.update(idx=idx, T=T_full, score=score)

        return best["T"]
    
    def create_template_from_H(self, T_m2c):
        near, far = 0.01, 5.0
        self.scene.camera.set_projection(self.K, near, far, self.intr.width, self.intr.height)

        # Extrinsics from current T_m2c
        eye, target, up = camera_eye_lookat_up_from_H(T_m2c)
        self.scene.camera.look_at(target, eye, up)

        # Render synthetic view of CAD at T_m2c
        color_img = self.renderer.render_to_image()
        depth_img = self.renderer.render_to_depth_image(z_in_view_space=True)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
        )
        src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(
        self.intr.width, self.intr.height, self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
    ))
        return src