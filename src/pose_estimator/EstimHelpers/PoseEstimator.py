import os
import glob
from ultralytics import YOLO
import pyrealsense2 as rs
from pose_estimator.EstimHelpers.template_creation import render_lego_views
from pose_estimator.EstimHelpers.HelpersRealtime import preprocess_point_cloud_uniform, cloud_resolution, get_correspondences, run_teaser, chamfer_distance, camera_eye_lookat_up_from_H, uniform_downsample_farthest_point
import copy
import numpy as np
import cv2
import open3d as o3d

class PoseEstimator:
    def __init__(self, yolo_weights: str, cad_path: str, pcd_path: str, intr: rs.intrinsics, K: np.ndarray, target_points: int =200):
        """
        Initialize the 3D detection and registration pipeline.

        Args:
            yolo_weights (str): Path to the YOLO model weights file (.pt).
            cad_path (str): Directory containing CAD Files.
            pcd_path (str): Directory containing the templates.
            intr (rs.intrinsics): RealSense intrinsics object containing fx, fy, ppx, ppy, width, and height.
            K (np.ndarray): intrinsic used for projection/deprojection.
            target_points (int, optional): Target number of downsampled points for feature computation.
                Defaults to 100.
        """
        self.yolo = YOLO(yolo_weights)

        self.mesh = o3d.io.read_triangle_mesh(cad_path)
        self.mesh.compute_vertex_normals()

        self.target_points = target_points
        self.templates, self.fpfh_feats = self.load_templates(pcd_path, cad_path)
        self.K = K
        self.intr = intr

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.intr.width, self.intr.height)
        self.scene = self.renderer.scene
        self.scene.set_background([1, 1, 1, 1])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.add_geometry("mesh", self.mesh, mat)

    def load_templates(self, pcd_path: str, cad_path: str):
        """
        Loads or generates template point clouds from CAD models and computes their FPFH descriptors.

        This function prepares the template database used for registration or recognition.
        It first checks whether `.ply` point clouds already exist in the given `pcd_path`.
        If not, it automatically renders the CAD models (from `cad_path`) into point clouds
        using `render_lego_views()`. For each resulting `.ply` file, it loads the point cloud,
        downsamples it to a fixed number of points, estimates surface normals, and computes
        FPFH (Fast Point Feature Histogram) features for local geometric matching.

        Args:
            pcd_path (str): Directory containing the pre-rendered or existing template point clouds (.ply files).
            cad_path (str): Directory containing CAD models used to render point clouds if templates are missing.

        Returns:
            tuple[list[o3d.geometry.PointCloud], list[o3d.pipelines.registration.Feature]]:
                - `src_clouds`: List of downsampled Open3D point clouds corresponding to each template.
                - `fpfh_fts`: List of FPFH feature objects corresponding to each template.
        """
        ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))
        # create templates if there are no ply files in folder
        if not ply_files:
            render_lego_views(mesh_path=cad_path, output_dir=pcd_path)
            ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))

        src_clouds = []
        fpfh_fts = []
        for ply_file in ply_files:
            src = o3d.io.read_point_cloud(ply_file)
            if len(src.points) == 0:
                print("Empty point cloud!")
                return None, None
            
            src_down = uniform_downsample_farthest_point(src, self.target_points) # Downsample the pointcloud to target_poins 

            # Estimate normals with adaptive parameters
            nn_param = min(30, len(src_down.points) // 2)  # Adaptive neighbor count
            radius = 0.05  # You might want to adjust this based on your data scale
            
            src_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn_param)
            )
            
            # Compute FPFH features
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2.5, max_nn=100)
            )
            src_clouds.append(src_down)
            fpfh_fts.append(fpfh)

            print(f"Loaded: {ply_file} with {len(src.points)} points")
        return src_clouds, fpfh_fts
        
    def detect_mask(self, img_bgr, class_id=0, conf=0.8):
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
        results = self.yolo(source=img_bgr, conf=conf, device="0", save=False, verbose=False)

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
        dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, self.target_points)
        if not dst_down.has_normals():
            dst_down.estimate_normals()
        res = cloud_resolution(dst_down)
        noise_bound = 1.5 * res           # adaptive, often far better than a fixed 1mm
        match_max_dist = 4.0 * res        # cap for geometric plausibility

        best = dict(T=np.eye(4), score=np.inf)
        
        for src_down, src_fpfh in zip(self.templates, self.fpfh_feats):
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
            src_aligned = copy.deepcopy(src_down).transform(T_full)
            alpha = 1
        

            geom_err = chamfer_distance(src_aligned, dst_cloud)

            # Combined score
            score = alpha * geom_err

            if score < best["score"]:
                best.update(T=T_full, score=score)

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