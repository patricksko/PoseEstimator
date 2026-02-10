import os
import glob
import pyrealsense2 as rs
from pose_estimator.EstimHelpers.template_creation import render_templates
from pose_estimator.EstimHelpers.HelpersRealtime import run_teaser, camera_eye_lookat_up_from_H, alignment_score
import copy
import numpy as np
import open3d as o3d


class PoseEstimator():
    def __init__(self, cad_path: str, pcd_path: str, intr: rs.intrinsics, K: np.ndarray, target_points: int =200):
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
        
        if intr is None or K is None:
            return

        self.mesh = o3d.io.read_triangle_mesh(cad_path)
        self.mesh.compute_vertex_normals()

        self.target_points = target_points
        self.templates = self.load_templates(pcd_path, cad_path)
        self.K = K
        self.intr = intr
        self.voxel_size = 0.05

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
        corner=np.array([1,1,1])
        ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))
        # create templates if there are no ply files in folder
        if not ply_files:
            render_templates(mesh_path=cad_path, output_dir=pcd_path)
            ply_files = sorted(glob.glob(os.path.join(pcd_path, "*.ply")))

        src_clouds = []
        for ply_file in ply_files:
            src = o3d.io.read_point_cloud(ply_file)
            if len(src.points) == 0:
                print("Empty point cloud!")
                return None, None

            # src_down = uniform_downsample_farthest_point(src, self.target_points) # Downsample the pointcloud to target_poins 
            src_clouds.append(src)
    
        return src_clouds
    
    
    def find_best_template_teaser(self, dst_cloud):
        # dst_down, dst_fpfh = preprocess_point_cloud_uniform(dst_cloud, self.target_points)
        dst_down = dst_cloud.voxel_down_sample(self.voxel_size)
        best = dict(T=np.eye(4), score=np.inf, src=None)
        
        for src in self.templates:
            if src is None:
                continue
            src_down = src.voxel_down_sample(self.voxel_size)
            # correspondences = get_correspondences(src_down, dst_down, src_fpfh, dst_fpfh, distance_threshold=match_max_dist)

            H = run_teaser(src_down, dst_down, voxel_size=self.voxel_size)
            icp_result = o3d.pipelines.registration.registration_icp(
                src_down, dst_down, 0.05, H,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
            )
            refined_transform = icp_result.transformation  # refined

            T_full = refined_transform
            src_aligned = copy.deepcopy(src_down).transform(T_full)
            score = alignment_score(
                src_aligned, src_down, dst_down, self.voxel_size
            )


            if score < best["score"]:
                best["score"] = score
                best["T"] = H
                best["src"] = src_down

        return best["T"], best["src"]
    
    def create_template_from_H(self, T_m2c, target_points):
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
        pts = np.asarray(src.points)
        if pts.shape[0] > target_points:
            idx = np.random.choice(pts.shape[0], target_points, replace=False)
            src = src.select_by_index(idx)

        return src