import numpy as np
import open3d as o3d
import os
import pyrealsense2 as rs



def camera_eye_lookat_up_from_H(H):
    """
    Convert model→camera H to eye/target/up in model/world coords.
    - eye: camera center
    - target: eye + forward vector
    - up: image up
    """
    R = H[:3, :3]
    t = H[:3, 3]

    # camera center in world
    eye = -R.T @ t

    # camera axes in world
    forward = R.T @ np.array([0, 0, 1.0])   # camera +Z
    up      = R.T @ np.array([0, -1, 0])  # screen up

    target = eye + forward
    up = up / (np.linalg.norm(up) + 1e-12)
    return eye, target, up


def rs_get_intrinsics():
    """Get intrinsics from a running RealSense pipeline profile."""
    # color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    # intr = color_stream.get_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        640,
        480,
        1077.8, 1078.2,
        323.8, 279.7
    )


def render_lego_views():
    # ------------------------
    # Load RealSense intrinsics (optional)
    # ------------------------
    # pipeline = rs.pipeline()
    # cfg = rs.config()
    # cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # profile = pipeline.start(cfg)

    intr_o3d = rs_get_intrinsics()

    # ------------------------
    # Load mesh
    # ------------------------
    mesh_path = "./../../data/obj_000001.ply"
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.scale(0.001, center=mesh.get_center())
    mesh.translate(-mesh.get_center())

    # ------------------------
    # Example H (from dataset)
    # ------------------------
    H_mm = np.array([
        [ 0.8064637780189514, 0.46442028880119324, -0.36596444249153137, -10.217182159423828],
        [ 0.29991739988327026, -0.854698657989502, -0.42372122406959534, 18.80324935913086],
        [-0.5095740556716919, 0.23195669054985046, -0.8285712003707886, 961.5562744140625],
        [0, 0, 0, 1]
    ])

    # Convert translation mm→m
    H = H_mm.copy()
    H[:3, 3] *= 0.001

    # ------------------------
    # Set up renderer
    # ------------------------
    w, h = intr_o3d.width, intr_o3d.height
    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # white

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    # camera intrinsics (from RealSense, or use synthetic FOV)
    #fx, fy, cx, cy = intr_o3d.get_focal_length()[0], intr_o3d.get_focal_length()[1], intr_o3d.get_principal_point()[0], intr_o3d.get_principal_point()[1]
    near, far = 0.01, 2.0
    K = intr_o3d.intrinsic_matrix
    scene.camera.set_projection(K, near, far, w, h)

    # ------------------------
    # Extrinsics
    # ------------------------
    eye, target, up = camera_eye_lookat_up_from_H(H)
    scene.camera.look_at(target, eye, up)

    # ------------------------
    # Render
    # ------------------------
    output_dir = "./../../data/lego_homogenous"
    os.makedirs(output_dir, exist_ok=True)

    color_img = renderer.render_to_image()
    
    depth_img = renderer.render_to_depth_image(z_in_view_space=True)

    rgb_path   = os.path.join(output_dir, f"rgb.png")
    o3d.io.write_image(rgb_path, color_img)

    # Backproject depth to point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
    )
    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)

    # Flip to conventional Open3D coords (if needed)
    # pcd_cam.transform([[1, 0, 0, 0],
    #                    [0, -1, 0, 0],
    #                    [0, 0, -1, 0],
    #                    [0, 0, 0, 1]])

    o3d.visualization.draw_geometries([pcd_cam.paint_uniform_color([0,1,1])])


if __name__ == "__main__":
    render_lego_views()
