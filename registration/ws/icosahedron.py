import numpy as np
import open3d as o3d
import os
from pathlib import Path

def fx_from_fov(fov_deg, width):
    f = 0.5 * width / np.tan(np.deg2rad(fov_deg) / 2.0)
    return f

def o3d_lookat(eye, target, up):
    eye    = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up     = np.asarray(up, dtype=float)
    z = (eye - target); z /= (np.linalg.norm(z) + 1e-12)         # camera forward (+Z points out of camera)
    x = np.cross(up, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.vstack([x, y, z])        # world->cam rotation
    T[:3, 3]  = -T[:3, :3] @ eye            # world->cam translation
    return T                   

def crop_pointcloud_fov(pcd, fov_deg=60.0, look_dir=np.array([0, 0, 1])):
    """
    Crops a point cloud to simulate a camera FoV.
    - pcd: Open3D PointCloud
    - fov_deg: field of view in degrees
    - look_dir: viewing direction vector (camera looks along this)
    """
    points = np.asarray(pcd.points)
    look_dir = look_dir / np.linalg.norm(look_dir)
    fov_rad = np.deg2rad(fov_deg)
    # Compute angles between look_dir and each point vector
    norms = np.linalg.norm(points, axis=1)
    norms[norms == 0] = 1e-8
    dirs = points / norms[:, None]
    cos_angles = np.dot(dirs, look_dir)
    mask = cos_angles > np.cos(fov_rad / 2)
    return pcd.select_by_index(np.where(mask)[0])

def crop_pointcloud_fov_hpr(pcd, camera_position=np.array([0, 0, -1]),
                           camera_lookat=np.array([0, 0, 0]),
                           fov_deg=60.0, radius_scale=100.0):
    """
    FOV + Hidden Point Removal (robust radius).
    radius_scale: increase if you still see over-pruning (typical 20–200).
    """
    # View direction
    cam_dir = camera_lookat - camera_position
    cam_dir /= (np.linalg.norm(cam_dir) + 1e-12)

    # Angular FoV prefilter
    pts = np.asarray(pcd.points)
    vecs = pts - camera_position
    dist = np.linalg.norm(vecs, axis=1)
    safe_dist = np.maximum(dist, 1e-8)
    dirs = vecs / safe_dist[:, None]
    cos_angles = np.dot(dirs, cam_dir)
    mask_fov = cos_angles > np.cos(np.deg2rad(fov_deg) / 2.0)
    idx_fov = np.where(mask_fov)[0]
    if idx_fov.size == 0:
        return o3d.geometry.PointCloud()  # nothing in FoV

    pcd_fov = pcd.select_by_index(idx_fov)

    # --- Robust radius choice ---
    # Use the maximum distance in the *full* cloud as a baseline to avoid local underestimation
    max_dist_full = (np.linalg.norm(pts - camera_position, axis=1).max()
                     if len(pts) else 1.0)
    # Make radius big (HPR becomes conservative as radius grows)
    radius = max(1e-3, radius_scale * max_dist_full)

    # HPR can still be sensitive if the camera is inside the convex hull.
    # Keep the camera comfortably away from the object relative to its size.
    _, visible_idx = pcd_fov.hidden_point_removal(camera_position, radius)
    return pcd_fov.select_by_index(visible_idx)

def get_axis_aligned_camera_positions(distance=0.3):
    """
    Generate camera positions that are aligned with the block's principal axes
    to ensure 90-degree views of faces, edges, and corners.
    """
    positions = []
    
    # 1. Face-aligned views (6 positions) - looking directly at each face
    face_directions = [
        np.array([1, 0, 0]),   # +X face (right)
        np.array([-1, 0, 0]),  # -X face (left)
        np.array([0, 1, 0]),   # +Y face (top)
        np.array([0, -1, 0]),  # -Y face (bottom)
        np.array([0, 0, 1]),   # +Z face (front)
        np.array([0, 0, -1])   # -Z face (back)
    ]
    
    for direction in face_directions:
        positions.append({
            'eye': direction * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 1, 0]) if abs(direction[1]) < 0.9 else np.array([0, 0, 1]),
            'type': 'face'
        })
    
    # 2. Edge-aligned views (12 positions) - looking along edges
    edge_directions = [
        # XY plane edges (4)
        np.array([1, 1, 0]),    # +X+Y edge
        np.array([1, -1, 0]),   # +X-Y edge
        np.array([-1, 1, 0]),   # -X+Y edge
        np.array([-1, -1, 0]),  # -X-Y edge
        # XZ plane edges (4)
        np.array([1, 0, 1]),    # +X+Z edge
        np.array([1, 0, -1]),   # +X-Z edge
        np.array([-1, 0, 1]),   # -X+Z edge
        np.array([-1, 0, -1]),  # -X-Z edge
        # YZ plane edges (4)
        np.array([0, 1, 1]),    # +Y+Z edge
        np.array([0, 1, -1]),   # +Y-Z edge
        np.array([0, -1, 1]),   # -Y+Z edge
        np.array([0, -1, -1])   # -Y-Z edge
    ]
    
    for direction in edge_directions:
        direction_norm = direction / np.linalg.norm(direction)
        # Choose appropriate up vector
        if abs(direction_norm[1]) < 0.9:
            up_vec = np.array([0, 1, 0])
        else:
            up_vec = np.array([0, 0, 1])
            
        positions.append({
            'eye': direction_norm * distance,
            'target': np.array([0, 0, 0]),
            'up': up_vec,
            'type': 'edge'
        })
    
    # 3. Corner-aligned views (8 positions) - looking at corners
    corner_directions = [
        np.array([1, 1, 1]),     # +++
        np.array([1, 1, -1]),    # ++-
        np.array([1, -1, 1]),    # +-+
        np.array([1, -1, -1]),   # +--
        np.array([-1, 1, 1]),    # -++
        np.array([-1, 1, -1]),   # -+-
        np.array([-1, -1, 1]),   # --+
        np.array([-1, -1, -1])   # ---
    ]
    
    for direction in corner_directions:
        direction_norm = direction / np.linalg.norm(direction)
        positions.append({
            'eye': direction_norm * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 1, 0]),
            'type': 'corner'
        })
    
    return positions  # Total: 26 positions
def get_reduced_camera_positions(distance=0.3):
    """
    Generate 11 unique camera positions for a Lego block with axes:
    - Y: up (top/bottom)
    - X: long side
    - Z: short side
    """
    positions = []

    # --- Faces (4) ---
    face_dirs = [
        (np.array([0,  1,  0]), '0'),
        (np.array([0, -1,  0]), '1'),
    ]
    for d, name in face_dirs:
        positions.append({
            'eye': d * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 0, 1]),  # Y is up
            'type': name
        })

    face_dirs = [
        (np.array([1,  0,  0]), '2'),
        (np.array([0,  0,  1]), '3')
    ]
    for d, name in face_dirs:
        positions.append({
            'eye': d * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 1, 0]),  # Y is up
            'type': name
        })
    # --- Edges (4) ---
    edge_dirs = [
        (np.array([ 1,  1, 0]), '4'),
        (np.array([ 1, -1, 0]), '5'),
        (np.array([ 0,  1, 1]), '6'),
        (np.array([ 0, -1, 1]), '7'),
        (np.array([ 1, 0, 1]), '8'),
        (np.array([ 1, 0, -1]), '9')
    ]
    for d, name in edge_dirs:
        d = d / np.linalg.norm(d)
        positions.append({
            'eye': d * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 1, 0]),
            'type': name
        })

    # --- Corners (3) ---
    corner_dirs = [
        (np.array([ 1,  1, -1]), '10'),
        (np.array([ 1,  1,  1]), '11'),
        (np.array([ 1, -1,  1]), '12'),
        (np.array([ 1, -1, -1]), '13')
    ]
    for d, name in corner_dirs:
        d = d / np.linalg.norm(d)
        positions.append({
            'eye': d * distance,
            'target': np.array([0, 0, 0]),
            'up': np.array([0, 1, 0]),
            'type': name
        })

    return positions  # total: 14


def sample_mesh_points(mesh, num_points=50000):
    """Sample points from mesh surface"""
    return mesh.sample_points_uniformly(number_of_points=num_points)

def render_lego_views():
    """Main function to render 26 views of the Lego block"""
    
    # Load and prepare the mesh
    print("Loading Lego block mesh...")
    mesh_path = "./../../data/obj_000001.ply"  # Update path as needed
    
    if not os.path.exists(mesh_path):
        print(f"Error: Could not find {mesh_path}")
        print("Please update the mesh_path variable with the correct path to your .ply file")
        return
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Center the mesh at origin
    #mesh.translate(-mesh.get_center())
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Set blue color for the mesh
    mesh.paint_uniform_color([0.0, 0.0, 1.0])
    
    print(f"Mesh bounds: {mesh.get_axis_aligned_bounding_box()}")
    print(f"Mesh center: {mesh.get_center()}")
    
    # Sample points from the mesh for point cloud operations
    print("Sampling points from mesh...")
    point_cloud = sample_mesh_points(mesh, num_points=50000)
    point_cloud.paint_uniform_color([0.0, 0.0, 1.0])
    
    # Output directory
    output_dir = "./../../data/lego_views"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get camera positions
    camera_positions = get_reduced_camera_positions(distance=0.3)
    
    print(f"Generating {len(camera_positions)} views:")
    print(f"- {sum(1 for p in camera_positions if p['type'] == 'face')} face views")
    print(f"- {sum(1 for p in camera_positions if p['type'] == 'edge')} edge views") 
    print(f"- {sum(1 for p in camera_positions if p['type'] == 'corner')} corner views")
    
    w, h = 640, 480
    fov_deg = 60.0
    fx = fy = fx_from_fov(fov_deg, w)
    cx, cy = w / 2.0, h / 2.0

    renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
    scene = renderer.scene
    scene.set_background([1, 1, 1, 1])  # white

    # add mesh with a simple material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    scene.add_geometry("mesh", mesh, mat)

    # camera intrinsics
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    near, far = 0.01, 5.0
    K = intr.intrinsic_matrix  # 3x3 numpy array
    scene.camera.set_projection(K, near, far, w, h)

    for i, cam_pos in enumerate(camera_positions):
        eye, target, up = cam_pos["eye"], cam_pos["target"], cam_pos["up"]

        # set extrinsics
        T_world_cam = o3d_lookat(eye, target, up)         # world -> cam
        # OffscreenRenderer uses look_at (eye, center, up) directly for the view:
        scene.camera.look_at(target, eye, up)

        # render color & depth
        color_img = renderer.render_to_image()
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)  # linear depth in meters

        # save rgb/depth
        rgb_path   = os.path.join(output_dir, f"rgb_{i:02d}_{cam_pos['type']}.png")
        depth_path = os.path.join(output_dir, f"depth_{i:02d}_{cam_pos['type']}.npy")
        o3d.io.write_image(rgb_path, color_img)
        np.save(depth_path, np.asarray(depth_img, dtype=np.float32))  # meters

        # back-project to point cloud (camera frame)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
        )
        pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)  # points in camera coords
        pcd_cam.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
        # OPTIONAL: move cloud into world coords if you want it there
        T_cam_world = np.linalg.inv(T_world_cam)
        pcd_world = pcd_cam.transform(T_cam_world)

        # save PCD (choose which frame you want)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"pcd_cam_{i:02d}_{cam_pos['type']}.ply"), pcd_cam)
        # o3d.io.write_point_cloud(os.path.join(output_dir, f"pcd_world_{i:02d}_{cam_pos['type']}.ply"), pcd_world)
if __name__ == "__main__":
    render_lego_views()