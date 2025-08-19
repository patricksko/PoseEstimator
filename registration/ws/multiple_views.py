import numpy as np
import open3d as o3d
import os
from pathlib import Path

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
    
    # Create visualizer for off-screen rendering
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    
    # Add the mesh to visualizer
    vis.add_geometry(mesh)
    
    # Get render option and set background to white
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # White background
    render_option.mesh_show_back_face = True
    
    for i, cam_pos in enumerate(camera_positions):
        print(f"Rendering view {i+1}/{len(camera_positions)}: {cam_pos['type']}")
        
        # Set camera parameters
        ctr = vis.get_view_control()
        
        # Set camera position and orientation
        # Convert to Open3D camera parameters
        front = cam_pos['target'] - cam_pos['eye']  # Look direction
        front = front / np.linalg.norm(front)
        
        up = cam_pos['up']
        
        # Set lookat, eye, and up
        ctr.set_lookat(cam_pos['target'])
        ctr.set_front(-front)  # Open3D uses negative front direction
        ctr.set_up(up)
        ctr.set_zoom(0.8)  # Adjust zoom as needed
        
        # Update geometry and render
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = np.asarray(image)
        
        # Convert to uint8 and save
        image_uint8 = (image_np * 255).astype(np.uint8)
        image_o3d = o3d.geometry.Image(image_uint8)
        
        # Save RGB image
        rgb_filename = f"rgb_{i:02d}_{cam_pos['type']}.png"
        o3d.io.write_image(os.path.join(output_dir, rgb_filename), image_o3d)
        
        # Generate and save cropped point cloud for this view
        visible_points = crop_pointcloud_fov_hpr(
            point_cloud, 
            camera_position=cam_pos['eye'],
            camera_lookat=cam_pos['target'],
            fov_deg=60.0
        )
        
        # Save point cloud
        pcd_filename = f"pointcloud_{i:02d}_{cam_pos['type']}.ply"
        o3d.io.write_point_cloud(os.path.join(output_dir, pcd_filename), visible_points)
        
        # Save camera parameters
        camera_data = {
            'eye': cam_pos['eye'],
            'target': cam_pos['target'],
            'up': cam_pos['up'],
            'type': cam_pos['type'],
            'front_direction': front,
            'fov_deg': 60.0
        }
        
        param_filename = f"camera_{i:02d}_{cam_pos['type']}.npy"
        np.save(os.path.join(output_dir, param_filename), camera_data)
        
        print(f"  Saved: {rgb_filename}, {pcd_filename}, {param_filename}")
        print(f"  Visible points: {len(visible_points.points)}")
    
    # Clean up
    vis.destroy_window()
    
    print(f"\nCompleted! Generated {len(camera_positions)} views in '{output_dir}'")
    print("Files generated per view:")
    print("  - RGB image (.png)")
    print("  - Visible point cloud (.ply)")
    print("  - Camera parameters (.npy)")

if __name__ == "__main__":
    render_lego_views()