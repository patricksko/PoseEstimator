import numpy as np
import pyrender
import trimesh
import cv2
import os

def look_at(eye, target, up=np.array([0, 1, 0])):
    """
    Try different camera conventions to match BlenderProc
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)

    # Right vector
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # True up vector  
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    # Try BlenderProc convention - Option 1: Standard right-handed with -Z forward
    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward  # Z points away from target (towards camera)
    mat[:3, 3] = eye
    return mat


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
def convert_to_opencv_convention(opengl_pose):
    """
    Convert OpenGL camera pose to Blender camera convention
    Blender cameras: X=right, Y=down, Z=forward (towards target)
    """
    # Transformation matrix to convert from OpenGL to Blender camera convention
    # OpenGL: X=right, Y=up, Z=back
    # Blender: X=right, Y=down, Z=forward
    conversion_matrix = np.array([
        [1,  0,  0,  0],  # X stays the same
        [0, -1,  0,  0],  # Y flips (up -> down)
        [0,  0, -1,  0],  # Z flips (back -> forward)
        [0,  0,  0,  1]
    ])
    
    return conversion_matrix @ opengl_pose

def create_camera_frame(length=0.05):
    """
    Creates a small camera coordinate frame (X-red, Y-green, Z-blue)
    """
    # Vertices of lines
    vertices = np.array([
        [0, 0, 0], [length, 0, 0],  # X
        [0, 0, 0], [0, length, 0],  # Y
        [0, 0, 0], [0, 0, length],  # Z
    ])

    # Each pair of vertices is a line (indices into vertices)
    edges = np.array([
        [0, 1],  # X
        [2, 3],  # Y
        [4, 5],  # Z
    ])

    # Colors per entity (one per line)
    colors = np.array([
        [255, 0, 0, 255],   # X - red
        [0, 255, 0, 255],   # Y - green
        [0, 0, 255, 255],   # Z - blue
    ], dtype=np.uint8)

    # Create lines using indices
    lines = [trimesh.path.entities.Line(points=edge) for edge in edges]

    # Create Path3D
    path = trimesh.path.Path3D(entities=lines, vertices=vertices, colors=colors)
    return path


# Load and prepare the 3D model
mesh = trimesh.load("../../data/LegoBlock_variant.ply")

# Center the mesh at origin
mesh.vertices -= mesh.centroid

# DON'T apply coordinate transformation yet - let's see the original orientation first
print(f"Mesh bounds: {mesh.bounds}")
print(f"Mesh centroid: {mesh.centroid}")

# Set up mesh appearance BEFORE creating pyrender mesh
mesh.visual.vertex_colors = np.tile([0, 0, 255, 255], (len(mesh.vertices), 1))

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.0, 0.0, 1.0, 1.0],
    metallicFactor=0.1,
    roughnessFactor=0.8,
    alphaMode='OPAQUE',
    doubleSided=True
)

# Visualize the setup first (comment out this section once it works)
print("Visualizing camera setup...")
scene_vis = trimesh.Scene()
scene_vis.add_geometry(mesh)

# Add coordinate frame at origin
origin_frame = create_camera_frame(length=0.05)
scene_vis.add_geometry(origin_frame)

# Show a few camera positions for debugging
camera_positions = get_axis_aligned_camera_positions(distance=0.3)
for i, cam in enumerate(camera_positions[:]):  # Only show first 6 cameras
    cam_frame = create_camera_frame(length=0.02)
    cam_transform = look_at(cam['eye'], cam['target'], cam['up'])
    cam_frame.apply_transform(cam_transform)
    scene_vis.add_geometry(cam_frame)


scene_vis.show()

pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
scene = pyrender.Scene()
scene.add(pyrender_mesh)

# Set up the renderer
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# Define camera intrinsics
focal_length = 500.0
camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=320.0, cy=240.0)
cam_node = scene.add(camera)

# Add lighting
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
light_node = scene.add(light)

# Output folder
output_dir = "../../data/lego_views"
os.makedirs(output_dir, exist_ok=True)

# Get axis-aligned camera positions
camera_positions = get_axis_aligned_camera_positions(distance=0.3)

print(f"Generating {len(camera_positions)} views:")
print(f"- {sum(1 for p in camera_positions if p['type'] == 'face')} face views")
print(f"- {sum(1 for p in camera_positions if p['type'] == 'edge')} edge views") 
print(f"- {sum(1 for p in camera_positions if p['type'] == 'corner')} corner views")

for i, cam_pos in enumerate(camera_positions):
    # Create camera pose using look_at
    cam_pose = look_at(cam_pos['eye'], cam_pos['target'], cam_pos['up'])
    
    # Set camera pose
    scene.set_pose(cam_node, pose=cam_pose)
    
    # Position light to follow camera for better illumination
    light_pose = cam_pose.copy()
    scene.set_pose(light_node, pose=light_pose)
    
    # Render
    color, depth = renderer.render(scene)
    
    # Save RGB image
    rgb_filename = f"rgb_{i:02d}_{cam_pos['type']}.png"
    cv2.imwrite(os.path.join(output_dir, rgb_filename), color)
    
    # Save depth
    depth_filename = f"depth_{i:02d}_{cam_pos['type']}.npy"
    np.save(os.path.join(output_dir, depth_filename), depth)
    
    # Save pose and camera parameters for point cloud reconstruction
    cam_pose_opengl = cam_pose.copy()  # This is what pyrender uses
    cam_pose_blender = convert_to_opencv_convention(cam_pose_opengl)  # Convert to Blender
    cam_pose_mm = cam_pose_blender.copy()
    cam_pose_mm[:3, 3] *= 1000.0  # scale translation from m to mm

    pose_filename = f"pose_{i:02d}_{cam_pos['type']}.npy"
    camera_data = {
        'pose': cam_pose_mm,
        'intrinsics': {
            'fx': focal_length,
            'fy': focal_length,
            'cx': 320.0,
            'cy': 240.0
        },
        'eye': cam_pos['eye'] * 1000.0,  # also in mm if you want consistency
        'target': cam_pos['target'] * 1000.0,
        'up': cam_pos['up'],  # unit vector, no scaling
        'type': cam_pos['type']
    }
    np.save(os.path.join(output_dir, pose_filename), camera_data)
    
   



