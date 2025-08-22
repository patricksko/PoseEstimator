import blenderproc as bproc
import argparse
import os
import numpy as np
from pathlib import Path
import json

# -------------------- args --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str,
                    help='Path to JSON config')
args = parser.parse_args()

config_path = Path(args.config)
assert config_path.exists(), f'Config file {config_path} does not exist'

with open(config_path, 'r') as f:
    config = json.load(f)

# paths
bop_dataset_path = (config_path.parent / config["bop_path"]).resolve()
output_dir = (config_path.parent / config["save_path"]).resolve()
dataset_name = "Legoblock_full"

# parameters
num_frames = int(config['dataset'].get('num_frames', 1000))
radius = float(config['dataset'].get('camera_radius', 0.8))

# -------------------- init --------------------
bproc.init()

# load lego block
lego_objs = bproc.loader.load_bop_objs(
    bop_dataset_path=os.path.join(str(bop_dataset_path), dataset_name),
    model_type="cad",
    object_model_unit="mm"
)
lego = lego_objs[0]
lego.set_name("lego_block")
lego.set_shading_mode("auto")
lego.hide(False)
lego.set_cp("category_id", 1)

# place it at origin
lego.set_location([0., 0.0, 0.0])
lego.set_rotation_euler([np.pi/2, 0.0, 0.0])

# simple ground
plane = bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 0, -0.01])

# lighting
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0.5, 0.5, 1.5])
light.set_energy(150)

# load intrinsics (BOP style)
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(str(bop_dataset_path), dataset_name))

# renderer setup
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# -------------------- camera motion --------------------
poi = np.array([0.0, 0.0, 0.0])  # point of interest (lego center)
n1 = num_frames // 2
n2 = num_frames - n1

theta1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
z1 = np.full_like(theta1, 0.25)  # constant height

for i, th in enumerate(theta1):
    cam_loc = np.array([radius*np.cos(th), radius*np.sin(th), z1[i]], dtype=np.float32)
    R = bproc.camera.rotation_from_forward_vec(poi - cam_loc, inplane_rot=0.0)
    cam2world = bproc.math.build_transformation_mat(cam_loc, R)
    bproc.camera.add_camera_pose(cam2world, frame=i)

u = np.linspace(0, 1, n2, endpoint=True)
ease = 0.5 - 0.5*np.cos(np.pi*u)  # cosine ease

radius2 = radius * (1.0 - 0.8*ease)     # shrink to 20% of original
height2 = 0.25 + 0.55*ease              # go up ~0.8 m total height (tune)
theta2 = 2*np.pi + 1.0*np.pi*ease       # continue turning half-circle while rising

for k in range(n2):
    th = theta2[k]
    r  = radius2[k]
    z  = height2[k]
    cam_loc = np.array([r*np.cos(th), r*np.sin(th), z], dtype=np.float32)
    R = bproc.camera.rotation_from_forward_vec(poi - cam_loc, inplane_rot=0.0)
    cam2world = bproc.math.build_transformation_mat(cam_loc, R)
    bproc.camera.add_camera_pose(cam2world, frame=n1 + k)
    
data = bproc.renderer.render()

# -------------------- write BOP --------------------
bproc.writer.write_bop(
    os.path.join(str(output_dir), "bop_data"),
    target_objects=[lego],
    dataset=dataset_name,
    depths=data["depth"],
    colors=data["colors"],
    color_file_format="JPEG",
    ignore_dist_thres=10,
    frames_per_chunk=num_frames,
    depth_scale=0.1
)

print(f"[OK] Rendered {num_frames} frames into {output_dir}/bop_data")
