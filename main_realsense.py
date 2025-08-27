import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from EstimHelpers.HelpersRealtime import *
from ultralytics import YOLO
import time
from colorama import Fore, Style
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

WEIGHTS_PATH="./data/best.pt"
PCD_PATH = "./data/lego_views/"
CAD_PATH = "./data/obj_000001.ply"
TARGET_PTS    = 100

def timer_print(start_time, label):
    """Clean timer output"""
    elapsed = time.time() - start_time
    print(Fore.GREEN + f"  {label}: {elapsed:.3f}s" + Style.RESET_ALL)
    return elapsed

def main():
    # Setup:
    # Check if realsense is connected
    ctx = rs.context()
    if len(ctx.devices) == 0:
        raise RuntimeError("❌ No Intel RealSense device connected.")
    
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # enable depth stream
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # (optional) enable color stream
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() # get depth scale of realsense
    # Filters:
    decimation = rs.decimation_filter()
    spatial    = rs.spatial_filter()
    temporal   = rs.temporal_filter()
    hole_fill  = rs.hole_filling_filter()
    colorizer  = rs.colorizer()

    intr, K = rs_get_intrinsics(profile=profile) # get intrinsic info and intrinsic camera matrix

    cad_model = o3d.io.read_point_cloud(CAD_PATH)
    cad_points = np.asarray(cad_model.points)

    mesh = o3d.io.read_triangle_mesh(CAD_PATH)
    mesh.compute_vertex_normals()

    #Preload Templates for Template matching
    ply_files = sorted(glob.glob(os.path.join(PCD_PATH, "*.ply")))
    if not ply_files:
            raise RuntimeError(f"No PLYs in {PCD_PATH}")
    src_clouds = []
    for ply_file in ply_files:
        src = o3d.io.read_point_cloud(ply_file)
        src_clouds.append(src)
        print(f"Loaded: {ply_file} with {len(src.points)} points")
    
    # Preload YOLO model once
    yolo_model = YOLO(WEIGHTS_PATH)
    
    frame_counter = 0
    frame_id = 0
    initialized = False # Variable for initial orientation

    # Offscreen renderer (lazy-init once we know width/height)
    renderer = None
    scene = None
    render_w = render_h = None
    poses = []
    xyz_list = []
    rpy_list = []
    stats_plotted = False
    
    try:
        while True:
            frameset = align.process(pipe.wait_for_frames())
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            frame = depth_frame
            frame = spatial.process(frame)
            frame = temporal.process(frame)
            frame = hole_fill.process(frame)
            # Convert to numpy
            depth = np.asanyarray(frame.get_data())  # uint16 depth in sensor units
            color = np.asanyarray(color_frame.get_data())  # BGR
            if not initialized:
                while frame_counter!=10:
                    frameset = align.process(pipe.wait_for_frames())
                    depth_frame = frameset.get_depth_frame()
                    color_frame = frameset.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    frame = depth_frame
                    frame = spatial.process(frame)
                    frame = temporal.process(frame)
                    frame = hole_fill.process(frame)
                    # Convert to numpy
                    depth = np.asanyarray(frame.get_data())  # uint16 depth in sensor units
                    color = np.asanyarray(color_frame.get_data())  # BGR
                    mask = detect_mask(yolo_model, color)
                    if mask is None:
                        frame_counter = 0
                    frame_counter+=1
                
                depth_m = depth.astype(np.float32) * depth_scale
                depth_m[mask == 0] = 0.0 

                depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
                )
                dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
                dst_cloud, _ = dst_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
                # First guess with templates
                best_idx, H_init = find_best_template_teaser(dst_cloud, src_clouds, target_points=TARGET_PTS)
            
                # Candidate CAD for tracking going forward
                cand_cad = copy.deepcopy(src_clouds[best_idx])
                cand_cad.transform(H_init)

                # Show candidate guess
                o3d.visualization.draw_geometries([
                    cand_cad.paint_uniform_color([0, 1, 1]),
                    dst_cloud.paint_uniform_color([0, 1, 0])
                ], window_name="Init registration (ENTER=accept, SPACE=reject)")

                print("[INFO] Press Y/y to accept this guess. Or something else to try again")

                nb = input('Enter your input:')
                if nb.lower() == 'y':  
                    print("[INFO] Initial pose accepted.")
                    initialized = True
                    T_m2c = H_init.copy()
                    # prepare renderer once we know dims
                    render_w, render_h = intr.width, intr.height
                    renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
                    scene = renderer.scene
                    scene.set_background([1, 1, 1, 1])
                    mat = o3d.visualization.rendering.MaterialRecord()
                    mat.shader = "defaultLit"
                    scene.add_geometry("mesh", mesh, mat)
                    idx = 1  # start tracking on next frame
                    continue
                else:
                    print("[INFO] Initial pose rejected. Retrying...")
                    frame_counter = 0
                    continue
            all = time.time()
            # Projection (intrinsics) for this frame
            start = time.time()
            near, far = 0.01, 2.0
            scene.camera.set_projection(K, near, far, render_w, render_h)

            # Extrinsics from current T_m2c
            eye, target, up = camera_eye_lookat_up_from_H(T_m2c)
            scene.camera.look_at(target, eye, up)

            # Render synthetic view of CAD at T_m2c
            color_img = renderer.render_to_image()
            depth_img = renderer.render_to_depth_image(z_in_view_space=True)
    
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
            )
            src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
            timer_print(start, "Rendering")

            start = time.time()
            prev_down, _ = preprocess_point_cloud_uniform(src, TARGET_PTS, False)
            timer_print(start, "Preprocessing Previous")
            frame_id += 1
            if frame_id % 1 == 0:
                
                mask = detect_mask(yolo_model, color)
                if mask is None or mask.sum() == 0:   # no segmentation found
                    print("No mask detected")
                    exit()
                depth_m = depth.astype(np.float32) * depth_scale
                depth_m[mask == 0] = 0.0  

                # Create RGBD (Open3D expects RGB, depth in meters here because depth_scale=1.0)
                start = time.time()
                depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False
                )
                dst_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics_from_rs(intr))
                timer_print(start, "RGB Kamera")
                start = time.time()
                dst_down, _ = preprocess_point_cloud_uniform(dst_cloud, TARGET_PTS, False)
                timer_print(start, "Preprocessing Destination")
                start = time.time()
                icp_result = o3d.pipelines.registration.registration_icp(
                    prev_down, dst_down, 0.01, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                timer_print(start, "ICP")
                delta = icp_result.transformation  # refined
                T_new = delta @ T_m2c              # candidate new pose

                # alpha = 0.5  # smoothing factor (0=freeze, 1=no filter)

                # # --- Smooth translation ---
                # T_m2c[:3, 3] = (1 - alpha) * T_m2c[:3, 3] + alpha * T_new[:3, 3]

                # # --- Smooth rotation with SLERP ---
                # R_old = R.from_matrix(T_m2c[:3, :3])
                # R_new = R.from_matrix(T_new[:3, :3])

                # # key times [0,1], and two rotations stacked together
                # slerp = Slerp([0, 1], R.from_matrix([R_old.as_matrix(), R_new.as_matrix()]))

                # # evaluate at alpha (0 = old, 1 = new)
                # R_smooth = slerp([alpha])[0]

                # T_m2c[:3, :3] = R_smooth.as_matrix()

                T_m2c = T_new
                if len(poses) < 200:
                    poses.append(T_m2c.copy())

                # --- When 50 collected, compute stats and plot once ---
                if len(poses) == 200 and not stats_plotted:
                    for T in poses:
                        xyz_list.append(T[:3, 3])
                        rpy_list.append(R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True))

                    xyz_arr = np.vstack(xyz_list)   # shape (N,3)
                    rpy_arr = np.vstack(rpy_list)   # shape (N,3)
                    frames = np.arange(len(poses))

                    # Plot 6 subplots
                    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

                    labels = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]', 'X [m]', 'Y [m]', 'Z [m]']
                    data = [rpy_arr[:,0], rpy_arr[:,1], rpy_arr[:,2],
                            xyz_arr[:,0], xyz_arr[:,1], xyz_arr[:,2]]

                    for i, ax in enumerate(axes):
                        ax.plot(frames, data[i], '-o', markersize=2)
                        ax.set_ylabel(labels[i])
                        ax.grid(True)

                    axes[-1].set_xlabel("Frame index")

                    plt.tight_layout()
                    plt.show()

                print("="*50)
                timer_print(all, "Full Time")

               
            uv = project_points(cad_points, K, T_m2c)
            for (u, v) in uv:
                if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                    cv2.circle(color, (u, v), 1, (0, 0, 255), -1)  # red dots

            cv2.imshow("Live Tracking", color)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
           

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        pipe.stop()
if __name__ == "__main__":
    main()





