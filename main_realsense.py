import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
from EstimHelpers.HelpersRealtime import *
from ultralytics import YOLO
import time
from colorama import Fore, Style

WEIGHTS_PATH="./data/best.pt"
PCD_PATH = "./data/lego_views/"
CAD_PATH = "./data/obj_000001.ply"
TARGET_PTS    = 100
TRACK_EVERY = 1

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
    
    # Configure Realsense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # enable depth stream
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) # (optional) enable color stream
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale() # get depth scale of realsense
    
    # Filters of Realsense
    spatial    = rs.spatial_filter()
    temporal   = rs.temporal_filter()
    hole_fill  = rs.hole_filling_filter()

    intr, K = rs_get_intrinsics(profile=profile) # get intrinsic info and intrinsic camera matrix

    # Read CAD Model for comparision
    cad_model = o3d.io.read_point_cloud(CAD_PATH)
    cad_model.scale(0.001, center=cad_model.get_center())
    cad_model.translate(-cad_model.get_center())
    cad_points = np.asarray(cad_model.points)

    # Read CAD Model for later Rendering 
    mesh = o3d.io.read_triangle_mesh(CAD_PATH)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    mesh.scale(0.001, center=mesh.get_center())

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

    # Offscreen renderer
    renderer = None
    scene = None
    render_w = render_h = None
    poses = []
    xyz_list = []
    rpy_list = []
    stats_plotted = False
    
    try:
        while True:
            # Read Camera frames (color and depth) and align them
            frameset = align.process(pipe.wait_for_frames())
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            #preprocess the images
            frame = depth_frame
            frame = spatial.process(frame)
            frame = temporal.process(frame)
            frame = hole_fill.process(frame)

            # Convert to numpy
            depth = np.asanyarray(frame.get_data())  # uint16 depth in sensor units
            color = np.asanyarray(color_frame.get_data())  # BGR
            
            #First initialization of pose
            if not initialized:
                # Check if mask is available, and if its not a misdetection (should appear in at least 10 frames)
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
                    depth = np.asanyarray(frame.get_data())
                    color = np.asanyarray(color_frame.get_data())  
                    mask = detect_mask(yolo_model, color)
                    if mask is None or mask.sum() == 0:
                        frame_counter = 0
                    frame_counter+=1
                
                depth_m = depth.astype(np.float32) * depth_scale # mind depth scale of realsense
                depth_m[mask == 0] = 0.0 # create a mask of the depth image for pointcloud

                depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32)) # create open3d depth image for later processing
                color_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)) # create open3d color image for later processing
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
                    continue
                else:
                    print("[INFO] Initial pose rejected. Retrying...")
                    frame_counter = 0
                    continue
            all = time.time()
            # Projection (intrinsics) for this frame
            start = time.time()
            near, far = 0.01, 5.0
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
            if frame_id % TRACK_EVERY == 0:
                
                mask = detect_mask(yolo_model, color)
                if mask is None or mask.sum() == 0:   # no segmentation found
                    print("No mask detected")
                    continue
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
                # o3d.visualization.draw_geometries([
                #     src.paint_uniform_color([0, 1, 1]),
                #     dst_cloud.paint_uniform_color([1, 1, 0])
                    
                # ], window_name="Init registration (ENTER=accept, SPACE=reject)")
                # exit()
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
                T_m2c = delta @ T_m2c              # candidate new pose

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





