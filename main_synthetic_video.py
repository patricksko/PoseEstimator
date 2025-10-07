#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
from ultralytics import YOLO
import open3d as o3d
from teaserpp_python import _teaserpp as tpp
from EstimHelpers.HelpersRealtime import *#list_bop_frames, load_scene_camera, find_best_template_teaser, camera_eye_lookat_up_from_H, preprocess_point_cloud_uniform, get_correspondences, project_points
import glob
import copy


# -------------------- CONFIG: ----------------------------
BOP_SEQ_DIR   = "./datageneration/Blenderproc/output/output_blenderproc_video/bop_data/Legoblock_full/train_pbr/000000"  # where BlenderProc wrote the frames
TEMPLATES_DIR = "./data/lego_views"                          # your 11/26 template PLYs
CAD_MODEL_PLY = "./data/obj_000001.ply"                      # full CAD (for projection)
YOLO_WEIGHTS  = "./data/best.pt"
CLASS_ID      = 0
TRACK_EVERY   = 1   # run TEASER update every N frames
TARGET_PTS    = 100  # feature downsample size


def main():
    frames = list_bop_frames(BOP_SEQ_DIR)
    if not frames:
        raise RuntimeError(f"No frames found in {BOP_SEQ_DIR}")
    cam_json = os.path.join(BOP_SEQ_DIR, "scene_camera.json")
    _ = load_scene_camera(cam_json)  # optional sanity

    # Load templates + CAD
    template_paths = sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.ply")))
    if not template_paths:
        raise RuntimeError(f"No PLYs in {TEMPLATES_DIR}")
    templates = [o3d.io.read_point_cloud(p) for p in template_paths]
    cad_model = o3d.io.read_point_cloud(CAD_MODEL_PLY)
    cad_points = np.asarray(cad_model.points)
    

    mesh = o3d.io.read_triangle_mesh(CAD_MODEL_PLY)
    mesh.compute_vertex_normals()


    yolo = YOLO(YOLO_WEIGHTS)

    initialized = False
    T_m2c = np.eye(4)

    # Offscreen renderer (lazy-init once we know width/height)
    renderer = None
    scene = None
    render_w = render_h = None

    idx = 0
    while idx < len(frames):
        print("=" * 20, "loop", "=" * 20, "idx =", idx)
        fid, rgb_path, depth_path = frames[idx]
        print(f"reading frame {fid}")

        color = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # uint16

        # YOLO mask
        mask = detect_mask(yolo, color, class_id=CLASS_ID, conf=0.7, imgsz=640)

        # Scene point cloud (uses correct per-frame intrinsics)
        dst_cloud, K, intr_o3d = get_pointcloud(depth_raw, color, BOP_SEQ_DIR, mask)
        if len(dst_cloud.points) == 0:
            cv2.imshow("Synthetic Tracking", color)
            if cv2.waitKey(1) == 27:
                break
            continue

        if not initialized:
            print("[INFO] Initializing with templates…")
            best_idx, refined_transform = find_best_template_teaser(
                dst_cloud, templates, target_points=TARGET_PTS
            )
            cand_cad = copy.deepcopy(templates[best_idx])
            cand_cad.transform(refined_transform)
            o3d.visualization.draw_geometries([
                dst_cloud.paint_uniform_color([0, 1, 0]),
                cand_cad.paint_uniform_color([0, 1, 1])
            ], window_name="Init registration (ENTER=accept, anything else=retry)")
            nb = input('Accept? [y/N]: ')
            if nb.lower() == 'y':
                print("[INFO] Initial pose accepted.")
                initialized = True
                T_m2c = refined_transform.copy()
                # prepare renderer once we know dims
                render_w, render_h = intr_o3d.width, intr_o3d.height
                renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
                scene = renderer.scene
                scene.set_background([1, 1, 1, 1])
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultLit"
                scene.add_geometry("mesh", mesh, mat)
                idx = 1  # start tracking on next frame
                continue
            else:
                print("[INFO] Initial pose rejected. Retrying from start…")
                idx = 0
                continue

        # ensure renderer matches this frame's intr dims (rarely changes)
        if (render_w != intr_o3d.width) or (render_h != intr_o3d.height):
            render_w, render_h = intr_o3d.width, intr_o3d.height
            renderer = o3d.visualization.rendering.OffscreenRenderer(render_w, render_h)
            scene = renderer.scene
            scene.set_background([1, 1, 1, 1])
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            scene.add_geometry("mesh", mesh, mat)

        idx += 1
        start = time.time()

        # Projection (intrinsics) for this frame
        near, far = 0.01, 2.0
        scene.camera.set_projection(K, near, far, render_w, render_h)

        # Extrinsics from current T_m2c
        eye, target, up = camera_eye_lookat_up_from_H(T_m2c)
        scene.camera.look_at(target, eye, up)

        # Render synthetic view of CAD at T_m2c
        color_img = renderer.render_to_image()
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        # if idx % 5 == 0:
        #     o3d.visualization.draw_geometries([o3d.geometry.Image(color_img)])
        # Backproject to get "src" cloud in camera coords
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, depth_scale=1.0, depth_trunc=far, convert_rgb_to_intensity=False
        )
        src = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr_o3d)

        prev_down, _ = preprocess_point_cloud_uniform(src, TARGET_PTS, calc_fpfh=False)

        if idx % TRACK_EVERY == 0:
            dst_down, _ = preprocess_point_cloud_uniform(dst_cloud, TARGET_PTS, calc_fpfh=False)

            icp_result = o3d.pipelines.registration.registration_icp(
                prev_down, dst_down, 0.01, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            delta = icp_result.transformation  # refined
            T_m2c = delta@T_m2c
            
        end = time.time()
        print(end-start)
        uv = project_points(cad_points, K, T_m2c)
        for (u, v) in uv:
            if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                cv2.circle(color, (u, v), 1, (0, 0, 255), -1)

        cv2.imshow("Live Tracking", color)
        if cv2.waitKey(1) & 0xFF == 27:
            break
       

if __name__ == "__main__":
    main()
