import numpy as np   
import cv2                     # fundamental package for scientific computing
import open3d as o3d                # Intel RealSense cross-platform open-source API
from pose_estimator.EstimHelpers.HelpersRealtime import *
from pose_estimator.EstimHelpers.RealSenseClass import RealSenseCamera
from pose_estimator.EstimHelpers.PoseEstimator import PoseEstimator
from pose_estimator.EstimHelpers.template_creation import render_lego_views
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
    cam = RealSenseCamera() # call the constructor of RealSenseCamera
    intr, K = cam.rs_get_intrinsics()
    
    estimator = PoseEstimator(WEIGHTS_PATH, CAD_PATH, PCD_PATH, intr, K, TARGET_PTS) # call the constructor of PoseEstimator for 6d-Pose

    # Read CAD Model for comparision
    cad_model = o3d.io.read_point_cloud(CAD_PATH)
    cad_points = np.asarray(cad_model.points)
    
    frame_counter = 0
    frame_id = 0
    initialized = False # Variable for initial orientation
    errorcounter = 0
  
    try:
        while True:
            color = cam.get_rgbd() #get color frame

            #First initialization of pose
            if not initialized:
                # Check if mask is available, and if its not a misdetection
                while frame_counter!=10:
                    color = cam.get_rgbd()
                    mask = estimator.detect_mask(color)
                    if mask is None or mask.sum() == 0:
                        frame_counter = 0
                    frame_counter+=1
                
                dst_cloud = cam.get_pcd_from_rgbd(mask)
               
                # First guess with templates
                H_init = estimator.find_best_template_teaser(dst_cloud)
                # H_init = enforce_upright_pose_y_up(H_init)

                initialized = True
                T_m2c = H_init.copy()
                continue

            all = time.time()
            # Projection (intrinsics) for this frame
            start = time.time()
            src = estimator.create_template_from_H(T_m2c=T_m2c)
            timer_print(start, "Rendering")

            start = time.time()
            prev_down, _ = preprocess_point_cloud_uniform(src, TARGET_PTS, False)
            timer_print(start, "Preprocessing Previous")
            frame_id += 1
            if frame_id % TRACK_EVERY == 0:
                mask = estimator.detect_mask(color)
                if mask is None or mask.sum() == 0:   # no segmentation found
                    print("No mask detected")
                    errorcounter += 1
                    if errorcounter > 20:
                        initialized = False
                        frame_counter = 0
                        continue
                    continue
                else:
                    errorcounter = 0
                
                
                dst_cloud = cam.get_pcd_from_rgbd(mask)
             
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

               
            # uv = project_points(cad_points, K, T_m2c)
            draw_model_projection_with_axes(color, cad_points, K, T_m2c)

            cv2.imshow("Live Tracking", color)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
           

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        cam.stop()
if __name__ == "__main__":
    main()





