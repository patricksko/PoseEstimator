import numpy as np   
import cv2                     # fundamental package for scientific computing
import open3d as o3d                # Intel RealSense cross-platform open-source API
from pose_estimator.EstimHelpers.HelpersRealtime import *
from pose_estimator.EstimHelpers.RealSenseClass import RealSenseCamera
from pose_estimator.EstimHelpers.PoseEstimator import PoseEstimator
from pose_estimator.EstimHelpers.Detector import Detector
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
    
    estimator = PoseEstimator(CAD_PATH, PCD_PATH, intr, K, TARGET_PTS) # call the constructor of PoseEstimator for 6d-Pose
    detector = Detector(WEIGHTS_PATH)
    # Read CAD Model for comparision
    mesh = o3d.io.read_triangle_mesh(CAD_PATH)
    mesh.compute_vertex_normals()

    # Sample more points (e.g. 50,000 points)
    pcd = mesh.sample_points_uniformly(number_of_points=200)

    # or use Poisson disk sampling (more even distribution)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=50000)

    cad_points = np.asarray(pcd.points)
    
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
                    detections = detector.detect_mask(color)
                    print(detections)
                    if len(detections) == 0:
                        continue
                    mask = detections[0]["mask"]
                    if mask is None or mask.sum() == 0:
                        frame_counter = 0
                    frame_counter+=1
                
                dst_cloud = cam.get_pcd_from_rgbd(mask)
               
                # First guess with templates
                H_init = estimator.find_best_template_teaser(dst_cloud)
                H_init = enforce_upright_pose_y_up(H_init)

                initialized = True
                T_m2c = H_init.copy()
                
                continue

            all = time.time()
            # Projection (intrinsics) for this frame
            start = time.time()
            prev_down = estimator.create_template_from_H(T_m2c=T_m2c, target_points=TARGET_PTS)
            timer_print(start, "Rendering")

            start = time.time()
            # prev_down, _ = preprocess_point_cloud_uniform(src, TARGET_PTS, False)
            timer_print(start, "Preprocessing Previous")
            frame_id += 1
            if frame_id % TRACK_EVERY == 0:
                detections = detector.detect_mask(color)
                
                if len(detections) == 0:   # no segmentation found
                    print("No mask detected")
                    errorcounter += 1
                    if errorcounter > 5:
                        initialized = False
                        frame_counter = 0
                        continue
                    continue
                else:
                    mask = detections[0]["mask"]
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





