from EstimHelpers.detection_utils import detect_mask
import cv2 

if __name__ == "__main__":
    weights = "/home/skoumal/dev/ObjectDetection/detection/best.pt"
    img_path = "/home/skoumal/dev/ObjectDetection/datageneration/Blenderproc/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000000.jpg"
    out_dir = "/home/skoumal/dev/ObjectDetection/detection/"

    mask = detect_mask(weights, img_path, out_dir)

 