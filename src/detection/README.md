# 🚀 YOLO Object Detection – LegoBlock Example

This repository provides a workflow for **training and testing YOLO models** (with segmentation) on a custom dataset of Lego blocks.

---
## 🎨 1. Visualization of Segmentation Polygons
Use ```testrun.py``` to check whether YOLO’s segmentation outputs (or your manual annotations) are correct. This script overlays segmentation polygons on the original image for verification.

## 📂 2. Dataset Setup

Before training, configure the dataset paths in [`dataset.yaml`](./dataset.yaml).  
Update the file so it points to your dataset:

```bash
path: /home/skoumal/dev/ObjectDetection/detection/Blenderproc/output_blenderproc_full/bop_data/Legoblock  # dataset root dir
train: train_tuutuuut        # training set folder (relative to 'path')
val:   val_tuutuuut          # validation set folder (relative to 'path')
test:                        # (optional) test images folder
```
+ ```path``` -> absolute path to your dataset root directory
+ ```train```/```val```-> relative paths inside ```path``` where images + labels are stored
+ ```test``` (optional) -> for running evaluations after training

## 🏋️ 3. Training
Modify the training script (e.g. ```train.py```) at following positions:
+ ```data``` → path to your dataset.yaml file
+ ```epochs```, ```batch```, and ```lr0``` can be adjusted depending on dataset size and hardware
+ ```name``` → name of the run (used for output folders and weight naming)
+ Weights will be saved under:
```bash
/home/skoumal/dev/ObjectDetection/detection/YOLO/output_runs/Legoblock/
```
## 🔍 4. Testing / Prediction
You can test trained weights with ```predict.py```



# ✅ Summary
+ Verify segmentation polygons with ```testrun.py```
+ Configure dataset paths in ```dataset.yaml```
+ Train with YOLO (```train.py```)
+ Test with ```predict.py```
