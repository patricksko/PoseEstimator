# 🚀 YOLO Object Detection – LegoBlock Example

This repository provides a workflow for **training and testing YOLO models** (with segmentation) on a custom dataset of Lego blocks.

---

## 📂 1. Dataset Setup

Before training, configure the dataset paths in [`dataset.yaml`](./dataset.yaml).  
Update the file so it points to your dataset:

```yaml
path: /home/skoumal/dev/ObjectDetection/detection/Blenderproc/output_blenderproc_full/bop_data/Legoblock  # dataset root dir
train: train_tuutuuut        # training set folder (relative to 'path')
val:   val_tuutuuut          # validation set folder (relative to 'path')
test:                        # (optional) test images folder
```
