# Pose Estimator
This project aims to build a complete pipeline for 6D object pose estimation by combining synthetic data generation, object detection, and registration techniques.

**1. Data Generation with BlenderProc:**  
    We start by rendering synthetic datasets using BlenderProc. Synthetic data allows us to generate diverse, perfectly annotated images of objects under various lighting and pose conditions, which are difficult and costly to capture manually. This helps train more robust and      generalizable object detection models.

**2. Data Conversion to YOLO Format:**  
    After rendering, the generated annotations are converted into the YOLO format. YOLO is a popular and efficient object detection framework that requires labeled bounding boxes in a specific format. Converting the data ensures compatibility with YOLO’s training pipeline.

**3. Training Object Detection Model:**  
    Using the converted dataset, we train an object detection model with Ultralytics YOLO. This model learns to detect and localize objects in images, providing bounding boxes essential for downstream tasks.

**4. 6D Pose Estimation via Registration:**  
    Finally, to estimate the full 6D pose (3D position + orientation) of detected objects, we use TEASER++ for point cloud registration. Registration aligns the detected object with its CAD model, enabling precise pose estimation required for robotics and augmented reality applications.

This end-to-end pipeline leverages synthetic data and state-of-the-art methods to achieve accurate 6D pose estimation.

# Install
We recommend using **Conda** to create an isolated environment.

## 1. Create and activate a new Conda environment (Python 3.10)

### Linux / macOS / Windows
```bash
conda create -n pose_estimator_env python=3.10 -y
conda activate pose_estimator_env
```

## 2. Install dependencies
You can install the required Python packages in two ways
### Option A - Using ```requirements.txt```
```bash
pip install -r requirements.txt
```
### Option B - Using ```conda_env.yaml```
This will reproduce the environment exactly as specified:
```bash
conda env create -f conda_env.yaml
conda activate pose_estimator_env
```
If you already created the environment manually in Step 1 and just want to update it with the ```conda_env.yaml``` file:
```bash
conda env update -n pose_estimator_env -f conda_env.yaml --prune
```
## Notes
Use **Option A** if you just want to install Python Packages quickly.
Use **Option B** if you want reproducibility across machines.



```bash
conda env create -f conda_env.yaml
```
2. Install the required Python packages from requirements.txt:
 ```
pip install -r requirements.txt
```
# Usage
For detailed instructions on each component, please refer to the README files located within their respective subfolders.
