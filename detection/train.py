from ultralytics import YOLO
 
model = YOLO("yolo11n-seg.pt")

result = model.train(
    data="/home/skoumal/dev/ObjectDetection/detection/dataset.yaml",
    epochs = 300,
    imgsz=640,
    batch=16,
    optimizer="Adam",
    lr0=0.001,
    device="0",
    name="Legoblock",
    save=True,
    save_json=True,
    project="/home/skoumal/dev/ObjectDetection/detection/YOLO/output_runs",
    exist_ok=True,
    resume=False,
    patience=10,
)
