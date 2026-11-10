
from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train on our custom license plate dataset
model.train(
    data="data/lmv_yolo/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="mps",
    project="runs/license_plate",
    name="lmv_model"
)

print("License plate model training completed!")
