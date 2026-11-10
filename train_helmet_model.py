
from ultralytics import YOLO

# Load the YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the helmet detection model
results = model.train(
    data="data/helmet_dataset/extracted/helmet_data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="mps",
    workers=2,
    project="runs/helmet",
    name="helmet_yolov8n",
    patience=10,
    plots=True
)

print("Helmet model training completed!")
