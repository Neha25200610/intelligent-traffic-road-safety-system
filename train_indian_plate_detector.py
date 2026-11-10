
from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train Indian number plate detector
results = model.train(
    data="data/indian_plate_detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    workers=2,
    project="runs",
    name="indian_plate_detector",
    exist_ok=True,
    patience=15,
    save=True,
    plots=True
)

print("Indian number plate detector training completed!")

# Display the best model location
print(
    "Best model:",
    "runs/indian_plate_detector/weights/best.pt"
)
