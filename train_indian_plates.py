from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Train on Indian licence plate character dataset
model.train(
    data="data/indian_plates_roboflow/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    workers=2,
    project="runs/detect",
    name="indian_plate_characters"
)

print("Indian plate character training completed!")