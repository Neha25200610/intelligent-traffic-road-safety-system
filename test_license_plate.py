
from ultralytics import YOLO
import cv2

# Load custom trained model
model = YOLO(
    "runs/detect/runs/license_plate/lmv_model/weights/best.pt"
)

# Test image
image_path = "data/lmv_yolo/images/val/frame29.jpg"

# Run detection
results = model.predict(
    source=image_path,
    conf=0.25,
    save=True
)

# Display detection count
for result in results:
    print("License plates detected:", len(result.boxes))

print("Testing completed!")
