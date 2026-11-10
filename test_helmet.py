
from ultralytics import YOLO
import os

# Load trained helmet model
model = YOLO("models/helmet_model.pt")

# Test on one validation image
image_path = "data/helmet_dataset/extracted/valid/images"

image = os.listdir(image_path)[0]
image_path = os.path.join(image_path, image)

# Run helmet detection
results = model.predict(
    source=image_path,
    conf=0.25,
    device="mps",
    save=True
)

print("Helmet detection test completed!")
print("Test image:", image)
print("Detected objects:")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(model.names[class_id], round(confidence, 2))
