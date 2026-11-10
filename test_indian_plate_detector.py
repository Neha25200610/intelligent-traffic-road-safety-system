from ultralytics import YOLO

# Load trained Indian number plate detector
model = YOLO("runs/detect/runs/indian_plate_detector/weights/best.pt")

# Test on validation images
results = model.predict(
    source="data/indian_plate_detection/valid/images",
    conf=0.25,
    save=True
)

print("Indian plate detection testing completed!")
print("Images tested:", len(results))
print("Results saved in the runs/detect folder")