
import cv2
import csv
import os
from ultralytics import YOLO
from collections import Counter

# Paths
MODEL_PATH = "runs/detect/runs/detect/traffic_light_detector/weights/best.pt"
VIDEO_PATH = "data/videos/indian_signal.mp4"
OUTPUT_VIDEO = "outputs/videos/traffic_light_detected.mp4"
REPORT_PATH = "reports/traffic_light_report.csv"

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

class_counts = Counter()
frame_number = 0
total_detections = 0

# Process video
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.40, verbose=False)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])

            class_counts[label] += 1
            total_detections += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    frame_number += 1
    out.write(frame)

cap.release()
out.release()

# Save CSV report
with open(REPORT_PATH, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Traffic Light State", "Frame-level Detections"])

    for label in ["Red", "Yellow", "Green"]:
        writer.writerow([label, class_counts[label]])

    writer.writerow(["Total", total_detections])

print("\nTraffic Light Detection Completed!")
print("Frames processed:", frame_number)
print("Total detections:", total_detections)
print("Class counts:", dict(class_counts))
print("Output video:", OUTPUT_VIDEO)
print("CSV report:", REPORT_PATH)
