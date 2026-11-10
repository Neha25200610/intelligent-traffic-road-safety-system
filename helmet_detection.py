
from ultralytics import YOLO
import cv2
import csv
import os
from collections import Counter

# Load trained helmet model
model = YOLO("models/helmet_model.pt")

# Input and output paths
input_video = "data/videos/vedio traffic.mp4"
output_video = "outputs/videos/helmet_detected.mp4"
report_file = "reports/helmet_detection_report.csv"

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Open video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {input_video}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

counts = Counter()
rows = []
frame_number = 0

# Process video frames
while True:
    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    results = model.predict(
        frame,
        conf=0.35,
        device="mps",
        verbose=False
    )

    for result in results:
        annotated_frame = result.plot()

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            counts[class_name] += 1

            rows.append([
                frame_number,
                class_name,
                round(confidence, 3)
            ])

    writer.write(annotated_frame)

    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

cap.release()
writer.release()

# Save CSV report
with open(report_file, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Frame", "Detection", "Confidence"])
    csv_writer.writerows(rows)

print("\nHelmet detection completed!")
print("Output video:", output_video)
print("Report:", report_file)
print("Detection counts:", dict(counts))
