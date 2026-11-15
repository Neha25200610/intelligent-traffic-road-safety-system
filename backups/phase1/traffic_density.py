
import cv2
from ultralytics import YOLO
import os
import csv

input_video = "data/videos/vedio traffic.mp4"
report_file = "reports/traffic_density_report.csv"

os.makedirs("reports", exist_ok=True)

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open input traffic video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_number = 0
density_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

print("Starting traffic density analysis...")

with open(report_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Frame Number", "Vehicle Count", "Traffic Density"])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.25,
            classes=vehicle_classes,
            device="mps",
            verbose=False
        )

        vehicle_count = len(results[0].boxes)

        if vehicle_count <= 3:
            density = "LOW"
        elif vehicle_count <= 7:
            density = "MEDIUM"
        else:
            density = "HIGH"

        density_counts[density] += 1

        writer.writerow([
            frame_number + 1,
            vehicle_count,
            density
        ])

        frame_number += 1

        if frame_number % 50 == 0:
            print(f"Processed {frame_number}/{total_frames} frames")

cap.release()

print("\nTraffic density analysis completed!")
print("Total frames analysed:", frame_number)
print("Density summary:", density_counts)
print("Report saved:", report_file)
