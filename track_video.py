import cv2
from ultralytics import YOLO
import os
import csv

# File paths
input_video = "data/videos/vedio traffic.mp4"
output_video = "outputs/videos/vehicle_tracking.mp4"
report_file = "reports/vehicle_tracking_report.csv"

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open input traffic video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Store unique tracking IDs by vehicle type
tracked_vehicles = {
    "car": set(),
    "motorcycle": set(),
    "bus": set(),
    "truck": set()
}

frame_number = 0

print("Starting vehicle tracking...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        classes=vehicle_classes,
        device="mps",
        verbose=False
    )

    result = results[0]
    annotated_frame = result.plot()
    out.write(annotated_frame)

    if result.boxes is not None and result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(track_ids, class_ids):
            vehicle_name = model.names[class_id]
            if vehicle_name in tracked_vehicles:
                tracked_vehicles[vehicle_name].add(track_id)

    frame_number += 1

    if frame_number % 50 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")

cap.release()
out.release()

# Save unique vehicle tracking report
with open(report_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Vehicle Type", "Unique Tracked Vehicles"])

    for vehicle, ids in tracked_vehicles.items():
        writer.writerow([vehicle, len(ids)])

print("\nVehicle tracking completed!")
print("Tracking video:", output_video)
print("Tracking report:", report_file)
print("Unique tracked vehicles:")

for vehicle, ids in tracked_vehicles.items():
    print(vehicle, ":", len(ids))