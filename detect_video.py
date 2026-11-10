
import cv2
from ultralytics import YOLO
import os
import csv

# --------------------------------
# 1. File paths
# --------------------------------

input_video = "data/videos/12553362_1080_1920_30fps.mp4"
output_video = "outputs/videos/traffic_detected_new.mp4"
report_file = "reports/traffic_detection_report.csv"

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)

print("INPUT VIDEO:", os.path.abspath(input_video))
print("OUTPUT VIDEO:", os.path.abspath(output_video))

if not os.path.exists(input_video):
    raise FileNotFoundError(f"Input video not found: {input_video}")

# --------------------------------
# 2. Load YOLO model
# --------------------------------

print("\nLoading YOLOv8 model...")
model = YOLO("yolov8n.pt")

vehicle_classes = [2, 3, 5, 7]

# --------------------------------
# 3. Open input video
# --------------------------------

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise FileNotFoundError("Cannot open input traffic video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\nInput video details:")
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise ValueError("Invalid input video properties")

# --------------------------------
# 4. Create output video
# --------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video")

vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

frame_number = 0

# --------------------------------
# 5. Vehicle detection
# --------------------------------

print("\nStarting vehicle detection...")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.15,
        iou=0.45,
        imgsz=1280,
        classes=vehicle_classes,
        device="mps",
        verbose=False
    )

    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    for class_id in results[0].boxes.cls.tolist():
        name = model.names[int(class_id)]

        if name in vehicle_counts:
            vehicle_counts[name] += 1

    frame_number += 1

    if frame_number % 50 == 0:
        print(f"Processed {frame_number}/{total_frames} frames")

# --------------------------------
# 6. Release resources
# --------------------------------

cap.release()
out.release()

# --------------------------------
# 7. Save detection report
# --------------------------------

with open(report_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["Vehicle Type", "Detection Count"])

    for vehicle, count in vehicle_counts.items():
        writer.writerow([vehicle, count])

# --------------------------------
# 8. Final results
# --------------------------------

print("\nDetection completed!")
print("Frames processed:", frame_number)
print("Output video:", os.path.abspath(output_video))
print("Report:", os.path.abspath(report_file))
print("Vehicle detections:", vehicle_counts)

print("\nOutput file size:",
      round(os.path.getsize(output_video) / (1024 * 1024), 2), "MB")
