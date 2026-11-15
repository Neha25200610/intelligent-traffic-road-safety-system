
from ultralytics import YOLO
import cv2
import csv
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input and output paths
input_video = "data/videos/vedio traffic.mp4"
output_video = "outputs/videos/traffic_violations.mp4"
report_file = "reports/traffic_violations.csv"

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

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

# Stop-line position
stop_line_y = int(height * 0.70)

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Tracking variables
previous_positions = {}
crossed_vehicles = set()
rows = []

frame_number = 0
total_crossings = 0
total_detections = 0
frames_without_ids = 0

print("Starting traffic stop-line detection...")
print("Stop line Y-coordinate:", stop_line_y)

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(vehicle_classes.keys()),
        conf=0.25,
        device="mps",
        verbose=False
    )

    annotated = frame.copy()

    # Draw stop line
    cv2.line(
        annotated,
        (0, stop_line_y),
        (width, stop_line_y),
        (0, 0, 255),
        4
    )

    cv2.putText(
        annotated,
        "STOP LINE",
        (30, stop_line_y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    for result in results:

        # Check vehicle detections
        if result.boxes is None or len(result.boxes) == 0:
            continue

        total_detections += len(result.boxes)

        # Check tracking IDs
        if result.boxes.id is None:
            frames_without_ids += 1
            print(f"No tracking IDs in frame {frame_number}")
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            vehicle_name = vehicle_classes[class_id]

            # Bottom-center position
            center_y = y2

            # Debug information every 10 frames
            if frame_number % 10 == 0:
                print(
                    f"Frame: {frame_number}, "
                    f"Vehicle ID: {track_id}, "
                    f"Position: {center_y}, "
                    f"Stop line: {stop_line_y}"
                )

            color = (0, 255, 0)

            if track_id in crossed_vehicles:
                color = (0, 0, 255)

            # Draw vehicle box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                annotated,
                f"{vehicle_name} ID:{track_id}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # Detect crossing
            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                if (
                    previous_y < stop_line_y
                    and center_y >= stop_line_y
                    and track_id not in crossed_vehicles
                ):

                    crossed_vehicles.add(track_id)
                    total_crossings += 1

                    timestamp = frame_number / fps

                    rows.append([
                        track_id,
                        vehicle_name,
                        frame_number,
                        round(timestamp, 2),
                        "Stop-line crossing detected"
                    ])

                    print(
                        f"CROSSING DETECTED: "
                        f"ID {track_id}, "
                        f"{vehicle_name}, "
                        f"Time {timestamp:.2f}s"
                    )

            previous_positions[track_id] = center_y

    writer.write(annotated)

    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

# Release resources
cap.release()
writer.release()

# Save CSV report
with open(report_file, "w", newline="") as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Vehicle ID",
        "Vehicle Type",
        "Frame Number",
        "Timestamp (seconds)",
        "Event"
    ])

    csv_writer.writerows(rows)

print("\nTraffic stop-line detection completed!")
print("Total frames processed:", frame_number)
print("Total vehicle detections:", total_detections)
print("Frames without tracking IDs:", frames_without_ids)
print("Crossing events:", total_crossings)
print("Output video:", output_video)
print("Report:", report_file)
