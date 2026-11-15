
import cv2
import csv
import os
from collections import deque, Counter
from ultralytics import YOLO

from database import save_red_light_violations


# =====================================================
#                  PROJECT PATHS
# =====================================================

VEHICLE_MODEL_PATH = "yolov8n.pt"

TRAFFIC_LIGHT_MODEL_PATH = (
    "runs/detect/runs/detect/traffic_light_detector/weights/best.pt"
)

INPUT_VIDEO = "data/videos/rlr_occasions.mp4"

OUTPUT_VIDEO = "outputs/videos/red_light_violations.mp4"

CSV_REPORT = "reports/red_light_violations.csv"


# =====================================================
#                  CONFIGURATION
# =====================================================

STOP_LINE_Y = 425

VEHICLE_CONFIDENCE = 0.25
SIGNAL_CONFIDENCE = 0.10

SIGNAL_HISTORY_SIZE = 15

# COCO vehicle classes
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# =====================================================
#                  CREATE DIRECTORIES
# =====================================================

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# =====================================================
#                  LOAD YOLO MODELS
# =====================================================

print("Loading vehicle detection model...")

vehicle_model = YOLO(VEHICLE_MODEL_PATH)

print("Loading traffic light detection model...")

signal_model = YOLO(TRAFFIC_LIGHT_MODEL_PATH)


# =====================================================
#                  OPEN INPUT VIDEO
# =====================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open input video: {INPUT_VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# =====================================================
#                  TRACKING VARIABLES
# =====================================================

signal_history = deque(maxlen=SIGNAL_HISTORY_SIZE)

previous_positions = {}

violated_vehicle_ids = set()

rows = []

vehicle_detection_count = 0

signal_observations = Counter()

frame_number = 0


# =====================================================
#                  PROCESS VIDEO
# =====================================================

print("\nStarting red-light violation detection...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # -------------------------------------------------
    # TRAFFIC LIGHT DETECTION
    # -------------------------------------------------

    signal_results = signal_model(
        frame,
        conf=SIGNAL_CONFIDENCE,
        verbose=False
    )

    detected_signal_states = []

    for result in signal_results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            class_name = signal_model.names[class_id]

            state = str(class_name).strip().lower()

            if state in ["red", "yellow", "green"]:

                detected_signal_states.append(state)

                signal_observations[state.capitalize()] += 1

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                color = {
                    "red": (0, 0, 255),
                    "yellow": (0, 255, 255),
                    "green": (0, 255, 0)
                }[state]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{state.upper()} {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Update signal history when a signal is detected
    if detected_signal_states:

        current_signal = Counter(
            detected_signal_states
        ).most_common(1)[0][0]

        signal_history.append(current_signal)

    # Confirm signal state using recent history
    confirmed_signal = "UNKNOWN"

    if signal_history:

        signal_counts = Counter(signal_history)

        red_ratio = (
            signal_counts["red"] / len(signal_history)
        )

        if red_ratio >= 0.80:

            confirmed_signal = "RED"

        elif signal_counts["green"] > 0:

            confirmed_signal = "GREEN"

        elif signal_counts["yellow"] > 0:

            confirmed_signal = "YELLOW"

    # -------------------------------------------------
    # DRAW STOP LINE
    # -------------------------------------------------

    cv2.line(
        frame,
        (0, STOP_LINE_Y),
        (width, STOP_LINE_Y),
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        "STOP LINE",
        (20, STOP_LINE_Y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -------------------------------------------------
    # VEHICLE DETECTION AND TRACKING
    # -------------------------------------------------

    vehicle_results = vehicle_model.track(
        frame,
        conf=VEHICLE_CONFIDENCE,
        persist=True,
        tracker="bytetrack.yaml",
        device="mps",
        verbose=False
    )

    for result in vehicle_results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        track_ids = (
            result.boxes.id.int().cpu().tolist()
        )

        class_ids = (
            result.boxes.cls.int().cpu().tolist()
        )

        confidences = (
            result.boxes.conf.cpu().tolist()
        )

        for box, track_id, class_id, confidence in zip(
            boxes,
            track_ids,
            class_ids,
            confidences
        ):

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_detection_count += 1

            vehicle_type = VEHICLE_CLASSES[class_id]

            x1, y1, x2, y2 = map(int, box)

            # Bottom-center position of vehicle
            current_y = int(y2)

            center_x = int((x1 + x2) / 2)

            # -------------------------------------------------
            # RED-LIGHT STOP-LINE CROSSING DETECTION
            # -------------------------------------------------

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                crossed_stop_line = (
                    previous_y < STOP_LINE_Y
                    and current_y >= STOP_LINE_Y
                )

                if (
                    crossed_stop_line
                    and confirmed_signal == "RED"
                    and track_id not in violated_vehicle_ids
                ):

                    violated_vehicle_ids.add(track_id)

                    timestamp = frame_number / fps

                    event = "Red-light stop-line crossing"

                    rows.append([
                        track_id,
                        vehicle_type,
                        frame_number,
                        round(timestamp, 2),
                        confirmed_signal,
                        event
                    ])

                    print(
                        f"Potential violation: "
                        f"Vehicle ID {track_id} | "
                        f"{vehicle_type} | "
                        f"Frame {frame_number} | "
                        f"{timestamp:.2f} seconds"
                    )

            previous_positions[track_id] = current_y

            # -------------------------------------------------
            # DRAW VEHICLE BOUNDING BOX
            # -------------------------------------------------

            if track_id in violated_vehicle_ids:

                box_color = (0, 0, 255)

                label = (
                    f"VIOLATION ID:{track_id} "
                    f"{vehicle_type}"
                )

            else:

                box_color = (0, 255, 0)

                label = (
                    f"ID:{track_id} {vehicle_type}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    # -------------------------------------------------
    # DISPLAY INFORMATION
    # -------------------------------------------------

    signal_color = {
        "RED": (0, 0, 255),
        "GREEN": (0, 255, 0),
        "YELLOW": (0, 255, 255),
        "UNKNOWN": (200, 200, 200)
    }.get(confirmed_signal, (200, 200, 200))

    cv2.putText(
        frame,
        f"Signal: {confirmed_signal}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        signal_color,
        2
    )

    cv2.putText(
        frame,
        f"Potential Violations: {len(rows)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -------------------------------------------------
    # WRITE OUTPUT VIDEO
    # -------------------------------------------------

    out.write(frame)

    if frame_number % 100 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# =====================================================
#                  RELEASE RESOURCES
# =====================================================

cap.release()
out.release()

cv2.destroyAllWindows()


# =====================================================
#                  SAVE CSV REPORT
# =====================================================

csv_columns = [
    "Vehicle ID",
    "Vehicle Type",
    "Frame Number",
    "Timestamp (seconds)",
    "Signal State",
    "Event"
]

with open(
    CSV_REPORT,
    "w",
    newline=""
) as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(csv_columns)

    csv_writer.writerows(rows)


print("\nCSV report saved successfully:", CSV_REPORT)


# =====================================================
#             SAVE VIOLATIONS TO POSTGRESQL
# =====================================================

try:

    save_red_light_violations(rows)

    print(
        f"Successfully saved {len(rows)} "
        "violation records to PostgreSQL."
    )

except Exception as e:

    print("PostgreSQL storage failed:", e)


# =====================================================
#                  FINAL SUMMARY
# =====================================================

print("\n====================================")
print(" RED-LIGHT VIOLATION DETECTION DONE")
print("====================================")

print("Total frames processed:", frame_number)

print("Total vehicle detections:", vehicle_detection_count)

print("Red signal observations:", signal_observations["Red"])

print("Green signal observations:", signal_observations["Green"])

print("Yellow signal observations:", signal_observations["Yellow"])

print("Potential red-light violations:", len(rows))

print("Unique vehicle IDs:", len(violated_vehicle_ids))

print("Output video:", OUTPUT_VIDEO)

print("CSV report:", CSV_REPORT)

print("====================================")
