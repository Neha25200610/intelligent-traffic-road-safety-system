
from ultralytics import YOLO
import easyocr
import cv2
import csv
import os
import re

# Load trained Indian number plate detection model
model = YOLO(
    "runs/detect/runs/indian_plate_detector/weights/best.pt"
)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Input and output paths
input_video = "data/videos/indian_anpr_sample.mp4"
output_video = "outputs/videos/number_plate_detected.mp4"
report_file = "reports/number_plate_report.csv"

os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Open input video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {input_video}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video
writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError("Could not create output video")

rows = []
frame_number = 0
total_plates = 0
readable_plates = 0

print("Starting Indian number plate detection and OCR...")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Detect number plates
    results = model.predict(
        frame,
        conf=0.25,
        device="mps",
        verbose=False
    )

    annotated_frame = frame.copy()

    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            # Keep coordinates within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            total_plates += 1

            # Crop number plate
            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Enlarge plate image
            enlarged = cv2.resize(
                plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )

            # Convert to grayscale
            gray = cv2.cvtColor(
                enlarged,
                cv2.COLOR_BGR2GRAY
            )

            # Reduce noise
            gray = cv2.bilateralFilter(
                gray, 9, 75, 75
            )

            # Improve contrast
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            enhanced = clahe.apply(gray)

            # Try multiple image versions with OCR
            candidates = []

            for image in [enlarged, gray, enhanced]:

                if len(image.shape) == 2:
                    ocr_image = image
                else:
                    ocr_image = cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2GRAY
                    )

                ocr_results = reader.readtext(
                    ocr_image,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

                for item in ocr_results:

                    detected_text = item[1].upper()

                    ocr_confidence = float(item[2])

                    cleaned_text = re.sub(
                        r'[^A-Z0-9]',
                        '',
                        detected_text
                    )

                    if len(cleaned_text) >= 4:
                        candidates.append(
                            (cleaned_text, ocr_confidence)
                        )

            # Select best OCR result
            if candidates:

                plate_text, ocr_score = max(
                    candidates,
                    key=lambda item: (
                        item[1],
                        len(item[0])
                    )
                )

                readable_plates += 1

            else:
                plate_text = "Text not readable"
                ocr_score = 0.0

            print(
                f"Frame {frame_number}: "
                f"{plate_text} | "
                f"YOLO confidence: {confidence:.3f} | "
                f"OCR confidence: {ocr_score:.3f}"
            )

            # Draw detection box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Draw detected text
            cv2.putText(
                annotated_frame,
                plate_text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Save detection report
            rows.append([
                frame_number,
                plate_text,
                round(confidence, 3),
                round(ocr_score, 3),
                x1, y1, x2, y2
            ])

    writer.write(annotated_frame)

    if frame_number % 100 == 0:
        print(f"Processed {frame_number} frames")

# Release resources
cap.release()
writer.release()

# Save CSV report
with open(report_file, "w", newline="") as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow([
        "Frame Number",
        "Detected Plate Text",
        "YOLO Detection Confidence",
        "OCR Confidence",
        "X1", "Y1", "X2", "Y2"
    ])

    csv_writer.writerows(rows)

print("\nIndian number plate detection and OCR completed!")
print("Total plate detections:", total_plates)
print("Readable plate results:", readable_plates)
print("Output video:", output_video)
print("Report:", report_file)
