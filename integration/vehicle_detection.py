import os
import csv
import subprocess
import cv2
import torch
from ultralytics import YOLO


def run_vehicle_detection(
    input_video,
    output_video,
    report_file,
    progress_callback=None
):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video not found: {input_video}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading YOLO model on {device}...")
    model = YOLO("yolov8n.pt")

    vehicle_classes = [2, 3, 5, 7]

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid input video properties")

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

    try:
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
                device=device,
                verbose=False
            )

            result = results[0]
            out.write(result.plot())

            if result.boxes is not None:
                for class_id in result.boxes.cls.tolist():
                    name = model.names[int(class_id)]

                    if name in vehicle_counts:
                        vehicle_counts[name] += 1

            frame_number += 1

            if progress_callback:
                progress_callback(frame_number, total_frames)

    finally:
        cap.release()
        out.release()

    with open(report_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Vehicle Type", "Detection Count"])

        for vehicle, count in vehicle_counts.items():
            writer.writerow([vehicle, count])

    h264_output_video = os.path.splitext(output_video)[0] + "_h264.mp4"

    print("Converting output video to H.264...")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", output_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            h264_output_video
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )

    return {
        "frames_processed": frame_number,
        "vehicle_counts": vehicle_counts,
        "output_video": h264_output_video,
        "report_file": report_file
    }
