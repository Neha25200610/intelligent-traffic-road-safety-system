from ultralytics import YOLO
import cv2

model = YOLO(
    "runs/detect/runs/detect/traffic_light_detector/weights/best.pt"
)

video_path = "data/videos/vedio traffic.mp4"
output_path = "outputs/videos/traffic_light_test.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print("Traffic light visualization completed!")
print("Output:", output_path)