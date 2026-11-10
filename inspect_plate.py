import cv2
import pandas as pd

video = "data/videos/vedio traffic.mp4"
report = "reports/number_plate_report.csv"

df = pd.read_csv(report)

# Select the detection with highest confidence
row = df.loc[df["Detection Confidence"].idxmax()]

cap = cv2.VideoCapture(video)

frame_no = int(row["Frame Number"])
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)

success, frame = cap.read()

if success:
    x1, y1, x2, y2 = [
        int(row[c]) for c in ["X1", "Y1", "X2", "Y2"]
    ]

    crop = frame[y1:y2, x1:x2]

    enlarged = cv2.resize(
        crop, None, fx=5, fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    cv2.imwrite(
        "outputs/videos/plate_crop_inspection.png",
        enlarged
    )

    print("Frame:", frame_no)
    print("Confidence:", row["Detection Confidence"])
    print("Crop size:", crop.shape)
    print("Saved: outputs/videos/plate_crop_inspection.png")
else:
    print("Could not read the selected frame")

cap.release()