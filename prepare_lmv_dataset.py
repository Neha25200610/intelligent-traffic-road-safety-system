
import json
import random
import shutil
from pathlib import Path

source = Path("data/LMV-LP-Sample")
output = Path("data/lmv_yolo")

pairs = []

for image in (source / "videos").rglob("*.jpg"):
    relative = image.relative_to(source / "videos")
    annotation = source / "jsons" / relative.with_suffix(".json")

    if annotation.exists():
        pairs.append((image, annotation))

random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.8)

for index, (image, annotation) in enumerate(pairs):
    subset = "train" if index < split else "val"

    with open(annotation) as f:
        data = json.load(f)

    labels = []

    for item in data.values():
        points = item["licPoly"]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        labels.append(
            f"0 {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        )

    image_dir = output / "images" / subset
    label_dir = output / "labels" / subset

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image, image_dir / image.name)

    label_file = label_dir / f"{image.stem}.txt"
    label_file.write_text("\n".join(labels))

yaml_path = output / "data.yaml"
yaml_path.write_text(
    f"path: {output.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: license_plate\n"
)

print("Dataset preparation completed!")
print("Total images:", len(pairs))
print("Training images:", split)
print("Validation images:", len(pairs) - split)
print("Output:", output)
