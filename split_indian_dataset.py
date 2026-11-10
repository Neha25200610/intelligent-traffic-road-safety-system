import random
import shutil
from pathlib import Path

base = Path("data/indian_plates_roboflow")

images_dir = base / "train/images"
labels_dir = base / "train/labels"

valid_images = base / "valid/images"
valid_labels = base / "valid/labels"

random.seed(42)

images = list(images_dir.glob("*.jpg"))
random.shuffle(images)

validation_count = int(len(images) * 0.2)
validation_images = images[:validation_count]

for image in validation_images:
    label = labels_dir / f"{image.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(f"Missing label: {label}")

    shutil.move(str(image), str(valid_images / image.name))
    shutil.move(str(label), str(valid_labels / label.name))

print("Dataset split completed!")
print("Validation images:", len(validation_images))
print("Training images:", len(list(images_dir.glob("*.jpg"))))