
from pathlib import Path
import shutil

SOURCE = Path("data/indian_plates_roboflow")
DEST = Path("data/indian_plate_detection")

for split in ["train", "valid"]:
    image_dir = SOURCE / split / "images"
    label_dir = SOURCE / split / "labels"

    output_images = DEST / split / "images"
    output_labels = DEST / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    count = 0

    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text().splitlines()

        # Keep only whole-plate annotations
        if len(lines) == 1:
            parts = lines[0].split()

            if parts[0] == "24":
                image_file = image_dir / (label_file.stem + ".jpg")

                if image_file.exists():
                    shutil.copy2(
                        image_file,
                        output_images / image_file.name
                    )

                    # Convert class ID 24 to class ID 0
                    parts[0] = "0"

                    new_label = " ".join(parts) + "\n"

                    (output_labels / label_file.name).write_text(
                        new_label
                    )

                    count += 1

    print(f"{split}: copied and corrected {count} images")

print("Plate detection dataset corrected!")
