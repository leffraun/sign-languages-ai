import mediapipe as mp
import cv2
import csv
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset - Gesture Speech"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUTPUT_PATH = PROJECT_ROOT / "landmarks" / "landmarks.csv"


# --------------------------------------------------
# 2. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)


# --------------------------------------------------
# 3. CSV header
# --------------------------------------------------

header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# --------------------------------------------------
# 4. Process the dataset
# --------------------------------------------------

total_images = 0
successful_images = 0
failed_images = 0

with HandLandmarker.create_from_options(options) as landmarker:

    with open(OUTPUT_PATH, "w", newline="") as csv_file:

        writer = csv.writer(csv_file)

        # Write column names
        writer.writerow(header)

        # Go through every class folder
        for class_folder in sorted(DATASET_PATH.iterdir()):

            if not class_folder.is_dir():
                continue

            label = class_folder.name

            print(f"\nProcessing class: {label}")

            # Get all JPG images
            images = list(class_folder.glob("*.jpg"))

            for image_path in images:

                total_images += 1

                try:
                    # Load image
                    image = cv2.imread(str(image_path))

                    if image is None:
                        failed_images += 1
                        continue

                    # Convert BGR → RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Convert to MediaPipe image
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image
                    )

                    # Detect hand
                    result = landmarker.detect(mp_image)

                    # Skip if no hand was detected
                    if not result.hand_landmarks:
                        failed_images += 1
                        continue

                    # Get first detected hand
                    hand = result.hand_landmarks[0]

                    # Make one row
                    row = [label]

                    for landmark in hand:
                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    # Save row
                    writer.writerow(row)

                    successful_images += 1

                except Exception as e:
                    failed_images += 1
                    print(f"Error processing {image_path}: {e}")

                # Progress
                if total_images % 100 == 0:
                    print(
                        f"Images processed: {total_images} | "
                        f"Successful: {successful_images} | "
                        f"Failed: {failed_images}"
                    )


# --------------------------------------------------
# 5. Final report
# --------------------------------------------------

print("\n==============================")
print("EXTRACTION COMPLETE")
print("==============================")

print(f"Total images:      {total_images}")
print(f"Successful:        {successful_images}")
print(f"Failed/skipped:    {failed_images}")
print(f"CSV saved to:      {OUTPUT_PATH}")
