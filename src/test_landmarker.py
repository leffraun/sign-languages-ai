import mediapipe as mp
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = str(PROJECT_ROOT / "models" / "hand_landmarker.task")
IMAGE_PATH = str(
    PROJECT_ROOT / "dataset" / "dataset - Gesture Speech" / "a" / "0.jpg"
)

# MediaPipe classes
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Create the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE
)

with HandLandmarker.create_from_options(options) as landmarker:

    # Load our test image
    image = cv2.imread(IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    # Detect the hand
    result = landmarker.detect(mp_image)

    # Check whether a hand was detected
    if not result.hand_landmarks:
        print("No hand detected!")
    else:
        print("Hand detected!")
        print(f"Number of hands: {len(result.hand_landmarks)}")

        # Get the first detected hand
        hand = result.hand_landmarks[0]

        # Print all 21 landmarks
        for i, landmark in enumerate(hand):
            print(
                f"Landmark {i}: "
                f"x={landmark.x:.4f}, "
                f"y={landmark.y:.4f}, "
                f"z={landmark.z:.4f}"
            )
