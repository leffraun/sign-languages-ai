import cv2
import mediapipe as mp
import joblib
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "sign_language_rf.pkl"
LANDMARK_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# --------------------------------------------------
# 2. Load trained Random Forest model
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(LANDMARK_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)


# --------------------------------------------------
# 4. Open webcam
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# --------------------------------------------------
# 5. Start hand landmark detection
# --------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        # Read frame
        success, frame = cap.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = landmarker.detect(mp_image)

        # If a hand is detected
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Extract the same 63 features used during training
            features = []

            for landmark in hand:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Create DataFrame with same feature names
            columns = []

            for i in range(21):
                columns.extend([
                    f"x{i}",
                    f"y{i}",
                    f"z{i}"
                ])

            input_data = pd.DataFrame(
                [features],
                columns=columns
            )

            # Predict alphabet
            prediction = model.predict(input_data)[0]

            # Display prediction
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Draw landmarks
            for landmark in hand:

                h, w, _ = frame.shape

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )

        else:

            cv2.putText(
                frame,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Show webcam
        cv2.imshow("Indian Sign Language Recognition", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------
# 6. Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()
