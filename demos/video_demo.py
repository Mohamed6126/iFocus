import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import mediapipe as mp


from src.config import OUTPUT_CSV_PATH, FACE_MODEL_PATH, BASE_DIR, IMG_SIZE, MODEL_PATH, OUTPUT_VIDEO_PATH
LABEL_MAP = {
    0: "alert",
    1: "microsleep",
    2: "yawning"
}
# Load MobileNetV2 classifier

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


# Configure MediaPipe Face Detector

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    ),
    running_mode=RunningMode.IMAGE,
    min_detection_confidence=0.7
)

face_detector = FaceDetector.create_from_options(options)

# Open video
VIDEO_PATH = "demos/demo_videos/videoplayback.mp4"
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {fps:.2f}")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"Total frames: {total_frames}")


# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    fps,
    (frame_width, frame_height)
)

# Store predictions
predictions = []
frame_number = 0


# Process video
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1
    timestamp = frame_number / fps

    # Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face
    result = face_detector.detect(mp_image)

    # We expect one driver face.
    # If multiple faces are detected, use the largest one.
    if result.detections:

        detection = max(
            result.detections,
            key=lambda d: (
                d.bounding_box.width *
                d.bounding_box.height
            )
        )

        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        width = bbox.width
        height = bbox.height

        # Expand the face bounding box so the model sees
        # some context around the face, similar to FL3D images.

        margin_x = int(width * 0.10)
        margin_y = int(height * 0.10)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)

        x2 = min(frame_width, x + width + margin_x)
        y2 = min(frame_height, y + height + margin_y)

        face = frame[y1:y2, x1:x2]

        if face.size != 0:
            # BGR → RGB
            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face_tensor = tf.convert_to_tensor(
                face_rgb,
                dtype=tf.float32
            )
            face_tensor = tf.image.resize_with_pad(
                face_tensor,
                IMG_SIZE,
                IMG_SIZE
            )

            # Add batch dimension
            face_tensor = tf.expand_dims(
                face_tensor,
                axis=0
            )

            # MobileNetV2 prediction
            prediction = model.predict(
                face_tensor,
                verbose=0
            )[0]

            class_id = int(np.argmax(prediction))
            confidence = float(prediction[class_id])

            label = LABEL_MAP[class_id]

            # Save prediction
            predictions.append({
                "frame": frame_number,
                "timestamp": timestamp,
                "label": label,
                "confidence": confidence
            })

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw classification
            text = f"{label} ({confidence:.2f})"

            cv2.putText(
                frame,
                text,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Timestamp
            time_text = f"{timestamp:.1f}s"

            cv2.putText(
                frame,
                time_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    else:

        # ----------------------------------------------------
        # No face detected
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # Write annotated frame
    writer.write(frame)

    cv2.imshow(
        "FLD3 Trip Analysis",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
cap.release()
writer.release()
face_detector.close()
cv2.destroyAllWindows()


# Save prediction data
predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv(
    OUTPUT_CSV_PATH,
    index=False
)


print()
print("=" * 50)
print("Trip analysis complete")
print("=" * 50)

print(f"Annotated video: {OUTPUT_VIDEO_PATH}")
print(f"Predictions CSV: {OUTPUT_CSV_PATH}")
print(f"Frames processed: {frame_number}")
print(f"Frames classified: {len(predictions_df)}")