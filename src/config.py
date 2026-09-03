from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "classification_frames"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_mobilenet.keras"

IMG_SIZE = 224
BATCH_SIZE = 32

LABEL_MAP = {
    "alert": 0,
    "microsleep": 1,
    "yawning": 2
}


OUTPUT_VIDEO_PATH = (
    BASE_DIR / "demos" / "demo_videos" / "trip_analyzed.mp4"
)

FACE_MODEL_PATH = (
    BASE_DIR / "models" / "blaze_face_short_range.tflite"
)


OUTPUT_CSV_PATH = (
    BASE_DIR / "demos" / "demo_videos" / "trip_predictions.csv"
)