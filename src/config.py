from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "classification_frames"
MODEL_DIR = BASE_DIR / "models"

IMG_SIZE = 224
BATCH_SIZE = 32

LABEL_MAP = {
    "alert": 0,
    "microsleep": 1,
    "yawning": 2
}