import json
import pandas as pd
import tensorflow as tf
from pathlib import Path
from src.config import DATASET_DIR, LABEL_MAP, IMG_SIZE, BATCH_SIZE

def load_annotations(filename: str) -> pd.DataFrame:
    with open(DATASET_DIR / filename, "r") as f:
        annotations = json.load(f)

    data = []
    for image_path, annotation in annotations.items():
        image_path = Path(image_path)
        data.append({
            "image_path": str(DATASET_DIR / image_path.relative_to("./classification_frames")),
            "label": annotation["driver_state"],
            "landmarks": annotation["landmarks"]
        })

    df = pd.DataFrame(data)
    df["label_id"] = df["label"].map(LABEL_MAP)
    return df

def _load_and_preprocess(path: str, label: int):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
    return tf.cast(image, tf.float32), label

def create_tf_dataset(df: pd.DataFrame, is_training: bool = False):
    ds = tf.data.Dataset.from_tensor_slices((df["image_path"].values, df["label_id"].values))
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        ds = ds.shuffle(1000)
        
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)