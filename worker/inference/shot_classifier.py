"""
Cricket shot classification inference (EfficientNetB0 + GRU).

This module mirrors the standalone test script you provided, but is structured
for reuse inside the worker process:

- The Keras model is built and weights are loaded ONCE per worker process.
- Subsequent calls reuse the same model instance to avoid re-loading weights.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

# Force TensorFlow to use CPU only for shot classification to avoid GPU contention
# Don't set CUDA_VISIBLE_DEVICES globally - it affects PyTorch models too!
# Instead, configure TensorFlow directly to not use GPU
import tensorflow as tf

# Configure TensorFlow to use CPU only (without affecting PyTorch)
try:
    # Hide GPUs from TensorFlow
    tf.config.set_visible_devices([], "GPU")
    logger = __import__('logging').getLogger(__name__)
    logger.debug("TensorFlow configured to use CPU only (PyTorch GPU access preserved)")
except Exception:
    pass

import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from config import settings

# Class mapping used in the training / Streamlit app
CLASSES: Dict[str, int] = {
    "cover": 0,
    "defense": 1,
    "flick": 2,
    "hook": 3,
    "late_cut": 4,
    "lofted": 5,
    "pull": 6,
    "square_cut": 7,
    "straight": 8,
    "sweep": 9,
}

_shot_classifier_model: tf.keras.Model | None = None


def _build_model(weights_path: str) -> tf.keras.Model:
    """Rebuild the EfficientNetB0 + GRU architecture and load trained weights."""
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True),
            layers.GRU(128),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.load_weights(weights_path)
    return model


def get_shot_classifier() -> tf.keras.Model:
    """
    Get or create the singleton shot classification model.

    The model is kept in memory for the lifetime of the worker process so that
    we don't repeatedly pay the cost of loading EfficientNet/GRU weights.
    """
    global _shot_classifier_model

    if _shot_classifier_model is not None:
        return _shot_classifier_model

    weights_path = os.getenv(
        "SHOT_CLASSIFIER_WEIGHTS_PATH",
        getattr(settings, "SHOT_CLASSIFIER_WEIGHTS_PATH", "model_weights.h5"),
    )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Shot classifier weights not found at '{weights_path}'. "
            "Set SHOT_CLASSIFIER_WEIGHTS_PATH to the correct .h5 file."
        )

    _shot_classifier_model = _build_model(weights_path)
    return _shot_classifier_model


def _format_frame(frame: np.ndarray, output_size) -> np.ndarray:
    """Pad and resize an image from a video to EfficientNet input size."""
    # Convert BGR (OpenCV) to RGB first, then to tf.uint8
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tf = tf.image.convert_image_dtype(frame_rgb, tf.uint8)
    frame_tf = tf.image.resize_with_pad(frame_tf, *output_size)
    return frame_tf.numpy()


def _frames_from_video_file(
    video_path: str,
    n_frames: int,
    output_size=(224, 224),
    frame_step: int = 1,
) -> np.ndarray:
    """
    Extract n_frames frames from a video, with a fixed step.

    This matches the logic in your original test script: we always take the
    first readable frame, then stride forward by `frame_step` between samples.
    Missing frames are padded with zero images.
    """
    result = []
    src = cv2.VideoCapture(str(video_path))
    src.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # First frame
    ret, frame = src.read()
    if ret:
        frame = _format_frame(frame, output_size)
        result.append(frame)
    else:
        # If the first frame can't be read, append a zero frame and exit
        result.append(np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8))

    # Subsequent frames
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()

        if ret:
            frame = _format_frame(frame, output_size)
            result.append(frame)
        else:
            # Append a zero-like frame if no more frames can be read
            result.append(np.zeros_like(result[0]))

    src.release()

    # Result is already RGB
    return np.array(result)


def classify_video(
    video_path: str,
    model: tf.keras.Model,
    frame_count: int,
    class_labels: Dict[str, int] | None = None,
) -> Tuple[str, float, np.ndarray]:
    """
    Classify a single video into one of the cricket shot classes.

    Returns:
        class_name: Predicted class label (e.g. "cover")
        confidence: Confidence in percent (0-100)
        probs: Raw probability vector over classes (np.ndarray of shape [num_classes])
    """
    if class_labels is None:
        class_labels = CLASSES

    frames = _frames_from_video_file(video_path, frame_count)
    frames = np.expand_dims(frames, axis=0)  # [1, T, H, W, C]

    predictions = model.predict(frames)
    predicted_class_idx = int(np.argmax(predictions, axis=1)[0])

    # Reverse lookup on the label dict
    # (keys are labels, values are indices)
    label_to_idx = class_labels
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_name = idx_to_label.get(predicted_class_idx, f"class_{predicted_class_idx}")

    confidence = float(predictions[0][predicted_class_idx]) * 100.0
    probs = predictions[0]

    return class_name, confidence, probs



