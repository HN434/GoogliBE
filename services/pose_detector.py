"""
Pose Detection Service using YOLOv8-pose with MediaPipe fallback
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Ultralytics YOLO not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available")

from config import settings
from models.schemas import Keypoint, BoundingBox, PoseDetection


# COCO keypoint names for YOLOv8-pose (17 keypoints)
YOLO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# MediaPipe Pose landmark names (33 keypoints - we'll map to COCO format)
MEDIAPIPE_TO_COCO_MAPPING = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class PoseDetector:
    """
    Pose detection service with YOLOv8-pose primary and MediaPipe fallback
    """

    def __init__(self):
        """Initialize pose detector with configured model"""
        self.device = self._get_device()
        self.model = None
        self.mediapipe_pose = None
        self.use_mediapipe = False

        print(f"🔧 Initializing PoseDetector on device: {self.device}")

        # Try to load YOLOv8-pose first
        if YOLO_AVAILABLE and not settings.USE_MEDIAPIPE_FALLBACK:
            self._load_yolo_model()

        # Load MediaPipe as fallback
        if not self.model and MEDIAPIPE_AVAILABLE:
            self._load_mediapipe_model()

        if not self.model and not self.mediapipe_pose:
            raise RuntimeError(
                "No pose detection model available. Install ultralytics or mediapipe."
            )

    def _get_device(self) -> str:
        """Determine the best available device"""
        if not settings.USE_GPU:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_yolo_model(self):
        """Load YOLOv8-pose model"""
        try:
            model_path = settings.POSE_MODEL
            print(f"📦 Loading YOLOv8-pose model: {model_path}")

            self.model = YOLO(model_path)
            self.model.to(self.device)

            print(f"✅ YOLOv8-pose loaded successfully on {self.device}")
            self.use_mediapipe = False

        except Exception as e:
            print(f"❌ Failed to load YOLOv8-pose: {e}")
            self.model = None

    def _load_mediapipe_model(self):
        """Load MediaPipe Pose as fallback"""
        try:
            print("📦 Loading MediaPipe Pose as fallback...")

            mp_pose = mp.solutions.pose
            self.mediapipe_pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=settings.MEDIAPIPE_MODEL_COMPLEXITY,
                min_detection_confidence=settings.POSE_CONFIDENCE_THRESHOLD,
                min_tracking_confidence=0.5
            )

            print("✅ MediaPipe Pose loaded successfully")
            self.use_mediapipe = True

        except Exception as e:
            print(f"❌ Failed to load MediaPipe: {e}")
            self.mediapipe_pose = None

    def detect_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[List[PoseDetection]]:
        """
        Detect poses in a batch of frames

        Args:
            frames: List of image frames (numpy arrays)

        Returns:
            List of detections per frame
        """
        if self.use_mediapipe:
            return self._detect_batch_mediapipe(frames)
        else:
            return self._detect_batch_yolo(frames)

    def _detect_batch_yolo(
        self,
        frames: List[np.ndarray]
    ) -> List[List[PoseDetection]]:
        """Batch detection using YOLOv8-pose"""
        results = []

        try:
            # Run batch inference
            predictions = self.model.predict(
                frames,
                conf=settings.POSE_CONFIDENCE_THRESHOLD,
                iou=settings.POSE_IOU_THRESHOLD,
                verbose=False,
                device=self.device
            )

            for pred in predictions:
                frame_detections = []

                if pred.keypoints is None or len(pred.keypoints) == 0:
                    results.append(frame_detections)
                    continue

                # Extract keypoints and boxes
                keypoints = pred.keypoints.xy.cpu().numpy()  # (N, 17, 2)
                keypoint_conf = pred.keypoints.conf.cpu().numpy()  # (N, 17)
                boxes = pred.boxes.xyxy.cpu().numpy()  # (N, 4)
                box_conf = pred.boxes.conf.cpu().numpy()  # (N,)

                for i in range(len(boxes)):
                    # Create keypoint list
                    kpts = []
                    for j, name in enumerate(YOLO_KEYPOINT_NAMES):
                        kpts.append(Keypoint(
                            name=name,
                            x=float(keypoints[i, j, 0]),
                            y=float(keypoints[i, j, 1]),
                            confidence=float(keypoint_conf[i, j])
                        ))

                    # Create bounding box
                    x1, y1, x2, y2 = boxes[i]
                    bbox = BoundingBox(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1)
                    )

                    # Create detection
                    detection = PoseDetection(
                        bbox=bbox,
                        keypoints=kpts,
                        confidence=float(box_conf[i])
                    )

                    frame_detections.append(detection)

                results.append(frame_detections)

        except Exception as e:
            print(f"⚠️  Error in YOLOv8 batch detection: {e}")
            results = [[] for _ in frames]

        return results

    def _detect_batch_mediapipe(
        self,
        frames: List[np.ndarray]
    ) -> List[List[PoseDetection]]:
        """Batch detection using MediaPipe (processes sequentially)"""
        results = []

        for frame in frames:
            frame_detections = []

            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                result = self.mediapipe_pose.process(frame_rgb)

                if result.pose_landmarks:
                    # Extract landmarks
                    landmarks = result.pose_landmarks.landmark
                    h, w = frame.shape[:2]

                    # Map MediaPipe landmarks to COCO format
                    kpts = []
                    xs, ys = [], []

                    for mp_idx, name in MEDIAPIPE_TO_COCO_MAPPING.items():
                        lm = landmarks[mp_idx]
                        x = lm.x * w
                        y = lm.y * h
                        xs.append(x)
                        ys.append(y)

                        kpts.append(Keypoint(
                            name=name,
                            x=float(x),
                            y=float(y),
                            confidence=float(lm.visibility)
                        ))

                    # Create bounding box from keypoints
                    if xs and ys:
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        # Add some padding
                        padding = 0.1
                        width = x_max - x_min
                        height = y_max - y_min
                        x_min = max(0, x_min - padding * width)
                        y_min = max(0, y_min - padding * height)
                        width = width * (1 + 2 * padding)
                        height = height * (1 + 2 * padding)

                        bbox = BoundingBox(
                            x=int(x_min),
                            y=int(y_min),
                            width=int(width),
                            height=int(height)
                        )

                        # Estimate overall confidence
                        avg_conf = np.mean([kp.confidence for kp in kpts])

                        detection = PoseDetection(
                            bbox=bbox,
                            keypoints=kpts,
                            confidence=float(avg_conf)
                        )

                        frame_detections.append(detection)

            except Exception as e:
                print(f"⚠️  Error in MediaPipe detection: {e}")

            results.append(frame_detections)

        return results

    def detect_single(self, frame: np.ndarray) -> List[PoseDetection]:
        """
        Detect poses in a single frame

        Args:
            frame: Image frame (numpy array)

        Returns:
            List of detections in the frame
        """
        return self.detect_batch([frame])[0]

    def cleanup(self):
        """Clean up resources"""
        if self.mediapipe_pose:
            self.mediapipe_pose.close()

        if self.model and settings.CLEAR_GPU_CACHE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        """Destructor to clean up resources"""
        self.cleanup()


# Singleton instance
_detector_instance = None


def get_pose_detector() -> PoseDetector:
    """Get or create singleton pose detector instance"""
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = PoseDetector()

    return _detector_instance
