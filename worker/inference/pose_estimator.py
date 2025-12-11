"""
RTMPose inference utilities for worker processes.

Loads RTMPose once and exposes a simple interface for frame-level inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from config import settings

logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6+ weights_only=True default
# Allow numpy objects in checkpoint loading (RTMPose checkpoints contain numpy arrays)
try:
    import numpy.core.multiarray as numpy_multiarray
    torch.serialization.add_safe_globals([
        numpy_multiarray._reconstruct,
        numpy_multiarray.scalar,
        np.ndarray,
        np.dtype,
    ])
    logger.debug("Added numpy safe globals for PyTorch checkpoint loading")
except (AttributeError, ImportError):
    # Fallback: if numpy structure is different, just add ndarray
    try:
        torch.serialization.add_safe_globals([np.ndarray, np.dtype])
        logger.debug("Added basic numpy safe globals for PyTorch checkpoint loading")
    except Exception as e:
        logger.warning(f"Could not add numpy safe globals: {e}")

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import PoseDataSample
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "mmpose, mmengine, and mmcv must be installed to use RTMPose inference."
    ) from exc


@dataclass
class PoseEstimatorResult:
    """Container for a single pose estimation output."""

    keypoints: np.ndarray  # Shape: [N, 2]
    scores: np.ndarray  # Shape: [N]
    bbox: List[float]

    @property
    def mean_confidence(self) -> float:
        if self.scores.size == 0:
            return 0.0
        return float(np.mean(self.scores))


class PoseEstimator:
    """
    Thin wrapper around RTMPose.

    Loads the model once and exposes an `infer` method that accepts a single frame.
    """

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cpu",
    ):
        if not config_path.exists():
            raise FileNotFoundError(f"RTMPose config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        logger.info(
            "Loading RTMPose model\n  config: %s\n  checkpoint: %s\n  device: %s",
            config_path,
            checkpoint_path,
            device,
        )
        
        # Monkey-patch torch.load to use weights_only=False for PyTorch 2.6+
        # This is safe for trusted RTMPose checkpoints
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            """Patched torch.load that forces weights_only=False for trusted checkpoints."""
            # Always use weights_only=False for checkpoint loading
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_torch_load
        
        try:
            self.model = init_model(
                config=str(config_path),
                checkpoint=str(checkpoint_path),
                device=device,
            )
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
        
        logger.info("RTMPose model successfully loaded")

    def infer(
        self,
        frame_bgr: np.ndarray,
        person_bboxes: Optional[List[List[float]]] = None,
        auto_detect: bool = True,
    ) -> List[PoseEstimatorResult]:
        """
        Run pose inference on a single frame.

        Args:
            frame_bgr: Frame in BGR format (OpenCV default).
            person_bboxes: Optional list of bounding boxes [x1, y1, x2, y2].
            auto_detect: If True and person_bboxes is None, auto-detect persons using YOLO.

        Returns:
            List of PoseEstimatorResult for each detected person.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame is empty or None.")

        height, width = frame_bgr.shape[:2]

        # Auto-detect persons if bboxes not provided
        if not person_bboxes and auto_detect:
            try:
                from worker.inference.person_detector import get_person_detector
                detector = get_person_detector()
                detected_bboxes = detector.detect(frame_bgr)
                if detected_bboxes:
                    person_bboxes = detected_bboxes
                    logger.debug(f"Auto-detected {len(person_bboxes)} person(s)")
            except Exception as e:
                logger.warning(f"Person detection failed, using full frame: {e}")
                person_bboxes = None

        # Default to a single bbox covering the whole frame if no detections
        # Format: [x1, y1, x2, y2] as numpy array with shape (4,)
        if not person_bboxes:
            person_bboxes = [np.array([0, 0, width - 1, height - 1], dtype=np.float32)]

        # Convert bboxes to numpy arrays if they're lists, ensure shape is (4,)
        # inference_topdown expects bboxes as numpy array with shape (N, 4) where N is number of persons
        bboxes_array = []
        for bbox in person_bboxes:
            if isinstance(bbox, list):
                bbox = np.array(bbox, dtype=np.float32)
            elif not isinstance(bbox, np.ndarray):
                bbox = np.array(bbox, dtype=np.float32)
            
            # Ensure bbox has shape (4,) - [x1, y1, x2, y2]
            if bbox.shape != (4,):
                raise ValueError(f"Bbox must have shape (4,), got {bbox.shape}")
            
            bboxes_array.append(bbox)
        
        # Stack into shape (N, 4) for inference_topdown
        bboxes_np = np.array(bboxes_array, dtype=np.float32) if bboxes_array else np.array([[0, 0, width - 1, height - 1]], dtype=np.float32)

        # Call inference_topdown - it expects bboxes as numpy array with shape (N, 4)
        # bbox_format='xyxy' means [x1, y1, x2, y2] format
        pose_data_samples: List[PoseDataSample] = inference_topdown(
            self.model,
            frame_bgr,
            bboxes=bboxes_np,
            bbox_format='xyxy',
        )

        results: List[PoseEstimatorResult] = []
        for sample in pose_data_samples:
            preds = sample.pred_instances
            if not hasattr(preds, "keypoints"):
                continue
            keypoints = preds.keypoints
            scores = preds.keypoint_scores if hasattr(preds, "keypoint_scores") else None

            # Some configs return batched predictions. We only handle the first instance per sample.
            if keypoints.ndim == 3:
                keypoints = keypoints[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]

            # Get bbox from predictions or use the input bbox
            if hasattr(preds, "bboxes") and preds.bboxes is not None and len(preds.bboxes) > 0:
                bbox = preds.bboxes[0].tolist()
            else:
                # Fallback: use the first input bbox
                bbox = bboxes_np[0].tolist() if len(bboxes_np) > 0 else [0, 0, 0, 0]

            results.append(
                PoseEstimatorResult(
                    keypoints=np.asarray(keypoints, dtype=np.float32),
                    scores=np.asarray(scores, dtype=np.float32)
                    if scores is not None
                    else np.zeros(keypoints.shape[0], dtype=np.float32),
                    bbox=bbox,
                )
            )

        return results


@lru_cache(maxsize=1)
def get_pose_estimator() -> PoseEstimator:
    """
    Return a singleton PoseEstimator instance.

    Loads RTMPose config + checkpoint paths from settings and caches the instance.
    """
    config_path = Path(settings.RTMPOSE_CONFIG_PATH).expanduser().resolve()
    checkpoint_path = Path(settings.RTMPOSE_CHECKPOINT_PATH).expanduser().resolve()
    device = settings.RTMPOSE_DEVICE or "cpu"

    estimator = PoseEstimator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    return estimator

