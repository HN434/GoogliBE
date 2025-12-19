"""
Simple RTMPose inference wrapper - fast and straightforward.
Optimized for speed without quality loss.
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

# Enable cuDNN benchmarking for consistent input sizes (significant speedup)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

try:
    from mmpose.apis import inference_topdown, init_model
    from mmpose.structures import PoseDataSample
except ImportError as exc:
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
        return float(self.scores.mean()) if self.scores.size else 0.0


class PoseEstimator:
    """Simple RTMPose wrapper - loads once, infers frames."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cuda",
    ):
        if not config_path.exists():
            raise FileNotFoundError(f"RTMPose config not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        logger.info(f"Loading RTMPose model (device={device})")

        # Patch torch.load for PyTorch 2.6+ compatibility
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)

        try:
            self.model = init_model(
                config=str(config_path),
                checkpoint=str(checkpoint_path),
                device=device,
            )
            self.model.eval()

            # FP16 for speed (2x faster, minimal quality loss)
            if device == "cuda" and settings.USE_HALF_PRECISION:
                self.model = self.model.half()
                logger.info("RTMPose running in FP16")

            # Compile for speed (10-20% faster)
            if device == "cuda" and hasattr(torch, "compile"):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    logger.info("RTMPose compiled with torch.compile")
                except Exception:
                    pass

            # Warm up the model with a dummy inference to optimize cuDNN
            if device == "cuda":
                try:
                    dummy_frame = np.zeros((256, 192, 3), dtype=np.uint8)
                    dummy_bbox = np.array([[0, 0, 191, 255]], dtype=np.float32)
                    with torch.inference_mode():
                        _ = inference_topdown(
                            self.model,
                            dummy_frame,
                            bboxes=dummy_bbox,
                            bbox_format="xyxy",
                        )
                    logger.debug("RTMPose warm-up completed")
                except Exception:
                    pass  # Warm-up is optional

        finally:
            torch.load = original_load

        logger.info("RTMPose loaded and optimized")

    def infer(
        self,
        frame_bgr: np.ndarray,
        person_bboxes: Optional[List[List[float]]] = None,
        auto_detect: bool = True,
    ) -> List[PoseEstimatorResult]:
        """Run pose inference on a single frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # Auto-detect persons if needed
        if not person_bboxes and auto_detect:
            try:
                from worker.inference.person_detector import get_person_detector
                detector = get_person_detector()
                person_bboxes = detector.detect(frame_bgr)
            except Exception:
                person_bboxes = None

        # Default to full frame if no bboxes
        if not person_bboxes:
            bboxes_np = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        else:
            bboxes_np = np.array(person_bboxes, dtype=np.float32)

        # Run inference
        with torch.inference_mode():
            samples: List[PoseDataSample] = inference_topdown(
                self.model,
                frame_bgr,
                bboxes=bboxes_np,
                bbox_format="xyxy",
            )

        # Convert results
        results = []
        for sample in samples:
            preds = sample.pred_instances
            if not hasattr(preds, "keypoints"):
                continue

            keypoints = preds.keypoints
            scores = getattr(preds, "keypoint_scores", None)

            # Handle batched output
            if keypoints.ndim == 3:
                keypoints = keypoints[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]

            # Convert to numpy
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            # Get bbox
            if hasattr(preds, "bboxes") and preds.bboxes is not None and len(preds.bboxes) > 0:
                bbox = preds.bboxes[0]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu().numpy().tolist()
                else:
                    bbox = bbox.tolist()
            else:
                bbox = bboxes_np[0].tolist() if len(bboxes_np) > 0 else [0, 0, 0, 0]

            results.append(
                PoseEstimatorResult(
                    keypoints=keypoints.astype(np.float32),
                    scores=scores.astype(np.float32) if scores is not None else np.zeros(keypoints.shape[0], dtype=np.float32),
                    bbox=bbox,
                )
            )

        return results

    def infer_batch(
        self,
        frames: List[np.ndarray],
        person_bboxes_list: Optional[List[Optional[List[List[float]]]]] = None,
        auto_detect: bool = True,
    ) -> List[List[PoseEstimatorResult]]:
        """
        Simple batch inference - just loops over frames and calls infer().
        No complex batching, just process each frame individually.
        """
        if not frames:
            return []

        results = []
        for i, frame in enumerate(frames):
            bboxes = person_bboxes_list[i] if person_bboxes_list and i < len(person_bboxes_list) else None
            frame_results = self.infer(frame, person_bboxes=bboxes, auto_detect=auto_detect)
            results.append(frame_results)

        return results


@lru_cache(maxsize=1)
def get_pose_estimator() -> PoseEstimator:
    """Return singleton PoseEstimator instance."""
    config_path = Path(settings.RTMPOSE_CONFIG_PATH).expanduser().resolve()
    checkpoint_path = Path(settings.RTMPOSE_CHECKPOINT_PATH).expanduser().resolve()
    device = settings.DEVICE

    return PoseEstimator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
