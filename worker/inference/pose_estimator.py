"""
Fast RTMPose-S batch inference for GPU (Tesla T4 optimized).

Key ideas:
- No manual crops
- No pipeline hacking
- Batch persons, not frames
- inference_topdown is the hot path
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
from mmpose.apis import init_model, inference_topdown
from mmpose.structures import PoseDataSample

logger = logging.getLogger(__name__)


@dataclass
class PoseEstimatorResult:
    keypoints: np.ndarray  # [K, 2]
    scores: np.ndarray     # [K]
    bbox: List[float]

    @property
    def mean_confidence(self) -> float:
        return float(self.scores.mean()) if self.scores.size else 0.0


class PoseEstimator:
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cuda",
    ):
        logger.info("Loading RTMPose-S (fast path)")
        # Store paths and device for introspection/logging
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self.model = init_model(
            config=str(self.config_path),
            checkpoint=str(self.checkpoint_path),
            device=device,
        )

        self.model.eval()

        # FP16 is safe & recommended on T4
        if device == "cuda" and settings.USE_HALF_PRECISION:
            self.model = self.model.half()
            logger.info("RTMPose running in FP16")

        # torch.compile gives ~10% on T4 if overhead is low
        if device == "cuda" and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("RTMPose compiled with torch.compile")
            except Exception:
                pass

    # ---------------------------------------------------------
    # FAST BATCH INFERENCE
    # ---------------------------------------------------------

    def infer_batch(
        self,
        frames: List[np.ndarray],
        person_bboxes_list: Optional[List[List[List[float]]]] = None,
        auto_detect: bool = True,
        max_batch_persons: int = 256,  # kept for API compatibility (unused in per-frame mode)
    ) -> List[List[PoseEstimatorResult]]:
        """
        Fast batched pose inference.

        Returns: List[frame][person]
        """

        if not frames:
            return []

        # -----------------------------------------------------
        # Step 1: detect persons (batched)
        # -----------------------------------------------------
        if person_bboxes_list is None:
            if not auto_detect:
                person_bboxes_list = [None] * len(frames)
            else:
                from worker.inference.person_detector import get_person_detector
                detector = get_person_detector()
                person_bboxes_list = detector.detect_batch(frames)

        try:
            inference_ctx = torch.inference_mode
        except AttributeError:
            inference_ctx = torch.no_grad

        output = [[] for _ in frames]

        with inference_ctx():
            # Process each frame independently; this avoids passing a list of
            # images to `inference_topdown`, which is not supported by some
            # mmpose versions and leads to `img_path` KeyError in the
            # loading pipeline.
            for frame_idx, (frame, persons) in enumerate(
                zip(frames, person_bboxes_list)
            ):
                if frame is None:
                    continue

                h, w = frame.shape[:2]

                if not persons:
                    persons = [[0, 0, w - 1, h - 1]]

                frame_bboxes = np.asarray(persons, dtype=np.float32)

                frame_results: List[Optional[PoseDataSample]] = inference_topdown(
                    self.model,
                    np.ascontiguousarray(frame),
                    bboxes=frame_bboxes,
                    bbox_format="xyxy",
                )

                for sample in frame_results:
                    if sample is None:
                        continue

                    preds = sample.pred_instances
                    if not hasattr(preds, "keypoints"):
                        continue

                    keypoints = preds.keypoints
                    scores = getattr(preds, "keypoint_scores", None)

                    if keypoints.ndim == 3:
                        keypoints = keypoints[0]
                    if scores is not None and scores.ndim == 2:
                        scores = scores[0]

                    if isinstance(keypoints, torch.Tensor):
                        keypoints = keypoints.cpu().numpy()
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()

                    if hasattr(preds, "bboxes") and preds.bboxes is not None:
                        bbox = preds.bboxes[0]
                        bbox = (
                            bbox.cpu().numpy().tolist()
                            if isinstance(bbox, torch.Tensor)
                            else bbox.tolist()
                        )
                    else:
                        bbox = [0, 0, 0, 0]

                    output[frame_idx].append(
                        PoseEstimatorResult(
                            keypoints=keypoints.astype(np.float32),
                            scores=(
                                scores.astype(np.float32)
                                if scores is not None
                                else np.zeros(keypoints.shape[0])
                            ),
                            bbox=bbox,
                        )
                    )

        return output


# ---------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_pose_estimator() -> PoseEstimator:
    return PoseEstimator(
        config_path=Path(settings.RTMPOSE_CONFIG_PATH),
        checkpoint_path=Path(settings.RTMPOSE_CHECKPOINT_PATH),
        device=settings.DEVICE,
    )
