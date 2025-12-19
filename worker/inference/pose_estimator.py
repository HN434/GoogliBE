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
        """Run pose inference on a single frame - optimized."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # Auto-detect if needed
        if not person_bboxes and auto_detect:
            try:
                from worker.inference.person_detector import get_person_detector
                detector = get_person_detector()
                person_bboxes = detector.detect(frame_bgr)
            except Exception:
                person_bboxes = None

        # Optimize bbox conversion - avoid repeated array creation
        if not person_bboxes:
            bboxes_np = np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)
        else:
            # Pre-allocate array with correct shape
            bboxes_np = np.asarray(person_bboxes, dtype=np.float32)
            if bboxes_np.ndim == 1:
                bboxes_np = bboxes_np.reshape(1, -1)

        # Infer with optimized context
        with torch.inference_mode():
            samples: List[PoseDataSample] = inference_topdown(
                self.model,
                frame_bgr,
                bboxes=bboxes_np,
                bbox_format="xyxy",
            )

        # Optimized result conversion - reduce allocations
        results = []
        for sample in samples:
            preds = sample.pred_instances
            if not hasattr(preds, "keypoints"):
                continue

            keypoints = preds.keypoints
            scores = getattr(preds, "keypoint_scores", None)

            # Handle batched output (take first if batched)
            if keypoints.ndim == 3:
                keypoints = keypoints[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]

            # Efficient tensor to numpy conversion (single CPU transfer)
            if isinstance(keypoints, torch.Tensor):
                keypoints = keypoints.detach().cpu().numpy()
            elif not isinstance(keypoints, np.ndarray):
                keypoints = np.asarray(keypoints)
                
            if scores is not None:
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                elif not isinstance(scores, np.ndarray):
                    scores = np.asarray(scores)

            # Get bbox efficiently
            if hasattr(preds, "bboxes") and preds.bboxes is not None and len(preds.bboxes) > 0:
                bbox_tensor = preds.bboxes[0]
                if isinstance(bbox_tensor, torch.Tensor):
                    bbox = bbox_tensor.detach().cpu().numpy().tolist()
                else:
                    bbox = bbox_tensor.tolist() if hasattr(bbox_tensor, 'tolist') else list(bbox_tensor)
            else:
                bbox = bboxes_np[0].tolist() if len(bboxes_np) > 0 else [0, 0, 0, 0]

            # Ensure correct dtype without extra copy if possible
            keypoints_f32 = keypoints.astype(np.float32, copy=False)
            scores_f32 = (
                scores.astype(np.float32, copy=False) 
                if scores is not None 
                else np.zeros(keypoints.shape[0], dtype=np.float32)
            )

            results.append(
                PoseEstimatorResult(
                    keypoints=keypoints_f32,
                    scores=scores_f32,
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
        Optimized batch inference - minimizes Python overhead.
        
        Processes frames efficiently by:
        1. Single inference_mode context
        2. Cached method lookups
        3. Pre-allocated arrays
        4. Minimal function calls
        """
        if not frames:
            return []

        num_frames = len(frames)
        output = [[] for _ in range(num_frames)]
        
        # Cache detector (reduces lookup overhead)
        detector = None
        if auto_detect:
            try:
                from worker.inference.person_detector import get_person_detector
                detector = get_person_detector()
            except Exception:
                pass
        
        # Cache method lookups for speed (reduces Python attribute lookup overhead)
        hasattr_func = hasattr
        isinstance_func = isinstance
        detach_cpu_numpy = lambda t: t.detach().cpu().numpy()
        
        # Process all frames in one inference context
        with torch.inference_mode():
            for frame_idx in range(num_frames):
                frame = frames[frame_idx]
                if frame is None or frame.size == 0:
                    continue
                
                h, w = frame.shape[:2]
                w1, h1 = w - 1, h - 1
                
                # Get bboxes efficiently
                if person_bboxes_list and frame_idx < len(person_bboxes_list):
                    bboxes = person_bboxes_list[frame_idx]
                elif auto_detect and detector:
                    bboxes = detector.detect(frame)
                else:
                    bboxes = None
                
                # Fast bbox conversion
                if not bboxes:
                    bboxes_np = np.array([[0, 0, w1, h1]], dtype=np.float32)
                else:
                    bboxes_np = np.asarray(bboxes, dtype=np.float32)
                    if bboxes_np.ndim == 1:
                        bboxes_np = bboxes_np.reshape(1, -1)
                
                # Frame already contiguous from video processor, but ensure it
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)
                
                # Single GPU call
                samples = inference_topdown(
                    self.model,
                    frame,
                    bboxes=bboxes_np,
                    bbox_format="xyxy",
                )
                
                # Fast result conversion - inline everything
                frame_results = output[frame_idx]
                frame_results_append = frame_results.append
                
                for sample in samples:
                    preds = sample.pred_instances
                    if not hasattr_func(preds, "keypoints"):
                        continue
                    
                    keypoints = preds.keypoints
                    scores = getattr(preds, "keypoint_scores", None) if hasattr_func(preds, "keypoint_scores") else None
                    
                    # Handle dimensions
                    if keypoints.ndim == 3:
                        keypoints = keypoints[0]
                    if scores is not None and scores.ndim == 2:
                        scores = scores[0]
                    
                    # Fast tensor conversion (single CPU transfer)
                    if isinstance_func(keypoints, torch.Tensor):
                        keypoints = detach_cpu_numpy(keypoints)
                    elif not isinstance_func(keypoints, np.ndarray):
                        keypoints = np.asarray(keypoints, dtype=np.float32)
                    
                    if scores is not None:
                        if isinstance_func(scores, torch.Tensor):
                            scores = detach_cpu_numpy(scores)
                        elif not isinstance_func(scores, np.ndarray):
                            scores = np.asarray(scores, dtype=np.float32)
                    
                    # Fast bbox extraction
                    if hasattr_func(preds, "bboxes") and preds.bboxes is not None and len(preds.bboxes) > 0:
                        bbox_t = preds.bboxes[0]
                        if isinstance_func(bbox_t, torch.Tensor):
                            bbox = detach_cpu_numpy(bbox_t).tolist()
                        else:
                            bbox = bbox_t.tolist() if hasattr_func(bbox_t, 'tolist') else list(bbox_t)
                    else:
                        bbox = bboxes_np[0].tolist() if len(bboxes_np) > 0 else [0, 0, 0, 0]
                    
                    # Create result - use copy=False when dtype matches
                    kpts_f32 = keypoints.astype(np.float32, copy=False)
                    scores_f32 = scores.astype(np.float32, copy=False) if scores is not None else np.zeros(keypoints.shape[0], dtype=np.float32)
                    
                    frame_results_append(
                        PoseEstimatorResult(
                            keypoints=kpts_f32,
                            scores=scores_f32,
                            bbox=bbox,
                        )
                    )
        
        return output


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
