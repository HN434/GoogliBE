"""
Person Detection using YOLO
Detects all persons in a frame and returns their bounding boxes for pose estimation
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available for person detection")


class PersonDetector:
    """
    Person detector using YOLO (regular detection, not pose).
    Detects all persons in a frame and returns bounding boxes.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Initialize person detector.
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt - will auto-download)
            conf_threshold: Confidence threshold for person detection
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is required for person detection. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO detection model"""
        try:
            logger.info(f"Loading YOLO person detector: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Enable FP16 for faster inference on T4 GPU if configured
            try:
                from config import settings
                import torch
                if settings.USE_HALF_PRECISION and torch.cuda.is_available():
                    # YOLO handles FP16 internally via the half parameter in predict calls
                    logger.info("✅ YOLO person detector loaded (FP16 will be used in inference)")
                else:
                    logger.info("✅ YOLO person detector loaded successfully")
            except Exception:
                logger.info("✅ YOLO person detector loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Detect all persons in a frame and return their bounding boxes.
        
        Args:
            frame_bgr: BGR image (numpy array) from OpenCV
            
        Returns:
            List of bounding boxes, each as numpy array [x1, y1, x2, y2]
            Returns empty list if no persons detected
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        
        try:
            # Run YOLO detection
            results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)
            
            if not results or len(results) == 0:
                return []
            
            # Extract person bounding boxes
            # YOLO class 0 is 'person'
            person_bboxes = []
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) in [x1, y1, x2, y2] format
                classes = result.boxes.cls.cpu().numpy()  # (N,)
                confidences = result.boxes.conf.cpu().numpy()  # (N,)
                
                # Filter for person class (class 0 in COCO)
                for i, cls in enumerate(classes):
                    if int(cls) == 0:  # Person class
                        bbox = boxes[i].astype(np.float32)
                        conf = confidences[i]
                        if conf >= self.conf_threshold:
                            person_bboxes.append(bbox)
            
            logger.debug(f"Detected {len(person_bboxes)} person(s) in frame")
            return person_bboxes
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}", exc_info=True)
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Detect all persons in a batch of frames using YOLO batch inference.
        This is much faster than calling detect() individually for each frame.
        
        Args:
            frames: List of BGR images (numpy arrays) from OpenCV
            
        Returns:
            List of lists of bounding boxes, one list per frame.
            Each bounding box is a numpy array [x1, y1, x2, y2]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not frames:
            return []
        
        try:
            # YOLO supports batch inference - pass list of frames
            # This processes all frames on GPU in one call - much more efficient!
            # Enable FP16 if configured for faster inference on T4 GPU
            try:
                from config import settings
                import torch
                use_half = settings.USE_HALF_PRECISION and torch.cuda.is_available()
            except Exception:
                use_half = False
            
            results = self.model(
                frames, 
                conf=self.conf_threshold, 
                verbose=False,
                half=use_half  # FP16 for faster inference
            )
            
            batch_bboxes = []
            for result in results:
                person_bboxes = []
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                    classes = result.boxes.cls.cpu().numpy()  # (N,)
                    confidences = result.boxes.conf.cpu().numpy()  # (N,)
                    
                    # Filter for person class (class 0 in COCO)
                    for i, cls in enumerate(classes):
                        if int(cls) == 0:  # Person class
                            bbox = boxes[i].astype(np.float32)
                            conf = confidences[i]
                            if conf >= self.conf_threshold:
                                person_bboxes.append(bbox)
                
                batch_bboxes.append(person_bboxes)
            
            avg_persons = sum(len(b) for b in batch_bboxes) / len(batch_bboxes) if batch_bboxes else 0
            logger.debug(f"Batch detection: {len(frames)} frames, avg {avg_persons:.1f} persons/frame")
            return batch_bboxes
            
        except Exception as e:
            logger.error(f"Error during batch person detection: {e}", exc_info=True)
            # Fallback to individual detection
            return [self.detect(frame) for frame in frames]


# Global detector instance (lazy-loaded)
_detector: Optional[PersonDetector] = None


def get_person_detector(model_path: str = "yolov8n.pt", conf_threshold: float = 0.5) -> PersonDetector:
    """
    Get or create a singleton PersonDetector instance.
    
    Args:
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold
        
    Returns:
        PersonDetector instance
    """
    global _detector
    
    if _detector is None:
        _detector = PersonDetector(model_path=model_path, conf_threshold=conf_threshold)
    
    return _detector

