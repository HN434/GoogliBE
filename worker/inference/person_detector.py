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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class PersonDetector:
    """
    Person detector using YOLO (regular detection, not pose).
    Detects all persons in a frame and returns their bounding boxes.
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
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> str:
        """
        Determine the best available device with proper validation.
        Validates CUDA devices to avoid 'Invalid device id' errors.
        Also handles CUDA_VISIBLE_DEVICES environment variable issues.
        """
        try:
            from config import settings
        except ImportError:
            # If config is not available, default to CPU
            logger.warning("Config not available, defaulting to CPU")
            return "cpu"
        
        # If GPU is disabled, use CPU
        if not settings.USE_GPU:
            return "cpu"
        
        # Check CUDA_VISIBLE_DEVICES environment variable
        # This can cause device ID mismatches if not handled properly
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible:
            logger.debug(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible}")
        
        # Check if a specific device is configured
        if hasattr(settings, 'DEVICE') and settings.DEVICE:
            requested_device = settings.DEVICE.lower()
            
            # Validate CUDA device
            if requested_device == "cuda" or requested_device.startswith("cuda:"):
                if not TORCH_AVAILABLE or not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return "cpu"
                
                # Get actual device count (may be affected by CUDA_VISIBLE_DEVICES)
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    logger.warning("No CUDA devices available, falling back to CPU")
                    return "cpu"
                
                # If specific CUDA device ID is provided (e.g., "cuda:0", "cuda:1")
                if ":" in requested_device:
                    try:
                        device_id = int(requested_device.split(":")[1])
                        # Validate device ID exists
                        if device_id >= device_count:
                            logger.warning(
                                f"CUDA device {device_id} not available (only {device_count} devices), "
                                "falling back to CPU"
                            )
                            return "cpu"
                        # Device ID is within valid range, return it
                        # Don't validate with get_device_properties as it can cause the same error
                        # YOLO will handle device selection when we call model.to(device)
                        return requested_device
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid CUDA device format: {requested_device}, falling back to CPU")
                        return "cpu"
                else:
                    # Just "cuda" - don't validate, just return it
                    # torch.cuda.is_available() and device_count > 0 are sufficient
                    # Validating with get_device_properties can cause the same error we're trying to avoid
                    return "cuda"
            
            # Validate MPS device (for Mac)
            elif requested_device == "mps":
                if not TORCH_AVAILABLE or not torch.backends.mps.is_available():
                    logger.warning("MPS requested but not available, falling back to CPU")
                    return "cpu"
                return "mps"
            
            # CPU is always valid
            elif requested_device == "cpu":
                return "cpu"
        
        # Auto-detect device if not specified
        # Use simple checks - don't validate with get_device_properties as it can cause errors
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # CUDA is available and devices exist - return "cuda"
                    # Don't validate with get_device_properties - it can trigger the same error
                    return "cuda"
                else:
                    logger.warning("CUDA is available but no devices found, falling back to CPU")
                    return "cpu"
            elif torch.backends.mps.is_available():
                return "mps"
        
        # Default to CPU
        return "cpu"
    
    def _load_model(self):
        """Load YOLO detection model with proper device configuration"""
        try:
            logger.info(f"Loading YOLO person detector: {self.model_path} on device: {self.device}")
            
            # Initialize YOLO with explicit device to avoid auto-detection issues
            # This prevents the "Invalid device id" error during initialization
            self.model = YOLO(self.model_path)
            
            # Explicitly set device after initialization
            # This ensures the model uses the validated device
            try:
                self.model.to(self.device)
            except Exception as device_error:
                logger.warning(
                    f"Failed to set device to {self.device}: {device_error}. "
                    "Falling back to CPU"
                )
                self.device = "cpu"
                self.model.to("cpu")
            
            # Verify model is on the correct device
            try:
                # YOLO model device can be checked via model.device property
                if hasattr(self.model, 'device'):
                    actual_device = str(self.model.device)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'device'):
                    actual_device = str(self.model.model.device)
                else:
                    actual_device = self.device  # Fallback to requested device
                
                logger.info(f"YOLO person detector device: {actual_device} (requested: {self.device})")
                
                if self.device.startswith("cuda") and not actual_device.startswith("cuda"):
                    logger.warning(f"⚠️ YOLO requested GPU ({self.device}) but model is on {actual_device}")
                elif self.device.startswith("cuda") and actual_device.startswith("cuda"):
                    logger.info(f"✅ YOLO confirmed on GPU: {actual_device}")
            except Exception as e:
                logger.debug(f"Could not verify YOLO device: {e}")
            
            # Enable FP16 for faster inference on T4 GPU if configured
            try:
                from config import settings
                if settings.USE_HALF_PRECISION and self.device.startswith("cuda"):
                    # YOLO handles FP16 internally via the half parameter in predict calls
                    logger.info("✅ YOLO person detector loaded (FP16 will be used in inference)")
                else:
                    logger.info(f"✅ YOLO person detector loaded successfully on {self.device}")
            except Exception:
                logger.info(f"✅ YOLO person detector loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            # Try fallback to CPU if CUDA fails
            if self.device != "cpu":
                logger.info("Attempting to load model on CPU as fallback...")
                try:
                    self.device = "cpu"
                    self.model = YOLO(self.model_path)
                    self.model.to("cpu")
                    logger.info("✅ YOLO person detector loaded successfully on CPU (fallback)")
                except Exception as fallback_error:
                    logger.error(f"Failed to load YOLO model even on CPU: {fallback_error}")
                    raise
            else:
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
            # Run YOLO detection with explicit device to avoid auto-detection issues
            results = self.model(
                frame_bgr, 
                conf=self.conf_threshold, 
                verbose=False,
                device=self.device  # Explicitly set device to avoid auto-detection
            )
            
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
                device=self.device,  # Explicitly set device to avoid auto-detection
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

