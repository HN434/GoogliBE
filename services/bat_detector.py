"""
Bat Detection Service using RT-DETR (Real-Time Detection Transformer)

This service detects cricket bats in video frames using a fine-tuned RT-DETR model.
RT-DETR provides accurate object detection with real-time performance.
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("WARNING: RF-DETR not available. Install: pip install rfdetr")

from config import settings
from models.schemas import BoundingBox


class BatDetection:
    """Represents a detected cricket bat in a frame"""
    
    def __init__(
        self,
        bbox: BoundingBox,
        confidence: float,
        bat_angle: Optional[float] = None,
        bat_center: Optional[Tuple[float, float]] = None,
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.bat_angle = bat_angle  # Angle in degrees
        self.bat_center = bat_center  # (x, y) center point
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
            },
            "confidence": float(self.confidence),
            "bat_angle": float(self.bat_angle) if self.bat_angle is not None else None,
            "bat_center": list(self.bat_center) if self.bat_center else None,
        }


class BatDetector:
    """
    Cricket bat detection using RT-DETR model
    
    RT-DETR (Real-Time Detection Transformer) provides:
    - High accuracy object detection
    - Real-time performance
    - Better than traditional YOLO for small objects
    """

    def __init__(self):
        """Initialize bat detector with RT-DETR model"""
        self.device = self._get_device()
        self.model = None
        self.enabled = settings.BAT_DETECTION_ENABLED
        
        if not self.enabled:
            print("🏏 Bat detection is DISABLED in config")
            return

        print(f"Initializing BatDetector on device: {self.device}")

        if not RFDETR_AVAILABLE:
            print("ERROR: RF-DETR not available. Install: pip install rfdetr")
            self.enabled = False
            return

        self._load_model()

    def _get_device(self) -> str:
        """Determine the best available device"""
        # Check what device was requested
        requested_device = settings.RTDETR_DEVICE
        
        # Validate the requested device is actually available
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, falling back to CPU")
                return "cpu"
            return "cuda"
        elif requested_device == "mps":
            if not torch.backends.mps.is_available():
                print("WARNING: MPS requested but not available, falling back to CPU")
                return "cpu"
            return "mps"
        elif requested_device == "cpu":
            return "cpu"
        
        # Auto-detect if no device specified
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Load RF-DETR model from checkpoint"""
        try:
            model_path = Path(settings.RTDETR_MODEL_PATH)
            
            if not model_path.exists():
                print(f"WARNING: RF-DETR model not found at: {model_path}")
                print(f"   Please update RTDETR_MODEL_PATH in config/environment")
                print(f"   Bat detection will be disabled until model is available")
                self.enabled = False
                return

            print(f"Loading RF-DETR model from: {model_path}")
            
            # Load RF-DETR model - it handles .pth files natively
            self.model = RFDETRBase(pretrain_weights=str(model_path))
            
            # Move to device
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
            
            # Set to eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Check and log model classes
            if hasattr(self.model, 'class_names'):
                print(f"   - Model classes: {self.model.class_names}")
            
            print(f"SUCCESS: RF-DETR bat detector loaded successfully")
            print(f"   - Device: {self.device}")
            print(f"   - Confidence threshold: {settings.BAT_CONFIDENCE_THRESHOLD}")
            print(f"   - Bat size range: {settings.BAT_MIN_WIDTH}-{settings.BAT_MAX_WIDTH}x{settings.BAT_MIN_HEIGHT}-{settings.BAT_MAX_HEIGHT} pixels")
            
        except Exception as e:
            print(f"ERROR: Failed to load RF-DETR model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.enabled = False

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        return_low_confidence: bool = False,
        low_confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> List[BatDetection]:
        """
        Detect cricket bats in a frame
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Override default confidence threshold
            return_low_confidence: If True, returns tuple (high_conf, low_conf) detections
            low_confidence_threshold: Minimum confidence for low-confidence detections
            
        Returns:
            List of BatDetection objects, or tuple of (high_conf, low_conf) if return_low_confidence=True
        """
        if not self.enabled or self.model is None:
            return ([], []) if return_low_confidence else []

        high_conf_threshold = confidence_threshold or settings.BAT_CONFIDENCE_THRESHOLD
        
        try:
            # Convert numpy array to PIL Image for RF-DETR
            from PIL import Image
            if isinstance(frame, np.ndarray):
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
            else:
                pil_image = frame
            
            # Run RF-DETR inference with lower threshold to catch all detections
            inference_threshold = low_confidence_threshold if return_low_confidence else high_conf_threshold
            result = self.model.predict(pil_image, threshold=inference_threshold)
            
            high_conf_detections = []
            low_conf_detections = []
            
            if result is not None and hasattr(result, 'xyxy') and len(result.xyxy) > 0:
                boxes = result.xyxy  # x1, y1, x2, y2
                confidences = result.confidence
                
                if debug:
                    print(f"[BAT DEBUG] Raw detections: {len(boxes)}")
                
                # Filter based on size (bats are smaller than people)
                filtered_reasons = []
                for i, (box, conf_val) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    aspect_ratio = box_height / (box_width + 1e-6)
                    
                    # Size-based filtering: skip if bbox is outside bat size range
                    # Bats in cricket videos are much smaller than people
                    if (box_height > settings.BAT_MAX_HEIGHT or 
                        box_width > settings.BAT_MAX_WIDTH or
                        box_height < settings.BAT_MIN_HEIGHT or 
                        box_width < settings.BAT_MIN_WIDTH):
                        filtered_reasons.append(f"Det {i}: {box_width:.0f}x{box_height:.0f} (size)")
                        continue
                    
                    # Skip if aspect ratio suggests it's a person (height >> width)
                    # Bats can be vertical (AR ~4-5) but people are even taller (AR > 6)
                    if aspect_ratio > 6.0:  # Very tall = likely person
                        filtered_reasons.append(f"Det {i}: AR={aspect_ratio:.1f} (person)")
                        continue
                    
                    # Create BoundingBox (convert from x1,y1,x2,y2 to x,y,width,height)
                    bbox = BoundingBox(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                    )
                    
                    # Calculate bat center and angle
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bat_center = (float(center_x), float(center_y))
                    
                    # Estimate bat angle from bbox orientation
                    width = x2 - x1
                    height = y2 - y1
                    bat_angle = np.degrees(np.arctan2(height, width))
                    
                    detection = BatDetection(
                        bbox=bbox,
                        confidence=float(conf_val),
                        bat_angle=float(bat_angle),
                        bat_center=bat_center,
                    )
                    
                    # Separate high and low confidence detections
                    if float(conf_val) >= high_conf_threshold:
                        high_conf_detections.append(detection)
                        if debug:
                            print(f"[BAT DEBUG]   Det {i}: {box_width:.0f}x{box_height:.0f}, AR={aspect_ratio:.1f}, conf={conf_val:.3f} -> HIGH CONF")
                    else:
                        low_conf_detections.append(detection)
                        if debug:
                            print(f"[BAT DEBUG]   Det {i}: {box_width:.0f}x{box_height:.0f}, AR={aspect_ratio:.1f}, conf={conf_val:.3f} -> LOW CONF")
                
                if debug and filtered_reasons:
                    print(f"[BAT DEBUG] Filtered {len(filtered_reasons)} detections:")
                    for reason in filtered_reasons[:5]:  # Show first 5
                        print(f"[BAT DEBUG]   {reason}")
                
                if debug:
                    print(f"[BAT DEBUG] Result: {len(high_conf_detections)} high, {len(low_conf_detections)} low confidence")
            
            if return_low_confidence:
                return (high_conf_detections, low_conf_detections)
            else:
                return high_conf_detections
            
        except Exception as e:
            print(f"ERROR: Bat detection failed: {e}")
            return []

    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> List[List[BatDetection]]:
        """
        Detect bats in a batch of frames (more efficient)
        
        Args:
            frames: List of input frames
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of lists of BatDetection objects (one list per frame)
        """
        if not self.enabled or self.model is None:
            return [[] for _ in frames]

        conf = confidence_threshold or settings.BAT_CONFIDENCE_THRESHOLD
        
        try:
            # Process each frame individually (RF-DETR batch processing handled internally)
            from PIL import Image
            all_detections = []
            
            for frame in frames:
                frame_detections = []
                
                # Convert numpy array to PIL Image
                if isinstance(frame, np.ndarray):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                else:
                    pil_image = frame
                
                # Run RF-DETR inference
                result = self.model.predict(pil_image, threshold=conf)
                
                if result is not None and hasattr(result, 'xyxy') and len(result.xyxy) > 0:
                    boxes = result.xyxy
                    confidences = result.confidence
                    
                    # Filter based on size (bats are smaller than people)
                    for box, conf in zip(boxes, confidences):
                        x1, y1, x2, y2 = box
                        box_width = x2 - x1
                        box_height = y2 - y1
                        
                        # Size-based filtering: skip if bbox is outside bat size range
                        if (box_height > settings.BAT_MAX_HEIGHT or 
                            box_width > settings.BAT_MAX_WIDTH or
                            box_height < settings.BAT_MIN_HEIGHT or 
                            box_width < settings.BAT_MIN_WIDTH):
                            continue
                        
                        # Skip if aspect ratio suggests it's a person (height >> width)
                        aspect_ratio = box_height / (box_width + 1e-6)
                        if aspect_ratio > 6.0:  # Very tall = likely person
                            continue
                        
                        # Convert from x1,y1,x2,y2 to x,y,width,height
                        bbox = BoundingBox(
                            x=int(x1),
                            y=int(y1),
                            width=int(x2 - x1),
                            height=int(y2 - y1),
                        )
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        bat_center = (float(center_x), float(center_y))
                        
                        width = x2 - x1
                        height = y2 - y1
                        bat_angle = np.degrees(np.arctan2(height, width))
                        
                        detection = BatDetection(
                            bbox=bbox,
                            confidence=float(conf),
                            bat_angle=float(bat_angle),
                            bat_center=bat_center,
                        )
                        frame_detections.append(detection)
                
                all_detections.append(frame_detections)
            
            return all_detections
            
        except Exception as e:
            print(f"ERROR: Batch bat detection failed: {e}")
            return [[] for _ in frames]

    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[BatDetection],
        color: Tuple[int, int, int] = (0, 255, 255),  # Yellow for bat
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bat detections on frame
        
        Args:
            frame: Input frame
            detections: List of bat detections
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            # Draw bounding box (convert from x,y,width,height to x1,y1,x2,y2)
            x1, y1 = det.bbox.x, det.bbox.y
            x2, y2 = det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw center point
            if det.bat_center:
                cx, cy = int(det.bat_center[0]), int(det.bat_center[1])
                cv2.circle(annotated, (cx, cy), 5, color, -1)
            
            # Draw label
            label = f"Bat {det.confidence:.2f}"
            if det.bat_angle is not None:
                label += f" | {det.bat_angle:.1f}°"
            
            # Add text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 8, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        
        return annotated


# Singleton instance
_bat_detector_instance: Optional[BatDetector] = None


def get_bat_detector() -> BatDetector:
    """Get or create singleton BatDetector instance"""
    global _bat_detector_instance
    
    if _bat_detector_instance is None:
        _bat_detector_instance = BatDetector()
    
    return _bat_detector_instance


def create_new_bat_detector() -> BatDetector:
    """Create a new BatDetector instance (for multi-processing)"""
    return BatDetector()

