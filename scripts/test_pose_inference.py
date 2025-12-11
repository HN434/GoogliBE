"""
Test script for RTMPose inference and RF-DETR bat detection.

Usage:
    # Pose detection only
    python scripts/test_pose_inference.py --video path/to/video.mp4 --max-frames 50 --save-viz
    
    # Pose + Bat detection
    python scripts/test_pose_inference.py --video path/to/video.mp4 --max-frames 50 --save-viz --detect-bat
    
    # Pose + Bat detection with temporal smoothing (recommended)
    python scripts/test_pose_inference.py --video path/to/video.mp4 --max-frames 50 --save-viz --detect-bat --temporal-smoothing
    
    # Save all outputs
    python scripts/test_pose_inference.py --video path/to/video.mp4 --max-frames 50 --save-viz --save-json --save-metrics --detect-bat --temporal-smoothing

Output Files:
    - {video_name}_combined_detections.json: Frame-by-frame person pose + bat detection data
    - {video_name}_pose_vis.mp4: Visualization video with drawn keypoints and bat bounding boxes
    - {video_name}_metrics.json: Aggregated pose metrics for the entire video
    
Temporal Smoothing:
    --temporal-smoothing enables intelligent recovery of low-confidence bat detections.
    If a bat was reliably detected in recent frames (within 100 pixels), low-confidence
    detections (0.3-0.5) in the current frame will be accepted. This helps maintain
    continuous bat tracking even when confidence temporarily drops.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

# Add project root to Python path to allow imports from worker, services, etc.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from worker.inference import PoseEstimatorResult, get_pose_estimator
from services.video_pose_metrics import compute_video_pose_metrics
from services.bat_detector import get_bat_detector, BatDetection


class BatTracker:
    """
    Simple bat tracker that assigns IDs and maintains temporal continuity.
    
    Features:
    - Assigns unique IDs to bats across frames
    - Temporal smoothing: recovers low-confidence detections near previous high-confidence ones
    - Connects bats across frames for continuous tracking
    """
    
    def __init__(
        self,
        history_frames: int = 5,
        distance_threshold: float = 100.0,
        min_confidence: float = 0.3,
        high_confidence: float = 0.5,
        max_age: int = 10,
    ):
        """
        Args:
            history_frames: Number of previous frames to check for temporal smoothing
            distance_threshold: Maximum distance (pixels) to consider same bat
            min_confidence: Minimum confidence to consider a detection
            high_confidence: Confidence threshold for "reliable" detections
            max_age: Maximum frames a bat can be missing before ID is retired
        """
        self.history_frames = history_frames
        self.distance_threshold = distance_threshold
        self.min_confidence = min_confidence
        self.high_confidence = high_confidence
        self.max_age = max_age
        
        self.history = []  # List of (frame_num, List[BatDetection])
        self.tracked_bats = {}  # bat_id -> (last_frame, last_center)
        self.next_bat_id = 1
    
    def _distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two centers."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _has_nearby_history(self, bat_center: Tuple[float, float]) -> bool:
        """Check if there was a high-confidence bat detection nearby in recent frames."""
        for frame_num, prev_bats in self.history[-self.history_frames:]:
            for prev_bat in prev_bats:
                if prev_bat.confidence >= self.high_confidence:
                    if prev_bat.bat_center:
                        dist = self._distance(bat_center, prev_bat.bat_center)
                        if dist <= self.distance_threshold:
                            return True
        return False
    
    def _assign_bat_id(self, frame_num: int, bat_center: Tuple[float, float]) -> int:
        """Assign bat ID by matching to tracked bats or creating new ID."""
        # Clean up old tracked bats
        to_remove = []
        for bat_id, (last_frame, last_center) in self.tracked_bats.items():
            if frame_num - last_frame > self.max_age:
                to_remove.append(bat_id)
        for bat_id in to_remove:
            del self.tracked_bats[bat_id]
        
        # Try to match with existing tracked bats
        best_match_id = None
        best_distance = float('inf')
        
        for bat_id, (last_frame, last_center) in self.tracked_bats.items():
            dist = self._distance(bat_center, last_center)
            if dist < self.distance_threshold and dist < best_distance:
                best_distance = dist
                best_match_id = bat_id
        
        # If matched, update tracking info
        if best_match_id is not None:
            self.tracked_bats[best_match_id] = (frame_num, bat_center)
            return best_match_id
        
        # Otherwise, create new ID
        new_id = self.next_bat_id
        self.next_bat_id += 1
        self.tracked_bats[new_id] = (frame_num, bat_center)
        return new_id
    
    def track_and_smooth(
        self,
        frame_num: int,
        raw_detections: List[BatDetection],
        low_conf_detections: List[BatDetection] = None,
    ) -> List[Tuple[int, BatDetection]]:
        """
        Apply temporal smoothing and assign bat IDs.
        
        Args:
            frame_num: Current frame number
            raw_detections: High-confidence detections (above threshold)
            low_conf_detections: Low-confidence detections (below threshold but above minimum)
            
        Returns:
            List of (bat_id, BatDetection) tuples
        """
        smoothed_detections = []
        
        # Add high-confidence detections
        smoothed_detections.extend(raw_detections)
        
        # Check low-confidence detections for temporal smoothing
        if low_conf_detections:
            for low_bat in low_conf_detections:
                if low_bat.confidence >= self.min_confidence and low_bat.bat_center:
                    # Check if this low-confidence detection is near a recent high-confidence one
                    if self._has_nearby_history(low_bat.bat_center):
                        smoothed_detections.append(low_bat)
        
        # Assign IDs to all smoothed detections
        tracked_results = []
        for bat_det in smoothed_detections:
            if bat_det.bat_center:
                bat_id = self._assign_bat_id(frame_num, bat_det.bat_center)
                tracked_results.append((bat_id, bat_det))
        
        # Update history
        self.history.append((frame_num, smoothed_detections))
        
        # Keep only recent history
        if len(self.history) > self.history_frames + 5:
            self.history = self.history[-self.history_frames:]
        
        return tracked_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pose_test")


def summarize_results(results: List[PoseEstimatorResult]) -> str:
    if not results:
        return "no detections"
    avg_conf = sum(r.mean_confidence for r in results) / len(results)
    return f"{len(results)} persons | mean conf={avg_conf:.3f}"


def draw_keypoints(frame: np.ndarray, results: List[PoseEstimatorResult]) -> np.ndarray:
    """
    Draw keypoints and skeleton on frame with color coding.
    COCO 17 keypoints format.
    Color scheme:
    - Head/Face: Cyan/Blue
    - Left Arm: Red
    - Right Arm: Orange
    - Torso: Green
    - Left Leg: Purple
    - Right Leg: Magenta
    """
    # COCO 17 keypoint connections (skeleton) with color assignments
    # Format: (connection, color_name)
    # Note: Head to shoulder center connection is drawn separately below
    skeleton_connections = [
        
        # Left arm (5: left shoulder, 7: left elbow, 9: left wrist)
        ([5, 7], 'left_arm'), ([7, 9], 'left_arm'),
        
        # Right arm (6: right shoulder, 8: right elbow, 10: right wrist)
        ([6, 8], 'right_arm'), ([8, 10], 'right_arm'),
        
        # Torso (shoulders and hips)
        ([5, 6], 'torso'),  # Shoulders
        ([11, 12], 'torso'),  # Hips
        ([5, 11], 'torso'),  # Left shoulder to left hip
        ([6, 12], 'torso'),  # Right shoulder to right hip
        
        # Left leg (11: left hip, 13: left knee, 15: left ankle)
        ([11, 13], 'left_leg'), ([13, 15], 'left_leg'),
        
        # Right leg (12: right hip, 14: right knee, 16: right ankle)
        ([12, 14], 'right_leg'), ([14, 16], 'right_leg'),
    ]
    
    vis_frame = frame.copy()
    
    # Define different color schemes for each person
    # Each person gets a slightly different color palette
    person_color_schemes = [
        # Person 1: Original colors
        {
            'head': (255, 255, 0),      # Cyan
            'left_arm': (0, 0, 255),    # Red
            'right_arm': (0, 165, 255), # Orange
            'torso': (0, 255, 0),       # Green
            'left_leg': (255, 0, 255),  # Magenta
            'right_leg': (128, 0, 128), # Purple
        },
        # Person 2: Brighter variants
        {
            'head': (255, 255, 100),    # Light Cyan
            'left_arm': (100, 100, 255), # Light Red
            'right_arm': (100, 200, 255), # Light Orange
            'torso': (100, 255, 100),   # Light Green
            'left_leg': (255, 100, 255), # Light Magenta
            'right_leg': (200, 100, 200), # Light Purple
        },
        # Person 3: Darker variants
        {
            'head': (200, 200, 0),      # Dark Cyan
            'left_arm': (0, 0, 200),    # Dark Red
            'right_arm': (0, 120, 200), # Dark Orange
            'torso': (0, 200, 0),       # Dark Green
            'left_leg': (200, 0, 200),  # Dark Magenta
            'right_leg': (100, 0, 100), # Dark Purple
        },
        # Person 4+: Cycle through variants
        {
            'head': (150, 255, 150),    # Greenish Cyan
            'left_arm': (200, 150, 0),  # Orange Red
            'right_arm': (0, 200, 255), # Cyan Orange
            'torso': (150, 150, 255),   # Blue Green
            'left_leg': (255, 150, 0), # Yellow Magenta
            'right_leg': (150, 0, 255), # Blue Purple
        },
    ]
    
    for person_idx, result in enumerate(results):
        # Get color scheme for this person (cycle if more than 4 persons)
        color_scheme = person_color_schemes[person_idx % len(person_color_schemes)]
        keypoints = result.keypoints  # Shape: (17, 2) or (17, 3)
        scores = result.scores  # Shape: (17,)
        
        # Extract x, y coordinates (ignore z if present)
        if keypoints.shape[1] >= 2:
            kpts_2d = keypoints[:, :2]
        else:
            continue
        
        # Calculate bounding box size for scaling visualization elements
        # Reference size: 200px average dimension (baseline for current sizes)
        bbox_size = 200.0  # Default reference size
        if result.bbox and len(result.bbox) >= 4:
            bbox_width = result.bbox[2] - result.bbox[0]
            bbox_height = result.bbox[3] - result.bbox[1]
            bbox_avg_dim = (bbox_width + bbox_height) / 2.0
            # Use bbox size if available, otherwise use default
            bbox_size = max(bbox_avg_dim, 50.0)  # Minimum 50px to avoid too small
        
        # Scale visualization elements based on bbox size
        # Baseline: 200px -> line thickness 4, keypoint radii 10/8
        # Smaller persons get proportionally smaller elements
        scale_factor = min(bbox_size / 200.0, 1.0)  # Cap at 1.0 (don't make larger than baseline)
        line_thickness = max(int(4 * scale_factor), 2)  # Minimum 2px for lines
        outer_radius = max(int(10 * scale_factor), 4)  # Minimum 4px
        inner_radius = max(int(8 * scale_factor), 3)  # Minimum 3px
        
        # Draw head to shoulder center connection (special case)
        # Calculate midpoint between left shoulder (5) and right shoulder (6)
        if len(kpts_2d) > 6:
            left_shoulder = kpts_2d[5]  # Left shoulder
            right_shoulder = kpts_2d[6]  # Right shoulder
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            nose = kpts_2d[0]  # Nose
            
            score_nose = scores[0] if 0 < len(scores) else 0.0
            score_left_shoulder = scores[5] if 5 < len(scores) else 0.0
            score_right_shoulder = scores[6] if 6 < len(scores) else 0.0
            
            # Draw if nose and at least one shoulder has reasonable confidence
            if score_nose > 0.3 and (score_left_shoulder > 0.3 or score_right_shoulder > 0.3):
                nose_pt = tuple(nose.astype(int))
                center_pt = tuple(int(c) for c in shoulder_center)
                head_color = color_scheme['head']
                cv2.line(vis_frame, nose_pt, center_pt, head_color, line_thickness)
        
        # Draw skeleton connections with color coding - scaled line thickness
        # Use person-specific color scheme
        for connection, color_name in skeleton_connections:
            idx1, idx2 = connection[0], connection[1]  # Already 0-indexed
            if 0 <= idx1 < len(kpts_2d) and 0 <= idx2 < len(kpts_2d):
                pt1 = tuple(kpts_2d[idx1].astype(int))
                pt2 = tuple(kpts_2d[idx2].astype(int))
                score1 = scores[idx1] if idx1 < len(scores) else 0.5
                score2 = scores[idx2] if idx2 < len(scores) else 0.5
                
                # Only draw if both keypoints have reasonable confidence
                if score1 > 0.3 and score2 > 0.3:
                    # Use person-specific color from scheme
                    color = color_scheme.get(color_name, (255, 255, 255))  # Default white
                    # Scaled line thickness for skeleton
                    cv2.line(vis_frame, pt1, pt2, color, line_thickness)
        
        # Draw keypoints - bigger and darker with body part colors
        # Skip eye and ear keypoints (indices 1, 2, 3, 4) - only show nose for head
        head_keypoints_to_skip = [1, 2, 3, 4]  # Eyes and ears - don't draw these
        
        for i, (x, y) in enumerate(kpts_2d):
            # Skip eye and ear keypoints - only show nose (index 0) for head
            if i in head_keypoints_to_skip:
                continue
                
            score = scores[i] if i < len(scores) else 0.0
            if score > 0.3:  # Only draw visible keypoints
                pt = (int(x), int(y))
                # Map keypoint index to body part and get person-specific color
                if i == 0:  # Nose (head)
                    base_color = color_scheme['head']
                elif i in [5, 7, 9]:  # Left arm
                    base_color = color_scheme['left_arm']
                elif i in [6, 8, 10]:  # Right arm
                    base_color = color_scheme['right_arm']
                elif i in [11, 12]:  # Hips (torso)
                    base_color = color_scheme['torso']
                elif i in [13, 15]:  # Left leg
                    base_color = color_scheme['left_leg']
                elif i in [14, 16]:  # Right leg
                    base_color = color_scheme['right_leg']
                else:
                    base_color = (255, 255, 255)  # Default white
                
                # Darken color based on confidence (higher confidence = brighter)
                if score > 0.7:
                    color = base_color  # Full brightness for high confidence
                else:
                    # Darken for medium confidence
                    color = tuple(int(c * 0.7) for c in base_color)
                
                # Scaled circles: outer circle (dark border), inner circle (filled)
                cv2.circle(vis_frame, pt, outer_radius, (0, 0, 0), -1)  # Black outer circle
                cv2.circle(vis_frame, pt, inner_radius, color, -1)  # Colored inner circle
    
    return vis_frame


def draw_bat_detections(
    frame: np.ndarray, 
    bat_detections: List[BatDetection],
    bat_ids: List[int] = None
) -> np.ndarray:
    """Draw bat detections on frame with yellow bounding boxes and IDs."""
    vis_frame = frame.copy()
    
    for idx, det in enumerate(bat_detections):
        bat_id = bat_ids[idx] if bat_ids and idx < len(bat_ids) else idx
        # Draw bounding box (convert from x,y,width,height to x1,y1,x2,y2)
        x1, y1 = det.bbox.x, det.bbox.y
        x2, y2 = det.bbox.x + det.bbox.width, det.bbox.y + det.bbox.height
        
        # Yellow color for bat
        color = (0, 255, 255)
        thickness = 3
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        if det.bat_center:
            cx, cy = int(det.bat_center[0]), int(det.bat_center[1])
            cv2.circle(vis_frame, (cx, cy), 6, color, -1)
            cv2.circle(vis_frame, (cx, cy), 8, (0, 0, 0), 2)  # Black outline
        
        # Draw label with background
        label = f"BAT#{bat_id} {det.confidence:.2f}"
        if det.bat_angle is not None:
            label += f" | {det.bat_angle:.1f}°"
        
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw text background
        cv2.rectangle(
            vis_frame,
            (x1, y1 - text_h - 12),
            (x1 + text_w + 12, y1),
            color,
            -1,
        )
        
        # Draw text
        cv2.putText(
            vis_frame,
            label,
            (x1 + 6, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
    
    return vis_frame


def save_results_json(results_per_frame: List[Dict[str, Any]], output_path: Path):
    """Save combined person pose and bat detection results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results_per_frame, f, indent=2, default=str)
    logger.info(f"Saved results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RTMPose inference on a local video")
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to a test video file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: process entire video)",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualized frames with keypoints drawn",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save keypoints data to JSON file",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Compute and save aggregated pose metrics JSON for the video",
    )
    parser.add_argument(
        "--detect-bat",
        action="store_true",
        help="Enable bat detection using RF-DETR model",
    )
    parser.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help="Enable temporal smoothing for bat detection (recovers low-confidence detections near previous high-confidence ones)",
    )
    parser.add_argument(
        "--debug-bat",
        action="store_true",
        help="Enable debug logging for bat detection (shows filtered detections)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/test_pose"),
        help="Output directory for saved files",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    estimator = get_pose_estimator()
    logger.info(
        "Loaded RTMPose (device=%s, config=%s)",
        estimator.device,
        estimator.config_path,
    )
    
    # Initialize bat detector if requested
    bat_detector = None
    if args.detect_bat:
        bat_detector = get_bat_detector()
        if bat_detector.enabled:
            logger.info("Loaded RT-DETR bat detector (device=%s)", bat_detector.device)
        else:
            logger.warning("Bat detection requested but model not available")
            bat_detector = None

    # Setup output directory
    if args.save_viz or args.save_json:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    # Get video properties for output video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_frames = args.max_frames if args.max_frames else total_frames
    logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} total frames")
    logger.info(f"Processing {max_frames} frame(s)")
    
    # Setup video writer if saving visualization
    video_writer = None
    if args.save_viz:
        output_video_path = args.output_dir / f"{args.video.stem}_pose_vis.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        logger.info(f"Will save visualization video to: {output_video_path}")

    processed_frames = 0
    total_persons = 0
    total_bats = 0
    total_recovered_bats = 0
    results_per_frame = []  # Combined person + bat data per frame
    
    # Initialize bat tracker if temporal smoothing enabled
    bat_tracker = None
    if args.temporal_smoothing:
        bat_tracker = BatTracker(
            history_frames=5,  # Look back 5 frames
            distance_threshold=100.0,  # 100 pixels max distance
            min_confidence=0.1,  # Accept detections above 0.3 if nearby history exists
            high_confidence=0.5,  # Threshold for "reliable" detections
            max_age=10,  # Max frames without detection before retiring ID
        )
        logger.info("Bat tracking with temporal smoothing enabled")

    while processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose detection
        pose_results = estimator.infer(frame)
        
        # Bat detection with optional tracking/smoothing
        bat_detections = []
        bat_ids = []
        if bat_detector:
            if bat_tracker:
                # Get both high and low confidence detections
                high_conf_bats, low_conf_bats = bat_detector.detect(
                    frame, 
                    return_low_confidence=True,
                    low_confidence_threshold=0.1,
                    debug=args.debug_bat
                )
                
                # Apply tracking and temporal smoothing
                tracked_bats = bat_tracker.track_and_smooth(
                    frame_num=processed_frames,
                    raw_detections=high_conf_bats,
                    low_conf_detections=low_conf_bats,
                )
                
                # Separate IDs and detections
                if tracked_bats:
                    bat_ids, bat_detections = zip(*tracked_bats)
                    bat_ids = list(bat_ids)
                    bat_detections = list(bat_detections)
                
                # Log smoothing info
                if low_conf_bats:
                    recovered = len(bat_detections) - len(high_conf_bats)
                    if recovered > 0:
                        logger.info(
                            "  Temporal smoothing recovered %d low-confidence bat(s)",
                            recovered
                        )
                        total_recovered_bats += recovered
            else:
                # Simple detection without smoothing
                bat_detections = bat_detector.detect(frame, debug=args.debug_bat)
                bat_ids = list(range(len(bat_detections)))  # Simple sequential IDs
        
        logger.info(
            "Frame %d -> %s | Bats: %d",
            processed_frames,
            summarize_results(pose_results),
            len(bat_detections),
        )

        # Store results for JSON export - combined person + bat data
        if args.save_json:
            frame_data = {
                "frame_number": processed_frames,
                "num_persons": len(pose_results),
                "persons": [],
                "num_bats": len(bat_detections),
                "bats": []
            }
            
            # Add person pose data
            for i, result in enumerate(pose_results):
                person_data = {
                    "person_id": i,
                    "keypoints": result.keypoints.tolist(),
                    "scores": result.scores.tolist(),
                    "bbox": result.bbox,
                    "mean_confidence": float(result.mean_confidence),
                }
                frame_data["persons"].append(person_data)
            
            # Add bat detection data with IDs
            if bat_detections:
                frame_data["bats"] = [
                    {
                        **bat.to_dict(),
                        "bat_id": bat_ids[i] if bat_ids and i < len(bat_ids) else i
                    }
                    for i, bat in enumerate(bat_detections)
                ]
            
            results_per_frame.append(frame_data)

        # Draw keypoints and bat detections, save visualization
        if args.save_viz:
            vis_frame = draw_keypoints(frame, pose_results)
            
            # Draw bat detections on top with IDs
            if bat_detections:
                vis_frame = draw_bat_detections(vis_frame, bat_detections, bat_ids)
            
            video_writer.write(vis_frame)
            
            # Also save first frame as image for quick preview
            if processed_frames == 0:
                preview_path = args.output_dir / f"{args.video.stem}_frame0_pose.jpg"
                cv2.imwrite(str(preview_path), vis_frame)
                logger.info(f"Saved preview frame to: {preview_path}")

        total_persons += len(pose_results)
        total_bats += len(bat_detections)
        processed_frames += 1

    cap.release()
    if video_writer:
        video_writer.release()
        logger.info(f"Saved visualization video with {processed_frames} frames")

    # Save JSON results - combined person + bat data
    metrics_data: Dict[str, Any] = {}
    if args.save_json:
        json_path = args.output_dir / f"{args.video.stem}_combined_detections.json"
        save_results_json(results_per_frame, json_path)
        logger.info(f"Saved combined person+bat detections to: {json_path}")

    # Optionally compute and save aggregated metrics using the same structure as the worker
    if args.save_metrics:
        logger.info("Computing aggregated pose metrics for test video")
        # Ensure results_per_frame structure matches worker output
        metrics_data = compute_video_pose_metrics(
            keypoints_data=results_per_frame,
            video_id=args.video.stem,
        )
        metrics_path = args.output_dir / f"{args.video.stem}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Saved aggregated pose metrics to: {metrics_path}")

    logger.info("Finished processing %d frame(s)", processed_frames)
    logger.info("Total persons detected: %d", total_persons)
    logger.info("Average persons per frame: %.3f", total_persons / processed_frames if processed_frames else 0.0)
    
    if bat_detector:
        logger.info("Total bats detected: %d", total_bats)
        logger.info("Average bats per frame: %.3f", total_bats / processed_frames if processed_frames else 0.0)
        
        if bat_tracker:
            unique_bat_ids = len(bat_tracker.tracked_bats) + len([
                bat_id for bat_id, (last_frame, _) in bat_tracker.tracked_bats.items()
                if processed_frames - last_frame > bat_tracker.max_age
            ])
            logger.info("Unique bats tracked: %d", bat_tracker.next_bat_id - 1)
            
            if total_recovered_bats > 0:
                logger.info("Temporal smoothing recovered: %d low-confidence detections", total_recovered_bats)
                logger.info("  Without smoothing: %d bats", total_bats - total_recovered_bats)
                logger.info("  With smoothing: %d bats (+%.1f%%)", 
                           total_bats, 
                           (total_recovered_bats / (total_bats - total_recovered_bats) * 100) if total_bats > total_recovered_bats else 0)


if __name__ == "__main__":
    main()

