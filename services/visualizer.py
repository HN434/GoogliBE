"""
Video Annotation and Visualization Service
Renders skeletons, bounding boxes, and event labels on video frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from models.schemas import PoseDetection, Event, Keypoint
from config import settings


# COCO skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    # Head
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),

    # Upper body
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),

    # Torso
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),

    # Lower body
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

# Color palette for different players
PLAYER_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
]

# Event type colors
EVENT_COLORS = {
    "cover_drive": (0, 255, 0),      # Green
    "pull": (0, 0, 255),              # Red
    "cut": (255, 165, 0),             # Orange
    "defensive": (128, 128, 128),     # Gray
    "sweep": (255, 255, 0),           # Cyan
    "glance": (255, 0, 255),          # Magenta
    "short_pitch": (0, 0, 128),       # Dark Red
}


class Visualizer:
    """
    Renders pose annotations on video frames
    """

    def __init__(self):
        """Initialize visualizer"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = settings.FONT_SCALE
        self.thickness = settings.SKELETON_THICKNESS
        self.bbox_thickness = settings.BBOX_THICKNESS

        print("🎨 Visualizer initialized")

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[PoseDetection],
        active_events: Optional[Dict[int, str]] = None,
        frame_info: Optional[str] = None
    ) -> np.ndarray:
        """
        Annotate frame with all visualizations

        Args:
            frame: Input frame (BGR)
            detections: List of pose detections
            active_events: Dict of player_id -> event_label
            frame_info: Additional frame information to display

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw each detection
        for detection in detections:
            player_id = detection.player_id if detection.player_id else 0
            color = self._get_player_color(player_id)

            # Draw bounding box
            if settings.DRAW_BBOXES:
                self._draw_bbox(annotated, detection, color, player_id)

            # Draw skeleton
            if settings.DRAW_SKELETON:
                self._draw_skeleton(annotated, detection.keypoints, color)

            # Draw keypoints
            self._draw_keypoints(annotated, detection.keypoints, color)

            # Draw event label if active
            if settings.DRAW_EVENT_LABELS and active_events and player_id in active_events:
                event_label = active_events[player_id]
                event_color = EVENT_COLORS.get(event_label, (255, 255, 255))
                self._draw_event_label(annotated, detection, event_label, event_color)

        # Draw frame info
        if frame_info:
            self._draw_frame_info(annotated, frame_info)

        return annotated

    def _draw_bbox(
        self,
        frame: np.ndarray,
        detection: PoseDetection,
        color: Tuple[int, int, int],
        player_id: int
    ):
        """Draw bounding box with player ID"""
        bbox = detection.bbox

        # Draw rectangle
        cv2.rectangle(
            frame,
            (bbox.x, bbox.y),
            (bbox.x + bbox.width, bbox.y + bbox.height),
            color,
            self.bbox_thickness
        )

        # Draw player ID and confidence
        if settings.DRAW_PLAYER_IDS and player_id > 0:
            label = f"Player {player_id}"
            label_with_conf = f"{label} ({detection.confidence:.2f})"

            # Background rectangle for text
            (text_w, text_h), _ = cv2.getTextSize(
                label_with_conf,
                self.font,
                self.font_scale * 0.6,
                1
            )

            cv2.rectangle(
                frame,
                (bbox.x, bbox.y - text_h - 10),
                (bbox.x + text_w + 10, bbox.y),
                color,
                -1
            )

            # Text
            cv2.putText(
                frame,
                label_with_conf,
                (bbox.x + 5, bbox.y - 5),
                self.font,
                self.font_scale * 0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint],
        color: Tuple[int, int, int]
    ):
        """Draw skeleton connections"""
        # Create keypoint lookup
        kp_dict = {kp.name: kp for kp in keypoints}

        # Draw each connection
        for start_name, end_name in SKELETON_CONNECTIONS:
            start_kp = kp_dict.get(start_name)
            end_kp = kp_dict.get(end_name)

            if start_kp and end_kp:
                # Check confidence threshold
                if start_kp.confidence > 0.5 and end_kp.confidence > 0.5:
                    start_pos = (int(start_kp.x), int(start_kp.y))
                    end_pos = (int(end_kp.x), int(end_kp.y))

                    # Draw line with alpha based on confidence
                    avg_conf = (start_kp.confidence + end_kp.confidence) / 2
                    alpha = int(avg_conf * 255)

                    cv2.line(
                        frame,
                        start_pos,
                        end_pos,
                        color,
                        self.thickness,
                        cv2.LINE_AA
                    )

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        keypoints: List[Keypoint],
        color: Tuple[int, int, int]
    ):
        """Draw keypoint circles"""
        for kp in keypoints:
            if kp.confidence > 0.5:
                pos = (int(kp.x), int(kp.y))

                # Draw circle
                radius = max(3, int(self.thickness * 1.5))
                cv2.circle(
                    frame,
                    pos,
                    radius,
                    color,
                    -1,
                    cv2.LINE_AA
                )

                # Draw outline
                cv2.circle(
                    frame,
                    pos,
                    radius,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    def _draw_event_label(
        self,
        frame: np.ndarray,
        detection: PoseDetection,
        event_label: str,
        color: Tuple[int, int, int]
    ):
        """Draw event label above detection"""
        # Position above bounding box
        bbox = detection.bbox
        x = bbox.x + bbox.width // 2
        y = bbox.y - 40

        # Format label
        label = event_label.replace("_", " ").title()

        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            self.font,
            self.font_scale * 0.8,
            2
        )

        # Draw background
        cv2.rectangle(
            frame,
            (x - text_w // 2 - 5, y - text_h - 5),
            (x + text_w // 2 + 5, y + 5),
            color,
            -1
        )

        # Draw border
        cv2.rectangle(
            frame,
            (x - text_w // 2 - 5, y - text_h - 5),
            (x + text_w // 2 + 5, y + 5),
            (255, 255, 255),
            2
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x - text_w // 2, y),
            self.font,
            self.font_scale * 0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    def _draw_frame_info(
        self,
        frame: np.ndarray,
        info: str
    ):
        """Draw frame information overlay"""
        # Position at top-left
        x, y = 10, 30

        # Background
        (text_w, text_h), _ = cv2.getTextSize(
            info,
            self.font,
            self.font_scale * 0.7,
            1
        )

        cv2.rectangle(
            frame,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + 5),
            (0, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            info,
            (x, y),
            self.font,
            self.font_scale * 0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    def _get_player_color(self, player_id: int) -> Tuple[int, int, int]:
        """Get color for player ID"""
        if player_id <= 0:
            return (128, 128, 128)  # Gray for untracked
        return PLAYER_COLORS[(player_id - 1) % len(PLAYER_COLORS)]

    def create_event_timeline(
        self,
        frame: np.ndarray,
        events: List[Event],
        current_frame: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Create timeline visualization at bottom of frame

        Args:
            frame: Input frame
            events: List of all events
            current_frame: Current frame number
            total_frames: Total frames in video

        Returns:
            Frame with timeline
        """
        h, w = frame.shape[:2]
        timeline_height = 60
        timeline_y = h - timeline_height

        # Draw timeline background
        cv2.rectangle(
            frame,
            (0, timeline_y),
            (w, h),
            (0, 0, 0),
            -1
        )

        # Draw event markers
        for event in events:
            start_x = int((event.start_frame / total_frames) * w)
            end_x = int((event.end_frame / total_frames) * w)
            keyframe_x = int((event.keyframe / total_frames) * w)

            # Get event color
            color = EVENT_COLORS.get(event.shot_type or event.event_type, (255, 255, 255))

            # Draw event span
            cv2.rectangle(
                frame,
                (start_x, timeline_y + 10),
                (end_x, timeline_y + 40),
                color,
                -1
            )

            # Draw keyframe marker
            cv2.line(
                frame,
                (keyframe_x, timeline_y + 5),
                (keyframe_x, timeline_y + 45),
                (255, 255, 255),
                2
            )

        # Draw current position
        current_x = int((current_frame / total_frames) * w)
        cv2.line(
            frame,
            (current_x, timeline_y),
            (current_x, h),
            (0, 255, 255),
            3
        )

        return frame


# Singleton instance
_visualizer_instance = None


def get_visualizer() -> Visualizer:
    """Get or create singleton visualizer instance"""
    global _visualizer_instance

    if _visualizer_instance is None:
        _visualizer_instance = Visualizer()

    return _visualizer_instance
