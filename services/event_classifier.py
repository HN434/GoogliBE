"""
Event Classification Service
Classifies cricket shots and detects short-pitch events using rule-based logic
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from models.schemas import PoseDetection, Event, EventMetrics
from config import settings


@dataclass
class EventCandidate:
    """Candidate event being tracked"""
    event_type: str
    shot_type: Optional[str]
    player_id: int
    start_frame: int
    keyframe: int
    confidences: List[float]
    metrics_history: List[Dict[str, float]]


class EventClassifier:
    """
    Rule-based event classifier for cricket shots and short-pitch detection
    """

    def __init__(self):
        """Initialize event classifier"""
        self.active_events: Dict[int, EventCandidate] = {}  # player_id -> event
        self.completed_events: List[Event] = []
        self.event_id_counter = 1

        # Event detection parameters
        self.min_event_frames = 5
        self.max_event_frames = 30
        self.event_cooldown_frames = 10

        self.player_cooldowns: Dict[int, int] = {}  # player_id -> frames_since_event

        print("🏏 EventClassifier initialized")

    def process_frame(
        self,
        detections: List[PoseDetection],
        frame_num: int,
        fps: float
    ) -> List[Event]:
        """
        Process detections in a frame and classify events

        Args:
            detections: List of pose detections with metrics
            frame_num: Current frame number
            fps: Video frames per second

        Returns:
            List of newly completed events
        """
        new_events = []

        # Update cooldowns
        for player_id in list(self.player_cooldowns.keys()):
            self.player_cooldowns[player_id] -= 1
            if self.player_cooldowns[player_id] <= 0:
                del self.player_cooldowns[player_id]

        for detection in detections:
            if detection.player_id is None or detection.metrics is None:
                continue

            player_id = detection.player_id

            # Skip if in cooldown
            if player_id in self.player_cooldowns:
                continue

            # Check for shot events
            if settings.SHOT_CLASSIFICATION_ENABLED:
                shot_type, confidence = self._classify_shot(detection.metrics)

                if shot_type and confidence >= settings.MIN_SHOT_CONFIDENCE:
                    # Start or update event
                    if player_id not in self.active_events:
                        self.active_events[player_id] = EventCandidate(
                            event_type="shot",
                            shot_type=shot_type,
                            player_id=player_id,
                            start_frame=frame_num,
                            keyframe=frame_num,
                            confidences=[confidence],
                            metrics_history=[detection.metrics]
                        )
                    else:
                        event = self.active_events[player_id]
                        event.confidences.append(confidence)
                        event.metrics_history.append(detection.metrics)

                        # Update keyframe if higher confidence
                        if confidence > max(event.confidences[:-1], default=0):
                            event.keyframe = frame_num

            # Check for short-pitch events
            if settings.SHORT_PITCH_DETECTION_ENABLED:
                is_short_pitch, confidence = self._detect_short_pitch(detection.metrics)

                if is_short_pitch and confidence >= settings.MIN_SHOT_CONFIDENCE:
                    if player_id not in self.active_events or \
                       self.active_events[player_id].event_type != "short_pitch":
                        self.active_events[player_id] = EventCandidate(
                            event_type="short_pitch",
                            shot_type=None,
                            player_id=player_id,
                            start_frame=frame_num,
                            keyframe=frame_num,
                            confidences=[confidence],
                            metrics_history=[detection.metrics]
                        )
                    else:
                        event = self.active_events[player_id]
                        event.confidences.append(confidence)
                        event.metrics_history.append(detection.metrics)

                        if confidence > max(event.confidences[:-1], default=0):
                            event.keyframe = frame_num

        # Finalize events that have ended
        completed_player_ids = []

        for player_id, event in self.active_events.items():
            event_duration = frame_num - event.start_frame

            # Check if event should be finalized
            should_finalize = False

            # Event is too long
            if event_duration > self.max_event_frames:
                should_finalize = True

            # Check if player still has high confidence in recent frames
            if len(event.confidences) >= self.min_event_frames:
                recent_confidences = event.confidences[-3:]
                if all(c < settings.MIN_SHOT_CONFIDENCE for c in recent_confidences):
                    should_finalize = True

            if should_finalize and event_duration >= self.min_event_frames:
                # Create completed event
                completed_event = self._create_event(event, frame_num, fps)
                new_events.append(completed_event)
                self.completed_events.append(completed_event)

                # Set cooldown
                self.player_cooldowns[player_id] = self.event_cooldown_frames

                completed_player_ids.append(player_id)

        # Remove completed events
        for player_id in completed_player_ids:
            del self.active_events[player_id]

        return new_events

    def _classify_shot(
        self,
        metrics: Dict[str, float]
    ) -> Tuple[Optional[str], float]:
        """
        Classify shot type based on metrics

        Args:
            metrics: Pose metrics dictionary

        Returns:
            (shot_type, confidence) tuple
        """
        # Extract relevant metrics
        bat_angle = metrics.get("bat_angle_degrees", 0)
        trunk_tilt = metrics.get("trunk_tilt", 0)
        shoulder_rotation = metrics.get("shoulder_rotation", 0)

        max_wrist_speed = max(
            metrics.get("left_wrist_velocity_px_per_sec", 0),
            metrics.get("right_wrist_velocity_px_per_sec", 0)
        )

        left_elbow_angle = metrics.get("left_elbow_angle", 0)
        right_elbow_angle = metrics.get("right_elbow_angle", 0)
        max_elbow_angle = max(left_elbow_angle, right_elbow_angle)

        left_knee_angle = metrics.get("left_knee_angle", 0)
        right_knee_angle = metrics.get("right_knee_angle", 0)

        # Rule-based classification
        confidences = {}

        # Cover Drive: Front foot forward, bat angle 30-60°, trunk rotation > 45°
        if 30 <= bat_angle <= 60 and abs(shoulder_rotation) > 45:
            front_foot_forward = min(left_knee_angle, right_knee_angle) < 160
            if front_foot_forward and max_wrist_speed > 100:
                confidences["cover_drive"] = 0.8

        # Pull Shot: High elbow, horizontal bat, backward step
        if max_elbow_angle > 120 and bat_angle < 45:
            if abs(shoulder_rotation) > 60:
                backward_step = max(left_knee_angle, right_knee_angle) > 160
                if backward_step and max_wrist_speed > 150:
                    confidences["pull"] = 0.85

        # Cut Shot: Lateral bat swing, raised elbow, minimal trunk rotation
        if 60 <= bat_angle <= 90:
            if 100 <= max_elbow_angle <= 140 and abs(shoulder_rotation) < 45:
                if max_wrist_speed > 120:
                    confidences["cut"] = 0.75

        # Defensive Shot: Vertical bat, low follow-through, minimal movement
        if bat_angle > 70:
            if abs(shoulder_rotation) < 20 and max_wrist_speed < 80:
                confidences["defensive"] = 0.7

        # Sweep Shot: Low bat angle, bent knee, trunk tilt
        if bat_angle < 30 and trunk_tilt > 20:
            if min(left_knee_angle, right_knee_angle) < 120:
                confidences["sweep"] = 0.7

        # Glance/Flick: Minimal bat movement, wrist action
        if 60 <= bat_angle <= 80:
            if abs(shoulder_rotation) < 30 and 50 < max_wrist_speed < 120:
                confidences["glance"] = 0.65

        # Return shot type with highest confidence
        if confidences:
            shot_type = max(confidences, key=confidences.get)
            confidence = confidences[shot_type]
            return shot_type, confidence

        return None, 0.0

    def _detect_short_pitch(
        self,
        metrics: Dict[str, float]
    ) -> Tuple[bool, float]:
        """
        Detect short-pitch delivery based on metrics

        Args:
            metrics: Pose metrics dictionary

        Returns:
            (is_short_pitch, confidence) tuple
        """
        # Extract relevant metrics
        trunk_tilt = metrics.get("trunk_tilt", 0)

        max_elbow_angle = max(
            metrics.get("left_elbow_angle", 0),
            metrics.get("right_elbow_angle", 0)
        )

        # Backward movement (positive y velocity in most coordinate systems)
        ankle_movement_y = metrics.get("left_ankle_velocity_y", 0) + \
                          metrics.get("right_ankle_velocity_y", 0)

        confidence = 0.0

        # High elbow position (defensive/ducking)
        if max_elbow_angle > 140:
            confidence += 0.3

        # Body lean back
        if trunk_tilt > 15:
            confidence += 0.3

        # Rapid backward step
        if ankle_movement_y > 50:  # pixels per second threshold
            confidence += 0.4

        is_short_pitch = confidence >= 0.6

        return is_short_pitch, min(confidence, 1.0)

    def _create_event(
        self,
        candidate: EventCandidate,
        end_frame: int,
        fps: float
    ) -> Event:
        """
        Create Event object from candidate

        Args:
            candidate: Event candidate
            end_frame: Ending frame number
            fps: Video frames per second

        Returns:
            Event object
        """
        # Compute average metrics at keyframe
        keyframe_idx = candidate.keyframe - candidate.start_frame
        keyframe_metrics = candidate.metrics_history[keyframe_idx] \
            if keyframe_idx < len(candidate.metrics_history) else {}

        # Extract event-specific metrics
        event_metrics = EventMetrics(
            bat_angle_degrees=keyframe_metrics.get("bat_angle_degrees"),
            front_foot_forward=self._check_front_foot_forward(keyframe_metrics),
            trunk_rotation_degrees=abs(keyframe_metrics.get("shoulder_rotation", 0)),
            elbow_angle_degrees=max(
                keyframe_metrics.get("left_elbow_angle", 0),
                keyframe_metrics.get("right_elbow_angle", 0)
            ) if keyframe_metrics else None,
            follow_through_speed_px_per_sec=max(
                keyframe_metrics.get("left_wrist_velocity_px_per_sec", 0),
                keyframe_metrics.get("right_wrist_velocity_px_per_sec", 0)
            ) if keyframe_metrics else None,
            backward_step_distance_px=None,  # Would need position tracking
            elbow_height_percentile=None,  # Would need normalization
            body_lean_back_degrees=keyframe_metrics.get("trunk_tilt")
        )

        # Average confidence
        avg_confidence = float(np.mean(candidate.confidences))

        event = Event(
            event_id=self.event_id_counter,
            event_type=candidate.event_type,
            shot_type=candidate.shot_type,
            player_id=candidate.player_id,
            start_frame=candidate.start_frame,
            end_frame=end_frame,
            start_time_seconds=candidate.start_frame / fps,
            end_time_seconds=end_frame / fps,
            keyframe=candidate.keyframe,
            confidence=avg_confidence,
            metrics=event_metrics
        )

        self.event_id_counter += 1

        return event

    def _check_front_foot_forward(self, metrics: Dict[str, float]) -> Optional[bool]:
        """Check if front foot is forward based on knee angles"""
        if not metrics:
            return None

        left_knee = metrics.get("left_knee_angle", 0)
        right_knee = metrics.get("right_knee_angle", 0)

        # Lower knee angle indicates bent knee (front foot forward)
        min_knee = min(left_knee, right_knee)

        return min_knee < 160 if min_knee > 0 else None

    def finalize_all_events(self, final_frame: int, fps: float) -> List[Event]:
        """
        Finalize all active events at end of video

        Args:
            final_frame: Final frame number
            fps: Video frames per second

        Returns:
            List of finalized events
        """
        finalized = []

        for event in self.active_events.values():
            if len(event.confidences) >= self.min_event_frames:
                completed_event = self._create_event(event, final_frame, fps)
                finalized.append(completed_event)
                self.completed_events.append(completed_event)

        self.active_events.clear()

        return finalized

    def get_all_events(self) -> List[Event]:
        """Get all completed events"""
        return self.completed_events

    def reset(self):
        """Reset classifier state"""
        self.active_events.clear()
        self.completed_events.clear()
        self.player_cooldowns.clear()
        self.event_id_counter = 1


# Singleton instance
_classifier_instance = None


def get_event_classifier() -> EventClassifier:
    """Get or create singleton event classifier instance"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = EventClassifier()

    return _classifier_instance


def create_new_event_classifier() -> EventClassifier:
    """Create a new event classifier instance (for multiple videos)"""
    return EventClassifier()
