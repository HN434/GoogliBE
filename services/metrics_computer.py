"""
Pose Metrics Computation Service
Computes joint angles, velocities, and other metrics from pose detections
"""

from typing import List, Dict, Optional
from collections import defaultdict, deque
import numpy as np

from models.schemas import PoseDetection, FramePoseData
from utils.geometry import (
    compute_joint_angles,
    compute_velocities,
    compute_angular_velocities,
    estimate_bat_angle,
    smooth_values
)
from config import settings


class MetricsComputer:
    """
    Computes pose metrics for tracked players across frames
    """

    def __init__(self):
        """Initialize metrics computer"""
        self.player_history = defaultdict(lambda: {
            "detections": deque(maxlen=settings.VELOCITY_SMOOTHING_WINDOW * 2),
            "angles": deque(maxlen=settings.VELOCITY_SMOOTHING_WINDOW * 2),
            "frame_times": deque(maxlen=settings.VELOCITY_SMOOTHING_WINDOW * 2)
        })

        print("📊 MetricsComputer initialized")

    def compute_frame_metrics(
        self,
        detections: List[PoseDetection],
        frame_num: int,
        fps: float
    ) -> List[PoseDetection]:
        """
        Compute metrics for all detections in a frame

        Args:
            detections: List of pose detections with player IDs
            frame_num: Current frame number
            fps: Video frames per second

        Returns:
            Detections with computed metrics
        """
        frame_time = frame_num / fps
        enriched_detections = []

        for detection in detections:
            if detection.player_id is None:
                # Skip untracked detections
                enriched_detections.append(detection)
                continue

            # Compute angles
            angles = compute_joint_angles(detection.keypoints)

            # Compute bat angle
            bat_angle = estimate_bat_angle(detection.keypoints)
            if bat_angle is not None:
                angles["bat_angle_degrees"] = bat_angle

            # Get player history
            history = self.player_history[detection.player_id]

            # Compute velocities if we have previous data
            velocities = {}
            angular_velocities = {}

            if len(history["detections"]) > 0:
                prev_detection = history["detections"][-1]
                prev_angles = history["angles"][-1]
                prev_time = history["frame_times"][-1]

                time_delta = frame_time - prev_time

                if time_delta > 0:
                    # Linear velocities
                    velocities = compute_velocities(
                        detection.keypoints,
                        prev_detection.keypoints,
                        time_delta
                    )

                    # Angular velocities
                    angular_velocities = compute_angular_velocities(
                        angles,
                        prev_angles,
                        time_delta
                    )

            # Combine all metrics
            metrics = {**angles, **velocities, **angular_velocities}

            # Create enriched detection
            enriched_detection = PoseDetection(
                player_id=detection.player_id,
                bbox=detection.bbox,
                keypoints=detection.keypoints,
                confidence=detection.confidence,
                metrics=metrics
            )

            # Update history
            history["detections"].append(detection)
            history["angles"].append(angles)
            history["frame_times"].append(frame_time)

            enriched_detections.append(enriched_detection)

        return enriched_detections

    def get_smoothed_metric(
        self,
        player_id: int,
        metric_name: str,
        window_size: Optional[int] = None
    ) -> Optional[float]:
        """
        Get smoothed value of a metric for a player

        Args:
            player_id: Player ID
            metric_name: Name of the metric
            window_size: Smoothing window size (uses config if None)

        Returns:
            Smoothed metric value or None
        """
        if window_size is None:
            window_size = settings.VELOCITY_SMOOTHING_WINDOW

        history = self.player_history.get(player_id)
        if not history:
            return None

        # Collect metric values from history
        values = []
        for detection in history["detections"]:
            if detection.metrics and metric_name in detection.metrics:
                values.append(detection.metrics[metric_name])

        if len(values) < 2:
            return None

        # Apply smoothing
        smoothed = smooth_values(values, window_size)

        return smoothed[-1] if smoothed else None

    def get_metric_statistics(
        self,
        player_id: int,
        metric_name: str
    ) -> Dict[str, float]:
        """
        Get statistics for a metric over time

        Args:
            player_id: Player ID
            metric_name: Name of the metric

        Returns:
            Dictionary with min, max, mean, std
        """
        history = self.player_history.get(player_id)
        if not history:
            return {}

        # Collect metric values
        values = []
        for detection in history["detections"]:
            if detection.metrics and metric_name in detection.metrics:
                values.append(detection.metrics[metric_name])

        if not values:
            return {}

        values_array = np.array(values)

        return {
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "median": float(np.median(values_array))
        }

    def compute_temporal_features(
        self,
        player_id: int,
        feature_window: int = 10
    ) -> Dict[str, any]:
        """
        Compute temporal features for event detection

        Args:
            player_id: Player ID
            feature_window: Number of frames to analyze

        Returns:
            Dictionary of temporal features
        """
        history = self.player_history.get(player_id)
        if not history or len(history["detections"]) < feature_window:
            return {}

        recent_detections = list(history["detections"])[-feature_window:]
        recent_angles = list(history["angles"])[-feature_window:]

        features = {}

        # Analyze wrist movement (bat swing indicator)
        if settings.COMPUTE_VELOCITIES:
            wrist_speeds = []
            for det in recent_detections:
                if det.metrics:
                    left_wrist_vel = det.metrics.get("left_wrist_velocity_px_per_sec", 0)
                    right_wrist_vel = det.metrics.get("right_wrist_velocity_px_per_sec", 0)
                    max_wrist_speed = max(left_wrist_vel, right_wrist_vel)
                    wrist_speeds.append(max_wrist_speed)

            if wrist_speeds:
                features["max_wrist_speed"] = float(np.max(wrist_speeds))
                features["avg_wrist_speed"] = float(np.mean(wrist_speeds))
                features["wrist_speed_change"] = float(wrist_speeds[-1] - wrist_speeds[0])

        # Analyze trunk rotation
        trunk_tilts = [a.get("trunk_tilt", 0) for a in recent_angles if "trunk_tilt" in a]
        if trunk_tilts:
            features["max_trunk_tilt"] = float(np.max(trunk_tilts))
            features["avg_trunk_tilt"] = float(np.mean(trunk_tilts))
            features["trunk_tilt_change"] = float(trunk_tilts[-1] - trunk_tilts[0])

        # Analyze elbow height (for pull shot / short-pitch detection)
        elbow_angles = []
        for a in recent_angles:
            left_elbow = a.get("left_elbow_angle", 0)
            right_elbow = a.get("right_elbow_angle", 0)
            max_elbow = max(left_elbow, right_elbow)
            elbow_angles.append(max_elbow)

        if elbow_angles:
            features["max_elbow_angle"] = float(np.max(elbow_angles))
            features["avg_elbow_angle"] = float(np.mean(elbow_angles))

        # Analyze foot movement (forward/backward)
        ankle_movements = []
        for det in recent_detections:
            if det.metrics:
                left_ankle_vel_y = det.metrics.get("left_ankle_velocity_y", 0)
                right_ankle_vel_y = det.metrics.get("right_ankle_velocity_y", 0)
                avg_ankle_vel_y = (left_ankle_vel_y + right_ankle_vel_y) / 2
                ankle_movements.append(avg_ankle_vel_y)

        if ankle_movements:
            # Positive y velocity = moving down (backward step in video coordinates)
            features["avg_ankle_movement_y"] = float(np.mean(ankle_movements))
            features["max_ankle_movement_y"] = float(np.max(np.abs(ankle_movements)))

        # Analyze bat angle
        bat_angles = [a.get("bat_angle_degrees", 0) for a in recent_angles if "bat_angle_degrees" in a]
        if bat_angles:
            features["avg_bat_angle"] = float(np.mean(bat_angles))
            features["bat_angle_range"] = float(np.max(bat_angles) - np.min(bat_angles))

        return features

    def reset_player_history(self, player_id: int):
        """Reset history for a specific player"""
        if player_id in self.player_history:
            del self.player_history[player_id]

    def reset(self):
        """Reset all player histories"""
        self.player_history.clear()


# Singleton instance
_metrics_computer_instance = None


def get_metrics_computer() -> MetricsComputer:
    """Get or create singleton metrics computer instance"""
    global _metrics_computer_instance

    if _metrics_computer_instance is None:
        _metrics_computer_instance = MetricsComputer()

    return _metrics_computer_instance


def create_new_metrics_computer() -> MetricsComputer:
    """Create a new metrics computer instance (for multiple videos)"""
    return MetricsComputer()
