"""
Multi-Object Tracking Service using DeepSORT
Tracks players across frames and assigns unique IDs
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque

try:
    from filterpy.kalman import KalmanFilter
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    print("⚠️  filterpy not available, tracking will use simple IoU matching")

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available, tracking may be less accurate")

from config import settings
from models.schemas import PoseDetection, BoundingBox


def bbox_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU value (0-1)
    """
    # Calculate intersection
    x1_min, y1_min = bbox1.x, bbox1.y
    x1_max, y1_max = bbox1.x + bbox1.width, bbox1.y + bbox1.height

    x2_min, y2_min = bbox2.x, bbox2.y
    x2_max, y2_max = bbox2.x + bbox2.width, bbox2.y + bbox2.height

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union
    bbox1_area = bbox1.width * bbox1.height
    bbox2_area = bbox2.width * bbox2.height
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def bbox_center(bbox: BoundingBox) -> Tuple[float, float]:
    """Get center point of bounding box"""
    return (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)


class Track:
    """
    Represents a single tracked object (player)
    """

    def __init__(self, track_id: int, detection: PoseDetection, frame_num: int):
        """
        Initialize track

        Args:
            track_id: Unique track ID
            detection: Initial detection
            frame_num: Frame number of initialization
        """
        self.track_id = track_id
        self.bbox = detection.bbox
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence

        self.hits = 1  # Number of consecutive matches
        self.age = 0  # Frames since creation
        self.time_since_update = 0  # Frames since last update

        self.first_frame = frame_num
        self.last_frame = frame_num

        self.history = deque(maxlen=30)  # Store recent detections
        self.history.append(detection)

        # Initialize Kalman filter if available
        if FILTERPY_AVAILABLE:
            self.kf = self._init_kalman_filter()
            self.predicted_bbox = None
        else:
            self.kf = None

    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter for bbox tracking"""
        # State: [x, y, w, h, vx, vy, vw, vh]
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        kf.F = np.eye(8)
        for i in range(4):
            kf.F[i, i + 4] = 1.0

        # Measurement matrix
        kf.H = np.eye(4, 8)

        # Covariance matrices
        kf.R *= 10.0  # Measurement noise
        kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocity
        kf.Q[4:, 4:] *= 0.01  # Process noise

        # Initialize state
        cx, cy = bbox_center(self.bbox)
        kf.x[:4] = [cx, cy, self.bbox.width, self.bbox.height]

        return kf

    def predict(self):
        """Predict next state using Kalman filter"""
        self.age += 1
        self.time_since_update += 1

        if self.kf is not None:
            self.kf.predict()

            # Extract predicted bbox
            cx, cy, w, h = self.kf.x[:4]
            self.predicted_bbox = BoundingBox(
                x=int(cx - w / 2),
                y=int(cy - h / 2),
                width=int(w),
                height=int(h)
            )

    def update(self, detection: PoseDetection, frame_num: int):
        """
        Update track with new detection

        Args:
            detection: New detection
            frame_num: Current frame number
        """
        self.time_since_update = 0
        self.hits += 1
        self.last_frame = frame_num

        self.bbox = detection.bbox
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence

        self.history.append(detection)

        # Update Kalman filter
        if self.kf is not None:
            cx, cy = bbox_center(detection.bbox)
            measurement = np.array([cx, cy, detection.bbox.width, detection.bbox.height])
            self.kf.update(measurement)

    def get_state(self) -> PoseDetection:
        """Get current detection with track ID"""
        detection = PoseDetection(
            player_id=self.track_id,
            bbox=self.bbox,
            keypoints=self.keypoints,
            confidence=self.confidence
        )
        return detection

    def is_confirmed(self) -> bool:
        """Check if track is confirmed (enough hits)"""
        return self.hits >= settings.MIN_TRACKING_HITS

    def is_stale(self) -> bool:
        """Check if track is too old without updates"""
        return self.time_since_update > settings.MAX_TRACKING_AGE


class PlayerTracker:
    """
    Multi-object tracker for cricket players
    """

    def __init__(self):
        """Initialize tracker"""
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

        # Statistics
        self.player_stats = defaultdict(lambda: {
            "first_frame": 0,
            "last_frame": 0,
            "total_detections": 0,
            "confidences": []
        })

        print("🎯 PlayerTracker initialized")

    def update(
        self,
        detections: List[PoseDetection],
        frame_num: int
    ) -> List[PoseDetection]:
        """
        Update tracker with new detections

        Args:
            detections: List of pose detections in current frame
            frame_num: Current frame number

        Returns:
            List of detections with assigned player IDs
        """
        self.frame_count = frame_num

        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = \
            self._associate_detections_to_tracks(detections)

        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            self.tracks[track_idx].update(detections[det_idx], frame_num)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(self.next_id, detections[det_idx], frame_num)
            self.tracks.append(new_track)
            self.next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks if not t.is_stale()]

        # Get confirmed tracks with IDs
        tracked_detections = []
        for track in self.tracks:
            if track.is_confirmed():
                detection = track.get_state()
                tracked_detections.append(detection)

                # Update statistics
                player_id = track.track_id
                stats = self.player_stats[player_id]

                if stats["first_frame"] == 0:
                    stats["first_frame"] = track.first_frame

                stats["last_frame"] = track.last_frame
                stats["total_detections"] += 1
                stats["confidences"].append(track.confidence)

        return tracked_detections

    def _associate_detections_to_tracks(
        self,
        detections: List[PoseDetection]
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Associate detections to existing tracks using IoU matching

        Returns:
            matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
        """
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks))), []

        # Compute IoU cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            # Use predicted bbox if available, otherwise use last bbox
            track_bbox = track.predicted_bbox if track.predicted_bbox else track.bbox

            for d, detection in enumerate(detections):
                iou = bbox_iou(track_bbox, detection.bbox)
                cost_matrix[t, d] = 1.0 - iou  # Convert to cost (lower is better)

        # Perform Hungarian algorithm matching
        if SCIPY_AVAILABLE:
            matched_tracks, matched_dets = linear_sum_assignment(cost_matrix)
            matched_tracks = matched_tracks.tolist()
            matched_dets = matched_dets.tolist()
        else:
            # Greedy matching as fallback
            matched_tracks, matched_dets = self._greedy_matching(cost_matrix)

        # Filter matches by IoU threshold
        filtered_matches_t = []
        filtered_matches_d = []

        for t, d in zip(matched_tracks, matched_dets):
            iou = 1.0 - cost_matrix[t, d]
            if iou >= settings.TRACKING_IOU_THRESHOLD:
                filtered_matches_t.append(t)
                filtered_matches_d.append(d)

        # Find unmatched tracks and detections
        unmatched_tracks = [
            t for t in range(len(self.tracks))
            if t not in filtered_matches_t
        ]

        unmatched_dets = [
            d for d in range(len(detections))
            if d not in filtered_matches_d
        ]

        return filtered_matches_t, filtered_matches_d, unmatched_tracks, unmatched_dets

    def _greedy_matching(
        self,
        cost_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Greedy matching as fallback when scipy is not available

        Args:
            cost_matrix: Cost matrix (tracks x detections)

        Returns:
            matched_tracks, matched_detections
        """
        matched_tracks = []
        matched_dets = []

        used_tracks = set()
        used_dets = set()

        # Sort by cost (lowest first)
        flat_indices = np.argsort(cost_matrix.ravel())

        for idx in flat_indices:
            t = idx // cost_matrix.shape[1]
            d = idx % cost_matrix.shape[1]

            if t not in used_tracks and d not in used_dets:
                matched_tracks.append(t)
                matched_dets.append(d)
                used_tracks.add(t)
                used_dets.add(d)

        return matched_tracks, matched_dets

    def get_player_statistics(self) -> Dict[int, Dict]:
        """
        Get statistics for all tracked players

        Returns:
            Dictionary of player statistics
        """
        stats = {}

        for player_id, player_data in self.player_stats.items():
            confidences = player_data["confidences"]
            avg_confidence = np.mean(confidences) if confidences else 0.0

            stats[player_id] = {
                "player_id": player_id,
                "first_frame": player_data["first_frame"],
                "last_frame": player_data["last_frame"],
                "total_detections": player_data["total_detections"],
                "avg_confidence": float(avg_confidence)
            }

        return stats

    def reset(self):
        """Reset tracker to initial state"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.player_stats.clear()


# Singleton instance
_tracker_instance = None


def get_tracker() -> PlayerTracker:
    """Get or create singleton tracker instance"""
    global _tracker_instance

    if _tracker_instance is None:
        _tracker_instance = PlayerTracker()

    return _tracker_instance


def create_new_tracker() -> PlayerTracker:
    """Create a new tracker instance (for multiple videos)"""
    return PlayerTracker()
