"""
Pose extraction using MediaPipe BlazePose
Extracts 3D human pose data from video frames
"""

import mediapipe as mp
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


class PoseExtractor:
    """Extract 3D pose data from video frames using MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0, 1, or 2 (2 = most accurate)
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Ball tracking state
        self.prev_frame_gray = None
        self.ball_trajectory = []

    def extract_pose_3d(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Extract 3D pose landmarks from a single frame

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detections with 3D landmarks, or None if no person detected
        """

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        # Extract 3D landmarks
        landmarks = []
        for landmark in results.pose_world_landmarks.landmark:
            landmarks.append({
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            })

        # Also get 2D normalized landmarks for reference
        landmarks_2d = []
        for landmark in results.pose_landmarks.landmark:
            landmarks_2d.append({
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
                "visibility": float(landmark.visibility)
            })

        detection = {
            "landmarks_3d": landmarks,  # World coordinates (meters)
            "landmarks_2d": landmarks_2d,  # Normalized image coordinates
            "confidence": self._calculate_average_confidence(landmarks)
        }

        return [detection]  # Return as list for multi-person support later

    def _calculate_average_confidence(self, landmarks: List[Dict]) -> float:
        """Calculate average visibility/confidence score"""
        if not landmarks:
            return 0.0

        total = sum(lm["visibility"] for lm in landmarks)
        return total / len(landmarks)

    def detect_players(self, all_poses: List[Dict]) -> List[Dict]:
        """
        Analyze all frames and organize into player sequences

        Args:
            all_poses: List of pose data from all frames

        Returns:
            List of player objects with role and frame sequences
        """

        if not all_poses:
            return []

        # For now, assume single player (batsman)
        # Future: Implement multi-player tracking and role detection

        player_frames = []

        for pose_frame in all_poses:
            if not pose_frame.get("detections"):
                continue

            # Take first detection (single player for now)
            detection = pose_frame["detections"][0]

            frame_data = {
                "frame_index": pose_frame["frame_index"],
                "timestamp": pose_frame["timestamp"],
                "landmarks_3d": detection["landmarks_3d"],
                "landmarks_2d": detection["landmarks_2d"],
                "confidence": detection["confidence"]
            }

            player_frames.append(frame_data)

        # Detect role based on pose characteristics
        role = self._detect_role(player_frames)

        player = {
            "player_id": "player-1",
            "role": role,
            "frames": player_frames,
            "stats": {
                "total_frames": len(player_frames),
                "avg_confidence": np.mean([f["confidence"] for f in player_frames]),
                "duration": player_frames[-1]["timestamp"] if player_frames else 0
            }
        }

        return [player]

    def _detect_role(self, frames: List[Dict]) -> str:
        """
        Detect player role based on pose patterns

        Args:
            frames: Player frame data

        Returns:
            Role string: 'batsman', 'bowler', or 'fielder'
        """

        if not frames:
            return "unknown"

        # Analyze pose characteristics
        # For cricket, we look for:
        # - Batsman: Sideways stance, hands together (holding bat)
        # - Bowler: Arm raised, forward motion
        # - Fielder: Various positions

        hands_together_count = 0
        arm_raised_count = 0

        for frame in frames[:min(30, len(frames))]:  # Sample first 30 frames
            landmarks = frame.get("landmarks_3d", [])

            if len(landmarks) < 33:
                continue

            # Check if hands are together (batsman indicator)
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]

            hand_distance = np.sqrt(
                (left_wrist["x"] - right_wrist["x"]) ** 2 +
                (left_wrist["y"] - right_wrist["y"]) ** 2 +
                (left_wrist["z"] - right_wrist["z"]) ** 2
            )

            if hand_distance < 0.3:  # Threshold for hands together
                hands_together_count += 1

            # Check if arm is raised (bowler indicator)
            right_wrist_y = right_wrist["y"]
            right_shoulder_y = landmarks[12]["y"]

            if right_wrist_y < right_shoulder_y - 0.3:  # Wrist above shoulder
                arm_raised_count += 1

        # Determine role
        total_frames_checked = min(30, len(frames))

        if hands_together_count > total_frames_checked * 0.5:
            return "batsman"
        elif arm_raised_count > total_frames_checked * 0.3:
            return "bowler"
        else:
            return "fielder"

    def detect_ball(self, frame: np.ndarray, frame_index: int) -> Optional[Dict]:
        """
        Detect cricket ball in frame using color and motion detection

        Args:
            frame: BGR image from OpenCV
            frame_index: Current frame number

        Returns:
            Ball position dict or None if not detected
        """

        # Convert to HSV for color detection (cricket ball is typically red)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for red color (cricket ball)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find the most circular contour (ball is round)
        best_ball = None
        best_circularity = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (ball shouldn't be too small or too large)
            if area < 20 or area > 5000:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity

                # Get center of contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Normalize to 0-1 range
                    height, width = frame.shape[:2]
                    best_ball = {
                        "x": cx / width,
                        "y": cy / height,
                        "radius": np.sqrt(area / np.pi) / width,  # Normalized
                        "confidence": float(circularity),
                        "frame_index": frame_index
                    }

        if best_ball:
            self.ball_trajectory.append(best_ball)

        return best_ball

    def get_ball_trajectory(self) -> List[Dict]:
        """
        Get complete ball trajectory

        Returns:
            List of ball positions across frames
        """
        return self.ball_trajectory

    def cleanup(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
        self.ball_trajectory = []


# MediaPipe Pose landmark indices (for reference)
POSE_LANDMARKS = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32
}
