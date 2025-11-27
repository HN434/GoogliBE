"""
Pose extraction using YOLOv8-pose
Extracts 3D human pose data from video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple


class YoloPoseExtractor:
    """Extract pose data from video frames using YOLOv8-pose"""

    def __init__(self):
        # Load YOLOv8 pose model
        # This will auto-download the model on first run
        self.model = YOLO('yolov8n-pose.pt')  # nano model for speed

        # Ball tracking state
        self.ball_trajectory = []

        # YOLO pose has 17 keypoints (COCO format)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

    def extract_pose_3d(self, frame: np.ndarray) -> Optional[List[Dict]]:
        """
        Extract pose landmarks from a single frame using YOLO

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detection dicts with landmarks
        """

        # Run YOLO pose detection
        results = self.model(frame, verbose=False)

        if not results or len(results) == 0:
            return None

        detections = []

        # Process each detected person
        for result in results:
            if result.keypoints is None or len(result.keypoints.data) == 0:
                continue

            for person_idx, keypoints in enumerate(result.keypoints.data):
                # keypoints shape: [17, 3] - 17 keypoints, each with (x, y, confidence)
                landmarks_2d = []
                landmarks_3d = []

                # Get frame dimensions for normalization
                height, width = frame.shape[:2]

                for kp_idx, kp in enumerate(keypoints):
                    x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])

                    # Normalize coordinates to 0-1 range for 2D
                    normalized_x = x / width if width > 0 else 0
                    normalized_y = y / height if height > 0 else 0

                    # 2D landmarks (normalized)
                    landmarks_2d.append({
                        'x': normalized_x,
                        'y': normalized_y,
                        'visibility': conf
                    })

                    # 3D landmarks (approximate - YOLO doesn't provide true 3D)
                    # We'll create pseudo-3D by using normalized coordinates
                    # Z-coordinate is approximated based on confidence and body part
                    z = 0.0  # Default depth

                    # Approximate depth based on typical human proportions
                    # Center points (hips) are reference
                    if kp_idx in [11, 12]:  # Hips
                        z = 0.0
                    elif kp_idx in [5, 6]:  # Shoulders
                        z = -0.1
                    elif kp_idx in [0, 1, 2, 3, 4]:  # Head
                        z = -0.15
                    elif kp_idx in [7, 8, 9, 10]:  # Arms
                        z = -0.05
                    elif kp_idx in [13, 14, 15, 16]:  # Legs
                        z = 0.05

                    landmarks_3d.append({
                        'x': normalized_x - 0.5,  # Center around 0
                        'y': -(normalized_y - 0.5),  # Invert Y for 3D space
                        'z': z,
                        'visibility': conf
                    })

                # Get bounding box if available
                bbox = None
                if result.boxes is not None and len(result.boxes.data) > person_idx:
                    box = result.boxes.data[person_idx]
                    x1, y1, x2, y2 = box[:4]
                    bbox = {
                        'x_min': float(x1) / width,
                        'y_min': float(y1) / height,
                        'x_max': float(x2) / width,
                        'y_max': float(y2) / height
                    }

                detections.append({
                    'landmarks_2d': landmarks_2d,
                    'landmarks_3d': landmarks_3d,
                    'bbox': bbox,
                    'person_id': person_idx
                })

        return detections if detections else None

    def convert_yolo_to_mediapipe_format(self, yolo_landmarks: List[Dict]) -> List[Dict]:
        """
        Convert YOLO 17-keypoint format to MediaPipe 33-landmark format
        for compatibility with existing frontend code

        YOLO keypoints (17):
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

        MediaPipe keypoints (33) - we'll map what we can and interpolate the rest
        """

        # Create 33-landmark array with default values
        mp_landmarks = []
        for i in range(33):
            mp_landmarks.append({
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'visibility': 0.0
            })

        # Map YOLO to MediaPipe indices
        # MediaPipe: 0=nose, 2=left_eye_inner, 5=right_eye_inner,
        #           7=left_ear, 8=right_ear, 11=left_shoulder, 12=right_shoulder,
        #           13=left_elbow, 14=right_elbow, 15=left_wrist, 16=right_wrist,
        #           23=left_hip, 24=right_hip, 25=left_knee, 26=right_knee,
        #           27=left_ankle, 28=right_ankle

        mapping = {
            0: 0,   # nose -> nose
            1: 2,   # left_eye -> left_eye_inner
            2: 5,   # right_eye -> right_eye_inner
            3: 7,   # left_ear -> left_ear
            4: 8,   # right_ear -> right_ear
            5: 11,  # left_shoulder -> left_shoulder
            6: 12,  # right_shoulder -> right_shoulder
            7: 13,  # left_elbow -> left_elbow
            8: 14,  # right_elbow -> right_elbow
            9: 15,  # left_wrist -> left_wrist
            10: 16, # right_wrist -> right_wrist
            11: 23, # left_hip -> left_hip
            12: 24, # right_hip -> right_hip
            13: 25, # left_knee -> left_knee
            14: 26, # right_knee -> right_knee
            15: 27, # left_ankle -> left_ankle
            16: 28, # right_ankle -> right_ankle
        }

        # Map available keypoints
        for yolo_idx, mp_idx in mapping.items():
            if yolo_idx < len(yolo_landmarks):
                mp_landmarks[mp_idx] = yolo_landmarks[yolo_idx].copy()

        # Interpolate missing landmarks
        # Eyes (1, 3, 4, 6)
        if mp_landmarks[2]['visibility'] > 0:
            mp_landmarks[1] = mp_landmarks[2].copy()
            mp_landmarks[3] = mp_landmarks[2].copy()
        if mp_landmarks[5]['visibility'] > 0:
            mp_landmarks[4] = mp_landmarks[5].copy()
            mp_landmarks[6] = mp_landmarks[5].copy()

        # Mouth (9, 10)
        if mp_landmarks[0]['visibility'] > 0:
            mp_landmarks[9] = mp_landmarks[0].copy()
            mp_landmarks[10] = mp_landmarks[0].copy()
            mp_landmarks[9]['y'] += 0.02
            mp_landmarks[10]['y'] += 0.02

        # Hands (17-22)
        if mp_landmarks[15]['visibility'] > 0:  # left wrist
            for i in range(17, 20):
                mp_landmarks[i] = mp_landmarks[15].copy()
                mp_landmarks[i]['x'] -= 0.02
        if mp_landmarks[16]['visibility'] > 0:  # right wrist
            for i in range(20, 23):
                mp_landmarks[i] = mp_landmarks[16].copy()
                mp_landmarks[i]['x'] += 0.02

        # Feet (29-32)
        if mp_landmarks[27]['visibility'] > 0:  # left ankle
            mp_landmarks[29] = mp_landmarks[27].copy()
            mp_landmarks[31] = mp_landmarks[27].copy()
            mp_landmarks[29]['y'] += 0.03
            mp_landmarks[31]['x'] -= 0.02
        if mp_landmarks[28]['visibility'] > 0:  # right ankle
            mp_landmarks[30] = mp_landmarks[28].copy()
            mp_landmarks[32] = mp_landmarks[28].copy()
            mp_landmarks[30]['y'] += 0.03
            mp_landmarks[32]['x'] += 0.02

        return mp_landmarks

    def detect_players(self, all_poses: List[Dict]) -> List[Dict]:
        """
        Identify unique players and assign roles
        """

        if not all_poses or len(all_poses) == 0:
            return []

        # Group detections by person_id
        player_frames = {}

        for pose_frame in all_poses:
            if 'detections' not in pose_frame or not pose_frame['detections']:
                continue

            for detection in pose_frame['detections']:
                person_id = detection.get('person_id', 0)

                if person_id not in player_frames:
                    player_frames[person_id] = []

                # Convert to MediaPipe format for compatibility
                landmarks_3d = self.convert_yolo_to_mediapipe_format(
                    detection['landmarks_3d']
                )
                landmarks_2d = self.convert_yolo_to_mediapipe_format(
                    detection['landmarks_2d']
                )

                player_frames[person_id].append({
                    'frame_index': pose_frame['frame_index'],
                    'timestamp': pose_frame['timestamp'],
                    'landmarks_3d': landmarks_3d,
                    'landmarks_2d': landmarks_2d
                })

        # Create player objects
        players = []
        for player_id, frames in player_frames.items():
            # Determine role (simplified)
            role = self._determine_player_role(frames)

            players.append({
                'player_id': player_id,
                'role': role,
                'frames': frames
            })

        return players

    def _determine_player_role(self, frames: List[Dict]) -> str:
        """
        Determine player role based on pose patterns
        """

        # Simple heuristic: check wrist positions
        hands_together_count = 0
        total_frames = len(frames)

        for frame in frames[:min(30, total_frames)]:
            landmarks = frame['landmarks_3d']

            if len(landmarks) < 17:
                continue

            left_wrist = landmarks[15]
            right_wrist = landmarks[16]

            if left_wrist['visibility'] > 0.5 and right_wrist['visibility'] > 0.5:
                distance = abs(left_wrist['x'] - right_wrist['x'])
                if distance < 0.15:  # Hands close together (batting stance)
                    hands_together_count += 1

        if hands_together_count > total_frames * 0.4:
            return "batsman"
        else:
            return "fielder"

    def detect_ball(self, frame: np.ndarray, frame_index: int) -> Optional[Dict]:
        """
        Detect cricket ball using color detection
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

            # Filter by size
            if area < 20 or area > 5000:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity > best_circularity and circularity > 0.6:
                best_circularity = circularity

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    height, width = frame.shape[:2]
                    best_ball = {
                        "x": cx / width,
                        "y": cy / height,
                        "radius": np.sqrt(area / np.pi) / width,
                        "confidence": float(circularity),
                        "frame_index": frame_index
                    }

        if best_ball:
            self.ball_trajectory.append(best_ball)

        return best_ball

    def get_ball_trajectory(self) -> List[Dict]:
        """Get complete ball trajectory"""
        return self.ball_trajectory

    def cleanup(self):
        """Release resources"""
        self.ball_trajectory = []
