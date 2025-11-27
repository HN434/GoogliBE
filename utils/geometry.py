"""
Geometry utilities for pose metrics computation
Includes joint angle calculations, velocities, and spatial transformations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from models.schemas import Keypoint


def calculate_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    point3: Tuple[float, float]
) -> float:
    """
    Calculate angle formed by three points (point2 is the vertex)

    Args:
        point1: First point (x, y)
        point2: Vertex point (x, y)
        point3: Third point (x, y)

    Returns:
        Angle in degrees (0-180)
    """
    # Convert to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    # Calculate vectors
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Avoid numerical errors

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def calculate_signed_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    point3: Tuple[float, float]
) -> float:
    """
    Calculate signed angle (-180 to 180) formed by three points

    Args:
        point1: First point (x, y)
        point2: Vertex point (x, y)
        point3: Third point (x, y)

    Returns:
        Signed angle in degrees (-180 to 180)
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)

    v1 = p1 - p2
    v2 = p3 - p2

    angle_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    # Normalize to [-pi, pi]
    while angle_rad > np.pi:
        angle_rad -= 2 * np.pi
    while angle_rad < -np.pi:
        angle_rad += 2 * np.pi

    return float(np.degrees(angle_rad))


def get_keypoint_position(
    keypoints: List[Keypoint],
    name: str
) -> Optional[Tuple[float, float]]:
    """
    Get position of a keypoint by name

    Args:
        keypoints: List of keypoints
        name: Keypoint name

    Returns:
        (x, y) position or None if not found
    """
    for kp in keypoints:
        if kp.name == name:
            if kp.confidence > 0.3:  # Minimum confidence threshold
                return (kp.x, kp.y)
    return None


def compute_joint_angles(keypoints: List[Keypoint]) -> Dict[str, float]:
    """
    Compute all relevant joint angles from keypoints

    Args:
        keypoints: List of detected keypoints

    Returns:
        Dictionary of joint angles
    """
    angles = {}

    # Left elbow angle (shoulder-elbow-wrist)
    left_shoulder = get_keypoint_position(keypoints, "left_shoulder")
    left_elbow = get_keypoint_position(keypoints, "left_elbow")
    left_wrist = get_keypoint_position(keypoints, "left_wrist")

    if all([left_shoulder, left_elbow, left_wrist]):
        angles["left_elbow_angle"] = calculate_angle(
            left_shoulder, left_elbow, left_wrist
        )

    # Right elbow angle
    right_shoulder = get_keypoint_position(keypoints, "right_shoulder")
    right_elbow = get_keypoint_position(keypoints, "right_elbow")
    right_wrist = get_keypoint_position(keypoints, "right_wrist")

    if all([right_shoulder, right_elbow, right_wrist]):
        angles["right_elbow_angle"] = calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

    # Left shoulder angle (hip-shoulder-elbow)
    left_hip = get_keypoint_position(keypoints, "left_hip")

    if all([left_hip, left_shoulder, left_elbow]):
        angles["left_shoulder_angle"] = calculate_angle(
            left_hip, left_shoulder, left_elbow
        )

    # Right shoulder angle
    right_hip = get_keypoint_position(keypoints, "right_hip")

    if all([right_hip, right_shoulder, right_elbow]):
        angles["right_shoulder_angle"] = calculate_angle(
            right_hip, right_shoulder, right_elbow
        )

    # Left knee angle (hip-knee-ankle)
    left_knee = get_keypoint_position(keypoints, "left_knee")
    left_ankle = get_keypoint_position(keypoints, "left_ankle")

    if all([left_hip, left_knee, left_ankle]):
        angles["left_knee_angle"] = calculate_angle(
            left_hip, left_knee, left_ankle
        )

    # Right knee angle
    right_knee = get_keypoint_position(keypoints, "right_knee")
    right_ankle = get_keypoint_position(keypoints, "right_ankle")

    if all([right_hip, right_knee, right_ankle]):
        angles["right_knee_angle"] = calculate_angle(
            right_hip, right_knee, right_ankle
        )

    # Left hip angle (shoulder-hip-knee)
    if all([left_shoulder, left_hip, left_knee]):
        angles["left_hip_angle"] = calculate_angle(
            left_shoulder, left_hip, left_knee
        )

    # Right hip angle
    if all([right_shoulder, right_hip, right_knee]):
        angles["right_hip_angle"] = calculate_angle(
            right_shoulder, right_hip, right_knee
        )

    # Trunk tilt (vertical deviation of shoulder-hip line)
    if left_shoulder and left_hip and right_shoulder and right_hip:
        # Calculate center points
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )
        hip_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2
        )

        # Calculate tilt from vertical
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        trunk_tilt = np.degrees(np.arctan2(abs(dx), abs(dy)))
        angles["trunk_tilt"] = float(trunk_tilt)

    # Shoulder rotation (shoulder line angle from horizontal)
    if left_shoulder and right_shoulder:
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        shoulder_rotation = np.degrees(np.arctan2(dy, dx))
        angles["shoulder_rotation"] = float(shoulder_rotation)

    return angles


def compute_velocities(
    current_keypoints: List[Keypoint],
    previous_keypoints: List[Keypoint],
    time_delta: float
) -> Dict[str, float]:
    """
    Compute velocities of keypoints between frames

    Args:
        current_keypoints: Keypoints in current frame
        previous_keypoints: Keypoints in previous frame
        time_delta: Time difference in seconds

    Returns:
        Dictionary of velocities (pixels per second)
    """
    velocities = {}

    if time_delta <= 0:
        return velocities

    # Key points to track velocities
    track_points = [
        "left_wrist", "right_wrist",
        "left_ankle", "right_ankle",
        "left_elbow", "right_elbow",
        "nose"
    ]

    for point_name in track_points:
        curr_pos = get_keypoint_position(current_keypoints, point_name)
        prev_pos = get_keypoint_position(previous_keypoints, point_name)

        if curr_pos and prev_pos:
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]

            # Compute speed (magnitude of velocity)
            speed = np.sqrt(dx**2 + dy**2) / time_delta
            velocities[f"{point_name}_velocity_px_per_sec"] = float(speed)

            # Store components as well
            velocities[f"{point_name}_velocity_x"] = float(dx / time_delta)
            velocities[f"{point_name}_velocity_y"] = float(dy / time_delta)

    return velocities


def compute_angular_velocities(
    current_angles: Dict[str, float],
    previous_angles: Dict[str, float],
    time_delta: float
) -> Dict[str, float]:
    """
    Compute angular velocities of joints

    Args:
        current_angles: Joint angles in current frame
        previous_angles: Joint angles in previous frame
        time_delta: Time difference in seconds

    Returns:
        Dictionary of angular velocities (degrees per second)
    """
    angular_velocities = {}

    if time_delta <= 0:
        return angular_velocities

    for angle_name in current_angles:
        if angle_name in previous_angles:
            delta_angle = current_angles[angle_name] - previous_angles[angle_name]
            angular_vel = delta_angle / time_delta
            angular_velocities[f"{angle_name}_velocity"] = float(angular_vel)

    return angular_velocities


def estimate_bat_angle(
    keypoints: List[Keypoint],
    is_right_handed: Optional[bool] = None
) -> Optional[float]:
    """
    Estimate bat angle from wrist and elbow positions

    Args:
        keypoints: List of detected keypoints
        is_right_handed: Player handedness (auto-detect if None)

    Returns:
        Bat angle in degrees from horizontal (0-180)
    """
    # Auto-detect handedness based on wrist positions
    if is_right_handed is None:
        left_wrist = get_keypoint_position(keypoints, "left_wrist")
        right_wrist = get_keypoint_position(keypoints, "right_wrist")

        if left_wrist and right_wrist:
            # Assume hand with lower wrist (higher y value) is holding bat
            is_right_handed = right_wrist[1] > left_wrist[1]
        else:
            is_right_handed = True  # Default assumption

    # Get wrist and elbow positions
    if is_right_handed:
        wrist = get_keypoint_position(keypoints, "right_wrist")
        elbow = get_keypoint_position(keypoints, "right_elbow")
    else:
        wrist = get_keypoint_position(keypoints, "left_wrist")
        elbow = get_keypoint_position(keypoints, "left_elbow")

    if not (wrist and elbow):
        return None

    # Calculate angle from horizontal
    dx = wrist[0] - elbow[0]
    dy = wrist[1] - elbow[1]

    angle_rad = np.arctan2(abs(dy), abs(dx))
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def compute_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Compute Euclidean distance between two points

    Args:
        point1: First point (x, y)
        point2: Second point (x, y)

    Returns:
        Distance in pixels
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return float(np.sqrt(dx**2 + dy**2))


def normalize_keypoints(
    keypoints: List[Keypoint],
    reference_height: Optional[float] = None
) -> List[Keypoint]:
    """
    Normalize keypoint coordinates by body height

    Args:
        keypoints: List of keypoints
        reference_height: Reference height (uses shoulder-ankle if None)

    Returns:
        Normalized keypoints
    """
    if reference_height is None:
        # Estimate height from shoulder to ankle
        left_shoulder = get_keypoint_position(keypoints, "left_shoulder")
        left_ankle = get_keypoint_position(keypoints, "left_ankle")

        if left_shoulder and left_ankle:
            reference_height = abs(left_ankle[1] - left_shoulder[1])
        else:
            return keypoints  # Can't normalize

    if reference_height <= 0:
        return keypoints

    # Create normalized copies
    normalized = []
    for kp in keypoints:
        normalized.append(Keypoint(
            name=kp.name,
            x=kp.x / reference_height,
            y=kp.y / reference_height,
            confidence=kp.confidence
        ))

    return normalized


def smooth_values(
    values: List[float],
    window_size: int = 5
) -> List[float]:
    """
    Apply moving average smoothing to a sequence of values

    Args:
        values: List of values to smooth
        window_size: Size of smoothing window

    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values

    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        window = values[start:end]
        smoothed.append(float(np.mean(window)))

    return smoothed
