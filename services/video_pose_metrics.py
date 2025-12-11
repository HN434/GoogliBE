"""
High-level video pose metrics for cricket batting analysis.

Consumes raw keypoints exported by the worker (per-frame COCO 17-keypoint arrays)
and produces aggregate metrics used for Bedrock analytics and the UI.
Also includes bat detection metrics computation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math


# COCO 17-keypoint index mapping (RTMPose / YOLO style)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


KeypointArray = List[float]  # [x, y] or [x, y, score]


def _get_point(
    keypoints: List[KeypointArray],
    scores: List[float],
    idx: int,
    min_conf: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Safely extract an (x, y) point by index with a confidence check."""
    if idx >= len(keypoints) or idx >= len(scores):
        return None
    if scores[idx] < min_conf:
        return None
    kp = keypoints[idx]
    if len(kp) < 2:
        return None
    return float(kp[0]), float(kp[1])


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _angle_deg(dx: float, dy: float) -> float:
    """Return absolute angle in degrees (0–180) given a vector."""
    return abs(math.degrees(math.atan2(dy, dx)))


def _safe_var(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _normalize_score(value: float, scale: float, clamp_min: float = 0.0, clamp_max: float = 1.0) -> float:
    """Map a non‑negative quantity into a [0, 1] 'good is high' score."""
    if scale <= 0:
        return 0.0
    score = 1.0 - min(value / scale, 1.0)
    return max(clamp_min, min(clamp_max, score))


def _extract_main_person(frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the primary batter in a frame: highest mean_confidence."""
    persons = frame_data.get("persons") or []
    if not persons:
        return None
    return max(persons, key=lambda p: float(p.get("mean_confidence", 0.0)))


def compute_video_pose_metrics(
    keypoints_data: List[Dict[str, Any]],
    video_id: str,
) -> Dict[str, Any]:
    """
    Compute high-level pose metrics for an entire video.

    This operates directly on the JSON structure produced by the worker and
    returns a compact metrics dictionary ready to be passed to Bedrock.
    """
    num_frames = len(keypoints_data)

    # Per-frame accumulators
    stance_widths: List[float] = []
    shoulder_tilts: List[float] = []
    spine_leans: List[float] = []
    bat_angles: List[float] = []
    backlift_heights: List[float] = []
    wrist_x_positions: List[float] = []
    nose_positions: List[Tuple[float, float]] = []
    hip_centers: List[Tuple[float, float]] = []
    ankle_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    frame_mean_confidences: List[float] = []
    torso_lengths: List[float] = []  # For normalization
    shoulder_centers: List[Tuple[float, float]] = []
    hip_rotations: List[float] = []  # Hip rotation angles
    shoulder_rotations: List[float] = []  # Shoulder rotation angles

    # Footwork tracking (front foot chosen as the one that moves forward more)
    left_ankle_ys: List[float] = []
    right_ankle_ys: List[float] = []
    left_ankle_xs: List[float] = []
    right_ankle_xs: List[float] = []

    for frame in keypoints_data:
        person = _extract_main_person(frame)
        if not person:
            continue

        keypoints: List[KeypointArray] = person.get("keypoints") or []
        scores: List[float] = person.get("scores") or []
        if not keypoints or not scores:
            continue

        # Confidence for this frame
        frame_mean_conf = float(person.get("mean_confidence", 0.0))
        frame_mean_confidences.append(frame_mean_conf)

        # Core keypoints
        nose = _get_point(keypoints, scores, NOSE)
        l_sh = _get_point(keypoints, scores, LEFT_SHOULDER)
        r_sh = _get_point(keypoints, scores, RIGHT_SHOULDER)
        l_hip = _get_point(keypoints, scores, LEFT_HIP)
        r_hip = _get_point(keypoints, scores, RIGHT_HIP)
        l_ank = _get_point(keypoints, scores, LEFT_ANKLE)
        r_ank = _get_point(keypoints, scores, RIGHT_ANKLE)
        l_wr = _get_point(keypoints, scores, LEFT_WRIST)
        r_wr = _get_point(keypoints, scores, RIGHT_WRIST)
        l_el = _get_point(keypoints, scores, LEFT_ELBOW)
        r_el = _get_point(keypoints, scores, RIGHT_ELBOW)
        l_knee = _get_point(keypoints, scores, LEFT_KNEE)
        r_knee = _get_point(keypoints, scores, RIGHT_KNEE)

        # Stance width (distance between ankles)
        if l_ank and r_ank:
            stance_widths.append(_distance(l_ank, r_ank))
            ankle_pairs.append((l_ank, r_ank))
            left_ankle_ys.append(l_ank[1])
            right_ankle_ys.append(r_ank[1])
            left_ankle_xs.append(l_ank[0])
            right_ankle_xs.append(r_ank[0])

        # Shoulder tilt and spine lean
        if l_sh and r_sh and l_hip and r_hip:
            shoulder_center = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)
            hip_center = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
            hip_centers.append(hip_center)
            shoulder_centers.append(shoulder_center)
            
            # Calculate torso length for normalization
            torso_length = _distance(shoulder_center, hip_center)
            if torso_length > 0:
                torso_lengths.append(torso_length)

            # Shoulder line vs horizontal
            dx_sh = r_sh[0] - l_sh[0]
            dy_sh = r_sh[1] - l_sh[1]
            shoulder_tilts.append(_angle_deg(dx_sh, dy_sh))

            # Spine line vs vertical
            dx_spine = shoulder_center[0] - hip_center[0]
            dy_spine = shoulder_center[1] - hip_center[1]
            # angle from vertical (swap dx/dy)
            spine_leans.append(_angle_deg(dx_spine, dy_spine))
            
            # Hip rotation: angle of hip line relative to frame horizontal
            hip_dx = r_hip[0] - l_hip[0]
            hip_dy = r_hip[1] - l_hip[1]
            hip_rotation = math.degrees(math.atan2(hip_dy, hip_dx))
            hip_rotations.append(hip_rotation)
            
            # Shoulder rotation: angle of shoulder line relative to frame horizontal
            shoulder_dx = r_sh[0] - l_sh[0]
            shoulder_dy = r_sh[1] - l_sh[1]
            shoulder_rotation = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
            shoulder_rotations.append(shoulder_rotation)

        # Head stability
        if nose:
            nose_positions.append(nose)

        # Bat angle: choose wrist lower in image as bat hand
        bat_wrist = None
        bat_elbow = None
        if l_wr and r_wr:
            if r_wr[1] > l_wr[1]:
                bat_wrist, bat_elbow = r_wr, r_el or r_sh
            else:
                bat_wrist, bat_elbow = l_wr, l_el or l_sh
        elif r_wr:
            bat_wrist, bat_elbow = r_wr, r_el or r_sh
        elif l_wr:
            bat_wrist, bat_elbow = l_wr, l_el or l_sh

        if bat_wrist and bat_elbow:
            dx_bat = bat_wrist[0] - bat_elbow[0]
            dy_bat = bat_wrist[1] - bat_elbow[1]
            bat_angle = _angle_deg(dx_bat, dy_bat)
            bat_angles.append(bat_angle)
            wrist_x_positions.append(bat_wrist[0])

            # Backlift height: vertical distance from hip center to wrist
            if l_hip and r_hip:
                hip_c = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
                backlift_heights.append(abs(hip_c[1] - bat_wrist[1]))

    # === Aggregate metrics ===
    # Stance metrics
    stance_width_avg_px = sum(stance_widths) / len(stance_widths) if stance_widths else 0.0
    stance_width_variance = _safe_var(stance_widths)

    # Head stability: normalized std-dev of nose position relative to stance width
    head_stability_score = 0.0
    if nose_positions and stance_width_avg_px > 0:
        xs = [p[0] for p in nose_positions]
        ys = [p[1] for p in nose_positions]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        std = math.sqrt(
            sum((x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in nose_positions)
            / max(len(nose_positions), 1)
        )
        # Larger std = worse stability; typically a few pixels to a fraction of stance width
        norm_std = std / stance_width_avg_px
        head_stability_score = _normalize_score(norm_std, scale=0.5)

    shoulder_tilt_avg_deg = sum(shoulder_tilts) / len(shoulder_tilts) if shoulder_tilts else 0.0
    spine_lean_avg_deg = sum(spine_leans) / len(spine_leans) if spine_leans else 0.0

    stance_metrics = {
        "stance_width_avg_px": round(stance_width_avg_px, 2),
        "stance_width_variance": round(stance_width_variance, 2),
        "head_stability_score": round(head_stability_score, 2),
        "shoulder_tilt_avg_deg": round(shoulder_tilt_avg_deg, 2),
        "spine_lean_avg_deg": round(spine_lean_avg_deg, 2),
    }

    # Backlift metrics
    bat_angle_avg_deg = sum(bat_angles) / len(bat_angles) if bat_angles else 0.0
    bat_angle_variance = _safe_var(bat_angles)
    backlift_height_avg_px = sum(backlift_heights) / len(backlift_heights) if backlift_heights else 0.0
    backlift_consistency_score = _normalize_score(bat_angle_variance, scale=400.0)
    
    # Normalized backlift height (relative to torso length - scale-invariant!)
    backlift_height_normalized = 0.0
    avg_torso_length = sum(torso_lengths) / len(torso_lengths) if torso_lengths else 0.0
    if avg_torso_length > 0:
        backlift_height_normalized = backlift_height_avg_px / avg_torso_length

    backlift_metrics = {
        "bat_angle_avg_deg": round(bat_angle_avg_deg, 2),
        "bat_angle_variance": round(bat_angle_variance, 2),
        "backlift_height_normalized": round(backlift_height_normalized, 2),
        "backlift_height_avg_px": round(backlift_height_avg_px, 2),
        "backlift_consistency_score": round(backlift_consistency_score, 2),
    }

    # Swing metrics (simple temporal stats over bat angle)
    downswing_angle_change_deg = 0.0
    downswing_smoothness = 0.0
    bat_path_deviation = 0.0

    if len(bat_angles) >= 2:
        downswing_angle_change_deg = max(bat_angles) - min(bat_angles)

        # Smoothness: variation of angle differences between consecutive frames
        diffs = [bat_angles[i + 1] - bat_angles[i] for i in range(len(bat_angles) - 1)]
        if len(diffs) >= 2:
            jerk = _safe_var(diffs)
            downswing_smoothness = _normalize_score(jerk, scale=800.0)

    if wrist_x_positions:
        mean_x = sum(wrist_x_positions) / len(wrist_x_positions)
        bat_path_deviation = math.sqrt(
            sum((x - mean_x) ** 2 for x in wrist_x_positions) / len(wrist_x_positions)
        )

    swing_metrics = {
        "downswing_angle_change_deg": round(downswing_angle_change_deg, 2),
        "downswing_smoothness": round(downswing_smoothness, 2),
        "bat_path_deviation": round(bat_path_deviation, 2),
    }

    # Footwork metrics
    front_foot_stride_avg_px = 0.0
    front_foot_stride_normalized = 0.0  # Normalized by stance width
    footwork_timing_score = 0.0
    front_foot_movement_initiation_frame: Optional[int] = None
    back_foot_pivot_angle_avg_deg = 0.0
    lateral_foot_movement = 0.0  # Side-to-side adjustment

    if left_ankle_ys and right_ankle_ys and left_ankle_xs and right_ankle_xs:
        # In image coords, larger y = further "forward" toward bowler (roughly)
        l_disp = left_ankle_ys[-1] - left_ankle_ys[0]
        r_disp = right_ankle_ys[-1] - right_ankle_ys[0]
        is_left_front = l_disp > r_disp

        front_series_y = left_ankle_ys if is_left_front else right_ankle_ys
        back_series_y = right_ankle_ys if is_left_front else left_ankle_ys
        front_series_x = left_ankle_xs if is_left_front else right_ankle_xs

        front_foot_stride_avg_px = abs(front_series_y[-1] - front_series_y[0])
        
        # Normalize stride by average stance width (scale-invariant!)
        if stance_width_avg_px > 0:
            front_foot_stride_normalized = front_foot_stride_avg_px / stance_width_avg_px

        # Lateral movement (side-to-side adjustment)
        lateral_foot_movement = abs(front_series_x[-1] - front_series_x[0])

        # Movement initiation: first frame where displacement exceeds 5% of stance width
        threshold = stance_width_avg_px * 0.05 if stance_width_avg_px > 0 else 5.0
        for i, y in enumerate(front_series_y):
            if abs(y - front_series_y[0]) >= threshold:
                front_foot_movement_initiation_frame = i
                break

        # Footwork timing score: earlier initiation = better (within first 40% of video)
        if front_foot_movement_initiation_frame is not None and num_frames > 0:
            timing_ratio = front_foot_movement_initiation_frame / num_frames
            # Good timing: 0.1-0.4 range, score higher for middle of that range
            if timing_ratio < 0.1:
                footwork_timing_score = 0.6  # Too early
            elif timing_ratio < 0.4:
                footwork_timing_score = 0.9  # Ideal
            elif timing_ratio < 0.6:
                footwork_timing_score = 0.7  # Acceptable
            else:
                footwork_timing_score = 0.4  # Late

        # Back foot "pivot" as relative change between back and front ankle vertical positions
        if len(front_series_y) >= 2 and len(back_series_y) >= 2:
            rel_angles: List[float] = []
            for f_y, b_y in zip(front_series_y, back_series_y):
                rel_y = f_y - b_y
                # map relative vertical offset into a pseudo-angle; small values expected
                rel_angles.append(abs(rel_y) * 0.2)
            back_foot_pivot_angle_avg_deg = sum(rel_angles) / len(rel_angles)

    footwork_metrics = {
        "front_foot_stride_normalized": round(front_foot_stride_normalized, 2),
        "front_foot_stride_avg_px": round(front_foot_stride_avg_px, 2),
        "lateral_foot_movement_px": round(lateral_foot_movement, 2),
        "footwork_timing_score": round(footwork_timing_score, 2),
        "front_foot_movement_initiation_frame": int(front_foot_movement_initiation_frame or 0),
        "back_foot_pivot_angle_avg_deg": round(back_foot_pivot_angle_avg_deg, 2),
    }

    # Alignment metrics
    shoulder_hip_alignment_avg_deg = 0.0
    hip_knee_symmetry_score = 0.0

    if shoulder_tilts and hip_centers and ankle_pairs:
        # Approximate hip line vs ankle line alignment
        align_angles: List[float] = []
        for (l_ank, r_ank), hip_center in zip(ankle_pairs, hip_centers):
            ankle_dx = r_ank[0] - l_ank[0]
            ankle_dy = r_ank[1] - l_ank[1]
            hip_dx = (hip_center[0] - (l_ank[0] + r_ank[0]) / 2.0)
            hip_dy = (hip_center[1] - (l_ank[1] + r_ank[1]) / 2.0)
            align_angles.append(_angle_deg(ankle_dx - hip_dx, ankle_dy - hip_dy))
        shoulder_hip_alignment_avg_deg = sum(align_angles) / len(align_angles)

    if left_ankle_ys and right_ankle_ys:
        # Symmetry via relative vertical difference between knees over time
        diffs: List[float] = []
        for l_y, r_y in zip(left_ankle_ys, right_ankle_ys):
            diffs.append(abs(l_y - r_y))
        avg_diff = sum(diffs) / len(diffs)
        # Normalize by average stance width
        norm = stance_width_avg_px or 1.0
        hip_knee_symmetry_score = _normalize_score(avg_diff / norm, scale=0.5)

    alignment_metrics = {
        "shoulder_hip_alignment_avg_deg": round(shoulder_hip_alignment_avg_deg, 2),
        "hip_knee_symmetry_score": round(hip_knee_symmetry_score, 2),
    }

    # Weight transfer metrics (enhanced with dynamics)
    com_shift_forward_px = 0.0
    com_shift_normalized = 0.0  # Normalized by torso length
    weight_transfer_velocity = 0.0  # Speed of weight shift
    weight_transfer_smoothness = 0.0  # How smooth is the transfer
    left_right_balance_ratio = 0.5
    balance_stability_score = 0.0  # How stable is the balance

    if hip_centers:
        com_y_start = hip_centers[0][1]
        com_y_end = hip_centers[-1][1]
        com_shift_forward_px = com_y_end - com_y_start
        
        # Normalize by average torso length (scale-invariant!)
        if avg_torso_length > 0:
            com_shift_normalized = com_shift_forward_px / avg_torso_length
        
        # Weight transfer velocity (change per frame, normalized)
        if num_frames > 1 and avg_torso_length > 0:
            weight_transfer_velocity = abs(com_shift_normalized) / num_frames
        
        # Weight transfer smoothness: variance of frame-to-frame changes
        if len(hip_centers) >= 3:
            com_y_values = [hc[1] for hc in hip_centers]
            frame_changes = [com_y_values[i+1] - com_y_values[i] for i in range(len(com_y_values)-1)]
            if frame_changes:
                change_variance = _safe_var(frame_changes)
                # Normalize and score (lower variance = smoother = better)
                norm_variance = change_variance / (avg_torso_length ** 2) if avg_torso_length > 0 else change_variance
                weight_transfer_smoothness = _normalize_score(norm_variance, scale=0.05)

    # Balance analysis: lateral stability and left/right distribution
    if ankle_pairs and hip_centers:
        left_count = 0
        right_count = 0
        lateral_deviations: List[float] = []
        
        for (l_ank, r_ank), hip_c in zip(ankle_pairs, hip_centers):
            mid_x = (l_ank[0] + r_ank[0]) / 2.0
            ankle_width = abs(r_ank[0] - l_ank[0])
            
            # Count left/right distribution
            if hip_c[0] < mid_x:
                left_count += 1
            else:
                right_count += 1
            
            # Measure lateral deviation from center (normalized by stance width)
            if ankle_width > 0:
                lateral_dev = abs(hip_c[0] - mid_x) / ankle_width
                lateral_deviations.append(lateral_dev)
        
        total = left_count + right_count or 1
        left_right_balance_ratio = right_count / total  # fraction of time over right side
        
        # Balance stability: lower deviation = more stable
        if lateral_deviations:
            avg_deviation = sum(lateral_deviations) / len(lateral_deviations)
            balance_stability_score = _normalize_score(avg_deviation, scale=0.5)

    weight_transfer_metrics = {
        "com_shift_normalized": round(com_shift_normalized, 2),
        "com_shift_forward_px": round(com_shift_forward_px, 2),
        "weight_transfer_velocity": round(weight_transfer_velocity, 3),
        "weight_transfer_smoothness": round(weight_transfer_smoothness, 2),
        "balance_stability_score": round(balance_stability_score, 2),
        "left_right_balance_ratio": round(left_right_balance_ratio, 2),
    }

    # Frame quality metrics
    mean_keypoint_confidence = sum(frame_mean_confidences) / len(frame_mean_confidences) if frame_mean_confidences else 0.0
    low_confidence_frames = sum(1 for c in frame_mean_confidences if c < 0.5)

    frame_quality = {
        "mean_keypoint_confidence": round(mean_keypoint_confidence, 2),
        "low_confidence_frames": int(low_confidence_frames),
    }

    # Advanced biomechanical metrics
    hip_shoulder_separation = 0.0  # X-factor (rotation difference)
    hip_shoulder_separation_max = 0.0  # Peak separation
    rotation_timing_score = 0.0  # When does rotation occur
    kinetic_chain_efficiency = 0.0  # Hip leads, then shoulders
    upper_body_rotation_range = 0.0  # Total rotation range
    
    if hip_rotations and shoulder_rotations and len(hip_rotations) == len(shoulder_rotations):
        # Hip-shoulder separation (X-factor): difference in rotation angles
        separations = [abs(sh - hp) for sh, hp in zip(shoulder_rotations, hip_rotations)]
        hip_shoulder_separation = sum(separations) / len(separations) if separations else 0.0
        hip_shoulder_separation_max = max(separations) if separations else 0.0
        
        # Rotation range (how much does upper body rotate)
        if shoulder_rotations:
            upper_body_rotation_range = max(shoulder_rotations) - min(shoulder_rotations)
        
        # Kinetic chain efficiency: hip should lead shoulder rotation
        # Analyze timing of peak rotations
        if len(hip_rotations) >= 3 and len(shoulder_rotations) >= 3:
            # Find peaks
            hip_peak_idx = hip_rotations.index(max(hip_rotations, key=abs))
            shoulder_peak_idx = shoulder_rotations.index(max(shoulder_rotations, key=abs))
            
            # Hip should peak before shoulder for good kinetic chain
            if hip_peak_idx < shoulder_peak_idx:
                # Calculate how much earlier (as fraction of video)
                lead_time = (shoulder_peak_idx - hip_peak_idx) / num_frames if num_frames > 0 else 0
                # Ideal lead time: 10-30% of video duration
                if 0.1 <= lead_time <= 0.3:
                    kinetic_chain_efficiency = 0.9
                elif 0.05 <= lead_time <= 0.4:
                    kinetic_chain_efficiency = 0.7
                else:
                    kinetic_chain_efficiency = 0.5
            else:
                # Shoulder rotating first or simultaneously = poor kinetic chain
                kinetic_chain_efficiency = 0.3
        
        # Rotation timing: when does most rotation occur (should be mid-to-late)
        if len(shoulder_rotations) >= 3:
            max_rotation_idx = shoulder_rotations.index(max(shoulder_rotations, key=abs))
            rotation_timing_ratio = max_rotation_idx / num_frames if num_frames > 0 else 0
            # Good timing: 0.4-0.7 range (middle of shot)
            if 0.4 <= rotation_timing_ratio <= 0.7:
                rotation_timing_score = 0.9
            elif 0.3 <= rotation_timing_ratio <= 0.8:
                rotation_timing_score = 0.7
            else:
                rotation_timing_score = 0.5

    biomechanical_metrics = {
        "hip_shoulder_separation_avg_deg": round(hip_shoulder_separation, 2),
        "hip_shoulder_separation_max_deg": round(hip_shoulder_separation_max, 2),
        "upper_body_rotation_range_deg": round(upper_body_rotation_range, 2),
        "kinetic_chain_efficiency": round(kinetic_chain_efficiency, 2),
        "rotation_timing_score": round(rotation_timing_score, 2),
    }

    result = {
        "video_id": video_id,
        "num_frames": num_frames,
        "stance_metrics": stance_metrics,
        "backlift_metrics": backlift_metrics,
        "swing_metrics": swing_metrics,
        "footwork_metrics": footwork_metrics,
        "alignment_metrics": alignment_metrics,
        "weight_transfer_metrics": weight_transfer_metrics,
        "biomechanical_metrics": biomechanical_metrics,
        "frame_quality": frame_quality,
    }
    
    return result


def add_bat_metrics_to_pose_metrics(
    pose_metrics: Dict[str, Any],
    bat_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Add bat detection metrics to existing pose metrics.
    
    This function is called when bat detection data is available
    (from pipeline processing with RT-DETR).
    
    Args:
        pose_metrics: Existing pose metrics dictionary
        bat_data: Bat detection data from RT-DETR
    
    Returns:
        Updated metrics dictionary with bat_metrics added
    """
    from services.bat_metrics import compute_bat_metrics
    
    bat_metrics = compute_bat_metrics(bat_data)
    pose_metrics["bat_metrics"] = bat_metrics
    
    return pose_metrics



