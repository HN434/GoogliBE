"""
Rule-based cricket shot classification for the worker pipeline using COCO keypoints.

This is a lightweight, pose‑driven classifier that mirrors the heuristics used
in the React `PoseAnalyzer` component and the FastAPI 3D backend, but operates
on the 17‑keypoint COCO format produced by RTMPose in the worker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

# COCO 17‑keypoint index mapping (kept consistent with services.video_pose_metrics)
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


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


def _calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Compute angle ABC (in degrees) in 2D, mirroring the frontend calculateAngle logic.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    v1 = (ax - bx, ay - by)
    v2 = (cx - bx, cy - by)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def _classify_frame_from_person(person: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Apply shot heuristics to a single person in one frame.

    Expects a dict with 'keypoints' (Nx2) and 'scores' (N).
    """
    keypoints: List[KeypointArray] = person.get("keypoints") or []
    scores: List[float] = person.get("scores") or []

    if not keypoints or not scores:
        return None

    # Extract critical joints
    r_sh = _get_point(keypoints, scores, RIGHT_SHOULDER)
    r_el = _get_point(keypoints, scores, RIGHT_ELBOW)
    r_wr = _get_point(keypoints, scores, RIGHT_WRIST)
    l_wr = _get_point(keypoints, scores, LEFT_WRIST)
    r_hip = _get_point(keypoints, scores, RIGHT_HIP)

    if not (r_sh and r_el and r_wr and r_hip):
        return None

    # Angles and relative positions
    elbow_angle = _calculate_angle(r_sh, r_el, r_wr)
    # Approximate body angle: shoulder‑hip relative to vertical
    body_ref = (r_hip[0], r_hip[1] + 1.0)
    shoulder_hip_angle = _calculate_angle(r_sh, r_hip, body_ref)

    shot = "Stance"
    confidence = 0.0

    # Cover Drive: extended arms, hands high
    if elbow_angle > 140.0 and r_wr[1] < r_sh[1] - 0.02:
        shot = "Cover Drive"
        confidence = 0.85
    # Pull Shot: bent arms, hands high and away from shoulder horizontally
    elif elbow_angle < 90.0 and r_wr[1] < r_sh[1] and abs(r_wr[0] - r_sh[0]) > 0.03:
        shot = "Pull Shot"
        confidence = 0.8
    # Flick / Leg Glance: wrists close together, towards leg side (x > hip)
    elif l_wr and abs(r_wr[0] - l_wr[0]) < 0.02 and r_wr[0] > r_hip[0]:
        shot = "Flick / Leg Glance"
        confidence = 0.75
    # Defense: relatively vertical bat, wrists near elbow height
    elif abs(r_wr[1] - r_el[1]) < 0.03 and elbow_angle < 120.0:
        shot = "Defense"
        confidence = 0.7

    return {
        "shot": shot,
        "confidence": confidence,
        "elbow_angle": elbow_angle,
        "shoulder_hip_angle": shoulder_hip_angle,
    }


def classify_shot_from_keypoints(keypoints_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Run rule-based shot classification over the full keypoints sequence
    produced by the worker (list of per-frame dicts).

    We pick the highest-confidence non‑'Stance'/'Unknown' frame as the summary
    shot; if none exist, we fall back to the best stance frame.
    """
    if not keypoints_data:
        return None

    best: Optional[Dict[str, Any]] = None

    def _score_key(pred: Dict[str, Any]) -> Tuple[int, float]:
        label = str(pred.get("shot", "Unknown"))
        base = 1
        if label not in ("Stance", "Unknown"):
            base = 2
        return (base, float(pred.get("confidence", 0.0) or 0.0))

    for frame in keypoints_data:
        persons = frame.get("persons") or []
        if not persons:
            continue

        # Choose main person by mean_confidence if available
        main_person = max(
            persons,
            key=lambda p: float(p.get("mean_confidence", 0.0)),
        )

        pred = _classify_frame_from_person(main_person)
        if not pred:
            continue

        pred["frame_number"] = frame.get("frame_number")

        if best is None or _score_key(pred) > _score_key(best):
            best = pred

    if not best:
        return None

    # Shape result similar to the old TensorFlow classifier output for Redis
    label_map = {
        "Cover Drive": "cover",
        "Pull Shot": "pull",
        "Flick / Leg Glance": "flick",
        "Defense": "defense",
        "Stance": "stance",
        "Unknown": "unknown",
    }

    class_label = label_map.get(best["shot"], "unknown")

    return {
        "shot_label": class_label,
        "shot_display": best["shot"],
        "confidence_percent": float(best["confidence"]) * 100.0,
        "elbow_angle": float(best.get("elbow_angle", 0.0)),
        "shoulder_hip_angle": float(best.get("shoulder_hip_angle", 0.0)),
        "frame_number": int(best.get("frame_number") or 0),
    }



