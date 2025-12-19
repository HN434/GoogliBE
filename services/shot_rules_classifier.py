"""
Rule-based cricket shot classification using 3D pose landmarks.

This mirrors the frontend `PoseAnalyzer` classifyShot logic, but runs on the
backend using the MediaPipe-style 3D landmarks produced by our pose extractors.

We deliberately keep this light-weight and dependency-free so it can be reused
from both the FastAPI 3D pose backend and (eventually) the worker pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math


# MediaPipe landmark indices (kept in sync with `POSE_LANDMARKS` in pose_extractor.py)
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
LEFT_WRIST = 15
RIGHT_HIP = 24


Landmark = Dict[str, float]
Frame = Dict[str, object]  # expects "landmarks_3d": List[Landmark]


def _calculate_angle(a: Landmark, b: Landmark, c: Landmark) -> float:
    """
    Calculate joint angle ABC (in degrees) given 3D points a, b, c.

    This mirrors the frontend `calculateAngle` helper.
    """
    if not a or not b or not c:
        return 0.0

    v1 = (a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])
    v2 = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])

    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def classify_shot_frame(landmarks_3d: List[Landmark]) -> Dict[str, float | str]:
    """
    Classify a single frame into a coarse shot type using pose heuristics.

    Heuristics mirror the React `classifyShot` logic:
      - Cover Drive: high front foot / extended arms (large elbow angle, hands high)
      - Pull Shot: bent arms, hands high and away from shoulder line
      - Flick / Leg Glance: wrists close together towards leg side
      - Defense: relatively vertical bat, hands close together near body
      - Fallback: "Stance" when none of the above triggers
    """
    if not landmarks_3d or len(landmarks_3d) <= max(
        RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_HIP
    ):
        return {"shot": "Unknown", "confidence": 0.0}

    right_wrist = landmarks_3d[RIGHT_WRIST]
    right_elbow = landmarks_3d[RIGHT_ELBOW]
    right_shoulder = landmarks_3d[RIGHT_SHOULDER]
    left_wrist = landmarks_3d[LEFT_WRIST]
    right_hip = landmarks_3d[RIGHT_HIP]

    # Basic visibility check
    for lm in (right_wrist, right_elbow, right_shoulder):
        if lm.get("visibility", 0.0) < 0.3:
            return {"shot": "Unknown", "confidence": 0.0}

    elbow_angle = _calculate_angle(right_shoulder, right_elbow, right_wrist)
    # Approximate body angle: shoulder–hip vs vertical
    shoulder_hip_angle = _calculate_angle(
        right_shoulder,
        right_hip,
        {"x": right_hip["x"], "y": right_hip["y"] + 1.0, "z": right_hip["z"]},
    )

    shot = "Stance"
    confidence = 0.0

    # Heuristics ported from frontend:
    # Cover Drive: High front foot, extended arms (large elbow angle, wrist significantly above shoulder)
    if elbow_angle > 140.0 and right_wrist["y"] < right_shoulder["y"] - 0.2:
        shot = "Cover Drive"
        confidence = 0.85
    # Pull Shot: Arms bent, wrist high and across body (large horizontal offset from shoulder)
    elif elbow_angle < 90.0 and right_wrist["y"] < right_shoulder["y"] and abs(
        right_wrist["x"] - right_shoulder["x"]
    ) > 0.3:
        shot = "Pull Shot"
        confidence = 0.8
    # Flick / Leg Glance: Wrists together, on leg side of body
    elif left_wrist and abs(right_wrist["x"] - left_wrist["x"]) < 0.2 and right_wrist[
        "x"
    ] > right_hip["x"]:
        shot = "Flick / Leg Glance"
        confidence = 0.75
    # Defense: Vertical-ish bat, hands close together near body
    elif abs(right_wrist["y"] - right_elbow["y"]) < 0.3 and elbow_angle < 120.0:
        shot = "Defense"
        confidence = 0.7

    return {
        "shot": shot,
        "confidence": confidence,
        "elbow_angle": elbow_angle,
        "shoulder_hip_angle": shoulder_hip_angle,
    }


def classify_shot_sequence(frames: List[Frame]) -> Optional[Dict[str, object]]:
    """
    Run frame-wise classification over a sequence and pick the best shot label.

    We return the highest-confidence non-"Stance"/"Unknown" prediction if any
    exists; otherwise we fall back to the best "Stance" frame.
    """
    if not frames:
        return None

    best: Optional[Dict[str, object]] = None

    for frame in frames:
        landmarks_3d = frame.get("landmarks_3d") or []
        if not isinstance(landmarks_3d, list) or not landmarks_3d:
            continue

        prediction = classify_shot_frame(landmarks_3d)
        shot = prediction.get("shot", "Unknown")
        confidence = float(prediction.get("confidence", 0.0) or 0.0)

        # Enrich with frame index/time if available
        prediction["frame_index"] = frame.get("frame_index")
        prediction["timestamp"] = frame.get("timestamp")

        def _score_key(p: Dict[str, object]) -> Tuple[int, float]:
            """Sort key: prefer non-stance, non-unknown first, then by confidence."""
            label = str(p.get("shot", "Unknown"))
            base = 1
            if label not in ("Stance", "Unknown"):
                base = 2
            return (base, float(p.get("confidence", 0.0) or 0.0))

        if best is None or _score_key(prediction) > _score_key(best):
            best = {
                "shot": shot,
                "confidence": confidence,
                "elbow_angle": prediction.get("elbow_angle", 0.0),
                "shoulder_hip_angle": prediction.get("shoulder_hip_angle", 0.0),
                "frame_index": prediction.get("frame_index"),
                "timestamp": prediction.get("timestamp"),
            }

    return best



