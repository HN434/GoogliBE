"""
Bat Detection Metrics Computation

Processes bat detection data from RT-DETR and computes aggregate metrics
for cricket batting analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import math


def compute_bat_metrics(bat_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate bat detection metrics from frame-by-frame bat detections.
    
    Args:
        bat_data: List of bat detection frames, each containing:
            - frame: frame number
            - time_seconds: timestamp
            - bats: list of bat detections with bbox, confidence, angle, center
    
    Returns:
        Dictionary with bat metrics:
        - bat_detected: bool (was bat detected in video)
        - detection_rate: fraction of frames with bat detected
        - avg_confidence: average detection confidence
        - bat_angle_avg_deg: average bat angle
        - bat_angle_range_deg: range of bat angles (rotation)
        - bat_movement_velocity: average bat movement speed
        - bat_path_length: total distance bat moved
        - bat_consistency_score: how consistently bat was detected
    """
    if not bat_data:
        return {
            "bat_detected": False,
            "detection_rate": 0.0,
            "avg_confidence": 0.0,
            "bat_angle_avg_deg": 0.0,
            "bat_angle_range_deg": 0.0,
            "bat_movement_velocity": 0.0,
            "bat_path_length": 0.0,
            "bat_consistency_score": 0.0,
        }
    
    # Extract bat data
    all_confidences: List[float] = []
    all_angles: List[float] = []
    all_centers: List[tuple] = []
    frames_with_bat = 0
    
    for frame_data in bat_data:
        bats = frame_data.get("bats", [])
        if bats:
            frames_with_bat += 1
            # Take the highest confidence bat if multiple detected
            best_bat = max(bats, key=lambda b: b.get("confidence", 0))
            
            all_confidences.append(best_bat.get("confidence", 0))
            
            if best_bat.get("bat_angle") is not None:
                all_angles.append(best_bat["bat_angle"])
            
            if best_bat.get("bat_center"):
                center = best_bat["bat_center"]
                if isinstance(center, list) and len(center) == 2:
                    all_centers.append((center[0], center[1]))
    
    # Basic metrics
    bat_detected = frames_with_bat > 0
    total_frames = len(bat_data)
    detection_rate = frames_with_bat / total_frames if total_frames > 0 else 0.0
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    # Angle metrics
    bat_angle_avg_deg = sum(all_angles) / len(all_angles) if all_angles else 0.0
    bat_angle_range_deg = (max(all_angles) - min(all_angles)) if len(all_angles) >= 2 else 0.0
    
    # Movement metrics
    bat_movement_velocity = 0.0
    bat_path_length = 0.0
    
    if len(all_centers) >= 2:
        # Calculate total path length
        distances = []
        for i in range(len(all_centers) - 1):
            x1, y1 = all_centers[i]
            x2, y2 = all_centers[i + 1]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(dist)
            bat_path_length += dist
        
        # Average velocity (distance per frame)
        bat_movement_velocity = sum(distances) / len(distances) if distances else 0.0
    
    # Consistency score: how consistently was bat detected
    # High score = detected in most frames with high confidence
    consistency_score = 0.0
    if bat_detected:
        # Combine detection rate and confidence
        consistency_score = (detection_rate * 0.6) + (avg_confidence * 0.4)
    
    return {
        "bat_detected": bat_detected,
        "detection_rate": round(detection_rate, 3),
        "avg_confidence": round(avg_confidence, 3),
        "bat_angle_avg_deg": round(bat_angle_avg_deg, 2),
        "bat_angle_range_deg": round(bat_angle_range_deg, 2),
        "bat_movement_velocity": round(bat_movement_velocity, 2),
        "bat_path_length": round(bat_path_length, 2),
        "bat_consistency_score": round(consistency_score, 3),
    }

