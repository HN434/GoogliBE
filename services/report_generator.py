"""
Report Generation Service
Generates JSON reports, CSV exports, and summary statistics
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter

from models.schemas import (
    AnalysisResult,
    VideoMetadata,
    ProcessingMetadata,
    Player,
    Event,
    FramePoseData,
    Summary
)
from config import settings


class ReportGenerator:
    """
    Generates analysis reports in various formats
    """

    def __init__(self):
        """Initialize report generator"""
        print("📄 ReportGenerator initialized")

    def generate_json_report(
        self,
        video_metadata: VideoMetadata,
        processing_metadata: ProcessingMetadata,
        players: List[Player],
        events: List[Event],
        pose_data: List[FramePoseData],
        output_path: str,
        bat_data: Optional[List[Dict]] = None
    ) -> AnalysisResult:
        """
        Generate comprehensive JSON report

        Args:
            video_metadata: Video file metadata
            processing_metadata: Processing information
            players: List of tracked players
            events: List of detected events
            pose_data: Frame-by-frame pose data (optional)
            output_path: Path to save JSON file

        Returns:
            AnalysisResult object
        """
        # Generate summary statistics
        summary = self._generate_summary(events)

        # Create analysis result
        result = AnalysisResult(
            video_metadata=video_metadata,
            processing_metadata=processing_metadata,
            players=players,
            events=events,
            pose_data=pose_data if settings.INCLUDE_POSE_DATA_IN_JSON else None,
            summary=summary
        )

        # Save to file
        if settings.EXPORT_JSON_REPORT:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w") as f:
                # Convert to dict and write
                result_dict = result.model_dump()
                
                # Add bat detection data if available
                if bat_data:
                    result_dict["bat_detections"] = bat_data
                
                json.dump(
                    result_dict,
                    f,
                    indent=2,
                    default=str
                )

            print(f"✅ JSON report saved: {output_file}")

        return result

    def generate_csv_timeseries(
        self,
        pose_data: List[FramePoseData],
        output_path: str
    ):
        """
        Generate CSV timeseries export

        Args:
            pose_data: Frame-by-frame pose data
            output_path: Path to save CSV file
        """
        if not settings.EXPORT_CSV_TIMESERIES:
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = [
                "frame",
                "time_seconds",
                "player_id",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "confidence"
            ]

            # Add keypoint columns
            if pose_data and len(pose_data) > 0 and len(pose_data[0].detections) > 0:
                first_detection = pose_data[0].detections[0]
                for kp in first_detection.keypoints:
                    header.extend([
                        f"{kp.name}_x",
                        f"{kp.name}_y",
                        f"{kp.name}_conf"
                    ])

                # Add metrics columns if available
                if first_detection.metrics:
                    for metric_name in first_detection.metrics.keys():
                        header.append(metric_name)

            writer.writerow(header)

            # Write data rows
            for frame_data in pose_data:
                for detection in frame_data.detections:
                    row = [
                        frame_data.frame,
                        frame_data.time_seconds,
                        detection.player_id if detection.player_id else "",
                        detection.bbox.x,
                        detection.bbox.y,
                        detection.bbox.width,
                        detection.bbox.height,
                        detection.confidence
                    ]

                    # Add keypoints
                    for kp in detection.keypoints:
                        row.extend([kp.x, kp.y, kp.confidence])

                    # Add metrics
                    if detection.metrics:
                        for metric_name in detection.metrics.keys():
                            row.append(detection.metrics.get(metric_name, ""))

                    writer.writerow(row)

        print(f"✅ CSV timeseries saved: {output_file}")

    def generate_events_csv(
        self,
        events: List[Event],
        output_path: str
    ):
        """
        Generate CSV export of events

        Args:
            events: List of events
            output_path: Path to save CSV file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "event_id",
                "event_type",
                "shot_type",
                "player_id",
                "start_frame",
                "end_frame",
                "start_time_seconds",
                "end_time_seconds",
                "keyframe",
                "confidence",
                "bat_angle",
                "trunk_rotation",
                "elbow_angle",
                "follow_through_speed"
            ])

            # Data rows
            for event in events:
                writer.writerow([
                    event.event_id,
                    event.event_type,
                    event.shot_type if event.shot_type else "",
                    event.player_id,
                    event.start_frame,
                    event.end_frame,
                    event.start_time_seconds,
                    event.end_time_seconds,
                    event.keyframe,
                    event.confidence,
                    event.metrics.bat_angle_degrees if event.metrics.bat_angle_degrees else "",
                    event.metrics.trunk_rotation_degrees if event.metrics.trunk_rotation_degrees else "",
                    event.metrics.elbow_angle_degrees if event.metrics.elbow_angle_degrees else "",
                    event.metrics.follow_through_speed_px_per_sec if event.metrics.follow_through_speed_px_per_sec else ""
                ])

        print(f"✅ Events CSV saved: {output_file}")

    def _generate_summary(self, events: List[Event]) -> Summary:
        """
        Generate summary statistics from events

        Args:
            events: List of detected events

        Returns:
            Summary object
        """
        # Count shot types
        shot_events = [e for e in events if e.event_type == "shot" and e.shot_type]
        shot_types = [e.shot_type for e in shot_events]
        shot_distribution = dict(Counter(shot_types))

        # Count short-pitch events
        short_pitch_count = sum(1 for e in events if e.event_type == "short_pitch")

        # Calculate average bat speed
        bat_speeds = []
        for event in shot_events:
            if event.metrics.follow_through_speed_px_per_sec:
                bat_speeds.append(event.metrics.follow_through_speed_px_per_sec)

        avg_bat_speed = sum(bat_speeds) / len(bat_speeds) if bat_speeds else None

        # Most frequent shot
        most_frequent = max(shot_distribution, key=shot_distribution.get) \
            if shot_distribution else None

        summary = Summary(
            total_shots_detected=len(shot_events),
            shot_type_distribution=shot_distribution,
            short_pitch_events=short_pitch_count,
            avg_bat_speed_px_per_sec=avg_bat_speed,
            most_frequent_shot=most_frequent
        )

        return summary

    def generate_summary_text(self, result: AnalysisResult) -> str:
        """
        Generate human-readable summary text

        Args:
            result: Analysis result

        Returns:
            Summary text
        """
        lines = [
            "=" * 60,
            "CRICKET POSE ANALYSIS SUMMARY",
            "=" * 60,
            "",
            f"Video: {result.video_metadata.filename}",
            f"Duration: {result.video_metadata.duration_seconds:.1f}s",
            f"Resolution: {result.video_metadata.resolution[0]}x{result.video_metadata.resolution[1]}",
            f"FPS: {result.video_metadata.fps:.1f}",
            f"Processed Frames: {result.video_metadata.processed_frames}/{result.video_metadata.total_frames}",
            "",
            f"Processing Time: {result.processing_metadata.processing_time_seconds:.1f}s",
            f"Model: {result.processing_metadata.model_used}",
            f"Tracker: {result.processing_metadata.tracker_used}",
            f"GPU Used: {'Yes' if result.processing_metadata.gpu_used else 'No'}",
            "",
            "=" * 60,
            "PLAYERS",
            "=" * 60,
        ]

        for player in result.players:
            lines.append(
                f"Player {player.player_id}: "
                f"{player.total_detections} detections, "
                f"avg confidence {player.avg_confidence:.2f}"
            )

        lines.extend([
            "",
            "=" * 60,
            "EVENTS SUMMARY",
            "=" * 60,
            f"Total Shots: {result.summary.total_shots_detected}",
            f"Short-Pitch Events: {result.summary.short_pitch_events}",
        ])

        if result.summary.avg_bat_speed_px_per_sec:
            lines.append(
                f"Average Bat Speed: {result.summary.avg_bat_speed_px_per_sec:.1f} px/s"
            )

        if result.summary.most_frequent_shot:
            lines.append(
                f"Most Frequent Shot: {result.summary.most_frequent_shot.replace('_', ' ').title()}"
            )

        if result.summary.shot_type_distribution:
            lines.extend([
                "",
                "Shot Distribution:"
            ])
            for shot_type, count in result.summary.shot_type_distribution.items():
                percentage = (count / result.summary.total_shots_detected) * 100
                lines.append(
                    f"  {shot_type.replace('_', ' ').title()}: "
                    f"{count} ({percentage:.1f}%)"
                )

        lines.extend([
            "",
            "=" * 60,
            "DETAILED EVENTS",
            "=" * 60,
        ])

        for event in result.events:
            event_label = event.shot_type if event.shot_type else event.event_type
            lines.append(
                f"[{event.start_time_seconds:.1f}s - {event.end_time_seconds:.1f}s] "
                f"Player {event.player_id}: "
                f"{event_label.replace('_', ' ').title()} "
                f"(confidence: {event.confidence:.2f})"
            )

        lines.append("=" * 60)

        return "\n".join(lines)


# Singleton instance
_report_generator_instance = None


def get_report_generator() -> ReportGenerator:
    """Get or create singleton report generator instance"""
    global _report_generator_instance

    if _report_generator_instance is None:
        _report_generator_instance = ReportGenerator()

    return _report_generator_instance
