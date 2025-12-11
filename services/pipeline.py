"""
Processing Pipeline Orchestrator
Coordinates all services to process cricket videos
"""

import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path
from datetime import datetime

from services.video_processor import VideoProcessor, VideoWriter, validate_video_file
from services.pose_detector import get_pose_detector, create_new_metrics_computer
from services.bat_detector import get_bat_detector, create_new_bat_detector
from services.tracker import create_new_tracker
from services.metrics_computer import create_new_metrics_computer
from services.event_classifier import create_new_event_classifier
from services.visualizer import get_visualizer
from services.report_generator import get_report_generator

from models.schemas import (
    VideoMetadata,
    ProcessingMetadata,
    Player,
    Event,
    FramePoseData,
    AnalysisResult
)
from config import settings


class ProcessingPipeline:
    """
    Main processing pipeline for cricket pose analysis
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize pipeline

        Args:
            progress_callback: Optional callback for progress updates
                              Signature: callback(progress: float, message: str, data: dict)
        """
        self.progress_callback = progress_callback

        # Initialize services
        self.pose_detector = get_pose_detector()
        self.bat_detector = get_bat_detector()
        self.visualizer = get_visualizer()
        self.report_generator = get_report_generator()

        # Per-video instances (created in process())
        self.tracker = None
        self.metrics_computer = None
        self.event_classifier = None

        print("🚀 ProcessingPipeline initialized")
        if self.bat_detector.enabled:
            print("   ✅ Bat detection: ENABLED")
        else:
            print("   ⚠️  Bat detection: DISABLED")

    def process(
        self,
        video_path: str,
        output_dir: str,
        job_id: str
    ) -> AnalysisResult:
        """
        Process a cricket video end-to-end

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            job_id: Unique job identifier

        Returns:
            AnalysisResult object
        """
        start_time = time.time()

        # Validate video
        self._update_progress(0, "Validating video file...")
        is_valid, error_msg = validate_video_file(video_path)
        if not is_valid:
            raise ValueError(f"Invalid video file: {error_msg}")

        # Create per-video service instances
        self.tracker = create_new_tracker()
        self.metrics_computer = create_new_metrics_computer()
        self.event_classifier = create_new_event_classifier()

        # Open video
        self._update_progress(5, "Loading video...")
        video_processor = VideoProcessor(video_path)

        video_info = video_processor.info
        print(f"📹 Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps")
        print(f"   Duration: {video_info.duration_seconds:.1f}s, Frames: {video_info.total_frames}")

        # Prepare output paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_path = output_path / f"{job_id}_report.json"
        csv_path = output_path / f"{job_id}_timeseries.csv"
        events_csv_path = output_path / f"{job_id}_events.csv"
        video_path_out = output_path / f"{job_id}_annotated.mp4"

        # Initialize video writer
        video_writer = None
        if settings.EXPORT_ANNOTATED_VIDEO:
            self._update_progress(10, "Initializing video writer...")
            video_writer = VideoWriter(
                str(video_path_out),
                video_info.width,
                video_info.height,
                settings.OUTPUT_VIDEO_FPS
            )

        # Storage for results
        all_pose_data = []
        all_bat_data = []  # Store bat detections
        active_events = {}  # player_id -> event_label

        # Process frames
        self._update_progress(15, "Processing frames...")

        frame_count = 0
        processed_frames = 0

        for batch in video_processor.get_batch_frames(
            batch_size=settings.BATCH_SIZE,
            sample_rate=None  # Uses config
        ):
            # Extract frames from batch
            frame_nums = [fn for fn, _ in batch]
            frames = [f for _, f in batch]

            # Pose detection
            detections_batch = self.pose_detector.detect_batch(frames)
            
            # Bat detection (parallel with pose detection)
            bat_detections_batch = []
            if self.bat_detector.enabled:
                bat_detections_batch = self.bat_detector.detect_batch(frames)
            else:
                bat_detections_batch = [[] for _ in frames]

            # Process each frame in batch
            for i, (frame_num, frame) in enumerate(batch):
                detections = detections_batch[i]
                bat_detections = bat_detections_batch[i]

                # Player tracking
                tracked_detections = self.tracker.update(detections, frame_num)

                # Metrics computation
                enriched_detections = self.metrics_computer.compute_frame_metrics(
                    tracked_detections,
                    frame_num,
                    video_info.fps
                )

                # Event classification
                new_events = self.event_classifier.process_frame(
                    enriched_detections,
                    frame_num,
                    video_info.fps
                )

                # Update active events for visualization
                for event in new_events:
                    active_events[event.player_id] = event.shot_type or event.event_type

                # Store pose data
                frame_data = FramePoseData(
                    frame=frame_num,
                    time_seconds=frame_num / video_info.fps,
                    detections=enriched_detections
                )
                all_pose_data.append(frame_data)
                
                # Store bat detection data
                if bat_detections:
                    bat_frame_data = {
                        "frame": frame_num,
                        "time_seconds": frame_num / video_info.fps,
                        "bats": [bat.to_dict() for bat in bat_detections]
                    }
                    all_bat_data.append(bat_frame_data)

                # Annotate frame
                if video_writer:
                    frame_info = f"Frame {frame_num}/{video_info.total_frames} | " \
                                f"Time: {frame_num / video_info.fps:.1f}s"

                    annotated_frame = self.visualizer.annotate_frame(
                        frame,
                        enriched_detections,
                        active_events,
                        frame_info
                    )
                    
                    # Draw bat detections on top
                    if bat_detections and self.bat_detector.enabled:
                        annotated_frame = self.bat_detector.visualize_detections(
                            annotated_frame,
                            bat_detections
                        )

                    video_writer.write_frame(annotated_frame)

                processed_frames += 1

            # Update progress
            frame_count += len(batch)
            progress = 15 + (70 * frame_count / video_info.total_frames)
            fps = frame_count / (time.time() - start_time)

            self._update_progress(
                progress,
                f"Processing frames... ({frame_count}/{video_info.total_frames})",
                {
                    "current_frame": frame_count,
                    "total_frames": video_info.total_frames,
                    "fps": fps
                }
            )

        # Finalize events
        self._update_progress(85, "Finalizing events...")
        final_events = self.event_classifier.finalize_all_events(
            video_info.total_frames - 1,
            video_info.fps
        )

        all_events = self.event_classifier.get_all_events()
        print(f"🎯 Detected {len(all_events)} events")

        # Get player statistics
        self._update_progress(88, "Computing player statistics...")
        player_stats = self.tracker.get_player_statistics()
        players = [
            Player(
                player_id=stats["player_id"],
                first_appearance_frame=stats["first_frame"],
                last_appearance_frame=stats["last_frame"],
                total_detections=stats["total_detections"],
                avg_confidence=stats["avg_confidence"]
            )
            for stats in player_stats.values()
        ]

        print(f"👥 Tracked {len(players)} players")

        # Create metadata
        processing_time = time.time() - start_time

        video_metadata = video_processor.get_metadata(
            Path(video_path).name,
            processed_frames
        )

        processing_metadata = ProcessingMetadata(
            model_used=settings.POSE_MODEL,
            tracker_used=settings.TRACKING_METHOD,
            processing_time_seconds=processing_time,
            gpu_used=settings.USE_GPU,
            timestamp=datetime.utcnow().isoformat()
        )

        # Generate reports
        self._update_progress(90, "Generating reports...")

        # Add bat detection summary to processing metadata
        if all_bat_data:
            print(f"🏏 Detected bats in {len(all_bat_data)} frames")
        
        result = self.report_generator.generate_json_report(
            video_metadata,
            processing_metadata,
            players,
            all_events,
            all_pose_data,
            str(json_path),
            bat_data=all_bat_data  # Pass bat detection data
        )

        if settings.EXPORT_CSV_TIMESERIES:
            self._update_progress(92, "Exporting CSV timeseries...")
            self.report_generator.generate_csv_timeseries(all_pose_data, str(csv_path))

        self._update_progress(94, "Exporting events CSV...")
        self.report_generator.generate_events_csv(all_events, str(events_csv_path))

        # Print summary
        self._update_progress(96, "Generating summary...")
        summary_text = self.report_generator.generate_summary_text(result)
        print("\n" + summary_text)

        # Cleanup
        self._update_progress(98, "Cleaning up...")
        video_processor.cleanup()
        if video_writer:
            video_writer.cleanup()

        # Final progress
        self._update_progress(
            100,
            f"Processing complete! ({processing_time:.1f}s)",
            {"result": result}
        )

        print(f"\n✅ Processing complete in {processing_time:.1f}s")
        print(f"   FPS: {processed_frames / processing_time:.1f}")

        return result

    def _update_progress(
        self,
        progress: float,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback(progress, message, data or {})


def process_video(
    video_path: str,
    output_dir: str,
    job_id: str,
    progress_callback: Optional[Callable] = None
) -> AnalysisResult:
    """
    Convenience function to process a video

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        job_id: Unique job identifier
        progress_callback: Optional progress callback

    Returns:
        AnalysisResult object
    """
    pipeline = ProcessingPipeline(progress_callback)
    return pipeline.process(video_path, output_dir, job_id)
