"""
Video Analysis Job
Worker-side job that processes a queued video
"""

import logging
import os
import socket
import tempfile
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

from worker.inference import get_pose_estimator
from worker.utils.db import (
    get_video_by_id,
    update_video_status,
    update_video_processing_started,
    update_video_progress,
    update_video_metadata,
    update_video_outputs,
    mark_video_completed,
)
from worker.utils.s3 import (
    download_from_s3,
    upload_to_s3,
    get_video_metadata,
    cleanup_temp_file,
)
from database.models import VideoStatus
from config import settings
from services.video_analytics_service import get_video_analytics_service
from services.bat_detector import get_bat_detector
# Helper function to publish video analysis messages to Redis (synchronous)
def _publish_video_analysis_sync(video_id: str, message: dict):
    """
    Publish video analysis message to Redis using synchronous client
    
    Args:
        video_id: Video identifier
        message: Message dictionary to publish
    """
    try:
        import redis
        import json
        from config import settings
        
        # Get Redis connection details
        redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
        
        # Parse Redis URL
        if redis_url.startswith("redis://"):
            parts = redis_url.replace("redis://", "").split("/")
            host_port = parts[0].split(":")
            host = host_port[0] if len(host_port) > 0 else "localhost"
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0
        else:
            host = "localhost"
            port = 6379
            db = 0
        
        # Create synchronous Redis client
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        
        # Publish to channel
        channel = f"video:{video_id}:analysis"
        message_json = json.dumps(message, default=str, ensure_ascii=False)
        result = redis_client.publish(channel, message_json)
        logger.info(f"Published to Redis channel {channel}, {result} subscribers notified")
        
    except Exception as e:
        logger.error(f"Failed to publish to Redis: {e}", exc_info=True)
        # Don't raise - allow job to continue even if Redis publish fails

logger = logging.getLogger(__name__)


def analyze_video_job(video_id: str):
    """
    Worker-side job that processes a queued video.
    
    Steps:
    1. Load video record from DB
    2. Update status to PROCESSING
    3. Download raw video from S3
    4. Extract metadata (duration, fps, width, height)
    5. Run pose model (RTMPose) – placeholder for now
    6. Save keypoints JSON to S3
    7. Generate overlay MP4 and upload it
    8. Update DB with output paths
    9. Set status to SUCCEEDED or FAILED
    
    Args:
        video_id: UUID string of the video to process
    """
    video_path = None
    temp_dir = None
    
    try:
        logger.info(f"Starting video analysis job for video {video_id}")

        # Lazily grab the shared pose estimator (already initialized at worker boot).
        pose_estimator = get_pose_estimator()
        
        # Initialize bat detector if enabled
        bat_detector = get_bat_detector()
        if bat_detector and bat_detector.enabled:
            logger.info("Bat detection enabled, will detect bats in each frame")
        else:
            bat_detector = None
            logger.info("Bat detection disabled or not available")
        
        # Step 1: Load video record from DB
        video = get_video_by_id(video_id)
        if not video:
            raise ValueError(f"Video not found: {video_id}")
        
        storage_type = getattr(video, 'storage_type', 's3')  # Default to s3 for backward compatibility
        storage_path = video.local_file_path if storage_type == 'local' else video.s3_raw_key
        logger.info(f"Loaded video record: storage_type={storage_type}, path={storage_path}, status: {video.status.value}")
        
        # Step 2: Update status to PROCESSING
        worker_id = socket.gethostname()
        if not update_video_processing_started(video_id, worker_id):
            raise RuntimeError("Failed to update video status to PROCESSING")
        
        update_video_progress(video_id, 5)
        
        # Step 3: Get video file (local or download from S3)
        if storage_type == 'local':
            # Use local file directly
            if not video.local_file_path or not os.path.exists(video.local_file_path):
                raise FileNotFoundError(f"Video file not found at local path: {video.local_file_path}")
            video_path = video.local_file_path
            logger.info(f"Using local video file: {video_path}")
            temp_dir = None  # No temp directory needed for local files
        else:
            # Download from S3
            logger.info(f"Downloading video from S3: {video.s3_raw_key}")
            temp_dir = tempfile.mkdtemp(prefix=f"video_{video_id}_")
            video_path = os.path.join(temp_dir, os.path.basename(video.s3_raw_key) or "video.mp4")
            download_from_s3(video.s3_raw_key, video_path)
        
        update_video_progress(video_id, 15)
        
        # Step 4: Extract metadata (duration, fps, width, height)
        logger.info("Extracting video metadata")
        metadata = get_video_metadata(video_path)

        # Open video for full-frame processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or (metadata.get("fps") if metadata else None)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (metadata.get("width") if metadata else None)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (metadata.get("height") if metadata else None)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

        # Update DB metadata
        update_video_metadata(
            video_id,
            duration_seconds=metadata.get("duration_seconds") if metadata else None,
            original_fps=fps,
            width=width,
            height=height,
        )

        logger.info(
            "Video opened for analysis: %sx%s @ %s fps, total_frames=%s",
            width, height, fps, total_frames,
        )

        update_video_progress(video_id, 25)

        # Step 5: Run pose model (RTMPose) and bat detection over all frames
        logger.info("Running RTMPose and bat detection over video frames")
        frame_index = 0
        processed_frames = 0
        total_persons = 0
        total_bats = 0
        frames_results = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Pose detection
                pose_results = pose_estimator.infer(frame)
                
                # Bat detection (if enabled)
                bat_detections = []
                if bat_detector:
                    bat_detections = bat_detector.detect(frame)

                # Log basic summary similar to test_pose_inference.py
                if pose_results:
                    avg_conf = float(
                        sum(r.mean_confidence for r in pose_results) / len(pose_results)
                    )
                    logger.info(
                        "Frame %d -> %d persons | mean conf=%.3f | %d bats",
                        frame_index,
                        len(pose_results),
                        avg_conf,
                        len(bat_detections),
                    )
                else:
                    logger.info("Frame %d -> no detections | %d bats", frame_index, len(bat_detections))

                # Build per-frame JSON structure with both persons and bats
                frame_data = {
                    "frame_number": frame_index,
                    "num_persons": len(pose_results),
                    "persons": [],
                    "num_bats": len(bat_detections),
                    "bats": [],
                }
                
                # Add person data
                for i, result in enumerate(pose_results):
                    person_data = {
                        "person_id": i,
                        "keypoints": result.keypoints.tolist(),
                        "scores": result.scores.tolist(),
                        "bbox": result.bbox,
                        "mean_confidence": float(result.mean_confidence),
                    }
                    frame_data["persons"].append(person_data)
                
                # Add bat data
                for i, bat_det in enumerate(bat_detections):
                    bat_data = {
                        "bat_id": i,
                        **bat_det.to_dict(),
                    }
                    frame_data["bats"].append(bat_data)

                frames_results.append(frame_data)

                total_persons += len(pose_results)
                total_bats += len(bat_detections)
                processed_frames += 1
                frame_index += 1

                # Rough progress update based on frames
                if total_frames:
                    progress = 25 + int(25 * processed_frames / max(total_frames, 1))
                    update_video_progress(video_id, min(progress, 50))

        finally:
            cap.release()

        # Assemble keypoints_data to match test_pose_inference.py:
        # a list of per-frame dicts with persons, keypoints, scores, bbox, etc.
        keypoints_data = frames_results

        update_video_progress(video_id, 50)
        
        # Step 6: Save keypoints JSON and binary format
        logger.info("Saving keypoints data")
        import json
        import msgpack
        import gzip
        
        # Create output directory
        if storage_type == 'local':
            output_dir = Path(video.local_file_path).parent
        else:
            output_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        
        keypoints_json_path = output_dir / f"{video_id}_keypoints.json"
        keypoints_binary_path = output_dir / f"{video_id}_keypoints.msgpack.gz"
        
        # Save JSON
        with open(keypoints_json_path, 'w') as f:
            json.dump(keypoints_data, f, indent=2)
        
        # Save binary format (msgpack + gzip compression)
        msgpack_data = msgpack.packb(keypoints_data, use_bin_type=True)
        with gzip.open(keypoints_binary_path, 'wb') as f:
            f.write(msgpack_data)
        
        keypoints_size = os.path.getsize(keypoints_json_path)
        binary_size = os.path.getsize(keypoints_binary_path)
        logger.info(f"Keypoints JSON size: {keypoints_size} bytes, Binary size: {binary_size} bytes")
        
        # Store based on storage type
        if storage_type == 'local':
            # Store locally
            update_video_outputs(
                video_id,
                keypoints_local_path=str(keypoints_binary_path),  # Store binary path
                analysis_binary_path=str(keypoints_binary_path),
            )
        else:
            # Upload to S3
            keypoints_s3_key = f"keypoints/{video_id}/keypoints.json"
            keypoints_binary_s3_key = f"keypoints/{video_id}/keypoints.msgpack.gz"
            
            upload_to_s3(str(keypoints_json_path), keypoints_s3_key, "application/json")
            upload_to_s3(str(keypoints_binary_path), keypoints_binary_s3_key, "application/octet-stream")
            
            if keypoints_size <= 1024 * 1024:  # <= 1MB, can store inline
                with open(keypoints_json_path, 'r') as f:
                    keypoints_jsonb = json.load(f)
                update_video_outputs(
                    video_id,
                    keypoints_s3_key=keypoints_s3_key,
                    keypoints_jsonb=keypoints_jsonb,
                )
            else:
                update_video_outputs(video_id, keypoints_s3_key=keypoints_s3_key)
        
        # Publish keypoints to Redis for WebSocket delivery
        _publish_video_analysis_sync(
            video_id,
            {
                "type": "keypoints",
                "video_id": video_id,
                "data": keypoints_data,
                "done": False
            }
        )
        
        update_video_progress(video_id, 70)
        
        # Step 7: Generate overlay MP4 and upload it
        # TODO: Implement video overlay generation
        logger.info("Generating overlay video (placeholder)")
        # This is where you would:
        # - Create overlay video with skeleton/keypoints drawn
        # - Render using OpenCV or similar
        # - Save as MP4
        
        # Placeholder: For now, we'll skip overlay generation
        # Uncomment and implement when ready:
        # overlay_video_path = os.path.join(temp_dir, f"{video_id}_overlay.mp4")
        # generate_overlay_video(video_path, keypoints_data, overlay_video_path)
        # overlay_s3_key = f"overlays/{video_id}/overlay.mp4"
        # upload_to_s3(overlay_video_path, overlay_s3_key, "video/mp4")
        # update_video_outputs(video_id, overlay_video_s3_key=overlay_s3_key)
        
        # For now, set overlay_s3_key to None
        overlay_s3_key = None
        
        update_video_progress(video_id, 85)
        
        # Step 8: Compute and store high-level pose metrics for Bedrock / UI
        from services.video_pose_metrics import compute_video_pose_metrics
        from services.shot_classifier import get_shot_classifier

        logger.info("Computing aggregated pose metrics for video %s", video_id)
        metrics = compute_video_pose_metrics(keypoints_data=keypoints_data, video_id=video_id)

        # Classify shot type from aggregated metrics
        shot_classification = None
        try:
            logger.info("Classifying shot type for video %s", video_id)
            shot_classifier = get_shot_classifier()
            shot_classification = shot_classifier.classify_shot_from_metrics(metrics)
            logger.info(
                "Shot classification for video %s: %s (confidence: %.2f)",
                video_id,
                shot_classification.get("primary_shot", "unknown"),
                shot_classification.get("confidence", 0.0),
            )
        except Exception as e:
            logger.error(
                "Failed to classify shot type for video %s: %s",
                video_id,
                e,
                exc_info=True,
            )

        # Optionally, generate Bedrock analytics immediately after metrics are ready.
        # This allows the frontend to fetch a pre-computed analytics JSON instead of
        # invoking Bedrock on every request.
        analytics = None
        if settings.BEDROCK_REGION and settings.BEDROCK_MODEL_ID:
            try:
                logger.info("Generating Bedrock analytics for video %s", video_id)
                analytics_service = get_video_analytics_service()
                analytics = analytics_service.generate_analytics(
                    video_id=str(video_id),
                    metrics=metrics,
                )
                logger.info("Successfully generated Bedrock analytics for video %s", video_id)
            except Exception as e:
                logger.error(
                    "Failed to generate Bedrock analytics for video %s: %s",
                    video_id,
                    e,
                    exc_info=True,
                )

        # If analytics were generated, embed them under 'bedrock_analytics' so both
        # raw metrics and AI commentary are available from a single JSON blob.
        metrics_payload = dict(metrics)
        if shot_classification is not None:
            metrics_payload["shot_classification"] = shot_classification
        if analytics is not None:
            metrics_payload["bedrock_analytics"] = analytics

        update_video_outputs(video_id, metrics_jsonb=metrics_payload)
        update_video_progress(video_id, 95)
        
        # Publish bedrock analysis to Redis for WebSocket delivery
        if analytics is not None:
            _publish_video_analysis_sync(
                video_id,
                {
                    "type": "bedrock_analysis",
                    "video_id": video_id,
                    "data": analytics,
                    "done": True
                }
            )
        
        # Step 9: Set status to SUCCEEDED
        if mark_video_completed(video_id):
            logger.info(f"Successfully completed video analysis for {video_id}")
        else:
            raise RuntimeError("Failed to mark video as completed")
        
        update_video_progress(video_id, 100)
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        
        # Update status to FAILED with error message
        error_message = str(e)
        update_video_status(video_id, VideoStatus.FAILED, error_message)

        raise  # Re-raise to mark RQ job as failed

    finally:
        # Cleanup temporary files (only if downloaded from S3)
        video = get_video_by_id(video_id) if video_id else None
        storage_type = getattr(video, 'storage_type', 's3') if video else 's3'

        if storage_type == 's3':
            # Only cleanup if we downloaded from S3
            if video_path and os.path.exists(video_path) and temp_dir:
                cleanup_temp_file(video_path)

            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


async def analyze_video_job_async(ctx, video_id: str, *args, **kwargs):
    """
    ARQ-compatible wrapper that runs the existing synchronous analyze_video_job
    in a thread executor so that heavy CPU/IO work doesn't block the event loop.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, analyze_video_job, video_id)