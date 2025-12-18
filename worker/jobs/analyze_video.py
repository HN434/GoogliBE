"""
Video Analysis Job
Worker-side job that processes a queued video
"""

import logging
import os
import socket
import tempfile
import threading
from typing import Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

from worker.inference import (
    get_pose_estimator,
    PoseEstimatorResult,
    get_shot_classifier,
    classify_shot_video,
    SHOT_CLASSES,
)
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
from services.video_processor import VideoProcessor, VideoWriter
from typing import List, Dict, Any
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
        password = None
        
        # Parse Redis URL
        if redis_url.startswith("redis://"):
            # Handle URL with password: redis://:password@host:port/db
            url_without_protocol = redis_url.replace("redis://", "")
            
            # Check if password is in URL
            if "@" in url_without_protocol:
                # Format: :password@host:port/db or username:password@host:port/db
                auth_and_rest = url_without_protocol.split("@")
                auth_part = auth_and_rest[0]
                rest = auth_and_rest[1]
                
                # Extract password (format: :password or username:password)
                if ":" in auth_part:
                    password = auth_part.split(":")[-1] if auth_part.startswith(":") else auth_part.split(":")[1]
                
                # Parse host, port, db from rest
                parts = rest.split("/")
                host_port = parts[0].split(":")
                host = host_port[0] if len(host_port) > 0 else "localhost"
                port = int(host_port[1]) if len(host_port) > 1 else 6379
                db = int(parts[1]) if len(parts) > 1 else 0
            else:
                # No password in URL, parse normally
                parts = url_without_protocol.split("/")
                host_port = parts[0].split(":")
                host = host_port[0] if len(host_port) > 0 else "localhost"
                port = int(host_port[1]) if len(host_port) > 1 else 6379
                db = int(parts[1]) if len(parts) > 1 else 0
                
                # Use password from settings if not in URL (for server environments)
                if settings.REDIS_PASSWORD:
                    password = settings.REDIS_PASSWORD
        else:
            host = "localhost"
            port = 6379
            db = 0
            # Use password from settings if provided (for server environments)
            if settings.REDIS_PASSWORD:
                password = settings.REDIS_PASSWORD
        
        # Create synchronous Redis client
        redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,  # None if not set (local), password if set (server)
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


def _convert_keypoints_data_to_results(frame_data: Dict[str, Any]) -> List[PoseEstimatorResult]:
    """
    Convert keypoints data from JSON format to PoseEstimatorResult format.
    
    Args:
        frame_data: Frame data dictionary with 'persons' list
        
    Returns:
        List of PoseEstimatorResult objects
    """
    results = []
    
    for person_data in frame_data.get("persons", []):
        keypoints = np.array(person_data.get("keypoints", []), dtype=np.float32)
        scores = np.array(person_data.get("scores", []), dtype=np.float32)
        bbox = person_data.get("bbox", [0, 0, 0, 0])
        
        if len(keypoints) > 0 and len(scores) > 0:
            results.append(
                PoseEstimatorResult(
                    keypoints=keypoints,
                    scores=scores,
                    bbox=bbox,
                )
            )
    
    return results


def _draw_skeletons_on_frame(
    frame: np.ndarray,
    results: List[PoseEstimatorResult]
) -> np.ndarray:
    """
    Draw skeletons for all detected persons on a frame.
    Each person gets a different color scheme for visibility.
    
    Args:
        frame: Input frame (BGR format)
        results: List of PoseEstimatorResult for each detected person
        
    Returns:
        Annotated frame with skeletons drawn
    """
    # COCO 17 keypoint connections (skeleton) with color assignments
    skeleton_connections = [
        # Left arm (5: left shoulder, 7: left elbow, 9: left wrist)
        ([5, 7], 'left_arm'), ([7, 9], 'left_arm'),
        # Right arm (6: right shoulder, 8: right elbow, 10: right wrist)
        ([6, 8], 'right_arm'), ([8, 10], 'right_arm'),
        # Torso (shoulders and hips)
        ([5, 6], 'torso'),  # Shoulders
        ([11, 12], 'torso'),  # Hips
        ([5, 11], 'torso'),  # Left shoulder to left hip
        ([6, 12], 'torso'),  # Right shoulder to right hip
        # Left leg (11: left hip, 13: left knee, 15: left ankle)
        ([11, 13], 'left_leg'), ([13, 15], 'left_leg'),
        # Right leg (12: right hip, 14: right knee, 16: right ankle)
        ([12, 14], 'right_leg'), ([14, 16], 'right_leg'),
    ]
    
    vis_frame = frame.copy()
    
    # Define different color schemes for each person
    person_color_schemes = [
        # Person 1: Original colors
        {
            'head': (255, 255, 0),      # Cyan
            'left_arm': (0, 0, 255),    # Red
            'right_arm': (0, 165, 255), # Orange
            'torso': (0, 255, 0),       # Green
            'left_leg': (255, 0, 255),  # Magenta
            'right_leg': (128, 0, 128), # Purple
        },
        # Person 2: Brighter variants
        {
            'head': (255, 255, 100),    # Light Cyan
            'left_arm': (100, 100, 255), # Light Red
            'right_arm': (100, 200, 255), # Light Orange
            'torso': (100, 255, 100),   # Light Green
            'left_leg': (255, 100, 255), # Light Magenta
            'right_leg': (200, 100, 200), # Light Purple
        },
        # Person 3: Darker variants
        {
            'head': (200, 200, 0),      # Dark Cyan
            'left_arm': (0, 0, 200),    # Dark Red
            'right_arm': (0, 120, 200), # Dark Orange
            'torso': (0, 200, 0),       # Dark Green
            'left_leg': (200, 0, 200),  # Dark Magenta
            'right_leg': (100, 0, 100), # Dark Purple
        },
        # Person 4+: Cycle through variants
        {
            'head': (150, 255, 150),    # Greenish Cyan
            'left_arm': (200, 150, 0),  # Orange Red
            'right_arm': (0, 200, 255), # Cyan Orange
            'torso': (150, 150, 255),   # Blue Green
            'left_leg': (255, 150, 0), # Yellow Magenta
            'right_leg': (150, 0, 255), # Blue Purple
        },
    ]
    
    for person_idx, result in enumerate(results):
        # Get color scheme for this person (cycle if more than 4 persons)
        color_scheme = person_color_schemes[person_idx % len(person_color_schemes)]
        keypoints = result.keypoints  # Shape: (17, 2) or (17, 3)
        scores = result.scores  # Shape: (17,)
        
        # Extract x, y coordinates (ignore z if present)
        if keypoints.shape[1] >= 2:
            kpts_2d = keypoints[:, :2]
        else:
            continue
        
        # Calculate bounding box size for scaling visualization elements
        bbox_size = 200.0  # Default reference size
        if result.bbox and len(result.bbox) >= 4:
            bbox_width = result.bbox[2] - result.bbox[0]
            bbox_height = result.bbox[3] - result.bbox[1]
            bbox_avg_dim = (bbox_width + bbox_height) / 2.0
            bbox_size = max(bbox_avg_dim, 50.0)  # Minimum 50px
        
        # Scale visualization elements based on bbox size
        scale_factor = min(bbox_size / 200.0, 1.0)  # Cap at 1.0
        line_thickness = max(int(4 * scale_factor), 2)  # Minimum 2px
        outer_radius = max(int(10 * scale_factor), 4)  # Minimum 4px
        inner_radius = max(int(8 * scale_factor), 3)  # Minimum 3px
        
        # Draw head to shoulder center connection (special case)
        if len(kpts_2d) > 6:
            left_shoulder = kpts_2d[5]  # Left shoulder
            right_shoulder = kpts_2d[6]  # Right shoulder
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                             (left_shoulder[1] + right_shoulder[1]) / 2)
            nose = kpts_2d[0]  # Nose
            
            score_nose = scores[0] if 0 < len(scores) else 0.0
            score_left_shoulder = scores[5] if 5 < len(scores) else 0.0
            score_right_shoulder = scores[6] if 6 < len(scores) else 0.0
            
            # Draw if nose and at least one shoulder has reasonable confidence
            if score_nose > 0.3 and (score_left_shoulder > 0.3 or score_right_shoulder > 0.3):
                nose_pt = tuple(nose.astype(int))
                center_pt = tuple(int(c) for c in shoulder_center)
                head_color = color_scheme['head']
                cv2.line(vis_frame, nose_pt, center_pt, head_color, line_thickness)
        
        # Draw skeleton connections with color coding
        for connection, color_name in skeleton_connections:
            idx1, idx2 = connection[0], connection[1]
            if 0 <= idx1 < len(kpts_2d) and 0 <= idx2 < len(kpts_2d):
                pt1 = tuple(kpts_2d[idx1].astype(int))
                pt2 = tuple(kpts_2d[idx2].astype(int))
                score1 = scores[idx1] if idx1 < len(scores) else 0.5
                score2 = scores[idx2] if idx2 < len(scores) else 0.5
                
                # Only draw if both keypoints have reasonable confidence
                if score1 > 0.3 and score2 > 0.3:
                    color = color_scheme.get(color_name, (255, 255, 255))
                    cv2.line(vis_frame, pt1, pt2, color, line_thickness)
        
        # Draw keypoints - skip eye and ear keypoints
        head_keypoints_to_skip = [1, 2, 3, 4]  # Eyes and ears
        
        for i, (x, y) in enumerate(kpts_2d):
            if i in head_keypoints_to_skip:
                continue
            
            score = scores[i] if i < len(scores) else 0.0
            if score > 0.3:  # Only draw visible keypoints
                pt = (int(x), int(y))
                # Map keypoint index to body part
                if i == 0:  # Nose (head)
                    base_color = color_scheme['head']
                elif i in [5, 7, 9]:  # Left arm
                    base_color = color_scheme['left_arm']
                elif i in [6, 8, 10]:  # Right arm
                    base_color = color_scheme['right_arm']
                elif i in [11, 12]:  # Hips (torso)
                    base_color = color_scheme['torso']
                elif i in [13, 15]:  # Left leg
                    base_color = color_scheme['left_leg']
                elif i in [14, 16]:  # Right leg
                    base_color = color_scheme['right_leg']
                else:
                    base_color = (255, 255, 255)  # Default white
                
                # Darken color based on confidence
                if score > 0.7:
                    color = base_color
                else:
                    color = tuple(int(c * 0.7) for c in base_color)
                
                # Draw circles: outer circle (dark border), inner circle (filled)
                cv2.circle(vis_frame, pt, outer_radius, (0, 0, 0), -1)  # Black outer
                cv2.circle(vis_frame, pt, inner_radius, color, -1)  # Colored inner
    
    return vis_frame


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

        # Start Pegasus analysis in the background as early as possible so it does
        # not block download or pose extraction.
        pegasus_analytics_container: Dict[str, Any] = {"value": None}
        pegasus_thread: Optional[threading.Thread] = None
        # Shot classification background thread (initialized later once video_path is known)
        shot_thread: Optional[threading.Thread] = None

        def _run_pegasus_analysis():
            """
            Background task that generates Pegasus analytics.
            It logs and publishes errors but never raises, so it cannot block
            keypoint extraction or cause the whole job to fail.
            """
            nonlocal pegasus_analytics_container

            if not (settings.BEDROCK_REGION and (settings.BEDROCK_PEGASUS_MODEL_ID or settings.BEDROCK_MODEL_ID)):
                logger.info("Pegasus analysis is not configured; skipping.")
                return

            try:
                logger.info("Generating Pegasus analysis for video %s via Bedrock", video_id)
                from services.video_analytics_service import get_pegasus_service

                pegasus_service = get_pegasus_service()

                # Generate S3 URI for Pegasus (Pegasus uses S3 URIs, not presigned URLs)
                if storage_type == "s3" and video.s3_raw_key and video.s3_bucket:
                    s3_uri = f"s3://{video.s3_bucket}/{video.s3_raw_key}"
                elif storage_type == "local" and video.local_file_path:
                    # For local storage, we'd need to upload to S3 first or use a different approach
                    logger.warning("Pegasus analysis requires S3 storage. Skipping for local video.")
                    s3_uri = None
                else:
                    s3_uri = None

                if not s3_uri:
                    logger.info("No valid S3 URI available for Pegasus; skipping analysis.")
                    return

                # Call Pegasus via Bedrock
                pegasus_analytics = pegasus_service.generate_pegasus_analytics(
                    video_id=str(video_id),
                    s3_uri=s3_uri,
                    s3_bucket=video.s3_bucket,
                )

                pegasus_analytics_container["value"] = pegasus_analytics

                logger.info(
                    "Successfully generated Pegasus analysis for video %s: %d shots detected",
                    video_id,
                    pegasus_analytics.get("total_shots", 0),
                )

                # Publish Pegasus analysis to Redis for WebSocket delivery
                _publish_video_analysis_sync(
                    video_id,
                    {
                        "type": "pegasus_analysis",
                        "video_id": video_id,
                        "data": pegasus_analytics,
                        "done": False,
                    },
                )

            except Exception as e:
                logger.error(
                    "Failed to generate Pegasus analysis for video %s: %s",
                    video_id,
                    e,
                    exc_info=True,
                )
                # Simple, generic Pegasus error – we no longer depend on video duration.
                error_code = "pegasus_error"
                error_message = (
                    "Googli AI could not analyse this video due to an internal Pegasus error. "
                    "Please try again in a few minutes."
                )

                # Send an error message over the analysis stream so the frontend can
                # show the correct reason without guessing. Do NOT raise.
                try:
                    _publish_video_analysis_sync(
                        video_id,
                        {
                            "type": "error",
                            "video_id": video_id,
                            "message": error_message,
                            "error_code": error_code,
                        },
                    )
                except Exception:
                    logger.exception("Failed to publish Pegasus error for video %s", video_id)

        # Kick off Pegasus analysis immediately (non-blocking).
        pegasus_thread = threading.Thread(target=_run_pegasus_analysis, name=f"pegasus-{video_id}", daemon=True)
        pegasus_thread.start()
        
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
            try:
                download_from_s3(video.s3_raw_key, video_path)
            except Exception as e:
                # Classify timeout vs generic download failure
                msg = str(e) if e else ""
                is_timeout = "timed out" in msg.lower() or "timeout" in msg.lower()
                error_code = "download_timeout" if is_timeout else "download_error"
                error_message = (
                    "Googli AI could not download your video from secure storage due to a network timeout. "
                    "Please try again in a few minutes."
                    if is_timeout
                    else "Googli AI could not download your video from secure storage. "
                         "Please try again; if the problem persists, re-upload the video."
                )

                # Send an error message over the analysis stream so the frontend can stop processing
                try:
                    _publish_video_analysis_sync(
                        video_id,
                        {
                            "type": "error",
                            "video_id": video_id,
                            "message": error_message,
                            "error_code": error_code,
                        },
                    )
                except Exception:
                    logger.exception("Failed to publish download error for video %s", video_id)

                # Mark video as failed via outer handler and abort further analysis
                raise RuntimeError(error_message) from e
        
        update_video_progress(video_id, 15)

        # Kick off shot classification in a background thread so it never blocks
        # core pose extraction or Pegasus analysis.

        def _run_shot_classification(local_video_path: str):
            """
            Background task that runs EfficientNetB0+GRU shot classification.

            Any errors are logged but never raised so that they cannot cause the
            overall video job to fail.
            """
            if not settings.SHOT_CLASSIFICATION_ENABLED:
                logger.info("Shot classification is disabled via settings; skipping.")
                return

            try:
                logger.info("Running shot classification for video %s", video_id)
                model = get_shot_classifier()

                # Use a small, fixed number of frames to keep this lightweight.
                frame_count = 30
                class_name, confidence, probs = classify_shot_video(
                    local_video_path,
                    model=model,
                    frame_count=frame_count,
                    class_labels=SHOT_CLASSES,
                )

                # Publish shot classification to Redis for WebSocket delivery
                try:
                    _publish_video_analysis_sync(
                        video_id,
                        {
                            "type": "shot_classification",
                            "video_id": video_id,
                            "data": {
                                "shot_label": class_name,
                                "confidence_percent": confidence,
                                "probabilities": probs.tolist(),
                                "classes": list(SHOT_CLASSES.keys()),
                            },
                            "done": True,
                        },
                    )
                    logger.info(
                        "Shot classification for video %s: %s (%.2f%%)",
                        video_id,
                        class_name,
                        confidence,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to publish shot classification for video %s: %s",
                        video_id,
                        e,
                        exc_info=True,
                    )

            except FileNotFoundError as e:
                logger.error(
                    "Shot classifier weights not found for video %s: %s", video_id, e
                )
            except Exception as e:
                logger.error(
                    "Error running shot classification for video %s: %s",
                    video_id,
                    e,
                    exc_info=True,
                )
        
        # Once we have video_path and the function defined, start the shot classifier thread.
        try:
            if video_path:
                shot_thread = threading.Thread(
                    target=_run_shot_classification,
                    args=(video_path,),
                    name=f"shot-classifier-{video_id}",
                    daemon=True,
                )
                shot_thread.start()
        except Exception as e:
            logger.error(
                "Failed to start shot classification thread for video %s: %s",
                video_id,
                e,
                exc_info=True,
            )
        
        # Step 4: Extract metadata (duration, fps, width, height)
        logger.info("Extracting video metadata")
        metadata = get_video_metadata(video_path)

        # Initialize VideoProcessor for batch processing (it will extract metadata too)
        video_processor = VideoProcessor(video_path)
        
        # Get metadata from VideoProcessor
        fps = video_processor.info.fps or (metadata.get("fps") if metadata else None)
        width = video_processor.info.width or (metadata.get("width") if metadata else None)
        height = video_processor.info.height or (metadata.get("height") if metadata else None)
        total_frames = video_processor.info.total_frames or None
        video_duration_seconds = video_processor.info.duration_seconds or (metadata.get("duration_seconds") if metadata else None)

        # Update DB metadata
        update_video_metadata(
            video_id,
            duration_seconds=video_duration_seconds,
            original_fps=fps,
            width=width,
            height=height,
        )

        logger.info(
            "Video opened for analysis: %sx%s @ %s fps, total_frames=%s",
            width, height, fps, total_frames,
        )

        update_video_progress(video_id, 25)

        update_video_progress(video_id, 30)

        # Step 6: Run pose model (RTMPose) in batches
        logger.info("Running RTMPose over video frames in batches")
        
        # Use a small warmup batch (size=1) for the very first inference to
        # reduce perceived latency, then switch to the configured batch size
        main_batch_size = settings.BATCH_SIZE or 24
        
        # Import for batch person detection
        from worker.inference.person_detector import get_person_detector
        
        frames_results = []
        processed_frames = 0
        total_persons = 0

        try:
            # Frame generator with sampling to reduce total frames processed
            # Sample every 2nd frame to roughly halve the work while keeping ~15 fps
            sample_rate = 2
            frame_iter = video_processor.frame_generator(sample_rate=sample_rate)

            # Effective total frames for progress calculations (how many frames we actually process)
            if total_frames:
                effective_total_frames = (total_frames + sample_rate - 1) // sample_rate
            else:
                effective_total_frames = None

            # Get person detector for batch detection
            person_detector = get_person_detector()

            # Skip warmup - start processing immediately with full batches for better GPU utilization

            # ----- Main loop: use configured batch size for remaining frames -----
            batch: list = []
            
            # Process frames sequentially - GPU operations are synchronous and don't benefit from threading
            # ThreadPoolExecutor adds overhead for GPU-bound tasks
            
            for frame_num, frame in frame_iter:
                batch.append((frame_num, frame))

                if len(batch) >= main_batch_size:
                    frame_nums = [fn for fn, _ in batch]
                    frames = [f for _, f in batch]

                    # OPTIMIZATION 1: Batch person detection (all frames at once on GPU)
                    # This detects persons on every frame, but processes them in one GPU call
                    detected_bboxes_batch = person_detector.detect_batch(frames)
                    
                    # OPTIMIZATION 2: Use PoseEstimator infer_batch helper to keep
                    # batched processing in a single call (ready for future true batching).
                    pose_results_batch = pose_estimator.infer_batch(
                        frames,
                        person_bboxes_list=detected_bboxes_batch,
                        auto_detect=False,
                    )

                    # Process each frame in the batch (optimized - minimal logging)
                    for frame_num, pose_results in zip(frame_nums, pose_results_batch):
                        # Build per-frame JSON structure (optimized)
                        frame_data = {
                            "frame_number": frame_num,
                            "num_persons": len(pose_results),
                            "persons": [
                                {
                                    "person_id": i,
                                    "keypoints": result.keypoints.tolist(),
                                    "scores": result.scores.tolist(),
                                    "bbox": result.bbox,
                                    "mean_confidence": float(result.mean_confidence),
                                }
                                for i, result in enumerate(pose_results)
                            ],
                            "num_bats": 0,
                            "bats": [],
                        }

                        frames_results.append(frame_data)
                        total_persons += len(pose_results)
                        processed_frames += 1
                    
                    # Log only once per batch (not per frame) for performance
                    if processed_frames % (main_batch_size * 4) == 0 or (
                        effective_total_frames and processed_frames >= effective_total_frames
                    ):
                        logger.info(
                            f"Processed {processed_frames}/{effective_total_frames or 'unknown'} sampled frames, "
                            f"{total_persons} total persons detected"
                        )

                    # Progress update less frequently (every 2 batches) to reduce overhead
                    if effective_total_frames and processed_frames % (main_batch_size * 2) == 0:
                        progress = 30 + int(25 * processed_frames / max(effective_total_frames, 1))
                        update_video_progress(video_id, min(progress, 55))

                    # Clear GPU cache periodically to prevent memory fragmentation
                    if settings.CLEAR_GPU_CACHE and processed_frames % (main_batch_size * 2) == 0:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.debug("Cleared GPU cache")

                    batch = []

            # Process any remaining frames that didn't fill a full batch
            if batch:
                frame_nums = [fn for fn, _ in batch]
                frames = [f for _, f in batch]

                # Batch person detection for remaining frames
                detected_bboxes_batch = person_detector.detect_batch(frames)
                
                # Pose estimation for remaining frames (batched helper)
                pose_results_batch = pose_estimator.infer_batch(
                    frames,
                    person_bboxes_list=detected_bboxes_batch,
                    auto_detect=False,
                )

                # Process remaining frames (optimized - minimal logging)
                for frame_num, pose_results in zip(frame_nums, pose_results_batch):
                    frame_data = {
                        "frame_number": frame_num,
                        "num_persons": len(pose_results),
                        "persons": [
                            {
                                "person_id": i,
                                "keypoints": result.keypoints.tolist(),
                                "scores": result.scores.tolist(),
                                "bbox": result.bbox,
                                "mean_confidence": float(result.mean_confidence),
                            }
                            for i, result in enumerate(pose_results)
                        ],
                        "num_bats": 0,
                        "bats": [],
                    }

                    frames_results.append(frame_data)
                    total_persons += len(pose_results)
                    processed_frames += 1

                if effective_total_frames:
                    progress = 30 + int(25 * processed_frames / max(effective_total_frames, 1))
                    update_video_progress(video_id, min(progress, 55))

        finally:
            video_processor.cleanup()

        # Assemble keypoints_data to match test_pose_inference.py:
        # a list of per-frame dicts with persons, keypoints, scores, bbox, etc.
        keypoints_data = frames_results

        update_video_progress(video_id, 55)

        # Publish full keypoints dataset to Redis for WebSocket delivery
        # Use background thread to avoid blocking
        def _publish_keypoints_async():
            try:
                _publish_video_analysis_sync(
                    video_id,
                    {
                        "type": "keypoints",
                        "video_id": video_id,
                        "data": keypoints_data,
                        "done": True,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to publish keypoints to Redis: {e}", exc_info=True)
        
        # Publish in background thread to not block completion
        publish_thread = threading.Thread(target=_publish_keypoints_async, daemon=True)
        publish_thread.start()
        
        update_video_progress(video_id, 95)
        
        # Wait for Pegasus analytics if available (non-blocking)
        if pegasus_thread is not None and pegasus_thread.is_alive():
            pegasus_thread.join(timeout=0)

        pegasus_analytics = pegasus_analytics_container.get("value")
        
        # Store only Pegasus analytics if available (no pose metrics)
        if pegasus_analytics is not None:
            update_video_outputs(video_id, metrics_jsonb={"pegasus_analytics": pegasus_analytics})
            logger.info("Stored Pegasus analytics for video %s", video_id)
        
        # Step 10: Set status to SUCCEEDED
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