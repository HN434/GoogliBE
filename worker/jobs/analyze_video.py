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

from worker.inference import get_pose_estimator, PoseEstimatorResult
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


def _generate_overlay_video(
    video_path: str,
    keypoints_data: List[Dict[str, Any]],
    output_path: str,
    progress_callback: Optional[callable] = None
) -> None:
    """
    Generate overlay video with skeletons drawn for all detected persons.
    
    Args:
        video_path: Path to input video
        keypoints_data: List of frame data dictionaries with keypoints
        output_path: Path to output overlay video
        progress_callback: Optional callback function(progress_percent) for progress updates
    """
    logger.info(f"Generating overlay video: {video_path} -> {output_path}")
    
    # Create a dictionary mapping frame numbers to keypoints data for quick lookup
    frame_data_map = {frame_data["frame_number"]: frame_data for frame_data in keypoints_data}
    
    # Open input video
    video_processor = VideoProcessor(video_path)
    width = video_processor.info.width
    height = video_processor.info.height
    fps = video_processor.info.fps
    total_frames = video_processor.info.total_frames
    
    logger.info(f"Video info: {width}x{height} @ {fps} fps, {total_frames} frames")
    
    # Create output video writer
    video_writer = VideoWriter(output_path, width, height, fps)
    
    try:
        frame_count = 0
        
        # Process all frames from the video
        for frame_num, frame in video_processor.frame_generator(sample_rate=1):
            # Look up keypoints data for this frame
            frame_data = frame_data_map.get(frame_num)
            
            if frame_data and frame_data.get("num_persons", 0) > 0:
                # Convert keypoints data to PoseEstimatorResult format
                pose_results = _convert_keypoints_data_to_results(frame_data)
                num_persons = len(pose_results)
                
                # Draw skeletons for all detected persons
                annotated_frame = _draw_skeletons_on_frame(frame, pose_results)
                
                if num_persons > 1:
                    logger.debug(
                        f"Frame {frame_num}: Drew skeletons for {num_persons} persons (multi-person detection)"
                    )
                else:
                    logger.debug(
                        f"Frame {frame_num}: Drew skeleton for {num_persons} person"
                    )
            else:
                # No detections for this frame, use original frame
                annotated_frame = frame
            
            # Write annotated frame to output video
            video_writer.write_frame(annotated_frame)
            frame_count += 1
            
            # Progress callback
            if progress_callback and total_frames > 0:
                progress = int(100 * frame_count / total_frames)
                progress_callback(progress)
        
        logger.info(f"Successfully generated overlay video: {output_path} ({frame_count} frames)")
        
    finally:
        video_writer.cleanup()
        video_processor.cleanup()


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

        # Initialize VideoProcessor for batch processing (it will extract metadata too)
        video_processor = VideoProcessor(video_path)
        
        # Get metadata from VideoProcessor
        fps = video_processor.info.fps or (metadata.get("fps") if metadata else None)
        width = video_processor.info.width or (metadata.get("width") if metadata else None)
        height = video_processor.info.height or (metadata.get("height") if metadata else None)
        total_frames = video_processor.info.total_frames or None

        # Update DB metadata
        update_video_metadata(
            video_id,
            duration_seconds=video_processor.info.duration_seconds or (metadata.get("duration_seconds") if metadata else None),
            original_fps=fps,
            width=width,
            height=height,
        )

        logger.info(
            "Video opened for analysis: %sx%s @ %s fps, total_frames=%s",
            width, height, fps, total_frames,
        )

        update_video_progress(video_id, 25)

        # Step 5: Generate Pegasus analysis from S3 URL via Bedrock (before keypoints extraction)
        pegasus_analytics = None
        if settings.BEDROCK_REGION and (settings.BEDROCK_PEGASUS_MODEL_ID or settings.BEDROCK_MODEL_ID):
            try:
                logger.info("Generating Pegasus analysis for video %s via Bedrock", video_id)
                from services.video_analytics_service import get_pegasus_service
                from services.s3_service import s3_service
                
                pegasus_service = get_pegasus_service()
                
                # Generate S3 URI for Pegasus (Pegasus uses S3 URIs, not presigned URLs)
                if storage_type == 's3' and video.s3_raw_key and video.s3_bucket:
                    s3_uri = f"s3://{video.s3_bucket}/{video.s3_raw_key}"
                elif storage_type == 'local' and video.local_file_path:
                    # For local storage, we'd need to upload to S3 first or use a different approach
                    # For now, skip Pegasus for local storage
                    logger.warning("Pegasus analysis requires S3 storage. Skipping for local video.")
                    s3_uri = None
                else:
                    s3_uri = None
                
                if s3_uri:
                    # Call Pegasus via Bedrock
                    pegasus_analytics = pegasus_service.generate_pegasus_analytics(
                        video_id=str(video_id),
                        s3_uri=s3_uri,
                        s3_bucket=video.s3_bucket
                    )
                    
                    logger.info(
                        "Successfully generated Pegasus analysis for video %s: %d shots detected",
                        video_id,
                        pegasus_analytics.get("total_shots", 0)
                    )
                    
                    # Publish Pegasus analysis to Redis for WebSocket delivery
                    _publish_video_analysis_sync(
                        video_id,
                        {
                            "type": "pegasus_analysis",
                            "video_id": video_id,
                            "data": pegasus_analytics,
                            "done": False
                        }
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to generate Pegasus analysis for video %s: %s",
                    video_id,
                    e,
                    exc_info=True,
                )
                # Continue with keypoints extraction even if Pegasus fails
        
        update_video_progress(video_id, 30)

        # Step 6: Run pose model (RTMPose) in batches
        logger.info("Running RTMPose over video frames in batches")
        
        batch_size = settings.BATCH_SIZE or 16
        
        frames_results = []
        processed_frames = 0
        total_persons = 0

        try:
            for batch in video_processor.get_batch_frames(batch_size=batch_size, sample_rate=None):
                # Extract frame numbers and frames from batch
                frame_nums = [fn for fn, _ in batch]
                frames = [f for _, f in batch]
                
                # Batch pose detection
                pose_results_batch = pose_estimator.infer_batch(frames, auto_detect=True)
                
                # Process each frame in the batch
                for frame_num, pose_results in zip(frame_nums, pose_results_batch):
                    # Log basic summary
                    if pose_results:
                        avg_conf = float(
                            sum(r.mean_confidence for r in pose_results) / len(pose_results)
                        )
                        num_persons = len(pose_results)
                        if num_persons > 1:
                            logger.info(
                                "Frame %d -> %d persons detected (multi-person) | mean conf=%.3f",
                                frame_num,
                                num_persons,
                                avg_conf,
                            )
                        else:
                            logger.info(
                                "Frame %d -> %d person | mean conf=%.3f",
                                frame_num,
                                num_persons,
                                avg_conf,
                            )
                    else:
                        logger.info("Frame %d -> no detections", frame_num)
                    
                    # Build per-frame JSON structure
                    frame_data = {
                        "frame_number": frame_num,
                        "num_persons": len(pose_results),
                        "persons": [],
                        "num_bats": 0,  # Bat detection disabled
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
                    
                    frames_results.append(frame_data)
                    
                    total_persons += len(pose_results)
                    processed_frames += 1
                
                # Progress update after each batch
                if total_frames:
                    progress = 30 + int(25 * processed_frames / max(total_frames, 1))
                    update_video_progress(video_id, min(progress, 55))
                
                logger.info(
                    f"Processed batch: {len(batch)} frames, total processed: {processed_frames}/{total_frames or 'unknown'}"
                )

        finally:
            video_processor.cleanup()

        # Assemble keypoints_data to match test_pose_inference.py:
        # a list of per-frame dicts with persons, keypoints, scores, bbox, etc.
        keypoints_data = frames_results

        update_video_progress(video_id, 55)
        
        # Step 7: Save keypoints JSON and binary format
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
        
        # Step 8: Generate overlay MP4 and upload it
        logger.info("Generating overlay video with skeletons for all detected persons")
        
        # Create overlay video path
        if storage_type == 'local':
            overlay_video_path = str(Path(video.local_file_path).parent / f"{video_id}_overlay.mp4")
        else:
            overlay_video_path = os.path.join(temp_dir, f"{video_id}_overlay.mp4")
        
        # Generate overlay video with progress updates
        def overlay_progress_callback(progress_percent: int):
            # Map overlay progress (0-100) to overall progress (70-85)
            overall_progress = 70 + int(15 * progress_percent / 100)
            update_video_progress(video_id, min(overall_progress, 85))
        
        try:
            _generate_overlay_video(
                video_path=video_path,
                keypoints_data=keypoints_data,
                output_path=overlay_video_path,
                progress_callback=overlay_progress_callback
            )
            
            # Upload or store overlay video
            if storage_type == 'local':
                # Store locally
                update_video_outputs(
                    video_id,
                    overlay_video_local_path=overlay_video_path,
                )
                overlay_s3_key = None
            else:
                # Upload to S3
                overlay_s3_key = f"overlays/{video_id}/overlay.mp4"
                upload_to_s3(overlay_video_path, overlay_s3_key, "video/mp4")
                update_video_outputs(video_id, overlay_video_s3_key=overlay_s3_key)
            
            logger.info(f"Successfully generated and stored overlay video for {video_id}")
            
        except Exception as e:
            logger.error(f"Failed to generate overlay video: {e}", exc_info=True)
            # Continue processing even if overlay generation fails
            overlay_s3_key = None
        
        update_video_progress(video_id, 80)
        
        # Step 9: Compute and store high-level pose metrics for UI
        from services.video_pose_metrics import compute_video_pose_metrics
        logger.info("Computing aggregated pose metrics for video %s", video_id)
        metrics = compute_video_pose_metrics(keypoints_data=keypoints_data, video_id=video_id)


        # Combine metrics with Pegasus analytics (if available)
        # We no longer generate Bedrock analytics from keypoints; rely on Pegasus only
        metrics_payload = dict(metrics)
        
        if pegasus_analytics is not None:
            # Use Pegasus analytics as the primary analysis
            metrics_payload["pegasus_analytics"] = pegasus_analytics
            logger.info("Stored Pegasus analytics in metrics payload for video %s", video_id)

        update_video_outputs(video_id, metrics_jsonb=metrics_payload)
        update_video_progress(video_id, 95)
        
        # Publish final analysis to Redis for WebSocket delivery
        if pegasus_analytics is not None:
            _publish_video_analysis_sync(
                video_id,
                {
                    "type": "pegasus_analysis",
                    "video_id": video_id,
                    "data": pegasus_analytics,
                    "done": True
                }
            )
        
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