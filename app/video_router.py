"""
Video Upload API Router
Handles presigned URL generation and upload completion
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid
import logging
import os
import shutil
import gzip
import msgpack
import asyncio
from pathlib import Path
from datetime import datetime

from models.schemas import (
    PresignedUrlRequest,
    PresignedUrlResponse,
    UploadCompleteRequest,
    UploadCompleteResponse,
)
from database.connection import get_db
from database.models import Video, VideoStatus
from services.s3_service import s3_service
from services.video_analytics_service import get_video_analytics_service
from services.redis_service import redis_service
from ws_manager.video_connection_manager import VideoWebSocketManager
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/videos", tags=["videos"])

# Global video WebSocket manager (initialized on first use)
video_ws_manager: Optional[VideoWebSocketManager] = None


def get_video_ws_manager() -> VideoWebSocketManager:
    """Get or create video WebSocket manager"""
    global video_ws_manager
    if video_ws_manager is None:
        video_ws_manager = VideoWebSocketManager()
        logger.info("✅ Video WebSocket manager initialized")
    return video_ws_manager


@router.post("/upload", response_model=UploadCompleteResponse)
async def upload_video_direct(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Direct video upload endpoint - stores video on server
    
    This endpoint accepts a video file upload and stores it locally on the server.
    After upload, the video is immediately queued for processing.
    
    Args:
        file: Video file to upload
        
    Returns:
        Upload confirmation with video status
    """
    try:
        # Validate content type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type. Expected video/*, got {file.content_type}"
            )
        
        # Generate unique video ID
        video_id = uuid.uuid4()
        
        # Create storage directory structure: videos/{year}/{month}/{video_id}/
        now = datetime.utcnow()
        storage_dir = Path(settings.VIDEO_STORAGE_DIR) / str(now.year) / f"{now.month:02d}" / str(video_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file locally
        file_extension = Path(file.filename).suffix if file.filename else ".mp4"
        local_file_path = storage_dir / f"video{file_extension}"
        
        # Get file size
        file_size = 0
        with open(local_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            file_size = local_file_path.stat().st_size
        
        # Validate file size
        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 if settings.MAX_UPLOAD_SIZE_MB else None
        if max_size_bytes and file_size > max_size_bytes:
            os.remove(local_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Create video record in database
        video = Video(
            id=video_id,
            storage_type="local",
            local_file_path=str(local_file_path),
            content_type=file.content_type,
            raw_size_bytes=file_size,
            status=VideoStatus.UPLOADED,
        )
        
        db.add(video)
        await db.commit()
        await db.refresh(video)
        
        logger.info(f"Uploaded video {video_id} to local storage: {local_file_path}")
        
        # Immediately mark as processing and enqueue worker
        try:
            from services.video_analysis_queue import enqueue_video_analysis
            job_id = await enqueue_video_analysis(video_id)
            logger.info(f"Enqueued video analysis job {job_id} for video {video_id}")
            
            # Update status to QUEUED (worker will change to PROCESSING when it starts)
            video.status = VideoStatus.QUEUED
            video.queue_job_id = job_id
            await db.commit()
            
        except Exception as e:
            logger.error(f"Failed to enqueue video analysis: {e}", exc_info=True)
            video.status = VideoStatus.FAILED
            video.error_message = f"Failed to enqueue analysis job: {str(e)}"
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to enqueue video analysis: {str(e)}"
            )
        
        return UploadCompleteResponse(
            video_id=str(video_id),
            status=video.status.value,
            message="Video uploaded successfully. Processing started. Connect to WebSocket for analysis results."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in direct upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/presigned-url", response_model=PresignedUrlResponse)
async def get_presigned_upload_url(
    request: PresignedUrlRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a presigned URL for video upload to S3
    
    The frontend will use this URL to directly upload the video file to S3.
    After upload, the frontend should call /api/videos/upload-complete to mark
    the upload as complete and trigger analysis.
    
    Args:
        request: Request containing filename, content_type, and optional file_size
        
    Returns:
        Presigned URL and video record information
    """
    try:
        # Generate unique video ID
        video_id = uuid.uuid4()
        
        # Generate S3 key (path) for the video
        # Format: videos/{year}/{month}/{video_id}/{filename}
        now = datetime.utcnow()
        s3_key = f"videos/{now.year}/{now.month:02d}/{video_id}/{request.filename}"
        
        # Validate content type
        if not request.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type. Expected video/*, got {request.content_type}"
            )
        
        # Validate file size if provided
        max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 if settings.MAX_UPLOAD_SIZE_MB else None
        if request.file_size_bytes and max_size_bytes and request.file_size_bytes > max_size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Check if S3 is enabled
        if not settings.USE_S3 or not settings.S3_BUCKET_NAME:
            raise HTTPException(
                status_code=400,
                detail="S3 storage is not enabled. Use /api/videos/upload for direct upload."
            )
        
        # Generate presigned URL
        expiration = settings.S3_PRESIGNED_URL_EXPIRATION
        presigned_data = s3_service.generate_presigned_upload_url(
            s3_key=s3_key,
            content_type=request.content_type,
            expiration=expiration,
            max_size_mb=settings.MAX_UPLOAD_SIZE_MB
        )
        
        # Create video record in database
        video = Video(
            id=video_id,
            storage_type="s3",
            s3_raw_key=s3_key,
            s3_bucket=settings.S3_BUCKET_NAME,
            content_type=request.content_type,
            raw_size_bytes=request.file_size_bytes,
            status=VideoStatus.UPLOADED,
        )
        
        db.add(video)
        await db.commit()
        await db.refresh(video)
        
        logger.info(f"Created video record {video_id} with S3 key {s3_key}")
        
        return PresignedUrlResponse(
            video_id=str(video_id),
            upload_url=presigned_data["upload_url"],
            s3_key=presigned_data["s3_key"],
            s3_bucket=presigned_data["s3_bucket"],
            expires_at=presigned_data["expires_at"],
            content_type=presigned_data["content_type"],
        )
        
    except RuntimeError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate presigned URL: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in presigned URL generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/upload-complete", response_model=UploadCompleteResponse)
async def mark_upload_complete(
    request: UploadCompleteRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark video upload as complete and trigger analysis
    
    This endpoint should be called by the frontend after successfully
    uploading the video to S3 using the presigned URL.
    
    Args:
        request: Request containing video_id, file_size, and optional checksum/output_options
        
    Returns:
        Confirmation with updated video status
    """
    try:
        # Parse video ID
        try:
            video_id = uuid.UUID(request.video_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video ID format: {request.video_id}"
            )
        
        # Fetch video record
        result = await db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()
        
        if not video:
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {request.video_id}"
            )
        
        # Verify upload status
        if video.status != VideoStatus.UPLOADED:
            raise HTTPException(
                status_code=400,
                detail=f"Video is not in UPLOADED status. Current status: {video.status.value}"
            )
        
        # Verify file exists (S3 or local)
        if video.storage_type == "s3":
            if not s3_service.check_object_exists(video.s3_raw_key):
                raise HTTPException(
                    status_code=400,
                    detail=f"Video file not found in S3 at key: {video.s3_raw_key}"
                )
            # Get S3 object metadata
            s3_metadata = s3_service.get_object_metadata(video.s3_raw_key)
        else:
            # Local storage - verify file exists
            if not video.local_file_path or not os.path.exists(video.local_file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Video file not found at local path: {video.local_file_path}"
                )
            s3_metadata = None
        
        # Update video record
        video.raw_size_bytes = request.file_size_bytes
        video.checksum = request.checksum
        video.output_options = request.output_options
        video.status = VideoStatus.QUEUED
        video.updated_at = datetime.utcnow()
        
        # Update from S3 metadata if available
        if s3_metadata:
            if s3_metadata.get("content_type"):
                video.content_type = s3_metadata["content_type"]
            if s3_metadata.get("content_length"):
                video.raw_size_bytes = s3_metadata["content_length"]
        
        await db.commit()
        await db.refresh(video)
        
        logger.info(f"Marked video {video_id} upload as complete, queued for processing")
        
        # Enqueue video analysis job
        try:
            from services.video_analysis_queue import enqueue_video_analysis
            job_id = await enqueue_video_analysis(video_id)
            logger.info(f"Enqueued video analysis job {job_id} for video {video_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue video analysis: {e}", exc_info=True)
            # Update video status to failed
            video.status = VideoStatus.FAILED
            video.error_message = f"Failed to enqueue analysis job: {str(e)}"
            await db.commit()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to enqueue video analysis: {str(e)}"
            )
        
        return UploadCompleteResponse(
            video_id=str(video_id),
            status=video.status.value,
            message="Upload completed successfully. Video queued for analysis."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error marking upload complete: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{video_id}/status")
async def get_video_status(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get current status of a video
    
    Args:
        video_id: UUID of the video
        
    Returns:
        Video status and metadata
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video ID format: {video_id}"
        )
    
    result = await db.execute(
        select(Video).where(Video.id == video_uuid)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )
    
    return {
        "video_id": str(video.id),
        "status": video.status.value,
        "progress_percent": video.progress_percent,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "updated_at": video.updated_at.isoformat() if video.updated_at else None,
        "processing_started_at": video.processing_started_at.isoformat() if video.processing_started_at else None,
        "processing_finished_at": video.processing_finished_at.isoformat() if video.processing_finished_at else None,
        "error_message": video.error_message,
        "duration_seconds": video.duration_seconds,
        "width": video.width,
        "height": video.height,
        "original_fps": video.original_fps,
    }


@router.get("/{video_id}/analysis")
async def get_analysis_data(
    video_id: str,
    format: str = "binary",  # "binary" or "json"
    db: AsyncSession = Depends(get_db)
):
    """
    Get analysis data for a completed video
    
    Returns analysis data in binary format (msgpack+gzip) by default for efficiency,
    or JSON format if requested.
    
    Args:
        video_id: UUID of the video
        format: Response format - "binary" (default) or "json"
        
    Returns:
        Analysis data in requested format
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video ID format: {video_id}"
        )
    
    result = await db.execute(
        select(Video).where(Video.id == video_uuid)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )
    
    if video.status != VideoStatus.SUCCEEDED:
        raise HTTPException(
            status_code=400,
            detail=f"Video analysis not completed. Current status: {video.status.value}"
        )
    
    # Try to get binary format first (preferred)
    analysis_path = None
    storage_type = getattr(video, 'storage_type', 's3')
    
    if storage_type == 'local' and video.analysis_binary_path:
        analysis_path = video.analysis_binary_path
        if not os.path.exists(analysis_path):
            analysis_path = None
    elif storage_type == 's3' and video.keypoints_s3_key:
        # For S3, we'd need to download first - for now, try to get from local path if available
        # In production, you might want to download from S3 here
        pass
    
    # Fallback to JSON if binary not available
    if not analysis_path:
        # Try to get JSON data
        if video.keypoints_jsonb:
            # Return inline JSON
            if format == "binary":
                # Convert JSON to binary
                json_data = video.keypoints_jsonb
                binary_data = gzip.compress(msgpack.packb(json_data, use_bin_type=True))
                return Response(
                    content=binary_data,
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f'attachment; filename="analysis_{video_id}.msgpack.gz"',
                        "Content-Encoding": "gzip"
                    }
                )
            else:
                return video.keypoints_jsonb
        else:
            raise HTTPException(
                status_code=404,
                detail="Analysis data not found. Video may not have completed processing."
            )
    
    # Return binary file
    if format == "binary":
        if not os.path.exists(analysis_path):
            raise HTTPException(
                status_code=404,
                detail=f"Analysis binary file not found at: {analysis_path}"
            )
        return FileResponse(
            analysis_path,
            media_type="application/octet-stream",
            filename=f"analysis_{video_id}.msgpack.gz",
            headers={"Content-Encoding": "gzip"}
        )
    else:
        # Read and convert binary to JSON
        try:
            with gzip.open(analysis_path, 'rb') as f:
                binary_data = f.read()
            json_data = msgpack.unpackb(binary_data, raw=False)
            return json_data
        except Exception as e:
            logger.error(f"Failed to read analysis binary file: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read analysis data: {str(e)}"
            )


@router.get("/{video_id}/metrics")
async def get_video_metrics(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Return aggregated pose metrics JSON for a processed video.

    This reads the `metrics_jsonb` column from the `videos` table and returns
    it to the frontend without any additional transformation.
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video ID format: {video_id}"
        )

    result = await db.execute(
        select(Video).where(Video.id == video_uuid)
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )

    if video.status != VideoStatus.SUCCEEDED:
        raise HTTPException(
            status_code=400,
            detail=f"Video analysis not completed. Current status: {video.status.value}"
        )

    if not video.metrics_jsonb:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found for this video."
        )

    return video.metrics_jsonb


@router.get("/{video_id}/bedrock-analytics")
async def get_video_bedrock_analytics(
    video_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate and return Bedrock-powered analytics for a processed video.

    - Reads `metrics_jsonb` directly from the `videos` table.
    - Sends the metrics as-is to Bedrock via `VideoAnalyticsService`.
    - Returns the parsed JSON analytics object to the frontend.

    This endpoint does not currently cache the Bedrock response; callers
    may want to cache on the frontend or at an API-gateway/CDN layer.
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid video ID format: {video_id}"
        )

    result = await db.execute(
        select(Video).where(Video.id == video_uuid)
    )
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found: {video_id}"
        )

    if video.status != VideoStatus.SUCCEEDED:
        raise HTTPException(
            status_code=400,
            detail=f"Video analysis not completed. Current status: {video.status.value}"
        )

    if not video.metrics_jsonb:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found for this video."
        )

    # If worker has already embedded Bedrock analytics under 'bedrock_analytics',
    # return that directly without calling Bedrock again.
    metrics_json = video.metrics_jsonb
    existing_analytics = metrics_json.get("bedrock_analytics") if isinstance(metrics_json, dict) else None
    if existing_analytics:
        return existing_analytics

    try:
        analytics_service = get_video_analytics_service()
    except RuntimeError as e:
        # Bedrock not configured
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    try:
        analytics = analytics_service.generate_analytics(
            video_id=str(video.id),
            metrics=metrics_json,
        )
    except (BotoCoreError, ClientError) as e:  # type: ignore[name-defined]
        # Surface Bedrock client errors with a generic message
        logger.error("Bedrock error while generating analytics for %s: %s", video_id, e, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="Failed to generate analytics from Bedrock.",
        )
    except json.JSONDecodeError as e:  # type: ignore[name-defined]
        logger.error("Failed to parse Bedrock analytics JSON for %s: %s", video_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Bedrock returned an invalid JSON payload for analytics.",
        )
    except Exception as e:
        logger.error("Unexpected error generating analytics for %s: %s", video_id, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while generating analytics.",
        )

    return analytics


@router.websocket("/ws/{video_id}")
async def websocket_video_analysis(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for video analysis results
    
    Connects to receive:
    1. Keypoints JSON (video overlay data)
    2. Bedrock analysis
    3. Completion message
    
    Args:
        websocket: WebSocket connection
        video_id: Video identifier
    """
    ws_manager = get_video_ws_manager()
    
    # Validate video_id format
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        await websocket.close(code=400, reason="Invalid video ID format")
        return
    
    logger.info(f"WebSocket connection request for video {video_id}")
    
    try:
        # Start background task to listen to Redis and forward messages
        # Start this BEFORE handling websocket to ensure subscription is ready
        logger.info(f"Starting Redis subscription task for video {video_id}")
        subscription_task = asyncio.create_task(
            _subscribe_and_forward_video_analysis(video_id)
        )
        
        # Give subscription a moment to establish
        await asyncio.sleep(0.1)
        logger.info(f"Subscription task started for video {video_id}, handling WebSocket")
        
        # Handle WebSocket connection
        await ws_manager.handle_websocket(websocket, video_id)
    except Exception as e:
        logger.error(f"Error in WebSocket handler for video {video_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass
    finally:
        # Cancel subscription task
        if 'subscription_task' in locals():
            subscription_task.cancel()
            try:
                await subscription_task
            except asyncio.CancelledError:
                pass


async def _subscribe_and_forward_video_analysis(video_id: str):
    """
    Subscribe to Redis Pub/Sub for video analysis and forward to WebSocket
    
    Args:
        video_id: Video identifier
    """
    ws_manager = get_video_ws_manager()
    
    # Ensure Redis is connected
    try:
        if not redis_service.redis_client:
            logger.info(f"Connecting to Redis for video {video_id} subscription")
            await redis_service.connect()
        logger.info(f"Redis connected, starting subscription for video {video_id}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
        return
    
    try:
        logger.info(f"Starting Redis subscription for video {video_id}")
        async for message in redis_service.subscribe_to_video_analysis(video_id):
            logger.info(f"Received message from Redis for video {video_id}: type={message.get('type')}")
            
            # Check if WebSocket connection still exists
            if not ws_manager.has_connection(video_id):
                logger.warning(f"WebSocket connection closed for video {video_id}, stopping subscription")
                break
            
            message_type = message.get("type")
            
            if message_type == "keypoints":
                # Send keypoints data
                keypoints_data = message.get("data", {})
                logger.info(f"Forwarding keypoints to WebSocket for video {video_id}")
                await ws_manager.send_keypoints(video_id, keypoints_data)
                logger.info(f"✅ Sent keypoints to WebSocket for video {video_id}")
                
            elif message_type == "bedrock_analysis":
                # Send bedrock analysis
                bedrock_data = message.get("data", {})
                logger.info(f"Forwarding bedrock analysis to WebSocket for video {video_id}")
                await ws_manager.send_bedrock_analysis(video_id, bedrock_data)
                logger.info(f"✅ Sent bedrock analysis to WebSocket for video {video_id}")
            else:
                logger.warning(f"Unknown message type received: {message_type}")
                
    except asyncio.CancelledError:
        logger.info(f"Subscription cancelled for video {video_id}")
    except Exception as e:
        logger.error(f"Error in subscription forwarding for video {video_id}: {e}", exc_info=True)

