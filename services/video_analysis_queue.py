"""
Video Analysis Queue Service
Handles enqueueing video analysis jobs using Redis Queue (RQ)
"""

import logging
import uuid
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from arq.connections import RedisSettings, create_pool

from database.models import Video, VideoStatus
from database.connection import async_session_maker
from config import settings

logger = logging.getLogger(__name__)


def _get_redis_settings() -> RedisSettings:
    """
    Build ARQ RedisSettings from REDIS_URL.
    """
    redis_url = settings.REDIS_URL or "redis://localhost:6379/0"

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

    return RedisSettings(host=host, port=port, database=db)


async def enqueue_video_analysis(video_id: uuid.UUID) -> str:
    """
    Enqueue a video analysis job
    
    This function:
    1. Pushes a job into the video-processing queue
    2. Stores the job ID in the database (queue_job_id)
    3. Updates status to QUEUED
    
    Args:
        video_id: UUID of the video to analyze
        
    Returns:
        Job ID string
        
    Raises:
        Exception: If video not found or queue operation fails
    """
    import asyncio
    
    try:
        # Get video record to verify it exists
        async with async_session_maker() as db:
            result = await db.execute(
                select(Video).where(Video.id == video_id)
            )
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValueError(f"Video not found: {video_id}")
            
            # ARQ enqueue: push job into the "video-processing" queue
            redis_settings = _get_redis_settings()
            redis = await create_pool(redis_settings)

            # args must be JSON-serializable
            job = await redis.enqueue_job(
                "worker.jobs.analyze_video.analyze_video_job_async",
                str(video_id),
                _queue_name="video-processing",
                _job_id=f"video-{video_id}",
                _timeout=600,
            )

            job_id = job.job_id
            logger.info(f"Enqueued video analysis job {job_id} for video {video_id}")
            
            # Update video record with job ID and status
            await db.execute(
                update(Video)
                .where(Video.id == video_id)
                .values(
                    queue_job_id=job_id,
                    status=VideoStatus.QUEUED,
                )
            )
            await db.commit()
            
            logger.info(f"Updated video {video_id} with job ID {job_id} and status QUEUED")
            
            return job_id
            
    except Exception as e:
        logger.error(f"Failed to enqueue video analysis for {video_id}: {e}", exc_info=True)
        # Update video status to failed if we can
        try:
            async with async_session_maker() as db:
                await db.execute(
                    update(Video)
                    .where(Video.id == video_id)
                    .values(
                        status=VideoStatus.FAILED,
                        error_message=f"Failed to enqueue job: {str(e)}",
                    )
                )
                await db.commit()
        except Exception as db_error:
            logger.error(f"Failed to update video status after enqueue error: {db_error}")
        
        raise

