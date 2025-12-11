"""
Database utilities for worker
Provides synchronous database operations for the worker
"""

import logging
from typing import Optional
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid

from database.models import Video, VideoStatus
from config import settings

logger = logging.getLogger(__name__)

# Synchronous database engine and session factory for worker
_engine = None
_SessionLocal = None


def get_db_engine():
    """Get or create synchronous database engine"""
    global _engine
    
    if _engine is None:
        database_url = (
            f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        _engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("Initialized synchronous database engine for worker")
    
    return _engine


def get_db_session() -> Session:
    """Get a synchronous database session"""
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_db_engine()
        _SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    
    return _SessionLocal()


def get_video_by_id(video_id: str) -> Optional[Video]:
    """
    Get video record by ID
    
    Args:
        video_id: UUID string of the video
        
    Returns:
        Video record or None if not found
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return None
    
    db = get_db_session()
    try:
        result = db.execute(
            select(Video).where(Video.id == video_uuid)
        )
        video = result.scalar_one_or_none()
        return video
    except Exception as e:
        logger.error(f"Error fetching video {video_id}: {e}", exc_info=True)
        db.rollback()
        return None
    finally:
        db.close()


def update_video_status(
    video_id: str,
    status: VideoStatus,
    error_message: Optional[str] = None
) -> bool:
    """
    Update video status
    
    Args:
        video_id: UUID string of the video
        status: New status
        error_message: Optional error message
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        db.execute(
            update(Video)
            .where(Video.id == video_uuid)
            .values(
                status=status,
                error_message=error_message,
            )
        )
        db.commit()
        logger.info(f"Updated video {video_id} status to {status.value}")
        return True
    except Exception as e:
        logger.error(f"Error updating video status: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def update_video_processing_started(video_id: str, worker_id: str) -> bool:
    """
    Mark video processing as started
    
    Args:
        video_id: UUID string of the video
        worker_id: Worker identifier (hostname)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        from datetime import datetime
        db.execute(
            update(Video)
            .where(Video.id == video_uuid)
            .values(
                status=VideoStatus.PROCESSING,
                processing_started_at=datetime.utcnow(),
                worker_id=worker_id,
                progress_percent=0,
            )
        )
        db.commit()
        logger.info(f"Marked video {video_id} processing started by worker {worker_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating video processing started: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def update_video_progress(video_id: str, progress_percent: int) -> bool:
    """
    Update video processing progress
    
    Args:
        video_id: UUID string of the video
        progress_percent: Progress percentage (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        db.execute(
            update(Video)
            .where(Video.id == video_uuid)
            .values(progress_percent=min(100, max(0, progress_percent)))
        )
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating video progress: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def update_video_metadata(
    video_id: str,
    duration_seconds: Optional[float] = None,
    original_fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> bool:
    """
    Update video metadata
    
    Args:
        video_id: UUID string of the video
        duration_seconds: Video duration
        original_fps: Original FPS
        width: Video width
        height: Video height
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        update_values = {}
        if duration_seconds is not None:
            update_values["duration_seconds"] = duration_seconds
        if original_fps is not None:
            update_values["original_fps"] = original_fps
        if width is not None:
            update_values["width"] = width
        if height is not None:
            update_values["height"] = height
        
        if update_values:
            db.execute(
                update(Video)
                .where(Video.id == video_uuid)
                .values(**update_values)
            )
            db.commit()
            logger.info(f"Updated video {video_id} metadata: {update_values}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating video metadata: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def update_video_outputs(
    video_id: str,
    keypoints_s3_key: Optional[str] = None,
    keypoints_local_path: Optional[str] = None,
    overlay_video_s3_key: Optional[str] = None,
    overlay_video_local_path: Optional[str] = None,
    keypoints_jsonb: Optional[dict] = None,
    metrics_jsonb: Optional[dict] = None,
    thumbnail_s3_key: Optional[str] = None,
    analysis_binary_path: Optional[str] = None,
) -> bool:
    """
    Update video output paths and data
    
    Args:
        video_id: UUID string of the video
        keypoints_s3_key: S3 key for keypoints JSON
        overlay_video_s3_key: S3 key for overlay video
        keypoints_jsonb: Inline keypoints JSON (for small videos)
        metrics_jsonb: Aggregated metrics JSON
        thumbnail_s3_key: S3 key for thumbnail
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        update_values = {}
        if keypoints_s3_key is not None:
            update_values["keypoints_s3_key"] = keypoints_s3_key
        if keypoints_local_path is not None:
            update_values["keypoints_local_path"] = keypoints_local_path
        if overlay_video_s3_key is not None:
            update_values["overlay_video_s3_key"] = overlay_video_s3_key
        if overlay_video_local_path is not None:
            update_values["overlay_video_local_path"] = overlay_video_local_path
        if keypoints_jsonb is not None:
            update_values["keypoints_jsonb"] = keypoints_jsonb
        if metrics_jsonb is not None:
            update_values["metrics_jsonb"] = metrics_jsonb
        if thumbnail_s3_key is not None:
            update_values["thumbnail_s3_key"] = thumbnail_s3_key
        if analysis_binary_path is not None:
            update_values["analysis_binary_path"] = analysis_binary_path
        
        if update_values:
            db.execute(
                update(Video)
                .where(Video.id == video_uuid)
                .values(**update_values)
            )
            db.commit()
            logger.info(f"Updated video {video_id} outputs")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating video outputs: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()


def mark_video_completed(video_id: str) -> bool:
    """
    Mark video processing as completed
    
    Args:
        video_id: UUID string of the video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except ValueError:
        logger.error(f"Invalid video ID format: {video_id}")
        return False
    
    db = get_db_session()
    try:
        from datetime import datetime
        db.execute(
            update(Video)
            .where(Video.id == video_uuid)
            .values(
                status=VideoStatus.SUCCEEDED,
                processing_finished_at=datetime.utcnow(),
                progress_percent=100,
            )
        )
        db.commit()
        logger.info(f"Marked video {video_id} as completed")
        return True
    except Exception as e:
        logger.error(f"Error marking video completed: {e}", exc_info=True)
        db.rollback()
        return False
    finally:
        db.close()

