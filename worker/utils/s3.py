"""
S3 utilities for worker
Handles downloading and uploading files to/from S3
"""

import boto3
import tempfile
import os
import logging
from typing import Optional, Dict
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

# S3 client instance
_s3_client = None


def get_s3_client():
    """Get or create S3 client"""
    global _s3_client
    
    if _s3_client is None:
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            raise RuntimeError("AWS credentials not configured")
        
        if not settings.S3_BUCKET_NAME:
            raise RuntimeError("S3 bucket name not configured")
        
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        logger.info(f"Initialized S3 client for bucket: {settings.S3_BUCKET_NAME}")
    
    return _s3_client


def download_from_s3(s3_key: str, local_path: Optional[str] = None) -> str:
    """
    Download file from S3 to local filesystem
    
    Args:
        s3_key: S3 key (path) of the file
        local_path: Optional local path. If not provided, uses temp directory
        
    Returns:
        Path to downloaded file
    """
    s3_client = get_s3_client()
    bucket_name = settings.S3_BUCKET_NAME
    
    if local_path is None:
        # Create temp file
        temp_dir = tempfile.gettempdir()
        filename = os.path.basename(s3_key) or "video"
        local_path = os.path.join(temp_dir, f"video_{os.path.basename(s3_key)}")
    
    logger.info(f"Downloading {s3_key} from S3 to {local_path}")
    
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        logger.info(f"Successfully downloaded {s3_key} to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}", exc_info=True)
        raise


def upload_to_s3(local_path: str, s3_key: str, content_type: Optional[str] = None) -> str:
    """
    Upload file to S3
    
    Args:
        local_path: Local file path
        s3_key: S3 key (path) where file should be stored
        content_type: Optional content type
        
    Returns:
        S3 key of uploaded file
    """
    s3_client = get_s3_client()
    bucket_name = settings.S3_BUCKET_NAME
    
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")
    
    logger.info(f"Uploading {local_path} to S3 as {s3_key}")
    
    try:
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        s3_client.upload_file(
            local_path,
            bucket_name,
            s3_key,
            ExtraArgs=extra_args if extra_args else None
        )
        logger.info(f"Successfully uploaded {local_path} to S3 as {s3_key}")
        return s3_key
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}", exc_info=True)
        raise


def get_video_metadata(video_path: str) -> Dict[str, any]:
    """
    Extract video metadata using OpenCV
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with duration_seconds, fps, width, height
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        duration_seconds = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        metadata = {
            "duration_seconds": duration_seconds,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
        
        logger.info(f"Extracted video metadata: {metadata}")
        return metadata
        
    except ImportError:
        logger.warning("OpenCV not available, cannot extract video metadata")
        return {}
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}", exc_info=True)
        return {}


def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

