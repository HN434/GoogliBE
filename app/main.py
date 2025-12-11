"""
FastAPI backend for 3D Cricket Replay System
Handles video upload, pose extraction, and 3D data generation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
import os
import uuid
import json
import shutil
import logging
from datetime import datetime

import httpx
# Lazy imports for heavy ML libraries - only load when actually needed
# from .pose_extractor import PoseExtractor
# from .pose_extractor_yolo import YoloPoseExtractor
from .video_utils import VideoProcessor
from .commentary_router import router as commentary_router, commentary_lifespan
from services.redis_service import redis_service
from config import settings
from utils.api_key_rotator import APIKeyRotator


# Combine lifespan events for both systems
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Combined lifespan manager for all systems"""
    import time
    startup_start = time.time()
    startup_logger = logging.getLogger(__name__)
    startup_logger.info("=" * 50)
    startup_logger.info("🚀 Starting FastAPI application...")
    startup_logger.info("=" * 50)
    
    async with commentary_lifespan(app):
        startup_time = time.time() - startup_start
        startup_logger.info("=" * 50)
        startup_logger.info(f"✅ FastAPI application ready in {startup_time:.2f}s")
        startup_logger.info("=" * 50)
        yield


app = FastAPI(
    title="Cricket Backend API",
    description="3D Cricket Replay & Live Commentary System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Include routers
app.include_router(commentary_router)
from .video_router import router as video_router
app.include_router(video_router)
from .chat_router import router as chat_router
app.include_router(chat_router)

# Configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job storage (use Redis/DB in production)
jobs_db: Dict[str, dict] = {}

# Live matches cache configuration
LIVE_MATCHES_CACHE_KEY = "live_matches:latest"
LIVE_MATCHES_CACHE_TTL = 600  # seconds


class JobStatus(BaseModel):
    job_id: str
    status: str  # 'processing', 'completed', 'failed'
    progress: int
    message: str
    result: Optional[dict] = None
    error: Optional[str] = None


class PoseData3D(BaseModel):
    fps: int
    total_frames: int
    duration: float
    players: List[dict]
    metadata: dict


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "3D Cricket Replay API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/upload", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload cricket video for 3D pose extraction

    Args:
        file: Video file (mp4, avi, mov)

    Returns:
        JobStatus with job_id for tracking progress
    """

    # Validate file
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a video file."
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")


    # need to update this logic to use the S3 directory

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

    # Initialize job status
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Video uploaded successfully. Processing queued.",
        "filename": file.filename,
        "upload_path": upload_path,
        "created_at": datetime.now().isoformat()
    }

    # Start background processing
    background_tasks.add_task(process_video, job_id, upload_path)

    return JobStatus(
        job_id=job_id,
        status="queued",
        progress=0,
        message="Video uploaded successfully. Processing will begin shortly."
    )


@app.get("/api/live-matches")
async def get_live_matches():
    """
    Proxy live matches endpoint with Redis caching.
    Returns cached data when available, otherwise fetches from RapidAPI.
    Uses key rotation to distribute load across multiple API keys.
    """
    api_keys = settings.get_rapidapi_keys()
    if not api_keys:
        raise HTTPException(
            status_code=500,
            detail="Live matches API keys are not configured"
        )

    # Attempt to serve from cache
    if redis_service.redis_client:
        print(f"Attempting to serve from cache: {LIVE_MATCHES_CACHE_KEY}")
        cached = await redis_service.redis_client.get(LIVE_MATCHES_CACHE_KEY)
        if cached:
            try:
                cached_data = json.loads(cached)
                print(f"Serving from cache")
                return JSONResponse(content=cached_data)

            except json.JSONDecodeError:
                # Cache is corrupt, fallback to fresh fetch
                print(f"Cache is corrupt")
                pass

    # Initialize key rotator for this request
    key_rotator = APIKeyRotator(api_keys)
    url = f"{settings.RAPIDAPI_BASE_URL.rstrip('/')}/matches/v1/live"
    
    last_error = None
    max_attempts = len(api_keys)
    
    for attempt in range(max_attempts):
        current_key = await key_rotator.get_next_key()
        headers = {
            "x-rapidapi-host": settings.RAPIDAPI_HOST,
            "x-rapidapi-key": current_key
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers)
                
                # Check for rate limit or auth errors
                if key_rotator.should_retry_with_different_key(response.status_code):
                    key_rotator.mark_key_rate_limited(current_key)
                    if attempt < max_attempts - 1:
                        continue  # Try next key
                
                response.raise_for_status()
                data = response.json()
                
                # Mark key as successful
                key_rotator.mark_key_success(current_key)
                
                # Store response in cache for future calls
                if redis_service.redis_client:
                    try:
                        await redis_service.redis_client.set(
                            LIVE_MATCHES_CACHE_KEY,
                            json.dumps(data),
                            ex=LIVE_MATCHES_CACHE_TTL
                        )
                    except Exception:
                        # Cache failures shouldn't break the API path
                        pass

                return JSONResponse(content=data)
                
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            
            # Mark key as problematic if it's a rate limit or auth error
            if key_rotator.should_retry_with_different_key(status_code):
                key_rotator.mark_key_rate_limited(current_key)
                if attempt < max_attempts - 1:
                    continue  # Try next key
            
            last_error = exc
            if attempt == max_attempts - 1:
                raise HTTPException(
                    status_code=status_code,
                    detail=f"Live matches provider returned {status_code}"
                )
                
        except httpx.RequestError as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to reach live matches provider: {exc}"
                )
    
    # If we get here, all keys failed
    raise HTTPException(
        status_code=502,
        detail=f"All API keys failed. Last error: {last_error}"
    )


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get processing status for a job

    Args:
        job_id: Job identifier

    Returns:
        Current job status and progress
    """

    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    job = jobs_db[job_id]

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/api/result/{job_id}", response_model=PoseData3D)
async def get_result(job_id: str):
    """
    Get 3D pose data result for completed job

    Args:
        job_id: Job identifier

    Returns:
        Complete 3D pose data for visualization
    """

    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    job = jobs_db[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed. Current status: {job['status']}"
        )

    if not job.get("result"):
        raise HTTPException(
            status_code=500,
            detail="Result data is missing"
        )

    return job["result"]


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files

    Args:
        job_id: Job identifier
    """

    if job_id not in jobs_db:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    job = jobs_db[job_id]

    # Delete files
    try:
        if os.path.exists(job.get("upload_path", "")):
            os.remove(job["upload_path"])

        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as e:
        print(f"Error deleting files: {e}")

    # Remove from database
    del jobs_db[job_id]

    return {"message": "Job deleted successfully"}


async def process_video(job_id: str, video_path: str):
    """
    Background task to process video and extract 3D pose data

    Args:
        job_id: Job identifier
        video_path: Path to uploaded video file
    """

    try:
        # Update status
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["progress"] = 10
        jobs_db[job_id]["message"] = "Initializing video processor..."

        # Initialize processors
        video_processor = VideoProcessor(video_path)
        # Use YOLO for pose detection - lazy import to avoid loading ML models at startup
        from .pose_extractor_yolo import YoloPoseExtractor
        pose_extractor = YoloPoseExtractor()

        # Extract video info
        jobs_db[job_id]["progress"] = 20
        jobs_db[job_id]["message"] = "Analyzing video..."

        video_info = video_processor.get_video_info()
        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        duration = video_info["duration"]

        # Process frames and extract poses
        jobs_db[job_id]["progress"] = 30
        jobs_db[job_id]["message"] = "Extracting poses from frames..."

        all_poses = []
        frame_count = 0

        for frame in video_processor.extract_frames():
            # Extract 3D pose from frame
            pose_result = pose_extractor.extract_pose_3d(frame)

            # Detect ball in frame
            ball_position = pose_extractor.detect_ball(frame, frame_count)

            if pose_result:
                all_poses.append({
                    "frame_index": frame_count,
                    "timestamp": frame_count / fps,
                    "detections": pose_result,
                    "ball": ball_position
                })

            frame_count += 1

            # Update progress
            progress = 30 + int((frame_count / total_frames) * 50)
            jobs_db[job_id]["progress"] = min(progress, 80)
            jobs_db[job_id]["message"] = f"Processing frame {frame_count}/{total_frames}..."

        # Detect players and roles
        jobs_db[job_id]["progress"] = 85
        jobs_db[job_id]["message"] = "Analyzing player roles..."

        players = pose_extractor.detect_players(all_poses)

        # Get ball trajectory
        ball_trajectory = pose_extractor.get_ball_trajectory()

        # Compile final result
        jobs_db[job_id]["progress"] = 95
        jobs_db[job_id]["message"] = "Finalizing 3D data..."

        result = {
            "fps": fps,
            "total_frames": len(all_poses),
            "duration": duration,
            "players": players,
            "ball_trajectory": ball_trajectory,
            "metadata": {
                "video_resolution": f"{video_info['width']}x{video_info['height']}",
                "processed_at": datetime.now().isoformat(),
                "job_id": job_id
            }
        }

        # Save result to file
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Update job status
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["message"] = "Processing completed successfully!"
        jobs_db[job_id]["result"] = result
        jobs_db[job_id]["completed_at"] = datetime.now().isoformat()

        # Cleanup
        pose_extractor.cleanup()

    except Exception as e:
        # Handle errors
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["progress"] = 0
        jobs_db[job_id]["message"] = "Processing failed"
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["failed_at"] = datetime.now().isoformat()

        print(f"Error processing video {job_id}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
