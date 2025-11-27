"""
Pydantic schemas for Cricket Pose Analysis API
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


# ===== Keypoint Models =====

class Keypoint(BaseModel):
    """Single keypoint detection"""
    name: str
    x: float
    y: float
    confidence: float = Field(ge=0.0, le=1.0)


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int


# ===== Pose Detection Models =====

class PoseDetection(BaseModel):
    """Single pose detection in a frame"""
    player_id: Optional[int] = None
    bbox: BoundingBox
    keypoints: List[Keypoint]
    confidence: float = Field(ge=0.0, le=1.0)
    metrics: Optional[Dict[str, float]] = None


class FramePoseData(BaseModel):
    """Pose data for a single frame"""
    frame: int
    time_seconds: float
    detections: List[PoseDetection]


# ===== Event Models =====

class EventMetrics(BaseModel):
    """Metrics associated with an event"""
    bat_angle_degrees: Optional[float] = None
    front_foot_forward: Optional[bool] = None
    trunk_rotation_degrees: Optional[float] = None
    elbow_angle_degrees: Optional[float] = None
    follow_through_speed_px_per_sec: Optional[float] = None
    backward_step_distance_px: Optional[float] = None
    elbow_height_percentile: Optional[float] = None
    body_lean_back_degrees: Optional[float] = None


class Event(BaseModel):
    """Detected event (shot or short-pitch)"""
    event_id: int
    event_type: str  # "shot" or "short_pitch"
    shot_type: Optional[str] = None  # cover_drive, pull, cut, etc.
    player_id: int
    start_frame: int
    end_frame: int
    start_time_seconds: float
    end_time_seconds: float
    keyframe: int
    confidence: float = Field(ge=0.0, le=1.0)
    metrics: EventMetrics


# ===== Player Models =====

class Player(BaseModel):
    """Tracked player information"""
    player_id: int
    first_appearance_frame: int
    last_appearance_frame: int
    total_detections: int
    avg_confidence: float


# ===== Summary Models =====

class Summary(BaseModel):
    """Analysis summary statistics"""
    total_shots_detected: int
    shot_type_distribution: Dict[str, int]
    short_pitch_events: int
    avg_bat_speed_px_per_sec: Optional[float] = None
    most_frequent_shot: Optional[str] = None


# ===== Video Metadata =====

class VideoMetadata(BaseModel):
    """Video file metadata"""
    filename: str
    duration_seconds: float
    fps: float
    resolution: List[int]  # [width, height]
    total_frames: int
    processed_frames: int
    sampling_rate: int


class ProcessingMetadata(BaseModel):
    """Processing pipeline metadata"""
    model_used: str
    tracker_used: str
    processing_time_seconds: float
    gpu_used: bool
    timestamp: str


# ===== Analysis Result =====

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    video_metadata: VideoMetadata
    processing_metadata: ProcessingMetadata
    players: List[Player]
    events: List[Event]
    pose_data: Optional[List[FramePoseData]] = None  # Optional, can be large
    summary: Summary


# ===== Job Management Models =====

class JobStatus(BaseModel):
    """Job processing status"""
    job_id: str
    status: str  # queued, processing, completed, failed
    progress: float = Field(ge=0.0, le=100.0)
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class JobCreate(BaseModel):
    """Job creation response"""
    job_id: str
    status: str
    message: str


# ===== Progress Update (WebSocket) =====

class ProgressUpdate(BaseModel):
    """Real-time progress update"""
    type: str  # progress, status, error, complete
    job_id: str
    status: str
    progress: float
    current_frame: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    timestamp: str


# ===== Configuration Models =====

class ProcessingConfig(BaseModel):
    """Processing configuration options"""
    sampling_rate_fps: Optional[int] = 10
    use_gpu: Optional[bool] = True
    batch_size: Optional[int] = 16
    pose_model: Optional[str] = "yolov8x-pose.pt"
    enable_tracking: Optional[bool] = True
    enable_shot_classification: Optional[bool] = True
    enable_short_pitch_detection: Optional[bool] = True
    export_annotated_video: Optional[bool] = True
    export_csv: Optional[bool] = True
