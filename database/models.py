"""
Database models for matches and commentaries
"""

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    Boolean,
    Index,
)
from sqlalchemy.orm import relationship, foreign
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from datetime import datetime
import uuid
import enum

from database.connection import Base


class Match(Base):
    """
    Match information table
    Stores match metadata and status
    """

    __tablename__ = "matches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(String(50), unique=True, nullable=False, index=True, comment="External match ID from API")

    # Teams
    team1_name = Column(String(200), nullable=True)
    team2_name = Column(String(200), nullable=True)
    team1_id = Column(Integer, nullable=True)
    team2_id = Column(Integer, nullable=True)

    # Status
    state = Column(String(50), nullable=True, index=True, comment="Match state (LIVE, COMPLETE, etc.)")
    status = Column(Text, nullable=True, comment="Match status text (includes result)")
    is_complete = Column(Boolean, default=False, index=True)

    # Match metadata
    match_format = Column(String(20), nullable=True, comment="T20, ODI, Test, etc.")
    series_name = Column(String(200), nullable=True)
    series_id = Column(Integer, nullable=True)
    match_desc = Column(String(200), nullable=True)
    match_start_timestamp = Column(DateTime, nullable=True)
    match_end_timestamp = Column(DateTime, nullable=True)

    # Winning info
    winning_team_id = Column(Integer, nullable=True)
    winning_team_name = Column(String(200), nullable=True)

    # Audit
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    extra_metadata = Column(JSON, nullable=True)

    # Relationships
    commentaries = relationship(
        "Commentary",
        back_populates="match",
        primaryjoin="Match.match_id==foreign(Commentary.match_id)",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_match_state_complete", "state", "is_complete"),
        Index("idx_match_created", "created_at"),
    )


class Commentary(Base):
    """
    Commentary lines table
    Stores individual commentary events
    """

    __tablename__ = "commentaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    match_id = Column(String(50), nullable=False, index=True, comment="External match ID")

    text = Column(Text, nullable=False)
    event_type = Column(String(50), nullable=True, index=True, comment="WICKET, FOUR, SIX, BALL, etc.")

    ball_number = Column(Integer, nullable=True, index=True)
    over_number = Column(Float, nullable=True, index=True)
    innings_id = Column(Integer, nullable=True, index=True)

    runs = Column(Integer, nullable=True)
    wickets = Column(Integer, nullable=True)

    batting_team_name = Column(String(512), nullable=True)

    timestamp = Column(DateTime, nullable=False, index=True, comment="When the event occurred")
    prev_commentary_id = Column(UUID(as_uuid=True), nullable=True, index=True, comment="Linked list pointer to previous commentary")
    extra_metadata = Column(JSON, nullable=True, comment="Additional commentary metadata")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship(
        "Match",
        back_populates="commentaries",
        primaryjoin="foreign(Commentary.match_id)==Match.match_id",
        viewonly=True,
    )

    __table_args__ = (
        Index("idx_commentary_match_timestamp", "match_id", "timestamp", unique=True),
        Index("idx_commentary_event_type", "event_type"),
        Index("idx_commentary_over_ball", "over_number", "ball_number"),
        Index("idx_commentary_created", "created_at"),
    )


class VideoStatus(str, enum.Enum):
    """Video processing status enum"""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Video(Base):
    """
    Video information table
    Stores video metadata, S3 paths, and processing status
    """

    __tablename__ = "videos"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Storage configuration
    storage_type = Column(String(20), default="local", nullable=False, comment="Storage type: 'local' or 's3'")
    # S3 storage paths (bucket stored separately in config for IAM flexibility)
    s3_raw_key = Column(String(512), nullable=True, comment="S3 key path to original uploaded video (if using S3)")
    s3_bucket = Column(String(255), nullable=True, comment="S3 bucket name (stored for reference, can be changed via IAM)")
    # Local storage path
    local_file_path = Column(String(512), nullable=True, comment="Local file system path to video (if using local storage)")

    # Upload metadata
    content_type = Column(String(100), nullable=True, comment="MIME type of uploaded video")
    raw_size_bytes = Column(Integer, nullable=True, comment="Size of uploaded video in bytes")

    # Video metadata (extracted during ingest)
    duration_seconds = Column(Float, nullable=True, comment="Video duration in seconds")
    original_fps = Column(Float, nullable=True, comment="Original video FPS")
    width = Column(Integer, nullable=True, comment="Video width in pixels")
    height = Column(Integer, nullable=True, comment="Video height in pixels")

    # Thumbnail
    thumbnail_s3_key = Column(String(512), nullable=True, comment="S3 key path to thumbnail image")

    # Processing status
    status = Column(
        ENUM(
            VideoStatus,
            name="video_status",
            create_type=False,
            # Ensure PostgreSQL enum values use the Enum *values* (e.g. "uploaded")
            # instead of the Enum names (e.g. "UPLOADED"), to match the existing DB type
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=VideoStatus.UPLOADED,
        nullable=False,
        index=True,
        comment="Processing pipeline state"
    )

    # Job tracking
    queue_job_id = Column(String(255), nullable=True, index=True, comment="Queue job identifier")
    worker_id = Column(String(255), nullable=True, comment="Worker instance identifier")
    progress_percent = Column(Integer, default=0, nullable=False, comment="Processing progress (0-100)")
    error_message = Column(Text, nullable=True, comment="Error message if processing failed")

    # Output paths
    keypoints_s3_key = Column(String(512), nullable=True, comment="S3 key path to keypoints JSON file")
    keypoints_local_path = Column(String(512), nullable=True, comment="Local path to keypoints JSON file (if using local storage)")
    overlay_video_s3_key = Column(String(512), nullable=True, comment="S3 key path to overlay MP4 video")
    overlay_video_local_path = Column(String(512), nullable=True, comment="Local path to overlay MP4 video (if using local storage)")
    # Binary format for analysis data (compressed/msgpack)
    analysis_binary_path = Column(String(512), nullable=True, comment="Path to binary-encoded analysis data (msgpack/compressed)")

    # Inline data (only for small videos)
    keypoints_jsonb = Column(JSONB, nullable=True, comment="Inline keypoints JSON (only for small videos)")

    # Metrics and analysis
    metrics_jsonb = Column(JSONB, nullable=True, comment="Aggregated metrics (contact_frame, max_bat_speed, avg_knee_angle, etc.)")

    # Analysis metadata
    analysis_model = Column(String(255), nullable=True, comment="Model used for analysis")
    analysis_model_version = Column(String(100), nullable=True, comment="Model version")
    analysis_fps = Column(Float, nullable=True, comment="FPS used during analysis")

    # Output options
    output_options = Column(JSON, nullable=True, comment="Requested output options (keypoints, overlay, both)")

    # Retention
    retention_expires_at = Column(DateTime, nullable=True, index=True, comment="TTL for S3 cleanup")

    # Integrity
    checksum = Column(String(64), nullable=True, comment="File checksum for deduplication/integrity")

    # Processing timestamps
    processing_started_at = Column(DateTime, nullable=True, comment="When processing started")
    processing_finished_at = Column(DateTime, nullable=True, comment="When processing finished")

    __table_args__ = (
        Index("idx_video_status", "status"),
        Index("idx_video_created", "created_at"),
        Index("idx_video_retention", "retention_expires_at"),
        Index("idx_video_queue_job", "queue_job_id"),
    )