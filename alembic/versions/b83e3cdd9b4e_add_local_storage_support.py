"""add_local_storage_support

Revision ID: b83e3cdd9b4e
Revises: c55081b6b0ea
Create Date: 2025-12-02 11:50:58.583983

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b83e3cdd9b4e'
down_revision: Union[str, None] = 'c55081b6b0ea'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    
    # Check if videos table exists
    if "videos" not in inspector.get_table_names():
        # Create enum type for video status if it doesn't exist
        op.execute("""
            DO $$ BEGIN
                CREATE TYPE video_status AS ENUM ('uploaded', 'queued', 'processing', 'succeeded', 'failed');
            EXCEPTION
                WHEN duplicate_object THEN null;
            END $$;
        """)
        
        # Create videos table with all columns including new local storage ones
        op.create_table('videos',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('storage_type', sa.String(length=20), nullable=False, server_default='s3', comment="Storage type: 'local' or 's3'"),
            sa.Column('s3_raw_key', sa.String(length=512), nullable=True, comment='S3 key path to original uploaded video (if using S3)'),
            sa.Column('s3_bucket', sa.String(length=255), nullable=True, comment='S3 bucket name (stored for reference, can be changed via IAM)'),
            sa.Column('local_file_path', sa.String(length=512), nullable=True, comment='Local file system path to video (if using local storage)'),
            sa.Column('content_type', sa.String(length=100), nullable=True, comment='MIME type of uploaded video'),
            sa.Column('raw_size_bytes', sa.Integer(), nullable=True, comment='Size of uploaded video in bytes'),
            sa.Column('duration_seconds', sa.Float(), nullable=True, comment='Video duration in seconds'),
            sa.Column('original_fps', sa.Float(), nullable=True, comment='Original video FPS'),
            sa.Column('width', sa.Integer(), nullable=True, comment='Video width in pixels'),
            sa.Column('height', sa.Integer(), nullable=True, comment='Video height in pixels'),
            sa.Column('thumbnail_s3_key', sa.String(length=512), nullable=True, comment='S3 key path to thumbnail image'),
            sa.Column('status', postgresql.ENUM('uploaded', 'queued', 'processing', 'succeeded', 'failed', name='video_status', create_type=False), nullable=False, server_default='uploaded'),
            sa.Column('queue_job_id', sa.String(length=255), nullable=True, comment='Queue job identifier'),
            sa.Column('worker_id', sa.String(length=255), nullable=True, comment='Worker instance identifier'),
            sa.Column('progress_percent', sa.Integer(), nullable=False, server_default='0', comment='Processing progress (0-100)'),
            sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if processing failed'),
            sa.Column('keypoints_s3_key', sa.String(length=512), nullable=True, comment='S3 key path to keypoints JSON file'),
            sa.Column('keypoints_local_path', sa.String(length=512), nullable=True, comment='Local path to keypoints JSON file (if using local storage)'),
            sa.Column('overlay_video_s3_key', sa.String(length=512), nullable=True, comment='S3 key path to overlay MP4 video'),
            sa.Column('overlay_video_local_path', sa.String(length=512), nullable=True, comment='Local path to overlay MP4 video (if using local storage)'),
            sa.Column('analysis_binary_path', sa.String(length=512), nullable=True, comment='Path to binary-encoded analysis data (msgpack/compressed)'),
            sa.Column('keypoints_jsonb', postgresql.JSONB(), nullable=True, comment='Inline keypoints JSON (only for small videos)'),
            sa.Column('metrics_jsonb', postgresql.JSONB(), nullable=True, comment='Aggregated metrics (contact_frame, max_bat_speed, avg_knee_angle, etc.)'),
            sa.Column('analysis_model', sa.String(length=255), nullable=True, comment='Model used for analysis'),
            sa.Column('analysis_model_version', sa.String(length=100), nullable=True, comment='Model version'),
            sa.Column('analysis_fps', sa.Float(), nullable=True, comment='FPS used during analysis'),
            sa.Column('output_options', sa.JSON(), nullable=True, comment='Requested output options (keypoints, overlay, both)'),
            sa.Column('retention_expires_at', sa.DateTime(), nullable=True, comment='TTL for S3 cleanup'),
            sa.Column('checksum', sa.String(length=64), nullable=True, comment='File checksum for deduplication/integrity'),
            sa.Column('processing_started_at', sa.DateTime(), nullable=True, comment='When processing started'),
            sa.Column('processing_finished_at', sa.DateTime(), nullable=True, comment='When processing finished'),
            sa.PrimaryKeyConstraint('id'),
        )
        
        # Create indexes
        op.create_index('idx_video_status', 'videos', ['status'], unique=False)
        op.create_index('idx_video_created', 'videos', ['created_at'], unique=False)
        op.create_index('idx_video_retention', 'videos', ['retention_expires_at'], unique=False)
        op.create_index('idx_video_queue_job', 'videos', ['queue_job_id'], unique=False)
    else:
        # Table exists, just add new columns
        # Check which columns already exist
        existing_columns = [col['name'] for col in inspector.get_columns('videos')]
        
        if 'storage_type' not in existing_columns:
            op.add_column('videos', sa.Column('storage_type', sa.String(length=20), nullable=False, server_default='s3', comment="Storage type: 'local' or 's3'"))
        if 'local_file_path' not in existing_columns:
            op.add_column('videos', sa.Column('local_file_path', sa.String(length=512), nullable=True, comment='Local file system path to video (if using local storage)'))
        if 'keypoints_local_path' not in existing_columns:
            op.add_column('videos', sa.Column('keypoints_local_path', sa.String(length=512), nullable=True, comment='Local path to keypoints JSON file (if using local storage)'))
        if 'overlay_video_local_path' not in existing_columns:
            op.add_column('videos', sa.Column('overlay_video_local_path', sa.String(length=512), nullable=True, comment='Local path to overlay MP4 video (if using local storage)'))
        if 'analysis_binary_path' not in existing_columns:
            op.add_column('videos', sa.Column('analysis_binary_path', sa.String(length=512), nullable=True, comment='Path to binary-encoded analysis data (msgpack/compressed)'))
        
        # Make s3_raw_key nullable if it's not already
        s3_raw_key_col = next((col for col in inspector.get_columns('videos') if col['name'] == 's3_raw_key'), None)
        if s3_raw_key_col and s3_raw_key_col['nullable'] is False:
            op.alter_column('videos', 's3_raw_key',
                           existing_type=sa.String(length=512),
                           nullable=True)


def downgrade() -> None:
    # Remove local storage columns
    op.alter_column('videos', 's3_raw_key',
                   existing_type=sa.String(length=512),
                   nullable=False,
                   existing_nullable=True)
    op.drop_column('videos', 'analysis_binary_path')
    op.drop_column('videos', 'overlay_video_local_path')
    op.drop_column('videos', 'keypoints_local_path')
    op.drop_column('videos', 'local_file_path')
    op.drop_column('videos', 'storage_type')
