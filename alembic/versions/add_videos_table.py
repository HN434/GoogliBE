"""add videos table

Revision ID: add_videos_table
Revises: 9093954b1087
Create Date: 2025-01-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'add_videos_table'
down_revision: Union[str, None] = '9093954b1087'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "videos" in inspector.get_table_names():
        return

    op.create_table(
        'videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('s3_raw_key', sa.String(length=512), nullable=False),
        sa.Column('s3_bucket', sa.String(length=255), nullable=True),
        sa.Column('content_type', sa.String(length=100), nullable=True),
        sa.Column('raw_size_bytes', sa.Integer(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('original_fps', sa.Float(), nullable=True),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('thumbnail_s3_key', sa.String(length=512), nullable=True),
        sa.Column('status', postgresql.ENUM('uploaded', 'queued', 'processing', 'succeeded', 'failed', name='video_status'), nullable=False, server_default='uploaded'),
        sa.Column('queue_job_id', sa.String(length=255), nullable=True),
        sa.Column('worker_id', sa.String(length=255), nullable=True),
        sa.Column('progress_percent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('keypoints_s3_key', sa.String(length=512), nullable=True),
        sa.Column('overlay_video_s3_key', sa.String(length=512), nullable=True),
        sa.Column('keypoints_jsonb', postgresql.JSONB(), nullable=True),
        sa.Column('metrics_jsonb', postgresql.JSONB(), nullable=True),
        sa.Column('analysis_model', sa.String(length=255), nullable=True),
        sa.Column('analysis_model_version', sa.String(length=100), nullable=True),
        sa.Column('analysis_fps', sa.Float(), nullable=True),
        sa.Column('output_options', sa.JSON(), nullable=True),
        sa.Column('retention_expires_at', sa.DateTime(), nullable=True),
        sa.Column('checksum', sa.String(length=64), nullable=True),
        sa.Column('processing_started_at', sa.DateTime(), nullable=True),
        sa.Column('processing_finished_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    
    # Create indexes
    op.create_index('idx_video_status', 'videos', ['status'], unique=False)
    op.create_index('idx_video_created', 'videos', ['created_at'], unique=False)
    op.create_index('idx_video_retention', 'videos', ['retention_expires_at'], unique=False)
    op.create_index('idx_video_queue_job', 'videos', ['queue_job_id'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "videos" not in inspector.get_table_names():
        return

    # Drop indexes
    op.drop_index('idx_video_queue_job', table_name='videos')
    op.drop_index('idx_video_retention', table_name='videos')
    op.drop_index('idx_video_created', table_name='videos')
    op.drop_index('idx_video_status', table_name='videos')
    
    # Drop table
    op.drop_table('videos')
    
    # Drop enum type
    op.execute("DROP TYPE IF EXISTS video_status")

