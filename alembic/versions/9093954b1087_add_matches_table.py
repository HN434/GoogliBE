"""add matches table

Revision ID: 9093954b1087
Revises: d29522fe52ba
Create Date: 2025-11-24 15:54:00.764664

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9093954b1087'
down_revision: Union[str, None] = 'd29522fe52ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "matches" in inspector.get_table_names():
        return

    op.create_table(
        'matches',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('match_id', sa.String(length=50), nullable=False),
        sa.Column('team1_name', sa.String(length=200), nullable=True),
        sa.Column('team2_name', sa.String(length=200), nullable=True),
        sa.Column('team1_id', sa.Integer(), nullable=True),
        sa.Column('team2_id', sa.Integer(), nullable=True),
        sa.Column('state', sa.String(length=50), nullable=True),
        sa.Column('status', sa.Text(), nullable=True),
        sa.Column('is_complete', sa.Boolean(), nullable=True),
        sa.Column('match_format', sa.String(length=20), nullable=True),
        sa.Column('series_name', sa.String(length=200), nullable=True),
        sa.Column('series_id', sa.Integer(), nullable=True),
        sa.Column('match_desc', sa.String(length=200), nullable=True),
        sa.Column('match_start_timestamp', sa.DateTime(), nullable=True),
        sa.Column('match_end_timestamp', sa.DateTime(), nullable=True),
        sa.Column('winning_team_id', sa.Integer(), nullable=True),
        sa.Column('winning_team_name', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('extra_metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('match_id'),
    )
    op.create_index('idx_match_state_complete', 'matches', ['state', 'is_complete'], unique=False)
    op.create_index('idx_match_created', 'matches', ['created_at'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "matches" not in inspector.get_table_names():
        return

    op.drop_index('idx_match_created', table_name='matches')
    op.drop_index('idx_match_state_complete', table_name='matches')
    op.drop_table('matches')
