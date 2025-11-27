"""add prev commentary link

Revision ID: c8a2d7939e0d
Revises: 5f9f460d1b3a
Create Date: 2025-11-25 18:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8a2d7939e0d'
down_revision: Union[str, None] = '5f9f460d1b3a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('commentaries', sa.Column('prev_commentary_id', sa.UUID(), nullable=True))
    op.create_index('idx_commentary_prev_id', 'commentaries', ['prev_commentary_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_commentary_prev_id', table_name='commentaries')
    op.drop_column('commentaries', 'prev_commentary_id')

