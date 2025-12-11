"""merge_heads

Revision ID: c55081b6b0ea
Revises: 41a7d49465c8, add_videos_table
Create Date: 2025-12-02 11:50:42.600122

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c55081b6b0ea'
down_revision: Union[str, None] = ('41a7d49465c8', 'add_videos_table')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
