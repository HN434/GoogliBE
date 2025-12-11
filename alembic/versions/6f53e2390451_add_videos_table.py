"""add_videos_table

Revision ID: 6f53e2390451
Revises: 41a7d49465c8
Create Date: 2025-11-28 11:28:54.291010

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f53e2390451'
down_revision: Union[str, None] = '41a7d49465c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
