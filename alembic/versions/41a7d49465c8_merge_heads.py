"""merge heads

Revision ID: 41a7d49465c8
Revises: 3244e178fd41, c8a2d7939e0d
Create Date: 2025-11-26 12:35:50.013282

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '41a7d49465c8'
down_revision: Union[str, None] = ('3244e178fd41', 'c8a2d7939e0d')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
