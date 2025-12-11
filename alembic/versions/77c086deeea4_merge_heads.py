"""Merge heads

Revision ID: 77c086deeea4
Revises: 6f53e2390451, b83e3cdd9b4e
Create Date: 2025-12-11 12:32:25.952470

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '77c086deeea4'
down_revision: Union[str, None] = ('6f53e2390451', 'b83e3cdd9b4e')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
