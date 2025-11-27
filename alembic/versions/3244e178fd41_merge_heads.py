"""merge heads

Revision ID: 3244e178fd41
Revises: 5f9f460d1b3a, 9093954b1087
Create Date: 2025-11-25 17:04:38.413648

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3244e178fd41'
down_revision: Union[str, None] = ('5f9f460d1b3a', '9093954b1087')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
