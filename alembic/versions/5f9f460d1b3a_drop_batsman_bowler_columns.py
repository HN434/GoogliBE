"""drop batsman and bowler columns

Revision ID: 5f9f460d1b3a
Revises: 7e88b9dcb33f
Create Date: 2025-11-25 17:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f9f460d1b3a'
down_revision: Union[str, None] = 'd29522fe52ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('commentaries', 'bowler')
    op.drop_column('commentaries', 'batsman')


def downgrade() -> None:
    op.add_column('commentaries', sa.Column('batsman', sa.VARCHAR(length=512), nullable=True))
    op.add_column('commentaries', sa.Column('bowler', sa.VARCHAR(length=512), nullable=True))

