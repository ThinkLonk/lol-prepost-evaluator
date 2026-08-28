"""add team and player media fields

Revision ID: ef2185db974b
Revises: e52c49458e50
Create Date: 2026-08-27 00:18:26.718095

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ef2185db974b"
down_revision: Union[str, Sequence[str], None] = "e52c49458e50"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable local-media paths for teams and players."""
    op.add_column(
        "team",
        sa.Column(
            "logo_file",
            sa.String(length=160),
            nullable=True,
        ),
    )
    op.add_column(
        "player",
        sa.Column(
            "photo_file",
            sa.String(length=160),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove team and player local-media paths."""
    op.drop_column("player", "photo_file")
    op.drop_column("team", "logo_file")
