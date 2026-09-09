"""add game temporal guard

Revision ID: f2c8a7319d44
Revises: a64d2f9c1b30
Create Date: 2026-09-02

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f2c8a7319d44"
down_revision: Union[str, Sequence[str], None] = "a64d2f9c1b30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Guard verified game timestamps and index temporal history lookups."""
    op.create_check_constraint(
        "ck_game_time_order",
        "game",
        "started_at IS NULL OR ended_at IS NULL OR started_at < ended_at",
    )
    op.create_index(
        "ix_game_ended_at",
        "game",
        ["ended_at"],
        unique=False,
        postgresql_where=sa.text("ended_at IS NOT NULL"),
    )


def downgrade() -> None:
    """Remove the temporal lookup index and ordering guard."""
    op.drop_index(
        "ix_game_ended_at",
        table_name="game",
        postgresql_where=sa.text("ended_at IS NOT NULL"),
    )
    op.drop_constraint(
        "ck_game_time_order",
        "game",
        type_="check",
    )
