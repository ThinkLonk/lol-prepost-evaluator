"""allow games without resolved series

Revision ID: 8f3a1c7d2e90
Revises: 3b7f4c29a6d1
Create Date: 2026-08-31

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f3a1c7d2e90"
down_revision: Union[str, Sequence[str], None] = "3b7f4c29a6d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Allow a game to belong to a series or directly to a stage."""
    op.alter_column(
        "game",
        "series_id",
        existing_type=sa.String(length=80),
        nullable=True,
    )
    op.add_column(
        "game",
        sa.Column(
            "stage_id",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "fk_game_stage",
        "game",
        "tournament_stage",
        ["stage_id"],
        ["stage_id"],
    )
    op.create_check_constraint(
        "ck_game_exactly_one_parent",
        "game",
        "(series_id IS NOT NULL AND stage_id IS NULL) OR "
        "(series_id IS NULL AND stage_id IS NOT NULL)",
    )


def downgrade() -> None:
    """Restore mandatory series ownership without inventing a series."""
    # Fail before dropping stage context if stage-backed games still exist.
    op.alter_column(
        "game",
        "series_id",
        existing_type=sa.String(length=80),
        nullable=False,
    )
    op.drop_constraint(
        "ck_game_exactly_one_parent",
        "game",
        type_="check",
    )
    op.drop_constraint(
        "fk_game_stage",
        "game",
        type_="foreignkey",
    )
    op.drop_column("game", "stage_id")
