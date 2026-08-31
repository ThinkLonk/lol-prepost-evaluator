"""prepare Oracle ETL schema

Revision ID: 3b7f4c29a6d1
Revises: ef2185db974b
Create Date: 2026-08-30

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3b7f4c29a6d1"
down_revision: Union[str, Sequence[str], None] = "ef2185db974b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make the core schema compatible with the approved Oracle ETL rules."""
    op.alter_column(
        "tournament",
        "region",
        existing_type=sa.String(length=30),
        nullable=True,
    )

    op.add_column(
        "team",
        sa.Column(
            "oracle_team_id",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.create_unique_constraint(
        "uq_team_oracle_team_id",
        "team",
        ["oracle_team_id"],
    )

    op.add_column(
        "player",
        sa.Column(
            "oracle_player_id",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.create_unique_constraint(
        "uq_player_oracle_player_id",
        "player",
        ["oracle_player_id"],
    )

    op.drop_constraint(
        "ck_game_winner_requires_end",
        "game",
        type_="check",
    )
    op.create_foreign_key(
        "fk_game_winner_participant",
        "game",
        "game_team",
        ["game_id", "winner_team_id"],
        ["game_id", "team_id"],
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    """Restore the schema that preceded Oracle ETL preparation."""
    op.drop_constraint(
        "fk_game_winner_participant",
        "game",
        type_="foreignkey",
    )
    op.create_check_constraint(
        "ck_game_winner_requires_end",
        "game",
        "winner_team_id IS NULL OR ended_at IS NOT NULL",
    )

    op.drop_constraint(
        "uq_player_oracle_player_id",
        "player",
        type_="unique",
    )
    op.drop_column("player", "oracle_player_id")

    op.drop_constraint(
        "uq_team_oracle_team_id",
        "team",
        type_="unique",
    )
    op.drop_column("team", "oracle_team_id")

    op.alter_column(
        "tournament",
        "region",
        existing_type=sa.String(length=30),
        nullable=False,
    )
