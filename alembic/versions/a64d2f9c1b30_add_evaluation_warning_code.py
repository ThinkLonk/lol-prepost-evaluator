"""add evaluation warning code

Revision ID: a64d2f9c1b30
Revises: 8f3a1c7d2e90
Create Date: 2026-09-01

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a64d2f9c1b30"
down_revision: Union[str, Sequence[str], None] = "8f3a1c7d2e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Require a stable machine-readable code for each evaluation warning."""
    op.add_column(
        "evaluation_warning",
        sa.Column(
            "warning_code",
            sa.String(length=50),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Remove the machine-readable evaluation warning code."""
    op.drop_column("evaluation_warning", "warning_code")
