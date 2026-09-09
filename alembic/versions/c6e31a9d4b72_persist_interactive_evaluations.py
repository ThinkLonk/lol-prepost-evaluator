"""Persist independent interactive analyses and immutable PRE/POST pairs.

Revision ID: c6e31a9d4b72
Revises: f2c8a7319d44
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "c6e31a9d4b72"
down_revision = "f2c8a7319d44"
branch_labels = None
depends_on = None


def _require_empty_evaluations():
    """Never invent provenance for legacy rows that need a separately reviewed import."""
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM evaluation)
               OR EXISTS (SELECT 1 FROM evaluation_warning) THEN
                RAISE EXCEPTION 'Evaluation persistence migration requires empty legacy tables';
            END IF;
        END $$
    """)


def upgrade() -> None:
    _require_empty_evaluations()
    op.create_table(
        "analysis_session",
        sa.Column("analysis_id", sa.String(100), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "evaluation_history",
        sa.Column("history_sha256", sa.String(64), primary_key=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("history_sha256 ~ '^[0-9a-f]{64}$'", name="ck_evaluation_history_hash"),
        sa.CheckConstraint("jsonb_typeof(payload) = 'object'", name="ck_evaluation_history_payload"),
    )
    op.alter_column("evaluation", "game_id", existing_type=sa.String(100), nullable=True)
    for column in (
        sa.Column("analysis_id", sa.String(100), nullable=True),
        sa.Column("inference_mode", sa.String(40), nullable=False),
        sa.Column("origin", sa.String(32), nullable=False),
        sa.Column("context_key", sa.String(64), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("inferred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provenance", postgresql.JSONB(), nullable=False),
        sa.Column("history_sha256", sa.String(64), nullable=False),
    ):
        op.add_column("evaluation", column)
    op.create_foreign_key(
        "fk_evaluation_analysis", "evaluation", "analysis_session", ["analysis_id"], ["analysis_id"],
    )
    op.create_foreign_key(
        "fk_evaluation_history", "evaluation", "evaluation_history",
        ["history_sha256"], ["history_sha256"],
    )
    for column in ("data_version", "model_version"):
        op.alter_column("evaluation", column, existing_type=sa.String(50), type_=sa.Text())
    op.create_check_constraint(
        "ck_evaluation_subject", "evaluation",
        "(game_id IS NULL AND analysis_id IS NOT NULL "
        "AND inference_mode = 'INTERACTIVE_ANALYSIS' AND origin = 'INTERACTIVE') OR "
        "(game_id IS NOT NULL AND analysis_id IS NULL "
        "AND inference_mode = 'REAL_RETROSPECTIVE_SIMULATION' AND origin = 'RETROSPECTIVE_IMPORT')",
    )
    for name, condition in (
        ("ck_evaluation_context_key", "context_key ~ '^[0-9a-f]{64}$'"),
        ("ck_evaluation_idempotency_key", "idempotency_key ~ '^[0-9a-f]{64}$'"),
        ("ck_evaluation_provenance", "jsonb_typeof(provenance) = 'object'"),
        ("ck_evaluation_snapshot", "jsonb_typeof(input_snapshot) = 'object'"),
    ):
        op.create_check_constraint(name, "evaluation", condition)
    op.create_unique_constraint("uq_evaluation_idempotency_key", "evaluation", ["idempotency_key"])
    op.create_index("ix_evaluation_analysis_type", "evaluation", ["analysis_id", "evaluation_type"])
    op.create_index(
        "uq_evaluation_one_active_analysis_model", "evaluation",
        ["analysis_id", "evaluation_type", "model_version"], unique=True,
        postgresql_where=sa.text("is_active IS TRUE AND analysis_id IS NOT NULL"),
    )
    op.add_column("evaluation_warning", sa.Column("position", sa.Integer(), nullable=False))
    op.add_column("evaluation_warning", sa.Column("details", postgresql.JSONB(), nullable=False))
    op.create_unique_constraint(
        "uq_evaluation_warning_position", "evaluation_warning", ["evaluation_id", "position"],
    )
    op.create_check_constraint("ck_evaluation_warning_position", "evaluation_warning", "position >= 0")
    op.create_check_constraint(
        "ck_evaluation_warning_details", "evaluation_warning", "jsonb_typeof(details) = 'object'",
    )
    _create_guards()


def _create_guards():
    op.execute("""
        CREATE FUNCTION mi_evaluation_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'Persisted evaluation records are immutable';
            END IF;
            IF TG_TABLE_NAME = 'evaluation'
               AND (to_jsonb(NEW) - 'is_active') = (to_jsonb(OLD) - 'is_active') THEN
                RETURN NEW;
            END IF;
            RAISE EXCEPTION 'Persisted evaluation records are immutable';
        END $$
    """)
    for table in ("analysis_session", "evaluation_history", "evaluation", "evaluation_warning"):
        op.execute(sa.text(
            f"CREATE TRIGGER {table}_immutable BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION mi_evaluation_immutable()"
        ))
    op.execute("""
        CREATE FUNCTION mi_evaluation_pre_parent() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE parent evaluation%ROWTYPE;
        BEGIN
            IF NEW.evaluation_type = 'POST' THEN
                SELECT * INTO parent FROM evaluation
                WHERE evaluation_id = NEW.pre_evaluation_id FOR UPDATE;
                IF NOT FOUND OR parent.evaluation_type <> 'PRE' OR NOT parent.is_active
                   OR parent.game_id IS DISTINCT FROM NEW.game_id
                   OR parent.analysis_id IS DISTINCT FROM NEW.analysis_id
                   OR parent.context_key IS DISTINCT FROM NEW.context_key
                   OR parent.history_cutoff_at IS DISTINCT FROM NEW.history_cutoff_at
                   OR parent.model_version IS DISTINCT FROM NEW.model_version
                   OR parent.data_version IS DISTINCT FROM NEW.data_version
                   OR parent.history_sha256 IS DISTINCT FROM NEW.history_sha256
                   OR parent.inference_mode IS DISTINCT FROM NEW.inference_mode
                   OR parent.origin IS DISTINCT FROM NEW.origin THEN
                    RAISE EXCEPTION 'POST requires its compatible active PRE';
                END IF;
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER evaluation_pre_parent BEFORE INSERT ON evaluation
        FOR EACH ROW EXECUTE FUNCTION mi_evaluation_pre_parent()
    """)
    op.execute("""
        CREATE FUNCTION mi_evaluation_active_parent() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM evaluation child JOIN evaluation parent
                  ON parent.evaluation_id = child.pre_evaluation_id
                WHERE child.evaluation_type = 'POST' AND child.is_active AND NOT parent.is_active
                  AND (child.evaluation_id = NEW.evaluation_id
                       OR parent.evaluation_id = NEW.evaluation_id)
            ) THEN
                RAISE EXCEPTION 'Active POST cannot reference an inactive PRE';
            END IF;
            RETURN NULL;
        END $$
    """)
    op.execute("""
        CREATE CONSTRAINT TRIGGER evaluation_active_parent AFTER INSERT OR UPDATE ON evaluation
        DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION mi_evaluation_active_parent()
    """)


def downgrade() -> None:
    _require_empty_evaluations()
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM analysis_session)
               OR EXISTS (SELECT 1 FROM evaluation_history) THEN
                RAISE EXCEPTION 'Cannot discard persisted analysis/history records';
            END IF;
        END $$
    """)
    op.execute("DROP TRIGGER evaluation_active_parent ON evaluation")
    op.execute("DROP FUNCTION mi_evaluation_active_parent()")
    op.execute("DROP TRIGGER evaluation_pre_parent ON evaluation")
    op.execute("DROP FUNCTION mi_evaluation_pre_parent()")
    for table in ("analysis_session", "evaluation_history", "evaluation", "evaluation_warning"):
        op.execute(sa.text(f"DROP TRIGGER {table}_immutable ON {table}"))
    op.execute("DROP FUNCTION mi_evaluation_immutable()")
    for name in ("ck_evaluation_warning_details", "ck_evaluation_warning_position"):
        op.drop_constraint(name, "evaluation_warning", type_="check")
    op.drop_constraint("uq_evaluation_warning_position", "evaluation_warning", type_="unique")
    op.drop_column("evaluation_warning", "details")
    op.drop_column("evaluation_warning", "position")
    op.drop_index("uq_evaluation_one_active_analysis_model", table_name="evaluation")
    op.drop_index("ix_evaluation_analysis_type", table_name="evaluation")
    op.drop_constraint("uq_evaluation_idempotency_key", "evaluation", type_="unique")
    for name in (
        "ck_evaluation_subject", "ck_evaluation_context_key", "ck_evaluation_idempotency_key",
        "ck_evaluation_provenance", "ck_evaluation_snapshot",
    ):
        op.drop_constraint(name, "evaluation", type_="check")
    op.drop_constraint("fk_evaluation_history", "evaluation", type_="foreignkey")
    op.drop_constraint("fk_evaluation_analysis", "evaluation", type_="foreignkey")
    for column in (
        "history_sha256", "provenance", "inferred_at", "idempotency_key", "context_key",
        "origin", "inference_mode", "analysis_id",
    ):
        op.drop_column("evaluation", column)
    for column in ("data_version", "model_version"):
        op.alter_column("evaluation", column, existing_type=sa.Text(), type_=sa.String(50))
    op.alter_column("evaluation", "game_id", existing_type=sa.String(100), nullable=False)
    op.drop_table("evaluation_history")
    op.drop_table("analysis_session")
