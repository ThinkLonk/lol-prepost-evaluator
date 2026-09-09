"""Opt-in, read-only PostgreSQL schema assertions for the applied migration chain."""

import os

import pytest
from sqlalchemy import Text, inspect, text
from sqlalchemy.dialects.postgresql import JSONB

CORE_TABLES = {
    "tournament",
    "tournament_stage",
    "series",
    "game_patch",
    "game",
    "team",
    "player",
    "team_membership",
    "game_team",
    "game_player",
    "champion",
    "evaluation",
    "evaluation_warning",
    "analysis_session",
    "evaluation_history",
}

EXPECTED_INDEXES = {
    "ix_game_scheduled_at",
    "ix_team_membership_player_validity",
    "ix_team_membership_team_validity",
    "ix_game_team_team_id",
    "ix_game_player_champion_id",
    "ix_game_player_player_id",
    "ix_evaluation_game_type",
    "uq_evaluation_one_active_model",
    "ix_game_ended_at",
    "ix_evaluation_analysis_type",
    "uq_evaluation_one_active_analysis_model",
}


@pytest.fixture(scope="module")
def schema_connection():
    if os.environ.get("MATCH_INSIGHT_STORAGE_DB_TESTS") != "1":
        pytest.skip("Read-only PostgreSQL schema tests require MATCH_INSIGHT_STORAGE_DB_TESTS=1")
    from match_insight.database.engine import engine

    with engine.connect() as connection:
        connection = connection.execution_options(isolation_level="REPEATABLE READ")
        with connection.begin():
            connection.execute(text("SET TRANSACTION READ ONLY"))
            assert connection.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
            connection.execute(text("SET LOCAL search_path TO public"))
            yield connection


def test_core_tables_exist(schema_connection) -> None:
    inspector = inspect(schema_connection)
    actual_tables = set(inspector.get_table_names(schema="public"))

    assert CORE_TABLES <= actual_tables


def test_core_constraint_counts(schema_connection) -> None:
    inspector = inspect(schema_connection)
    tables = sorted(CORE_TABLES)

    primary_key_count = sum(
        bool(inspector.get_pk_constraint(table).get("constrained_columns"))
        for table in tables
    )
    foreign_key_count = sum(
        len(inspector.get_foreign_keys(table)) for table in tables
    )
    check_constraint_count = sum(
        len(inspector.get_check_constraints(table)) for table in tables
    )
    unique_constraint_count = sum(
        len(inspector.get_unique_constraints(table)) for table in tables
    )

    # f2c8a7319d44 adds one temporal check; c6e31a9d4b72 adds two storage
    # tables, two FKs, nine checks and two unique constraints to the old schema.
    assert primary_key_count == 15
    assert foreign_key_count == 19
    assert check_constraint_count == 18
    assert unique_constraint_count == 7


def test_oracle_identity_columns_are_nullable_and_unique(schema_connection) -> None:
    inspector = inspect(schema_connection)
    team_columns = {
        column["name"]: column for column in inspector.get_columns("team")
    }
    player_columns = {
        column["name"]: column for column in inspector.get_columns("player")
    }

    assert team_columns["oracle_team_id"]["nullable"] is True
    assert player_columns["oracle_player_id"]["nullable"] is True

    team_unique_constraint = next(
        (
            constraint
            for constraint in inspector.get_unique_constraints("team")
            if tuple(constraint.get("column_names") or ())
            == ("oracle_team_id",)
        ),
        None,
    )
    player_unique_constraint = next(
        (
            constraint
            for constraint in inspector.get_unique_constraints("player")
            if tuple(constraint.get("column_names") or ())
            == ("oracle_player_id",)
        ),
        None,
    )

    assert team_unique_constraint is not None
    assert team_unique_constraint["name"] == "uq_team_oracle_team_id"
    assert player_unique_constraint is not None
    assert player_unique_constraint["name"] == "uq_player_oracle_player_id"


def test_tournament_region_is_nullable(schema_connection) -> None:
    inspector = inspect(schema_connection)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("tournament")
    }

    assert columns["region"]["nullable"] is True


def test_game_has_exactly_one_series_or_stage_parent(schema_connection) -> None:
    inspector = inspect(schema_connection)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("game")
    }
    foreign_keys = inspector.get_foreign_keys("game")
    check_constraints = inspector.get_check_constraints("game")

    assert columns["series_id"]["nullable"] is True
    assert columns["stage_id"]["nullable"] is True

    series_foreign_key = next(
        (
            foreign_key
            for foreign_key in foreign_keys
            if tuple(foreign_key.get("constrained_columns") or ())
            == ("series_id",)
            and foreign_key.get("referred_table") == "series"
            and tuple(foreign_key.get("referred_columns") or ())
            == ("series_id",)
        ),
        None,
    )
    stage_foreign_key = next(
        (
            foreign_key
            for foreign_key in foreign_keys
            if tuple(foreign_key.get("constrained_columns") or ())
            == ("stage_id",)
            and foreign_key.get("referred_table") == "tournament_stage"
            and tuple(foreign_key.get("referred_columns") or ())
            == ("stage_id",)
        ),
        None,
    )
    parent_check = next(
        (
            constraint
            for constraint in check_constraints
            if constraint.get("name") == "ck_game_exactly_one_parent"
        ),
        None,
    )

    assert series_foreign_key is not None
    assert stage_foreign_key is not None
    assert stage_foreign_key["name"] == "fk_game_stage"
    assert parent_check is not None

    sql_text = str(parent_check.get("sqltext", "")).lower()
    sql_text = " ".join(
        sql_text.replace("(", " ").replace(")", " ").split()
    )
    assert (
        "series_id is not null and stage_id is null or "
        "series_id is null and stage_id is not null"
    ) in sql_text


def test_game_winner_constraint_uses_participating_team(schema_connection) -> None:
    inspector = inspect(schema_connection)
    check_constraint_names = {
        constraint["name"]
        for constraint in inspector.get_check_constraints("game")
    }
    foreign_keys = inspector.get_foreign_keys("game")
    winner_foreign_key = next(
        (
            foreign_key
            for foreign_key in foreign_keys
            if tuple(foreign_key.get("constrained_columns") or ())
            == ("game_id", "winner_team_id")
            and foreign_key.get("referred_table") == "game_team"
            and tuple(foreign_key.get("referred_columns") or ())
            == ("game_id", "team_id")
        ),
        None,
    )

    assert "ck_game_winner_requires_end" not in check_constraint_names
    assert winner_foreign_key is not None
    assert winner_foreign_key["name"] == "fk_game_winner_participant"

    options = winner_foreign_key.get("options") or {}

    assert options.get("deferrable") is True
    assert str(options.get("initially", "")).upper() == "DEFERRED"


def test_pre_evaluation_foreign_key_exists(schema_connection) -> None:
    inspector = inspect(schema_connection)
    foreign_keys = inspector.get_foreign_keys("evaluation")

    pre_reference_exists = any(
        foreign_key["constrained_columns"] == ["pre_evaluation_id"]
        and foreign_key["referred_table"] == "evaluation"
        and foreign_key["referred_columns"] == ["evaluation_id"]
        for foreign_key in foreign_keys
    )

    assert pre_reference_exists


def test_active_evaluation_partial_unique_index_exists(schema_connection) -> None:
    inspector = inspect(schema_connection)
    indexes = inspector.get_indexes("evaluation")

    partial_index = next(
        (
            index
            for index in indexes
            if index["name"] == "uq_evaluation_one_active_model"
        ),
        None,
    )

    assert partial_index is not None
    assert partial_index["unique"] is True
    assert partial_index["column_names"] == [
        "game_id",
        "evaluation_type",
        "model_version",
    ]

    where_clause = partial_index.get("dialect_options", {}).get(
        "postgresql_where"
    )

    assert where_clause is not None
    assert "is_active is true" in str(where_clause).lower()

def test_reference_media_columns_exist(schema_connection) -> None:
    inspector = inspect(schema_connection)

    team_columns = {
        column["name"] for column in inspector.get_columns("team")
    }
    player_columns = {
        column["name"] for column in inspector.get_columns("player")
    }
    champion_columns = {
        column["name"] for column in inspector.get_columns("champion")
    }

    assert "logo_file" in team_columns
    assert "photo_file" in player_columns
    assert "image_file" in champion_columns


def test_required_indexes_and_applied_revision_exist(schema_connection) -> None:
    inspector = inspect(schema_connection)
    indexes = {index["name"] for table in CORE_TABLES for index in inspector.get_indexes(table)}
    assert EXPECTED_INDEXES <= indexes
    assert schema_connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all() == [
        "c6e31a9d4b72",
    ]
    game_checks = {item["name"] for item in inspector.get_check_constraints("game")}
    assert "ck_game_time_order" in game_checks


def test_persistence_subject_history_and_provenance_schema(schema_connection) -> None:
    inspector = inspect(schema_connection)
    columns = {column["name"]: column for column in inspector.get_columns("evaluation")}
    assert columns["game_id"]["nullable"] is True
    assert columns["analysis_id"]["nullable"] is True
    assert columns["inferred_at"]["nullable"] is True
    for name in ("context_key", "idempotency_key", "history_sha256", "inference_mode", "origin",
                 "provenance", "input_snapshot", "data_version", "model_version"):
        assert columns[name]["nullable"] is False
    assert isinstance(columns["data_version"]["type"], Text)
    assert isinstance(columns["model_version"]["type"], Text)
    assert isinstance(columns["input_snapshot"]["type"], JSONB)
    assert isinstance(columns["provenance"]["type"], JSONB)
    for name in ("created_at", "inferred_at", "history_cutoff_at"):
        assert columns[name]["type"].timezone is True

    foreign_keys = {item["name"]: item for item in inspector.get_foreign_keys("evaluation")}
    assert foreign_keys["fk_evaluation_analysis"]["constrained_columns"] == ["analysis_id"]
    assert foreign_keys["fk_evaluation_analysis"]["referred_table"] == "analysis_session"
    assert foreign_keys["fk_evaluation_history"]["constrained_columns"] == ["history_sha256"]
    assert foreign_keys["fk_evaluation_history"]["referred_table"] == "evaluation_history"
    checks = {item["name"]: item["sqltext"]
              for item in inspector.get_check_constraints("evaluation")}
    for name in ("ck_evaluation_subject", "ck_evaluation_context_key", "ck_evaluation_idempotency_key",
                 "ck_evaluation_provenance", "ck_evaluation_snapshot"):
        assert name in checks
    subject = checks["ck_evaluation_subject"]
    assert "INTERACTIVE_ANALYSIS" in subject and "REAL_RETROSPECTIVE_SIMULATION" in subject
    assert "INTERACTIVE" in subject and "RETROSPECTIVE_IMPORT" in subject
    unique = {item["name"]: item["column_names"]
              for item in inspector.get_unique_constraints("evaluation")}
    assert unique["uq_evaluation_idempotency_key"] == ["idempotency_key"]

    history = {column["name"]: column for column in inspector.get_columns("evaluation_history")}
    assert isinstance(history["payload"]["type"], JSONB)
    assert history["payload"]["nullable"] is False
    assert inspector.get_pk_constraint("evaluation_history")["constrained_columns"] == [
        "history_sha256",
    ]
    assert inspector.get_pk_constraint("analysis_session")["constrained_columns"] == ["analysis_id"]


def test_warning_order_and_analysis_active_index(schema_connection) -> None:
    inspector = inspect(schema_connection)
    columns = {column["name"]: column for column in inspector.get_columns("evaluation_warning")}
    for name in ("warning_code", "position", "details"):
        assert columns[name]["nullable"] is False
    assert columns["warning_code"]["type"].length == 50
    assert isinstance(columns["details"]["type"], JSONB)
    unique = {item["name"]: item["column_names"]
              for item in inspector.get_unique_constraints("evaluation_warning")}
    assert unique["uq_evaluation_warning_position"] == ["evaluation_id", "position"]
    index = next(item for item in inspector.get_indexes("evaluation")
                 if item["name"] == "uq_evaluation_one_active_analysis_model")
    assert index["unique"] is True
    assert index["column_names"] == ["analysis_id", "evaluation_type", "model_version"]
    predicate = str(index["dialect_options"]["postgresql_where"]).lower()
    assert "is_active is true" in predicate and "analysis_id is not null" in predicate


def test_persistence_immutability_and_parent_triggers_exist(schema_connection) -> None:
    rows = schema_connection.execute(text("""
        SELECT c.relname AS table_name, t.tgname AS trigger_name, p.proname AS function_name,
               t.tgtype, t.tgenabled, t.tgdeferrable, t.tginitdeferred
        FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_proc p ON p.oid = t.tgfoid
        WHERE n.nspname = 'public' AND NOT t.tgisinternal
          AND c.relname IN ('evaluation', 'evaluation_warning', 'evaluation_history', 'analysis_session')
    """)).mappings().all()
    by_name = {row["trigger_name"]: row for row in rows}
    for table in ("analysis_session", "evaluation_history", "evaluation", "evaluation_warning"):
        trigger = by_name[f"{table}_immutable"]
        assert trigger["table_name"] == table
        assert trigger["function_name"] == "mi_evaluation_immutable"
        assert trigger["tgtype"] == 27  # ROW | BEFORE | DELETE | UPDATE
        assert trigger["tgenabled"] == "O"
    parent = by_name["evaluation_pre_parent"]
    assert parent["function_name"] == "mi_evaluation_pre_parent"
    assert parent["tgtype"] == 7  # ROW | BEFORE | INSERT
    assert parent["tgenabled"] == "O"
    active = by_name["evaluation_active_parent"]
    assert active["function_name"] == "mi_evaluation_active_parent"
    assert active["tgtype"] == 21  # ROW | AFTER | INSERT | UPDATE
    assert active["tgenabled"] == "O"
    assert active["tgdeferrable"] is True and active["tginitdeferred"] is True
