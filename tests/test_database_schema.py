"""Kiểm thử tích hợp tối thiểu cho lược đồ PostgreSQL."""

from sqlalchemy import inspect

from match_insight.database.engine import engine

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
}


def test_core_tables_exist() -> None:
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names(schema="public"))

    assert CORE_TABLES <= actual_tables


def test_core_constraint_counts() -> None:
    inspector = inspect(engine)
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

    assert primary_key_count == 13
    assert foreign_key_count == 15
    assert check_constraint_count == 8
    assert unique_constraint_count == 3


def test_pre_evaluation_foreign_key_exists() -> None:
    inspector = inspect(engine)
    foreign_keys = inspector.get_foreign_keys("evaluation")

    pre_reference_exists = any(
        foreign_key["constrained_columns"] == ["pre_evaluation_id"]
        and foreign_key["referred_table"] == "evaluation"
        and foreign_key["referred_columns"] == ["evaluation_id"]
        for foreign_key in foreign_keys
    )

    assert pre_reference_exists


def test_active_evaluation_partial_unique_index_exists() -> None:
    inspector = inspect(engine)
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