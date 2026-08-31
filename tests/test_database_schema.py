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
    assert foreign_key_count == 17
    assert check_constraint_count == 8
    assert unique_constraint_count == 5


def test_oracle_identity_columns_are_nullable_and_unique() -> None:
    inspector = inspect(engine)
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


def test_tournament_region_is_nullable() -> None:
    inspector = inspect(engine)
    columns = {
        column["name"]: column
        for column in inspector.get_columns("tournament")
    }

    assert columns["region"]["nullable"] is True


def test_game_has_exactly_one_series_or_stage_parent() -> None:
    inspector = inspect(engine)
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


def test_game_winner_constraint_uses_participating_team() -> None:
    inspector = inspect(engine)
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

    
def test_reference_media_columns_exist() -> None:
    inspector = inspect(engine)

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
