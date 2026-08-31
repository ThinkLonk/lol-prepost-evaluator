from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from sqlalchemy import func, insert, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from match_insight.data_processing.oracle_transform import (
    ACTION_COLUMNS,
    GAME_COLUMNS,
    GAME_PATCH_COLUMNS,
    GAME_PLAYER_COLUMNS,
    GAME_TEAM_COLUMNS,
    ISSUE_COLUMNS,
    PLAYER_COLUMNS,
    SERIES_BLOCKED,
    SERIES_COLUMNS,
    TARGET_TABLE_ORDER,
    TEAM_COLUMNS,
    TOURNAMENT_COLUMNS,
    TOURNAMENT_STAGE_COLUMNS,
    UNMAPPED_CHAMPION_COLUMNS,
    OracleDryRunResult,
    OracleSeriesAnalysis,
    OracleTargetRecords,
)
from match_insight.database.engine import engine
from match_insight.database.models import (
    Champion,
    Evaluation,
    EvaluationWarning,
    Game,
    GamePatch,
    GamePlayer,
    GameTeam,
    Player,
    Team,
    TeamMembership,
    Tournament,
    TournamentStage,
)
from match_insight.database.oracle_apply import (
    ALREADY_APPLIED,
    APPROVED_INITIAL_STATE,
    TableApplyStats,
    apply_oracle_records,
    build_approval_contract,
    classify_approval_state,
)
from scripts.apply_oracle_etl import execute_approved_apply


@dataclass(frozen=True)
class ApplyFixture:
    result: OracleDryRunResult
    ids: dict[str, str]


def _frame(
    rows: list[dict[str, object]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=list(columns))


def _issue(
    entity: str,
    source_key: str,
    reason: str,
    detail: str = "",
) -> dict[str, str]:
    return {
        "entity": entity,
        "source_key": source_key,
        "reason": reason,
        "detail": detail,
    }


def _action(table: str, source_key: str, action: str) -> dict[str, str]:
    return {
        "table": table,
        "source_key": source_key,
        "action": action,
    }


def _make_apply_fixture(
    prefix: str,
    *,
    champion_id: str,
    existing_blue_references: bool,
) -> ApplyFixture:
    ids = {
        "patch": f"{prefix}-patch",
        "tournament": f"{prefix}-tournament",
        "stage": f"{prefix}-stage",
        "blue_team": f"{prefix}-blue-team",
        "red_team": f"{prefix}-red-team",
        "blue_player": f"{prefix}-blue-player",
        "red_player": f"{prefix}-red-player",
        "game": f"{prefix}-game",
        "unresolved_team": f"{prefix}-unresolved-team",
        "unresolved_player": f"{prefix}-unresolved-player",
        "rejected_game": f"{prefix}-rejected-game",
    }
    records = OracleTargetRecords(
        game_patches=_frame(
            [
                {
                    "patch_id": ids["patch"],
                    "patch_name": ids["patch"],
                }
            ],
            GAME_PATCH_COLUMNS,
        ),
        tournaments=_frame(
            [
                {
                    "tournament_id": ids["tournament"],
                    "name": "Apply Test League",
                    "region": None,
                    "season": 2025,
                }
            ],
            TOURNAMENT_COLUMNS,
        ),
        tournament_stages=_frame(
            [
                {
                    "stage_id": ids["stage"],
                    "tournament_id": ids["tournament"],
                    "name": "Spring",
                }
            ],
            TOURNAMENT_STAGE_COLUMNS,
        ),
        series=_frame([], SERIES_COLUMNS),
        teams=_frame(
            [
                {
                    "team_id": ids["blue_team"],
                    "oracle_team_id": f"{prefix}-oracle-blue-team",
                    "canonical_name": "Updated Blue Team",
                    "display_name": "Updated Blue Team",
                    "logo_file": None,
                },
                {
                    "team_id": ids["red_team"],
                    "oracle_team_id": f"{prefix}-oracle-red-team",
                    "canonical_name": "Red Team",
                    "display_name": "Red Team",
                    "logo_file": None,
                },
            ],
            TEAM_COLUMNS,
        ),
        players=_frame(
            [
                {
                    "player_id": ids["blue_player"],
                    "oracle_player_id": f"{prefix}-oracle-blue-player",
                    "canonical_name": "Updated Blue Player",
                    "display_name": "Updated Blue Player",
                    "photo_file": None,
                },
                {
                    "player_id": ids["red_player"],
                    "oracle_player_id": f"{prefix}-oracle-red-player",
                    "canonical_name": "Red Player",
                    "display_name": "Red Player",
                    "photo_file": None,
                },
            ],
            PLAYER_COLUMNS,
        ),
        games=_frame(
            [
                {
                    "game_id": ids["game"],
                    "series_id": None,
                    "stage_id": ids["stage"],
                    "patch_id": ids["patch"],
                    "game_number": 1,
                    "scheduled_at": pd.Timestamp("2025-01-01T12:00:00Z"),
                    "started_at": None,
                    "ended_at": None,
                    "winner_team_id": ids["blue_team"],
                }
            ],
            GAME_COLUMNS,
        ),
        game_teams=_frame(
            [
                {
                    "game_id": ids["game"],
                    "team_id": ids["blue_team"],
                    "side": "BLUE",
                    "confirmation_status": "SOURCE_REPORTED",
                },
                {
                    "game_id": ids["game"],
                    "team_id": ids["red_team"],
                    "side": "RED",
                    "confirmation_status": "SOURCE_REPORTED",
                },
            ],
            GAME_TEAM_COLUMNS,
        ),
        game_players=_frame(
            [
                {
                    "game_id": ids["game"],
                    "side": "BLUE",
                    "player_id": ids["blue_player"],
                    "role": "TOP",
                    "champion_id": champion_id,
                    "confirmation_status": "SOURCE_REPORTED",
                },
                {
                    "game_id": ids["game"],
                    "side": "RED",
                    "player_id": ids["red_player"],
                    "role": "TOP",
                    "champion_id": champion_id,
                    "confirmation_status": "SOURCE_REPORTED",
                },
            ],
            GAME_PLAYER_COLUMNS,
        ),
    )
    blue_action = "UPDATE" if existing_blue_references else "INSERT"
    actions = _frame(
        [
            _action("game_patch", ids["patch"], "INSERT"),
            _action("tournament", ids["tournament"], "INSERT"),
            _action("tournament_stage", ids["stage"], "INSERT"),
            _action("team", ids["blue_team"], blue_action),
            _action("team", ids["red_team"], "INSERT"),
            _action("player", ids["blue_player"], blue_action),
            _action("player", ids["red_player"], "INSERT"),
            _action("game", ids["game"], "INSERT"),
            _action("game_team", f"{ids['game']}|BLUE", "INSERT"),
            _action("game_team", f"{ids['game']}|RED", "INSERT"),
            _action("game_player", f"{ids['game']}|BLUE|TOP", "INSERT"),
            _action("game_player", f"{ids['game']}|RED|TOP", "INSERT"),
        ],
        ACTION_COLUMNS,
    )
    analysis = OracleSeriesAnalysis(
        status=SERIES_BLOCKED,
        series_records=pd.DataFrame(),
        diagnostics=pd.DataFrame(
            [
                {
                    "source_key": ids["game"],
                    "status": SERIES_BLOCKED,
                    "reason": "EXPLICIT_BEST_OF_MISSING",
                    "detail": "",
                }
            ]
        ),
        game_assignments=pd.DataFrame(
            [
                {
                    "gameid": ids["game"],
                    "series_id": pd.NA,
                    "series_status": SERIES_BLOCKED,
                    "series_reason": "EXPLICIT_BEST_OF_MISSING",
                }
            ]
        ),
    )
    result = OracleDryRunResult(
        records=records,
        unresolved_records=_frame(
            [
                _issue(
                    "team_participation",
                    ids["unresolved_team"],
                    "NO_SOURCE_ID_CANDIDATE",
                ),
                _issue(
                    "player_row",
                    ids["unresolved_player"],
                    "NO_SOURCE_ID_CANDIDATE",
                ),
            ],
            ISSUE_COLUMNS,
        ),
        rejected_records=_frame(
            [
                _issue(
                    "game",
                    ids["rejected_game"],
                    "GAME_METADATA_INVALID",
                    "MISSING_SPLIT",
                )
            ],
            ISSUE_COLUMNS,
        ),
        skipped_records=_frame([], ISSUE_COLUMNS),
        unmapped_champions=_frame([], UNMAPPED_CHAMPION_COLUMNS),
        series_analysis=analysis,
        actions=actions,
        source_counts={
            "source_rows": 25,
            "player_rows": 20,
            "team_rows": 5,
            "distinct_games": 2,
        },
    )
    return ApplyFixture(result=result, ids=ids)


def _approval_summary() -> dict[str, object]:
    transformed = {
        "game_patch": 1,
        "tournament": 1,
        "tournament_stage": 1,
        "series": 0,
        "team": 2,
        "player": 2,
        "game": 1,
        "game_team": 2,
        "game_player": 2,
    }
    actions = {
        table: {
            "inserted_expected": transformed[table],
            "updated_expected": 0,
            "skipped": 0,
        }
        for table in TARGET_TABLE_ORDER
    }
    return {
        "mode": "DRY_RUN",
        "source_filename": "oracle-fixture.csv",
        "source_path": str(Path("oracle-fixture.csv").resolve()),
        "source_sha256": "fixture-sha256",
        "source_sha256_before": "fixture-sha256",
        "source_sha256_after": "fixture-sha256",
        "source_sha256_unchanged": True,
        "run_timestamp_utc": "2026-08-31T12:00:00+00:00",
        "source_rows": 25,
        "player_rows": 20,
        "team_rows": 5,
        "distinct_games": 2,
        "transformed_records": transformed,
        "actions": actions,
        "action_totals": {
            "inserted_expected": sum(transformed.values()),
            "updated_expected": 0,
            "skipped": 0,
        },
        "target_snapshot_rows": {
            "champion": 173,
            **{table: 0 for table in TARGET_TABLE_ORDER},
        },
        "skipped": 0,
        "unresolved": 2,
        "rejected": 1,
        "unmapped_champions": 0,
        "series_status": SERIES_BLOCKED,
        "game_transform_status": "PARTIAL",
        "transform_ready": False,
        "dry_run_status": "PARTIAL",
        "cutoff_status": "UNAVAILABLE",
        "transaction_status": "READ_ONLY_ROLLED_BACK",
        "postgresql_changed": False,
        "reconciliation_valid": True,
        "grain_reconciliation_valid": True,
    }


@pytest.fixture
def database_connection() -> Iterator[Connection]:
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            yield connection
        finally:
            if transaction.is_active:
                transaction.rollback()


def _count_for_prefix(
    connection: Connection,
    model: type[object],
    column: object,
    prefix: str,
) -> int:
    statement = (
        select(func.count())
        .select_from(model)
        .where(column.like(f"{prefix}%"))
    )
    return int(connection.scalar(statement) or 0)


def _protected_counts(
    connection: Connection,
    ids: dict[str, str],
) -> dict[str, int]:
    evaluation_count = connection.scalar(
        select(func.count())
        .select_from(Evaluation)
        .where(Evaluation.game_id == ids["game"])
    )
    warning_count = connection.scalar(
        select(func.count())
        .select_from(EvaluationWarning)
        .join(
            Evaluation,
            Evaluation.evaluation_id == EvaluationWarning.evaluation_id,
        )
        .where(Evaluation.game_id == ids["game"])
    )
    membership_count = connection.scalar(
        select(func.count())
        .select_from(TeamMembership)
        .where(
            TeamMembership.team_id.in_(
                [ids["blue_team"], ids["red_team"]]
            )
        )
    )
    return {
        "evaluation": int(evaluation_count or 0),
        "evaluation_warning": int(warning_count or 0),
        "team_membership": int(membership_count or 0),
    }


def test_approval_contract_accepts_initial_and_idempotent_replay() -> None:
    approved = _approval_summary()
    approved_before = copy.deepcopy(approved)
    contract = build_approval_contract(approved)

    assert classify_approval_state(approved, copy.deepcopy(approved)) == (
        APPROVED_INITIAL_STATE
    )

    replay = copy.deepcopy(approved)
    replay["target_snapshot_rows"] = contract[
        "predicted_post_snapshot_rows"
    ]
    replay["actions"] = contract["replay_actions"]
    replay["skipped"] = sum(
        int(value) for value in approved["transformed_records"].values()
    )

    assert classify_approval_state(approved, replay) == ALREADY_APPLIED
    assert approved == approved_before


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("source_sha256", "changed-sha256"),
        ("source_sha256_after", "changed-sha256"),
        ("reconciliation_valid", False),
        ("grain_reconciliation_valid", False),
        ("postgresql_changed", True),
    ),
)
def test_approval_contract_rejects_unsafe_report(
    field: str,
    unsafe_value: object,
) -> None:
    summary = _approval_summary()
    summary[field] = unsafe_value

    with pytest.raises(ValueError):
        build_approval_contract(summary)


def test_approval_state_rejects_semantic_or_snapshot_drift() -> None:
    approved = _approval_summary()
    semantic_drift = copy.deepcopy(approved)
    semantic_drift["unresolved"] = 3
    snapshot_drift = copy.deepcopy(approved)
    snapshot_drift["target_snapshot_rows"]["game"] = 7

    with pytest.raises(ValueError):
        classify_approval_state(approved, semantic_drift)
    with pytest.raises(ValueError):
        classify_approval_state(approved, snapshot_drift)


@pytest.mark.parametrize(
    ("allow_partial", "commit", "message"),
    (
        (False, True, "allow-partial"),
        (True, False, "commit"),
    ),
)
def test_cli_requires_explicit_partial_and_commit_guards(
    allow_partial: bool,
    commit: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        execute_approved_apply(
            source=Path("must-not-be-read.csv"),
            approved_summary_path=Path("must-not-be-read.json"),
            allow_partial=allow_partial,
            commit=commit,
        )


@pytest.mark.parametrize("batch_size", (0, -1, True))
def test_apply_rejects_invalid_batch_size_before_writing(
    database_connection: Connection,
    batch_size: int,
) -> None:
    fixture = _make_apply_fixture(
        f"oa{uuid4().hex[:8]}",
        champion_id="missing-champion",
        existing_blue_references=False,
    )

    with pytest.raises(ValueError, match="batch_size"):
        apply_oracle_records(
            database_connection,
            fixture.result,
            batch_size=batch_size,
        )


def test_apply_rejects_record_action_mismatch_before_writing(
    database_connection: Connection,
) -> None:
    fixture = _make_apply_fixture(
        f"oa{uuid4().hex[:8]}",
        champion_id="missing-champion",
        existing_blue_references=False,
    )
    mismatched = replace(
        fixture.result,
        actions=fixture.result.actions.iloc[:-1].copy(),
    )

    with pytest.raises(ValueError, match="identical keys"):
        apply_oracle_records(database_connection, mismatched)


def test_apply_rejects_insert_conflict_with_exact_rowcount(
    database_connection: Connection,
) -> None:
    fixture = _make_apply_fixture(
        f"oa{uuid4().hex[:8]}",
        champion_id="missing-champion",
        existing_blue_references=False,
    )
    patch_id = fixture.ids["patch"]
    database_connection.execute(
        insert(GamePatch),
        [{"patch_id": patch_id, "patch_name": patch_id}],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"Oracle game_patch insert conflict or rowcount drift: "
            r"expected 1, wrote 0\."
        ),
    ):
        apply_oracle_records(database_connection, fixture.result)

    assert database_connection.scalar(
        select(func.count())
        .select_from(GamePatch)
        .where(GamePatch.patch_id == patch_id)
    ) == 1


def test_apply_maps_logical_game_team_links_and_preserves_existing_data(
    database_connection: Connection,
) -> None:
    prefix = f"oa{uuid4().hex[:8]}"
    champion_id = f"{prefix}-champion"
    fixture = _make_apply_fixture(
        prefix,
        champion_id=champion_id,
        existing_blue_references=True,
    )
    ids = fixture.ids
    logo_file = f"assets/teams/{prefix}-blue.png"
    photo_file = f"assets/players/{prefix}-blue.png"
    image_file = f"assets/champions/{prefix}.png"

    database_connection.execute(
        insert(Champion),
        [
            {
                "champion_id": champion_id,
                "canonical_name": "ApplyChampion",
                "display_name": "Apply Champion",
                "image_file": image_file,
            }
        ],
    )
    database_connection.execute(
        insert(Team),
        [
            {
                "team_id": ids["blue_team"],
                "oracle_team_id": f"{prefix}-oracle-blue-team",
                "canonical_name": "Old Blue Team",
                "display_name": "Old Blue Team",
                "logo_file": logo_file,
            }
        ],
    )
    database_connection.execute(
        insert(Player),
        [
            {
                "player_id": ids["blue_player"],
                "oracle_player_id": f"{prefix}-oracle-blue-player",
                "canonical_name": "Old Blue Player",
                "display_name": "Old Blue Player",
                "photo_file": photo_file,
            }
        ],
    )
    protected_counts_before = _protected_counts(database_connection, ids)
    assert protected_counts_before == {
        "evaluation": 0,
        "evaluation_warning": 0,
        "team_membership": 0,
    }

    applied = apply_oracle_records(database_connection, fixture.result)
    database_connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))

    assert applied.state == APPROVED_INITIAL_STATE
    assert applied.per_table["team"] == TableApplyStats(
        inserted=1,
        updated=1,
        skipped=0,
    )
    assert applied.per_table["player"] == TableApplyStats(
        inserted=1,
        updated=1,
        skipped=0,
    )
    assert applied.per_table["game_team"].inserted == 2
    assert applied.per_table["game_player"].inserted == 2

    game = database_connection.execute(
        select(
            Game.series_id,
            Game.stage_id,
            Game.scheduled_at,
            Game.started_at,
            Game.ended_at,
            Game.winner_team_id,
        ).where(Game.game_id == ids["game"])
    ).one()
    assert game.series_id is None
    assert game.stage_id == ids["stage"]
    assert pd.Timestamp(game.scheduled_at).tz_convert("UTC") == pd.Timestamp(
        "2025-01-01T12:00:00Z"
    )
    assert game.started_at is None
    assert game.ended_at is None
    assert game.winner_team_id == ids["blue_team"]

    participation_rows = database_connection.execute(
        select(GameTeam.side, GamePlayer.player_id, GamePlayer.champion_id)
        .join(GamePlayer, GamePlayer.game_team_id == GameTeam.game_team_id)
        .where(GameTeam.game_id == ids["game"])
    ).all()
    participants = {
        side: (player_id, mapped_champion_id)
        for side, player_id, mapped_champion_id in participation_rows
    }
    assert participants == {
        "BLUE": (ids["blue_player"], champion_id),
        "RED": (ids["red_player"], champion_id),
    }

    blue_team = database_connection.execute(
        select(Team.display_name, Team.logo_file).where(
            Team.team_id == ids["blue_team"]
        )
    ).one()
    blue_player = database_connection.execute(
        select(Player.display_name, Player.photo_file).where(
            Player.player_id == ids["blue_player"]
        )
    ).one()
    champion = database_connection.execute(
        select(Champion.image_file).where(
            Champion.champion_id == champion_id
        )
    ).one()
    assert blue_team.display_name == "Updated Blue Team"
    assert blue_team.logo_file == logo_file
    assert blue_player.display_name == "Updated Blue Player"
    assert blue_player.photo_file == photo_file
    assert champion.image_file == image_file

    assert database_connection.scalar(
        select(func.count())
        .select_from(Team)
        .where(Team.team_id == ids["unresolved_team"])
    ) == 0
    assert database_connection.scalar(
        select(func.count())
        .select_from(Player)
        .where(Player.player_id == ids["unresolved_player"])
    ) == 0
    assert database_connection.scalar(
        select(func.count())
        .select_from(Game)
        .where(Game.game_id == ids["rejected_game"])
    ) == 0

    protected_counts_after = _protected_counts(database_connection, ids)
    assert protected_counts_after == protected_counts_before


def test_apply_twice_does_not_duplicate_records(
    database_connection: Connection,
) -> None:
    prefix = f"oa{uuid4().hex[:8]}"
    champion_id = f"{prefix}-champion"
    fixture = _make_apply_fixture(
        prefix,
        champion_id=champion_id,
        existing_blue_references=False,
    )
    database_connection.execute(
        insert(Champion),
        [
            {
                "champion_id": champion_id,
                "canonical_name": "ReplayChampion",
                "display_name": "Replay Champion",
                "image_file": None,
            }
        ],
    )

    first = apply_oracle_records(database_connection, fixture.result)
    replay_result = replace(
        fixture.result,
        actions=fixture.result.actions.assign(action="SKIP"),
    )
    second = apply_oracle_records(database_connection, replay_result)
    database_connection.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))

    assert first.state == APPROVED_INITIAL_STATE
    assert second.state == ALREADY_APPLIED
    expected_skips = {
        "game_patch": 1,
        "tournament": 1,
        "tournament_stage": 1,
        "series": 0,
        "team": 2,
        "player": 2,
        "game": 1,
        "game_team": 2,
        "game_player": 2,
    }
    for table, skipped in expected_skips.items():
        assert second.per_table[table] == TableApplyStats(
            inserted=0,
            updated=0,
            skipped=skipped,
        )

    assert _count_for_prefix(
        database_connection,
        GamePatch,
        GamePatch.patch_id,
        prefix,
    ) == 1
    assert _count_for_prefix(
        database_connection,
        Tournament,
        Tournament.tournament_id,
        prefix,
    ) == 1
    assert _count_for_prefix(
        database_connection,
        TournamentStage,
        TournamentStage.stage_id,
        prefix,
    ) == 1
    assert _count_for_prefix(
        database_connection,
        Team,
        Team.team_id,
        prefix,
    ) == 2
    assert _count_for_prefix(
        database_connection,
        Player,
        Player.player_id,
        prefix,
    ) == 2
    assert _count_for_prefix(
        database_connection,
        Game,
        Game.game_id,
        prefix,
    ) == 1
    game_team_count = database_connection.scalar(
        select(func.count())
        .select_from(GameTeam)
        .where(GameTeam.game_id == fixture.ids["game"])
    )
    game_player_count = database_connection.scalar(
        select(func.count())
        .select_from(GamePlayer)
        .join(GameTeam, GameTeam.game_team_id == GamePlayer.game_team_id)
        .where(GameTeam.game_id == fixture.ids["game"])
    )
    assert game_team_count == 2
    assert game_player_count == 2


def test_late_foreign_key_error_rolls_back_the_entire_apply() -> None:
    prefix = f"oa{uuid4().hex[:8]}"
    missing_champion_id = f"{prefix}-missing-champion"
    fixture = _make_apply_fixture(
        prefix,
        champion_id=missing_champion_id,
        existing_blue_references=False,
    )

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with pytest.raises(IntegrityError):
                apply_oracle_records(connection, fixture.result)
        finally:
            if transaction.is_active:
                transaction.rollback()

    with engine.connect() as verification_connection:
        assert _count_for_prefix(
            verification_connection,
            GamePatch,
            GamePatch.patch_id,
            prefix,
        ) == 0
        assert _count_for_prefix(
            verification_connection,
            Tournament,
            Tournament.tournament_id,
            prefix,
        ) == 0
        assert _count_for_prefix(
            verification_connection,
            TournamentStage,
            TournamentStage.stage_id,
            prefix,
        ) == 0
        assert _count_for_prefix(
            verification_connection,
            Team,
            Team.team_id,
            prefix,
        ) == 0
        assert _count_for_prefix(
            verification_connection,
            Player,
            Player.player_id,
            prefix,
        ) == 0
        assert _count_for_prefix(
            verification_connection,
            Game,
            Game.game_id,
            prefix,
        ) == 0
        assert verification_connection.scalar(
            select(func.count())
            .select_from(GameTeam)
            .where(GameTeam.game_id == fixture.ids["game"])
        ) == 0
        assert verification_connection.scalar(
            select(func.count())
            .select_from(GamePlayer)
            .join(
                GameTeam,
                GameTeam.game_team_id == GamePlayer.game_team_id,
            )
            .where(GameTeam.game_id == fixture.ids["game"])
        ) == 0
