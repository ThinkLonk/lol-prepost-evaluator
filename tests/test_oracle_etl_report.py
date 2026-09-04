from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from match_insight.data_processing.oracle_etl_report import (
    ETL_REPORT_FILENAME,
    ETL_SUMMARY_FILENAME,
    REJECTED_GAMES_FILENAME,
    REJECTED_RECORDS_FILENAME,
    UNMAPPED_CHAMPIONS_FILENAME,
    UNRESOLVED_IDENTITIES_FILENAME,
    _sorted_frame,
    build_etl_report_summary,
    build_grain_reconciliation,
    derive_dry_run_state,
    write_etl_report_outputs,
)
from match_insight.data_processing.oracle_transform import (
    ACTION_COLUMNS,
    GAME_COLUMNS,
    GAME_PATCH_COLUMNS,
    GAME_PLAYER_COLUMNS,
    GAME_TEAM_COLUMNS,
    ISSUE_COLUMNS,
    PLAYER_COLUMNS,
    REJECTED_GAME_COLUMNS,
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


def _make_result() -> OracleDryRunResult:
    games = _frame(
        [
            {
                "game_id": "G1",
                "series_id": None,
                "stage_id": "STAGE-1",
                "patch_id": "15.1",
                "game_number": 1,
                "scheduled_at": None,
                "started_at": None,
                "ended_at": None,
                "winner_team_id": "TEAM-BLUE",
            }
        ],
        GAME_COLUMNS,
    )
    game_teams = _frame(
        [
            {
                "game_id": "G1",
                "team_id": "TEAM-BLUE",
                "side": "BLUE",
                "confirmation_status": "SOURCE_REPORTED",
            },
            {
                "game_id": "G1",
                "team_id": "TEAM-RED",
                "side": "RED",
                "confirmation_status": "SOURCE_REPORTED",
            },
        ],
        GAME_TEAM_COLUMNS,
    )
    roles = ("TOP", "JUNGLE", "MID", "BOT", "SUPPORT")
    game_players = _frame(
        [
            {
                "game_id": "G1",
                "side": side,
                "player_id": f"PLAYER-{side}-{role}",
                "role": role,
                "champion_id": f"CHAMPION-{side}-{role}",
                "confirmation_status": "SOURCE_REPORTED",
            }
            for side in ("BLUE", "RED")
            for role in roles
        ],
        GAME_PLAYER_COLUMNS,
    )
    records = OracleTargetRecords(
        game_patches=_frame(
            [{"patch_id": "15.1", "patch_name": "15.1"}],
            GAME_PATCH_COLUMNS,
        ),
        tournaments=_frame(
            [
                {
                    "tournament_id": "TOURNAMENT-1",
                    "name": "Test League",
                    "region": None,
                    "season": 2025,
                }
            ],
            TOURNAMENT_COLUMNS,
        ),
        tournament_stages=_frame(
            [
                {
                    "stage_id": "STAGE-1",
                    "tournament_id": "TOURNAMENT-1",
                    "name": "Spring",
                }
            ],
            TOURNAMENT_STAGE_COLUMNS,
        ),
        series=_frame([], SERIES_COLUMNS),
        teams=_frame([], TEAM_COLUMNS),
        players=_frame([], PLAYER_COLUMNS),
        games=games,
        game_teams=game_teams,
        game_players=game_players,
    )

    unresolved = _frame(
        [
            _issue("series", "G1", "EXPLICIT_BEST_OF_MISSING"),
            _issue("series", "G2", "SOURCE_SERIES_KEY_MISSING"),
            _issue("team_participation", "G2|Blue", "TEAM_UNRESOLVED"),
            _issue("player_row", "G2|Blue|top|10", "NO_SOURCE_ID_CANDIDATE"),
            _issue("player_row", "G2|Red|mid|11", "NO_SOURCE_ID_CANDIDATE"),
        ],
        ISSUE_COLUMNS,
    )
    rejected_rows = [
        _issue(
            "game",
            "G2",
            "GAME_LINEUP_INCOMPLETE",
            "NO_SOURCE_ID_CANDIDATE,TEAM_ROLE_SET_INCOMPLETE",
        ),
        _issue(
            "game_player",
            "G2|Blue|top|10",
            "NO_SOURCE_ID_CANDIDATE",
            "Lineup preflight failed for this source row.",
        ),
        _issue("game_team", "G2|Blue|20", "GAME_TEAM_BLOCKED_BY_GAME"),
        _issue("game_team", "G2|Red|21", "GAME_TEAM_BLOCKED_BY_GAME"),
    ]
    rejected_rows.extend(
        _issue(
            "game_player",
            f"G2|side|role|{index}",
            "GAME_PLAYER_BLOCKED_BY_GAME",
        )
        for index in range(10)
    )
    rejected = _frame(rejected_rows, ISSUE_COLUMNS)
    rejected_games = _frame(
        [
            {
                "gameid": "G2",
                "primary_reason": "GAME_LINEUP_INCOMPLETE",
                "secondary_reasons": json.dumps(
                    [
                        "NO_SOURCE_ID_CANDIDATE",
                        "TEAM_ROLE_SET_INCOMPLETE",
                    ]
                ),
                "league": "Test League",
                "source_year": 2025,
                "calendar_date": "2025-01-02",
                "patch": "15.1",
                "split": "Spring",
                "datacompleteness": "partial",
                "blue_team": "Blue Team",
                "red_team": "Red Team",
                "detail": (
                    "NO_SOURCE_ID_CANDIDATE,TEAM_ROLE_SET_INCOMPLETE"
                ),
            }
        ],
        REJECTED_GAME_COLUMNS,
    )
    analysis = OracleSeriesAnalysis(
        status=SERIES_BLOCKED,
        series_records=pd.DataFrame(),
        diagnostics=pd.DataFrame(
            [
                {
                    "source_key": "G1",
                    "status": SERIES_BLOCKED,
                    "reason": "EXPLICIT_BEST_OF_MISSING",
                    "detail": "",
                },
                {
                    "source_key": "G2",
                    "status": SERIES_BLOCKED,
                    "reason": "SOURCE_SERIES_KEY_MISSING",
                    "detail": "",
                },
            ]
        ),
        game_assignments=pd.DataFrame(
            [
                {
                    "gameid": "G1",
                    "series_id": pd.NA,
                    "series_status": SERIES_BLOCKED,
                    "series_reason": "EXPLICIT_BEST_OF_MISSING",
                }
            ]
        ),
    )
    return OracleDryRunResult(
        records=records,
        unresolved_records=unresolved,
        rejected_records=rejected,
        skipped_records=_frame([], ISSUE_COLUMNS),
        unmapped_champions=_frame([], UNMAPPED_CHAMPION_COLUMNS),
        series_analysis=analysis,
        actions=_frame([], ACTION_COLUMNS),
        source_counts={
            "source_rows": 24,
            "player_rows": 20,
            "team_rows": 4,
            "distinct_games": 2,
        },
        rejected_games=rejected_games,
    )


def _make_transform_summary(result: OracleDryRunResult) -> dict[str, object]:
    records = result.records
    counts = {
        "game_patch": len(records.game_patches),
        "tournament": len(records.tournaments),
        "tournament_stage": len(records.tournament_stages),
        "series": len(records.series),
        "team": len(records.teams),
        "player": len(records.players),
        "game": len(records.games),
        "game_team": len(records.game_teams),
        "game_player": len(records.game_players),
    }
    actions = {
        table: {
            "inserted_expected": int(counts[table]),
            "updated_expected": 0,
            "skipped": 0,
        }
        for table in TARGET_TABLE_ORDER
    }
    unresolved_reason_counts = {
        str(key): int(value)
        for key, value in result.unresolved_records["reason"]
        .value_counts()
        .sort_index()
        .items()
    }
    rejected_reason_counts = {
        str(key): int(value)
        for key, value in result.rejected_records["reason"]
        .value_counts()
        .sort_index()
        .items()
    }
    return {
        "mode": "DRY_RUN",
        **result.source_counts,
        "transformed_records": counts,
        "actions": actions,
        "skipped": 0,
        "unresolved": len(result.unresolved_records),
        "unresolved_reason_counts": unresolved_reason_counts,
        "rejected": len(result.rejected_records),
        "rejected_reason_counts": rejected_reason_counts,
        "primary_rejected_games": len(result.rejected_games),
        "primary_rejection_reason_counts": {
            "GAME_LINEUP_INCOMPLETE": 1
        },
        "game_metadata_detail_counts": {},
        "unmapped_champions": 0,
        "series_status": SERIES_BLOCKED,
        "series_game_status_counts": {SERIES_BLOCKED: 1},
        "series_diagnostic_reason_counts": {
            "EXPLICIT_BEST_OF_MISSING": 1,
            "SOURCE_SERIES_KEY_MISSING": 1,
        },
        "game_transform_status": "PARTIAL",
        "cutoff_status": "UNAVAILABLE",
        "transaction_status": "READ_ONLY_ROLLED_BACK",
        "postgresql_changed": False,
    }


def _build_summary(
    source_path: Path | None = None,
) -> tuple[OracleDryRunResult, dict[str, object]]:
    result = _make_result()
    report_source = source_path or Path("oracle-fixture.csv")
    summary = build_etl_report_summary(
        result=result,
        transform_summary=_make_transform_summary(result),
        source_path=report_source,
        source_sha256_before="fixture-sha256",
        source_sha256_after="fixture-sha256",
        target_snapshot_rows={"game": 0, "champion": 173},
        run_timestamp_utc=datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
        reconciliation_valid=True,
    )
    return result, summary


def test_summary_has_trace_status_and_exact_grain_reconciliation() -> None:
    result = _make_result()
    transform_summary = _make_transform_summary(result)
    summary_before = copy.deepcopy(transform_summary)
    unresolved_before = result.unresolved_records.copy(deep=True)

    summary = build_etl_report_summary(
        result=result,
        transform_summary=transform_summary,
        source_path=Path("oracle-fixture.csv"),
        source_sha256_before="fixture-sha256",
        source_sha256_after="fixture-sha256",
        target_snapshot_rows={"game": 0, "champion": 173},
        run_timestamp_utc=datetime(2026, 8, 31, 12, tzinfo=timezone.utc),
        reconciliation_valid=True,
    )

    assert summary["source_filename"] == "oracle-fixture.csv"
    assert summary["source_sha256"] == "fixture-sha256"
    assert summary["source_sha256_unchanged"] is True
    assert summary["run_timestamp_utc"] == "2026-08-31T12:00:00+00:00"
    assert summary["target_snapshot_rows"] == {"champion": 173, "game": 0}
    assert summary["dry_run_status"] == "PARTIAL"
    assert summary["transform_ready"] is False
    assert summary["unresolved_identity_count"] == 3
    assert summary["series_unresolved_count"] == 2
    assert summary["series_diagnostic_count"] == 2
    assert summary["series_diagnostic_status_counts"] == {"BLOCKED": 2}

    reconciliation = summary["grain_reconciliation"]
    assert reconciliation["source_rows"] == {
        "source": 24,
        "player_rows": 20,
        "team_rows": 4,
        "accounted": 24,
        "difference": 0,
    }
    assert reconciliation["games"] == {
        "source": 2,
        "transformed": 1,
        "primary_rejected": 1,
        "primary_reason_counts": {"GAME_LINEUP_INCOMPLETE": 1},
        "duplicate_transformed_rows": 0,
        "duplicate_primary_rows": 0,
        "accepted_rejected_overlap": 0,
        "accounted": 2,
        "difference": 0,
    }
    assert reconciliation["game_teams"]["difference"] == 0
    assert reconciliation["game_players"]["difference"] == 0
    assert reconciliation["target_cardinality_valid"] is True
    assert summary["grain_reconciliation_valid"] is True
    assert transform_summary == summary_before
    pd.testing.assert_frame_equal(result.unresolved_records, unresolved_before)


def test_writer_creates_exact_outputs_and_identity_only_csv(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "oracle-fixture.csv"
    raw_path.write_text("raw-source\n", encoding="utf-8")
    result, summary = _build_summary(raw_path)
    report_dir = tmp_path / "reports"
    summary_before = copy.deepcopy(summary)
    unresolved_before = result.unresolved_records.copy(deep=True)
    rejected_before = result.rejected_records.copy(deep=True)
    games_before = result.records.games.copy(deep=True)

    outputs = write_etl_report_outputs(
        summary=summary,
        result=result,
        report_dir=report_dir,
        source_path=raw_path,
    )

    expected_names = {
        ETL_SUMMARY_FILENAME,
        UNRESOLVED_IDENTITIES_FILENAME,
        UNMAPPED_CHAMPIONS_FILENAME,
        REJECTED_RECORDS_FILENAME,
        REJECTED_GAMES_FILENAME,
        ETL_REPORT_FILENAME,
    }
    assert {Path(path).name for path in outputs.values()} == expected_names
    assert raw_path.read_text(encoding="utf-8") == "raw-source\n"

    written_summary = json.loads(
        (report_dir / ETL_SUMMARY_FILENAME).read_text(encoding="utf-8")
    )
    assert written_summary["dry_run_status"] == "PARTIAL"
    assert written_summary["unresolved"] == 5
    identities = pd.read_csv(report_dir / UNRESOLVED_IDENTITIES_FILENAME)
    assert len(identities) == 3
    assert set(identities["entity"]) == {"team_participation", "player_row"}
    assert tuple(identities.columns) == ISSUE_COLUMNS
    unmapped = pd.read_csv(report_dir / UNMAPPED_CHAMPIONS_FILENAME)
    assert unmapped.empty
    assert tuple(unmapped.columns) == UNMAPPED_CHAMPION_COLUMNS
    rejected = pd.read_csv(report_dir / REJECTED_RECORDS_FILENAME)
    assert len(rejected) == len(result.rejected_records)
    assert tuple(rejected.columns) == ISSUE_COLUMNS
    traced = rejected.loc[
        rejected["reason"].eq("NO_SOURCE_ID_CANDIDATE")
    ].iloc[0]
    assert traced["entity"] == "game_player"
    assert traced["source_key"] == "G2|Blue|top|10"
    assert traced["detail"] == (
        "Lineup preflight failed for this source row."
    )
    rejected_games = pd.read_csv(report_dir / REJECTED_GAMES_FILENAME)
    assert len(rejected_games) == len(result.rejected_games)
    assert tuple(rejected_games.columns) == REJECTED_GAME_COLUMNS
    assert rejected_games["gameid"].is_unique
    assert rejected_games.iloc[0]["primary_reason"] == (
        "GAME_LINEUP_INCOMPLETE"
    )
    markdown = (report_dir / ETL_REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Dry-run status | PARTIAL" in markdown
    assert "Cutoff status | UNAVAILABLE" in markdown
    assert "record-level evidence" in markdown
    assert summary == summary_before
    pd.testing.assert_frame_equal(result.unresolved_records, unresolved_before)
    pd.testing.assert_frame_equal(result.rejected_records, rejected_before)
    pd.testing.assert_frame_equal(result.records.games, games_before)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("mode", "APPLY"),
        ("source_sha256_unchanged", False),
        ("source_sha256_after", "different-sha256"),
        ("source_sha256", "different-sha256"),
        ("transaction_status", "COMMITTED"),
        ("postgresql_changed", True),
        ("reconciliation_valid", False),
        ("grain_reconciliation_valid", False),
    ),
)
def test_writer_rejects_unsafe_state(
    tmp_path: Path,
    field: str,
    unsafe_value: object,
) -> None:
    result, summary = _build_summary()
    summary[field] = unsafe_value

    with pytest.raises(ValueError, match="Unsafe ETL report state"):
        write_etl_report_outputs(
            summary=summary,
            result=result,
            report_dir=tmp_path / "reports",
            source_path=tmp_path / "oracle-fixture.csv",
        )


def test_writer_refuses_to_overwrite_raw_source(tmp_path: Path) -> None:
    raw_path = tmp_path / ETL_SUMMARY_FILENAME
    result, summary = _build_summary(raw_path)

    with pytest.raises(ValueError, match="raw source"):
        write_etl_report_outputs(
            summary=summary,
            result=result,
            report_dir=tmp_path,
            source_path=raw_path,
        )


def test_reconciliation_rejects_duplicate_primary_rows() -> None:
    base = _make_result()
    duplicate_primary = pd.concat(
        [base.rejected_games, base.rejected_games],
        ignore_index=True,
    )
    result = replace(base, rejected_games=duplicate_primary)

    reconciliation = build_grain_reconciliation(result)

    assert reconciliation["games"]["duplicate_primary_rows"] == 1
    assert reconciliation["valid"] is False


def test_reconciliation_rejects_accepted_rejected_overlap() -> None:
    base = _make_result()
    overlapping = base.rejected_games.copy(deep=True)
    overlapping.loc[:, "gameid"] = "G1"
    result = replace(base, rejected_games=overlapping)

    reconciliation = build_grain_reconciliation(result)

    assert reconciliation["games"]["accepted_rejected_overlap"] == 1
    assert reconciliation["valid"] is False


def test_reconciliation_rejects_duplicate_transformed_game_rows() -> None:
    base = _make_result()
    duplicate_games = pd.concat(
        [base.records.games, base.records.games],
        ignore_index=True,
    )
    records = replace(base.records, games=duplicate_games)

    reconciliation = build_grain_reconciliation(
        replace(base, records=records)
    )

    assert reconciliation["games"]["duplicate_transformed_rows"] == 1
    assert reconciliation["valid"] is False


@pytest.mark.parametrize("child", ("game_teams", "game_players"))
def test_reconciliation_rejects_incomplete_target_cardinality(
    child: str,
) -> None:
    base = _make_result()
    frame = getattr(base.records, child).iloc[:-1].copy()
    records = replace(base.records, **{child: frame})

    reconciliation = build_grain_reconciliation(
        replace(base, records=records)
    )

    assert reconciliation["target_cardinality_valid"] is False
    assert reconciliation["valid"] is False


@pytest.mark.parametrize("child", ("game_teams", "game_players"))
def test_reconciliation_rejects_orphan_child_with_unchanged_total(
    child: str,
) -> None:
    base = _make_result()
    frame = getattr(base.records, child).copy(deep=True)
    frame.loc[frame.index[-1], "game_id"] = "GHOST"
    records = replace(base.records, **{child: frame})

    reconciliation = build_grain_reconciliation(
        replace(base, records=records)
    )

    assert reconciliation[child]["orphan_rows"] == 1
    assert reconciliation[child]["invalid_game_count"] == 1
    assert reconciliation["target_cardinality_valid"] is False
    assert reconciliation["valid"] is False


def test_writer_rejects_stale_result_reconciliation(tmp_path: Path) -> None:
    source_path = tmp_path / "oracle-fixture.csv"
    result, summary = _build_summary(source_path)
    duplicate_primary = pd.concat(
        [result.rejected_games, result.rejected_games],
        ignore_index=True,
    )
    stale_result = replace(result, rejected_games=duplicate_primary)
    report_dir = tmp_path / "reports"

    with pytest.raises(ValueError, match="result reconciliation"):
        write_etl_report_outputs(
            summary=summary,
            result=stale_result,
            report_dir=report_dir,
            source_path=source_path,
        )

    assert not report_dir.exists()


def test_rejected_game_sort_accepts_mixed_source_year_trace_types() -> None:
    rows = [
        {
            "gameid": "G2",
            "primary_reason": "GAME_METADATA_INVALID",
            "secondary_reasons": "[]",
            "league": "League",
            "source_year": 2025,
            "calendar_date": "2025-01-01",
            "patch": "15.1",
            "split": "Spring",
            "datacompleteness": "complete",
            "blue_team": "Blue",
            "red_team": "Red",
            "detail": "CONFLICTING_YEAR",
        },
        {
            "gameid": "G1",
            "primary_reason": "GAME_METADATA_INVALID",
            "secondary_reasons": "[]",
            "league": "League",
            "source_year": "[2024, 2025]",
            "calendar_date": "2025-01-02",
            "patch": "15.1",
            "split": "Spring",
            "datacompleteness": "complete",
            "blue_team": "Blue",
            "red_team": "Red",
            "detail": "CONFLICTING_YEAR",
        },
    ]

    sorted_frame = _sorted_frame(
        _frame(rows, REJECTED_GAME_COLUMNS),
        columns=REJECTED_GAME_COLUMNS,
        sort_by=("gameid",),
    )

    assert sorted_frame["gameid"].tolist() == ["G1", "G2"]
    assert sorted_frame["source_year"].tolist() == ["[2024, 2025]", 2025]


@pytest.mark.parametrize(
    ("game_status", "series_status", "issues", "expected"),
    (
        ("BLOCKED", SERIES_BLOCKED, 0, (False, "BLOCKED")),
        ("PARTIAL", SERIES_BLOCKED, 0, (False, "PARTIAL")),
        ("READY", SERIES_BLOCKED, 0, (True, "READY_WITH_ISSUES")),
        ("READY", "READY", 0, (True, "READY")),
        ("READY", "READY", 1, (True, "READY_WITH_ISSUES")),
    ),
)
def test_derive_dry_run_state(
    game_status: str,
    series_status: str,
    issues: int,
    expected: tuple[bool, str],
) -> None:
    assert derive_dry_run_state(
        {
            "game_transform_status": game_status,
            "series_status": series_status,
            "unresolved": issues,
            "rejected": 0,
            "unmapped_champions": 0,
        }
    ) == expected
