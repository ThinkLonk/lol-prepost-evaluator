"""Build and write traceable Oracle ETL dry-run reports."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pandas as pd

from match_insight.data_processing.oracle_transform import (
    GAME_TRANSFORM_BLOCKED,
    GAME_TRANSFORM_PARTIAL,
    GAME_TRANSFORM_READY,
    ISSUE_COLUMNS,
    REJECTED_GAME_COLUMNS,
    SERIES_READY,
    UNMAPPED_CHAMPION_COLUMNS,
    OracleDryRunResult,
)

ETL_SUMMARY_FILENAME: Final[str] = "oracle_2025_etl_summary.json"
UNRESOLVED_IDENTITIES_FILENAME: Final[str] = (
    "oracle_2025_unresolved_identities.csv"
)
UNMAPPED_CHAMPIONS_FILENAME: Final[str] = (
    "oracle_2025_unmapped_champions.csv"
)
REJECTED_RECORDS_FILENAME: Final[str] = "oracle_2025_rejected_records.csv"
REJECTED_GAMES_FILENAME: Final[str] = "oracle_2025_rejected_games.csv"
ETL_REPORT_FILENAME: Final[str] = "oracle_2025_etl_report.md"

IDENTITY_ISSUE_ENTITIES: Final[frozenset[str]] = frozenset(
    {"team_participation", "player_row"}
)
_EXPECTED_SIDES: Final[frozenset[str]] = frozenset({"BLUE", "RED"})
_EXPECTED_ROLES: Final[frozenset[str]] = frozenset(
    {"TOP", "JUNGLE", "MID", "BOT", "SUPPORT"}
)


def _mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Report field '{field_name}' must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _value_counts(dataframe: pd.DataFrame, column: str) -> dict[str, int]:
    if dataframe.empty:
        return {}
    if column not in dataframe.columns:
        raise ValueError(f"Report dataframe is missing column: {column}")
    counts = dataframe[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _issue_count(
    dataframe: pd.DataFrame,
    *,
    entity: str,
    reason: str,
) -> int:
    if dataframe.empty:
        return 0
    required = {"entity", "reason"}
    if not required <= set(dataframe.columns):
        missing = ", ".join(sorted(required - set(dataframe.columns)))
        raise ValueError(f"Issue dataframe is missing columns: {missing}")
    mask = dataframe["entity"].eq(entity) & dataframe["reason"].eq(reason)
    return int(mask.sum())


def _child_integrity(
    dataframe: pd.DataFrame,
    *,
    accepted_ids: set[str],
    expected_keys: frozenset[tuple[str, ...]],
    key_columns: tuple[str, ...],
    identity_column: str,
) -> dict[str, object]:
    """Validate exact per-game child shape and parent membership."""
    required = {"game_id", identity_column, *key_columns}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Target child dataframe is missing columns: " + ", ".join(missing)
        )

    parent_ids = dataframe["game_id"].astype("string")
    missing_parent_rows = int(parent_ids.isna().sum())
    parent_text = parent_ids.fillna("").astype(str)
    orphan_mask = ~parent_text.isin(accepted_ids)
    orphan_rows = int(orphan_mask.sum())
    duplicate_keys = int(
        dataframe.duplicated(subset=["game_id", *key_columns]).sum()
    )
    duplicate_identities = int(
        dataframe.duplicated(subset=["game_id", identity_column]).sum()
    )
    invalid_games: list[str] = []

    for game_id in sorted(accepted_ids):
        game_rows = dataframe.loc[parent_text.eq(game_id)]
        observed_keys = frozenset(
            tuple(str(value) for value in row)
            for row in game_rows.loc[:, list(key_columns)].itertuples(
                index=False,
                name=None,
            )
        )
        identities = game_rows[identity_column].astype("string")
        if (
            len(game_rows) != len(expected_keys)
            or observed_keys != expected_keys
            or bool(identities.isna().any())
            or bool(identities.duplicated().any())
        ):
            invalid_games.append(game_id)

    valid = (
        missing_parent_rows == 0
        and orphan_rows == 0
        and duplicate_keys == 0
        and duplicate_identities == 0
        and not invalid_games
    )
    return {
        "orphan_rows": orphan_rows,
        "missing_parent_rows": missing_parent_rows,
        "duplicate_logical_keys": duplicate_keys,
        "duplicate_identities": duplicate_identities,
        "invalid_game_count": len(invalid_games),
        "invalid_game_ids": invalid_games,
        "valid": valid,
    }


def derive_dry_run_state(
    summary: Mapping[str, object],
) -> tuple[bool, str]:
    """Return transform readiness and the user-facing dry-run status."""

    game_status = str(summary.get("game_transform_status", ""))
    if game_status not in {
        GAME_TRANSFORM_READY,
        GAME_TRANSFORM_PARTIAL,
        GAME_TRANSFORM_BLOCKED,
    }:
        raise ValueError(f"Unsupported game transform status: {game_status}")

    transform_ready = game_status == GAME_TRANSFORM_READY
    if game_status == GAME_TRANSFORM_BLOCKED:
        return transform_ready, "BLOCKED"
    if game_status == GAME_TRANSFORM_PARTIAL:
        return transform_ready, "PARTIAL"

    has_issues = any(
        int(summary.get(key, 0)) > 0
        for key in ("unresolved", "rejected", "unmapped_champions")
    )
    series_ready = summary.get("series_status") == SERIES_READY
    if has_issues or not series_ready:
        return transform_ready, "READY_WITH_ISSUES"
    return transform_ready, "READY"


def build_grain_reconciliation(
    result: OracleDryRunResult,
) -> dict[str, object]:
    """Reconcile unique game dispositions and their complete child grains."""

    diagnostics = result.rejected_records
    primary = result.rejected_games
    source_rows = int(result.source_counts["source_rows"])
    source_games = int(result.source_counts["distinct_games"])
    source_team_rows = int(result.source_counts["team_rows"])
    source_player_rows = int(result.source_counts["player_rows"])

    accepted_list = result.records.games["game_id"].astype(str).tolist()
    accepted_ids = set(accepted_list)
    rejected_list = primary["gameid"].astype(str).tolist()
    rejected_ids = set(rejected_list)
    duplicate_accepted = len(accepted_list) - len(accepted_ids)
    duplicate_rejected = len(rejected_list) - len(rejected_ids)
    overlap = len(accepted_ids & rejected_ids)
    accounted_ids = accepted_ids | rejected_ids
    transformed_games = len(accepted_list)

    transformed_team_rows = int(len(result.records.game_teams))
    team_integrity = _child_integrity(
        result.records.game_teams,
        accepted_ids=accepted_ids,
        expected_keys=frozenset((side,) for side in _EXPECTED_SIDES),
        key_columns=("side",),
        identity_column="team_id",
    )
    team_parent_blocked = _issue_count(
        diagnostics,
        entity="game_team",
        reason="GAME_TEAM_BLOCKED_BY_GAME",
    )
    accounted_team_rows = transformed_team_rows + team_parent_blocked

    transformed_player_rows = int(len(result.records.game_players))
    player_integrity = _child_integrity(
        result.records.game_players,
        accepted_ids=accepted_ids,
        expected_keys=frozenset(
            (side, role)
            for side in _EXPECTED_SIDES
            for role in _EXPECTED_ROLES
        ),
        key_columns=("side", "role"),
        identity_column="player_id",
    )
    player_parent_blocked = _issue_count(
        diagnostics,
        entity="game_player",
        reason="GAME_PLAYER_BLOCKED_BY_GAME",
    )
    accounted_player_rows = transformed_player_rows + player_parent_blocked

    games = {
        "source": source_games,
        "transformed": transformed_games,
        "primary_rejected": len(rejected_list),
        "primary_reason_counts": _value_counts(primary, "primary_reason"),
        "duplicate_transformed_rows": duplicate_accepted,
        "duplicate_primary_rows": duplicate_rejected,
        "accepted_rejected_overlap": overlap,
        "accounted": len(accounted_ids),
        "difference": source_games - len(accounted_ids),
    }
    game_teams = {
        "source": source_team_rows,
        "transformed": transformed_team_rows,
        "parent_game_blocked": team_parent_blocked,
        "accounted": accounted_team_rows,
        "difference": source_team_rows - accounted_team_rows,
        **team_integrity,
    }
    game_players = {
        "source": source_player_rows,
        "transformed": transformed_player_rows,
        "parent_game_blocked": player_parent_blocked,
        "accounted": accounted_player_rows,
        "difference": source_player_rows - accounted_player_rows,
        **player_integrity,
    }
    classified_rows = source_team_rows + source_player_rows
    source = {
        "source": source_rows,
        "player_rows": source_player_rows,
        "team_rows": source_team_rows,
        "accounted": classified_rows,
        "difference": source_rows - classified_rows,
    }
    target_cardinality_valid = bool(
        team_integrity["valid"] and player_integrity["valid"]
    )
    valid = (
        all(
            int(section["difference"]) == 0
            for section in (source, games, game_teams, game_players)
        )
        and duplicate_accepted == 0
        and duplicate_rejected == 0
        and overlap == 0
        and target_cardinality_valid
    )
    return {
        "source_rows": source,
        "games": games,
        "game_teams": game_teams,
        "game_players": game_players,
        "target_cardinality_valid": target_cardinality_valid,
        "valid": valid,
    }


def build_etl_report_summary(
    *,
    result: OracleDryRunResult,
    transform_summary: Mapping[str, object],
    source_path: str | Path,
    source_sha256_before: str,
    source_sha256_after: str,
    target_snapshot_rows: Mapping[str, int],
    run_timestamp_utc: datetime,
    reconciliation_valid: bool,
) -> dict[str, object]:
    """Enrich the transform summary with trace and report reconciliation."""

    if run_timestamp_utc.tzinfo is None or run_timestamp_utc.utcoffset() is None:
        raise ValueError("run_timestamp_utc must be timezone-aware.")

    summary = copy.deepcopy(dict(transform_summary))
    source = Path(source_path).resolve()
    sha_unchanged = source_sha256_before == source_sha256_after
    transform_ready, dry_run_status = derive_dry_run_state(summary)
    grain_reconciliation = build_grain_reconciliation(result)

    unresolved = result.unresolved_records
    identity_mask = unresolved["entity"].isin(IDENTITY_ISSUE_ENTITIES)
    unresolved_entity_counts = _value_counts(unresolved, "entity")
    rejected_entity_counts = _value_counts(result.rejected_records, "entity")
    series_diagnostics = result.series_analysis.diagnostics
    action_totals = {
        "inserted_expected": 0,
        "updated_expected": 0,
        "skipped": 0,
    }
    for table_counts in _mapping(
        summary.get("actions"),
        field_name="actions",
    ).values():
        counts = _mapping(table_counts, field_name="actions table")
        for action_name in action_totals:
            action_totals[action_name] += int(counts.get(action_name, 0))

    summary.update(
        {
            "source_filename": source.name,
            "source_path": str(source),
            "source_sha256": source_sha256_before,
            "source_sha256_before": source_sha256_before,
            "source_sha256_after": source_sha256_after,
            "source_sha256_unchanged": sha_unchanged,
            "run_timestamp_utc": run_timestamp_utc.astimezone(
                timezone.utc
            ).isoformat(),
            "target_snapshot_rows": {
                str(key): int(value)
                for key, value in sorted(target_snapshot_rows.items())
            },
            "action_totals": action_totals,
            "unresolved_entity_counts": unresolved_entity_counts,
            "unresolved_identity_count": int(identity_mask.sum()),
            "series_unresolved_count": int(
                unresolved["entity"].eq("series").sum()
            ),
            "series_diagnostic_count": int(len(series_diagnostics)),
            "series_diagnostic_status_counts": _value_counts(
                series_diagnostics,
                "status",
            ),
            "rejected_entity_counts": rejected_entity_counts,
            "primary_rejected_games": int(len(result.rejected_games)),
            "primary_rejection_reason_counts": _value_counts(
                result.rejected_games,
                "primary_reason",
            ),
            "unmapped_champion_reason_counts": _value_counts(
                result.unmapped_champions,
                "reason",
            ),
            "transform_ready": transform_ready,
            "dry_run_status": dry_run_status,
            "reconciliation_valid": bool(reconciliation_valid),
            "grain_reconciliation": grain_reconciliation,
            "grain_reconciliation_valid": bool(
                grain_reconciliation["valid"]
            ),
        }
    )
    return summary


def _markdown_table(rows: list[tuple[str, object]]) -> list[str]:
    lines = ["| Chỉ số | Giá trị |", "| --- | --- |"]
    for label, value in rows:
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        rendered = rendered.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {label} | {rendered} |")
    return lines


def build_etl_markdown_report(summary: Mapping[str, object]) -> str:
    """Render a concise Vietnamese report without hiding partial coverage."""

    transformed = _mapping(
        summary.get("transformed_records"),
        field_name="transformed_records",
    )
    reconciliation = _mapping(
        summary.get("grain_reconciliation"),
        field_name="grain_reconciliation",
    )
    source_rows = _mapping(
        reconciliation.get("source_rows"),
        field_name="source_rows",
    )
    games = _mapping(reconciliation.get("games"), field_name="games")
    game_teams = _mapping(
        reconciliation.get("game_teams"),
        field_name="game_teams",
    )
    game_players = _mapping(
        reconciliation.get("game_players"),
        field_name="game_players",
    )

    lines: list[str] = [
        "# Oracle 2025 ETL Dry-run Report",
        "",
        "> Báo cáo này mô tả dry-run và không xác nhận dữ liệu "
        "đã sẵn sàng để apply. Không có bản ghi PostgreSQL nào "
        "được insert hoặc update.",
        "",
        "## 1. Nguồn và lần chạy",
        "",
        *_markdown_table(
            [
                ("Filename", summary.get("source_filename")),
                ("Source path", summary.get("source_path")),
                ("SHA-256", summary.get("source_sha256")),
                ("SHA không đổi", summary.get("source_sha256_unchanged")),
                ("Run timestamp UTC", summary.get("run_timestamp_utc")),
                ("Mode", summary.get("mode")),
            ]
        ),
        "",
        "## 2. Trạng thái an toàn",
        "",
        *_markdown_table(
            [
                ("Dry-run status", summary.get("dry_run_status")),
                ("Transform ready", summary.get("transform_ready")),
                ("Series status", summary.get("series_status")),
                ("Cutoff status", summary.get("cutoff_status")),
                ("Transaction", summary.get("transaction_status")),
                ("PostgreSQL changed", summary.get("postgresql_changed")),
                ("Reconciliation valid", summary.get("reconciliation_valid")),
            ]
        ),
        "",
        "## 3. Source grain và target records",
        "",
        *_markdown_table(
            [
                ("Source rows", summary.get("source_rows")),
                ("Player rows", summary.get("player_rows")),
                ("Team rows", summary.get("team_rows")),
                ("Distinct games", summary.get("distinct_games")),
                ("Transformed records", transformed),
                ("Action totals", summary.get("action_totals")),
            ]
        ),
        "",
        "## 4. Reconciliation theo grain",
        "",
        *_markdown_table(
            [
                ("Source rows", source_rows),
                ("Game", games),
                ("Game-team", game_teams),
                ("Game-player", game_players),
                (
                    "Target cardinality valid",
                    reconciliation.get("target_cardinality_valid"),
                ),
                (
                    "Grain reconciliation valid",
                    summary.get("grain_reconciliation_valid"),
                ),
            ]
        ),
        "",
        "Mỗi source game phải có đúng một disposition: accepted hoặc primary "
        "rejected. Game accepted phải emit đúng 2 game-team và 10 game-player.",
        "",
        "## 5. Unresolved và rejected",
        "",
        *_markdown_table(
            [
                ("Unresolved tổng", summary.get("unresolved")),
                (
                    "Unresolved identity",
                    summary.get("unresolved_identity_count"),
                ),
                (
                    "Series unresolved",
                    summary.get("series_unresolved_count"),
                ),
                (
                    "Series diagnostics tổng",
                    summary.get("series_diagnostic_count"),
                ),
                (
                    "Series diagnostic status",
                    summary.get("series_diagnostic_status_counts"),
                ),
                (
                    "Series diagnostic reasons",
                    summary.get("series_diagnostic_reason_counts"),
                ),
                (
                    "Unresolved theo entity",
                    summary.get("unresolved_entity_counts"),
                ),
                ("Rejected diagnostics", summary.get("rejected")),
                (
                    "Rejected theo reason",
                    summary.get("rejected_reason_counts"),
                ),
                (
                    "Game metadata detail",
                    summary.get("game_metadata_detail_counts"),
                ),
                (
                    "Primary rejected games",
                    summary.get("primary_rejected_games"),
                ),
                (
                    "Primary rejection reasons",
                    summary.get("primary_rejection_reason_counts"),
                ),
            ]
        ),
        "",
        "Rejected là diagnostic ở nhiều grain và có thể gồm cascade hoặc "
        "group diagnostic; không được diễn giải như số raw rows "
        "độc lập. Primary rejected-game chỉ là aggregate bổ sung, không thay "
        "thế record-level evidence.",
        "",
        "## 6. Champion và giới hạn",
        "",
        *_markdown_table(
            [
                ("Unmapped champions", summary.get("unmapped_champions")),
                (
                    "Unmapped reasons",
                    summary.get("unmapped_champion_reason_counts"),
                ),
            ]
        ),
        "",
        "Nguồn không có explicit game-end timestamp nên "
        "`cutoff_status=UNAVAILABLE`. `date` không được diễn giải là scheduled, "
        "started, ended hoặc cutoff; báo cáo này không cho phép chuyển sang "
        "feature hoặc "
        "huấn luyện mô hình theo thời gian.",
        "",
    ]
    return "\n".join(lines)


def _sorted_frame(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    sort_by: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Report dataframe is missing columns: " + ", ".join(missing)
        )
    result = dataframe.loc[:, list(columns)].copy(deep=True)
    if not result.empty:
        sort_columns = columns if sort_by is None else sort_by
        unknown_sort_columns = sorted(set(sort_columns) - set(columns))
        if unknown_sort_columns:
            raise ValueError(
                "Report sort columns are not in its schema: "
                + ", ".join(unknown_sort_columns)
            )
        result = result.sort_values(
            by=list(sort_columns),
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
    return result


def _validate_report_summary(summary: Mapping[str, object]) -> None:
    sha_before = summary.get("source_sha256_before")
    sha_after = summary.get("source_sha256_after")
    source_sha = summary.get("source_sha256")
    checks = (
        (summary.get("mode") == "DRY_RUN", "mode must be DRY_RUN"),
        (
            summary.get("source_sha256_unchanged") is True
            and isinstance(sha_before, str)
            and sha_before == sha_after
            and source_sha == sha_before,
            "source SHA-256 must be unchanged",
        ),
        (
            summary.get("transaction_status") == "READ_ONLY_ROLLED_BACK",
            "transaction must be read-only and rolled back",
        ),
        (
            summary.get("postgresql_changed") is False,
            "postgresql_changed must be false",
        ),
        (
            summary.get("reconciliation_valid") is True,
            "structural reconciliation must be valid",
        ),
        (
            summary.get("grain_reconciliation_valid") is True,
            "grain reconciliation must be valid",
        ),
    )
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise ValueError("Unsafe ETL report state: " + "; ".join(failures))


def write_etl_report_outputs(
    *,
    summary: Mapping[str, object],
    result: OracleDryRunResult,
    report_dir: str | Path,
    source_path: str | Path,
) -> dict[str, str]:
    """Write the six approved artifacts after safety validation."""

    _validate_report_summary(summary)
    current_reconciliation = build_grain_reconciliation(result)
    if (
        current_reconciliation.get("valid") is not True
        or summary.get("grain_reconciliation") != current_reconciliation
    ):
        raise ValueError(
            "Unsafe ETL report state: result reconciliation is invalid or stale"
        )
    resolved_dir = Path(report_dir).resolve()
    resolved_source = Path(source_path).resolve()
    summary_source = summary.get("source_path")
    if not isinstance(summary_source, str):
        raise ValueError("ETL report summary is missing source_path.")
    if Path(summary_source).resolve() != resolved_source:
        raise ValueError("ETL report source path does not match its summary.")
    paths = {
        "etl_summary_path": resolved_dir / ETL_SUMMARY_FILENAME,
        "unresolved_identities_path": (
            resolved_dir / UNRESOLVED_IDENTITIES_FILENAME
        ),
        "unmapped_champions_path": (
            resolved_dir / UNMAPPED_CHAMPIONS_FILENAME
        ),
        "rejected_records_path": resolved_dir / REJECTED_RECORDS_FILENAME,
        "rejected_games_path": resolved_dir / REJECTED_GAMES_FILENAME,
        "etl_report_path": resolved_dir / ETL_REPORT_FILENAME,
    }
    resolved_paths = [path.resolve() for path in paths.values()]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("ETL report output paths must be distinct.")
    if resolved_source in resolved_paths:
        raise ValueError("ETL report output must not overwrite the raw source.")

    unresolved_identities = _sorted_frame(
        result.unresolved_records.loc[
            result.unresolved_records["entity"].isin(IDENTITY_ISSUE_ENTITIES)
        ],
        columns=ISSUE_COLUMNS,
    )
    unmapped_champions = _sorted_frame(
        result.unmapped_champions,
        columns=UNMAPPED_CHAMPION_COLUMNS,
    )
    rejected_records = _sorted_frame(
        result.rejected_records,
        columns=ISSUE_COLUMNS,
    )
    rejected_games = _sorted_frame(
        result.rejected_games,
        columns=REJECTED_GAME_COLUMNS,
        sort_by=("gameid",),
    )

    resolved_dir.mkdir(parents=True, exist_ok=True)
    paths["etl_summary_path"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    unresolved_identities.to_csv(
        paths["unresolved_identities_path"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    unmapped_champions.to_csv(
        paths["unmapped_champions_path"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    rejected_records.to_csv(
        paths["rejected_records_path"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    rejected_games.to_csv(
        paths["rejected_games_path"],
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    paths["etl_report_path"].write_text(
        build_etl_markdown_report(summary),
        encoding="utf-8",
    )
    return {key: str(path.resolve()) for key, path in paths.items()}
