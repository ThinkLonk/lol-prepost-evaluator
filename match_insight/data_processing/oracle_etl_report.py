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
ETL_REPORT_FILENAME: Final[str] = "oracle_2025_etl_report.md"

IDENTITY_ISSUE_ENTITIES: Final[frozenset[str]] = frozenset(
    {"team_participation", "player_row"}
)

_PLAYER_ROW_REJECTION_REASONS: Final[tuple[str, ...]] = (
    "GAME_PLAYER_DEPENDENCY_INVALID",
    "DUPLICATE_GAME_TEAM_ROLE",
    "GAME_PLAYER_BLOCKED_BY_INCOMPLETE_ROLE_SET",
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


def _accepted_player_identity_keys(result: OracleDryRunResult) -> set[str]:
    accepted_game_ids = set(result.records.games["game_id"].astype(str))
    if not accepted_game_ids or result.unresolved_records.empty:
        return set()

    unresolved_players = result.unresolved_records.loc[
        result.unresolved_records["entity"].eq("player_row"),
        "source_key",
    ]
    accepted_keys: set[str] = set()
    for source_key in unresolved_players.tolist():
        key = str(source_key)
        parts = key.rsplit("|", 3)
        if len(parts) == 4 and parts[0] in accepted_game_ids:
            accepted_keys.add(key)
    return accepted_keys


def _issue_count_excluding_keys(
    dataframe: pd.DataFrame,
    *,
    entity: str,
    reason: str,
    excluded_source_keys: set[str],
) -> int:
    if dataframe.empty:
        return 0
    required = {"entity", "source_key", "reason"}
    if not required <= set(dataframe.columns):
        missing = ", ".join(sorted(required - set(dataframe.columns)))
        raise ValueError(f"Issue dataframe is missing columns: {missing}")
    mask = dataframe["entity"].eq(entity) & dataframe["reason"].eq(reason)
    if excluded_source_keys:
        mask &= ~dataframe["source_key"].astype(str).isin(excluded_source_keys)
    return int(mask.sum())


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
    """Partition source grains without treating group diagnostics as rows."""

    rejected = result.rejected_records
    source_rows = int(result.source_counts["source_rows"])
    source_games = int(result.source_counts["distinct_games"])
    source_team_rows = int(result.source_counts["team_rows"])
    source_player_rows = int(result.source_counts["player_rows"])
    classified_rows = source_team_rows + source_player_rows

    transformed_games = int(len(result.records.games))
    metadata_invalid = _issue_count(
        rejected,
        entity="game",
        reason="GAME_METADATA_INVALID",
    )
    dependency_invalid = _issue_count(
        rejected,
        entity="game",
        reason="GAME_DEPENDENCY_INVALID",
    )
    accounted_games = transformed_games + metadata_invalid + dependency_invalid

    transformed_team_rows = int(len(result.records.game_teams))
    team_parent_blocked = _issue_count(
        rejected,
        entity="game_team",
        reason="GAME_TEAM_BLOCKED_BY_GAME",
    )
    accounted_team_rows = transformed_team_rows + team_parent_blocked

    transformed_player_rows = int(len(result.records.game_players))
    player_parent_blocked = _issue_count(
        rejected,
        entity="game_player",
        reason="GAME_PLAYER_BLOCKED_BY_GAME",
    )
    accepted_unresolved_keys = _accepted_player_identity_keys(result)
    player_row_rejections = {
        reason: _issue_count_excluding_keys(
            rejected,
            entity="game_player",
            reason=reason,
            excluded_source_keys=accepted_unresolved_keys,
        )
        for reason in _PLAYER_ROW_REJECTION_REASONS
    }
    raw_player_row_rejections = sum(
        _issue_count(
            rejected,
            entity="game_player",
            reason=reason,
        )
        for reason in _PLAYER_ROW_REJECTION_REASONS
    )
    unresolved_on_accepted_games = len(accepted_unresolved_keys)
    overlapping_unresolved_diagnostics = (
        raw_player_row_rejections - sum(player_row_rejections.values())
    )
    accounted_player_rows = (
        transformed_player_rows
        + player_parent_blocked
        + sum(player_row_rejections.values())
        + unresolved_on_accepted_games
    )
    incomplete_group_diagnostics = _issue_count(
        rejected,
        entity="game_player",
        reason="GAME_TEAM_ROLES_INCOMPLETE",
    )

    games = {
        "source": source_games,
        "transformed": transformed_games,
        "metadata_invalid": metadata_invalid,
        "dependency_invalid": dependency_invalid,
        "accounted": accounted_games,
        "difference": source_games - accounted_games,
    }
    game_teams = {
        "source": source_team_rows,
        "transformed": transformed_team_rows,
        "parent_game_blocked": team_parent_blocked,
        "accounted": accounted_team_rows,
        "difference": source_team_rows - accounted_team_rows,
    }
    game_players = {
        "source": source_player_rows,
        "transformed": transformed_player_rows,
        "parent_game_blocked": player_parent_blocked,
        "row_rejections": player_row_rejections,
        "unresolved_identity_on_transformed_game": (
            unresolved_on_accepted_games
        ),
        "incomplete_role_group_diagnostics": incomplete_group_diagnostics,
        "overlapping_unresolved_row_diagnostics": (
            overlapping_unresolved_diagnostics
        ),
        "accounted": accounted_player_rows,
        "difference": source_player_rows - accounted_player_rows,
    }
    source = {
        "source": source_rows,
        "player_rows": source_player_rows,
        "team_rows": source_team_rows,
        "accounted": classified_rows,
        "difference": source_rows - classified_rows,
    }
    valid = all(
        int(section["difference"]) == 0
        for section in (source, games, game_teams, game_players)
    )
    return {
        "source_rows": source,
        "games": games,
        "game_teams": game_teams,
        "game_players": game_players,
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
                    "Grain reconciliation valid",
                    summary.get("grain_reconciliation_valid"),
                ),
            ]
        ),
        "",
        "`incomplete_role_group_diagnostics` là diagnostic cấp game-side "
        "và không được cộng như một source player row độc lập.",
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
            ]
        ),
        "",
        "Rejected là diagnostic ở nhiều grain và có thể gồm cascade hoặc "
        "group diagnostic; không được diễn giải như số raw rows "
        "độc lập.",
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
        "`cutoff_status=UNAVAILABLE`. `date` không được diễn giải là thời "
        "điểm kết thúc và báo cáo này không cho phép chuyển sang "
        "feature hoặc "
        "huấn luyện mô hình theo thời gian.",
        "",
    ]
    return "\n".join(lines)


def _sorted_frame(
    dataframe: pd.DataFrame,
    *,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            "Report dataframe is missing columns: " + ", ".join(missing)
        )
    result = dataframe.loc[:, list(columns)].copy(deep=True)
    if not result.empty:
        result = result.sort_values(
            by=list(columns),
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
    """Write the five approved artifacts after safety validation."""

    _validate_report_summary(summary)
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
    paths["etl_report_path"].write_text(
        build_etl_markdown_report(summary),
        encoding="utf-8",
    )
    return {key: str(path.resolve()) for key, path in paths.items()}
