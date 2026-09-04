"""Inspect the Oracle dry-run and optionally write approved Step 6E reports."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Connection
from sqlalchemy.sql import Select

from match_insight.data_processing.oracle_audit import (
    calculate_sha256,
    resolve_source_path,
)
from match_insight.data_processing.oracle_etl import (
    OracleCoreData,
    prepare_oracle_core_data,
)
from match_insight.data_processing.oracle_etl_report import (
    build_etl_report_summary,
    derive_dry_run_state,
    write_etl_report_outputs,
)
from match_insight.data_processing.oracle_identity import (
    ChampionEntityReference,
    PlayerEntityReference,
    TeamEntityReference,
    resolve_oracle_identities,
)
from match_insight.data_processing.oracle_transform import (
    GAME_TRANSFORM_BLOCKED,
    GAME_TRANSFORM_PARTIAL,
    GAME_TRANSFORM_READY,
    ROLE_MAP,
    OracleDryRunResult,
    OracleTargetSnapshot,
    analyze_series_evidence,
    build_dry_run_summary,
    transform_oracle_targets,
)
from match_insight.database.engine import engine
from match_insight.database.models import (
    Champion,
    Game,
    GamePatch,
    GamePlayer,
    GameTeam,
    Player,
    Team,
    Tournament,
    TournamentStage,
)
from match_insight.database.models import Series as MatchSeries

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oracle"
    / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
)


@dataclass(frozen=True)
class OracleDatabaseSnapshot:
    """Materialized target and champion rows from one database snapshot."""

    targets: OracleTargetSnapshot
    champions: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Accept the source and an explicit opt-in report directory."""
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run Oracle target records against a read-only PostgreSQL "
            "snapshot and print reconciliation to stdout."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=(
            "Oracle CSV path. Defaults to the audited 2025 raw file under "
            "data/raw/oracle."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Write the six approved Step 6E reports under this directory. "
            "If omitted, the command remains console-only."
        ),
    )
    return parser.parse_args()


def _json_value(value: object) -> str:
    """Encode deterministic one-line JSON for reconciliation output."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _materialize(
    connection: Connection,
    statement: Select[tuple[object, ...]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Execute a SELECT and fully materialize its rows."""
    rows = connection.execute(statement).mappings().all()
    return pd.DataFrame(
        [dict(row) for row in rows],
        columns=list(columns),
    )


def load_database_snapshot(
    connection: Connection | None = None,
) -> OracleDatabaseSnapshot:
    """Read targets, owning a read-only transaction only when needed."""
    owns_connection = connection is None
    context = (
        engine.connect().execution_options(isolation_level="REPEATABLE READ")
        if owns_connection
        else nullcontext(connection)
    )
    with context as active_connection:
        connection = active_connection
        transaction = connection.begin() if owns_connection else None
        try:
            if transaction is not None:
                # This must remain the first SQL statement in the transaction.
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")

            game_patches = _materialize(
                connection,
                select(
                    GamePatch.patch_id,
                    GamePatch.patch_name,
                ).order_by(GamePatch.patch_id),
                ("patch_id", "patch_name"),
            )
            tournaments = _materialize(
                connection,
                select(
                    Tournament.tournament_id,
                    Tournament.name,
                    Tournament.region,
                    Tournament.season,
                ).order_by(Tournament.tournament_id),
                ("tournament_id", "name", "region", "season"),
            )
            tournament_stages = _materialize(
                connection,
                select(
                    TournamentStage.stage_id,
                    TournamentStage.tournament_id,
                    TournamentStage.name,
                ).order_by(TournamentStage.stage_id),
                ("stage_id", "tournament_id", "name"),
            )
            series = _materialize(
                connection,
                select(
                    MatchSeries.series_id,
                    MatchSeries.stage_id,
                    MatchSeries.best_of,
                ).order_by(MatchSeries.series_id),
                ("series_id", "stage_id", "best_of"),
            )
            teams = _materialize(
                connection,
                select(
                    Team.team_id,
                    Team.oracle_team_id,
                    Team.canonical_name,
                    Team.display_name,
                    Team.logo_file,
                ).order_by(Team.team_id),
                (
                    "team_id",
                    "oracle_team_id",
                    "canonical_name",
                    "display_name",
                    "logo_file",
                ),
            )
            players = _materialize(
                connection,
                select(
                    Player.player_id,
                    Player.oracle_player_id,
                    Player.canonical_name,
                    Player.display_name,
                    Player.photo_file,
                ).order_by(Player.player_id),
                (
                    "player_id",
                    "oracle_player_id",
                    "canonical_name",
                    "display_name",
                    "photo_file",
                ),
            )
            games = _materialize(
                connection,
                select(
                    Game.game_id,
                    Game.series_id,
                    Game.stage_id,
                    Game.patch_id,
                    Game.game_number,
                    Game.scheduled_at,
                    Game.started_at,
                    Game.ended_at,
                    Game.winner_team_id,
                ).order_by(Game.game_id),
                (
                    "game_id",
                    "series_id",
                    "stage_id",
                    "patch_id",
                    "game_number",
                    "scheduled_at",
                    "started_at",
                    "ended_at",
                    "winner_team_id",
                ),
            )
            game_teams = _materialize(
                connection,
                select(
                    GameTeam.game_id,
                    GameTeam.team_id,
                    GameTeam.side,
                    GameTeam.confirmation_status,
                ).order_by(GameTeam.game_id, GameTeam.side),
                (
                    "game_id",
                    "team_id",
                    "side",
                    "confirmation_status",
                ),
            )
            game_players = _materialize(
                connection,
                select(
                    GameTeam.game_id.label("game_id"),
                    GameTeam.side.label("side"),
                    GamePlayer.player_id,
                    GamePlayer.role,
                    GamePlayer.champion_id,
                    GamePlayer.confirmation_status,
                )
                .join(
                    GameTeam,
                    GamePlayer.game_team_id == GameTeam.game_team_id,
                )
                .order_by(
                    GameTeam.game_id,
                    GameTeam.side,
                    GamePlayer.role,
                ),
                (
                    "game_id",
                    "side",
                    "player_id",
                    "role",
                    "champion_id",
                    "confirmation_status",
                ),
            )
            champions = _materialize(
                connection,
                select(
                    Champion.champion_id,
                    Champion.canonical_name,
                    Champion.display_name,
                    Champion.image_file,
                ).order_by(Champion.champion_id),
                (
                    "champion_id",
                    "canonical_name",
                    "display_name",
                    "image_file",
                ),
            )
        finally:
            if transaction is not None:
                transaction.rollback()

    if champions.empty:
        raise ValueError(
            "Champion reference table is empty; mapping cannot be verified."
        )

    return OracleDatabaseSnapshot(
        targets=OracleTargetSnapshot(
            game_patches=game_patches,
            tournaments=tournaments,
            tournament_stages=tournament_stages,
            series=series,
            teams=teams,
            players=players,
            games=games,
            game_teams=game_teams,
            game_players=game_players,
        ),
        champions=champions,
    )


def materialize_database_snapshot(
    connection: Connection,
) -> OracleDatabaseSnapshot:
    """Materialize the Oracle target snapshot in a caller-owned transaction."""
    return load_database_snapshot(connection=connection)


def _identity_references(
    snapshot: OracleDatabaseSnapshot,
) -> tuple[
    tuple[TeamEntityReference, ...],
    tuple[PlayerEntityReference, ...],
    tuple[ChampionEntityReference, ...],
]:
    """Build immutable identity references from materialized rows."""
    team_references = tuple(
        TeamEntityReference(
            team_id=str(row.team_id),
            oracle_team_id=row.oracle_team_id,
            canonical_name=str(row.canonical_name),
            display_name=str(row.display_name),
            logo_file=row.logo_file,
        )
        for row in snapshot.targets.teams.itertuples(index=False)
    )
    player_references = tuple(
        PlayerEntityReference(
            player_id=str(row.player_id),
            oracle_player_id=row.oracle_player_id,
            canonical_name=str(row.canonical_name),
            display_name=str(row.display_name),
            photo_file=row.photo_file,
        )
        for row in snapshot.targets.players.itertuples(index=False)
    )
    champion_references = tuple(
        ChampionEntityReference(
            champion_id=str(row.champion_id),
            canonical_name=str(row.canonical_name),
            display_name=str(row.display_name),
            image_file=row.image_file,
        )
        for row in snapshot.champions.itertuples(index=False)
    )
    return team_references, player_references, champion_references


def _snapshot_counts(
    snapshot: OracleDatabaseSnapshot,
) -> dict[str, int]:
    """Count every materialized table at its reconciliation grain."""
    targets = snapshot.targets
    return {
        "champion": int(len(snapshot.champions)),
        "game": int(len(targets.games)),
        "game_patch": int(len(targets.game_patches)),
        "game_player": int(len(targets.game_players)),
        "game_team": int(len(targets.game_teams)),
        "player": int(len(targets.players)),
        "series": int(len(targets.series)),
        "team": int(len(targets.teams)),
        "tournament": int(len(targets.tournaments)),
        "tournament_stage": int(len(targets.tournament_stages)),
    }


def database_snapshot_counts(
    snapshot: OracleDatabaseSnapshot,
) -> dict[str, int]:
    """Expose deterministic snapshot counts to the approved apply CLI."""
    return _snapshot_counts(snapshot)


def _value_counts(dataframe: pd.DataFrame, column: str) -> dict[str, int]:
    """Return deterministic counts for a diagnostic column."""
    counts = dataframe[column].astype("string").value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _summary_mapping(
    summary: dict[str, object],
    key: str,
) -> dict[str, object]:
    """Read a required mapping from the transform summary."""
    value = summary.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Dry-run summary field '{key}' must be a mapping.")
    return {str(item_key): item_value for item_key, item_value in value.items()}


def validate_reconciliation(
    result: OracleDryRunResult,
    summary: dict[str, object],
) -> None:
    """Validate dynamic counts and target-shape invariants before success."""
    records = result.records
    frames = {
        "game_patch": records.game_patches,
        "tournament": records.tournaments,
        "tournament_stage": records.tournament_stages,
        "series": records.series,
        "team": records.teams,
        "player": records.players,
        "game": records.games,
        "game_team": records.game_teams,
        "game_player": records.game_players,
    }
    transformed = _summary_mapping(summary, "transformed_records")
    actions = _summary_mapping(summary, "actions")
    _summary_mapping(summary, "game_metadata_detail_counts")
    _summary_mapping(summary, "primary_rejection_reason_counts")
    _summary_mapping(summary, "series_diagnostic_reason_counts")
    failed: list[str] = []

    for table, frame in frames.items():
        if int(transformed.get(table, -1)) != len(frame):
            failed.append(f"{table}_record_count")
        table_actions = actions.get(table)
        if not isinstance(table_actions, dict):
            failed.append(f"{table}_actions_shape")
            continue
        action_total = sum(int(value) for value in table_actions.values())
        if action_total != len(frame):
            failed.append(f"{table}_action_count")

    count_checks = {
        key: summary.get(key) == result.source_counts[key]
        for key in (
            "source_rows",
            "player_rows",
            "team_rows",
            "distinct_games",
        )
    }
    count_checks.update(
        {
            "skipped": int(summary.get("skipped", -1))
            == len(result.skipped_records),
            "unresolved": int(summary.get("unresolved", -1))
            == len(result.unresolved_records),
            "rejected": int(summary.get("rejected", -1))
            == len(result.rejected_records),
            "primary_rejected_games": int(
                summary.get("primary_rejected_games", -1)
            )
            == len(result.rejected_games),
            "unmapped_champions": int(
                summary.get("unmapped_champions", -1)
            )
            == len(result.unmapped_champions),
        }
    )
    failed.extend(
        name for name, passed in count_checks.items() if not passed
    )

    games = records.games
    if not games.empty:
        parent_counts = games[["series_id", "stage_id"]].notna().sum(axis=1)
        if bool(parent_counts.ne(1).any()):
            failed.append("game_parent_xor")
        if bool(
            games[["scheduled_at", "started_at", "ended_at"]]
            .notna()
            .any()
            .any()
        ):
            failed.append("game_source_time_policy")
        if bool(games.duplicated(subset=["game_id"]).any()):
            failed.append("duplicate_game")

    emitted_games = len(games)
    source_games = int(result.source_counts["distinct_games"])
    if emitted_games == 0:
        expected_game_status = GAME_TRANSFORM_BLOCKED
    elif emitted_games == source_games:
        expected_game_status = GAME_TRANSFORM_READY
    else:
        expected_game_status = GAME_TRANSFORM_PARTIAL
    if summary.get("game_transform_status") != expected_game_status:
        failed.append("game_transform_status")

    game_teams = records.game_teams
    if len(game_teams) != 2 * len(games):
        failed.append("game_team_cardinality")
    if bool(game_teams.duplicated(subset=["game_id", "side"]).any()):
        failed.append("duplicate_game_side")
    if bool(game_teams.duplicated(subset=["game_id", "team_id"]).any()):
        failed.append("duplicate_game_team")
    for _, group in game_teams.groupby("game_id", sort=False, dropna=False):
        if set(group["side"].astype(str)) != {"BLUE", "RED"}:
            failed.append("game_side_set")

    if not games.empty:
        winner_membership = games[["game_id", "winner_team_id"]].merge(
            game_teams[["game_id", "team_id"]],
            left_on=["game_id", "winner_team_id"],
            right_on=["game_id", "team_id"],
            how="left",
        )
        if bool(winner_membership["team_id"].isna().any()):
            failed.append("winner_team_membership")

    game_players = records.game_players
    if len(game_players) != 10 * len(games):
        failed.append("game_player_cardinality")
    if bool(game_players["champion_id"].isna().any()):
        failed.append("game_player_champion_required")
    if bool(
        game_players.duplicated(
            subset=["game_id", "side", "role"]
        ).any()
    ):
        failed.append("duplicate_game_team_role")
    if not game_players.empty:
        unique_players = game_players.groupby("game_id")["player_id"].nunique()
        if bool(unique_players.ne(10).any()):
            failed.append("game_unique_players")
        expected_roles = set(ROLE_MAP.values())
        for _, group in game_players.groupby(
            ["game_id", "side"],
            sort=False,
            dropna=False,
        ):
            if (
                len(group) != 5
                or set(group["role"].astype(str)) != expected_roles
            ):
                failed.append("game_side_role_set")

    rejected_game_ids = result.rejected_games["gameid"].astype(str)
    accepted_game_ids = set(games["game_id"].astype(str))
    if bool(rejected_game_ids.duplicated().any()):
        failed.append("duplicate_primary_rejection")
    if accepted_game_ids & set(rejected_game_ids):
        failed.append("accepted_rejected_overlap")
    if len(accepted_game_ids) + len(rejected_game_ids) != source_games:
        failed.append("game_disposition_count")

    stages = records.tournament_stages
    if bool(stages["name"].eq("UNSPECIFIED").any()):
        failed.append("unspecified_stage")

    if failed:
        raise ValueError(
            "Dry-run reconciliation failed: " + ", ".join(sorted(set(failed)))
        )


def build_transform_result(
    core_data: OracleCoreData,
    snapshot: OracleDatabaseSnapshot,
) -> tuple[OracleDryRunResult, dict[str, object]]:
    """Resolve and transform Oracle rows against one materialized snapshot."""
    (
        team_references,
        player_references,
        champion_references,
    ) = _identity_references(snapshot)
    identity_result = resolve_oracle_identities(
        core_data,
        team_references=team_references,
        player_references=player_references,
        champion_references=champion_references,
    )
    series_analysis = analyze_series_evidence(core_data, identity_result)
    result = transform_oracle_targets(
        core_data,
        identity_result,
        target_snapshot=snapshot.targets,
        series_analysis=series_analysis,
    )
    summary = dict(build_dry_run_summary(result))
    validate_reconciliation(result, summary)
    return result, summary


def print_inspection_result(
    *,
    source_path: Path,
    source_sha256_before: str,
    source_sha256_after: str,
    snapshot: OracleDatabaseSnapshot,
    result: OracleDryRunResult,
    summary: dict[str, object],
) -> None:
    """Print dry-run reconciliation only; never create report files."""
    diagnostics = result.series_analysis.diagnostics
    unmapped = result.unmapped_champions
    transform_ready, dry_run_status = derive_dry_run_state(summary)

    print(f"source={source_path}")
    print(f"source_sha256_before={source_sha256_before}")
    print(f"source_sha256_after={source_sha256_after}")
    print("source_sha256_unchanged=true")
    print(f"mode={summary['mode']}")
    print(f"source_rows={summary['source_rows']}")
    print(f"player_rows={summary['player_rows']}")
    print(f"team_rows={summary['team_rows']}")
    print(f"distinct_games={summary['distinct_games']}")
    print(f"target_snapshot_rows={_json_value(_snapshot_counts(snapshot))}")
    print(f"series_status={summary['series_status']}")
    print(f"game_transform_status={summary['game_transform_status']}")
    print(
        "game_metadata_detail_counts="
        f"{_json_value(summary['game_metadata_detail_counts'])}"
    )
    print(f"series_diagnostic_rows={len(diagnostics)}")
    print(
        "series_diagnostic_status_counts="
        f"{_json_value(_value_counts(diagnostics, 'status'))}"
    )
    print(
        "series_diagnostic_reason_counts="
        f"{_json_value(summary['series_diagnostic_reason_counts'])}"
    )
    print(
        "series_game_status_counts="
        f"{_json_value(summary['series_game_status_counts'])}"
    )
    print(
        "transformed_records="
        f"{_json_value(summary['transformed_records'])}"
    )
    print(f"action_counts={_json_value(summary['actions'])}")
    print(f"skipped={summary['skipped']}")
    print(f"unresolved={summary['unresolved']}")
    print(
        "unresolved_reason_counts="
        f"{_json_value(summary['unresolved_reason_counts'])}"
    )
    print(f"rejected={summary['rejected']}")
    print(
        "rejected_reason_counts="
        f"{_json_value(summary['rejected_reason_counts'])}"
    )
    print(f"primary_rejected_games={summary['primary_rejected_games']}")
    print(
        "primary_rejection_reason_counts="
        f"{_json_value(summary['primary_rejection_reason_counts'])}"
    )
    print(f"unmapped_champions={summary['unmapped_champions']}")
    print(
        "unmapped_champion_reason_counts="
        f"{_json_value(_value_counts(unmapped, 'reason'))}"
    )
    print(f"transform_ready={str(transform_ready).lower()}")
    print(f"dry_run_status={dry_run_status}")
    print("reconciliation_valid=true")
    print(f"cutoff_status={summary['cutoff_status']}")
    print(f"transaction_status={summary['transaction_status']}")
    print(f"database_transaction={summary['transaction_status']}")
    print(
        "postgresql_changed="
        f"{str(summary['postgresql_changed']).lower()}"
    )


def main() -> None:
    """Run the read-only transform and optionally write Step 6E reports."""
    args = parse_args()
    source_path = resolve_source_path(args.source)
    source_sha256_before = calculate_sha256(source_path)
    core_data = prepare_oracle_core_data(source_path)
    database_snapshot = load_database_snapshot()
    result, summary = build_transform_result(
        core_data,
        database_snapshot,
    )
    summary["transaction_status"] = "READ_ONLY_ROLLED_BACK"
    summary["postgresql_changed"] = False
    source_sha256_after = calculate_sha256(source_path)

    if source_sha256_before != source_sha256_after:
        raise RuntimeError(
            "Oracle source SHA-256 changed during transform dry-run."
        )

    report_paths: dict[str, str] = {}
    if args.report_dir is not None:
        report_summary = build_etl_report_summary(
            result=result,
            transform_summary=summary,
            source_path=source_path,
            source_sha256_before=source_sha256_before,
            source_sha256_after=source_sha256_after,
            target_snapshot_rows=database_snapshot_counts(database_snapshot),
            run_timestamp_utc=datetime.now(timezone.utc),
            reconciliation_valid=True,
        )
        report_paths = write_etl_report_outputs(
            summary=report_summary,
            result=result,
            report_dir=args.report_dir,
            source_path=source_path,
        )

    print_inspection_result(
        source_path=source_path,
        source_sha256_before=source_sha256_before,
        source_sha256_after=source_sha256_after,
        snapshot=database_snapshot,
        result=result,
        summary=summary,
    )
    for output_name, output_path in report_paths.items():
        print(f"{output_name}={output_path}")


if __name__ == "__main__":
    main()
