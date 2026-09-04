"""CLI read-only để kiểm tra resolve định danh Oracle ở Bước 6C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from match_insight.data_processing.oracle_audit import (
    calculate_sha256,
    resolve_source_path,
)
from match_insight.data_processing.oracle_etl import (
    prepare_oracle_core_data,
)
from match_insight.data_processing.oracle_identity import (
    CHAMPION_MAPPED,
    CHAMPION_UNMAPPED,
    RECOVERED_UNIQUE,
    SOURCE_ID,
    TARGET_CONFLICT,
    TARGET_EXISTING,
    TARGET_NEW_REQUIRED,
    UNRESOLVED,
    ChampionEntityReference,
    OracleIdentityResult,
    PlayerEntityReference,
    TeamEntityReference,
    build_identity_summary,
    resolve_oracle_identities,
)
from match_insight.database.engine import engine
from match_insight.database.models import (
    Champion,
    Player,
    Team,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oracle"
    / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
)


def parse_args() -> argparse.Namespace:
    """Đọc đường dẫn CSV nguồn; không nhận output path."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve Oracle team/player/champion bằng transaction "
            "PostgreSQL read-only và chỉ in reconciliation ra console."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=(
            "Đường dẫn CSV nguồn. Mặc định dùng file Oracle 2025 "
            "trong data/raw/oracle."
        ),
    )
    return parser.parse_args()


def _json_value(value: object) -> str:
    """Mã hóa output ổn định để người dùng đối chiếu."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_reference_snapshots() -> tuple[
    tuple[TeamEntityReference, ...],
    tuple[PlayerEntityReference, ...],
    tuple[ChampionEntityReference, ...],
]:
    """SELECT reference trong một transaction read-only rồi rollback."""
    with engine.connect().execution_options(
        isolation_level="REPEATABLE READ"
    ) as connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql(
                "SET TRANSACTION READ ONLY"
            )
            team_rows = connection.execute(
                select(
                    Team.team_id,
                    Team.oracle_team_id,
                    Team.canonical_name,
                    Team.display_name,
                    Team.logo_file,
                ).order_by(Team.team_id)
            ).mappings()
            team_references = tuple(
                TeamEntityReference(
                    team_id=str(row["team_id"]),
                    oracle_team_id=row["oracle_team_id"],
                    canonical_name=str(
                        row["canonical_name"]
                    ),
                    display_name=str(row["display_name"]),
                    logo_file=row["logo_file"],
                )
                for row in team_rows
            )

            player_rows = connection.execute(
                select(
                    Player.player_id,
                    Player.oracle_player_id,
                    Player.canonical_name,
                    Player.display_name,
                    Player.photo_file,
                ).order_by(Player.player_id)
            ).mappings()
            player_references = tuple(
                PlayerEntityReference(
                    player_id=str(row["player_id"]),
                    oracle_player_id=(
                        row["oracle_player_id"]
                    ),
                    canonical_name=str(
                        row["canonical_name"]
                    ),
                    display_name=str(row["display_name"]),
                    photo_file=row["photo_file"],
                )
                for row in player_rows
            )

            champion_rows = connection.execute(
                select(
                    Champion.champion_id,
                    Champion.canonical_name,
                    Champion.display_name,
                    Champion.image_file,
                ).order_by(Champion.champion_id)
            ).mappings()
            champion_references = tuple(
                ChampionEntityReference(
                    champion_id=str(row["champion_id"]),
                    canonical_name=str(
                        row["canonical_name"]
                    ),
                    display_name=str(row["display_name"]),
                    image_file=row["image_file"],
                )
                for row in champion_rows
            )
        finally:
            transaction.rollback()

    if not champion_references:
        raise ValueError(
            "Champion reference table is empty; mapping cannot be verified."
        )

    return (
        team_references,
        player_references,
        champion_references,
    )


def _summary_dict(
    summary: dict[str, object],
    key: str,
) -> dict[str, int]:
    """Đọc một mapping count đã tạo nội bộ."""
    value = summary.get(key)
    if not isinstance(value, dict):
        raise TypeError(
            f"Identity summary field '{key}' must be a mapping."
        )
    return {
        str(item_key): int(item_value)
        for item_key, item_value in value.items()
    }


def validate_reconciliation(
    *,
    result: OracleIdentityResult,
    summary: dict[str, object],
    expected_team_groups: int,
    expected_player_rows: int,
) -> None:
    """Chặn output nếu một trạng thái không bao phủ đúng grain."""
    team_counts = _summary_dict(
        summary,
        "team_resolution_counts",
    )
    player_counts = _summary_dict(
        summary,
        "player_resolution_counts",
    )
    champion_counts = _summary_dict(
        summary,
        "champion_mapping_counts",
    )
    team_target_counts = _summary_dict(
        summary,
        "team_target_counts",
    )
    player_target_counts = _summary_dict(
        summary,
        "player_target_counts",
    )
    team_eligible = int(
        summary["distinct_team_source_ids_eligible"]
    )
    player_eligible = int(
        summary["distinct_player_source_ids_eligible"]
    )

    checks = {
        "team_result_rows": (
            len(result.team_participations)
            == expected_team_groups
        ),
        "team_resolution_total": (
            sum(team_counts.values())
            == expected_team_groups
        ),
        "player_result_rows": (
            len(result.player_rows)
            == expected_player_rows
        ),
        "player_resolution_total": (
            sum(player_counts.values())
            == expected_player_rows
        ),
        "champion_mapping_total": (
            sum(champion_counts.values())
            == expected_player_rows
        ),
        "team_target_total": (
            sum(team_target_counts.values())
            == team_eligible
        ),
        "player_target_total": (
            sum(player_target_counts.values())
            == player_eligible
        ),
    }
    failed_checks = [
        check_name
        for check_name, passed in checks.items()
        if not passed
    ]
    if failed_checks:
        raise ValueError(
            "Identity reconciliation failed: "
            + ", ".join(failed_checks)
        )


def print_inspection_result(
    *,
    source_path: Path,
    source_sha256_before: str,
    source_sha256_after: str,
    source_rows: int,
    source_player_rows: int,
    source_team_rows: int,
    distinct_games: int,
    team_reference_rows: int,
    player_reference_rows: int,
    champion_reference_rows: int,
    summary: dict[str, object],
) -> None:
    """In reconciliation theo grain, không ghi report."""
    team_counts = _summary_dict(
        summary,
        "team_resolution_counts",
    )
    player_counts = _summary_dict(
        summary,
        "player_resolution_counts",
    )
    champion_counts = _summary_dict(
        summary,
        "champion_mapping_counts",
    )
    team_target_counts = _summary_dict(
        summary,
        "team_target_counts",
    )
    player_target_counts = _summary_dict(
        summary,
        "player_target_counts",
    )
    sha256_unchanged = (
        source_sha256_before == source_sha256_after
    )

    print(f"source={source_path}")
    print(
        "source_sha256_before="
        f"{source_sha256_before}"
    )
    print(
        "source_sha256_after="
        f"{source_sha256_after}"
    )
    print(
        "source_sha256_unchanged="
        f"{str(sha256_unchanged).lower()}"
    )
    print(f"source_rows={source_rows}")
    print(f"source_player_rows={source_player_rows}")
    print(f"source_team_rows={source_team_rows}")
    print(f"distinct_games={distinct_games}")
    print(f"team_reference_rows={team_reference_rows}")
    print(f"player_reference_rows={player_reference_rows}")
    print(
        f"champion_reference_rows={champion_reference_rows}"
    )
    print(f"team_groups={summary['team_groups']}")
    print(
        "team_resolved_source_id="
        f"{team_counts.get(SOURCE_ID, 0)}"
    )
    print(
        "team_recovered_unique_context="
        f"{team_counts.get(RECOVERED_UNIQUE, 0)}"
    )
    print(
        "team_unresolved="
        f"{team_counts.get(UNRESOLVED, 0)}"
    )
    print(
        "team_reason_counts="
        f"{_json_value(summary['team_reason_counts'])}"
    )
    print(
        "distinct_team_source_ids_eligible="
        f"{summary['distinct_team_source_ids_eligible']}"
    )
    print(
        "team_target_existing="
        f"{team_target_counts.get(TARGET_EXISTING, 0)}"
    )
    print(
        "team_target_new_required="
        f"{team_target_counts.get(TARGET_NEW_REQUIRED, 0)}"
    )
    print(
        "team_target_conflict="
        f"{team_target_counts.get(TARGET_CONFLICT, 0)}"
    )
    print(
        "player_resolved_source_id="
        f"{player_counts.get(SOURCE_ID, 0)}"
    )
    print(
        "player_recovered_unique="
        f"{player_counts.get(RECOVERED_UNIQUE, 0)}"
    )
    print(
        "player_unresolved="
        f"{player_counts.get(UNRESOLVED, 0)}"
    )
    print(
        "player_reason_counts="
        f"{_json_value(summary['player_reason_counts'])}"
    )
    print(
        "distinct_player_source_ids_eligible="
        f"{summary['distinct_player_source_ids_eligible']}"
    )
    print(
        "player_target_existing="
        f"{player_target_counts.get(TARGET_EXISTING, 0)}"
    )
    print(
        "player_target_new_required="
        f"{player_target_counts.get(TARGET_NEW_REQUIRED, 0)}"
    )
    print(
        "player_target_conflict="
        f"{player_target_counts.get(TARGET_CONFLICT, 0)}"
    )
    print(f"champion_rows={summary['player_rows']}")
    print(
        "champion_mapped="
        f"{champion_counts.get(CHAMPION_MAPPED, 0)}"
    )
    print(
        "champion_unmapped="
        f"{champion_counts.get(CHAMPION_UNMAPPED, 0)}"
    )
    print(
        "champion_reason_counts="
        f"{_json_value(summary['champion_reason_counts'])}"
    )
    print(
        "unmapped_champions="
        f"{_json_value(summary['unmapped_champions'])}"
    )
    print("reconciliation_valid=true")
    print("cutoff_status=UNAVAILABLE")
    print("database_transaction=READ_ONLY_ROLLED_BACK")
    print("postgresql_changed=false")


def main() -> None:
    """Chạy resolve read-only và in kết quả để người dùng kiểm tra."""
    args = parse_args()
    source_path = resolve_source_path(args.source)
    source_sha256_before = calculate_sha256(source_path)
    core_data = prepare_oracle_core_data(source_path)
    (
        team_references,
        player_references,
        champion_references,
    ) = load_reference_snapshots()
    identity_result = resolve_oracle_identities(
        core_data,
        team_references=team_references,
        player_references=player_references,
        champion_references=champion_references,
    )
    summary = build_identity_summary(identity_result)
    source_sha256_after = calculate_sha256(source_path)

    if source_sha256_before != source_sha256_after:
        raise RuntimeError(
            "Oracle source SHA-256 changed during identity inspection."
        )

    validate_reconciliation(
        result=identity_result,
        summary=summary,
        expected_team_groups=len(core_data.team_rows),
        expected_player_rows=len(core_data.player_rows),
    )
    print_inspection_result(
        source_path=source_path,
        source_sha256_before=source_sha256_before,
        source_sha256_after=source_sha256_after,
        source_rows=len(core_data.dataframe),
        source_player_rows=len(core_data.player_rows),
        source_team_rows=len(core_data.team_rows),
        distinct_games=int(
            core_data.dataframe["gameid"].nunique(
                dropna=True
            )
        ),
        team_reference_rows=len(team_references),
        player_reference_rows=len(player_references),
        champion_reference_rows=len(champion_references),
        summary=summary,
    )


if __name__ == "__main__":
    main()
