"""CLI đồng bộ team từ Leaguepedia hoặc staging CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from match_insight.data_processing.leaguepedia import (
    LEAGUEPEDIA_API_URL,
    LEAGUEPEDIA_CLIENT_NAME,
)
from match_insight.data_processing.reference_common import (
    file_sha256,
    write_sync_manifest,
)
from match_insight.data_processing.teams import (
    TeamRecord,
    fetch_teams,
    load_teams_csv,
    resolve_team_logo_urls,
    sync_teams,
    write_teams_csv,
)
from match_insight.database.engine import engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "reference" / "teams.csv"
)
DEFAULT_RAW_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "leaguepedia"
    / "teams.json"
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Đồng bộ team metadata và logo từ "
            "Leaguepedia hoặc staging teams.csv."
        )
    )
    parser.add_argument(
        "--source",
        choices=("leaguepedia", "csv"),
        default="leaguepedia",
        help=(
            "Nguồn đầu vào. Mặc định tải toàn bộ "
            "Leaguepedia Teams."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Đường dẫn teams.csv. Với Leaguepedia đây "
            "là staging output; với csv đây là input."
        ),
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT,
        help=(
            "Đường dẫn raw Leaguepedia Teams snapshot."
        ),
    )
    parser.add_argument(
        "--refresh-media",
        action="store_true",
        help=(
            "Tải lại logo ngay cả khi file local "
            "đã tồn tại."
        ),
    )
    return parser.parse_args()


def _resolve_team_media(
    records: list[TeamRecord],
) -> list[TeamRecord]:
    """Ánh xạ logo qua imageinfo và báo tiến độ cho CLI."""
    print(
        "media_mapping_started="
        f"{len(records)} source=mediawiki_imageinfo"
    )
    resolved = resolve_team_logo_urls(records)
    print(
        "media_mapping_resolved="
        f"{sum(record.logo_url is not None for record in resolved)} "
        "media_mapping_missing="
        f"{sum(record.logo_url is None for record in resolved)}"
    )
    return resolved


def prepare_records(
    source: str,
    input_path: Path,
    raw_output_path: Path,
) -> tuple[list[TeamRecord], dict[str, object]]:
    """Chuẩn bị records và metadata nguồn cho manifest."""
    if source == "leaguepedia":
        records = fetch_teams(
            snapshot_path=raw_output_path,
        )

        if not records:
            raise RuntimeError(
                "Leaguepedia không trả team record nào."
            )

        records = _resolve_team_media(records)

        write_teams_csv(
            path=input_path,
            records=records,
        )

        source_details: dict[str, object] = {
            "type": "leaguepedia_cargo",
            "api_url": LEAGUEPEDIA_API_URL,
            "client": LEAGUEPEDIA_CLIENT_NAME,
            "media_url_resolution": "mediawiki_imageinfo",
            "table": "Teams",
            "raw_snapshot": str(raw_output_path),
            "raw_sha256": file_sha256(raw_output_path),
            "staging_file": str(input_path),
            "staging_sha256": file_sha256(input_path),
            "record_count": len(records),
        }

        return records, source_details

    if source == "csv":
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy team input file: "
                f"{input_path}"
            )

        records = load_teams_csv(input_path)

        if not records:
            raise RuntimeError(
                "teams.csv không có team record nào."
            )

        records = _resolve_team_media(records)

        source_details = {
            "type": "curated_csv",
            "input_file": str(input_path),
            "input_sha256": file_sha256(input_path),
            "media_url_resolution": "mediawiki_imageinfo",
            "record_count": len(records),
        }

        return records, source_details

    raise ValueError(
        f"Team source không được hỗ trợ: {source}"
    )


def print_stats(
    inserted: int,
    updated: int,
    skipped: int,
    media_downloaded: int,
    media_skipped: int,
) -> None:
    """In kết quả theo hợp đồng chung của sync script."""
    print(
        f"inserted={inserted} "
        f"updated={updated} "
        f"skipped={skipped} "
        f"media_downloaded={media_downloaded} "
        f"media_skipped={media_skipped}"
    )


def main() -> None:
    """Chạy toàn bộ team sync trong một transaction."""
    args = parse_args()
    input_path = args.input.resolve()
    raw_output_path = args.raw_output.resolve()

    records, source_details = prepare_records(
        source=args.source,
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    with Session(engine) as session:
        stats = sync_teams(
            session=session,
            records=records,
            project_root=PROJECT_ROOT,
            refresh_media=args.refresh_media,
        )
        session.commit()

    write_sync_manifest(
        asset_dir=PROJECT_ROOT / "assets" / "teams",
        source=source_details,
        stats=stats,
    )

    print_stats(
        inserted=stats.inserted,
        updated=stats.updated,
        skipped=stats.skipped,
        media_downloaded=stats.media_downloaded,
        media_skipped=stats.media_skipped,
    )


if __name__ == "__main__":
    main()
