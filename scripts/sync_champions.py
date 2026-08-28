"""CLI đồng bộ champion reference data từ Riot Data Dragon."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy.orm import Session

from match_insight.data_processing.champions import (
    fetch_champions,
    sync_champions,
)
from match_insight.data_processing.reference_common import write_sync_manifest
from match_insight.database.engine import engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Đồng bộ champion metadata và square assets "
            "từ Riot Data Dragon."
        )
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Data Dragon version cố định, ví dụ 16.16.1.",
    )
    parser.add_argument(
        "--locale",
        default="en_US",
        help="Locale metadata; mặc định en_US.",
    )
    parser.add_argument(
        "--refresh-media",
        action="store_true",
        help="Tải lại ảnh ngay cả khi file local đã tồn tại.",
    )
    return parser.parse_args()


def print_stats(
    inserted: int,
    updated: int,
    skipped: int,
    media_downloaded: int,
    media_skipped: int,
) -> None:
    """In kết quả theo hợp đồng chung của các sync script."""
    print(
        f"inserted={inserted} "
        f"updated={updated} "
        f"skipped={skipped} "
        f"media_downloaded={media_downloaded} "
        f"media_skipped={media_skipped}"
    )


def main() -> None:
    """Chạy toàn bộ champion sync trong một transaction."""
    args = parse_args()
    records, source_url = fetch_champions(
        version=args.version,
        locale=args.locale,
    )

    if not records:
        raise RuntimeError(
            "Data Dragon không trả về champion record nào."
        )

    with Session(engine) as session:
        stats = sync_champions(
            session=session,
            records=records,
            project_root=PROJECT_ROOT,
            refresh_media=args.refresh_media,
        )
        session.commit()

    write_sync_manifest(
        asset_dir=PROJECT_ROOT / "assets" / "champions",
        source={
            "type": "riot_data_dragon",
            "url": source_url,
            "version": args.version,
            "locale": args.locale,
            "record_count": len(records),
        },
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