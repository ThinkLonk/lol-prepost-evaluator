"""Nạp dữ liệu đội và logo từ CSV manifest đã được kiểm soát."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from sqlalchemy.orm import Session

from match_insight.data_processing.leaguepedia import (
    fetch_cargo_rows,
    fetch_file_urls,
)
from match_insight.data_processing.media_policy import load_media_exclusions
from match_insight.data_processing.reference_common import (
    SyncStats,
    asset_stem,
    bounded,
    canonical_key,
    clean_label,
    download_image,
    validate_stable_id,
)
from match_insight.database.models import Team

_LEAGUEPEDIA_WIKI_URL = "https://lol.fandom.com/wiki"
_LEAGUEPEDIA_SOURCE_NAME = "leaguepedia_cargo"
_LEAGUEPEDIA_FILE_PREFIX = "/wiki/Special:Redirect/file/"
_LEAGUEPEDIA_TEAM_FIELDS = (
    "Teams._pageID=SourcePageId",
    "Teams.Name=Name",
    "Teams.OverviewPage=OverviewPage",
    "Teams.Image=Image",
)

_REQUIRED_COLUMNS = (
    "team_id",
    "canonical_name",
    "display_name",
    "record_source",
    "record_source_url",
    "logo_url",
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamRecord:
    """Một team record đã đọc từ curated CSV."""

    team_id: str
    canonical_name: str
    display_name: str
    record_source: str
    record_source_url: str
    logo_url: str | None


def _leaguepedia_file_url(
    image_name: str,
) -> str | None:
    """Chuyển tên file wiki thành URL redirect tải được."""
    cleaned_name = image_name.strip()

    if not cleaned_name:
        return None

    encoded_name = quote(
        cleaned_name.replace(" ", "_"),
        safe="",
    )

    return (
        f"{_LEAGUEPEDIA_WIKI_URL}"
        f"/Special:Redirect/file/{encoded_name}"
    )


def _leaguepedia_file_name(url: str) -> str | None:
    """Lấy tên file từ URL redirect do adapter Leaguepedia tạo."""
    parsed = urlparse(url)

    if (
        (parsed.hostname or "").casefold()
        != "lol.fandom.com"
        or not parsed.path.startswith(
            _LEAGUEPEDIA_FILE_PREFIX
        )
    ):
        return None

    file_name = unquote(
        parsed.path[len(_LEAGUEPEDIA_FILE_PREFIX) :]
    ).strip()
    return file_name or None


def resolve_team_logo_urls(
    records: list[TeamRecord],
) -> list[TeamRecord]:
    """Thay Fandom redirect bằng URL CDN lấy qua imageinfo API."""
    file_names_by_team_id = {
        record.team_id: file_name
        for record in records
        if record.logo_url is not None
        and (
            file_name := _leaguepedia_file_name(
                record.logo_url
            )
        )
        is not None
    }

    if not file_names_by_team_id:
        return records

    direct_urls = fetch_file_urls(
        tuple(file_names_by_team_id.values())
    )
    resolved_records: list[TeamRecord] = []
    missing_count = 0

    for record in records:
        file_name = file_names_by_team_id.get(
            record.team_id
        )

        if file_name is None:
            resolved_records.append(record)
            continue

        direct_url = direct_urls.get(file_name)

        if direct_url is None:
            missing_count += 1

        resolved_records.append(
            replace(record, logo_url=direct_url)
        )

    if missing_count:
        _LOGGER.warning(
            "Leaguepedia imageinfo không ánh xạ được %s logo; "
            "các logo này được giữ ở trạng thái chưa ghi nhận.",
            missing_count,
        )

    return resolved_records


def fetch_teams(
    snapshot_path: Path | None = None,
) -> list[TeamRecord]:
    """Lấy toàn bộ team reference từ Leaguepedia Cargo."""
    rows = fetch_cargo_rows(
        table="Teams",
        fields=_LEAGUEPEDIA_TEAM_FIELDS,
        order_by="Teams._pageID",
        snapshot_path=snapshot_path,
    )

    records: list[TeamRecord] = []
    seen_team_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        source_page_id = (
            row.get("SourcePageId") or ""
        ).strip()
        overview_page = (
            row.get("OverviewPage") or ""
        ).strip()
        source_name = (row.get("Name") or "").strip()
        image_name = (row.get("Image") or "").strip()

        valid_page_id = (
            source_page_id.isascii()
            and source_page_id.isdigit()
            and int(source_page_id) > 0
        )

        if not valid_page_id:
            raise ValueError(
                f"Leaguepedia Teams row {row_number} có "
                f"SourcePageId không hợp lệ: "
                f"{source_page_id!r}"
            )

        if not overview_page:
            raise ValueError(
                f"Leaguepedia Teams row {row_number} "
                "thiếu OverviewPage."
            )

        team_id = f"lp_team_{source_page_id}"

        if team_id in seen_team_ids:
            raise ValueError(
                f"Leaguepedia Teams chứa team ID trùng: "
                f"{team_id}"
            )

        seen_team_ids.add(team_id)

        canonical_name = clean_label(
            source_name or overview_page,
            field_name="canonical_name",
        )

        records.append(
            TeamRecord(
                team_id=team_id,
                canonical_name=canonical_name,
                display_name=canonical_name,
                record_source=_LEAGUEPEDIA_SOURCE_NAME,
                record_source_url=(
                    f"{_LEAGUEPEDIA_WIKI_URL}"
                    f"/Special:Redirect/page/"
                    f"{source_page_id}"
                ),
                logo_url=_leaguepedia_file_url(image_name),
            )
        )

    return records


def _read_cell(
    row: dict[str, str | None],
    column: str,
) -> str:
    """Đọc một CSV cell và chuẩn hóa khoảng trắng biên."""
    return (row.get(column) or "").strip()


def load_teams_csv(path: Path) -> list[TeamRecord]:
    """Đọc và kiểm tra cấu trúc teams.csv."""
    records: list[TeamRecord] = []
    seen_team_ids: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        headers = set(reader.fieldnames or [])
        missing_columns = set(_REQUIRED_COLUMNS) - headers

        if missing_columns:
            raise ValueError(
                f"teams.csv thiếu cột: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            team_id = _read_cell(row, "team_id")
            canonical_name = _read_cell(row, "canonical_name")
            display_name = _read_cell(row, "display_name")
            record_source = _read_cell(row, "record_source")
            record_source_url = _read_cell(
                row,
                "record_source_url",
            )
            logo_url = _read_cell(row, "logo_url") or None

            required_values = {
                "team_id": team_id,
                "canonical_name": canonical_name,
                "display_name": display_name,
                "record_source": record_source,
                "record_source_url": record_source_url,
            }
            missing_values = [
                column
                for column, value in required_values.items()
                if not value
            ]

            if missing_values:
                raise ValueError(
                    f"teams.csv dòng {row_number} thiếu giá trị: "
                    f"{missing_values}"
                )

            if team_id in seen_team_ids:
                raise ValueError(
                    f"teams.csv chứa team_id trùng: {team_id}"
                )

            seen_team_ids.add(team_id)
            records.append(
                TeamRecord(
                    team_id=team_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    record_source=record_source,
                    record_source_url=record_source_url,
                    logo_url=logo_url,
                )
            )

    return records


def write_teams_csv(
    path: Path,
    records: list[TeamRecord],
) -> None:
    """Ghi teams.csv staging bằng thao tác nguyên tử."""
    if not records:
        raise ValueError(
            "Không ghi teams.csv khi danh sách team rỗng."
        )

    team_ids = [record.team_id for record in records]

    if len(team_ids) != len(set(team_ids)):
        raise ValueError(
            "Không ghi teams.csv vì có team_id trùng."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        f"{path.suffix}.part"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=_REQUIRED_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()

            for record in records:
                writer.writerow(
                    {
                        "team_id": record.team_id,
                        "canonical_name": (
                            record.canonical_name
                        ),
                        "display_name": record.display_name,
                        "record_source": (
                            record.record_source
                        ),
                        "record_source_url": (
                            record.record_source_url
                        ),
                        "logo_url": record.logo_url or "",
                    }
                )

        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sync_teams(
    session: Session,
    records: list[TeamRecord],
    project_root: Path,
    refresh_media: bool = False,
) -> SyncStats:
    """Tải logo và upsert team theo team_id."""
    stats = SyncStats()
    asset_dir = project_root / "assets" / "teams"
    excluded_ids = load_media_exclusions(project_root, "teams")
    team_ids = [record.team_id for record in records]

    if len(team_ids) != len(set(team_ids)):
        raise ValueError(
            "Danh sách team chứa team_id trùng."
        )

    for record in records:
        team_id = validate_stable_id(
            record.team_id,
            field_name="team_id",
            maximum=64,
        )
        canonical_name = bounded(
            clean_label(
                record.canonical_name,
                field_name="canonical_name",
            ),
            field_name="canonical_name",
            maximum=150,
        )
        display_name = bounded(
            clean_label(
                record.display_name,
                field_name="display_name",
            ),
            field_name="display_name",
            maximum=150,
        )

        clean_label(
            record.record_source,
            field_name="record_source",
        )
        clean_label(
            record.record_source_url,
            field_name="record_source_url",
        )
        canonical_key(canonical_name)

        team = session.get(Team, team_id)
        logo_file = team.logo_file if team is not None else None

        if record.logo_url is not None and team_id in excluded_ids:
            stats.media_skipped += 1
        if record.logo_url is not None and team_id not in excluded_ids:
            logo_file, downloaded = download_image(
                url=record.logo_url,
                asset_dir=asset_dir,
                stem=asset_stem(
                    stable_id=team_id,
                    canonical_name=canonical_name,
                ),
                project_root=project_root,
                refresh=refresh_media,
            )

            if downloaded:
                stats.media_downloaded += 1
            else:
                stats.media_skipped += 1

        if team is None:
            session.add(
                Team(
                    team_id=team_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    logo_file=logo_file,
                )
            )
            stats.inserted += 1
            continue

        desired_values = {
            "canonical_name": canonical_name,
            "display_name": display_name,
            "logo_file": logo_file,
        }
        changed = any(
            getattr(team, field_name) != value
            for field_name, value in desired_values.items()
        )

        if not changed:
            stats.skipped += 1
            continue

        for field_name, value in desired_values.items():
            setattr(team, field_name, value)

        stats.updated += 1

    session.flush()
    return stats
