"""Nạp reference tuyển thủ chuyên nghiệp và ảnh từ Leaguepedia."""

from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from sqlalchemy import select
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
from match_insight.database.models import (
    GamePlayer,
    Player,
    TeamMembership,
)

_LEAGUEPEDIA_WIKI_URL = "https://lol.fandom.com/wiki"
_LEAGUEPEDIA_SOURCE_NAME = "leaguepedia_cargo"
_LEAGUEPEDIA_FILE_PREFIX = "/wiki/Special:Redirect/file/"
_PLAYER_SNAPSHOT_SCHEMA_VERSION = 2
_LEAGUEPEDIA_PLAYER_FIELDS = (
    "Players._pageID=SourcePageId",
    "Players.OverviewPage=SourcePageName",
    "Players.ID=PlayerHandle",
    "Players.Image=Image",
    "Players.IsPersonality=IsPersonality",
)
_LEAGUEPEDIA_GAME_EVIDENCE_FIELDS = (
    "PlayerLeagueHistory.Player=PlayerPage",
)
_LEAGUEPEDIA_TOURNAMENT_ROSTER_FIELDS = (
    "TournamentPlayers.Player=PlayerPage",
    "TournamentPlayers.Role=Role",
)
_LEAGUEPEDIA_CURRENT_ROSTER_FIELDS = (
    "ListplayerCurrent.Link=PlayerPage",
    "ListplayerCurrent.Role=Role",
)
_LEAGUEPEDIA_PLAYER_IMAGE_TABLES = (
    "PlayerImages=PI,Tournaments=T"
)
_LEAGUEPEDIA_PLAYER_IMAGE_JOIN = (
    "PI.Tournament=T.OverviewPage"
)
_LEAGUEPEDIA_PLAYER_IMAGE_EFFECTIVE_DATE = (
    "COALESCE(PI.SortDate,T.DateStartFuzzy,T.Date)"
)
_LEAGUEPEDIA_PLAYER_IMAGE_FIELDS = (
    "PI.Link=PlayerPage",
    "PI.FileName=FileName",
    "PI.Tournament=Tournament",
    "PI.IsProfileImage=IsProfileImage",
    "PI.SortDate=SortDate",
    "T.DateStartFuzzy=TournamentDateStartFuzzy",
    "T.Date=TournamentDate",
    (
        f"{_LEAGUEPEDIA_PLAYER_IMAGE_EFFECTIVE_DATE}="
        "EffectiveSortDate"
    ),
)

_PROFILE_IMAGE_YEAR_PATTERN = re.compile(
    r"(?<!\d)((?:19|20)\d{2})(?!\d)"
)
_MEDIAWIKI_REVISION_PATTERN = re.compile(
    r"(?:^|[?&])cb=(\d{8,14})(?:&|$)"
)
_PROFILE_IMAGE_PHASE_MONTHS = (
    (re.compile(r"\ball[- ]star\b"), 11),
    (re.compile(r"\b(?:worlds?|world championship|wcs?|wc)\b"), 10),
    (re.compile(r"\b(?:split\s*3|fall|autumn)\b"), 9),
    (
        re.compile(
            r"\b(?:s(?:pl|p)it\s*2|spilt\s*2|summer|closing|clausura)\b"
        ),
        7,
    ),
    (re.compile(r"\b(?:msi|mid[- ]season|first stand)\b"), 5),
    (
        re.compile(
            r"\b(?:split\s*1|spring|opening|apertura|ouverture)\b"
        ),
        3,
    ),
    (re.compile(r"\bkickoff\b"), 2),
    (re.compile(r"\bwinter\b"), 1),
)

_IN_GAME_ROLE_ALIASES = {
    "ad",
    "ad carry",
    "adc",
    "bot",
    "bot lane",
    "bot laner",
    "bottom",
    "bottom lane",
    "bottom laner",
    "jungle",
    "jungler",
    "mid",
    "mid lane",
    "mid laner",
    "middle",
    "middle lane",
    "middle laner",
    "marksman",
    "sup",
    "support",
    "top",
    "top lane",
    "top laner",
}

_REQUIRED_COLUMNS = (
    "player_id",
    "canonical_name",
    "display_name",
    "record_source",
    "record_source_url",
    "photo_url",
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerRecord:
    """Một player record đã chuẩn hóa từ nguồn reference."""

    player_id: str
    canonical_name: str
    display_name: str
    record_source: str
    record_source_url: str
    photo_url: str | None


@dataclass(frozen=True)
class PlayerSelectionStats:
    """Dấu vết lọc hồ sơ Leaguepedia về đúng grain tuyển thủ."""

    catalog_total: int
    included: int
    included_by_game: int
    included_by_roster_only: int
    excluded_personality: int
    excluded_without_player_evidence: int
    duplicate_handle_records: int
    media_available: int
    media_selected_by_manual_override: int = 0
    media_selected_by_playerimages_sort_date: int = 0
    media_selected_by_tournament_start_fuzzy: int = 0
    media_selected_by_tournament_date: int = 0
    media_selected_by_effective_sort_date: int = 0
    media_selected_by_filename_date: int = 0
    media_selected_by_revision_timestamp: int = 0
    media_selected_without_date: int = 0


@dataclass(frozen=True)
class PlayerSelectionResult:
    """Danh sách tuyển thủ đã lọc cùng thống kê truy vết."""

    records: list[PlayerRecord]
    stats: PlayerSelectionStats
    media_urls_are_direct: bool


@dataclass(frozen=True)
class PlayerPruneResult:
    """Kết quả dọn các player Leaguepedia ngoài tập hợp hợp lệ."""

    deleted: int
    protected: int
    photo_files: tuple[str, ...]


@dataclass(frozen=True)
class PlayerImageSortRefreshStats:
    """Dấu vết làm giàu ngày xếp ảnh trong player snapshot."""

    fetched_rows: int
    matched_rows: int
    unmatched_snapshot_rows: int
    new_source_rows: int
    duplicate_source_rows_collapsed: int


@dataclass
class PlayerMediaSyncResult(SyncStats):
    """Kết quả media kèm ID để checkpoint theo từng batch."""

    succeeded_player_ids: tuple[str, ...] = ()
    failed_player_ids: tuple[str, ...] = ()


def _page_key(value: str) -> str:
    """Chuẩn hóa tên trang chỉ để đối chiếu, không thay stable ID."""

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    )
    return " ".join(
        normalized.replace("_", " ").split()
    ).casefold()


def _cargo_boolean(value: object) -> bool:
    """Đọc Boolean Cargo từ các dạng chuỗi thường gặp."""

    if isinstance(value, bool):
        return value

    return str(value or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
    }


def is_in_game_role(value: str) -> bool:
    """Nhận diện đúng năm nhóm vai trò thi đấu của dự án."""

    normalized = unicodedata.normalize(
        "NFKC",
        value,
    ).casefold()
    parts = re.split(r"[/,;|&]+", normalized)

    for part in parts:
        role = " ".join(
            re.sub(r"[^a-z]+", " ", part).split()
        )

        if role in _IN_GAME_ROLE_ALIASES:
            return True

    return False


def _evidence_page_keys(
    rows: list[dict[str, str]],
    *,
    require_in_game_role: bool,
) -> set[str]:
    """Lấy trang có bằng chứng vai trò thi đấu hợp lệ."""

    page_keys: set[str] = set()

    for row in rows:
        if require_in_game_role and not is_in_game_role(
            row.get("Role") or ""
        ):
            continue

        page_key = _page_key(
            row.get("PlayerPage") or ""
        )

        if page_key:
            page_keys.add(page_key)

    return page_keys


def _parse_profile_image_date(value: object) -> date | None:
    """Đọc ngày ISO từ Cargo mà không chấp nhận ngày không hợp lệ."""

    cleaned = str(value or "").strip()

    if not cleaned:
        return None

    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        return None


def _filename_profile_image_date(
    file_name: str,
) -> date | None:
    """Suy ngày tương đối từ năm/giai đoạn chỉ khi nguồn thiếu ngày."""

    normalized = unicodedata.normalize(
        "NFKC",
        file_name,
    ).casefold()
    years = [
        int(year)
        for year in _PROFILE_IMAGE_YEAR_PATTERN.findall(
            normalized
        )
    ]

    if not years:
        return None

    phase_month = 1

    for pattern, month in _PROFILE_IMAGE_PHASE_MONTHS:
        if pattern.search(normalized):
            phase_month = month
            break

    phase_day = (
        20
        if re.search(r"\bplayoffs?\b", normalized)
        else 1
    )
    return date(years[-1], phase_month, phase_day)


def _mediawiki_revision_timestamp(
    photo_url: str,
) -> str:
    """Lấy timestamp revision để phá hòa, không coi là ngày thi đấu."""

    matched = _MEDIAWIKI_REVISION_PATTERN.search(
        photo_url
    )

    if matched is None:
        return ""

    normalized = matched.group(1).ljust(14, "0")

    try:
        datetime.strptime(normalized, "%Y%m%d%H%M%S")
    except ValueError:
        return ""

    return normalized


def _profile_image_effective_date(
    row: dict[str, str],
) -> tuple[date | None, str]:
    """Trả ngày xếp ảnh và nguồn bằng chứng đã dùng."""

    leaguepedia_fields = (
        ("SortDate", "playerimages_sort_date"),
        (
            "TournamentDateStartFuzzy",
            "tournament_start_fuzzy",
        ),
        ("TournamentDate", "tournament_date"),
        ("EffectiveSortDate", "effective_sort_date"),
    )

    for field_name, date_source in leaguepedia_fields:
        leaguepedia_date = _parse_profile_image_date(
            row.get(field_name)
        )

        if leaguepedia_date is not None:
            return leaguepedia_date, date_source

    filename_date = _filename_profile_image_date(
        row.get("FileName") or ""
    )

    if filename_date is not None:
        return filename_date, "filename_date"

    revision_timestamp = _mediawiki_revision_timestamp(
        row.get("PhotoUrl") or ""
    )

    if revision_timestamp:
        try:
            return (
                date.fromisoformat(
                    "-".join(
                        (
                            revision_timestamp[:4],
                            revision_timestamp[4:6],
                            revision_timestamp[6:8],
                        )
                    )
                ),
                "revision_timestamp",
            )
        except ValueError:
            pass

    return None, "undated"


def _profile_image_sort_key(
    row: dict[str, str],
) -> tuple[int, int, int, int, str, str, str]:
    """Tạo khóa xếp ảnh ổn định, có timestamp upload chỉ để phá hòa."""

    effective_date, date_source = _profile_image_effective_date(
        row
    )
    source_rank = {
        "playerimages_sort_date": 4,
        "tournament_start_fuzzy": 3,
        "tournament_date": 2,
        "effective_sort_date": 2,
        "filename_date": 1,
        "revision_timestamp": 0,
        "undated": 0,
    }[date_source]
    has_semantic_date = date_source in {
        "playerimages_sort_date",
        "tournament_start_fuzzy",
        "tournament_date",
        "effective_sort_date",
        "filename_date",
    }
    file_name = (row.get("FileName") or "").strip()
    return (
        int(has_semantic_date),
        (
            effective_date.toordinal()
            if has_semantic_date
            and effective_date is not None
            else 0
        ),
        int(row.get("_SelectionOrigin") == "players_image"),
        source_rank,
        _mediawiki_revision_timestamp(
            row.get("PhotoUrl") or ""
        ),
        file_name.casefold(),
        file_name,
    )


def _profile_images_by_page(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Chọn ảnh profile mới nhất cho mỗi trang tuyển thủ."""

    candidates: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        page_key = _page_key(
            row.get("PlayerPage") or ""
        )
        file_name = (row.get("FileName") or "").strip()

        if (
            not page_key
            or not file_name
            or not _cargo_boolean(
                row.get("IsProfileImage")
            )
        ):
            continue

        candidates.setdefault(page_key, []).append(row)

    return {
        page_key: max(
            page_rows,
            key=_profile_image_sort_key,
        )
        for page_key, page_rows in candidates.items()
    }


def _profile_image_rows_by_file(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Lập chỉ mục ảnh profile để làm giàu candidate từ Players.Image."""

    candidates: dict[
        tuple[str, str], list[dict[str, str]]
    ] = {}

    for row in rows:
        row_key = (
            _page_key(row.get("PlayerPage") or ""),
            _page_key(row.get("FileName") or ""),
        )

        if not all(row_key) or not _cargo_boolean(
            row.get("IsProfileImage")
        ):
            continue

        candidates.setdefault(row_key, []).append(row)

    return {
        row_key: max(page_rows, key=_profile_image_sort_key)
        for row_key, page_rows in candidates.items()
    }


def _direct_media_file_name(photo_url: str) -> str | None:
    """Lấy filename từ URL CDN MediaWiki để kiểm tra Image/PhotoUrl."""

    parsed = urlparse(photo_url)
    path_parts = [
        unquote(part)
        for part in parsed.path.split("/")
        if part
    ]

    if not path_parts:
        return None

    revision_positions = [
        index
        for index, part in enumerate(path_parts)
        if part.casefold() == "revision"
    ]

    if revision_positions:
        revision_index = revision_positions[-1]

        if revision_index == 0:
            return None

        return path_parts[revision_index - 1]

    return path_parts[-1]


def _manual_profile_image_candidate(
    *,
    player_page: str,
    image_name: str,
    photo_url: str,
    profile_rows_by_file: dict[
        tuple[str, str], dict[str, str]
    ],
) -> dict[str, str] | None:
    """Biến Players.Image thành candidate thay vì override vô điều kiện."""

    cleaned_image_name = image_name.strip()
    direct_file_name = _direct_media_file_name(photo_url)

    if not cleaned_image_name and direct_file_name is not None:
        cleaned_image_name = direct_file_name

    if not cleaned_image_name:
        return None

    matching_row = profile_rows_by_file.get(
        (
            _page_key(player_page),
            _page_key(cleaned_image_name),
        )
    )
    candidate = (
        dict(matching_row)
        if matching_row is not None
        else {
            "PlayerPage": player_page,
            "FileName": cleaned_image_name,
            "IsProfileImage": "1",
        }
    )
    candidate["_SelectionOrigin"] = "players_image"

    if (
        photo_url
        and direct_file_name is not None
        and _page_key(direct_file_name)
        == _page_key(cleaned_image_name)
    ):
        candidate["PhotoUrl"] = photo_url

    if not (candidate.get("PhotoUrl") or "").strip():
        candidate["PhotoUrl"] = (
            _leaguepedia_file_url(cleaned_image_name) or ""
        )

    return candidate


def _leaguepedia_file_url(
    image_name: str,
) -> str | None:
    """Chuyển tên file MediaWiki thành URL redirect."""

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


def _leaguepedia_file_name(
    url: str,
) -> str | None:
    """Lấy tên file từ URL redirect do adapter tạo."""

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


def resolve_player_photo_urls(
    records: list[PlayerRecord],
) -> list[PlayerRecord]:
    """Thay URL Fandom redirect bằng URL CDN từ imageinfo."""

    file_names_by_player_id = {
        record.player_id: file_name
        for record in records
        if record.photo_url is not None
        and (
            file_name := _leaguepedia_file_name(
                record.photo_url
            )
        )
        is not None
    }

    if not file_names_by_player_id:
        return records

    direct_urls = fetch_file_urls(
        tuple(file_names_by_player_id.values())
    )
    resolved_records: list[PlayerRecord] = []
    missing_count = 0

    for record in records:
        file_name = file_names_by_player_id.get(
            record.player_id
        )

        if file_name is None:
            resolved_records.append(record)
            continue

        direct_url = direct_urls.get(file_name)

        if direct_url is None:
            missing_count += 1

        resolved_records.append(
            replace(
                record,
                photo_url=direct_url,
            )
        )

    if missing_count:
        _LOGGER.warning(
            "Leaguepedia imageinfo không ánh xạ được %s ảnh "
            "tuyển thủ; các ảnh này được giữ ở trạng thái "
            "chưa ghi nhận.",
            missing_count,
        )

    return resolved_records


def fetch_player_image_rows(
    snapshot_path: Path | None = None,
) -> list[dict[str, str]]:
    """Lấy ngày xếp ảnh đúng theo truy vấn infobox Leaguepedia."""

    return fetch_cargo_rows(
        table=_LEAGUEPEDIA_PLAYER_IMAGE_TABLES,
        fields=_LEAGUEPEDIA_PLAYER_IMAGE_FIELDS,
        join_on=_LEAGUEPEDIA_PLAYER_IMAGE_JOIN,
        where="PI.IsProfileImage=1",
        order_by=(
            "PI.Link, "
            f"{_LEAGUEPEDIA_PLAYER_IMAGE_EFFECTIVE_DATE}, "
            "PI.FileName"
        ),
        snapshot_path=snapshot_path,
    )


def fetch_players(
    snapshot_path: Path | None = None,
) -> PlayerSelectionResult:
    """Lấy catalog và bằng chứng để chọn đúng tuyển thủ toàn cầu."""

    catalog_rows = fetch_cargo_rows(
        table="Players",
        fields=_LEAGUEPEDIA_PLAYER_FIELDS,
        order_by="Players._pageID",
    )

    game_evidence_rows = fetch_cargo_rows(
        table="PlayerLeagueHistory",
        fields=_LEAGUEPEDIA_GAME_EVIDENCE_FIELDS,
        where="PlayerLeagueHistory.TotalGames>0",
        order_by="PlayerLeagueHistory.Player",
        group_by="PlayerLeagueHistory.Player",
    )
    tournament_roster_rows = fetch_cargo_rows(
        table="TournamentPlayers",
        fields=_LEAGUEPEDIA_TOURNAMENT_ROSTER_FIELDS,
        order_by=(
            "TournamentPlayers.Player, "
            "TournamentPlayers.Role"
        ),
        group_by=(
            "TournamentPlayers.Player, "
            "TournamentPlayers.Role"
        ),
    )
    current_roster_rows = fetch_cargo_rows(
        table="ListplayerCurrent",
        fields=_LEAGUEPEDIA_CURRENT_ROSTER_FIELDS,
        order_by=(
            "ListplayerCurrent.Link, "
            "ListplayerCurrent.Role"
        ),
        group_by=(
            "ListplayerCurrent.Link, "
            "ListplayerCurrent.Role"
        ),
    )
    image_rows = fetch_player_image_rows()
    roster_rows = [
        *(
            {
                **row,
                "EvidenceSource": "TournamentPlayers",
            }
            for row in tournament_roster_rows
        ),
        *(
            {
                **row,
                "EvidenceSource": "ListplayerCurrent",
            }
            for row in current_roster_rows
        ),
    ]
    result = _records_from_leaguepedia_rows(
        catalog_rows=catalog_rows,
        game_evidence_rows=game_evidence_rows,
        roster_rows=roster_rows,
        image_rows=image_rows,
    )

    if snapshot_path is not None:
        write_players_snapshot(
            path=snapshot_path,
            catalog_rows=catalog_rows,
            game_evidence_rows=game_evidence_rows,
            roster_rows=roster_rows,
            image_rows=image_rows,
            media_urls_are_direct=False,
        )

    return result


def _records_from_leaguepedia_rows(
    catalog_rows: list[dict[str, str]],
    game_evidence_rows: list[dict[str, str]],
    roster_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    *,
    direct_media_urls: bool = False,
) -> PlayerSelectionResult:
    """Lọc hồ sơ có bằng chứng thi đấu rồi tạo player records."""

    game_page_keys = _evidence_page_keys(
        game_evidence_rows,
        require_in_game_role=False,
    )
    roster_page_keys = _evidence_page_keys(
        roster_rows,
        require_in_game_role=True,
    )
    profile_images = _profile_images_by_page(image_rows)
    profile_rows_by_file = _profile_image_rows_by_file(
        image_rows
    )
    candidates: list[
        tuple[dict[str, str], bool, bool]
    ] = []
    seen_player_ids: set[str] = set()
    excluded_personality = 0
    excluded_without_evidence = 0

    for row_number, row in enumerate(
        catalog_rows,
        start=1,
    ):
        source_page_id = (
            row.get("SourcePageId") or ""
        ).strip()
        source_page_name = (
            row.get("SourcePageName") or ""
        ).strip()
        player_handle = (
            row.get("PlayerHandle") or ""
        ).strip()
        valid_page_id = (
            source_page_id.isascii()
            and source_page_id.isdigit()
            and int(source_page_id) > 0
        )

        if not valid_page_id:
            raise ValueError(
                f"Leaguepedia Players row {row_number} có "
                f"SourcePageId không hợp lệ: "
                f"{source_page_id!r}"
            )

        if not source_page_name:
            raise ValueError(
                f"Leaguepedia Players row {row_number} "
                "thiếu SourcePageName."
            )

        player_id = f"lp_player_{source_page_id}"

        if player_id in seen_player_ids:
            raise ValueError(
                "Leaguepedia Players chứa player ID trùng: "
                f"{player_id}"
            )

        seen_player_ids.add(player_id)

        page_key = _page_key(source_page_name)
        has_game_evidence = page_key in game_page_keys
        has_roster_evidence = page_key in roster_page_keys

        if not has_game_evidence and not has_roster_evidence:
            if _cargo_boolean(row.get("IsPersonality")):
                excluded_personality += 1
            else:
                excluded_without_evidence += 1

            continue

        candidates.append(
            (
                row,
                has_game_evidence,
                has_roster_evidence,
            )
        )

    handle_keys = [
        _page_key(
            (row.get("PlayerHandle") or "").strip()
            or (row.get("SourcePageName") or "").strip()
        )
        for row, _, _ in candidates
    ]
    handle_counts = Counter(handle_keys)
    duplicate_handle_records = sum(
        count
        for count in handle_counts.values()
        if count > 1
    )
    records: list[PlayerRecord] = []
    included_by_game = 0
    included_by_roster_only = 0
    media_selection_sources: Counter[str] = Counter()

    for row, has_game_evidence, _ in candidates:
        source_page_id = (
            row.get("SourcePageId") or ""
        ).strip()
        source_page_name = (
            row.get("SourcePageName") or ""
        ).strip()
        player_handle = (
            row.get("PlayerHandle") or ""
        ).strip()
        player_id = f"lp_player_{source_page_id}"

        canonical_name = clean_label(
            player_handle or source_page_name,
            field_name="canonical_name",
        )
        display_name = canonical_name

        if handle_counts[_page_key(canonical_name)] > 1:
            display_name = clean_label(
                source_page_name,
                field_name="display_name",
            )

        manual_image_name = (
            row.get("Image") or ""
        ).strip()
        manual_photo_url = (
            row.get("PhotoUrl") or ""
        ).strip()
        profile_image = profile_images.get(
            _page_key(source_page_name)
        )
        manual_candidate = _manual_profile_image_candidate(
            player_page=source_page_name,
            image_name=manual_image_name,
            photo_url=manual_photo_url,
            profile_rows_by_file=profile_rows_by_file,
        )

        image_candidates = [
            candidate
            for candidate in (
                profile_image,
                manual_candidate,
            )
            if candidate is not None
        ]
        selected_image = (
            max(image_candidates, key=_profile_image_sort_key)
            if image_candidates
            else None
        )
        image_name = (
            (selected_image.get("FileName") or "").strip()
            if selected_image is not None
            else ""
        )
        direct_photo_url = (
            (selected_image.get("PhotoUrl") or "").strip()
            if selected_image is not None
            else ""
        )

        if has_game_evidence:
            included_by_game += 1
        else:
            included_by_roster_only += 1

        photo_url = (
            direct_photo_url or None
            if direct_media_urls
            else _leaguepedia_file_url(image_name)
        )

        if photo_url is not None:
            if (
                selected_image is not None
                and selected_image.get("_SelectionOrigin")
                == "players_image"
            ):
                media_selection_sources[
                    "manual_override"
                ] += 1
            elif selected_image is not None:
                _, date_source = (
                    _profile_image_effective_date(
                        selected_image
                    )
                )
                media_selection_sources[date_source] += 1

        records.append(
            PlayerRecord(
                player_id=player_id,
                canonical_name=canonical_name,
                display_name=display_name,
                record_source=_LEAGUEPEDIA_SOURCE_NAME,
                record_source_url=(
                    f"{_LEAGUEPEDIA_WIKI_URL}"
                    f"/Special:Redirect/page/"
                    f"{source_page_id}"
                ),
                photo_url=photo_url,
            )
        )

    stats = PlayerSelectionStats(
        catalog_total=len(catalog_rows),
        included=len(records),
        included_by_game=included_by_game,
        included_by_roster_only=(
            included_by_roster_only
        ),
        excluded_personality=excluded_personality,
        excluded_without_player_evidence=(
            excluded_without_evidence
        ),
        duplicate_handle_records=(
            duplicate_handle_records
        ),
        media_available=sum(
            record.photo_url is not None
            for record in records
        ),
        media_selected_by_manual_override=(
            media_selection_sources["manual_override"]
        ),
        media_selected_by_playerimages_sort_date=(
            media_selection_sources[
                "playerimages_sort_date"
            ]
        ),
        media_selected_by_tournament_start_fuzzy=(
            media_selection_sources[
                "tournament_start_fuzzy"
            ]
        ),
        media_selected_by_tournament_date=(
            media_selection_sources["tournament_date"]
        ),
        media_selected_by_effective_sort_date=(
            media_selection_sources[
                "effective_sort_date"
            ]
        ),
        media_selected_by_filename_date=(
            media_selection_sources["filename_date"]
        ),
        media_selected_by_revision_timestamp=(
            media_selection_sources[
                "revision_timestamp"
            ]
        ),
        media_selected_without_date=(
            media_selection_sources["undated"]
        ),
    )

    return PlayerSelectionResult(
        records=records,
        stats=stats,
        media_urls_are_direct=direct_media_urls,
    )


def write_players_snapshot(
    path: Path,
    catalog_rows: list[dict[str, str]],
    game_evidence_rows: list[dict[str, str]],
    roster_rows: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    *,
    media_urls_are_direct: bool,
) -> None:
    """Ghi bundle raw đủ để tái hiện bước chọn tuyển thủ."""

    if not catalog_rows:
        raise ValueError(
            "Không ghi player snapshot khi catalog rỗng."
        )

    payload: dict[str, object] = {
        "schema_version": _PLAYER_SNAPSHOT_SCHEMA_VERSION,
        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "type": "leaguepedia_player_reference",
            "api_url": (
                "https://lol.fandom.com/api.php"
            ),
            "catalog_table": "Players",
            "evidence_tables": [
                "PlayerLeagueHistory",
                "TournamentPlayers",
                "ListplayerCurrent",
            ],
            "media_table": "PlayerImages",
            "media_sort_policy": (
                "COALESCE(PlayerImages.SortDate,"
                "Tournaments.DateStartFuzzy,Tournaments.Date)"
            ),
        },
        "media_urls_are_direct": (
            media_urls_are_direct
        ),
        "catalog_row_count": len(catalog_rows),
        "game_evidence_row_count": len(
            game_evidence_rows
        ),
        "roster_row_count": len(roster_rows),
        "image_row_count": len(image_rows),
        "catalog_rows": catalog_rows,
        "game_evidence_rows": game_evidence_rows,
        "roster_rows": roster_rows,
        "image_rows": image_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")

    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _player_image_row_key(
    row: dict[str, str],
) -> tuple[str, str]:
    """Tạo khóa ghép ổn định giữa snapshot ảnh và Cargo enrichment."""

    return (
        _page_key(row.get("PlayerPage") or ""),
        _page_key(row.get("FileName") or ""),
    )


def refresh_players_snapshot_image_sort_dates(
    path: Path,
    *,
    query_snapshot_path: Path | None = None,
    reuse_query_snapshot: bool = False,
) -> PlayerImageSortRefreshStats:
    """Bổ sung ngày giải vào snapshot ảnh, giữ URL media đã tải."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "players.json khong phai JSON hop le."
        ) from error

    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != _PLAYER_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError(
            "players snapshot phải dùng schema_version=2."
        )

    image_rows = payload.get("image_rows")
    declared_count = payload.get("image_row_count")

    if (
        not isinstance(image_rows, list)
        or not isinstance(declared_count, int)
        or isinstance(declared_count, bool)
        or declared_count != len(image_rows)
    ):
        raise ValueError(
            "players snapshot có image_rows không hợp lệ."
        )

    normalized_existing_rows: list[dict[str, str]] = []

    for row_number, row in enumerate(image_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                "players snapshot image row "
                f"{row_number} không phải object."
            )

        normalized_existing_rows.append(
            {
                key: "" if value is None else str(value)
                for key, value in row.items()
                if isinstance(key, str)
            }
        )

    if reuse_query_snapshot:
        if query_snapshot_path is None:
            raise ValueError(
                "reuse_query_snapshot yêu cầu query_snapshot_path."
            )

        try:
            query_payload = json.loads(
                query_snapshot_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Player image sort snapshot không phải JSON hợp lệ."
            ) from error

        query_source = (
            query_payload.get("source")
            if isinstance(query_payload, dict)
            else None
        )
        query_rows = (
            query_payload.get("rows")
            if isinstance(query_payload, dict)
            else None
        )
        query_row_count = (
            query_payload.get("row_count")
            if isinstance(query_payload, dict)
            else None
        )

        if (
            not isinstance(query_source, dict)
            or query_source.get("table")
            != _LEAGUEPEDIA_PLAYER_IMAGE_TABLES
            or query_source.get("join_on")
            != _LEAGUEPEDIA_PLAYER_IMAGE_JOIN
            or not isinstance(query_rows, list)
            or not isinstance(query_row_count, int)
            or isinstance(query_row_count, bool)
            or query_row_count != len(query_rows)
        ):
            raise ValueError(
                "Player image sort snapshot không đúng truy vấn/row_count."
            )

        fetched_rows = []

        for row_number, row in enumerate(query_rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    "Player image sort row "
                    f"{row_number} không phải object."
                )

            fetched_rows.append(
                {
                    key: "" if value is None else str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
            )
    else:
        fetched_rows = fetch_player_image_rows(
            snapshot_path=query_snapshot_path
        )

    fetched_by_key: dict[
        tuple[str, str], dict[str, str]
    ] = {}
    duplicate_source_rows_collapsed = 0

    for row in fetched_rows:
        row_key = _player_image_row_key(row)

        if not all(row_key):
            continue

        existing_row = fetched_by_key.get(row_key)

        if existing_row is not None:
            duplicate_source_rows_collapsed += 1
            fetched_by_key[row_key] = max(
                (existing_row, row),
                key=_profile_image_sort_key,
            )
        else:
            fetched_by_key[row_key] = row

    enriched_fields = (
        "Tournament",
        "SortDate",
        "SortDate__precision",
        "TournamentDateStartFuzzy",
        "TournamentDateStartFuzzy__precision",
        "TournamentDate",
        "TournamentDate__precision",
        "EffectiveSortDate",
    )
    enriched_rows: list[dict[str, str]] = []
    matched_keys: set[tuple[str, str]] = set()

    for row in normalized_existing_rows:
        row_key = _player_image_row_key(row)
        fetched_row = fetched_by_key.get(row_key)

        if fetched_row is None:
            enriched_rows.append(row)
            continue

        matched_keys.add(row_key)
        enriched_row = dict(row)

        for field_name in enriched_fields:
            enriched_row[field_name] = (
                fetched_row.get(field_name) or ""
            )

        enriched_rows.append(enriched_row)

    source = payload.get("source")

    if not isinstance(source, dict):
        raise ValueError(
            "players snapshot thiếu source object."
        )

    refreshed_at_utc = datetime.now(
        timezone.utc
    ).isoformat()
    source["media_sort_policy"] = (
        "COALESCE(PlayerImages.SortDate,"
        "Tournaments.DateStartFuzzy,Tournaments.Date)"
    )
    source["media_sort_enrichment"] = {
        "fetched_at_utc": refreshed_at_utc,
        "table": _LEAGUEPEDIA_PLAYER_IMAGE_TABLES,
        "join_on": _LEAGUEPEDIA_PLAYER_IMAGE_JOIN,
        "row_count": len(fetched_rows),
        "unique_player_file_count": len(fetched_by_key),
        "duplicate_source_rows_collapsed": (
            duplicate_source_rows_collapsed
        ),
        "matched_image_rows": len(matched_keys),
        "unmatched_snapshot_rows": (
            len(enriched_rows) - len(matched_keys)
        ),
        "new_source_rows": (
            len(fetched_by_key) - len(matched_keys)
        ),
        "query_snapshot": (
            str(query_snapshot_path)
            if query_snapshot_path is not None
            else None
        ),
    }
    payload["source"] = source
    payload["image_rows"] = enriched_rows
    temporary = path.with_suffix(f"{path.suffix}.part")

    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

    return PlayerImageSortRefreshStats(
        fetched_rows=len(fetched_rows),
        matched_rows=len(matched_keys),
        unmatched_snapshot_rows=(
            len(enriched_rows) - len(matched_keys)
        ),
        new_source_rows=(
            len(fetched_by_key) - len(matched_keys)
        ),
        duplicate_source_rows_collapsed=(
            duplicate_source_rows_collapsed
        ),
    )


def load_players_snapshot(
    path: Path,
) -> PlayerSelectionResult:
    """Đọc bundle browser và tái hiện đúng bước chọn tuyển thủ."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "players.json khong phai JSON hop le."
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            "players.json phai chua mot JSON object."
        )

    schema_version = payload.get("schema_version")

    if schema_version != _PLAYER_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "players snapshot phải dùng schema_version=2; "
            "snapshot Players cũ chứa cả personality/staff và "
            "không được phép đồng bộ."
        )

    source = payload.get("source")

    expected_evidence_tables = {
        "PlayerLeagueHistory",
        "TournamentPlayers",
        "ListplayerCurrent",
    }
    source_evidence_tables = (
        source.get("evidence_tables")
        if isinstance(source, dict)
        else None
    )

    if (
        not isinstance(source, dict)
        or source.get("catalog_table") != "Players"
        or not isinstance(source_evidence_tables, list)
        or any(
            not isinstance(table_name, str)
            for table_name in source_evidence_tables
        )
        or len(source_evidence_tables)
        != len(expected_evidence_tables)
        or set(source_evidence_tables)
        != expected_evidence_tables
        or source.get("media_table") != "PlayerImages"
    ):
        raise ValueError(
            "players snapshot không khai báo đủ catalog, "
            "bằng chứng tuyển thủ và nguồn ảnh hợp lệ."
        )

    def normalized_rows(
        field_name: str,
        count_field_name: str,
    ) -> list[dict[str, str]]:
        rows = payload.get(field_name)
        declared_count = payload.get(count_field_name)

        if not isinstance(rows, list):
            raise ValueError(
                f"players snapshot thiếu {field_name}."
            )

        if (
            not isinstance(declared_count, int)
            or isinstance(declared_count, bool)
            or declared_count != len(rows)
        ):
            raise ValueError(
                "players snapshot có "
                f"{count_field_name} không khớp."
            )

        result: list[dict[str, str]] = []

        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"{field_name} row {row_number} "
                    "không phải object."
                )

            result.append(
                {
                    key: "" if value is None else str(value)
                    for key, value in row.items()
                    if isinstance(key, str)
                }
            )

        return result

    media_urls_are_direct = payload.get(
        "media_urls_are_direct"
    )

    if not isinstance(media_urls_are_direct, bool):
        raise ValueError(
            "players snapshot thiếu media_urls_are_direct."
        )

    return _records_from_leaguepedia_rows(
        catalog_rows=normalized_rows(
            "catalog_rows",
            "catalog_row_count",
        ),
        game_evidence_rows=normalized_rows(
            "game_evidence_rows",
            "game_evidence_row_count",
        ),
        roster_rows=normalized_rows(
            "roster_rows",
            "roster_row_count",
        ),
        image_rows=normalized_rows(
            "image_rows",
            "image_row_count",
        ),
        direct_media_urls=media_urls_are_direct,
    )


def _read_cell(
    row: dict[str, str | None],
    column: str,
) -> str:
    """Đọc một CSV cell và loại bỏ khoảng trắng biên."""

    return (row.get(column) or "").strip()


def load_players_csv(
    path: Path,
) -> list[PlayerRecord]:
    """Đọc và kiểm tra cấu trúc players.csv."""

    records: list[PlayerRecord] = []
    seen_player_ids: set[str] = set()

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
                "players.csv thiếu cột: "
                f"{sorted(missing_columns)}"
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            player_id = _read_cell(
                row,
                "player_id",
            )
            canonical_name = _read_cell(
                row,
                "canonical_name",
            )
            display_name = _read_cell(
                row,
                "display_name",
            )
            record_source = _read_cell(
                row,
                "record_source",
            )
            record_source_url = _read_cell(
                row,
                "record_source_url",
            )
            photo_url = (
                _read_cell(row, "photo_url") or None
            )

            required_values = {
                "player_id": player_id,
                "canonical_name": canonical_name,
                "display_name": display_name,
                "record_source": record_source,
                "record_source_url": record_source_url,
            }
            missing_values = [
                column
                for column, value
                in required_values.items()
                if not value
            ]

            if missing_values:
                raise ValueError(
                    f"players.csv dòng {row_number} "
                    "thiếu giá trị: "
                    f"{missing_values}"
                )

            if player_id in seen_player_ids:
                raise ValueError(
                    "players.csv chứa player_id trùng: "
                    f"{player_id}"
                )

            seen_player_ids.add(player_id)

            records.append(
                PlayerRecord(
                    player_id=player_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    record_source=record_source,
                    record_source_url=(
                        record_source_url
                    ),
                    photo_url=photo_url,
                )
            )

    return records


def write_players_csv(
    path: Path,
    records: list[PlayerRecord],
) -> None:
    """Ghi players.csv staging bằng thao tác nguyên tử."""

    if not records:
        raise ValueError(
            "Không ghi players.csv khi danh sách "
            "player rỗng."
        )

    player_ids = [
        record.player_id for record in records
    ]

    if len(player_ids) != len(set(player_ids)):
        raise ValueError(
            "Không ghi players.csv vì có "
            "player_id trùng."
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
                        "player_id": record.player_id,
                        "canonical_name": (
                            record.canonical_name
                        ),
                        "display_name": (
                            record.display_name
                        ),
                        "record_source": (
                            record.record_source
                        ),
                        "record_source_url": (
                            record.record_source_url
                        ),
                        "photo_url": (
                            record.photo_url or ""
                        ),
                    }
                )

        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sync_player_metadata(
    session: Session,
    records: list[PlayerRecord],
) -> SyncStats:
    """Upsert metadata player, không thực hiện I/O media."""

    stats = SyncStats()
    player_ids = [
        record.player_id for record in records
    ]

    if len(player_ids) != len(set(player_ids)):
        raise ValueError(
            "Danh sách player chứa player_id trùng."
        )

    for record in records:
        player_id = validate_stable_id(
            record.player_id,
            field_name="player_id",
            maximum=64,
        )
        canonical_name = bounded(
            clean_label(
                record.canonical_name,
                field_name="canonical_name",
            ),
            field_name="canonical_name",
            maximum=120,
        )
        display_name = bounded(
            clean_label(
                record.display_name,
                field_name="display_name",
            ),
            field_name="display_name",
            maximum=120,
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

        player = session.get(
            Player,
            player_id,
        )

        if player is None:
            session.add(
                Player(
                    player_id=player_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    photo_file=None,
                )
            )
            stats.inserted += 1
            continue

        desired_values = {
            "canonical_name": canonical_name,
            "display_name": display_name,
        }
        changed = any(
            getattr(player, field_name) != value
            for field_name, value
            in desired_values.items()
        )

        if not changed:
            stats.skipped += 1
            continue

        for field_name, value in (
            desired_values.items()
        ):
            setattr(
                player,
                field_name,
                value,
            )

        stats.updated += 1

    session.flush()
    return stats


def sync_player_media(
    session: Session,
    records: list[PlayerRecord],
    project_root: Path,
    refresh_media: bool = False,
) -> PlayerMediaSyncResult:
    """Tải một batch ảnh rồi liên kết đường dẫn local vào player."""

    stats = PlayerMediaSyncResult()
    asset_dir = project_root / "assets" / "players"
    excluded_ids = load_media_exclusions(project_root, "players")
    player_ids = [
        record.player_id for record in records
    ]

    if len(player_ids) != len(set(player_ids)):
        raise ValueError(
            "Batch media chứa player_id trùng."
        )

    photo_files: dict[str, str] = {}
    failed_player_ids: list[str] = []

    # Không truy cập session trong vòng lặp này. HTTP chậm hoặc Ctrl+C vì
    # vậy không giữ một transaction PostgreSQL đang mở.
    for record in records:
        if record.photo_url is None:
            continue
        if record.player_id in excluded_ids:
            stats.media_skipped += 1
            continue

        player_id = validate_stable_id(
            record.player_id,
            field_name="player_id",
            maximum=64,
        )
        canonical_name = bounded(
            clean_label(
                record.canonical_name,
                field_name="canonical_name",
            ),
            field_name="canonical_name",
            maximum=120,
        )
        canonical_key(canonical_name)

        try:
            photo_file, downloaded = download_image(
                url=record.photo_url,
                asset_dir=asset_dir,
                stem=asset_stem(
                    stable_id=player_id,
                    canonical_name=canonical_name,
                ),
                project_root=project_root,
                refresh=refresh_media,
            )
        except (RuntimeError, ValueError) as error:
            _LOGGER.warning(
                "Khong tai duoc anh cua %s; "
                "giu photo_file hien tai va tiep tuc: %s",
                player_id,
                error,
            )
            stats.media_skipped += 1
            failed_player_ids.append(player_id)
            continue

        photo_files[player_id] = bounded(
            photo_file,
            field_name="photo_file",
            maximum=160,
        )

        if downloaded:
            stats.media_downloaded += 1
        else:
            stats.media_skipped += 1

    for player_id, photo_file in photo_files.items():
        player = session.get(Player, player_id)

        if player is None:
            raise RuntimeError(
                "Không thể liên kết media trước khi đồng bộ "
                f"metadata player: {player_id}."
            )

        if player.photo_file != photo_file:
            player.photo_file = photo_file

    session.flush()
    stats.succeeded_player_ids = tuple(
        photo_files
    )
    stats.failed_player_ids = tuple(
        failed_player_ids
    )
    return stats


def prune_missing_leaguepedia_players(
    session: Session,
    keep_player_ids: set[str],
) -> PlayerPruneResult:
    """Xóa player Leaguepedia ngoài tập mới nếu chưa được tham chiếu."""

    invalid_keep_ids = {
        player_id
        for player_id in keep_player_ids
        if not player_id.startswith("lp_player_")
    }

    if invalid_keep_ids:
        raise ValueError(
            "Tập keep_player_ids chứa ID không thuộc "
            "Leaguepedia."
        )

    leaguepedia_id_filter = Player.player_id.like(
        r"lp\_player\_%",
        escape="\\",
    )
    current_ids = set(
        session.scalars(
            select(Player.player_id).where(
                leaguepedia_id_filter
            )
        )
    )
    stale_ids = current_ids - keep_player_ids

    if not stale_ids:
        return PlayerPruneResult(
            deleted=0,
            protected=0,
            photo_files=(),
        )

    ordered_stale_ids = tuple(sorted(stale_ids))
    referenced_ids = set(
        session.scalars(
            select(TeamMembership.player_id).where(
                TeamMembership.player_id.in_(
                    ordered_stale_ids
                )
            )
        )
    )
    referenced_ids.update(
        session.scalars(
            select(GamePlayer.player_id).where(
                GamePlayer.player_id.in_(
                    ordered_stale_ids
                )
            )
        )
    )
    removable_ids = stale_ids - referenced_ids
    photo_files: list[str] = []

    if removable_ids:
        ordered_removable_ids = tuple(
            sorted(removable_ids)
        )
        players = session.scalars(
            select(Player).where(
                Player.player_id.in_(
                    ordered_removable_ids
                )
            )
        )

        for player in players:
            if player.photo_file:
                photo_files.append(player.photo_file)

            session.delete(player)

    session.flush()

    return PlayerPruneResult(
        deleted=len(removable_ids),
        protected=len(referenced_ids),
        photo_files=tuple(sorted(set(photo_files))),
    )


def delete_pruned_player_media(
    project_root: Path,
    relative_paths: tuple[str, ...],
) -> int:
    """Xóa đúng các ảnh của player đã prune, sau khi DB commit."""

    resolved_root = project_root.resolve()
    asset_root = (
        resolved_root / "assets" / "players"
    ).resolve()
    targets: list[Path] = []

    for relative_path in relative_paths:
        if Path(relative_path).is_absolute():
            raise ValueError(
                "Không xóa media bằng đường dẫn tuyệt đối: "
                f"{relative_path}"
            )

        target = (resolved_root / relative_path).resolve()

        try:
            target.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(
                "Không xóa media nằm ngoài assets/players: "
                f"{relative_path}"
            ) from error

        targets.append(target)

    deleted = 0

    for target in targets:
        if target.is_file():
            target.unlink()
            deleted += 1

    return deleted
