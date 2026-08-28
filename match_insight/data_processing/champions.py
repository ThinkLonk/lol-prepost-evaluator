"""Đồng bộ danh mục tướng và square assets từ Riot Data Dragon."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from sqlalchemy.orm import Session

from match_insight.data_processing.reference_common import (
    SyncStats,
    asset_stem,
    bounded,
    canonical_key,
    clean_label,
    download_image,
    validate_stable_id,
)
from match_insight.database.models import Champion

_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
_LOCALE_PATTERN = re.compile(r"[a-z]{2}_[A-Z]{2}")


@dataclass(frozen=True)
class ChampionRecord:
    """Một champion record đã đọc từ Data Dragon."""

    champion_id: str
    canonical_name: str
    display_name: str
    image_url: str


def fetch_champions(
    version: str,
    locale: str = "en_US",
) -> tuple[list[ChampionRecord], str]:
    """Đọc champion metadata từ một Data Dragon version cố định."""
    if _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(
            "Data Dragon version phải có dạng ba phần, ví dụ 16.16.1."
        )

    if _LOCALE_PATTERN.fullmatch(locale) is None:
        raise ValueError(
            "Data Dragon locale phải có dạng như en_US hoặc vi_VN."
        )

    source_url = (
        "https://ddragon.leagueoflegends.com/"
        f"cdn/{version}/data/{locale}/champion.json"
    )

    request = Request(
        source_url,
        headers={
            "User-Agent": "match-insight-reference-sync/1.0",
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.load(response)

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "Data Dragon không trả về trường data hợp lệ."
        )

    records: list[ChampionRecord] = []

    for raw_record in data.values():
        if not isinstance(raw_record, dict):
            raise ValueError(
                "Data Dragon chứa champion record không hợp lệ."
            )

        image = raw_record.get("image")

        if not isinstance(image, dict):
            raise ValueError(
                f"Tướng {raw_record.get('id')} không có image hợp lệ."
            )

        image_file = image.get("full")

        if not isinstance(image_file, str) or not image_file:
            raise ValueError(
                f"Tướng {raw_record.get('id')} không có image.full."
            )

        champion_id = str(raw_record["key"])
        canonical_name = str(raw_record["id"])
        display_name = str(raw_record["name"])
        image_url = (
            "https://ddragon.leagueoflegends.com/"
            f"cdn/{version}/img/champion/{image_file}"
        )

        records.append(
            ChampionRecord(
                champion_id=champion_id,
                canonical_name=canonical_name,
                display_name=display_name,
                image_url=image_url,
            )
        )

    try:
        records.sort(key=lambda record: int(record.champion_id))
    except ValueError as error:
        raise ValueError(
            "Data Dragon champion key phải là số nguyên."
        ) from error

    return records, source_url


def sync_champions(
    session: Session,
    records: list[ChampionRecord],
    project_root: Path,
    refresh_media: bool = False,
) -> SyncStats:
    """Tải ảnh và upsert champion theo champion_id."""
    stats = SyncStats()
    asset_dir = project_root / "assets" / "champions"

    champion_ids = [record.champion_id for record in records]

    if len(champion_ids) != len(set(champion_ids)):
        raise ValueError(
            "Nguồn champion chứa champion_id trùng."
        )

    for record in records:
        champion_id = validate_stable_id(
            record.champion_id,
            field_name="champion_id",
            maximum=30,
        )
        canonical_name = bounded(
            clean_label(
                record.canonical_name,
                field_name="canonical_name",
            ),
            field_name="canonical_name",
            maximum=80,
        )
        display_name = bounded(
            clean_label(
                record.display_name,
                field_name="display_name",
            ),
            field_name="display_name",
            maximum=80,
        )

        canonical_key(canonical_name)

        image_file, downloaded = download_image(
            url=record.image_url,
            asset_dir=asset_dir,
            stem=asset_stem(
                stable_id=champion_id,
                canonical_name=canonical_name,
            ),
            project_root=project_root,
            refresh=refresh_media,
        )

        if downloaded:
            stats.media_downloaded += 1
        else:
            stats.media_skipped += 1

        champion = session.get(Champion, champion_id)

        if champion is None:
            session.add(
                Champion(
                    champion_id=champion_id,
                    canonical_name=canonical_name,
                    display_name=display_name,
                    image_file=image_file,
                )
            )
            stats.inserted += 1
            continue

        desired_values = {
            "canonical_name": canonical_name,
            "display_name": display_name,
            "image_file": image_file,
        }
        changed = any(
            getattr(champion, field_name) != value
            for field_name, value in desired_values.items()
        )

        if not changed:
            stats.skipped += 1
            continue

        for field_name, value in desired_values.items():
            setattr(champion, field_name, value)

        stats.updated += 1

    session.flush()
    return stats