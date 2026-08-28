"""Đọc dữ liệu reference từ Leaguepedia Cargo bằng mwrogue."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

from match_insight.config import load_local_environment

LEAGUEPEDIA_API_URL = "https://lol.fandom.com/api.php"
LEAGUEPEDIA_CLIENT_NAME = "mwrogue"
LEAGUEPEDIA_USER_AGENT_PRODUCT = (
    "MatchInsightReferenceSync/1.0"
)
CARGO_PAGE_SIZE = 500

_MAX_ATTEMPTS = 4
_IMAGEINFO_BATCH_SIZE = 50
_CARGO_PAGE_INTERVAL_SECONDS = 20
_IMAGEINFO_BATCH_INTERVAL_SECONDS = 2
_GENERIC_RETRY_DELAYS_SECONDS = (1, 2, 4)
_RATE_LIMIT_RETRY_DELAYS_SECONDS = (30, 60, 120)
_USERNAME_ENV = "WIKI_USERNAME_MATCH_INSIGHT"
_PASSWORD_ENV = "WIKI_PASSWORD_MATCH_INSIGHT"

_LOGGER = logging.getLogger(__name__)


class CargoQueryClient(Protocol):
    """Hợp đồng tối thiểu của mwrogue CargoClient."""

    def query(
        self,
        *,
        tables: str,
        fields: str,
        join_on: str | None = None,
        where: str | None = None,
        group_by: str | None = None,
        order_by: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        auto_continue: bool = True,
    ) -> object:
        """Thực hiện một Cargo query."""


class MediaWikiApiClient(Protocol):
    """Hợp đồng tối thiểu của mwclient Site cho MediaWiki API."""

    def api(
        self,
        action: str,
        http_method: str = "POST",
        **kwargs: object,
    ) -> object:
        """Thực hiện một MediaWiki API request."""


class LeaguepediaSite(Protocol):
    """Các client cần dùng từ một phiên mwrogue đã xác thực."""

    cargo_client: CargoQueryClient
    client: MediaWikiApiClient


def _load_credentials() -> tuple[str, str]:
    """Đọc Fandom bot credentials mà không ghi ra log."""
    load_local_environment()
    username = (os.getenv(_USERNAME_ENV) or "").strip()
    password = (os.getenv(_PASSWORD_ENV) or "").strip()

    missing = [
        variable
        for variable, value in (
            (_USERNAME_ENV, username),
            (_PASSWORD_ENV, password),
        )
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Thiếu Leaguepedia credential: "
            f"{', '.join(missing)}."
        )

    if "@" not in username:
        raise ValueError(
            f"{_USERNAME_ENV} phải có dạng "
            "AccountName@BotName."
        )

    return username, password


def _create_leaguepedia_site() -> LeaguepediaSite:
    """Tạo phiên Leaguepedia đã xác thực bằng mwrogue."""
    try:
        from mwrogue.auth_credentials import AuthCredentials
        from mwrogue.esports_client import EsportsClient
    except ImportError as error:
        raise RuntimeError(
            "Chưa cài dependency mwrogue. Hãy cài lại "
            "requirements-dev.txt."
        ) from error

    username, password = _load_credentials()
    credentials = AuthCredentials(
        username=username,
        password=password,
    )

    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            site = EsportsClient(
                "lol",
                credentials=credentials,
                user_agent=(
                    f"{LEAGUEPEDIA_USER_AGENT_PRODUCT} "
                    f"({username})"
                ),
            )
        except Exception as error:
            last_error = error

            if _http_status_code(error) == 403:
                raise RuntimeError(
                    "Leaguepedia chặn Python bằng HTTP 403; "
                    "hãy dùng browser snapshot của script sync "
                    "tương ứng."
                ) from error

            if (
                not _is_rate_limited(error)
                or attempt == _MAX_ATTEMPTS
            ):
                raise RuntimeError(
                    "Không khởi tạo được phiên Leaguepedia "
                    "đã xác thực."
                ) from error

            retry_delay = _retry_delay_seconds(
                error=error,
                attempt=attempt,
            )
            _LOGGER.warning(
                "Leaguepedia từ chối khởi tạo phiên với "
                "HTTP %s; thử lại sau %s giây.",
                _http_status_code(error),
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            return cast(LeaguepediaSite, site)

    raise RuntimeError(
        "Không khởi tạo được phiên Leaguepedia "
        "đã xác thực."
    ) from last_error


def _create_cargo_client() -> CargoQueryClient:
    """Lấy Cargo client từ phiên Leaguepedia đã xác thực."""
    return _create_leaguepedia_site().cargo_client


def _normalize_cargo_rows(
    payload: object,
) -> list[dict[str, str]]:
    """Chuẩn hóa Cargo rows thành mapping chuỗi ổn định."""
    if not isinstance(payload, list):
        raise ValueError(
            "mwrogue Cargo query phải trả về một list."
        )

    rows: list[dict[str, str]] = []

    for item_number, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Cargo item {item_number} không phải object."
            )

        row: dict[str, str] = {}

        for field_name, value in item.items():
            if not isinstance(field_name, str):
                raise ValueError(
                    "Cargo field name phải là chuỗi."
                )

            row[field_name] = (
                "" if value is None else str(value)
            )

        rows.append(row)

    return rows


def _http_status_code(
    error: Exception,
) -> int | None:
    """Đọc HTTP status từ exception mà không phụ thuộc requests."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)

    return (
        status_code
        if isinstance(status_code, int)
        else None
    )


def _is_rate_limited(error: Exception) -> bool:
    """Nhận diện rate limit từ API code hoặc HTTP status."""
    return (
        getattr(error, "code", None) == "ratelimited"
        or _http_status_code(error) == 429
    )


def _retry_delay_seconds(
    error: Exception,
    attempt: int,
) -> int:
    """Chọn thời gian chờ theo loại lỗi và lần thử vừa thất bại."""
    delays = (
        _RATE_LIMIT_RETRY_DELAYS_SECONDS
        if _is_rate_limited(error)
        else _GENERIC_RETRY_DELAYS_SECONDS
    )
    return delays[attempt - 1]


def _file_name_key(value: str) -> str:
    """Chuẩn hóa tên file MediaWiki để đối chiếu kết quả imageinfo."""
    cleaned = value.strip()

    if cleaned.casefold().startswith("file:"):
        cleaned = cleaned[5:]

    return " ".join(
        cleaned.replace("_", " ").split()
    ).casefold()


def _normalize_imageinfo_urls(
    payload: object,
) -> dict[str, str]:
    """Đọc mapping tên file sang URL CDN từ MediaWiki imageinfo."""
    if not isinstance(payload, dict):
        raise ValueError(
            "MediaWiki imageinfo phải trả về một object."
        )

    query = payload.get("query")

    if not isinstance(query, dict):
        raise ValueError(
            "MediaWiki imageinfo thiếu query object."
        )

    pages = query.get("pages")

    if not isinstance(pages, list):
        raise ValueError(
            "MediaWiki imageinfo pages phải là một list."
        )

    resolved: dict[str, str] = {}
    aliases: dict[str, str] = {}

    for alias_group in ("normalized", "redirects"):
        alias_rows = query.get(alias_group, [])

        if not isinstance(alias_rows, list):
            continue

        for alias_row in alias_rows:
            if not isinstance(alias_row, dict):
                continue

            source_name = alias_row.get("from")
            target_name = alias_row.get("to")

            if isinstance(source_name, str) and isinstance(
                target_name, str
            ):
                aliases[_file_name_key(source_name)] = (
                    _file_name_key(target_name)
                )

    for page in pages:
        if not isinstance(page, dict) or "missing" in page:
            continue

        title = page.get("title")
        imageinfo = page.get("imageinfo")

        if (
            not isinstance(title, str)
            or not isinstance(imageinfo, list)
            or not imageinfo
            or not isinstance(imageinfo[0], dict)
        ):
            continue

        direct_url = imageinfo[0].get("url")

        if not isinstance(direct_url, str):
            continue

        parsed_url = urlparse(direct_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
        ):
            continue

        resolved[_file_name_key(title)] = direct_url

    for _ in range(2):
        for source_key, target_key in aliases.items():
            direct_url = resolved.get(target_key)

            if direct_url is not None:
                resolved[source_key] = direct_url

    return resolved


def _query_imageinfo_batch(
    api_client: MediaWikiApiClient,
    file_names: tuple[str, ...],
) -> dict[str, str]:
    """Tra URL CDN cho một batch tên file với retry có giới hạn."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            payload = api_client.api(
                "query",
                prop="imageinfo",
                iiprop="url",
                titles="|".join(
                    f"File:{name}" for name in file_names
                ),
                redirects=True,
                formatversion=2,
            )
        except Exception as error:
            last_error = error

            if _http_status_code(error) == 403:
                raise RuntimeError(
                    "Leaguepedia chặn Python imageinfo bằng "
                    "HTTP 403; hãy dùng browser snapshot có URL "
                    "media trực tiếp."
                ) from error

            if attempt == _MAX_ATTEMPTS:
                raise RuntimeError(
                    "Không tra được Leaguepedia imageinfo "
                    f"sau {_MAX_ATTEMPTS} lần."
                ) from error

            retry_delay = _retry_delay_seconds(
                error=error,
                attempt=attempt,
            )
            _LOGGER.warning(
                "Leaguepedia imageinfo lỗi %s; thử lại sau %s giây.",
                (
                    "ratelimited"
                    if _is_rate_limited(error)
                    else type(error).__name__
                ),
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            return _normalize_imageinfo_urls(payload)

    raise RuntimeError(
        "Không tra được Leaguepedia imageinfo."
    ) from last_error


def fetch_file_urls(
    file_names: tuple[str, ...],
    batch_size: int = _IMAGEINFO_BATCH_SIZE,
    api_client: MediaWikiApiClient | None = None,
) -> dict[str, str]:
    """Tra URL CDN trực tiếp cho các tên file Leaguepedia theo batch."""
    if not 1 <= batch_size <= _IMAGEINFO_BATCH_SIZE:
        raise ValueError(
            f"batch_size phải nằm trong khoảng "
            f"1..{_IMAGEINFO_BATCH_SIZE}."
        )

    cleaned_names = tuple(
        dict.fromkeys(
            name.strip()
            for name in file_names
            if name.strip()
        )
    )

    if not cleaned_names:
        return {}

    client = (
        api_client
        if api_client is not None
        else _create_leaguepedia_site().client
    )
    resolved_by_key: dict[str, str] = {}

    for start in range(0, len(cleaned_names), batch_size):
        batch = cleaned_names[start : start + batch_size]
        resolved_by_key.update(
            _query_imageinfo_batch(
                api_client=client,
                file_names=batch,
            )
        )

        if start + batch_size < len(cleaned_names):
            time.sleep(
                _IMAGEINFO_BATCH_INTERVAL_SECONDS
            )

    return {
        original_name: resolved_by_key[key]
        for original_name in cleaned_names
        if (key := _file_name_key(original_name))
        in resolved_by_key
    }


def _query_cargo_page(
    cargo_client: CargoQueryClient,
    table: str,
    fields: tuple[str, ...],
    order_by: str,
    join_on: str | None,
    where: str | None,
    group_by: str | None,
    page_size: int,
    offset: int,
) -> list[dict[str, str]]:
    """Tải một Cargo page và retry lỗi kết nối giới hạn."""
    last_error: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            query_arguments: dict[str, object] = {
                "tables": table,
                "fields": ", ".join(fields),
                "where": where,
                "order_by": order_by,
                "limit": page_size,
                "offset": offset,
                "auto_continue": False,
            }

            if join_on is not None:
                query_arguments["join_on"] = join_on

            if group_by is not None:
                query_arguments["group_by"] = group_by

            payload = cargo_client.query(
                **query_arguments,
            )
        except Exception as error:
            last_error = error

            if _http_status_code(error) == 403:
                raise RuntimeError(
                    "Leaguepedia chặn Python Cargo bằng HTTP 403; "
                    "hãy dùng browser snapshot của script sync "
                    "tương ứng."
                ) from error

            if attempt == _MAX_ATTEMPTS:
                failure_reason = (
                    "vẫn bị giới hạn tần suất"
                    if _is_rate_limited(error)
                    else "không gọi được"
                )
                raise RuntimeError(
                    f"Leaguepedia Cargo {failure_reason} tại "
                    f"offset={offset} sau {_MAX_ATTEMPTS} lần."
                ) from error

            retry_delay = _retry_delay_seconds(
                error=error,
                attempt=attempt,
            )
            error_kind = (
                "ratelimited"
                if _is_rate_limited(error)
                else type(error).__name__
            )
            _LOGGER.warning(
                "Leaguepedia Cargo lỗi %s tại offset=%s; "
                "thử lại sau %s giây.",
                error_kind,
                offset,
                retry_delay,
            )
            time.sleep(retry_delay)
        else:
            return _normalize_cargo_rows(payload)

    raise RuntimeError(
        "Không gọi được Leaguepedia Cargo."
    ) from last_error


def write_cargo_snapshot(
    path: Path,
    table: str,
    fields: tuple[str, ...],
    order_by: str,
    where: str | None,
    page_size: int,
    rows: list[dict[str, str]],
    group_by: str | None = None,
    join_on: str | None = None,
) -> None:
    """Ghi raw Cargo rows và metadata truy vết."""
    if not rows:
        raise ValueError(
            "Không ghi Cargo snapshot khi rows rỗng."
        )

    snapshot: dict[str, object] = {
        "fetched_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source": {
            "type": "leaguepedia_cargo",
            "api_url": LEAGUEPEDIA_API_URL,
            "client": LEAGUEPEDIA_CLIENT_NAME,
            "table": table,
            "fields": list(fields),
            "order_by": order_by,
            "where": where,
            "group_by": group_by,
            "join_on": join_on,
            "page_size": page_size,
        },
        "row_count": len(rows),
        "rows": rows,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(
        f"{path.suffix}.part"
    )
    content = json.dumps(
        snapshot,
        ensure_ascii=False,
        indent=2,
    )

    try:
        temporary.write_text(
            content,
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_cargo_rows(
    table: str,
    fields: tuple[str, ...],
    order_by: str,
    where: str | None = None,
    group_by: str | None = None,
    join_on: str | None = None,
    page_size: int = CARGO_PAGE_SIZE,
    snapshot_path: Path | None = None,
    cargo_client: CargoQueryClient | None = None,
) -> list[dict[str, str]]:
    """Tải đầy đủ Cargo table bằng limit/offset."""
    if not table.strip():
        raise ValueError("Cargo table không được để trống.")

    if not fields:
        raise ValueError("Cargo fields không được để trống.")

    if not order_by.strip():
        raise ValueError(
            "Cargo order_by không được để trống."
        )

    if not 1 <= page_size <= CARGO_PAGE_SIZE:
        raise ValueError(
            f"page_size phải nằm trong khoảng "
            f"1..{CARGO_PAGE_SIZE}."
        )

    client = (
        cargo_client
        if cargo_client is not None
        else _create_cargo_client()
    )
    rows: list[dict[str, str]] = []
    offset = 0

    while True:
        page_rows = _query_cargo_page(
            cargo_client=client,
            table=table,
            fields=fields,
            order_by=order_by,
            join_on=join_on,
            where=where,
            group_by=group_by,
            page_size=page_size,
            offset=offset,
        )
        rows.extend(page_rows)

        if len(page_rows) < page_size:
            break

        offset += page_size
        _LOGGER.info(
            "Đã tải %s Cargo rows; nghỉ %s giây trước trang kế tiếp.",
            len(rows),
            _CARGO_PAGE_INTERVAL_SECONDS,
        )
        time.sleep(_CARGO_PAGE_INTERVAL_SECONDS)

    if snapshot_path is not None:
        write_cargo_snapshot(
            path=snapshot_path,
            table=table,
            fields=fields,
            order_by=order_by,
            where=where,
            page_size=page_size,
            rows=rows,
            group_by=group_by,
            join_on=join_on,
        )

    return rows
