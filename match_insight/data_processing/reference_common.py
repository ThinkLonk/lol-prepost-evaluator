"""Tiện ích chung cho dữ liệu tham chiếu và media local."""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_STABLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_DOWNLOAD_ATTEMPTS = 4
_DOWNLOAD_TIMEOUT_SECONDS = 45
_FANDOM_MEDIA_INTERVAL_SECONDS = 1
_FANDOM_RATE_LIMIT_DELAYS_SECONDS = (30, 60, 120)
_RETRYABLE_HTTP_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}

_LOGGER = logging.getLogger(__name__)


@dataclass
class SyncStats:
    """Bộ đếm kết quả của một lần đồng bộ."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    media_downloaded: int = 0
    media_skipped: int = 0


def clean_label(value: str, field_name: str) -> str:
    """Chuẩn hóa Unicode và khoảng trắng nhưng giữ cách viết hiển thị."""
    cleaned = " ".join(unicodedata.normalize("NFKC", value).split())

    if not cleaned:
        raise ValueError(f"{field_name} không được để trống.")

    return cleaned


def canonical_key(value: str) -> str:
    """Tạo khóa so sánh tên, không dùng thay cho entity ID."""
    return clean_label(value, "canonical_name").casefold()


def bounded(value: str, field_name: str, maximum: int) -> str:
    """Kiểm tra độ dài tương thích với cột PostgreSQL."""

    if len(value) > maximum:
        raise ValueError(f"{field_name} dài {len(value)} ký tự, vượt giới hạn {maximum}.")

    return value


def validate_stable_id(
    value: str,
    field_name: str,
    maximum: int,
) -> str:
    """Kiểm tra cú pháp và độ dài stable ID; chưa mã hóa thành tên file."""
    cleaned = clean_label(value, field_name)

    if len(cleaned) > maximum:
        raise ValueError(f"{field_name} dài {len(cleaned)} ký tự, vượt giới hạn {maximum}.")

    if _STABLE_ID_PATTERN.fullmatch(cleaned) is None:
        raise ValueError(
            f"{field_name} chỉ được chứa chữ, số, dấu chấm, gạch dưới, hai chấm hoặc gạch ngang."
        )

    return cleaned


def _filename_slug(value: str) -> str:
    """Chuyển canonical name thành thành phần tên file an toàn."""
    canonical = canonical_key(value).replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", canonical)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")

    if slug:
        return slug[:70]

    return sha256(value.encode("utf-8")).hexdigest()[:12]


def asset_stem(stable_id: str, canonical_name: str) -> str:
    """Tạo tên media, phân biệt dấu phân cách namespace với gạch dưới."""
    safe_id = validate_stable_id(
        stable_id,
        field_name="stable_id",
        maximum=100,
    )
    encoded_id = safe_id.replace(":", "~")
    return f"{encoded_id}-{_filename_slug(canonical_name)}"


def _detect_image_extension(payload: bytes) -> str:
    """Xác định định dạng bằng magic bytes, không tin extension URL."""
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"

    if payload.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"

    raise ValueError(
        "Nội dung tải về không phải PNG, JPEG, WebP hoặc GIF hợp lệ."
    )


def _relative_path(path: Path, project_root: Path) -> str:
    """Trả về đường dẫn POSIX tương đối bên trong project."""
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("Media phải nằm bên trong project root.") from error

    return relative.as_posix()


def _find_existing_images(asset_dir: Path, stem: str) -> list[Path]:
    """Tìm ảnh đã tồn tại đúng stem và các extension được hỗ trợ."""
    return [
        asset_dir / f"{stem}{extension}"
        for extension in _IMAGE_EXTENSIONS
        if (asset_dir / f"{stem}{extension}").is_file()
    ]


def download_image(
    url: str,
    asset_dir: Path,
    stem: str,
    project_root: Path,
    refresh: bool = False,
) -> tuple[str, bool]:
    """Tải ảnh nguyên tử và trả về (relative_path, downloaded)."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Media URL phải dùng http hoặc https.")

    asset_dir.mkdir(parents=True, exist_ok=True)
    existing_images = _find_existing_images(asset_dir, stem)

    if len(existing_images) > 1:
        raise ValueError(f"Phát hiện nhiều file media cùng stem: {stem}.")

    if existing_images and not refresh:
        return _relative_path(existing_images[0], project_root), False

    payload = _download_image_payload(url)

    if len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn 10 MB.")

    extension = _detect_image_extension(payload)
    target = asset_dir / f"{stem}{extension}"
    temporary = target.with_suffix(f"{target.suffix}.part")

    temporary.write_bytes(payload)
    temporary.replace(target)

    if _is_fandom_url(url):
        time.sleep(_FANDOM_MEDIA_INTERVAL_SECONDS)

    return _relative_path(target, project_root), True


def _is_fandom_url(url: str) -> bool:
    """Nhận diện URL trang hoặc CDN media thuộc hệ sinh thái Fandom."""
    hostname = (urlparse(url).hostname or "").casefold()
    return (
        hostname == "fandom.com"
        or hostname.endswith(".fandom.com")
        or hostname == "static.wikia.nocookie.net"
    )


def _is_retryable_media_http_error(
    error: HTTPError,
    url: str,
) -> bool:
    """Cho phép retry 403 riêng cho media Fandom bị giới hạn tạm thời."""
    return error.code in _RETRYABLE_HTTP_CODES or (
        error.code == 403 and _is_fandom_url(url)
    )


def _media_retry_delay_seconds(
    error: Exception | None,
    url: str,
    attempt: int,
) -> int:
    """Chọn backoff dài cho rate limit Fandom, ngắn cho lỗi mạng khác."""
    if (
        isinstance(error, HTTPError)
        and error.code in {403, 429}
        and _is_fandom_url(url)
    ):
        return _FANDOM_RATE_LIMIT_DELAYS_SECONDS[
            attempt - 1
        ]

    return 2 ** (attempt - 1)


def _download_image_payload(url: str) -> bytes:
    """Tải image bytes với retry giới hạn cho lỗi mạng tạm thời."""
    last_error: Exception | None = None
    is_fandom_url = _is_fandom_url(url)

    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        headers = {
            "User-Agent": "match-insight-reference-sync/1.0",
        }

        if is_fandom_url:
            headers.update(
                {
                    "Accept": (
                        "image/webp,image/apng,"
                        "image/png,image/jpeg,*/*;q=0.8"
                    ),
                    "Referer": "https://lol.fandom.com/",
                }
            )

        request = Request(
            url,
            headers=headers,
        )

        try:
            with urlopen(
                request,
                timeout=_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                payload = response.read(
                    _MAX_IMAGE_BYTES + 1
                )

            return payload
        except HTTPError as error:
            last_error = error
            retryable = _is_retryable_media_http_error(
                error=error,
                url=url,
            )

            if not retryable or attempt == _MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"Không tải được media URL {url}: HTTP {error.code}.") from error
        except (
            TimeoutError,
            URLError,
            ConnectionResetError,
        ) as error:
            last_error = error

            if attempt == _MAX_DOWNLOAD_ATTEMPTS:
                raise RuntimeError(
                    f"Không tải được media sau {_MAX_DOWNLOAD_ATTEMPTS} lần: {url}"
                ) from error

        retry_delay = _media_retry_delay_seconds(
            error=last_error,
            url=url,
            attempt=attempt,
        )
        _LOGGER.warning(
            "Media request lỗi tại %s; thử lại sau %s giây.",
            urlparse(url).hostname or "unknown-host",
            retry_delay,
        )
        time.sleep(retry_delay)

    raise RuntimeError(f"Không tải được media URL: {url}") from last_error


def file_sha256(path: Path) -> str:
    """Tính SHA-256 để truy vết chính xác file CSV đầu vào."""
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(64 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def write_sync_manifest(
    asset_dir: Path,
    source: dict[str, object],
    stats: SyncStats,
) -> None:
    """Ghi metadata của lần sync gần nhất bằng thao tác nguyên tử."""
    asset_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "synced_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "stats": asdict(stats),
    }

    target = asset_dir / "_sync_manifest.json"
    temporary = asset_dir / "_sync_manifest.json.part"
    content = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )

    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
