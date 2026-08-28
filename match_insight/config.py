import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_LOCAL_ENV_KEYS = frozenset(
    {
        "DATABASE_URL",
        "WIKI_USERNAME_MATCH_INSIGHT",
        "WIKI_PASSWORD_MATCH_INSIGHT",
    }
)


def load_local_environment() -> None:
    """Nạp các biến cục bộ được cho phép từ file .env."""
    if not _ENV_FILE.is_file():
        return

    for raw_line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")
        normalized_key = key.strip()

        if (
            separator
            and normalized_key in _LOCAL_ENV_KEYS
            and not os.getenv(normalized_key)
        ):
            os.environ[normalized_key] = value.strip()


class Settings:
    """Các cấu hình bắt buộc của ứng dụng."""

    def __init__(self) -> None:
        load_local_environment()

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("Thiếu biến môi trường DATABASE_URL.")

        self.database_url = database_url
