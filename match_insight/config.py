import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def _load_local_database_url() -> None:
    """Nạp DATABASE_URL từ .env nếu hệ điều hành chưa cung cấp biến này."""
    if os.getenv("DATABASE_URL") or not _ENV_FILE.is_file():
        return

    for raw_line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw_line.partition("=")

        if separator and key.strip() == "DATABASE_URL":
            os.environ["DATABASE_URL"] = value.strip()
            return


class Settings:
    """Các cấu hình bắt buộc của ứng dụng."""

    def __init__(self) -> None:
        _load_local_database_url()

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("Thiếu biến môi trường DATABASE_URL.")

        self.database_url = database_url