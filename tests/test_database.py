"""Kiểm thử tối thiểu cho cấu hình và kết nối PostgreSQL."""

from sqlalchemy.engine import make_url

from match_insight.config import Settings
from match_insight.database.engine import check_database_connection


def test_database_url_configuration() -> None:
    url = make_url(Settings().database_url)

    assert url.drivername == "postgresql+psycopg"
    assert url.database


def test_database_connection_returns_one() -> None:
    assert check_database_connection() == 1