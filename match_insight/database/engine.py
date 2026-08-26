"""Khởi tạo và kiểm tra SQLAlchemy engine cho PostgreSQL."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from match_insight.config import Settings

engine: Engine = create_engine(Settings().database_url)


def check_database_connection() -> int:
    """Kết nối PostgreSQL và trả về kết quả của SELECT 1."""
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1")).scalar_one()

    return result