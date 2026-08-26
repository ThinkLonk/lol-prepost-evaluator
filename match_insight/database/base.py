"""Declarative Base dùng chung cho các SQLAlchemy ORM model."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Lớp gốc của toàn bộ ORM model trong hệ thống."""