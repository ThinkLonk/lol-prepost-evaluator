"""Kiểm tra khói cho bộ khung project."""

from match_insight import SYSTEM_NAME
from match_insight.ui.app import run


def test_system_name() -> None:
    assert SYSTEM_NAME == "Match Insight"


def test_streamlit_entrypoint_is_callable() -> None:
    assert callable(run)
