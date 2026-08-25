"""Màn hình Streamlit tối thiểu dùng để kiểm tra bộ khung project."""

import streamlit as st

from match_insight import SYSTEM_NAME


def run() -> None:
    """Hiển thị trạng thái khởi tạo, chưa chứa logic nghiệp vụ."""
    st.set_page_config(page_title=SYSTEM_NAME)
    st.title(SYSTEM_NAME)
    st.info("Project Python đã được khởi tạo thành công.")
