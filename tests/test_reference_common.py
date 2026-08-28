"""Kiểm thử tiện ích reference data và normalization."""

import pytest

from match_insight.data_processing.reference_common import (
    _detect_image_extension,
    asset_stem,
    bounded,
    canonical_key,
    clean_label,
    validate_stable_id,
)


def test_detect_image_extension_supports_gif() -> None:
    assert _detect_image_extension(b"GIF89a-test") == ".gif"


def test_clean_label_normalizes_unicode_and_spaces() -> None:
    result = clean_label(
        "  Example   Team  ",
        field_name="name",
    )

    assert result == "Example Team"


def test_canonical_key_is_case_insensitive() -> None:
    first = canonical_key("Example Team")
    second = canonical_key("  example team  ")

    assert first == second
    assert first == "example team"


def test_asset_stem_contains_stable_id_and_normalized_name() -> None:
    result = asset_stem(
        stable_id="team_001",
        canonical_name="Đội Tuyển Ví Dụ",
    )

    assert result == "team_001-doi-tuyen-vi-du"


def test_invalid_stable_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_stable_id(
            "../team",
            field_name="team_id",
            maximum=64,
        )


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        clean_label(
            "   ",
            field_name="display_name",
        )


def test_value_over_database_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        bounded(
            value="a" * 81,
            field_name="canonical_name",
            maximum=80,
        )
