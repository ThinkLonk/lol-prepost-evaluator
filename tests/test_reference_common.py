"""Kiểm thử tiện ích reference data và normalization."""

import pytest

import match_insight.data_processing.reference_common as reference_common
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


def test_namespaced_oracle_stable_id_is_accepted() -> None:
    team_id = "oe:team:84bc703e28859788770611d94cf02ac"
    player_id = "oe:player:c659697694306de62d978569b84c344"
    tournament_id = "oracle:tournament:lpl-2025"

    assert validate_stable_id(team_id, "team_id", 64) == team_id
    assert validate_stable_id(player_id, "player_id", 64) == player_id
    assert validate_stable_id(tournament_id, "tournament_id", 64) == tournament_id


def test_asset_stem_sanitizes_colons_for_filesystem() -> None:
    result = asset_stem(
        stable_id="oe:team:001",
        canonical_name="Đội Oracle",
    )

    assert result == "oe~team~001-doi-oracle"


def test_asset_stems_distinguish_namespace_and_underscore_ids() -> None:
    ids = (
        "oe:team:001",
        "oe_team_001",
        "oe:team_001",
        "oe_team:001",
    )

    stems = [asset_stem(value, "Đội Oracle") for value in ids]

    assert len(set(stems)) == len(ids)
    assert stems[1] == "oe_team_001-doi-oracle"


@pytest.mark.parametrize("refresh", (False, True))
def test_namespaced_media_does_not_reuse_or_replace_underscore_media(
    tmp_path,
    monkeypatch,
    refresh,
) -> None:
    existing = tmp_path / "oe_team_001-doi-oracle.png"
    old_payload = b"\x89PNG\r\n\x1a\nold-fixture"
    new_payload = b"\x89PNG\r\n\x1a\nnew-fixture"
    existing.write_bytes(old_payload)
    calls = []

    def fake_download(url):
        calls.append(url)
        return new_payload

    monkeypatch.setattr(
        reference_common,
        "_download_image_payload",
        fake_download,
    )
    url = "https://example.invalid/image.png"

    relative_path, downloaded = reference_common.download_image(
        url=url,
        asset_dir=tmp_path,
        stem=asset_stem("oe:team:001", "Đội Oracle"),
        project_root=tmp_path,
        refresh=refresh,
    )

    assert downloaded is True
    assert relative_path == "oe~team~001-doi-oracle.png"
    assert (tmp_path / relative_path).read_bytes() == new_payload
    assert existing.read_bytes() == old_payload
    assert calls == [url]
