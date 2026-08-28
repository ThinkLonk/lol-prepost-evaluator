"""Kiểm thử đọc champion metadata mà không truy cập Internet."""

from __future__ import annotations

import json

import pytest

import match_insight.data_processing.champions as champion_module
from match_insight.data_processing.champions import fetch_champions


class FakeResponse:
    """Response tối thiểu thay cho kết nối Data Dragon."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_fetch_champions_maps_data_dragon_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload: dict[str, object] = {
        "data": {
            "Aatrox": {
                "id": "Aatrox",
                "key": "266",
                "name": "Aatrox",
                "image": {
                    "full": "Aatrox.png",
                },
            }
        }
    }

    def fake_urlopen(
        request: object,
        timeout: int,
    ) -> FakeResponse:
        assert request is not None
        assert timeout == 30
        return FakeResponse(payload)

    monkeypatch.setattr(
        champion_module,
        "urlopen",
        fake_urlopen,
    )

    records, source_url = fetch_champions(
        version="16.16.1",
        locale="en_US",
    )

    assert len(records) == 1
    assert records[0].champion_id == "266"
    assert records[0].canonical_name == "Aatrox"
    assert records[0].display_name == "Aatrox"
    assert records[0].image_url.endswith(
        "/cdn/16.16.1/img/champion/Aatrox.png"
    )
    assert source_url.endswith(
        "/cdn/16.16.1/data/en_US/champion.json"
    )


def test_fetch_champions_rejects_unpinned_version() -> None:
    with pytest.raises(ValueError):
        fetch_champions(
            version="latest",
            locale="en_US",
        )


def test_fetch_champions_rejects_invalid_locale() -> None:
    with pytest.raises(ValueError):
        fetch_champions(
            version="16.16.1",
            locale="../../etc",
        )