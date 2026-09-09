"""Unit test kiểm thử logic trích xuất payload nạp dữ liệu thời gian."""

from __future__ import annotations

from datetime import datetime

from scripts.apply_oracle_riot_v5_backfill import extract_valid_backfill_payloads


def test_extract_valid_backfill_payloads_success():
    sample_list = [
        {
            "game_id": "LOLTMNT01_189329",
            "status": "SUCCESS",
            "start_timestamp_ms": 1737468667522,
            "end_timestamp_ms": 1737470553120,
            "started_at": "2025-01-21T14:11:07.522Z",
            "ended_at": "2025-01-21T14:42:33.120Z",
        },
        {
            "game_id": "11715-11715_game_1",
            "status": "PAGE_MISSING",
            "started_at": None,
            "ended_at": None,
        },
    ]

    valid = extract_valid_backfill_payloads(sample_list)
    assert len(valid) == 1
    item = valid[0]
    assert item["game_id"] == "LOLTMNT01_189329"
    assert isinstance(item["started_at"], datetime)
    assert isinstance(item["ended_at"], datetime)
    assert item["ended_at"] > item["started_at"]
    assert item["started_at"].tzinfo is not None


def test_extract_valid_backfill_payloads_filter_invalid_order():
    sample_list = [
        {
            "game_id": "LOLTMNT01_189329",
            "status": "SUCCESS",
            "start_timestamp_ms": 1737470553120,
            "end_timestamp_ms": 1737468667522,  # end < start
            "started_at": "2025-01-21T14:42:33.120Z",
            "ended_at": "2025-01-21T14:11:07.522Z",
        }
    ]

    valid = extract_valid_backfill_payloads(sample_list)
    assert len(valid) == 0


def test_extract_valid_backfill_payloads_filter_empty_list():
    valid = extract_valid_backfill_payloads([])
    assert valid == []
