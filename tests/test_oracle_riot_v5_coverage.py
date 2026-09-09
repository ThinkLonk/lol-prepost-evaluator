"""Unit test cho logic kiểm định độ bao phủ và tính toàn vẹn thời gian Riot V5."""

from __future__ import annotations

from scripts.audit_oracle_riot_v5_coverage import validate_v5_record


def test_validate_v5_record_valid():
    valid_record = {
        "game_id": "LOLTMNT01_189329",
        "status": "SUCCESS",
        "start_timestamp_ms": 1737468667522,
        "end_timestamp_ms": 1737470553120,
        "started_at": "2025-01-21T14:11:07.522Z",
        "ended_at": "2025-01-21T14:42:33.120Z",
        "game_duration_sec": 1779,
        "patch": "15.1.649.4112",
        "participants_count": 10,
    }
    is_valid, reason = validate_v5_record(valid_record)
    assert is_valid is True
    assert reason == "VALID"


def test_validate_v5_record_invalid_order():
    invalid_record = {
        "game_id": "LOLTMNT01_189329",
        "status": "SUCCESS",
        "start_timestamp_ms": 1737470553120,
        "end_timestamp_ms": 1737468667522,  # end < start
        "started_at": "2025-01-21T14:42:33.120Z",
        "ended_at": "2025-01-21T14:11:07.522Z",
    }
    is_valid, reason = validate_v5_record(invalid_record)
    assert is_valid is False
    assert reason == "INVALID_TIME_ORDER"


def test_validate_v5_record_equal_timestamps():
    invalid_record = {
        "game_id": "LOLTMNT01_189329",
        "status": "SUCCESS",
        "start_timestamp_ms": 1737468667522,
        "end_timestamp_ms": 1737468667522,  # end == start
        "started_at": "2025-01-21T14:11:07.522Z",
        "ended_at": "2025-01-21T14:11:07.522Z",
    }
    is_valid, reason = validate_v5_record(invalid_record)
    assert is_valid is False
    assert reason == "INVALID_TIME_ORDER"


def test_validate_v5_record_negative_timestamp():
    invalid_record = {
        "game_id": "LOLTMNT01_189329",
        "status": "SUCCESS",
        "start_timestamp_ms": -100,
        "end_timestamp_ms": 1737468667522,
        "started_at": "2025-01-21T14:11:07.522Z",
        "ended_at": "2025-01-21T14:42:33.120Z",
    }
    is_valid, reason = validate_v5_record(invalid_record)
    assert is_valid is False
    assert reason == "NON_POSITIVE_TIMESTAMPS"


def test_validate_v5_record_not_success():
    record = {
        "game_id": "11715-11715_game_1",
        "status": "PAGE_MISSING",
        "started_at": None,
        "ended_at": None,
    }
    is_valid, reason = validate_v5_record(record)
    assert is_valid is False
    assert reason == "PAGE_MISSING"


def test_validate_v5_record_missing_iso():
    record = {
        "game_id": "LOLTMNT01_189329",
        "status": "SUCCESS",
        "start_timestamp_ms": 1737468667522,
        "end_timestamp_ms": 1737470553120,
        "started_at": None,
        "ended_at": None,
    }
    is_valid, reason = validate_v5_record(record)
    assert is_valid is False
    assert reason == "MISSING_ISO_TIMESTAMPS"
