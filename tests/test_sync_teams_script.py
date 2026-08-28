"""Kiểm thử điều phối nguồn của sync_teams CLI."""

from pathlib import Path

import pytest

import scripts.sync_teams as sync_script
from match_insight.data_processing.teams import (
    TeamRecord,
    write_teams_csv,
)


def _sample_team_record() -> TeamRecord:
    return TeamRecord(
        team_id="lp_team_123",
        canonical_name="Test Team",
        display_name="Test Team",
        record_source="leaguepedia_cargo",
        record_source_url=(
            "https://lol.fandom.com/wiki/"
            "Special:Redirect/page/123"
        ),
        logo_url=None,
    )


def test_prepare_records_from_leaguepedia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "reference" / "teams.csv"
    raw_output_path = (
        tmp_path / "raw" / "leaguepedia" / "teams.json"
    )
    expected_record = _sample_team_record()
    observed_snapshot_path: Path | None = None
    observed_staging_path: Path | None = None

    def fake_fetch_teams(
        snapshot_path: Path | None = None,
    ) -> list[TeamRecord]:
        nonlocal observed_snapshot_path
        observed_snapshot_path = snapshot_path
        return [expected_record]

    def fake_write_teams_csv(
        path: Path,
        records: list[TeamRecord],
    ) -> None:
        nonlocal observed_staging_path
        observed_staging_path = path
        assert records == [expected_record]

    def fake_file_sha256(path: Path) -> str:
        if path == raw_output_path:
            return "raw-hash"

        if path == input_path:
            return "staging-hash"

        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(
        sync_script,
        "fetch_teams",
        fake_fetch_teams,
    )
    monkeypatch.setattr(
        sync_script,
        "write_teams_csv",
        fake_write_teams_csv,
    )
    monkeypatch.setattr(
        sync_script,
        "file_sha256",
        fake_file_sha256,
    )

    records, source = sync_script.prepare_records(
        source="leaguepedia",
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    assert records == [expected_record]
    assert observed_snapshot_path == raw_output_path
    assert observed_staging_path == input_path
    assert source == {
        "type": "leaguepedia_cargo",
        "api_url": sync_script.LEAGUEPEDIA_API_URL,
        "client": sync_script.LEAGUEPEDIA_CLIENT_NAME,
        "media_url_resolution": "mediawiki_imageinfo",
        "table": "Teams",
        "raw_snapshot": str(raw_output_path),
        "raw_sha256": "raw-hash",
        "staging_file": str(input_path),
        "staging_sha256": "staging-hash",
        "record_count": 1,
    }


def test_prepare_records_from_csv(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "teams.csv"
    raw_output_path = tmp_path / "unused.json"
    expected_record = _sample_team_record()

    write_teams_csv(
        path=input_path,
        records=[expected_record],
    )

    records, source = sync_script.prepare_records(
        source="csv",
        input_path=input_path,
        raw_output_path=raw_output_path,
    )

    assert records == [expected_record]
    assert source["type"] == "curated_csv"
    assert source["input_file"] == str(input_path)
    assert source["media_url_resolution"] == (
        "mediawiki_imageinfo"
    )
    assert source["record_count"] == 1
    assert isinstance(source["input_sha256"], str)


def test_prepare_records_rejects_empty_leaguepedia(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_teams(
        snapshot_path: Path | None = None,
    ) -> list[TeamRecord]:
        assert snapshot_path is not None
        return []

    monkeypatch.setattr(
        sync_script,
        "fetch_teams",
        fake_fetch_teams,
    )

    with pytest.raises(
        RuntimeError,
        match="không trả team record",
    ):
        sync_script.prepare_records(
            source="leaguepedia",
            input_path=tmp_path / "teams.csv",
            raw_output_path=tmp_path / "teams.json",
        )
