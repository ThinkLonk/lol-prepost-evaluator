"""Kiểm thử đọc curated team CSV."""

from pathlib import Path

import pytest

import match_insight.data_processing.teams as team_module
from match_insight.data_processing.teams import (
    TeamRecord,
    fetch_teams,
    load_teams_csv,
    write_teams_csv,
)


def test_load_teams_csv_reads_valid_record(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    csv_path.write_text(
        (
            "team_id,canonical_name,display_name,"
            "record_source,record_source_url,logo_url\n"
            "team_001,Example Team,Example Team,"
            "manual_curated,https://source.example/team,"
            "https://media.example/team.png\n"
        ),
        encoding="utf-8",
    )

    records = load_teams_csv(csv_path)

    assert len(records) == 1
    assert records[0].team_id == "team_001"
    assert records[0].canonical_name == "Example Team"
    assert records[0].display_name == "Example Team"
    assert records[0].record_source == "manual_curated"
    assert records[0].logo_url == "https://media.example/team.png"


def test_load_teams_csv_allows_missing_logo(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    csv_path.write_text(
        (
            "team_id,canonical_name,display_name,"
            "record_source,record_source_url,logo_url\n"
            "team_001,Example Team,Example Team,"
            "manual_curated,https://source.example/team,\n"
        ),
        encoding="utf-8",
    )

    records = load_teams_csv(csv_path)

    assert len(records) == 1
    assert records[0].logo_url is None


def test_load_teams_csv_rejects_duplicate_team_id(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    csv_path.write_text(
        (
            "team_id,canonical_name,display_name,"
            "record_source,record_source_url,logo_url\n"
            "team_001,Example Team,Example Team,"
            "manual_curated,https://source.example/team,\n"
            "team_001,Renamed Team,Renamed Team,"
            "manual_curated,https://source.example/team-new,\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="team_id trùng",
    ):
        load_teams_csv(csv_path)


def test_load_teams_csv_rejects_missing_column(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    csv_path.write_text(
        "team_id,canonical_name\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="thiếu cột",
    ):
        load_teams_csv(csv_path)


def test_load_teams_csv_accepts_header_only(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    csv_path.write_text(
        (
            "team_id,canonical_name,display_name,"
            "record_source,record_source_url,logo_url\n"
        ),
        encoding="utf-8",
    )

    records = load_teams_csv(csv_path)

    assert records == []


def test_fetch_teams_maps_stable_ids_and_media_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        assert arguments == {
            "table": "Teams",
            "fields": (
                "Teams._pageID=SourcePageId",
                "Teams.Name=Name",
                "Teams.OverviewPage=OverviewPage",
                "Teams.Image=Image",
            ),
            "order_by": "Teams._pageID",
            "snapshot_path": None,
        }

        return [
            {
                "SourcePageId": "123",
                "Name": "Test Team",
                "OverviewPage": "Test Team",
                "Image": "Test Team (EU).png",
            },
            {
                "SourcePageId": "456",
                "Name": "",
                "OverviewPage": "Fallback Team",
                "Image": "",
            },
        ]

    monkeypatch.setattr(
        team_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    records = fetch_teams()

    assert len(records) == 2

    assert records[0].team_id == "lp_team_123"
    assert records[0].canonical_name == "Test Team"
    assert records[0].display_name == "Test Team"
    assert records[0].record_source == "leaguepedia_cargo"
    assert records[0].record_source_url == (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/page/123"
    )
    assert records[0].logo_url == (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/file/"
        "Test_Team_%28EU%29.png"
    )

    assert records[1].team_id == "lp_team_456"
    assert records[1].canonical_name == "Fallback Team"
    assert records[1].logo_url is None


def test_fetch_teams_rejects_invalid_source_page_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        assert arguments

        return [
            {
                "SourcePageId": "",
                "Name": "Invalid Team",
                "OverviewPage": "Invalid Team",
                "Image": "",
            }
        ]

    monkeypatch.setattr(
        team_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    with pytest.raises(
        ValueError,
        match="SourcePageId không hợp lệ",
    ):
        fetch_teams()


def test_fetch_teams_rejects_duplicate_source_page_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        assert arguments

        return [
            {
                "SourcePageId": "123",
                "Name": "First Team",
                "OverviewPage": "First Team",
                "Image": "",
            },
            {
                "SourcePageId": "123",
                "Name": "Duplicate Team",
                "OverviewPage": "Duplicate Team",
                "Image": "",
            },
        ]

    monkeypatch.setattr(
        team_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    with pytest.raises(
        ValueError,
        match="team ID trùng",
    ):
        fetch_teams()

def test_write_teams_csv_round_trips_records(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "reference" / "teams.csv"
    records = [
        TeamRecord(
            team_id="lp_team_123",
            canonical_name="First Team",
            display_name="First Team",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            logo_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/file/First_Team.png"
            ),
        ),
        TeamRecord(
            team_id="lp_team_456",
            canonical_name="Second Team",
            display_name="Second Team",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/456"
            ),
            logo_url=None,
        ),
    ]

    write_teams_csv(
        path=csv_path,
        records=records,
    )
    loaded_records = load_teams_csv(csv_path)

    assert loaded_records == records
    assert csv_path.is_file()
    assert not csv_path.with_suffix(
        ".csv.part"
    ).exists()


def test_write_teams_csv_rejects_empty_records(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    original_content = "existing content\n"
    csv_path.write_text(
        original_content,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="danh sách team rỗng",
    ):
        write_teams_csv(
            path=csv_path,
            records=[],
        )

    assert csv_path.read_text(
        encoding="utf-8",
    ) == original_content


def test_write_teams_csv_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "teams.csv"
    original_content = "existing content\n"
    csv_path.write_text(
        original_content,
        encoding="utf-8",
    )

    duplicate_records = [
        TeamRecord(
            team_id="lp_team_123",
            canonical_name="First Team",
            display_name="First Team",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            logo_url=None,
        ),
        TeamRecord(
            team_id="lp_team_123",
            canonical_name="Duplicate Team",
            display_name="Duplicate Team",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            logo_url=None,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="team_id trùng",
    ):
        write_teams_csv(
            path=csv_path,
            records=duplicate_records,
        )

    assert csv_path.read_text(
        encoding="utf-8",
    ) == original_content
