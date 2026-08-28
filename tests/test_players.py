"""Kiểm thử đọc curated player CSV."""

import json
from pathlib import Path

import pytest

import match_insight.data_processing.players as player_module
from match_insight.data_processing.players import (
    PlayerRecord,
    delete_pruned_player_media,
    fetch_players,
    load_players_csv,
    load_players_snapshot,
    prune_missing_leaguepedia_players,
    write_players_csv,
)
from match_insight.database.models import Player


def test_load_players_csv_reads_valid_record(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        (
            "player_id,canonical_name,display_name,"
            "record_source,record_source_url,photo_url\n"
            "player_001,Example Player,Example Player,"
            "manual_curated,https://source.example/player,"
            "https://media.example/player.jpg\n"
        ),
        encoding="utf-8",
    )

    records = load_players_csv(csv_path)

    assert len(records) == 1
    assert records[0].player_id == "player_001"
    assert records[0].canonical_name == "Example Player"
    assert records[0].display_name == "Example Player"
    assert records[0].record_source == "manual_curated"
    assert records[0].photo_url == "https://media.example/player.jpg"


def test_load_players_csv_allows_missing_photo(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        (
            "player_id,canonical_name,display_name,"
            "record_source,record_source_url,photo_url\n"
            "player_001,Example Player,Example Player,"
            "manual_curated,https://source.example/player,\n"
        ),
        encoding="utf-8",
    )

    records = load_players_csv(csv_path)

    assert len(records) == 1
    assert records[0].photo_url is None


def test_load_players_csv_rejects_duplicate_player_id(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        (
            "player_id,canonical_name,display_name,"
            "record_source,record_source_url,photo_url\n"
            "player_001,Example Player,Example Player,"
            "manual_curated,https://source.example/player,\n"
            "player_001,Renamed Player,Renamed Player,"
            "manual_curated,https://source.example/player-new,\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="player_id trùng",
    ):
        load_players_csv(csv_path)


def test_load_players_csv_rejects_missing_column(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        "player_id,canonical_name\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="thiếu cột",
    ):
        load_players_csv(csv_path)


def test_load_players_csv_accepts_header_only(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    csv_path.write_text(
        (
            "player_id,canonical_name,display_name,"
            "record_source,record_source_url,photo_url\n"
        ),
        encoding="utf-8",
    )

    records = load_players_csv(csv_path)

    assert records == []


def test_load_players_snapshot_filters_personality_and_uses_profile_image(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "players.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {
                    "catalog_table": "Players",
                    "evidence_tables": [
                        "PlayerLeagueHistory",
                        "TournamentPlayers",
                        "ListplayerCurrent",
                    ],
                    "media_table": "PlayerImages",
                },
                "media_urls_are_direct": True,
                "catalog_row_count": 3,
                "game_evidence_row_count": 1,
                "roster_row_count": 1,
                "image_row_count": 1,
                "catalog_rows": [
                    {
                        "SourcePageId": "123",
                        "SourcePageName": "Faker",
                        "PlayerHandle": "Faker",
                        "Image": "",
                        "PhotoUrl": "",
                        "IsPersonality": "0",
                    },
                    {
                        "SourcePageId": "456",
                        "SourcePageName": "Chobra",
                        "PlayerHandle": "Chobra",
                        "Image": "Chobra.png",
                        "PhotoUrl": "https://media.example/chobra.png",
                        "IsPersonality": "1",
                    },
                    {
                        "SourcePageId": "789",
                        "SourcePageName": "Rookie",
                        "PlayerHandle": "Rookie",
                        "Image": "",
                        "PhotoUrl": "",
                        "IsPersonality": "0",
                    },
                ],
                "game_evidence_rows": [
                    {"PlayerPage": "Faker"},
                ],
                "roster_rows": [
                    {"PlayerPage": "Rookie", "Role": "Support"},
                ],
                "image_rows": [
                    {
                        "PlayerPage": "Faker",
                        "FileName": "Faker 2026.png",
                        "IsProfileImage": "1",
                        "SortDate": "2026-01-01",
                        "PhotoUrl": (
                            "https://static.wikia.nocookie.net/"
                            "example/Faker.png"
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    selection = load_players_snapshot(snapshot_path)
    records = selection.records

    assert len(records) == 2
    assert records[0].player_id == "lp_player_123"
    assert records[0].photo_url == (
        "https://static.wikia.nocookie.net/"
        "example/Faker.png"
    )
    assert records[1].player_id == "lp_player_789"
    assert records[1].photo_url is None
    assert selection.stats.catalog_total == 3
    assert selection.stats.included_by_game == 1
    assert selection.stats.included_by_roster_only == 1
    assert selection.stats.excluded_personality == 1


def test_load_players_snapshot_rejects_unsafe_legacy_schema(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "players.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "source": {"table": "Players"},
                "row_count": 1,
                "rows": [
                    {
                        "SourcePageId": "123",
                        "SourcePageName": "Faker",
                        "PlayerHandle": "Faker",
                        "Image": "Faker.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="schema_version=2",
    ):
        load_players_snapshot(snapshot_path)


def test_load_players_snapshot_rejects_missing_evidence_sources(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "players.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {
                    "catalog_table": "Players",
                    "evidence_tables": ["PlayerLeagueHistory"],
                    "media_table": "PlayerImages",
                },
                "media_urls_are_direct": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="không khai báo đủ",
    ):
        load_players_snapshot(snapshot_path)


def test_refresh_snapshot_image_sort_dates_preserves_media_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "players.json"
    query_snapshot_path = tmp_path / "player-image-sort-dates.json"
    original_photo_url = (
        "https://static.wikia.nocookie.net/example/gumayusi.png"
        "?cb=20260203040506"
    )
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {
                    "catalog_table": "Players",
                    "evidence_tables": [
                        "PlayerLeagueHistory",
                        "TournamentPlayers",
                        "ListplayerCurrent",
                    ],
                    "media_table": "PlayerImages",
                },
                "media_urls_are_direct": True,
                "image_row_count": 2,
                "image_rows": [
                    {
                        "PlayerPage": "Gumayusi",
                        "FileName": "HLE Gumayusi 2026 Split 1.png",
                        "IsProfileImage": "1",
                        "PhotoUrl": original_photo_url,
                    },
                    {
                        "PlayerPage": "Legacy",
                        "FileName": "Legacy.png",
                        "IsProfileImage": "1",
                        "PhotoUrl": "https://example.test/legacy.png",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch_player_image_rows(
        snapshot_path: Path | None = None,
    ) -> list[dict[str, str]]:
        assert snapshot_path == query_snapshot_path
        return [
            {
                "PlayerPage": "Gumayusi",
                "FileName": "HLE Gumayusi 2026 Split 1.png",
                "Tournament": "LCK 2026 Split 1",
                "SortDate": "",
                "TournamentDateStartFuzzy": "2026-01-14",
                "TournamentDate": "2026-01-14",
                "EffectiveSortDate": "2026-01-14",
            },
            {
                "PlayerPage": "New player",
                "FileName": "New player 2026.png",
                "EffectiveSortDate": "2026-02-01",
            },
        ]

    monkeypatch.setattr(
        player_module,
        "fetch_player_image_rows",
        fake_fetch_player_image_rows,
    )

    stats = player_module.refresh_players_snapshot_image_sort_dates(
        snapshot_path,
        query_snapshot_path=query_snapshot_path,
    )
    refreshed = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert refreshed["image_rows"][0]["PhotoUrl"] == original_photo_url
    assert refreshed["image_rows"][0]["EffectiveSortDate"] == (
        "2026-01-14"
    )
    assert refreshed["image_rows"][1]["PhotoUrl"] == (
        "https://example.test/legacy.png"
    )
    assert refreshed["source"]["media_sort_enrichment"][
        "matched_image_rows"
    ] == 1
    assert stats.fetched_rows == 2
    assert stats.matched_rows == 1
    assert stats.unmatched_snapshot_rows == 1
    assert stats.new_source_rows == 1
    assert stats.duplicate_source_rows_collapsed == 0


def test_refresh_snapshot_reuses_query_and_collapses_duplicate_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "players.json"
    query_snapshot_path = tmp_path / "sort-dates.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "source": {},
                "image_row_count": 1,
                "image_rows": [
                    {
                        "PlayerPage": "Example",
                        "FileName": "Example.png",
                        "IsProfileImage": "1",
                        "PhotoUrl": "https://example.test/example.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    query_rows = [
        {
            "PlayerPage": "Example",
            "FileName": "Example.png",
            "IsProfileImage": "1",
            "SortDate": "2020-01-01",
            "EffectiveSortDate": "2020-01-01",
        },
        {
            "PlayerPage": "Example",
            "FileName": "Example.png",
            "IsProfileImage": "1",
            "SortDate": "2022-01-01",
            "EffectiveSortDate": "2022-01-01",
        },
    ]
    query_snapshot_path.write_text(
        json.dumps(
            {
                "source": {
                    "table": "PlayerImages=PI,Tournaments=T",
                    "join_on": "PI.Tournament=T.OverviewPage",
                },
                "row_count": len(query_rows),
                "rows": query_rows,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        player_module,
        "fetch_player_image_rows",
        lambda **arguments: pytest.fail("network query must not run"),
    )

    stats = player_module.refresh_players_snapshot_image_sort_dates(
        snapshot_path,
        query_snapshot_path=query_snapshot_path,
        reuse_query_snapshot=True,
    )
    refreshed = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert refreshed["image_rows"][0]["SortDate"] == "2022-01-01"
    assert stats.fetched_rows == 2
    assert stats.matched_rows == 1
    assert stats.duplicate_source_rows_collapsed == 1


def test_fetch_players_maps_stable_ids_and_media_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        calls.append(arguments)
        table = arguments["table"]

        if table == "Players":
            return [
                {
                    "SourcePageId": "123",
                    "SourcePageName": "Faker",
                    "PlayerHandle": "Faker",
                    "Image": "",
                    "IsPersonality": "0",
                },
                {
                    "SourcePageId": "456",
                    "SourcePageName": "Fallback Player",
                    "PlayerHandle": "",
                    "Image": "",
                    "IsPersonality": "0",
                },
            ]

        if table == "PlayerLeagueHistory":
            assert arguments == {
                "table": "PlayerLeagueHistory",
                "fields": (
                    "PlayerLeagueHistory.Player=PlayerPage",
                ),
                "where": "PlayerLeagueHistory.TotalGames>0",
                "order_by": "PlayerLeagueHistory.Player",
                "group_by": "PlayerLeagueHistory.Player",
            }
            return [{"PlayerPage": "Faker"}]

        if table == "TournamentPlayers":
            return [
                {
                    "PlayerPage": "Fallback Player",
                    "Role": "Jungle",
                }
            ]

        if table == "PlayerImages=PI,Tournaments=T":
            assert arguments == {
                "table": "PlayerImages=PI,Tournaments=T",
                "fields": player_module._LEAGUEPEDIA_PLAYER_IMAGE_FIELDS,
                "join_on": "PI.Tournament=T.OverviewPage",
                "where": "PI.IsProfileImage=1",
                "order_by": (
                    "PI.Link, "
                    "COALESCE(PI.SortDate,T.DateStartFuzzy,T.Date), "
                    "PI.FileName"
                ),
                "snapshot_path": None,
            }
            return [
                {
                    "PlayerPage": "Faker",
                    "FileName": "Faker 2026.png",
                    "IsProfileImage": "1",
                    "SortDate": "2026-01-01",
                    "EffectiveSortDate": "2026-01-01",
                }
            ]

        return []

    monkeypatch.setattr(
        player_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    selection = fetch_players()
    records = selection.records

    assert len(records) == 2

    assert records[0].player_id == "lp_player_123"
    assert records[0].canonical_name == "Faker"
    assert records[0].display_name == "Faker"
    assert records[0].record_source == "leaguepedia_cargo"
    assert records[0].record_source_url == (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/page/123"
    )
    assert records[0].photo_url == (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/file/Faker_2026.png"
    )

    assert records[1].player_id == "lp_player_456"
    assert records[1].canonical_name == (
        "Fallback Player"
    )
    assert records[1].photo_url is None
    assert selection.stats.included == 2
    assert selection.stats.included_by_game == 1
    assert selection.stats.included_by_roster_only == 1
    assert {call["table"] for call in calls} == {
        "Players",
        "PlayerLeagueHistory",
        "TournamentPlayers",
        "ListplayerCurrent",
        "PlayerImages=PI,Tournaments=T",
    }


def test_profile_image_ranking_uses_effective_sort_date() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Gumayusi",
                "FileName": "T1 Gumayusi 2021 Worlds.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2021-10-05",
            },
            {
                "PlayerPage": "Gumayusi",
                "FileName": "HLE Gumayusi 2026 Split 1.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2026-01-14",
            },
        ]
    )

    assert selected["gumayusi"]["FileName"] == (
        "HLE Gumayusi 2026 Split 1.png"
    )


def test_profile_image_ranking_prefers_explicit_sortdate_over_effective() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example portrait.png",
                "IsProfileImage": "1",
                "SortDate": "2026-01-01",
                "EffectiveSortDate": "2020-01-01",
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example 2025.png",
                "IsProfileImage": "1",
            },
        ]
    )

    assert selected["example"]["FileName"] == "Example portrait.png"


def test_profile_image_ranking_uses_filename_phase_as_fallback() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example 2025 Spring.png",
                "IsProfileImage": "1",
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example 2025 Worlds.png",
                "IsProfileImage": "1",
            },
        ]
    )

    assert selected["example"]["FileName"] == (
        "Example 2025 Worlds.png"
    )


def test_profile_image_ranking_prefers_semantic_date_over_newer_upload() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example 2026.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/old.png"
                    "?cb=20200102030405"
                ),
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example portrait.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/new.png"
                    "?cb=20270102030405"
                ),
            },
        ]
    )

    assert selected["example"]["FileName"] == "Example 2026.png"


def test_profile_image_ranking_uses_revision_only_without_semantic_date() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example alpha.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/a.png"
                    "?cb=20240102030405"
                ),
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example beta.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/b.png"
                    "?cb=20250102030405"
                ),
            },
        ]
    )

    assert selected["example"]["FileName"] == "Example beta.png"


def test_profile_image_ranking_rejects_numeric_invalid_revision_date() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example invalid.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/a.png"
                    "?cb=20259999299999"
                ),
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example valid.png",
                "IsProfileImage": "1",
                "PhotoUrl": (
                    "https://static.wikia.nocookie.net/example/b.png"
                    "?cb=20250101000000"
                ),
            },
        ]
    )

    assert selected["example"]["FileName"] == "Example valid.png"


def test_profile_image_ranking_is_deterministic_for_malformed_rows() -> None:
    selected = player_module._profile_images_by_page(
        [
            {
                "PlayerPage": "Example",
                "FileName": "Example B.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "not-a-date",
                "PhotoUrl": "https://example.test/b.png?cb=bad",
            },
            {
                "PlayerPage": "Example",
                "FileName": "Example A.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2025-99-99",
            },
            {
                "PlayerPage": "Ignored",
                "FileName": "Ignored 2099.png",
                "IsProfileImage": "0",
            },
            {
                "PlayerPage": "",
                "FileName": "Missing page 2099.png",
                "IsProfileImage": "1",
            },
        ]
    )

    assert selected["example"]["FileName"] == "Example B.png"
    assert "ignored" not in selected


def test_manual_player_image_competes_with_newer_profile_image() -> None:
    old_url = (
        "https://static.wikia.nocookie.net/example/old.png/"
        "revision/latest?cb=20190101000000"
    )
    new_url = (
        "https://static.wikia.nocookie.net/example/new.png/"
        "revision/latest?cb=20240101000000"
    )
    selection = player_module._records_from_leaguepedia_rows(
        catalog_rows=[
            {
                "SourcePageId": "123",
                "SourcePageName": "Example",
                "PlayerHandle": "Example",
                "Image": "old.png",
                "PhotoUrl": old_url,
                "IsPersonality": "0",
            }
        ],
        game_evidence_rows=[{"PlayerPage": "Example"}],
        roster_rows=[],
        image_rows=[
            {
                "PlayerPage": "Example",
                "FileName": "old.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2019-01-01",
                "PhotoUrl": old_url,
            },
            {
                "PlayerPage": "Example",
                "FileName": "new.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2024-01-01",
                "PhotoUrl": new_url,
            },
        ],
        direct_media_urls=True,
    )

    assert selection.records[0].photo_url == new_url
    assert selection.stats.media_selected_by_effective_sort_date == 1
    assert selection.stats.media_selected_by_manual_override == 0


def test_newer_manual_player_image_can_win_profile_ranking() -> None:
    manual_url = (
        "https://static.wikia.nocookie.net/example/"
        "Example_2026.png/revision/latest?cb=20260101000000"
    )
    profile_url = (
        "https://static.wikia.nocookie.net/example/"
        "Example_2024.png/revision/latest?cb=20240101000000"
    )
    selection = player_module._records_from_leaguepedia_rows(
        catalog_rows=[
            {
                "SourcePageId": "123",
                "SourcePageName": "Example",
                "PlayerHandle": "Example",
                "Image": "Example 2026.png",
                "PhotoUrl": manual_url,
                "IsPersonality": "0",
            }
        ],
        game_evidence_rows=[{"PlayerPage": "Example"}],
        roster_rows=[],
        image_rows=[
            {
                "PlayerPage": "Example",
                "FileName": "Example 2024.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2024-01-01",
                "PhotoUrl": profile_url,
            }
        ],
        direct_media_urls=True,
    )

    assert selection.records[0].photo_url == manual_url
    assert selection.stats.media_selected_by_manual_override == 1


def test_manual_image_without_url_reuses_matching_profile_url() -> None:
    profile_url = (
        "https://static.wikia.nocookie.net/example/"
        "Example_2026.png/revision/latest?cb=20260101000000"
    )
    selection = player_module._records_from_leaguepedia_rows(
        catalog_rows=[
            {
                "SourcePageId": "123",
                "SourcePageName": "Example",
                "PlayerHandle": "Example",
                "Image": "Example 2026.png",
                "PhotoUrl": "",
                "IsPersonality": "0",
            }
        ],
        game_evidence_rows=[{"PlayerPage": "Example"}],
        roster_rows=[],
        image_rows=[
            {
                "PlayerPage": "Example",
                "FileName": "Example 2026.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2026-01-01",
                "PhotoUrl": profile_url,
            }
        ],
        direct_media_urls=True,
    )

    assert selection.records[0].photo_url == profile_url


def test_mismatched_manual_image_url_uses_safe_file_redirect() -> None:
    profile_url = (
        "https://static.wikia.nocookie.net/example/"
        "Example_2025.png/revision/latest?cb=20250101000000"
    )
    mismatched_url = (
        "https://static.wikia.nocookie.net/example/"
        "Different_person.png/revision/latest?cb=20260101000000"
    )
    selection = player_module._records_from_leaguepedia_rows(
        catalog_rows=[
            {
                "SourcePageId": "123",
                "SourcePageName": "Example",
                "PlayerHandle": "Example",
                "Image": "Example 2026.png",
                "PhotoUrl": mismatched_url,
                "IsPersonality": "0",
            }
        ],
        game_evidence_rows=[{"PlayerPage": "Example"}],
        roster_rows=[],
        image_rows=[
            {
                "PlayerPage": "Example",
                "FileName": "Example 2025.png",
                "IsProfileImage": "1",
                "EffectiveSortDate": "2025-01-01",
                "PhotoUrl": profile_url,
            }
        ],
        direct_media_urls=True,
    )

    assert selection.records[0].photo_url == (
        "https://lol.fandom.com/wiki/Special:Redirect/file/"
        "Example_2026.png"
    )


def test_selection_keeps_former_player_now_marked_personality() -> None:
    selection = player_module._records_from_leaguepedia_rows(
        catalog_rows=[
            {
                "SourcePageId": "321",
                "SourcePageName": "Former Pro",
                "PlayerHandle": "Former Pro",
                "Image": "",
                "IsPersonality": "1",
            }
        ],
        game_evidence_rows=[
            {
                "PlayerPage": "Former Pro",
                "Role": "Top Laner",
            }
        ],
        roster_rows=[],
        image_rows=[],
    )

    assert [record.player_id for record in selection.records] == [
        "lp_player_321"
    ]
    assert selection.stats.included_by_game == 1
    assert selection.stats.excluded_personality == 0


@pytest.mark.parametrize(
    "role",
    [
        "Top",
        "Jungler",
        "Mid Lane",
        "AD",
        "ADC",
        "Marksman",
        "Support",
    ],
)
def test_in_game_role_accepts_only_project_positions(
    role: str,
) -> None:
    assert player_module.is_in_game_role(role) is True
    assert player_module.is_in_game_role("Caster") is False
    assert player_module.is_in_game_role("Head Coach") is False


def test_fetch_players_rejects_invalid_source_page_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        if arguments["table"] != "Players":
            return []

        return [
            {
                "SourcePageId": "",
                "SourcePageName": "Invalid Player",
                "PlayerHandle": "Invalid Player",
                "Image": "",
            }
        ]

    monkeypatch.setattr(
        player_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    with pytest.raises(
        ValueError,
        match="SourcePageId không hợp lệ",
    ):
        fetch_players()


def test_fetch_players_rejects_missing_source_page_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        if arguments["table"] != "Players":
            return []

        return [
            {
                "SourcePageId": "123",
                "SourcePageName": "",
                "PlayerHandle": "Missing Page",
                "Image": "",
            }
        ]

    monkeypatch.setattr(
        player_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    with pytest.raises(
        ValueError,
        match="thiếu SourcePageName",
    ):
        fetch_players()


def test_fetch_players_rejects_duplicate_source_page_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_cargo_rows(
        **arguments: object,
    ) -> list[dict[str, str]]:
        if arguments["table"] != "Players":
            return []

        return [
            {
                "SourcePageId": "123",
                "SourcePageName": "First Player",
                "PlayerHandle": "First Player",
                "Image": "",
            },
            {
                "SourcePageId": "123",
                "SourcePageName": "Duplicate Player",
                "PlayerHandle": "Duplicate Player",
                "Image": "",
            },
        ]

    monkeypatch.setattr(
        player_module,
        "fetch_cargo_rows",
        fake_fetch_cargo_rows,
    )

    with pytest.raises(
        ValueError,
        match="player ID trùng",
    ):
        fetch_players()


def test_resolve_player_photo_urls_uses_imageinfo_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redirect_url = (
        "https://lol.fandom.com/wiki/"
        "Special:Redirect/file/Faker.png"
    )
    direct_url = (
        "https://static.wikia.nocookie.net/"
        "test/images/a/ab/Faker.png"
    )
    records = [
        PlayerRecord(
            player_id="lp_player_123",
            canonical_name="Faker",
            display_name="Faker",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            photo_url=redirect_url,
        )
    ]

    def fake_fetch_file_urls(
        file_names: tuple[str, ...],
    ) -> dict[str, str]:
        assert file_names == ("Faker.png",)
        return {"Faker.png": direct_url}

    monkeypatch.setattr(
        player_module,
        "fetch_file_urls",
        fake_fetch_file_urls,
    )

    resolved = (
        player_module.resolve_player_photo_urls(
            records
        )
    )

    assert resolved[0].photo_url == direct_url
    assert records[0].photo_url == redirect_url


def test_resolve_player_photo_urls_marks_missing_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = PlayerRecord(
        player_id="lp_player_123",
        canonical_name="Test Player",
        display_name="Test Player",
        record_source="leaguepedia_cargo",
        record_source_url=(
            "https://lol.fandom.com/wiki/"
            "Special:Redirect/page/123"
        ),
        photo_url=(
            "https://lol.fandom.com/wiki/"
            "Special:Redirect/file/Missing.png"
        ),
    )

    monkeypatch.setattr(
        player_module,
        "fetch_file_urls",
        lambda file_names: {},
    )

    resolved = (
        player_module.resolve_player_photo_urls(
            [record]
        )
    )

    assert resolved[0].photo_url is None


def test_write_players_csv_round_trips_records(
    tmp_path: Path,
) -> None:
    csv_path = (
        tmp_path / "reference" / "players.csv"
    )
    records = [
        PlayerRecord(
            player_id="lp_player_123",
            canonical_name="First Player",
            display_name="First Player",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            photo_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/file/"
                "First_Player.png"
            ),
        ),
        PlayerRecord(
            player_id="lp_player_456",
            canonical_name="Second Player",
            display_name="Second Player",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/456"
            ),
            photo_url=None,
        ),
    ]

    write_players_csv(
        path=csv_path,
        records=records,
    )
    loaded_records = load_players_csv(csv_path)

    assert loaded_records == records
    assert csv_path.is_file()
    assert not csv_path.with_suffix(
        ".csv.part"
    ).exists()


def test_write_players_csv_rejects_empty_records(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    original_content = "existing content\n"
    csv_path.write_text(
        original_content,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="danh sách player rỗng",
    ):
        write_players_csv(
            path=csv_path,
            records=[],
        )

    assert csv_path.read_text(
        encoding="utf-8",
    ) == original_content


def test_write_players_csv_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "players.csv"
    original_content = "existing content\n"
    csv_path.write_text(
        original_content,
        encoding="utf-8",
    )

    duplicate_records = [
        PlayerRecord(
            player_id="lp_player_123",
            canonical_name="First Player",
            display_name="First Player",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            photo_url=None,
        ),
        PlayerRecord(
            player_id="lp_player_123",
            canonical_name="Duplicate Player",
            display_name="Duplicate Player",
            record_source="leaguepedia_cargo",
            record_source_url=(
                "https://lol.fandom.com/wiki/"
                "Special:Redirect/page/123"
            ),
            photo_url=None,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="player_id trùng",
    ):
        write_players_csv(
            path=csv_path,
            records=duplicate_records,
        )

    assert csv_path.read_text(
        encoding="utf-8",
    ) == original_content


def test_prune_missing_players_preserves_foreign_key_references() -> None:
    stale_player = Player(
        player_id="lp_player_200",
        canonical_name="Caster Only",
        display_name="Caster Only",
        photo_file="assets/players/lp_player_200.png",
    )

    class FakeSession:
        def __init__(self) -> None:
            self.results = [
                {
                    "lp_player_100",
                    "lp_player_200",
                    "lp_player_300",
                },
                {"lp_player_300"},
                set(),
                [stale_player],
            ]
            self.deleted: list[Player] = []
            self.flushed = False

        def scalars(self, statement: object) -> object:
            del statement
            return self.results.pop(0)

        def delete(self, player: Player) -> None:
            self.deleted.append(player)

        def flush(self) -> None:
            self.flushed = True

    session = FakeSession()
    result = prune_missing_leaguepedia_players(
        session=session,  # type: ignore[arg-type]
        keep_player_ids={"lp_player_100"},
    )

    assert result.deleted == 1
    assert result.protected == 1
    assert result.photo_files == (
        "assets/players/lp_player_200.png",
    )
    assert session.deleted == [stale_player]
    assert session.flushed is True


def test_delete_pruned_media_validates_all_paths_before_deleting(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "assets" / "players"
    asset_dir.mkdir(parents=True)
    player_photo = asset_dir / "lp_player_200.png"
    outside_file = tmp_path / "outside.png"
    player_photo.write_bytes(b"player")
    outside_file.write_bytes(b"outside")

    with pytest.raises(
        ValueError,
        match="ngoài assets/players",
    ):
        delete_pruned_player_media(
            project_root=tmp_path,
            relative_paths=(
                "assets/players/lp_player_200.png",
                "outside.png",
            ),
        )

    assert player_photo.is_file()
    assert outside_file.is_file()

    assert delete_pruned_player_media(
        project_root=tmp_path,
        relative_paths=(
            "assets/players/lp_player_200.png",
        ),
    ) == 1
    assert not player_photo.exists()
    assert outside_file.is_file()
