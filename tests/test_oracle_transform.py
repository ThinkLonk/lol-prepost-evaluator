from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from match_insight.data_processing.oracle_etl import (
    ORACLE_CORE_COLUMNS,
    OracleCoreData,
    classify_oracle_rows,
    inspect_position_values,
    normalize_oracle_core_dataframe,
)
from match_insight.data_processing.oracle_identity import (
    ChampionEntityReference,
    OracleIdentityResult,
    PlayerEntityReference,
    TeamEntityReference,
    resolve_oracle_identities,
)
from match_insight.data_processing.oracle_transform import (
    GAME_COLUMNS,
    GAME_PATCH_COLUMNS,
    GAME_PLAYER_COLUMNS,
    GAME_TEAM_COLUMNS,
    GAME_TRANSFORM_BLOCKED,
    GAME_TRANSFORM_PARTIAL,
    GAME_TRANSFORM_READY,
    PLAYER_COLUMNS,
    ROLE_MAP,
    SERIES_BLOCKED,
    SERIES_COLUMNS,
    SERIES_READY,
    SIDE_MAP,
    SOURCE_CHAMPION_UNMAPPED,
    SOURCE_REPORTED,
    TARGET_TABLE_ORDER,
    TEAM_COLUMNS,
    TOURNAMENT_COLUMNS,
    TOURNAMENT_STAGE_COLUMNS,
    OracleDryRunResult,
    OracleTargetSnapshot,
    analyze_series_evidence,
    build_dry_run_summary,
    transform_oracle_targets,
)
from scripts.inspect_oracle_transform import (
    OracleDatabaseSnapshot,
    print_inspection_result,
    validate_reconciliation,
)

SOURCE_POSITIONS = ("top", "jng", "mid", "bot", "sup")

TARGET_FRAME_COLUMNS = {
    "game_patches": GAME_PATCH_COLUMNS,
    "tournaments": TOURNAMENT_COLUMNS,
    "tournament_stages": TOURNAMENT_STAGE_COLUMNS,
    "series": SERIES_COLUMNS,
    "teams": TEAM_COLUMNS,
    "players": PLAYER_COLUMNS,
    "games": GAME_COLUMNS,
    "game_teams": GAME_TEAM_COLUMNS,
    "game_players": GAME_PLAYER_COLUMNS,
}

TABLE_TO_FRAME = {
    "game_patch": "game_patches",
    "tournament": "tournaments",
    "tournament_stage": "tournament_stages",
    "series": "series",
    "team": "teams",
    "player": "players",
    "game": "games",
    "game_team": "game_teams",
    "game_player": "game_players",
}


def make_game_rows(
    *,
    gameid: str = "SERIES-bo3_game_1",
    game_number: int = 1,
    url: str = "https://example.test/game/1",
    league: str = "TEST",
    source_year: int = 2026,
    split: str | None = "Spring",
    playoffs: int = 0,
    date: str = "2025-01-01T12:00:00Z",
    patch: str = "15.10",
) -> list[dict[str, object]]:
    """Create one complete game using only the approved Oracle core columns."""
    rows: list[dict[str, object]] = []
    configurations = (
        ("Blue", "Blue Team", "TEAM-BLUE", 1, 100, 1),
        ("Red", "Red Team", "TEAM-RED", 6, 200, 0),
    )

    for side, team_name, team_id, first_participant, team_participant, result in (
        configurations
    ):
        for offset, position in enumerate(SOURCE_POSITIONS):
            participant_id = first_participant + offset
            rows.append(
                {
                    "gameid": gameid,
                    "datacompleteness": "complete",
                    "url": url,
                    "league": league,
                    "year": source_year,
                    "split": split,
                    "playoffs": playoffs,
                    "date": date,
                    "game": game_number,
                    "patch": patch,
                    "participantid": participant_id,
                    "side": side,
                    "position": position,
                    "playername": f"{side} {position} Player",
                    "playerid": f"PLAYER-{side.upper()}-{position}",
                    "teamname": team_name,
                    "teamid": team_id,
                    "champion": f"Champion-{side}-{position}",
                    "gamelength": 2100,
                    "result": result,
                }
            )

        rows.append(
            {
                "gameid": gameid,
                "datacompleteness": "complete",
                "url": url,
                "league": league,
                "year": source_year,
                "split": split,
                "playoffs": playoffs,
                "date": date,
                "game": game_number,
                "patch": patch,
                "participantid": team_participant,
                "side": side,
                "position": "team",
                "playername": None,
                "playerid": None,
                "teamname": team_name,
                "teamid": team_id,
                "champion": None,
                "gamelength": 2100,
                "result": result,
            }
        )

    return rows


def make_series_rows(game_count: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game_number in range(1, game_count + 1):
        rows.extend(
            make_game_rows(
                gameid=f"SERIES-bo3_game_{game_number}",
                game_number=game_number,
                url=f"https://example.test/game/{game_number}",
                date=f"2025-01-0{game_number}T12:00:00Z",
            )
        )
    return rows


def make_core_data(rows: Sequence[dict[str, object]]) -> OracleCoreData:
    dataframe = pd.DataFrame(rows).loc[:, list(ORACLE_CORE_COLUMNS)]
    normalized = normalize_oracle_core_dataframe(dataframe)
    player_rows, team_rows = classify_oracle_rows(normalized)
    position_counts, missing_position_rows = inspect_position_values(normalized)
    return OracleCoreData(
        dataframe=normalized,
        player_rows=player_rows,
        team_rows=team_rows,
        position_counts=position_counts,
        missing_position_rows=missing_position_rows,
    )


def make_champion_references(
    core_data: OracleCoreData,
    *,
    excluded_names: frozenset[str] = frozenset(),
) -> tuple[ChampionEntityReference, ...]:
    names = sorted(
        {
            str(value)
            for value in core_data.player_rows["champion"].dropna().tolist()
            if str(value) not in excluded_names
        }
    )
    return tuple(
        ChampionEntityReference(
            champion_id=name,
            canonical_name=name,
            display_name=name,
            image_file=f"champions/{name}.png",
        )
        for name in names
    )


def make_identity_result(
    core_data: OracleCoreData,
    *,
    team_references: Sequence[TeamEntityReference] = (),
    player_references: Sequence[PlayerEntityReference] = (),
    excluded_champions: frozenset[str] = frozenset(),
) -> OracleIdentityResult:
    return resolve_oracle_identities(
        core_data,
        team_references=team_references,
        player_references=player_references,
        champion_references=make_champion_references(
            core_data,
            excluded_names=excluded_champions,
        ),
    )


def make_existing_references() -> tuple[
    tuple[TeamEntityReference, ...],
    tuple[PlayerEntityReference, ...],
]:
    team_references = (
        TeamEntityReference(
            team_id="db-team-blue",
            oracle_team_id="TEAM-BLUE",
            canonical_name="Curated Blue",
            display_name="Curated Blue",
            logo_file="teams/blue.png",
        ),
        TeamEntityReference(
            team_id="db-team-red",
            oracle_team_id="TEAM-RED",
            canonical_name="Curated Red",
            display_name="Curated Red",
            logo_file="teams/red.png",
        ),
    )
    player_references = tuple(
        PlayerEntityReference(
            player_id=f"db-player-{side.casefold()}-{position}",
            oracle_player_id=f"PLAYER-{side}-{position}",
            canonical_name=f"Curated {side} {position}",
            display_name=f"Curated {side} {position}",
            photo_file=f"players/{side.casefold()}-{position}.png",
        )
        for side in ("BLUE", "RED")
        for position in SOURCE_POSITIONS
    )
    return team_references, player_references


def snapshot_from_references(
    team_references: Sequence[TeamEntityReference],
    player_references: Sequence[PlayerEntityReference],
) -> OracleTargetSnapshot:
    teams = pd.DataFrame(
        [
            {
                "team_id": item.team_id,
                "oracle_team_id": item.oracle_team_id,
                "canonical_name": item.canonical_name,
                "display_name": item.display_name,
                "logo_file": item.logo_file,
            }
            for item in team_references
        ],
        columns=list(TEAM_COLUMNS),
    )
    players = pd.DataFrame(
        [
            {
                "player_id": item.player_id,
                "oracle_player_id": item.oracle_player_id,
                "canonical_name": item.canonical_name,
                "display_name": item.display_name,
                "photo_file": item.photo_file,
            }
            for item in player_references
        ],
        columns=list(PLAYER_COLUMNS),
    )
    return OracleTargetSnapshot(teams=teams, players=players)


def snapshot_from_result(result: OracleDryRunResult) -> OracleTargetSnapshot:
    records = result.records
    return OracleTargetSnapshot(
        game_patches=records.game_patches.copy(deep=True),
        tournaments=records.tournaments.copy(deep=True),
        tournament_stages=records.tournament_stages.copy(deep=True),
        series=records.series.copy(deep=True),
        teams=records.teams.copy(deep=True),
        players=records.players.copy(deep=True),
        games=records.games.copy(deep=True),
        game_teams=records.game_teams.copy(deep=True),
        game_players=records.game_players.copy(deep=True),
    )


def test_transform_preserves_inputs_and_uses_exact_target_columns() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    source_before = core_data.dataframe.copy(deep=True)
    teams_before = identities.team_participations.copy(deep=True)
    players_before = identities.player_rows.copy(deep=True)

    result = transform_oracle_targets(core_data, identities)

    pd.testing.assert_frame_equal(core_data.dataframe, source_before)
    pd.testing.assert_frame_equal(identities.team_participations, teams_before)
    pd.testing.assert_frame_equal(identities.player_rows, players_before)
    assert tuple(core_data.dataframe.columns) == ORACLE_CORE_COLUMNS
    for frame_name, columns in TARGET_FRAME_COLUMNS.items():
        assert tuple(getattr(result.records, frame_name).columns) == columns


def test_patch_source_year_and_context_ids_are_preserved_and_deterministic() -> None:
    core_data = make_core_data(make_game_rows(source_year=2026, patch="15.10"))
    identities = make_identity_result(core_data)

    first = transform_oracle_targets(core_data, identities)
    second = transform_oracle_targets(core_data, identities)

    assert first.records.game_patches.to_dict("records") == [
        {"patch_id": "15.10", "patch_name": "15.10"}
    ]
    assert first.records.tournaments.iloc[0]["season"] == 2026
    assert first.records.tournaments.iloc[0]["region"] is None
    assert first.records.tournament_stages.iloc[0]["name"] == "Spring"
    assert (
        first.records.tournaments.iloc[0]["tournament_id"]
        == second.records.tournaments.iloc[0]["tournament_id"]
    )
    assert (
        first.records.tournament_stages.iloc[0]["stage_id"]
        == second.records.tournament_stages.iloc[0]["stage_id"]
    )


def test_conflicting_metadata_in_one_game_is_rejected() -> None:
    rows = make_game_rows()
    rows[0]["patch"] = "15.11"
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    assert result.records.games.empty
    metadata_rejections = result.rejected_records.loc[
        result.rejected_records["reason"].eq("GAME_METADATA_INVALID")
    ]
    assert len(metadata_rejections) == 1
    assert "CONFLICTING_PATCH" in str(metadata_rejections.iloc[0]["detail"])
    summary = build_dry_run_summary(result)
    assert summary["game_transform_status"] == GAME_TRANSFORM_BLOCKED
    assert summary["game_metadata_detail_counts"] == {
        "CONFLICTING_PATCH": 1
    }


def test_game_transform_status_is_partial_when_only_valid_games_emit() -> None:
    rows = make_series_rows()
    rows[-1]["patch"] = "15.11"
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)
    summary = build_dry_run_summary(result)

    assert len(result.records.games) == 1
    assert summary["game_transform_status"] == GAME_TRANSFORM_PARTIAL
    assert summary["game_metadata_detail_counts"] == {
        "CONFLICTING_PATCH": 1
    }


def test_series_is_ready_only_with_source_key_and_explicit_best_of() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_READY
    assert analysis.series_records.to_dict("records") == [
        {
            "series_id": "SERIES-bo3",
            "source_series_key": "SERIES-bo3",
            "league": "TEST",
            "year": 2026,
            "split": "Spring",
            "playoffs": 0,
            "best_of": 3,
        }
    ]
    assert set(analysis.game_assignments["series_status"]) == {SERIES_READY}
    game = result.records.games.iloc[0]
    assert game["series_id"] == "SERIES-bo3"
    assert pd.isna(game["stage_id"])


def test_series_accepts_controlled_url_key_and_explicit_best_of() -> None:
    core_data = make_core_data(
        make_game_rows(
            gameid="GAME-001",
            url=(
                "https://example.test/game?"
                "matchid=SERIES-URL&bestof=3"
            ),
        )
    )
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)

    assert analysis.status == SERIES_READY
    assert analysis.series_records.iloc[0]["series_id"] == (
        "example.test?matchid=SERIES-URL"
    )
    assert analysis.series_records.iloc[0]["best_of"] == 3


@pytest.mark.parametrize(
    ("replacement_url", "expected_reason"),
    (
        (None, "URL_MIXED_MISSING"),
        ("https://other.test/game/1", "URL_VALUES_CONFLICT"),
    ),
)
def test_inconsistent_optional_url_blocks_only_series_evidence(
    replacement_url: str | None,
    expected_reason: str,
) -> None:
    rows = make_game_rows()
    rows[0]["url"] = replacement_url
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert expected_reason in str(analysis.diagnostics.iloc[0]["reason"])
    assert len(result.records.games) == 1
    game = result.records.games.iloc[0]
    assert pd.isna(game["series_id"])
    assert pd.notna(game["stage_id"])
    assert "GAME_METADATA_INVALID" not in set(
        result.rejected_records["reason"]
    )


def test_malformed_url_is_invalid_series_evidence_but_game_still_emits() -> None:
    core_data = make_core_data(
        make_game_rows(url="https://[invalid")
    )
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert "INVALID_URL_EVIDENCE" in str(
        analysis.diagnostics.iloc[0]["reason"]
    )
    assert len(result.records.games) == 1
    game = result.records.games.iloc[0]
    assert pd.isna(game["series_id"])
    assert pd.notna(game["stage_id"])
    assert "GAME_METADATA_INVALID" not in set(
        result.rejected_records["reason"]
    )


@pytest.mark.parametrize(
    ("gameid", "expected_reason"),
    (
        ("SERIES-A_game_1", "EXPLICIT_BEST_OF_MISSING"),
        ("GAME-bo3", "SOURCE_SERIES_KEY_MISSING"),
    ),
)
def test_series_without_complete_explicit_evidence_is_blocked(
    gameid: str,
    expected_reason: str,
) -> None:
    core_data = make_core_data(make_game_rows(gameid=gameid))
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert analysis.game_assignments.iloc[0]["series_status"] == SERIES_BLOCKED
    assert expected_reason in str(analysis.game_assignments.iloc[0]["series_reason"])
    assert result.records.series.empty
    assert len(result.records.games) == 1
    game = result.records.games.iloc[0]
    assert pd.isna(game["series_id"])
    assert pd.notna(game["stage_id"])
    assert len(result.records.game_teams) == 2
    assert len(result.records.game_players) == 10
    assert "GAME_BLOCKED_BY_SERIES_ANALYSIS" not in set(
        result.rejected_records["reason"]
    )


def test_ready_series_is_kept_when_another_game_has_no_series_evidence() -> None:
    rows = make_game_rows()
    rows.extend(
        make_game_rows(
            gameid="GAME-002",
            game_number=2,
            url="https://example.test/game/2",
            date="2025-01-02T12:00:00Z",
        )
    )
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records["series_id"].tolist() == ["SERIES-bo3"]
    assert result.records.series["series_id"].tolist() == ["SERIES-bo3"]
    games = result.records.games.set_index("game_id")
    assert games.loc["SERIES-bo3_game_1", "series_id"] == "SERIES-bo3"
    assert pd.isna(games.loc["SERIES-bo3_game_1", "stage_id"])
    assert pd.isna(games.loc["GAME-002", "series_id"])
    assert pd.notna(games.loc["GAME-002", "stage_id"])
    summary = build_dry_run_summary(result)
    assert summary["series_status"] == SERIES_BLOCKED
    assert summary["game_transform_status"] == GAME_TRANSFORM_READY


def test_conflicting_explicit_series_evidence_is_blocked() -> None:
    core_data = make_core_data(
        make_game_rows(
            url="https://example.test/game/1?bestof=5",
        )
    )
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert "EXPLICIT_BEST_OF_CONFLICT" in str(
        analysis.game_assignments.iloc[0]["series_reason"]
    )
    assert result.records.series.empty
    assert len(result.records.games) == 1
    game = result.records.games.iloc[0]
    assert pd.isna(game["series_id"])
    assert pd.notna(game["stage_id"])
    assert not result.unresolved_records.loc[
        result.unresolved_records["entity"].eq("series")
    ].empty
    cascade_reasons = {
        "GAME_BLOCKED_BY_SERIES_ANALYSIS",
        "GAME_TEAM_BLOCKED_BY_GAME",
        "GAME_PLAYER_BLOCKED_BY_GAME",
    }
    assert cascade_reasons.isdisjoint(result.rejected_records["reason"])
    summary = build_dry_run_summary(result)
    assert summary["series_status"] == SERIES_BLOCKED
    assert summary["game_transform_status"] == GAME_TRANSFORM_READY


def test_url_identifier_cannot_point_to_multiple_source_series() -> None:
    rows = make_game_rows(
        gameid="SERIES-A-bo3_game_1",
        url="https://example.test/game?matchid=SHARED&bestof=3&game=1",
    )
    rows.extend(
        make_game_rows(
            gameid="SERIES-B-bo3_game_1",
            url="https://example.test/game?matchid=SHARED&bestof=3&game=2",
        )
    )
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert set(analysis.game_assignments["series_status"]) == {SERIES_BLOCKED}
    assert "URL_TO_SERIES_KEY_CONFLICT" in set(analysis.diagnostics["reason"])
    assert len(result.records.games) == 2
    assert result.records.games["series_id"].isna().all()
    assert result.records.games["stage_id"].notna().all()


def test_normalized_url_whitespace_still_detects_cross_game_conflict() -> None:
    rows = make_game_rows(
        gameid="SERIES-A-bo3_game_1",
        url="  https://example.test/game/shared  ",
    )
    rows.extend(
        make_game_rows(
            gameid="SERIES-B-bo3_game_1",
            url="\thttps://example.test/game/shared\n",
        )
    )
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    assert all(
        "URL_TO_GAMEID_CONFLICT" in str(reason)
        for reason in analysis.diagnostics["reason"]
    )
    assert len(result.records.games) == 2
    assert result.records.games["series_id"].isna().all()
    assert result.records.games["stage_id"].notna().all()
    assert "GAME_METADATA_INVALID" not in set(
        result.rejected_records["reason"]
    )


def test_exact_url_cannot_point_to_multiple_gameids_in_one_series() -> None:
    shared_url = "https://example.test/game/shared"
    rows = make_game_rows(
        gameid="SERIES-bo3_game_1",
        game_number=1,
        url=shared_url,
    )
    rows.extend(
        make_game_rows(
            gameid="SERIES-bo3_game_2",
            game_number=2,
            url=shared_url,
        )
    )
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    analysis = analyze_series_evidence(core_data, identities)
    result = transform_oracle_targets(
        core_data,
        identities,
        series_analysis=analysis,
    )

    assert analysis.status == SERIES_BLOCKED
    assert analysis.series_records.empty
    reasons = ",".join(analysis.game_assignments["series_reason"].astype(str))
    assert "URL_TO_GAMEID_CONFLICT" in reasons
    assert len(result.records.games) == 2
    assert result.records.games["series_id"].isna().all()
    assert result.records.games["stage_id"].notna().all()


def test_game_records_preserve_time_winner_sides_roles_and_player_champions() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    game = result.records.games.iloc[0]
    assert game["game_id"] == "SERIES-bo3_game_1"
    assert game["series_id"] == "SERIES-bo3"
    assert pd.isna(game["stage_id"])
    assert game["scheduled_at"] == pd.Timestamp("2025-01-01T12:00:00Z")
    assert game["started_at"] is None
    assert game["ended_at"] is None
    assert game["winner_team_id"] == "TEAM-BLUE"
    assert set(result.records.game_teams["side"]) == {"BLUE", "RED"}
    assert set(result.records.game_teams["confirmation_status"]) == {
        SOURCE_REPORTED
    }
    assert set(result.records.game_players["role"]) == set(ROLE_MAP.values())
    assert len(result.records.game_players) == 10
    assert set(result.records.game_players["confirmation_status"]) == {
        SOURCE_REPORTED
    }
    for _, side_rows in result.records.game_players.groupby("side"):
        assert set(side_rows["role"]) == set(ROLE_MAP.values())

    expected_champions = {
        (
            SIDE_MAP[str(row.side).casefold()],
            ROLE_MAP[str(row.position).casefold()],
        ): str(row.champion_id)
        for row in identities.player_rows.itertuples(index=False)
    }
    actual_champions = {
        (str(row.side), str(row.role)): str(row.champion_id)
        for row in result.records.game_players.itertuples(index=False)
    }
    assert actual_champions == expected_champions


def test_unresolved_team_is_reported_and_rejects_the_game() -> None:
    rows = make_game_rows()
    for row in rows:
        if row["side"] == "Blue":
            row["teamid"] = None
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    assert result.records.games.empty
    assert not result.unresolved_records.loc[
        result.unresolved_records["entity"].eq("team_participation")
    ].empty
    assert not result.unresolved_records.loc[
        result.unresolved_records["entity"].eq("series")
    ].empty


def test_unresolved_player_is_reported_without_hiding_the_row_rejection() -> None:
    rows = make_game_rows()
    for row in rows:
        if row["side"] == "Blue" and row["position"] == "top":
            row["playerid"] = None
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    assert len(result.records.games) == 1
    assert len(result.records.game_players) == 5
    assert not result.unresolved_records.loc[
        result.unresolved_records["entity"].eq("player_row")
    ].empty
    assert "GAME_TEAM_ROLES_INCOMPLETE" in set(
        result.rejected_records["reason"]
    )
    assert "GAME_PLAYER_BLOCKED_BY_INCOMPLETE_ROLE_SET" in set(
        result.rejected_records["reason"]
    )


def test_unmapped_champion_is_explicit_and_keeps_player_participation() -> None:
    missing_name = "Champion-Blue-top"
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(
        core_data,
        excluded_champions=frozenset({missing_name}),
    )

    result = transform_oracle_targets(core_data, identities)

    assert result.unmapped_champions.to_dict("records") == [
        {
            "gameid": "SERIES-bo3_game_1",
            "side": "Blue",
            "position": "top",
            "champion": missing_name,
            "reason": "NO_CHAMPION_MATCH",
        }
    ]
    player = result.records.game_players.loc[
        result.records.game_players["side"].eq("BLUE")
        & result.records.game_players["role"].eq("TOP")
    ].iloc[0]
    assert pd.isna(player["champion_id"])
    assert player["confirmation_status"] == SOURCE_CHAMPION_UNMAPPED


def test_new_entities_use_exact_oracle_ids_as_target_primary_keys() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    assert set(result.records.teams["team_id"]) == {"TEAM-BLUE", "TEAM-RED"}
    assert set(result.records.teams["team_id"]) == set(
        result.records.teams["oracle_team_id"]
    )
    assert set(result.records.players["player_id"]) == set(
        result.records.players["oracle_player_id"]
    )
    assert result.records.teams["display_name"].notna().all()
    assert result.records.players["display_name"].notna().all()


def test_conflicting_new_entity_names_are_rejected_without_selecting_one() -> None:
    rows = make_series_rows()
    for row in rows:
        if row["game"] == 2 and row["side"] == "Blue":
            row["teamname"] = "Renamed Blue Team"
        if (
            row["game"] == 2
            and row["side"] == "Red"
            and row["position"] == "top"
        ):
            row["playername"] = "Renamed Red Top"
    core_data = make_core_data(rows)
    identities = make_identity_result(core_data)

    result = transform_oracle_targets(core_data, identities)

    conflicts = result.rejected_records.loc[
        result.rejected_records["reason"].eq("SOURCE_NAME_CONFLICT")
    ]
    assert set(conflicts["entity"]) == {"team", "player"}
    assert "TEAM-BLUE" not in set(result.records.teams["team_id"])
    assert "PLAYER-RED-top" not in set(result.records.players["player_id"])


@pytest.mark.parametrize("entity", ("team", "player"))
def test_target_primary_key_collision_is_rejected(entity: str) -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    if entity == "team":
        snapshot = OracleTargetSnapshot(
            teams=pd.DataFrame(
                [
                    {
                        "team_id": "TEAM-BLUE",
                        "oracle_team_id": "DIFFERENT-TEAM",
                        "canonical_name": "Occupied",
                        "display_name": "Occupied",
                        "logo_file": "teams/occupied.png",
                    }
                ],
                columns=list(TEAM_COLUMNS),
            )
        )
        colliding_id = "TEAM-BLUE"
        frame_name = "teams"
        id_column = "team_id"
    else:
        snapshot = OracleTargetSnapshot(
            players=pd.DataFrame(
                [
                    {
                        "player_id": "PLAYER-BLUE-top",
                        "oracle_player_id": "DIFFERENT-PLAYER",
                        "canonical_name": "Occupied",
                        "display_name": "Occupied",
                        "photo_file": "players/occupied.png",
                    }
                ],
                columns=list(PLAYER_COLUMNS),
            )
        )
        colliding_id = "PLAYER-BLUE-top"
        frame_name = "players"
        id_column = "player_id"

    result = transform_oracle_targets(
        core_data,
        identities,
        target_snapshot=snapshot,
    )

    collisions = result.rejected_records.loc[
        result.rejected_records["reason"].eq("TARGET_PRIMARY_KEY_COLLISION")
    ]
    assert entity in set(collisions["entity"])
    assert colliding_id not in set(getattr(result.records, frame_name)[id_column])


def test_existing_entity_names_and_media_are_preserved_and_skipped() -> None:
    core_data = make_core_data(make_game_rows())
    team_references, player_references = make_existing_references()
    identities = make_identity_result(
        core_data,
        team_references=team_references,
        player_references=player_references,
    )
    snapshot = snapshot_from_references(team_references, player_references)
    teams_before = snapshot.teams.copy(deep=True)
    players_before = snapshot.players.copy(deep=True)

    result = transform_oracle_targets(
        core_data,
        identities,
        target_snapshot=snapshot,
    )

    assert set(result.records.teams["display_name"]) == {
        "Curated Blue",
        "Curated Red",
    }
    assert set(result.records.teams["logo_file"]) == {
        "teams/blue.png",
        "teams/red.png",
    }
    assert result.records.players["display_name"].str.startswith("Curated ").all()
    assert result.records.players["photo_file"].notna().all()
    pd.testing.assert_frame_equal(snapshot.teams, teams_before)
    pd.testing.assert_frame_equal(snapshot.players, players_before)
    entity_actions = result.actions.loc[
        result.actions["table"].isin(["team", "player"])
    ]
    assert set(entity_actions["action"]) == {"SKIP"}


def test_snapshot_oracle_id_recovers_existing_target_and_media_fail_closed() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    existing_team = {
        "team_id": "db-team-blue",
        "oracle_team_id": "TEAM-BLUE",
        "canonical_name": "Curated Blue",
        "display_name": "Curated Blue",
        "logo_file": "teams/blue.png",
    }
    existing_player = {
        "player_id": "db-player-blue-top",
        "oracle_player_id": "PLAYER-BLUE-top",
        "canonical_name": "Curated Blue Top",
        "display_name": "Curated Blue Top",
        "photo_file": "players/blue-top.png",
    }
    snapshot = OracleTargetSnapshot(
        teams=pd.DataFrame([existing_team], columns=list(TEAM_COLUMNS)),
        players=pd.DataFrame([existing_player], columns=list(PLAYER_COLUMNS)),
    )

    result = transform_oracle_targets(
        core_data,
        identities,
        target_snapshot=snapshot,
    )

    team = result.records.teams.loc[
        result.records.teams["team_id"].eq("db-team-blue")
    ].iloc[0]
    player = result.records.players.loc[
        result.records.players["player_id"].eq("db-player-blue-top")
    ].iloc[0]
    assert team.to_dict() == existing_team
    assert player.to_dict() == existing_player


def test_incomplete_snapshot_schema_is_rejected_before_reconciliation() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    snapshot = OracleTargetSnapshot(
        teams=pd.DataFrame([{"team_id": "unrelated-target"}])
    )

    with pytest.raises(ValueError, match="missing source-ID columns"):
        transform_oracle_targets(
            core_data,
            identities,
            target_snapshot=snapshot,
        )


def test_game_team_side_swap_is_rejected_before_action_forecast() -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    first = transform_oracle_targets(core_data, identities)
    snapshot = snapshot_from_result(first)
    snapshot.game_teams.loc[
        snapshot.game_teams["side"].eq("BLUE"), "side"
    ] = "TEMP"
    snapshot.game_teams.loc[
        snapshot.game_teams["side"].eq("RED"), "side"
    ] = "BLUE"
    snapshot.game_teams.loc[
        snapshot.game_teams["side"].eq("TEMP"), "side"
    ] = "RED"

    with pytest.raises(ValueError, match="alternate unique key"):
        transform_oracle_targets(
            core_data,
            identities,
            target_snapshot=snapshot,
        )


def test_dynamic_reconciliation_matches_records_actions_and_source_grains() -> None:
    core_data = make_core_data(make_series_rows())
    identities = make_identity_result(core_data)
    first = transform_oracle_targets(core_data, identities)
    second = transform_oracle_targets(
        core_data,
        identities,
        target_snapshot=snapshot_from_result(first),
    )

    summary = build_dry_run_summary(second)

    assert summary["source_rows"] == len(core_data.dataframe)
    assert summary["player_rows"] == len(core_data.player_rows)
    assert summary["team_rows"] == len(core_data.team_rows)
    assert summary["distinct_games"] == core_data.dataframe["gameid"].nunique()
    assert summary["series_status"] == SERIES_READY
    assert summary["game_transform_status"] == GAME_TRANSFORM_READY
    assert summary["game_metadata_detail_counts"] == {}
    assert summary["postgresql_changed"] is False
    assert summary["transaction_status"] == "NOT_OPENED"

    expected_skipped = 0
    for table in TARGET_TABLE_ORDER:
        frame = getattr(second.records, TABLE_TO_FRAME[table])
        expected_count = len(frame)
        expected_skipped += expected_count
        assert summary["transformed_records"][table] == expected_count
        assert summary["actions"][table] == {
            "inserted_expected": 0,
            "updated_expected": 0,
            "skipped": expected_count,
        }

    assert summary["skipped"] == expected_skipped
    assert len(second.skipped_records) == expected_skipped
    assert summary["unresolved"] == len(second.unresolved_records)
    assert summary["rejected"] == len(second.rejected_records)
    assert summary["unmapped_champions"] == len(second.unmapped_champions)
    validate_reconciliation(second, dict(summary))


@pytest.mark.parametrize("parent_state", ("BOTH", "NEITHER"))
def test_reconciliation_requires_exactly_one_game_parent(
    parent_state: str,
) -> None:
    core_data = make_core_data(make_game_rows())
    identities = make_identity_result(core_data)
    result = transform_oracle_targets(core_data, identities)
    games = result.records.games.copy(deep=True)
    if parent_state == "BOTH":
        games.loc[:, "stage_id"] = result.records.tournament_stages.iloc[0][
            "stage_id"
        ]
    else:
        games.loc[:, ["series_id", "stage_id"]] = None
    invalid_result = replace(
        result,
        records=replace(result.records, games=games),
    )
    summary = dict(build_dry_run_summary(invalid_result))

    with pytest.raises(ValueError, match="game_parent_xor"):
        validate_reconciliation(invalid_result, summary)


def test_cli_does_not_block_valid_direct_stage_games_by_series_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    core_data = make_core_data(make_game_rows(gameid="GAME-bo3"))
    identities = make_identity_result(core_data)
    result = transform_oracle_targets(core_data, identities)
    summary = dict(build_dry_run_summary(result))
    snapshot = OracleDatabaseSnapshot(
        targets=OracleTargetSnapshot(),
        champions=pd.DataFrame([{"champion_id": "fixture"}]),
    )

    validate_reconciliation(result, summary)
    print_inspection_result(
        source_path=Path("oracle-fixture.csv"),
        source_sha256_before="fixture-sha256",
        source_sha256_after="fixture-sha256",
        snapshot=snapshot,
        result=result,
        summary=summary,
    )

    output = capsys.readouterr().out
    assert "series_status=BLOCKED\n" in output
    assert "game_transform_status=READY\n" in output
    assert "game_metadata_detail_counts={}\n" in output
    assert "transform_ready=true\n" in output
    assert "dry_run_status=READY_WITH_ISSUES\n" in output
