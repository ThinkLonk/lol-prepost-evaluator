from __future__ import annotations

from collections.abc import Sequence

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
    CHAMPION_MAPPED,
    CHAMPION_UNMAPPED,
    RECOVERED_UNIQUE,
    SOURCE_ID,
    TARGET_CONFLICT,
    TARGET_EXISTING,
    TARGET_NEW_REQUIRED,
    UNRESOLVED,
    ChampionEntityReference,
    PlayerEntityReference,
    TeamEntityReference,
    build_identity_summary,
    map_oracle_champions,
    resolve_oracle_identities,
    resolve_player_identities,
    resolve_team_identities,
)


def make_row(**overrides: object) -> dict[str, object]:
    """Tạo một source row đủ 20 cột Bước 6B."""
    row: dict[str, object] = {
        "gameid": "GAME-001",
        "datacompleteness": "complete",
        "url": "https://example.test/game/1",
        "league": "TEST",
        "year": 2025,
        "split": "Spring",
        "playoffs": 0,
        "date": "2025-01-01T12:00:00Z",
        "game": 1,
        "patch": "15.10",
        "participantid": 1,
        "side": "Blue",
        "position": "top",
        "playername": "Player One",
        "playerid": "PLAYER-1",
        "teamname": "Team One",
        "teamid": "TEAM-1",
        "champion": "Aatrox",
        "gamelength": 2100,
        "result": 1,
    }
    row.update(overrides)
    return row


def make_frame(
    rows: Sequence[dict[str, object]],
) -> pd.DataFrame:
    """Chuẩn hóa fixture bằng đúng contract Bước 6B."""
    dataframe = pd.DataFrame(rows).loc[
        :,
        list(ORACLE_CORE_COLUMNS),
    ]
    return normalize_oracle_core_dataframe(
        dataframe
    )


def make_core_data(
    rows: Sequence[dict[str, object]],
) -> OracleCoreData:
    """Tạo OracleCoreData mà không đọc CSV hoặc database."""
    dataframe = make_frame(rows)
    player_rows, team_rows = classify_oracle_rows(
        dataframe
    )
    position_counts, missing_rows = (
        inspect_position_values(dataframe)
    )
    return OracleCoreData(
        dataframe=dataframe,
        player_rows=player_rows,
        team_rows=team_rows,
        position_counts=position_counts,
        missing_position_rows=missing_rows,
    )


def test_team_uses_single_source_id_for_game_side() -> None:
    dataframe = make_frame(
        [
            make_row(teamid="OE-Team_01"),
            make_row(
                participantid=100,
                position="team",
                playerid=None,
                playername=None,
                champion=None,
                teamid="OE-Team_01",
            ),
        ]
    )

    resolved = resolve_team_identities(dataframe)

    assert len(resolved) == 1
    assert resolved.at[0, "resolved_oracle_team_id"] == (
        "OE-Team_01"
    )
    assert resolved.at[0, "team_resolution_method"] == (
        SOURCE_ID
    )


def test_empty_team_source_is_rejected() -> None:
    empty = pd.DataFrame(
        columns=("gameid", "side", "teamid", "teamname")
    )

    with pytest.raises(
        ValueError,
        match="Oracle source rows cannot be empty",
    ):
        resolve_team_identities(empty)


def test_team_recovers_only_from_unique_canonical_name() -> None:
    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-SEED",
                teamid="TEAM-UNIQUE",
                teamname="Téam One",
            ),
            make_row(
                gameid="GAME-MISSING",
                teamid=None,
                teamname="  TÉAM   ONE  ",
            ),
        ]
    )

    resolved = resolve_team_identities(dataframe)
    recovered = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert recovered["resolved_oracle_team_id"] == (
        "TEAM-UNIQUE"
    )
    assert recovered["team_resolution_method"] == (
        RECOVERED_UNIQUE
    )
    assert recovered["team_resolution_reason"] == (
        "UNIQUE_CANONICAL_NAME"
    )


def test_team_source_id_conflict_is_unresolved() -> None:
    dataframe = make_frame(
        [
            make_row(teamid="TEAM-A"),
            make_row(
                participantid=2,
                position="jng",
                teamid="TEAM-B",
            ),
        ]
    )

    resolved = resolve_team_identities(dataframe)

    assert pd.isna(
        resolved.at[0, "resolved_oracle_team_id"]
    )
    assert resolved.at[0, "team_resolution_method"] == (
        UNRESOLVED
    )
    assert resolved.at[0, "team_resolution_reason"] == (
        "SOURCE_ID_CONFLICT"
    )


@pytest.mark.parametrize(
    ("include_ambiguous_evidence", "expected_reason"),
    [
        (True, "AMBIGUOUS_SOURCE_ID_CANDIDATES"),
        (False, "NO_SOURCE_ID_CANDIDATE"),
    ],
)
def test_team_without_unique_name_candidate_is_unresolved(
    include_ambiguous_evidence: bool,
    expected_reason: str,
) -> None:
    rows = [
        make_row(
            gameid="GAME-MISSING",
            teamid=None,
            teamname=(
                "Shared Team"
                if include_ambiguous_evidence
                else "Unknown Team"
            ),
        )
    ]
    if include_ambiguous_evidence:
        rows.extend(
            [
                make_row(
                    gameid="GAME-A",
                    teamid="TEAM-A",
                    teamname="Shared Team",
                ),
                make_row(
                    gameid="GAME-B",
                    teamid="TEAM-B",
                    teamname="Shared Team",
                ),
            ]
        )

    resolved = resolve_team_identities(
        make_frame(rows)
    )
    missing_group = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert missing_group["team_resolution_method"] == (
        UNRESOLVED
    )
    assert missing_group["team_resolution_reason"] == (
        expected_reason
    )


def test_player_source_id_has_priority_and_is_preserved() -> None:
    dataframe = make_frame(
        [
            make_row(
                playerid="OE-Player_01",
                playername="Shared Handle",
            ),
            make_row(
                gameid="GAME-002",
                playerid="OTHER-PLAYER",
                playername="Shared Handle",
            ),
        ]
    )
    teams = resolve_team_identities(dataframe)

    resolved = resolve_player_identities(
        dataframe,
        teams,
    )

    assert resolved.at[0, "resolved_oracle_player_id"] == (
        "OE-Player_01"
    )
    assert resolved.at[0, "player_resolution_method"] == (
        SOURCE_ID
    )


def test_player_recovers_from_unique_full_context() -> None:
    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-SEED",
                playerid="PLAYER-UNIQUE",
                playername="Player Shared",
            ),
            make_row(
                gameid="GAME-MISSING",
                playerid=None,
                playername="  PLAYER   SHARED ",
            ),
        ]
    )
    teams = resolve_team_identities(dataframe)

    resolved = resolve_player_identities(
        dataframe,
        teams,
    )
    recovered = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert recovered["resolved_oracle_player_id"] == (
        "PLAYER-UNIQUE"
    )
    assert recovered["player_resolution_method"] == (
        RECOVERED_UNIQUE
    )
    assert recovered["player_resolution_reason"] == (
        "UNIQUE_FULL_CONTEXT"
    )


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("teamid", "TEAM-OTHER"),
        ("position", "jng"),
        ("league", "OTHER-LEAGUE"),
        ("year", 2026),
        ("split", "Summer"),
    ],
)
def test_player_different_context_is_not_merged(
    field_name: str,
    different_value: object,
) -> None:
    missing_row = make_row(
        gameid="GAME-MISSING",
        playerid=None,
        playername="Player Shared",
    )
    missing_row[field_name] = different_value
    if field_name == "teamid":
        missing_row["teamname"] = "Other Team"

    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-SEED",
                playerid="PLAYER-SEED",
                playername="Player Shared",
            ),
            missing_row,
        ]
    )
    teams = resolve_team_identities(dataframe)

    resolved = resolve_player_identities(
        dataframe,
        teams,
    )
    missing_result = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert pd.isna(
        missing_result["resolved_oracle_player_id"]
    )
    assert missing_result["player_resolution_method"] == (
        UNRESOLVED
    )
    assert missing_result["player_resolution_reason"] == (
        "NO_SOURCE_ID_CANDIDATE"
    )


def test_player_full_context_ambiguity_is_unresolved() -> None:
    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-A",
                playerid="PLAYER-A",
                playername="Shared Player",
            ),
            make_row(
                gameid="GAME-B",
                playerid="PLAYER-B",
                playername="Shared Player",
            ),
            make_row(
                gameid="GAME-MISSING",
                playerid=None,
                playername="Shared Player",
            ),
        ]
    )
    teams = resolve_team_identities(dataframe)

    resolved = resolve_player_identities(
        dataframe,
        teams,
    )
    missing_result = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert missing_result["player_resolution_method"] == (
        UNRESOLVED
    )
    assert missing_result["player_resolution_reason"] == (
        "AMBIGUOUS_SOURCE_ID_CANDIDATES"
    )
    assert missing_result[
        "player_resolution_candidates"
    ] == ("PLAYER-A", "PLAYER-B")


def test_player_with_unresolved_team_cannot_be_recovered() -> None:
    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-SEED",
                teamid="TEAM-SEED",
                teamname="Seed Team",
                playerid="PLAYER-SEED",
                playername="Shared Player",
            ),
            make_row(
                gameid="GAME-MISSING",
                teamid=None,
                teamname="Unknown Team",
                playerid=None,
                playername="Shared Player",
            ),
        ]
    )
    teams = resolve_team_identities(dataframe)

    resolved = resolve_player_identities(
        dataframe,
        teams,
    )
    missing_result = resolved.loc[
        resolved["gameid"].eq("GAME-MISSING")
    ].iloc[0]

    assert missing_result["player_resolution_method"] == (
        UNRESOLVED
    )
    assert missing_result["player_resolution_reason"] == (
        "TEAM_UNRESOLVED"
    )


def test_target_mapping_requires_exact_oracle_id() -> None:
    dataframe = make_frame(
        [
            make_row(
                gameid="GAME-EXACT",
                teamid="TEAM-EXACT",
                teamname="Exact Team",
                playerid="PLAYER-EXACT",
                playername="Exact Player",
            ),
            make_row(
                gameid="GAME-NAME",
                teamid="TEAM-NAME-ONLY",
                teamname="Name Only Team",
                playerid="PLAYER-NAME-ONLY",
                playername="Name Only Player",
            ),
        ]
    )
    team_references = (
        TeamEntityReference(
            team_id="target-team-exact",
            oracle_team_id="TEAM-EXACT",
            canonical_name="Different Name",
            display_name="Different Name",
            logo_file="assets/teams/exact.png",
        ),
        TeamEntityReference(
            team_id="target-team-name",
            oracle_team_id=None,
            canonical_name="Name Only Team",
            display_name="Name Only Team",
            logo_file="assets/teams/name.png",
        ),
    )
    player_references = (
        PlayerEntityReference(
            player_id="target-player-exact",
            oracle_player_id="PLAYER-EXACT",
            canonical_name="Different Name",
            display_name="Different Name",
            photo_file="assets/players/exact.png",
        ),
        PlayerEntityReference(
            player_id="target-player-name",
            oracle_player_id=None,
            canonical_name="Name Only Player",
            display_name="Name Only Player",
            photo_file="assets/players/name.png",
        ),
    )

    teams = resolve_team_identities(
        dataframe,
        references=team_references,
    )
    players = resolve_player_identities(
        dataframe,
        teams,
        references=player_references,
    )
    exact_team = teams.loc[
        teams["gameid"].eq("GAME-EXACT")
    ].iloc[0]
    name_team = teams.loc[
        teams["gameid"].eq("GAME-NAME")
    ].iloc[0]
    exact_player = players.loc[
        players["gameid"].eq("GAME-EXACT")
    ].iloc[0]
    name_player = players.loc[
        players["gameid"].eq("GAME-NAME")
    ].iloc[0]

    assert exact_team["target_team_id"] == (
        "target-team-exact"
    )
    assert exact_team["team_target_status"] == (
        TARGET_EXISTING
    )
    assert name_team["team_target_status"] == (
        TARGET_NEW_REQUIRED
    )
    assert pd.isna(name_team["target_team_id"])
    assert exact_player["target_player_id"] == (
        "target-player-exact"
    )
    assert exact_player["player_target_status"] == (
        TARGET_EXISTING
    )
    assert name_player["player_target_status"] == (
        TARGET_NEW_REQUIRED
    )
    assert pd.isna(name_player["target_player_id"])


def test_duplicate_target_oracle_id_is_a_conflict() -> None:
    dataframe = make_frame([make_row()])
    team_references = (
        TeamEntityReference(
            "team-a",
            "TEAM-1",
            "Team A",
            "Team A",
        ),
        TeamEntityReference(
            "team-b",
            "TEAM-1",
            "Team B",
            "Team B",
        ),
    )
    player_references = (
        PlayerEntityReference(
            "player-a",
            "PLAYER-1",
            "Player A",
            "Player A",
        ),
        PlayerEntityReference(
            "player-b",
            "PLAYER-1",
            "Player B",
            "Player B",
        ),
    )

    teams = resolve_team_identities(
        dataframe,
        references=team_references,
    )
    players = resolve_player_identities(
        dataframe,
        teams,
        references=player_references,
    )

    assert teams.at[0, "team_resolution_method"] == (
        SOURCE_ID
    )
    assert teams.at[0, "team_target_status"] == (
        TARGET_CONFLICT
    )
    assert pd.isna(teams.at[0, "target_team_id"])
    assert players.at[0, "player_resolution_method"] == (
        SOURCE_ID
    )
    assert players.at[0, "player_target_status"] == (
        TARGET_CONFLICT
    )
    assert pd.isna(players.at[0, "target_player_id"])


def test_champion_mapping_uses_unique_canonical_or_display_name() -> None:
    dataframe = make_frame(
        [
            make_row(champion="Aatrox"),
            make_row(
                gameid="GAME-002",
                champion="Wukong",
            ),
        ]
    )
    references = (
        ChampionEntityReference(
            champion_id="266",
            canonical_name="Aatrox",
            display_name="Aatrox",
        ),
        ChampionEntityReference(
            champion_id="62",
            canonical_name="MonkeyKing",
            display_name="Wukong",
        ),
    )

    mapped = map_oracle_champions(
        dataframe,
        references,
    )

    assert mapped["champion_id"].tolist() == [
        "266",
        "62",
    ]
    assert mapped["champion_mapping_status"].eq(
        CHAMPION_MAPPED
    ).all()


def test_unmapped_champions_and_inputs_are_preserved() -> None:
    core_data = make_core_data(
        [
            make_row(
                participantid=1,
                position="top",
                playerid="PLAYER-1",
                champion="Unknown Champion",
            ),
            make_row(
                participantid=2,
                position="jng",
                playerid="PLAYER-2",
                champion="Shared Champion",
            ),
            make_row(
                participantid=3,
                position="mid",
                playerid="PLAYER-3",
                champion=None,
            ),
        ]
    )
    dataframe_before = core_data.dataframe.copy(
        deep=True
    )
    player_rows_before = core_data.player_rows.copy(
        deep=True
    )
    team_references = (
        TeamEntityReference(
            "target-team",
            "TEAM-1",
            "Team One",
            "Team One",
            "assets/teams/team.png",
        ),
    )
    player_references = tuple(
        PlayerEntityReference(
            f"target-player-{index}",
            f"PLAYER-{index}",
            f"Player {index}",
            f"Player {index}",
            f"assets/players/player-{index}.png",
        )
        for index in range(1, 4)
    )
    champion_references = (
        ChampionEntityReference(
            "1",
            "Shared Champion",
            "First Shared",
            "assets/champions/1.png",
        ),
        ChampionEntityReference(
            "2",
            "Second Shared",
            "Shared Champion",
            "assets/champions/2.png",
        ),
    )
    references_before = (
        team_references,
        player_references,
        champion_references,
    )

    result = resolve_oracle_identities(
        core_data,
        team_references=team_references,
        player_references=player_references,
        champion_references=champion_references,
    )
    summary = build_identity_summary(result)

    pd.testing.assert_frame_equal(
        core_data.dataframe,
        dataframe_before,
    )
    pd.testing.assert_frame_equal(
        core_data.player_rows,
        player_rows_before,
    )
    assert references_before == (
        team_references,
        player_references,
        champion_references,
    )
    assert result.player_rows[
        "champion_mapping_status"
    ].eq(CHAMPION_UNMAPPED).all()
    assert result.player_rows[
        "champion_mapping_reason"
    ].tolist() == [
        "NO_CHAMPION_MATCH",
        "AMBIGUOUS_CHAMPION_MATCH",
        "MISSING_SOURCE_NAME",
    ]
    assert sum(
        summary["team_resolution_counts"].values()
    ) == len(result.team_participations)
    assert sum(
        summary["player_resolution_counts"].values()
    ) == len(result.player_rows)
    assert sum(
        summary["champion_mapping_counts"].values()
    ) == len(result.player_rows)
    assert team_references[0].logo_file == (
        "assets/teams/team.png"
    )
    assert player_references[0].photo_file == (
        "assets/players/player-1.png"
    )
    assert champion_references[0].image_file == (
        "assets/champions/1.png"
    )
