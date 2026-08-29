from __future__ import annotations

import pandas as pd
import pytest

from match_insight.data_processing.oracle_audit import (
    audit_game_structure,
    audit_identifiers,
    audit_oracle_dataframe,
    audit_player_rows,
    audit_results,
    audit_team_rows,
    audit_time,
    to_python_int,
)

PLAYER_POSITIONS = (
    "top",
    "jng",
    "mid",
    "bot",
    "sup",
)


def make_valid_game() -> pd.DataFrame:
    """Tạo một ván hợp lệ gồm 12 dòng."""
    rows: list[dict[str, object]] = []

    team_configurations = (
        {
            "side": "Blue",
            "participant_ids": range(1, 6),
            "teamid": "TEAM-BLUE",
            "teamname": "Blue Team",
            "result": 1,
            "team_participantid": 100,
        },
        {
            "side": "Red",
            "participant_ids": range(6, 11),
            "teamid": "TEAM-RED",
            "teamname": "Red Team",
            "result": 0,
            "team_participantid": 200,
        },
    )

    for configuration in team_configurations:
        side = str(configuration["side"])
        teamid = str(configuration["teamid"])
        teamname = str(configuration["teamname"])
        result = int(configuration["result"])
        participant_ids = configuration[
            "participant_ids"
        ]

        if not isinstance(participant_ids, range):
            raise TypeError(
                "participant_ids must be a range."
            )

        for participantid, position in zip(
            participant_ids,
            PLAYER_POSITIONS,
            strict=True,
        ):
            row: dict[str, object] = {
                "gameid": "GAME-001",
                "participantid": participantid,
                "date": "2025-01-01T12:00:00Z",
                "year": 2025,
                "position": position,
                "side": side,
                "playername": (
                    f"{side} {position} Player"
                ),
                "playerid": (
                    f"{side.upper()}-{participantid}"
                ),
                "teamname": teamname,
                "teamid": teamid,
                "patch": "15.1",
                "champion": (
                    f"Champion-{participantid}"
                ),
                "result": result,
                "datacompleteness": "complete",
                "game_end_time": (
                    "2025-01-01T12:35:00Z"
                ),
            }

            for index in range(1, 6):
                row[f"pick{index}"] = None
                row[f"ban{index}"] = None

            rows.append(row)

        team_row: dict[str, object] = {
            "gameid": "GAME-001",
            "participantid": configuration[
                "team_participantid"
            ],
            "date": "2025-01-01T12:00:00Z",
            "year": 2025,
            "position": "team",
            "side": side,
            "playername": None,
            "playerid": None,
            "teamname": teamname,
            "teamid": teamid,
            "patch": "15.1",
            "champion": None,
            "result": result,
            "datacompleteness": "complete",
            "game_end_time": (
                "2025-01-01T12:35:00Z"
            ),
        }

        for index in range(1, 6):
            team_row[f"pick{index}"] = (
                f"{side}-Pick-{index}"
            )
            team_row[f"ban{index}"] = (
                f"{side}-Ban-{index}"
            )

        rows.append(team_row)

    return pd.DataFrame(rows)


@pytest.fixture
def valid_game_dataframe() -> pd.DataFrame:
    """Fixture một game hợp lệ, độc lập với CSV thật."""
    return make_valid_game()


def issue_count(
    issues: pd.DataFrame,
    issue_code: str,
) -> int:
    """Đếm một issue code mà không gây cảnh báo kiểu."""
    return to_python_int(
        issues["issue_code"].eq(issue_code).sum(),
        field_name=f"test_{issue_code}",
    )


def test_valid_sample(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    summary, issues = audit_oracle_dataframe(
        valid_game_dataframe
    )

    assert issues.empty
    assert summary["issues"] == {
        "total_issues": 0,
        "affected_games": 0,
        "by_code": {},
        "by_group": {},
        "by_severity": {},
    }


def test_missing_required_column(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.drop(
        columns=["champion"]
    )

    _, issues = audit_oracle_dataframe(dataframe)

    assert issue_count(
        issues,
        "MISSING_REQUIRED_COLUMN",
    ) == 1

    assert issue_count(
        issues,
        "MISSING_PLAYER_CHAMPION",
    ) == 0


def test_duplicate_gameid_participantid(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    duplicated_row = valid_game_dataframe.iloc[
        [0]
    ]

    dataframe = pd.concat(
        [
            valid_game_dataframe,
            duplicated_row,
        ],
        ignore_index=True,
    )

    issues = audit_identifiers(dataframe)

    assert issue_count(
        issues,
        "DUPLICATE_GAME_PARTICIPANT",
    ) == 2


def test_invalid_row_count(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.loc[
        ~valid_game_dataframe[
            "participantid"
        ].eq(10)
    ].copy()

    issues, summary = audit_game_structure(
        dataframe
    )

    assert issue_count(
        issues,
        "INVALID_GAME_ROW_COUNT",
    ) == 1

    assert issue_count(
        issues,
        "INVALID_PLAYER_ROW_COUNT",
    ) == 1

    assert summary[
        "games_with_valid_structure"
    ] == 0


def test_missing_playerid(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    row_index = dataframe.index[
        dataframe["participantid"].eq(1)
    ][0]

    dataframe.at[row_index, "playerid"] = pd.NA

    issues, summary = audit_player_rows(
        dataframe
    )

    assert issue_count(
        issues,
        "MISSING_PLAYER_ID",
    ) == 1

    assert summary["missing_values"] == {
        "playerid": 1,
        "teamid": 0,
        "playername": 0,
        "teamname": 0,
        "position": 0,
        "side": 0,
        "patch": 0,
        "champion": 0,
    }


def test_missing_champion(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    row_index = dataframe.index[
        dataframe["participantid"].eq(1)
    ][0]

    dataframe.at[row_index, "champion"] = pd.NA

    issues, _ = audit_player_rows(dataframe)

    assert issue_count(
        issues,
        "MISSING_PLAYER_CHAMPION",
    ) == 1


def test_missing_pick(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    row_index = dataframe.index[
        dataframe["participantid"].eq(100)
    ][0]

    dataframe.at[row_index, "pick3"] = pd.NA

    issues, summary = audit_team_rows(dataframe)

    assert issue_count(
        issues,
        "MISSING_TEAM_PICK",
    ) == 1

    assert summary[
        "games_missing_at_least_one_pick"
    ] == 1


def test_missing_ban(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    row_index = dataframe.index[
        dataframe["participantid"].eq(200)
    ][0]

    dataframe.at[row_index, "ban4"] = pd.NA

    issues, summary = audit_team_rows(dataframe)

    assert issue_count(
        issues,
        "MISSING_TEAM_BAN",
    ) == 1

    assert summary[
        "games_missing_at_least_one_ban"
    ] == 1


def test_year_date_mismatch(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    row_index = dataframe.index[0]

    dataframe.at[row_index, "year"] = 2026

    issues, summary = audit_time(dataframe)

    assert issue_count(
        issues,
        "YEAR_DATE_MISMATCH",
    ) == 1

    assert summary["year_date_mismatch_rows"] == 1


def test_invalid_result_structure(
    valid_game_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_game_dataframe.copy()
    team_mask = dataframe["position"].eq("team")

    dataframe.loc[team_mask, "result"] = 1

    issues, summary = audit_results(dataframe)

    assert issue_count(
        issues,
        "INVALID_RESULT_STRUCTURE",
    ) == 1

    assert summary[
        "games_with_two_team_winners"
    ] == 1
