from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from match_insight.data_processing.oracle_audit import (
    calculate_sha256,
)
from match_insight.data_processing.oracle_etl import (
    ORACLE_CORE_COLUMNS,
    ORACLE_INTEGER_COLUMNS,
    classify_oracle_rows,
    inspect_position_values,
    normalize_oracle_core_dataframe,
    prepare_oracle_core_data,
    read_oracle_core_csv,
)

PLAYER_POSITIONS = (
    "top",
    "jng",
    "mid",
    "bot",
    "sup",
)


def make_source_dataframe() -> pd.DataFrame:
    """Tạo một game nguồn gồm 10 player rows và 2 team rows."""
    rows: list[dict[str, object]] = []
    team_configurations = (
        {
            "side": "Blue",
            "teamname": "Blue Team",
            "teamid": "TEAM-BLUE",
            "participant_ids": range(1, 6),
            "team_participant_id": 100,
            "result": 1,
        },
        {
            "side": "Red",
            "teamname": "Red Team",
            "teamid": "TEAM-RED",
            "participant_ids": range(6, 11),
            "team_participant_id": 200,
            "result": 0,
        },
    )

    for configuration in team_configurations:
        side = str(configuration["side"])
        teamname = str(configuration["teamname"])
        teamid = str(configuration["teamid"])
        result = int(configuration["result"])
        participant_ids = configuration[
            "participant_ids"
        ]

        if not isinstance(participant_ids, range):
            raise TypeError(
                "participant_ids must be a range."
            )

        for participant_id, position in zip(
            participant_ids,
            PLAYER_POSITIONS,
            strict=True,
        ):
            rows.append(
                {
                    "gameid": "GAME-001",
                    "datacompleteness": "complete",
                    "url": "https://example.test/game/1",
                    "league": "TEST",
                    "year": 2026,
                    "split": "Spring",
                    "playoffs": 0,
                    "date": "2025-01-01T12:00:00Z",
                    "game": 1,
                    "patch": "15.10",
                    "participantid": participant_id,
                    "side": side,
                    "position": position,
                    "playername": (
                        f"{side} {position} Player"
                    ),
                    "playerid": (
                        f"{side.upper()}-{participant_id}"
                    ),
                    "teamname": teamname,
                    "teamid": teamid,
                    "champion": (
                        f"Champion-{participant_id}"
                    ),
                    "gamelength": 2100,
                    "result": result,
                    "kills": participant_id,
                }
            )

        rows.append(
            {
                "gameid": "GAME-001",
                "datacompleteness": "complete",
                "url": "https://example.test/game/1",
                "league": "TEST",
                "year": 2026,
                "split": "Spring",
                "playoffs": 0,
                "date": "2025-01-01T12:00:00Z",
                "game": 1,
                "patch": "15.10",
                "participantid": configuration[
                    "team_participant_id"
                ],
                "side": side,
                "position": "team",
                "playername": None,
                "playerid": None,
                "teamname": teamname,
                "teamid": teamid,
                "champion": None,
                "gamelength": 2100,
                "result": result,
                "kills": 0,
            }
        )

    return pd.DataFrame(rows).loc[
        :,
        [*ORACLE_CORE_COLUMNS, "kills"],
    ]


def write_source_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> Path:
    """Ghi fixture CSV tạm; không dùng CSV raw của dự án."""
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )
    return path


@pytest.fixture
def oracle_csv_path(tmp_path: Path) -> Path:
    """Tạo CSV nguồn tạm cho từng test."""
    return write_source_csv(
        make_source_dataframe(),
        tmp_path / "oracle_fixture.csv",
    )


def test_read_selects_exact_core_columns_and_preserves_patch(
    oracle_csv_path: Path,
) -> None:
    dataframe = read_oracle_core_csv(
        oracle_csv_path
    )

    assert tuple(dataframe.columns) == (
        ORACLE_CORE_COLUMNS
    )
    assert "kills" not in dataframe.columns
    assert all(
        pd.api.types.is_string_dtype(dtype)
        for dtype in dataframe.dtypes
    )
    assert dataframe["patch"].unique().tolist() == [
        "15.10"
    ]


def test_missing_core_column_is_rejected(
    tmp_path: Path,
) -> None:
    source_dataframe = make_source_dataframe().drop(
        columns=["champion"]
    )
    source_path = write_source_csv(
        source_dataframe,
        tmp_path / "missing_column.csv",
    )

    with pytest.raises(
        ValueError,
        match="champion",
    ):
        read_oracle_core_csv(source_path)


def test_normalize_uses_nullable_types_and_keeps_source_year(
    oracle_csv_path: Path,
) -> None:
    dataframe = normalize_oracle_core_dataframe(
        read_oracle_core_csv(oracle_csv_path)
    )

    for column in ORACLE_INTEGER_COLUMNS:
        assert str(dataframe[column].dtype) == "Int64"

    assert isinstance(
        dataframe["date"].dtype,
        pd.DatetimeTZDtype,
    )
    assert str(dataframe["date"].dt.tz) == "UTC"
    assert dataframe.at[0, "date"] == pd.Timestamp(
        "2025-01-01T12:00:00Z"
    )
    assert dataframe["year"].eq(2026).all()
    assert dataframe["patch"].eq("15.10").all()
    assert "calendar_year" not in dataframe.columns
    assert "started_at" not in dataframe.columns
    assert "ended_at" not in dataframe.columns


def test_normalize_distinguishes_zero_from_missing(
    tmp_path: Path,
) -> None:
    source_dataframe = make_source_dataframe()
    source_dataframe["playoffs"] = source_dataframe[
        "playoffs"
    ].astype("string")
    source_dataframe.at[0, "playoffs"] = "0"
    source_dataframe.at[1, "playoffs"] = pd.NA
    source_path = write_source_csv(
        source_dataframe,
        tmp_path / "nullable_integer.csv",
    )

    dataframe = normalize_oracle_core_dataframe(
        read_oracle_core_csv(source_path)
    )

    assert str(dataframe["playoffs"].dtype) == (
        "Int64"
    )
    assert dataframe.at[0, "playoffs"] == 0
    assert pd.isna(dataframe.at[1, "playoffs"])


@pytest.mark.parametrize(
    "invalid_year",
    ["not-a-number", "2025.5"],
)
def test_invalid_integer_value_is_rejected(
    invalid_year: str,
    tmp_path: Path,
) -> None:
    source_dataframe = make_source_dataframe()
    source_dataframe["year"] = source_dataframe[
        "year"
    ].astype("string")
    source_dataframe.at[0, "year"] = invalid_year
    source_path = write_source_csv(
        source_dataframe,
        tmp_path / "invalid_integer.csv",
    )

    with pytest.raises(
        ValueError,
        match="year",
    ):
        normalize_oracle_core_dataframe(
            read_oracle_core_csv(source_path)
        )


def test_inspect_position_values_reports_actual_source_values(
    oracle_csv_path: Path,
) -> None:
    dataframe = normalize_oracle_core_dataframe(
        read_oracle_core_csv(oracle_csv_path)
    )

    position_counts, missing_rows = (
        inspect_position_values(dataframe)
    )

    assert position_counts == {
        "bot": 2,
        "jng": 2,
        "mid": 2,
        "sup": 2,
        "team": 2,
        "top": 2,
    }
    assert missing_rows == 0


def test_classify_rows_uses_position_only(
    tmp_path: Path,
) -> None:
    source_dataframe = make_source_dataframe()
    team_mask = source_dataframe["position"].eq(
        "team"
    )
    source_dataframe.loc[
        ~team_mask,
        "participantid",
    ] = 100
    source_dataframe.loc[
        team_mask,
        "participantid",
    ] = 1
    source_path = write_source_csv(
        source_dataframe,
        tmp_path / "position_only.csv",
    )
    dataframe = normalize_oracle_core_dataframe(
        read_oracle_core_csv(source_path)
    )

    player_rows, team_rows = classify_oracle_rows(
        dataframe
    )

    assert len(player_rows) == 10
    assert len(team_rows) == 2
    assert set(player_rows.index).isdisjoint(
        team_rows.index
    )
    assert set(player_rows.index) | set(
        team_rows.index
    ) == set(dataframe.index)


@pytest.mark.parametrize(
    "invalid_position",
    [pd.NA, "coach"],
)
def test_invalid_position_fails_before_classification(
    invalid_position: object,
    tmp_path: Path,
) -> None:
    source_dataframe = make_source_dataframe()
    source_dataframe.at[0, "position"] = (
        invalid_position
    )
    source_path = write_source_csv(
        source_dataframe,
        tmp_path / "invalid_position.csv",
    )
    dataframe = normalize_oracle_core_dataframe(
        read_oracle_core_csv(source_path)
    )

    position_counts, missing_rows = (
        inspect_position_values(dataframe)
    )
    if pd.isna(invalid_position):
        assert missing_rows == 1
    else:
        assert position_counts["coach"] == 1

    with pytest.raises(
        ValueError,
        match="position",
    ):
        classify_oracle_rows(dataframe)


def test_prepare_core_data_preserves_raw_file(
    oracle_csv_path: Path,
) -> None:
    bytes_before = oracle_csv_path.read_bytes()
    sha256_before = calculate_sha256(
        oracle_csv_path
    )

    core_data = prepare_oracle_core_data(
        oracle_csv_path
    )

    bytes_after = oracle_csv_path.read_bytes()
    sha256_after = calculate_sha256(
        oracle_csv_path
    )

    assert bytes_after == bytes_before
    assert sha256_after == sha256_before
    assert len(core_data.dataframe) == 12
    assert len(core_data.player_rows) == 10
    assert len(core_data.team_rows) == 2
    assert core_data.missing_position_rows == 0
