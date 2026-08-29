from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import pandas as pd

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "gameid",
    "participantid",
    "date",
    "year",
    "position",
    "side",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "patch",
    "champion",
    "result",
    "datacompleteness",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
)

OPTIONAL_COLUMNS: Final[tuple[str, ...]] = (
    "url",
    "league",
    "split",
    "playoffs",
    "game",
    "gamelength",
)

PLAYER_POSITIONS: Final[frozenset[str]] = frozenset(
    {
        "top",
        "jng",
        "mid",
        "bot",
        "sup",
    }
)

TEAM_POSITION: Final[str] = "team"

PLAYER_PARTICIPANT_IDS: Final[frozenset[int]] = frozenset(
    range(1, 11)
)

TEAM_PARTICIPANT_IDS: Final[frozenset[int]] = frozenset(
    {
        100,
        200,
    }
)

VALID_SIDES: Final[frozenset[str]] = frozenset(
    {
        "Blue",
        "Red",
    }
)

VALID_RESULTS: Final[frozenset[int]] = frozenset(
    {
        0,
        1,
    }
)

ISSUE_COLUMNS: Final[tuple[str, ...]] = (
    "gameid",
    "participantid",
    "issue_code",
    "issue_group",
    "severity",
    "message",
)

PICK_COLUMNS: Final[tuple[str, ...]] = (
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
)

BAN_COLUMNS: Final[tuple[str, ...]] = (
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
)

GAME_END_TIME_CANDIDATES: Final[tuple[str, ...]] = (
    "game_end_time",
    "gameendtime",
    "game_end",
    "gameend",
    "end_time",
    "endtime",
)

VALID_DATA_COMPLETENESS_VALUES: Final[
    frozenset[str]
] = frozenset(
    {
        "complete",
        "partial",
    }
)

BENCHMARK_EXPECTED: Final[dict[str, int]] = {
    "total_rows": 120492,
    "distinct_games": 10041,
    "games_with_valid_structure": 10041,
    "complete_games": 9221,
    "partial_games": 820,
    "player_rows_missing_playerid": 1586,
    "player_rows_missing_teamid": 3030,
    "player_rows_missing_patch": 0,
    "player_rows_missing_champion": 0,
    "games_missing_pick": 26,
    "games_missing_ban": 226,
    "duplicate_game_participant_rows": 0,
    "games_year_2025": 9793,
    "games_year_2026": 248,
}


def resolve_source_path(source_path: str | Path) -> Path:
    """Kiểm tra và trả về đường dẫn tuyệt đối của CSV nguồn."""
    path = Path(source_path)

    if not path.is_absolute():
        path = Path.cwd() / path

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Oracle CSV does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Oracle CSV path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Oracle source must be a CSV file: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Oracle CSV is empty: {path}"
        )

    return path


def read_oracle_csv(source_path: str | Path) -> pd.DataFrame:
    """Đọc toàn bộ CSV mà không thay đổi dữ liệu nguồn."""
    path = resolve_source_path(source_path)

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Could not read Oracle CSV: {path}"
        ) from error


def find_missing_required_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Trả về required columns không tồn tại trong DataFrame."""
    actual_columns = set(dataframe.columns)

    return [
        column
        for column in REQUIRED_COLUMNS
        if column not in actual_columns
    ]


def find_present_optional_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Trả về optional columns đang tồn tại."""
    actual_columns = set(dataframe.columns)

    return [
        column
        for column in OPTIONAL_COLUMNS
        if column in actual_columns
    ]


def build_schema_snapshot(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Tạo snapshot schema; không thực hiện làm sạch dữ liệu."""
    return {
        "total_rows": int(dataframe.shape[0]),
        "total_columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "pandas_dtypes": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "missing_required_columns": (
            find_missing_required_columns(dataframe)
        ),
        "present_optional_columns": (
            find_present_optional_columns(dataframe)
        ),
    }

def calculate_sha256(source_path: str | Path) -> str:
    """Tính SHA-256 bằng cách đọc file theo từng khối."""
    path = resolve_source_path(source_path)
    digest = sha256()
    block_size = 1024 * 1024

    with path.open("rb") as source_file:
        for block in iter(
            lambda: source_file.read(block_size),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest().upper()


def build_file_metadata(
    source_path: str | Path,
) -> dict[str, object]:
    """Thu thập metadata mà không thay đổi file nguồn."""
    path = resolve_source_path(source_path)
    file_stat = path.stat()

    return {
        "filename": path.name,
        "source_path": str(path),
        "file_size_bytes": int(file_stat.st_size),
        "sha256": calculate_sha256(path),
        "last_modified_utc": datetime.fromtimestamp(
            file_stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "audit_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }


def create_issue(
    *,
    issue_code: str,
    issue_group: str,
    severity: str,
    message: str,
    gameid: object = None,
    participantid: object = None,
) -> dict[str, object]:
    """Tạo một issue record phẳng, không dùng class hierarchy."""
    return {
        "gameid": gameid,
        "participantid": participantid,
        "issue_code": issue_code,
        "issue_group": issue_group,
        "severity": severity,
        "message": message,
    }


def issues_to_dataframe(
    issues: list[dict[str, object]],
) -> pd.DataFrame:
    """Chuyển danh sách issue thành DataFrame có schema ổn định."""
    return pd.DataFrame.from_records(
        issues,
        columns=ISSUE_COLUMNS,
    )


def audit_schema(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ghi issue cho required columns không tồn tại."""
    issues: list[dict[str, object]] = []

    for column in find_missing_required_columns(dataframe):
        issues.append(
            create_issue(
                issue_code="MISSING_REQUIRED_COLUMN",
                issue_group="SCHEMA",
                severity="ERROR",
                message=(
                    "Required audit column is missing: "
                    f"{column}"
                ),
            )
        )

    return issues_to_dataframe(issues)

def missing_value_mask(series: pd.Series) -> pd.Series:
    """Xem null hoặc chuỗi chỉ chứa khoảng trắng là missing."""
    string_values = series.astype("string")
    blank_mask = string_values.str.strip().eq("")

    return (
        series.isna()
        | blank_mask.fillna(False)
    ).astype(bool)


def to_issue_value(value: Any) -> object:
    """Đổi pandas/numpy scalar thành giá trị phù hợp cho issue."""
    if pd.isna(value):
        return None

    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    return value

def to_python_int(
    value: Any,
    *,
    field_name: str,
) -> int:
    """Chuyển một pandas scalar không thiếu thành int Python."""
    if pd.isna(value):
        raise ValueError(
            f"{field_name} cannot be missing."
        )

    return int(value)

def build_row_masks(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Phân loại player/team mà không thay đổi DataFrame nguồn."""
    false_mask = pd.Series(
        False,
        index=dataframe.index,
        dtype=bool,
    )

    player_mask = false_mask.copy()
    team_mask = false_mask.copy()

    if "position" in dataframe.columns:
        position_values = dataframe["position"].astype("string")

        player_mask = position_values.isin(
            PLAYER_POSITIONS
        ).fillna(False)

        team_mask = position_values.eq(
            TEAM_POSITION
        ).fillna(False)

    if "participantid" not in dataframe.columns:
        return (
            player_mask.astype(bool),
            team_mask.astype(bool),
        )

    participant_ids = pd.to_numeric(
        dataframe["participantid"],
        errors="coerce",
    )

    participant_player_mask = participant_ids.isin(
        PLAYER_PARTICIPANT_IDS
    )

    participant_team_mask = participant_ids.isin(
        TEAM_PARTICIPANT_IDS
    )

    unclassified_position_mask = ~(
        player_mask | team_mask
    )

    player_mask = player_mask | (
        unclassified_position_mask
        & participant_player_mask
    )

    team_mask = team_mask | (
        unclassified_position_mask
        & participant_team_mask
    )

    return (
        player_mask.astype(bool),
        team_mask.astype(bool),
    )


def audit_identifiers(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Audit missing key và duplicate game-participant."""
    issues: list[dict[str, object]] = []
    actual_columns = set(dataframe.columns)

    if "gameid" in actual_columns:
        missing_game_mask = missing_value_mask(
            dataframe["gameid"]
        )

        for row_index in dataframe.index[missing_game_mask]:
            participantid = None

            if "participantid" in actual_columns:
                participantid = to_issue_value(
                    dataframe.at[
                        row_index,
                        "participantid",
                    ]
                )

            issues.append(
                create_issue(
                    gameid=None,
                    participantid=participantid,
                    issue_code="MISSING_GAME_ID",
                    issue_group="IDENTIFIER",
                    severity="ERROR",
                    message="gameid is missing.",
                )
            )

    if "participantid" in actual_columns:
        missing_participant_mask = missing_value_mask(
            dataframe["participantid"]
        )

        for row_index in dataframe.index[
            missing_participant_mask
        ]:
            gameid = None

            if "gameid" in actual_columns:
                gameid = to_issue_value(
                    dataframe.at[row_index, "gameid"]
                )

            issues.append(
                create_issue(
                    gameid=gameid,
                    participantid=None,
                    issue_code="MISSING_PARTICIPANT_ID",
                    issue_group="IDENTIFIER",
                    severity="ERROR",
                    message="participantid is missing.",
                )
            )

    key_columns_exist = {
        "gameid",
        "participantid",
    }.issubset(actual_columns)

    if key_columns_exist:
        valid_key_mask = (
            ~missing_value_mask(dataframe["gameid"])
            & ~missing_value_mask(
                dataframe["participantid"]
            )
        )

        duplicate_mask = (
            valid_key_mask
            & dataframe.duplicated(
                subset=[
                    "gameid",
                    "participantid",
                ],
                keep=False,
            )
        )

        for row_index in dataframe.index[duplicate_mask]:
            issues.append(
                create_issue(
                    gameid=to_issue_value(
                        dataframe.at[row_index, "gameid"]
                    ),
                    participantid=to_issue_value(
                        dataframe.at[
                            row_index,
                            "participantid",
                        ]
                    ),
                    issue_code=(
                        "DUPLICATE_GAME_PARTICIPANT"
                    ),
                    issue_group="IDENTIFIER",
                    severity="ERROR",
                    message=(
                        "Duplicate gameid and participantid "
                        "combination."
                    ),
                )
            )

    return issues_to_dataframe(issues)

def count_distribution(
    values: pd.Series,
) -> dict[str, int]:
    """Tạo phân bố đếm có thể ghi vào JSON."""
    counts = values.value_counts().sort_index()

    return {
        str(to_issue_value(value)): to_python_int(
            count,
            field_name="distribution_count",
        )
        for value, count in counts.items()
    }


def audit_game_structure(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Kiểm tra số dòng player/team của từng game."""
    issues: list[dict[str, object]] = []

    dependent_columns = (
        "gameid",
        "participantid",
        "position",
    )

    missing_dependencies = [
        column
        for column in dependent_columns
        if column not in dataframe.columns
    ]

    if missing_dependencies:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": missing_dependencies,
            "distinct_games": None,
        }

        return issues_to_dataframe(issues), summary

    player_mask, team_mask = build_row_masks(dataframe)

    valid_game_mask = ~missing_value_mask(
        dataframe["gameid"]
    )

    structure_rows = pd.DataFrame(
        {
            "gameid": dataframe.loc[
                valid_game_mask,
                "gameid",
            ],
            "is_player": player_mask.loc[
                valid_game_mask
            ].astype(int),
            "is_team": team_mask.loc[
                valid_game_mask
            ].astype(int),
        }
    )

    structure = structure_rows.groupby(
        "gameid",
        sort=False,
    ).agg(
        total_rows=("gameid", "size"),
        player_rows=("is_player", "sum"),
        team_rows=("is_team", "sum"),
    )

    invalid_total_mask = structure["total_rows"].ne(12)
    invalid_player_mask = structure["player_rows"].ne(10)
    invalid_team_mask = structure["team_rows"].ne(2)

    invalid_total_gameids = [
        to_issue_value(gameid)
        for gameid in structure.index[invalid_total_mask]
    ]

    invalid_player_gameids = [
        to_issue_value(gameid)
        for gameid in structure.index[invalid_player_mask]
    ]

    invalid_team_gameids = [
        to_issue_value(gameid)
        for gameid in structure.index[invalid_team_mask]
    ]

    for gameid in invalid_total_gameids:
        actual_count = to_python_int(
            structure.at[gameid, "total_rows"],
            field_name="total_rows",
        )   

        issues.append(
            create_issue(
                gameid=gameid,
                issue_code="INVALID_GAME_ROW_COUNT",
                issue_group="STRUCTURE",
                severity="ERROR",
                message=(
                    "Game has "
                    f"{actual_count} rows; expected 12."
                ),
            )
        )

    for gameid in invalid_player_gameids:
        actual_count = to_python_int(
            structure.at[gameid, "player_rows"],
            field_name="player_rows",
        )
        issues.append(
            create_issue(
                gameid=gameid,
                issue_code=(
                    "INVALID_PLAYER_ROW_COUNT"
                ),
                issue_group="STRUCTURE",
                severity="ERROR",
                message=(
                    "Game has "
                    f"{actual_count} player rows; "
                    "expected 10."
                ),
            )
        )

    for gameid in invalid_team_gameids:
        actual_count = to_python_int(
            structure.at[gameid, "team_rows"],
            field_name="team_rows",
        )

        issues.append(
            create_issue(
                gameid=gameid,
                issue_code="INVALID_TEAM_ROW_COUNT",
                issue_group="STRUCTURE",
                severity="ERROR",
                message=(
                    "Game has "
                    f"{actual_count} team rows; "
                    "expected 2."
                ),
            )
        )

    valid_structure_mask = (
        structure["total_rows"].eq(12)
        & structure["player_rows"].eq(10)
        & structure["team_rows"].eq(2)
    )

    summary = {
    "check_skipped": False,
    "distinct_games": len(structure),
    "games_with_12_rows": to_python_int(
        structure["total_rows"].eq(12).sum(),
        field_name="games_with_12_rows",
    ),
    "games_with_10_player_rows": to_python_int(
        structure["player_rows"].eq(10).sum(),
        field_name="games_with_10_player_rows",
    ),
    "games_with_2_team_rows": to_python_int(
        structure["team_rows"].eq(2).sum(),
        field_name="games_with_2_team_rows",
    ),
    "games_with_valid_structure": to_python_int(
        valid_structure_mask.sum(),
        field_name="games_with_valid_structure",
    ),
    "rows_per_game_distribution": (
        count_distribution(structure["total_rows"])
    ),
    "player_rows_per_game_distribution": (
        count_distribution(structure["player_rows"])
    ),
    "team_rows_per_game_distribution": (
        count_distribution(structure["team_rows"])
    ),
    "invalid_game_row_count_gameids": (
        invalid_total_gameids
    ),
    "invalid_player_row_count_gameids": (
        invalid_player_gameids
    ),
    "invalid_team_row_count_gameids": (
        invalid_team_gameids
    ),
}

    return issues_to_dataframe(issues), summary

def get_issue_identifiers(
    dataframe: pd.DataFrame,
    row_index: Any,
) -> tuple[object, object]:
    """Lấy gameid và participantid an toàn cho issue."""
    gameid: object = None
    participantid: object = None

    if "gameid" in dataframe.columns:
        gameid = to_issue_value(
            dataframe.at[row_index, "gameid"]
        )

    if "participantid" in dataframe.columns:
        participantid = to_issue_value(
            dataframe.at[row_index, "participantid"]
        )

    return gameid, participantid


def append_masked_row_issues(
    *,
    dataframe: pd.DataFrame,
    row_mask: pd.Series,
    issues: list[dict[str, object]],
    issue_code: str,
    issue_group: str,
    severity: str,
    message: str,
) -> None:
    """Thêm một issue cho mỗi dòng khớp mask."""
    selected_mask = row_mask.fillna(False).astype(bool)

    for row_index in dataframe.index[selected_mask]:
        gameid, participantid = get_issue_identifiers(
            dataframe,
            row_index,
        )

        issues.append(
            create_issue(
                gameid=gameid,
                participantid=participantid,
                issue_code=issue_code,
                issue_group=issue_group,
                severity=severity,
                message=message,
            )
        )

def audit_player_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit các trường và cấu trúc của player rows."""
    issues: list[dict[str, object]] = []

    classification_columns = (
        "participantid",
        "position",
    )

    missing_classification_columns = [
        column
        for column in classification_columns
        if column not in dataframe.columns
    ]

    if missing_classification_columns:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": (
                missing_classification_columns
            ),
            "player_row_count": None,
        }

        return issues_to_dataframe(issues), summary

    player_mask, _ = build_row_masks(dataframe)

    field_checks: tuple[
        tuple[str, str, str, str],
        ...,
    ] = (
        (
            "playerid",
            "MISSING_PLAYER_ID",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "teamid",
            "MISSING_TEAM_ID",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "playername",
            "MISSING_PLAYER_NAME",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "teamname",
            "MISSING_TEAM_NAME",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "position",
            "MISSING_POSITION",
            "CONTENT",
            "ERROR",
        ),
        (
            "side",
            "MISSING_SIDE",
            "CONTENT",
            "ERROR",
        ),
        (
            "patch",
            "MISSING_PATCH",
            "CONTENT",
            "ERROR",
        ),
        (
            "champion",
            "MISSING_PLAYER_CHAMPION",
            "CONTENT",
            "ERROR",
        ),
    )

    missing_counts: dict[str, int | None] = {}

    for (
        column,
        issue_code,
        issue_group,
        severity,
    ) in field_checks:
        if column not in dataframe.columns:
            missing_counts[column] = None
            continue

        field_missing_mask = (
            player_mask
            & missing_value_mask(dataframe[column])
        )

        missing_counts[column] = to_python_int(
            field_missing_mask.sum(),
            field_name=f"missing_{column}",
        )

        append_masked_row_issues(
            dataframe=dataframe,
            row_mask=field_missing_mask,
            issues=issues,
            issue_code=issue_code,
            issue_group=issue_group,
            severity=severity,
            message=(
                f"{column} is missing on player row."
            ),
        )

    position_missing_mask = missing_value_mask(
        dataframe["position"]
    )

    invalid_position_mask = (
        player_mask
        & ~position_missing_mask
        & ~dataframe["position"].isin(
            PLAYER_POSITIONS
        )
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=invalid_position_mask,
        issues=issues,
        issue_code="INVALID_POSITION",
        issue_group="CONTENT",
        severity="ERROR",
        message=(
            "Player row has an invalid or unknown "
            "position."
        ),
    )

    duplicate_role_mask = pd.Series(
        False,
        index=dataframe.index,
        dtype=bool,
    )

    duplicate_dependencies = {
        "gameid",
        "side",
        "position",
    }

    if duplicate_dependencies.issubset(dataframe.columns):
        duplicate_candidate_mask = (
            player_mask
            & ~missing_value_mask(
                dataframe["gameid"]
            )
            & ~missing_value_mask(
                dataframe["side"]
            )
            & dataframe["position"].isin(
                PLAYER_POSITIONS
            )
        )

        duplicate_candidates = dataframe.loc[
            duplicate_candidate_mask,
            [
                "gameid",
                "side",
                "position",
            ],
        ]

        duplicate_role_mask.loc[
            duplicate_candidates.index
        ] = duplicate_candidates.duplicated(
            subset=[
                "gameid",
                "side",
                "position",
            ],
            keep=False,
        )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=duplicate_role_mask,
        issues=issues,
        issue_code="DUPLICATE_TEAM_ROLE",
        issue_group="STRUCTURE",
        severity="ERROR",
        message=(
            "Position is duplicated within the same "
            "game and side."
        ),
    )

    incomplete_team_role_count: int | None = None
    player_team_group_count: int | None = None

    role_dependencies = {
        "gameid",
        "side",
        "position",
    }

    if role_dependencies.issubset(dataframe.columns):
        valid_group_mask = (
            player_mask
            & ~missing_value_mask(
                dataframe["gameid"]
            )
            & ~missing_value_mask(
                dataframe["side"]
            )
        )

        role_rows = dataframe.loc[
            valid_group_mask,
            [
                "gameid",
                "side",
                "position",
            ],
        ]

        grouped_roles = role_rows.groupby(
            [
                "gameid",
                "side",
            ],
            sort=False,
        )

        player_team_group_count = 0
        incomplete_team_role_count = 0

        for raw_group_key, group_rows in grouped_roles:
            group_key: Any = raw_group_key
            gameid = to_issue_value(group_key[0])
            side = to_issue_value(group_key[1])

            player_team_group_count += 1

            observed_positions: set[str] = set()

            for value in group_rows["position"].tolist():
                if (
                    isinstance(value, str)
                    and value in PLAYER_POSITIONS
                ):
                    observed_positions.add(value)

            missing_positions = sorted(
                PLAYER_POSITIONS
                - observed_positions
            )

            if not missing_positions:
                continue

            incomplete_team_role_count += 1

            issues.append(
                create_issue(
                    gameid=gameid,
                    participantid=None,
                    issue_code=(
                        "INCOMPLETE_TEAM_ROLES"
                    ),
                    issue_group="STRUCTURE",
                    severity="ERROR",
                    message=(
                        f"Side {side!r} is missing "
                        "required positions: "
                        f"{', '.join(missing_positions)}."
                    ),
                )
            )

    summary = {
        "check_skipped": False,
        "player_row_count": to_python_int(
            player_mask.sum(),
            field_name="player_row_count",
        ),
        "missing_values": missing_counts,
        "invalid_position_rows": to_python_int(
            invalid_position_mask.sum(),
            field_name="invalid_position_rows",
        ),
        "duplicated_team_role_rows": to_python_int(
            duplicate_role_mask.sum(),
            field_name="duplicated_team_role_rows",
        ),
        "player_team_group_count": (
            player_team_group_count
        ),
        "incomplete_team_role_groups": (
            incomplete_team_role_count
        ),
    }

    return issues_to_dataframe(issues), summary


def timestamp_to_iso(value: Any) -> str | None:
    """Chuyển pandas timestamp thành chuỗi ISO."""
    if pd.isna(value):
        return None

    return pd.Timestamp(value).isoformat()


def audit_time(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit date, year và khả năng xác định game end."""
    issues: list[dict[str, object]] = []

    dependencies = (
        "gameid",
        "date",
        "year",
    )

    missing_dependencies = [
        column
        for column in dependencies
        if column not in dataframe.columns
    ]

    if missing_dependencies:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": missing_dependencies,
        }

        return issues_to_dataframe(issues), summary

    parsed_dates: pd.Series = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
        utc=True,
        format="mixed",
    )

    missing_date_mask = missing_value_mask(
        dataframe["date"]
    )

    invalid_date_mask = (
        ~missing_date_mask
        & parsed_dates.isna()
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=missing_date_mask,
        issues=issues,
        issue_code="MISSING_DATE",
        issue_group="TIME",
        severity="ERROR",
        message="date is missing.",
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=invalid_date_mask,
        issues=issues,
        issue_code="INVALID_DATE",
        issue_group="TIME",
        severity="ERROR",
        message="date cannot be parsed.",
    )

    year_values = pd.Series(
        pd.to_numeric(
            dataframe["year"],
            errors="coerce",
        ),
        index=dataframe.index,
    )

    missing_year_mask = missing_value_mask(
        dataframe["year"]
    )

    invalid_year_mask = (
        ~missing_year_mask
        & (
            year_values.isna()
            | (
                year_values.notna()
                & year_values.mod(1).ne(0)
            )
        )
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=missing_year_mask,
        issues=issues,
        issue_code="MISSING_YEAR",
        issue_group="TIME",
        severity="ERROR",
        message="year is missing.",
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=invalid_year_mask,
        issues=issues,
        issue_code="INVALID_YEAR",
        issue_group="TIME",
        severity="ERROR",
        message=(
            "year is not a valid integer year."
        ),
    )

    valid_date_year_mask = (
        ~missing_date_mask
        & ~invalid_date_mask
        & ~missing_year_mask
        & ~invalid_year_mask
    )

    parsed_date_years = parsed_dates.dt.year

    year_date_mismatch_mask = (
        valid_date_year_mask
        & year_values.ne(parsed_date_years)
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=year_date_mismatch_mask,
        issues=issues,
        issue_code="YEAR_DATE_MISMATCH",
        issue_group="TIME",
        severity="ERROR",
        message=(
            "year differs from the year parsed "
            "from date."
        ),
    )

    valid_game_mask = ~missing_value_mask(
        dataframe["gameid"]
    )

    all_gameids = [
        to_issue_value(value)
        for value in dataframe.loc[
            valid_game_mask,
            "gameid",
        ].drop_duplicates().tolist()
    ]

    valid_game_year_mask = (
        valid_game_mask
        & ~missing_year_mask
        & ~invalid_year_mask
    )

    game_year_rows = dataframe.loc[
        valid_game_year_mask,
        ["gameid"],
    ]

    year_distribution_by_game: dict[str, int] = {}
    inconsistent_game_yearids: list[object] = []

    for raw_gameid, group_rows in (
        game_year_rows.groupby(
            "gameid",
            sort=False,
        )
    ):
        gameid: Any = to_issue_value(raw_gameid)
        observed_years: set[int] = set()

        for row_index in group_rows.index:
            year_value: Any = year_values.at[
                row_index
            ]

            if pd.isna(year_value):
                continue

            observed_years.add(
                to_python_int(
                    year_value,
                    field_name="game_year",
                )
            )

        if len(observed_years) != 1:
            inconsistent_game_yearids.append(
                gameid
            )

            issues.append(
                create_issue(
                    gameid=gameid,
                    participantid=None,
                    issue_code=(
                        "INCONSISTENT_GAME_YEAR"
                    ),
                    issue_group="TIME",
                    severity="ERROR",
                    message=(
                        "Rows in the same game contain "
                        "different year values."
                    ),
                )
            )

            continue

        game_year = next(iter(observed_years))
        year_key = str(game_year)

        year_distribution_by_game[year_key] = (
            year_distribution_by_game.get(
                year_key,
                0,
            )
            + 1
        )

    valid_dates = parsed_dates.dropna()

    min_date = (
        None
        if valid_dates.empty
        else timestamp_to_iso(valid_dates.min())
    )

    max_date = (
        None
        if valid_dates.empty
        else timestamp_to_iso(valid_dates.max())
    )

    end_time_columns = [
        column
        for column in GAME_END_TIME_CANDIDATES
        if column in dataframe.columns
    ]

    has_explicit_game_end_time = bool(
        end_time_columns
    )

    if has_explicit_game_end_time:
        cutoff_assessment = {
            "status": "REQUIRES_VALIDATION",
            "message": (
                "A candidate game-end column exists, "
                "but its semantics, timezone and "
                "completeness must be validated."
            ),
        }
    else:
        cutoff_assessment = {
            "status": "UNAVAILABLE",
            "message": (
                "The source has no explicit game-end "
                "timestamp. date must not be assumed "
                "to be the game-end cutoff."
            ),
        }

        issues.append(
            create_issue(
                gameid=None,
                participantid=None,
                issue_code=(
                    "MISSING_EXPLICIT_GAME_END_TIME"
                ),
                issue_group="TIME",
                severity="WARNING",
                message=(
                    "No explicit game-end timestamp "
                    "exists in the source schema."
                ),
            )
        )

    games_with_valid_year = sum(
        year_distribution_by_game.values()
    )

    summary = {
        "check_skipped": False,
        "missing_date_rows": to_python_int(
            missing_date_mask.sum(),
            field_name="missing_date_rows",
        ),
        "invalid_date_rows": to_python_int(
            invalid_date_mask.sum(),
            field_name="invalid_date_rows",
        ),
        "min_date_utc": min_date,
        "max_date_utc": max_date,
        "missing_year_rows": to_python_int(
            missing_year_mask.sum(),
            field_name="missing_year_rows",
        ),
        "invalid_year_rows": to_python_int(
            invalid_year_mask.sum(),
            field_name="invalid_year_rows",
        ),
        "year_date_mismatch_rows": to_python_int(
            year_date_mismatch_mask.sum(),
            field_name="year_date_mismatch_rows",
        ),
        "year_distribution_by_game": (
            year_distribution_by_game
        ),
        "games_with_valid_year": (
            games_with_valid_year
        ),
        "games_without_valid_year": (
            len(all_gameids)
            - games_with_valid_year
            - len(inconsistent_game_yearids)
        ),
        "games_with_inconsistent_year": len(
            inconsistent_game_yearids
        ),
        "inconsistent_game_yearids": (
            inconsistent_game_yearids
        ),
        "explicit_game_end_time_columns": (
            end_time_columns
        ),
        "has_explicit_game_end_time": (
            has_explicit_game_end_time
        ),
        "cutoff_assessment": cutoff_assessment,
    }

    return issues_to_dataframe(issues), summary


def result_counts_for_mask(
    result_values: pd.Series,
    row_mask: pd.Series,
) -> dict[str, int]:
    """Đếm result hợp lệ trong phạm vi row mask."""
    valid_mask = (
        row_mask.fillna(False).astype(bool)
        & result_values.isin(VALID_RESULTS)
    )

    counts = (
        result_values.loc[valid_mask]
        .value_counts()
        .sort_index()
    )

    return {
        str(
            to_python_int(
                value,
                field_name="result_value",
            )
        ): to_python_int(
            count,
            field_name="result_count",
        )
        for value, count in counts.items()
    }


def audit_results(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Kiểm tra miền result và tính nhất quán trong ván."""
    issues: list[dict[str, object]] = []

    dependencies = (
        "gameid",
        "participantid",
        "position",
        "side",
        "result",
    )

    missing_dependencies = [
        column
        for column in dependencies
        if column not in dataframe.columns
    ]

    if missing_dependencies:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": missing_dependencies,
        }

        return issues_to_dataframe(issues), summary

    player_mask, team_mask = build_row_masks(dataframe)

    result_values = pd.Series(
        pd.to_numeric(
            dataframe["result"],
            errors="coerce",
        ),
        index=dataframe.index,
    )

    missing_result_mask = missing_value_mask(
        dataframe["result"]
    )

    invalid_result_mask = (
        ~missing_result_mask
        & ~result_values.isin(VALID_RESULTS)
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=missing_result_mask,
        issues=issues,
        issue_code="MISSING_RESULT",
        issue_group="CONTENT",
        severity="ERROR",
        message="result is missing.",
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=invalid_result_mask,
        issues=issues,
        issue_code="INVALID_RESULT_VALUE",
        issue_group="CONTENT",
        severity="ERROR",
        message=(
            "result is outside the observed valid set "
            "{0, 1}."
        ),
    )

    valid_game_mask = ~missing_value_mask(
        dataframe["gameid"]
    )

    all_gameids = [
        to_issue_value(value)
        for value in dataframe.loc[
            valid_game_mask,
            "gameid",
        ].drop_duplicates().tolist()
    ]

    team_rows = dataframe.loc[
        team_mask & valid_game_mask,
        ["gameid"],
    ]

    team_results_by_game: dict[Any, list[int]] = {}
    team_row_counts_by_game: dict[Any, int] = {}

    for raw_gameid, group_rows in team_rows.groupby(
        "gameid",
        sort=False,
    ):
        gameid: Any = to_issue_value(raw_gameid)
        valid_team_results: list[int] = []

        for row_index in group_rows.index:
            result_value: Any = result_values.at[
                row_index
            ]

            if result_value not in VALID_RESULTS:
                continue

            valid_team_results.append(
                to_python_int(
                    result_value,
                    field_name="team_result",
                )
            )

        team_results_by_game[gameid] = (
            valid_team_results
        )
        team_row_counts_by_game[gameid] = len(
            group_rows
        )

    invalid_result_structure_gameids: list[object] = []
    two_team_winner_gameids: list[object] = []
    games_without_team_winner: list[object] = []

    for gameid in all_gameids:
        team_results = team_results_by_game.get(
            gameid,
            [],
        )

        team_row_count = team_row_counts_by_game.get(
            gameid,
            0,
        )

        winner_count = team_results.count(1)
        loser_count = team_results.count(0)

        if winner_count == 2:
            two_team_winner_gameids.append(gameid)

        if winner_count == 0:
            games_without_team_winner.append(gameid)

        valid_structure = (
            team_row_count == 2
            and winner_count == 1
            and loser_count == 1
        )

        if valid_structure:
            continue

        invalid_result_structure_gameids.append(
            gameid
        )

        issues.append(
            create_issue(
                gameid=gameid,
                participantid=None,
                issue_code=(
                    "INVALID_RESULT_STRUCTURE"
                ),
                issue_group="CONTENT",
                severity="ERROR",
                message=(
                    "Game must have exactly one winning "
                    "team and one losing team. "
                    f"Observed team results: "
                    f"{team_results}."
                ),
            )
        )

    expected_team_results: dict[
        tuple[Any, Any],
        int,
    ] = {}

    valid_team_side_mask = (
        team_mask
        & valid_game_mask
        & ~missing_value_mask(dataframe["side"])
    )

    team_side_rows = dataframe.loc[
        valid_team_side_mask,
        [
            "gameid",
            "side",
        ],
    ]

    for raw_group_key, group_rows in (
        team_side_rows.groupby(
            [
                "gameid",
                "side",
            ],
            sort=False,
        )
    ):
        if len(group_rows) != 1:
            continue

        group_key: Any = raw_group_key
        row_index = group_rows.index[0]

        team_result: Any = result_values.at[
            row_index
        ]

        if team_result not in VALID_RESULTS:
            continue

        key = (
            to_issue_value(group_key[0]),
            to_issue_value(group_key[1]),
        )

        expected_team_results[key] = to_python_int(
            team_result,
            field_name="expected_team_result",
        )

    valid_player_side_mask = (
        player_mask
        & valid_game_mask
        & ~missing_value_mask(dataframe["side"])
    )

    player_team_result_mismatch_count = 0

    for row_index in dataframe.index[
        valid_player_side_mask
    ]:
        gameid, participantid = get_issue_identifiers(
            dataframe,
            row_index,
        )

        side = to_issue_value(
            dataframe.at[row_index, "side"]
        )

        key = (gameid, side)

        if key not in expected_team_results:
            continue

        player_result: Any = result_values.at[
            row_index
        ]

        if player_result not in VALID_RESULTS:
            continue

        player_result_int = to_python_int(
            player_result,
            field_name="player_result",
        )

        expected_result = expected_team_results[key]

        if player_result_int == expected_result:
            continue

        player_team_result_mismatch_count += 1

        issues.append(
            create_issue(
                gameid=gameid,
                participantid=participantid,
                issue_code=(
                    "PLAYER_TEAM_RESULT_MISMATCH"
                ),
                issue_group="CONTENT",
                severity="ERROR",
                message=(
                    "Player result does not match the "
                    f"team row for side {side!r}."
                ),
            )
        )

    all_rows_mask = pd.Series(
        True,
        index=dataframe.index,
        dtype=bool,
    )

    invalid_observed_values = sorted(
        {
            str(to_issue_value(value))
            for value in dataframe.loc[
                invalid_result_mask,
                "result",
            ].drop_duplicates().tolist()
        }
    )

    summary = {
        "check_skipped": False,
        "valid_result_values": [0, 1],
        "result_value_counts": result_counts_for_mask(
            result_values,
            all_rows_mask,
        ),
        "player_result_value_counts": (
            result_counts_for_mask(
                result_values,
                player_mask,
            )
        ),
        "team_result_value_counts": (
            result_counts_for_mask(
                result_values,
                team_mask,
            )
        ),
        "missing_result_rows": to_python_int(
            missing_result_mask.sum(),
            field_name="missing_result_rows",
        ),
        "invalid_result_rows": to_python_int(
            invalid_result_mask.sum(),
            field_name="invalid_result_rows",
        ),
        "invalid_observed_values": (
            invalid_observed_values
        ),
        "games_checked": len(all_gameids),
        "games_with_valid_result_structure": (
            len(all_gameids)
            - len(invalid_result_structure_gameids)
        ),
        "games_with_invalid_result_structure": len(
            invalid_result_structure_gameids
        ),
        "invalid_result_structure_gameids": (
            invalid_result_structure_gameids
        ),
        "games_with_two_team_winners": len(
            two_team_winner_gameids
        ),
        "two_team_winner_gameids": (
            two_team_winner_gameids
        ),
        "games_without_team_winner": len(
            games_without_team_winner
        ),
        "games_without_team_winner_gameids": (
            games_without_team_winner
        ),
        "player_team_result_mismatch_rows": (
            player_team_result_mismatch_count
        ),
    }

    return issues_to_dataframe(issues), summary


def build_missing_cells(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Tạo missing mask theo từng cột mà không sửa dữ liệu."""
    return pd.DataFrame(
        {
            column: missing_value_mask(
                dataframe[column]
            )
            for column in columns
        },
        index=dataframe.index,
    )


def gameids_for_mask(
    dataframe: pd.DataFrame,
    row_mask: pd.Series,
) -> list[object]:
    """Lấy danh sách gameid riêng biệt của các dòng khớp mask."""
    if "gameid" not in dataframe.columns:
        return []

    valid_mask = (
        row_mask.fillna(False).astype(bool)
        & ~missing_value_mask(dataframe["gameid"])
    )

    values = dataframe.loc[
        valid_mask,
        "gameid",
    ].drop_duplicates()

    return [
        to_issue_value(value)
        for value in values.tolist()
    ]


def audit_team_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit identifiers, pick/ban và sides của team rows."""
    issues: list[dict[str, object]] = []

    classification_columns = (
        "participantid",
        "position",
    )

    missing_classification_columns = [
        column
        for column in classification_columns
        if column not in dataframe.columns
    ]

    if missing_classification_columns:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": (
                missing_classification_columns
            ),
            "team_row_count": None,
        }

        return issues_to_dataframe(issues), summary

    _, team_mask = build_row_masks(dataframe)

    field_checks: tuple[
        tuple[str, str, str, str],
        ...,
    ] = (
        (
            "teamid",
            "MISSING_TEAM_ID",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "teamname",
            "MISSING_TEAM_NAME",
            "IDENTIFIER",
            "WARNING",
        ),
        (
            "side",
            "MISSING_SIDE",
            "CONTENT",
            "ERROR",
        ),
    )

    missing_counts: dict[str, int | None] = {}

    for (
        column,
        issue_code,
        issue_group,
        severity,
    ) in field_checks:
        if column not in dataframe.columns:
            missing_counts[column] = None
            continue

        field_missing_mask = (
            team_mask
            & missing_value_mask(dataframe[column])
        )

        missing_counts[column] = to_python_int(
            field_missing_mask.sum(),
            field_name=f"team_missing_{column}",
        )

        append_masked_row_issues(
            dataframe=dataframe,
            row_mask=field_missing_mask,
            issues=issues,
            issue_code=issue_code,
            issue_group=issue_group,
            severity=severity,
            message=(
                f"{column} is missing on team row."
            ),
        )

    pick_missing_field_counts: dict[str, int] | None = None
    team_rows_missing_pick: int | None = None
    missing_pick_gameids: list[object] | None = None

    if set(PICK_COLUMNS).issubset(dataframe.columns):
        pick_missing_cells = build_missing_cells(
            dataframe,
            PICK_COLUMNS,
        )

        team_pick_missing_mask = (
            team_mask
            & pick_missing_cells.any(axis=1)
        )

        pick_missing_field_counts = {
            column: to_python_int(
                (
                    team_mask
                    & pick_missing_cells[column]
                ).sum(),
                field_name=f"missing_{column}",
            )
            for column in PICK_COLUMNS
        }

        team_rows_missing_pick = to_python_int(
            team_pick_missing_mask.sum(),
            field_name="team_rows_missing_pick",
        )

        missing_pick_gameids = gameids_for_mask(
            dataframe,
            team_pick_missing_mask,
        )

        append_masked_row_issues(
            dataframe=dataframe,
            row_mask=team_pick_missing_mask,
            issues=issues,
            issue_code="MISSING_TEAM_PICK",
            issue_group="CONTENT",
            severity="WARNING",
            message=(
                "At least one pick1-pick5 value is "
                "missing on team row."
            ),
        )

    ban_missing_field_counts: dict[str, int] | None = None
    team_rows_missing_ban: int | None = None
    missing_ban_gameids: list[object] | None = None

    if set(BAN_COLUMNS).issubset(dataframe.columns):
        ban_missing_cells = build_missing_cells(
            dataframe,
            BAN_COLUMNS,
        )

        team_ban_missing_mask = (
            team_mask
            & ban_missing_cells.any(axis=1)
        )

        ban_missing_field_counts = {
            column: to_python_int(
                (
                    team_mask
                    & ban_missing_cells[column]
                ).sum(),
                field_name=f"missing_{column}",
            )
            for column in BAN_COLUMNS
        }

        team_rows_missing_ban = to_python_int(
            team_ban_missing_mask.sum(),
            field_name="team_rows_missing_ban",
        )

        missing_ban_gameids = gameids_for_mask(
            dataframe,
            team_ban_missing_mask,
        )

        append_masked_row_issues(
            dataframe=dataframe,
            row_mask=team_ban_missing_mask,
            issues=issues,
            issue_code="MISSING_TEAM_BAN",
            issue_group="CONTENT",
            severity="WARNING",
            message=(
                "At least one ban1-ban5 value is "
                "missing on team row."
            ),
        )

    invalid_team_side_gameids: list[object] | None = None

    side_dependencies = {
        "gameid",
        "side",
    }

    if side_dependencies.issubset(dataframe.columns):
        valid_game_mask = ~missing_value_mask(
            dataframe["gameid"]
        )

        all_gameids = [
            to_issue_value(value)
            for value in dataframe.loc[
                valid_game_mask,
                "gameid",
            ].drop_duplicates().tolist()
        ]

        team_side_rows = dataframe.loc[
            team_mask & valid_game_mask,
            [
                "gameid",
                "side",
            ],
        ]

        sides_by_game: dict[Any, set[str]] = {}

        for raw_gameid, group_rows in (
            team_side_rows.groupby(
                "gameid",
                sort=False,
            )
        ):
            gameid: Any = to_issue_value(raw_gameid)
            observed_sides: set[str] = set()

            for value in group_rows["side"].tolist():
                if isinstance(value, str):
                    observed_sides.add(value)

            sides_by_game[gameid] = observed_sides

        invalid_team_side_gameids = []

        for gameid in all_gameids:
            observed_sides = sides_by_game.get(
                gameid,
                set(),
            )

            if observed_sides == set(VALID_SIDES):
                continue

            invalid_team_side_gameids.append(gameid)

            observed_text = (
                ", ".join(sorted(observed_sides))
                if observed_sides
                else "<none>"
            )

            issues.append(
                create_issue(
                    gameid=gameid,
                    participantid=None,
                    issue_code=(
                        "INVALID_TEAM_SIDE_STRUCTURE"
                    ),
                    issue_group="STRUCTURE",
                    severity="ERROR",
                    message=(
                        "Game does not have the two "
                        "required team sides. Observed: "
                        f"{observed_text}."
                    ),
                )
            )

    summary = {
        "check_skipped": False,
        "team_row_count": to_python_int(
            team_mask.sum(),
            field_name="team_row_count",
        ),
        "missing_values": missing_counts,
        "pick_missing_field_counts": (
            pick_missing_field_counts
        ),
        "team_rows_missing_pick": (
            team_rows_missing_pick
        ),
        "games_missing_at_least_one_pick": (
            None
            if missing_pick_gameids is None
            else len(missing_pick_gameids)
        ),
        "missing_pick_gameids": missing_pick_gameids,
        "ban_missing_field_counts": (
            ban_missing_field_counts
        ),
        "team_rows_missing_ban": (
            team_rows_missing_ban
        ),
        "games_missing_at_least_one_ban": (
            None
            if missing_ban_gameids is None
            else len(missing_ban_gameids)
        ),
        "missing_ban_gameids": missing_ban_gameids,
        "games_with_invalid_team_sides": (
            None
            if invalid_team_side_gameids is None
            else len(invalid_team_side_gameids)
        ),
        "invalid_team_side_gameids": (
            invalid_team_side_gameids
        ),
    }

    return issues_to_dataframe(issues), summary


def audit_data_completeness(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit nhãn datacompleteness ở cấp dòng và ván."""
    issues: list[dict[str, object]] = []

    dependencies = (
        "gameid",
        "datacompleteness",
    )

    missing_dependencies = [
        column
        for column in dependencies
        if column not in dataframe.columns
    ]

    if missing_dependencies:
        summary: dict[str, object] = {
            "check_skipped": True,
            "missing_dependencies": missing_dependencies,
        }

        return issues_to_dataframe(issues), summary

    completeness_values = dataframe[
        "datacompleteness"
    ].astype("string")

    missing_completeness_mask = missing_value_mask(
        dataframe["datacompleteness"]
    )

    unknown_completeness_mask = (
        ~missing_completeness_mask
        & ~completeness_values.isin(
            VALID_DATA_COMPLETENESS_VALUES
        )
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=missing_completeness_mask,
        issues=issues,
        issue_code="MISSING_DATA_COMPLETENESS",
        issue_group="CONTENT",
        severity="WARNING",
        message="datacompleteness is missing.",
    )

    append_masked_row_issues(
        dataframe=dataframe,
        row_mask=unknown_completeness_mask,
        issues=issues,
        issue_code="UNKNOWN_DATA_COMPLETENESS",
        issue_group="CONTENT",
        severity="WARNING",
        message=(
            "datacompleteness is outside the observed "
            "set {'complete', 'partial'}."
        ),
    )

    row_distribution: dict[str, int] = {}

    row_counts = dataframe[
        "datacompleteness"
    ].value_counts(dropna=False)

    for raw_value, raw_count in row_counts.items():
        value: Any = raw_value
        count: Any = raw_count

        if pd.isna(value):
            value_key = "<MISSING>"
        else:
            value_key = str(
                to_issue_value(value)
            )

        row_distribution[value_key] = to_python_int(
            count,
            field_name="datacompleteness_count",
        )

    valid_game_mask = ~missing_value_mask(
        dataframe["gameid"]
    )

    game_rows = dataframe.loc[
        valid_game_mask,
        [
            "gameid",
            "datacompleteness",
        ],
    ]

    game_distribution = {
        "complete": 0,
        "partial": 0,
        "other_or_inconsistent": 0,
    }

    invalid_gameids: list[object] = []

    for raw_gameid, group_rows in game_rows.groupby(
        "gameid",
        sort=False,
    ):
        gameid: Any = to_issue_value(raw_gameid)
        observed_values: set[str] = set()

        for raw_value in group_rows[
            "datacompleteness"
        ].tolist():
            value: Any = raw_value

            if pd.isna(value):
                observed_values.add("<MISSING>")
            else:
                observed_values.add(
                    str(to_issue_value(value))
                )

        if observed_values == {"complete"}:
            game_distribution["complete"] += 1
            continue

        if observed_values == {"partial"}:
            game_distribution["partial"] += 1
            continue

        game_distribution[
            "other_or_inconsistent"
        ] += 1

        invalid_gameids.append(gameid)

        observed_text = ", ".join(
            sorted(observed_values)
        )

        issues.append(
            create_issue(
                gameid=gameid,
                participantid=None,
                issue_code=(
                    "INVALID_GAME_DATA_COMPLETENESS"
                ),
                issue_group="CONTENT",
                severity="WARNING",
                message=(
                    "Game has unknown, missing or "
                    "inconsistent datacompleteness "
                    f"values: {observed_text}."
                ),
            )
        )

    summary = {
        "check_skipped": False,
        "row_value_distribution": row_distribution,
        "game_value_distribution": game_distribution,
        "missing_rows": to_python_int(
            missing_completeness_mask.sum(),
            field_name="missing_completeness_rows",
        ),
        "unknown_rows": to_python_int(
            unknown_completeness_mask.sum(),
            field_name="unknown_completeness_rows",
        ),
        "complete_games": game_distribution[
            "complete"
        ],
        "partial_games": game_distribution[
            "partial"
        ],
        "games_with_other_or_inconsistent_status": (
            game_distribution[
                "other_or_inconsistent"
            ]
        ),
        "invalid_gameids": invalid_gameids,
    }

    return issues_to_dataframe(issues), summary


def combine_issue_dataframes(
    *issue_frames: pd.DataFrame,
) -> pd.DataFrame:
    """Ghép issue frames và giữ schema cột ổn định."""
    non_empty_frames = [
        frame.loc[:, list(ISSUE_COLUMNS)]
        for frame in issue_frames
        if not frame.empty
    ]

    if not non_empty_frames:
        return issues_to_dataframe([])

    return pd.concat(
        non_empty_frames,
        ignore_index=True,
    )


def issue_counts_by_column(
    issues: pd.DataFrame,
    column: str,
) -> dict[str, int]:
    """Đếm issues theo một cột phân loại."""
    if issues.empty:
        return {}

    counts = issues[column].value_counts(
        dropna=False
    )

    result: dict[str, int] = {}

    for raw_value, raw_count in counts.items():
        value: Any = raw_value
        count: Any = raw_count

        if pd.isna(value):
            key = "<MISSING>"
        else:
            key = str(to_issue_value(value))

        result[key] = to_python_int(
            count,
            field_name=f"issue_count_{column}",
        )

    return result


def summary_value_to_int(
    value: Any,
    *,
    field_name: str,
) -> int | None:
    """Đọc số nguyên tùy chọn từ summary."""
    if value is None or isinstance(value, bool):
        return None

    if pd.isna(value):
        return None

    return to_python_int(
        value,
        field_name=field_name,
    )


def nested_summary_int(
    summary: dict[str, object],
    parent_key: str,
    child_key: str,
) -> int | None:
    """Đọc một số nguyên trong mapping lồng nhau."""
    parent_value = summary.get(parent_key)

    if not isinstance(parent_value, dict):
        return None

    child_value: Any = parent_value.get(child_key)

    return summary_value_to_int(
        child_value,
        field_name=f"{parent_key}_{child_key}",
    )


def build_benchmark_comparison(
    *,
    schema_summary: dict[str, object],
    structure_summary: dict[str, object],
    player_summary: dict[str, object],
    team_summary: dict[str, object],
    time_summary: dict[str, object],
    completeness_summary: dict[str, object],
    identifier_issues: pd.DataFrame,
) -> dict[str, object]:
    """So sánh kết quả audit độc lập với benchmark tham chiếu."""
    duplicate_count = to_python_int(
        identifier_issues["issue_code"]
        .eq("DUPLICATE_GAME_PARTICIPANT")
        .sum(),
        field_name="duplicate_game_participant_rows",
    )

    actual_metrics: dict[str, int | None] = {
        "total_rows": summary_value_to_int(
            schema_summary.get("total_rows"),
            field_name="total_rows",
        ),
        "distinct_games": summary_value_to_int(
            structure_summary.get("distinct_games"),
            field_name="distinct_games",
        ),
        "games_with_valid_structure": (
            summary_value_to_int(
                structure_summary.get(
                    "games_with_valid_structure"
                ),
                field_name=(
                    "games_with_valid_structure"
                ),
            )
        ),
        "complete_games": summary_value_to_int(
            completeness_summary.get(
                "complete_games"
            ),
            field_name="complete_games",
        ),
        "partial_games": summary_value_to_int(
            completeness_summary.get(
                "partial_games"
            ),
            field_name="partial_games",
        ),
        "player_rows_missing_playerid": (
            nested_summary_int(
                player_summary,
                "missing_values",
                "playerid",
            )
        ),
        "player_rows_missing_teamid": (
            nested_summary_int(
                player_summary,
                "missing_values",
                "teamid",
            )
        ),
        "player_rows_missing_patch": (
            nested_summary_int(
                player_summary,
                "missing_values",
                "patch",
            )
        ),
        "player_rows_missing_champion": (
            nested_summary_int(
                player_summary,
                "missing_values",
                "champion",
            )
        ),
        "games_missing_pick": summary_value_to_int(
            team_summary.get(
                "games_missing_at_least_one_pick"
            ),
            field_name="games_missing_pick",
        ),
        "games_missing_ban": summary_value_to_int(
            team_summary.get(
                "games_missing_at_least_one_ban"
            ),
            field_name="games_missing_ban",
        ),
        "duplicate_game_participant_rows": (
            duplicate_count
        ),
        "games_year_2025": nested_summary_int(
            time_summary,
            "year_distribution_by_game",
            "2025",
        ),
        "games_year_2026": nested_summary_int(
            time_summary,
            "year_distribution_by_game",
            "2026",
        ),
    }

    metric_comparisons: dict[
        str,
        dict[str, object],
    ] = {}

    for metric, expected in BENCHMARK_EXPECTED.items():
        actual = actual_metrics.get(metric)

        if actual is None:
            status = "NOT_AVAILABLE"
            difference = None
        elif actual == expected:
            status = "MATCH"
            difference = 0
        else:
            status = "MISMATCH"
            difference = actual - expected

        metric_comparisons[metric] = {
            "expected": expected,
            "actual": actual,
            "difference": difference,
            "status": status,
        }

    all_metrics_match = all(
        comparison["status"] == "MATCH"
        for comparison in metric_comparisons.values()
    )

    return {
        "all_metrics_match": all_metrics_match,
        "metrics": metric_comparisons,
    }


def audit_oracle_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Chạy toàn bộ audit trên DataFrame đã đọc."""
    schema_summary = build_schema_snapshot(
        dataframe
    )
    schema_issues = audit_schema(dataframe)

    identifier_issues = audit_identifiers(dataframe)

    structure_issues, structure_summary = (
        audit_game_structure(dataframe)
    )

    player_issues, player_summary = (
        audit_player_rows(dataframe)
    )

    team_issues, team_summary = audit_team_rows(
        dataframe
    )

    result_issues, result_summary = audit_results(
        dataframe
    )

    time_issues, time_summary = audit_time(
        dataframe
    )

    (
        completeness_issues,
        completeness_summary,
    ) = audit_data_completeness(dataframe)

    issues = combine_issue_dataframes(
        schema_issues,
        identifier_issues,
        structure_issues,
        player_issues,
        team_issues,
        result_issues,
        time_issues,
        completeness_issues,
    )

    issue_summary = {
        "total_issues": len(issues),
        "affected_games": (
            0
            if issues.empty
            else to_python_int(
                issues["gameid"].dropna().nunique(),
                field_name="affected_games",
            )
        ),
        "by_code": issue_counts_by_column(
            issues,
            "issue_code",
        ),
        "by_group": issue_counts_by_column(
            issues,
            "issue_group",
        ),
        "by_severity": issue_counts_by_column(
            issues,
            "severity",
        ),
    }

    benchmark_comparison = build_benchmark_comparison(
        schema_summary=schema_summary,
        structure_summary=structure_summary,
        player_summary=player_summary,
        team_summary=team_summary,
        time_summary=time_summary,
        completeness_summary=completeness_summary,
        identifier_issues=identifier_issues,
    )

    summary: dict[str, object] = {
        "file": None,
        "schema": schema_summary,
        "data_completeness": completeness_summary,
        "identifiers": {
            "issue_count": len(identifier_issues),
        },
        "game_structure": structure_summary,
        "player_rows": player_summary,
        "team_rows": team_summary,
        "results": result_summary,
        "time": time_summary,
        "issues": issue_summary,
        "benchmark_comparison": (
            benchmark_comparison
        ),
    }

    return summary, issues


def audit_oracle_file(
    source_path: str | Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Đọc CSV một lần và chạy toàn bộ audit."""
    path = resolve_source_path(source_path)
    dataframe = read_oracle_csv(path)

    summary, issues = audit_oracle_dataframe(
        dataframe
    )

    summary["file"] = build_file_metadata(path)

    return summary, issues


def json_default(value: Any) -> Any:
    """Chuyển các scalar đặc biệt sang dạng JSON hỗ trợ."""
    if isinstance(value, Path):
        return str(value)

    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    if pd.isna(value):
        return None

    raise TypeError(
        f"Value is not JSON serializable: "
        f"{type(value).__name__}"
    )


def summary_mapping(
    summary: dict[str, object],
    key: str,
) -> dict[str, Any]:
    """Lấy một mapping lồng trong summary."""
    value = summary.get(key)

    if not isinstance(value, dict):
        return {}

    return {
        str(item_key): item_value
        for item_key, item_value in value.items()
    }


def markdown_cell(value: Any) -> str:
    """Định dạng một giá trị an toàn trong Markdown table."""
    if value is None:
        text = "N/A"
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            default=json_default,
        )
    else:
        text = str(value)

    return (
        text.replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def append_markdown_table(
    lines: list[str],
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
) -> None:
    """Thêm Markdown table vào danh sách dòng."""
    lines.append(
        "| "
        + " | ".join(headers)
        + " |"
    )

    lines.append(
        "| "
        + " | ".join("---" for _ in headers)
        + " |"
    )

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in row
            )
            + " |"
        )

    lines.append("")


def build_markdown_report(
    summary: dict[str, object],
) -> str:
    """Tạo báo cáo Markdown chỉ quan sát và mô tả."""
    file_summary = summary_mapping(
        summary,
        "file",
    )
    schema_summary = summary_mapping(
        summary,
        "schema",
    )
    completeness_summary = summary_mapping(
        summary,
        "data_completeness",
    )
    structure_summary = summary_mapping(
        summary,
        "game_structure",
    )
    player_summary = summary_mapping(
        summary,
        "player_rows",
    )
    team_summary = summary_mapping(
        summary,
        "team_rows",
    )
    result_summary = summary_mapping(
        summary,
        "results",
    )
    time_summary = summary_mapping(
        summary,
        "time",
    )
    issue_summary = summary_mapping(
        summary,
        "issues",
    )
    benchmark_summary = summary_mapping(
        summary,
        "benchmark_comparison",
    )

    player_missing = summary_mapping(
        player_summary,
        "missing_values",
    )
    team_missing = summary_mapping(
        team_summary,
        "missing_values",
    )
    cutoff_assessment = summary_mapping(
        time_summary,
        "cutoff_assessment",
    )
    benchmark_metrics = summary_mapping(
        benchmark_summary,
        "metrics",
    )

    lines: list[str] = [
        "# Oracle 2025 Data Quality Audit",
        "",
        "> Báo cáo này chỉ quan sát và ghi nhận chất lượng "
        "dữ liệu. Không làm sạch, sửa, loại dòng hoặc nhập "
        "dữ liệu vào cơ sở dữ liệu.",
        "",
        "## 1. Thông tin file",
        "",
    ]

    append_markdown_table(
        lines,
        (
            "Thuộc tính",
            "Giá trị",
        ),
        [
            ("Filename", file_summary.get("filename")),
            (
                "Source path",
                file_summary.get("source_path"),
            ),
            (
                "File size (bytes)",
                file_summary.get("file_size_bytes"),
            ),
            ("SHA-256", file_summary.get("sha256")),
            (
                "Last modified UTC",
                file_summary.get("last_modified_utc"),
            ),
            (
                "Audit timestamp UTC",
                file_summary.get("audit_timestamp_utc"),
            ),
        ],
    )

    lines.extend(
        [
            "## 2. Schema",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Total rows",
                schema_summary.get("total_rows"),
            ),
            (
                "Total columns",
                schema_summary.get("total_columns"),
            ),
            (
                "Missing required columns",
                schema_summary.get(
                    "missing_required_columns"
                ),
            ),
            (
                "Present optional columns",
                schema_summary.get(
                    "present_optional_columns"
                ),
            ),
        ],
    )

    column_names = schema_summary.get(
        "column_names"
    )

    if isinstance(column_names, list):
        lines.extend(
            [
                "### Column names",
                "",
                ", ".join(
                    f"`{markdown_cell(column)}`"
                    for column in column_names
                ),
                "",
            ]
        )

    pandas_dtypes = schema_summary.get(
        "pandas_dtypes"
    )

    if isinstance(pandas_dtypes, dict):
        dtype_rows = [
            (
                column,
                dtype,
            )
            for column, dtype in pandas_dtypes.items()
        ]

        lines.extend(
            [
                "### Inferred pandas dtypes",
                "",
            ]
        )

        append_markdown_table(
            lines,
            (
                "Column",
                "dtype",
            ),
            dtype_rows,
        )

    lines.extend(
        [
            "## 3. Tổng quan issues",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Total issues",
                issue_summary.get("total_issues"),
            ),
            (
                "Affected games",
                issue_summary.get("affected_games"),
            ),
            (
                "By severity",
                issue_summary.get("by_severity"),
            ),
            (
                "By group",
                issue_summary.get("by_group"),
            ),
            (
                "By code",
                issue_summary.get("by_code"),
            ),
        ],
    )

    lines.extend(
        [
            "## 4. Missing values quan trọng",
            "",
        ]
    )

    missing_rows = [
        (
            "Player row",
            field,
            count,
        )
        for field, count in player_missing.items()
    ]

    missing_rows.extend(
        (
            "Team row",
            field,
            count,
        )
        for field, count in team_missing.items()
    )

    append_markdown_table(
        lines,
        (
            "Row type",
            "Field",
            "Missing rows",
        ),
        missing_rows,
    )

    lines.extend(
        [
            "## 5. Cấu trúc game",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Distinct games",
                structure_summary.get("distinct_games"),
            ),
            (
                "Games with 12 rows",
                structure_summary.get(
                    "games_with_12_rows"
                ),
            ),
            (
                "Games with 10 player rows",
                structure_summary.get(
                    "games_with_10_player_rows"
                ),
            ),
            (
                "Games with 2 team rows",
                structure_summary.get(
                    "games_with_2_team_rows"
                ),
            ),
            (
                "Games with valid structure",
                structure_summary.get(
                    "games_with_valid_structure"
                ),
            ),
        ],
    )

    lines.extend(
        [
            "## 6. Data completeness và pick/ban",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Complete games",
                completeness_summary.get(
                    "complete_games"
                ),
            ),
            (
                "Partial games",
                completeness_summary.get(
                    "partial_games"
                ),
            ),
            (
                "Games missing pick",
                team_summary.get(
                    "games_missing_at_least_one_pick"
                ),
            ),
            (
                "Games missing ban",
                team_summary.get(
                    "games_missing_at_least_one_ban"
                ),
            ),
            (
                "Pick missing by field",
                team_summary.get(
                    "pick_missing_field_counts"
                ),
            ),
            (
                "Ban missing by field",
                team_summary.get(
                    "ban_missing_field_counts"
                ),
            ),
        ],
    )

    lines.extend(
        [
            "Thiếu pick hoặc ban được ghi nhận là quality "
            "issue. Báo cáo không tự sửa hoặc loại các ván "
            "liên quan.",
            "",
            "## 7. Result",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Result distribution",
                result_summary.get(
                    "result_value_counts"
                ),
            ),
            (
                "Invalid result rows",
                result_summary.get(
                    "invalid_result_rows"
                ),
            ),
            (
                "Invalid result structure games",
                result_summary.get(
                    "games_with_invalid_result_structure"
                ),
            ),
            (
                "Player-team mismatches",
                result_summary.get(
                    "player_team_result_mismatch_rows"
                ),
            ),
        ],
    )

    lines.extend(
        [
            "Kết quả thật trong bước này chỉ được dùng để "
            "audit tính nhất quán.",
            "",
            "## 8. Thời gian",
            "",
        ]
    )

    append_markdown_table(
        lines,
        (
            "Chỉ số",
            "Giá trị",
        ),
        [
            (
                "Minimum date UTC",
                time_summary.get("min_date_utc"),
            ),
            (
                "Maximum date UTC",
                time_summary.get("max_date_utc"),
            ),
            (
                "Missing date rows",
                time_summary.get(
                    "missing_date_rows"
                ),
            ),
            (
                "Invalid date rows",
                time_summary.get(
                    "invalid_date_rows"
                ),
            ),
            (
                "Year distribution by game",
                time_summary.get(
                    "year_distribution_by_game"
                ),
            ),
            (
                "Year-date mismatch rows",
                time_summary.get(
                    "year_date_mismatch_rows"
                ),
            ),
            (
                "Explicit game-end columns",
                time_summary.get(
                    "explicit_game_end_time_columns"
                ),
            ),
            (
                "Cutoff status",
                cutoff_assessment.get("status"),
            ),
            (
                "Cutoff assessment",
                cutoff_assessment.get("message"),
            ),
        ],
    )

    lines.extend(
        [
            "Không suy tạo thời điểm kết thúc từ `date` "
            "hoặc từ thời lượng ván.",
            "",
            "## 9. Benchmark comparison",
            "",
        ]
    )

    benchmark_rows: list[tuple[Any, ...]] = []

    for metric, raw_comparison in (
        benchmark_metrics.items()
    ):
        if not isinstance(raw_comparison, dict):
            continue

        benchmark_rows.append(
            (
                metric,
                raw_comparison.get("expected"),
                raw_comparison.get("actual"),
                raw_comparison.get("difference"),
                raw_comparison.get("status"),
            )
        )

    append_markdown_table(
        lines,
        (
            "Metric",
            "Expected",
            "Actual",
            "Difference",
            "Status",
        ),
        benchmark_rows,
    )

    lines.extend(
        [
            "## 10. Kết luận sơ bộ",
            "",
            (
                "- Benchmark comparison: "
                f"`{benchmark_summary.get('all_metrics_match')}`."
            ),
            (
                "- Tổng issues ghi nhận: "
                f"`{issue_summary.get('total_issues')}`."
            ),
            "- Các issue được giữ nguyên để truy vết.",
            "- Audit không quyết định xóa, sửa, điền thiếu "
            "hoặc nhập dữ liệu.",
            "",
        ]
    )

    return "\n".join(lines)


def validate_output_paths(
    summary: dict[str, object],
    output_paths: tuple[Path, ...],
) -> None:
    """Ngăn output trùng nhau hoặc ghi đè nguồn raw."""
    resolved_outputs = [
        path.resolve()
        for path in output_paths
    ]

    if len(set(resolved_outputs)) != len(
        resolved_outputs
    ):
        raise ValueError(
            "Audit output paths must be distinct."
        )

    file_summary = summary_mapping(
        summary,
        "file",
    )
    source_path_value = file_summary.get(
        "source_path"
    )

    if not isinstance(source_path_value, str):
        return

    source_path = Path(source_path_value).resolve()

    if source_path in resolved_outputs:
        raise ValueError(
            "Audit output must not overwrite "
            "the raw source file."
        )


def write_audit_outputs(
    *,
    summary: dict[str, object],
    issues: pd.DataFrame,
    summary_path: str | Path,
    issues_path: str | Path,
    report_path: str | Path,
) -> dict[str, str]:
    """Ghi JSON summary, CSV issues và Markdown report."""
    resolved_summary_path = Path(
        summary_path
    ).resolve()
    resolved_issues_path = Path(
        issues_path
    ).resolve()
    resolved_report_path = Path(
        report_path
    ).resolve()

    if resolved_summary_path.suffix.lower() != ".json":
        raise ValueError(
            "Summary output must use .json."
        )

    if resolved_issues_path.suffix.lower() != ".csv":
        raise ValueError(
            "Issues output must use .csv."
        )

    if resolved_report_path.suffix.lower() != ".md":
        raise ValueError(
            "Report output must use .md."
        )

    output_paths = (
        resolved_summary_path,
        resolved_issues_path,
        resolved_report_path,
    )

    validate_output_paths(
        summary,
        output_paths,
    )

    for output_path in output_paths:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    summary_json = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=json_default,
    )

    resolved_summary_path.write_text(
        summary_json + "\n",
        encoding="utf-8",
    )

    issues.loc[
        :,
        list(ISSUE_COLUMNS),
    ].to_csv(
        resolved_issues_path,
        index=False,
        encoding="utf-8",
    )

    markdown_report = build_markdown_report(
        summary
    )

    resolved_report_path.write_text(
        markdown_report,
        encoding="utf-8",
    )

    return {
        "summary_path": str(
            resolved_summary_path
        ),
        "issues_path": str(
            resolved_issues_path
        ),
        "report_path": str(
            resolved_report_path
        ),
    }
