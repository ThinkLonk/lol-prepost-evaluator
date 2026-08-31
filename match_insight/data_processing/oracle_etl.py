"""Đọc và chuẩn hóa các cột Oracle cốt lõi cho ETL Bước 6B.

Cột ``date`` trong mô-đun này chỉ là timestamp nguồn. Không suy diễn
thời điểm bắt đầu, thời điểm kết thúc hoặc mốc cutoff của ván đấu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from match_insight.data_processing.oracle_audit import resolve_source_path

ORACLE_CORE_COLUMNS: Final[tuple[str, ...]] = (
    "gameid",
    "datacompleteness",
    "url",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "champion",
    "gamelength",
    "result",
)

ORACLE_INTEGER_COLUMNS: Final[tuple[str, ...]] = (
    "year",
    "playoffs",
    "game",
    "participantid",
    "gamelength",
    "result",
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

VALID_POSITION_VALUES: Final[frozenset[str]] = (
    PLAYER_POSITIONS | {TEAM_POSITION}
)


@dataclass(frozen=True)
class OracleCoreData:
    """Các DataFrame cốt lõi sau khi đọc, chuẩn hóa và phân loại."""

    dataframe: pd.DataFrame
    player_rows: pd.DataFrame
    team_rows: pd.DataFrame
    position_counts: dict[str, int]
    missing_position_rows: int


def _missing_string_mask(series: pd.Series) -> pd.Series:
    """Xác định giá trị thiếu hoặc chuỗi chỉ chứa khoảng trắng."""
    values = series.astype("string")
    blank_mask = values.str.strip().eq("").fillna(False)
    return (values.isna() | blank_mask).astype(bool)


def _missing_core_columns(columns: pd.Index) -> list[str]:
    """Liệt kê cột cốt lõi không xuất hiện theo đúng thứ tự hợp đồng."""
    actual_columns = set(columns)
    return [
        column
        for column in ORACLE_CORE_COLUMNS
        if column not in actual_columns
    ]


def _read_csv_header(source_path: Path) -> pd.Index:
    """Đọc riêng header để báo lỗi schema trước khi đọc dữ liệu."""
    try:
        header = pd.read_csv(
            source_path,
            nrows=0,
            encoding="utf-8",
            keep_default_na=False,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            f"Could not read Oracle CSV header: {source_path}"
        ) from error

    return header.columns


def read_oracle_core_csv(
    source_path: str | Path,
) -> pd.DataFrame:
    """Đọc đúng 20 cột cốt lõi và giữ giá trị nguồn ở dạng chuỗi."""
    path = resolve_source_path(source_path)
    missing_columns = _missing_core_columns(
        _read_csv_header(path)
    )

    if missing_columns:
        raise ValueError(
            "Oracle CSV is missing core columns: "
            + ", ".join(missing_columns)
        )

    try:
        dataframe = pd.read_csv(
            path,
            usecols=list(ORACLE_CORE_COLUMNS),
            dtype="string",
            encoding="utf-8",
            keep_default_na=False,
            low_memory=False,
        )
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            f"Could not read Oracle core columns: {path}"
        ) from error

    return dataframe.loc[
        :,
        list(ORACLE_CORE_COLUMNS),
    ].copy()


def _invalid_samples(
    series: pd.Series,
    invalid_mask: pd.Series,
) -> list[str]:
    """Lấy tối đa năm giá trị lỗi để thông báo có căn cứ."""
    return (
        series.loc[invalid_mask]
        .astype("string")
        .drop_duplicates()
        .head(5)
        .tolist()
    )


def _normalize_nullable_integer(
    series: pd.Series,
    *,
    column: str,
) -> pd.Series:
    """Chuyển cột số nguyên sang Int64 mà không che giá trị lỗi."""
    source_values = series.astype("string")
    missing_mask = source_values.isna()
    numeric_values = pd.to_numeric(
        source_values,
        errors="coerce",
    )

    non_numeric_mask = (~missing_mask) & numeric_values.isna()
    if bool(non_numeric_mask.any()):
        samples = _invalid_samples(
            source_values,
            non_numeric_mask,
        )
        raise ValueError(
            f"Invalid integer values in Oracle column '{column}': {samples}"
        )

    non_missing_values = numeric_values.dropna()
    non_finite_mask = ~non_missing_values.map(
        lambda value: pd.notna(value)
        and float("-inf") < float(value) < float("inf")
    )
    if bool(non_finite_mask.any()):
        invalid_indices = non_finite_mask.loc[
            non_finite_mask
        ].index
        samples = (
            source_values.loc[invalid_indices]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Non-finite integer values in Oracle column '{column}': {samples}"
        )

    fractional_mask = non_missing_values.mod(1).ne(0)
    if bool(fractional_mask.any()):
        invalid_indices = fractional_mask.loc[
            fractional_mask
        ].index
        samples = (
            source_values.loc[invalid_indices]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"Fractional values in Oracle integer column '{column}': {samples}"
        )

    try:
        return numeric_values.astype("Int64")
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            f"Oracle column '{column}' contains integers outside Int64 range."
        ) from error


def _normalize_source_timestamp(
    series: pd.Series,
) -> pd.Series:
    """Parse ``date`` thành UTC nhưng không gán ngữ nghĩa game start/end."""
    source_values = series.astype("string")
    parsed_values = pd.to_datetime(
        source_values,
        errors="coerce",
        format="mixed",
        utc=True,
    )
    invalid_mask = source_values.notna() & parsed_values.isna()

    if bool(invalid_mask.any()):
        samples = _invalid_samples(
            source_values,
            invalid_mask,
        )
        raise ValueError(
            "Invalid source timestamps in Oracle column 'date': "
            f"{samples}"
        )

    return parsed_values


def normalize_oracle_core_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Chuẩn hóa kiểu dữ liệu mà không đổi ý nghĩa giá trị nguồn."""
    missing_columns = _missing_core_columns(
        dataframe.columns
    )
    if missing_columns:
        raise ValueError(
            "Oracle DataFrame is missing core columns: "
            + ", ".join(missing_columns)
        )

    normalized = dataframe.loc[
        :,
        list(ORACLE_CORE_COLUMNS),
    ].copy()

    for column in ORACLE_CORE_COLUMNS:
        values = normalized[column].astype("string")
        normalized[column] = values.mask(
            _missing_string_mask(values),
            pd.NA,
        )

    for column in ORACLE_INTEGER_COLUMNS:
        normalized[column] = (
            _normalize_nullable_integer(
                normalized[column],
                column=column,
            )
        )

    normalized["date"] = _normalize_source_timestamp(
        normalized["date"]
    )

    return normalized


def inspect_position_values(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, int], int]:
    """Ghi nhận actual ``position`` values trước khi phân loại dòng."""
    if "position" not in dataframe.columns:
        raise ValueError(
            "Oracle DataFrame is missing core column: position"
        )

    position_values = dataframe["position"].astype(
        "string"
    )
    missing_mask = _missing_string_mask(
        position_values
    )
    counts_series = (
        position_values.loc[~missing_mask]
        .value_counts()
        .sort_index()
    )
    counts = {
        str(value): int(count)
        for value, count in counts_series.items()
    }

    return counts, int(missing_mask.sum())


def _validate_position_values(
    position_counts: dict[str, int],
    missing_position_rows: int,
) -> None:
    """Từ chối position thiếu hoặc không thuộc vocabulary nguồn đã biết."""
    unexpected_values = sorted(
        set(position_counts) - VALID_POSITION_VALUES
    )

    if missing_position_rows or unexpected_values:
        raise ValueError(
            "Oracle position values cannot be classified: "
            f"missing_rows={missing_position_rows}, "
            f"unexpected_values={unexpected_values}, "
            f"actual_counts={position_counts}"
        )


def _select_classified_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tạo hai tập dòng chỉ từ giá trị ``position`` đã kiểm tra."""
    position_values = dataframe["position"].astype(
        "string"
    )
    player_mask = position_values.isin(
        PLAYER_POSITIONS
    ).fillna(False)
    team_mask = position_values.eq(
        TEAM_POSITION
    ).fillna(False)

    if bool((player_mask & team_mask).any()):
        raise ValueError(
            "Oracle player/team row masks overlap."
        )

    if not bool((player_mask | team_mask).all()):
        raise ValueError(
            "Oracle player/team row masks do not cover every source row."
        )

    return (
        dataframe.loc[player_mask].copy(),
        dataframe.loc[team_mask].copy(),
    )


def classify_oracle_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kiểm tra actual position rồi phân loại player/team rows."""
    position_counts, missing_position_rows = (
        inspect_position_values(dataframe)
    )
    _validate_position_values(
        position_counts,
        missing_position_rows,
    )
    return _select_classified_rows(dataframe)


def prepare_oracle_core_data(
    source_path: str | Path,
) -> OracleCoreData:
    """Đọc, chuẩn hóa và phân loại CSV mà không ghi output hay database."""
    source_dataframe = read_oracle_core_csv(
        source_path
    )
    normalized_dataframe = (
        normalize_oracle_core_dataframe(
            source_dataframe
        )
    )
    position_counts, missing_position_rows = (
        inspect_position_values(
            normalized_dataframe
        )
    )
    _validate_position_values(
        position_counts,
        missing_position_rows,
    )
    player_rows, team_rows = _select_classified_rows(
        normalized_dataframe
    )

    return OracleCoreData(
        dataframe=normalized_dataframe,
        player_rows=player_rows,
        team_rows=team_rows,
        position_counts=position_counts,
        missing_position_rows=missing_position_rows,
    )
