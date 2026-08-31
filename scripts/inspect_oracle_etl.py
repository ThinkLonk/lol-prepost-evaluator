"""CLI read-only để kiểm tra transform Oracle cốt lõi của Bước 6B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from match_insight.data_processing.oracle_audit import (
    calculate_sha256,
    resolve_source_path,
)
from match_insight.data_processing.oracle_etl import (
    OracleCoreData,
    prepare_oracle_core_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oracle"
    / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
)


def parse_args() -> argparse.Namespace:
    """Đọc duy nhất đường dẫn CSV nguồn."""
    parser = argparse.ArgumentParser(
        description=(
            "Đọc và kiểm tra các cột Oracle cốt lõi mà không ghi file "
            "hoặc truy cập PostgreSQL."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=(
            "Đường dẫn CSV nguồn. Mặc định dùng file Oracle 2025 "
            "trong data/raw/oracle."
        ),
    )
    return parser.parse_args()


def _json_value(value: object) -> str:
    """Mã hóa list/map ổn định để output CLI dễ đối chiếu."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def print_inspection_result(
    *,
    source_path: Path,
    source_sha256_before: str,
    source_sha256_after: str,
    core_data: OracleCoreData,
) -> None:
    """In các số liệu được tính trực tiếp từ transform read-only."""
    dataframe = core_data.dataframe
    sha256_unchanged = (
        source_sha256_before == source_sha256_after
    )
    position_values = list(
        core_data.position_counts
    )
    etl_dtypes = {
        column: str(dtype)
        for column, dtype in dataframe.dtypes.items()
    }

    print(f"source={source_path}")
    print(
        "source_sha256_before="
        f"{source_sha256_before}"
    )
    print(
        "source_sha256_after="
        f"{source_sha256_after}"
    )
    print(
        "source_sha256_unchanged="
        f"{str(sha256_unchanged).lower()}"
    )
    print(f"source_rows={len(dataframe)}")
    print(
        f"player_rows={len(core_data.player_rows)}"
    )
    print(f"team_rows={len(core_data.team_rows)}")
    print(
        "distinct_games="
        f"{int(dataframe['gameid'].nunique(dropna=True))}"
    )
    print(
        "position_values="
        f"{_json_value(position_values)}"
    )
    print(
        "position_counts="
        f"{_json_value(core_data.position_counts)}"
    )
    print(
        "position_missing_rows="
        f"{core_data.missing_position_rows}"
    )
    print(f"etl_column_count={dataframe.shape[1]}")
    print(
        "etl_columns="
        f"{_json_value(dataframe.columns.tolist())}"
    )
    print(
        "etl_dtypes="
        f"{_json_value(etl_dtypes)}"
    )
    print("cutoff_status=UNAVAILABLE")
    print("postgresql_changed=false")


def main() -> None:
    """Chạy inspect mà không tạo output hoặc kết nối database."""
    args = parse_args()
    source_path = resolve_source_path(
        args.source
    )
    source_sha256_before = calculate_sha256(
        source_path
    )
    core_data = prepare_oracle_core_data(
        source_path
    )
    source_sha256_after = calculate_sha256(
        source_path
    )

    print_inspection_result(
        source_path=source_path,
        source_sha256_before=(
            source_sha256_before
        ),
        source_sha256_after=source_sha256_after,
        core_data=core_data,
    )

    if source_sha256_before != source_sha256_after:
        raise RuntimeError(
            "Oracle source SHA-256 changed during read-only inspection."
        )


if __name__ == "__main__":
    main()
