"""CLI chạy audit chất lượng dữ liệu Oracle's Elixir."""

from __future__ import annotations

import argparse
from pathlib import Path

from match_insight.data_processing.oracle_audit import (
    audit_oracle_file,
    summary_mapping,
    write_audit_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "oracle"
    / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "oracle_2025_summary.json"
)

DEFAULT_ISSUES_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "oracle_2025_issues.csv"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_quality"
    / "oracle_2025_report.md"
)


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit CSV Oracle's Elixir và xuất báo cáo "
            "chất lượng dữ liệu. Không làm sạch hoặc nhập "
            "dữ liệu vào database."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help=(
            "Đường dẫn CSV nguồn. Mặc định dùng file "
            "Oracle 2025 trong data/raw/oracle."
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Đường dẫn JSON summary.",
    )

    parser.add_argument(
        "--issues-output",
        type=Path,
        default=DEFAULT_ISSUES_PATH,
        help="Đường dẫn CSV issues.",
    )

    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Đường dẫn Markdown report.",
    )

    return parser.parse_args()


def print_audit_result(
    *,
    source_path: Path,
    summary: dict[str, object],
    issue_shape: tuple[int, int],
    output_paths: dict[str, str],
) -> None:
    """In kết quả thực thi ngắn gọn."""
    issue_summary = summary_mapping(
        summary,
        "issues",
    )
    benchmark_summary = summary_mapping(
        summary,
        "benchmark_comparison",
    )

    print(f"source={source_path.resolve()}")
    print(
        "total_issues="
        f"{issue_summary.get('total_issues')}"
    )
    print(
        "affected_games="
        f"{issue_summary.get('affected_games')}"
    )
    print(
        "issues_by_severity="
        f"{issue_summary.get('by_severity')}"
    )
    print(
        "benchmark_all_match="
        f"{benchmark_summary.get('all_metrics_match')}"
    )
    print(f"issues_shape={issue_shape}")

    for output_name, output_path in (
        output_paths.items()
    ):
        print(f"{output_name}={output_path}")


def main() -> None:
    """Chạy audit và ghi ba output chất lượng dữ liệu."""
    args = parse_args()

    summary, issues = audit_oracle_file(
        args.source
    )

    output_paths = write_audit_outputs(
        summary=summary,
        issues=issues,
        summary_path=args.summary_output,
        issues_path=args.issues_output,
        report_path=args.report_output,
    )

    print_audit_result(
        source_path=args.source,
        summary=summary,
        issue_shape=issues.shape,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
