"""Apply an approved Oracle ETL dry-run in one PostgreSQL transaction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from match_insight.data_processing.oracle_audit import (
    calculate_sha256,
    resolve_source_path,
)
from match_insight.data_processing.oracle_etl import prepare_oracle_core_data
from match_insight.data_processing.oracle_etl_report import (
    build_etl_report_summary,
)
from match_insight.data_processing.oracle_transform import (
    TARGET_TABLE_ORDER,
    OracleDryRunResult,
)
from match_insight.database.engine import engine
from match_insight.database.models import (
    Evaluation,
    EvaluationWarning,
    TeamMembership,
)
from match_insight.database.oracle_apply import (
    ALREADY_APPLIED,
    APPROVED_INITIAL_STATE,
    OracleApplyResult,
    apply_oracle_records,
    build_approval_contract,
    classify_approval_state,
)
from scripts.inspect_oracle_transform import (
    build_transform_result,
    database_snapshot_counts,
    materialize_database_snapshot,
)


def parse_args() -> argparse.Namespace:
    """Require an approved partial report and an explicit commit decision."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply the approved Oracle ETL records atomically to PostgreSQL."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Audited Oracle raw CSV used by the approved dry-run.",
    )
    parser.add_argument(
        "--approved-summary",
        type=Path,
        required=True,
        help="Approved oracle_2025_etl_summary.json path.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        required=True,
        help="Explicitly accept the approved PARTIAL reconciliation scope.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        required=True,
        help="Explicitly authorize the single transactional PostgreSQL apply.",
    )
    return parser.parse_args()


def _json_value(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_approved_summary(path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() != ".json":
        raise ValueError("The approved ETL summary must use .json.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The approved ETL summary must contain a JSON object.")
    return resolved, {str(key): value for key, value in payload.items()}


def _protected_counts(connection: Connection) -> dict[str, int]:
    """Read tables that Step 6F must never populate or modify."""
    return {
        "evaluation": int(
            connection.execute(
                select(func.count()).select_from(Evaluation)
            ).scalar_one()
        ),
        "evaluation_warning": int(
            connection.execute(
                select(func.count()).select_from(EvaluationWarning)
            ).scalar_one()
        ),
        "team_membership": int(
            connection.execute(
                select(func.count()).select_from(TeamMembership)
            ).scalar_one()
        ),
    }


def _require_protected_state(
    *,
    before: dict[str, int],
    after: dict[str, int] | None = None,
) -> None:
    if before["evaluation"] != 0 or before["evaluation_warning"] != 0:
        raise RuntimeError(
            "Step 6F requires evaluation and evaluation_warning to be empty."
        )
    if after is None:
        return
    if after["evaluation"] != 0 or after["evaluation_warning"] != 0:
        raise RuntimeError(
            "Oracle apply changed an evaluation table unexpectedly."
        )
    if after["team_membership"] != before["team_membership"]:
        raise RuntimeError(
            "Oracle apply changed team_membership unexpectedly."
        )


def _enriched_current_summary(
    *,
    result: OracleDryRunResult,
    transform_summary: dict[str, object],
    source_path: Path,
    source_sha256_before: str,
    source_sha256_after: str,
    target_snapshot_rows: dict[str, int],
) -> dict[str, object]:
    summary = dict(transform_summary)
    summary["transaction_status"] = "WRITE_TRANSACTION_OPEN"
    summary["postgresql_changed"] = False
    return build_etl_report_summary(
        result=result,
        transform_summary=summary,
        source_path=source_path,
        source_sha256_before=source_sha256_before,
        source_sha256_after=source_sha256_after,
        target_snapshot_rows=target_snapshot_rows,
        run_timestamp_utc=datetime.now(timezone.utc),
        reconciliation_valid=True,
    )


def _apply_stats(result: OracleApplyResult) -> dict[str, dict[str, int]]:
    return {
        table: {
            "inserted": int(result.per_table[table].inserted),
            "updated": int(result.per_table[table].updated),
            "skipped": int(result.per_table[table].skipped),
        }
        for table in TARGET_TABLE_ORDER
    }


def _changed_rows(stats: dict[str, dict[str, int]]) -> int:
    return sum(
        counts["inserted"] + counts["updated"]
        for counts in stats.values()
    )


def execute_approved_apply(
    *,
    source: str | Path,
    approved_summary_path: str | Path,
    allow_partial: bool,
    commit: bool,
) -> dict[str, object]:
    """Re-run, apply, and verify the approved records before one commit."""
    if not allow_partial:
        raise ValueError("Step 6F requires explicit --allow-partial approval.")
    if not commit:
        raise ValueError("Step 6F requires explicit --commit authorization.")

    source_path = resolve_source_path(source)
    approved_path, approved_summary = _load_approved_summary(
        approved_summary_path
    )
    build_approval_contract(approved_summary)

    if approved_summary.get("dry_run_status") != "PARTIAL":
        raise ValueError(
            "--allow-partial requires an approved PARTIAL dry-run summary."
        )

    source_sha256_before = calculate_sha256(source_path)
    core_data = prepare_oracle_core_data(source_path)
    source_sha256_after_read = calculate_sha256(source_path)
    if source_sha256_before != source_sha256_after_read:
        raise RuntimeError("Oracle source changed while it was being prepared.")

    output: dict[str, object]
    with engine.connect().execution_options(
        isolation_level="SERIALIZABLE"
    ) as connection:
        with connection.begin():
            protected_before = _protected_counts(connection)
            _require_protected_state(before=protected_before)

            initial_snapshot = materialize_database_snapshot(connection)
            current_result, current_transform_summary = build_transform_result(
                core_data,
                initial_snapshot,
            )
            current_summary = _enriched_current_summary(
                result=current_result,
                transform_summary=current_transform_summary,
                source_path=source_path,
                source_sha256_before=source_sha256_before,
                source_sha256_after=source_sha256_after_read,
                target_snapshot_rows=database_snapshot_counts(
                    initial_snapshot
                ),
            )
            approval_state = classify_approval_state(
                approved_summary,
                current_summary,
            )
            if approval_state not in {
                APPROVED_INITIAL_STATE,
                ALREADY_APPLIED,
            }:
                raise RuntimeError(
                    f"Unsupported Oracle approval state: {approval_state}"
                )

            apply_result = apply_oracle_records(connection, current_result)
            if apply_result.state != approval_state:
                raise RuntimeError(
                    "Apply state does not match the approved dry-run state."
                )

            post_snapshot = materialize_database_snapshot(connection)
            post_result, post_transform_summary = build_transform_result(
                core_data,
                post_snapshot,
            )
            source_sha256_final = calculate_sha256(source_path)
            if source_sha256_final != source_sha256_before:
                raise RuntimeError(
                    "Oracle source changed before the transaction could commit."
                )
            post_summary = _enriched_current_summary(
                result=post_result,
                transform_summary=post_transform_summary,
                source_path=source_path,
                source_sha256_before=source_sha256_before,
                source_sha256_after=source_sha256_final,
                target_snapshot_rows=database_snapshot_counts(post_snapshot),
            )
            post_state = classify_approval_state(
                approved_summary,
                post_summary,
            )
            if post_state != ALREADY_APPLIED:
                raise RuntimeError(
                    "Post-apply verification is not idempotent; rolling back."
                )

            protected_after = _protected_counts(connection)
            _require_protected_state(
                before=protected_before,
                after=protected_after,
            )
            stats = _apply_stats(apply_result)
            output = {
                "source": str(source_path),
                "approved_summary": str(approved_path),
                "source_sha256_before": source_sha256_before,
                "source_sha256_final": source_sha256_final,
                "source_sha256_unchanged": True,
                "mode": "APPLY",
                "approval_state": approval_state,
                "approved_dry_run_status": approved_summary.get(
                    "dry_run_status"
                ),
                "current_dry_run_status": current_summary.get(
                    "dry_run_status"
                ),
                "transformed_records": current_summary.get(
                    "transformed_records"
                ),
                "planned_action_counts": current_summary.get("actions"),
                "apply_counts": stats,
                "pre_target_snapshot_rows": database_snapshot_counts(
                    initial_snapshot
                ),
                "post_target_snapshot_rows": database_snapshot_counts(
                    post_snapshot
                ),
                "post_approval_state": post_state,
                "post_action_counts": post_summary.get("actions"),
                "evaluation_count": protected_after["evaluation"],
                "evaluation_warning_count": protected_after[
                    "evaluation_warning"
                ],
                "team_membership_before": protected_before[
                    "team_membership"
                ],
                "team_membership_after": protected_after[
                    "team_membership"
                ],
                "idempotency_preview": True,
                "transaction_status": "COMMITTED",
                "postgresql_changed": _changed_rows(stats) > 0,
            }
    return output


def print_apply_result(output: dict[str, object]) -> None:
    """Print deterministic apply and post-apply reconciliation values."""
    scalar_keys = (
        "source",
        "approved_summary",
        "source_sha256_before",
        "source_sha256_final",
        "source_sha256_unchanged",
        "mode",
        "approval_state",
        "approved_dry_run_status",
        "current_dry_run_status",
    )
    for key in scalar_keys:
        value = output[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(f"{key}={value}")

    mapping_keys = (
        "transformed_records",
        "planned_action_counts",
        "apply_counts",
        "pre_target_snapshot_rows",
        "post_target_snapshot_rows",
        "post_action_counts",
    )
    for key in mapping_keys:
        print(f"{key}={_json_value(output[key])}")

    trailing_keys = (
        "post_approval_state",
        "evaluation_count",
        "evaluation_warning_count",
        "team_membership_before",
        "team_membership_after",
        "idempotency_preview",
        "transaction_status",
        "postgresql_changed",
    )
    for key in trailing_keys:
        value = output[key]
        if isinstance(value, bool):
            value = str(value).lower()
        print(f"{key}={value}")


def main() -> None:
    """Run the explicitly authorized Oracle PostgreSQL apply."""
    args = parse_args()
    try:
        output = execute_approved_apply(
            source=args.source,
            approved_summary_path=args.approved_summary,
            allow_partial=args.allow_partial,
            commit=args.commit,
        )
    except Exception:
        print("transaction_status=ROLLED_BACK")
        print("postgresql_changed=false")
        raise
    print_apply_result(output)


if __name__ == "__main__":
    main()
