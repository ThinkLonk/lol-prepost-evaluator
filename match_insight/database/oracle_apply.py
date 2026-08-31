"""Transactional PostgreSQL writes for approved Oracle ETL records.

The caller owns the transaction.  This module neither opens nor commits one;
any exception is therefore available to the caller as a full-apply rollback.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd
from sqlalchemy import and_, bindparam, func, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.engine import Connection
from sqlalchemy.sql.schema import Table

from match_insight.data_processing.oracle_transform import (
    TARGET_TABLE_ORDER,
    OracleDryRunResult,
)
from match_insight.database.models import (
    Game,
    GamePatch,
    GamePlayer,
    GameTeam,
    Player,
    Series,
    Team,
    Tournament,
    TournamentStage,
)

APPROVED_INITIAL_STATE: Final[str] = "APPROVED_INITIAL_STATE"
ALREADY_APPLIED: Final[str] = "ALREADY_APPLIED"

_ACTION_INSERT: Final[str] = "INSERT"
_ACTION_UPDATE: Final[str] = "UPDATE"
_ACTION_SKIP: Final[str] = "SKIP"
_SUMMARY_ACTION_FIELDS: Final[tuple[str, ...]] = (
    "inserted_expected",
    "updated_expected",
    "skipped",
)
_VOLATILE_SUMMARY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "action_totals",
        "actions",
        "database_transaction",
        "postgresql_changed",
        "run_timestamp_utc",
        "skipped",
        "source_path",
        "target_snapshot_rows",
        "transaction_status",
    }
)

_FRAME_ATTRIBUTES: Final[dict[str, str]] = {
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
_TABLES: Final[dict[str, Table]] = {
    "game_patch": GamePatch.__table__,
    "tournament": Tournament.__table__,
    "tournament_stage": TournamentStage.__table__,
    "series": Series.__table__,
    "team": Team.__table__,
    "player": Player.__table__,
    "game": Game.__table__,
    "game_team": GameTeam.__table__,
    "game_player": GamePlayer.__table__,
}
_LOGICAL_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "game_patch": ("patch_id",),
    "tournament": ("tournament_id",),
    "tournament_stage": ("stage_id",),
    "series": ("series_id",),
    "team": ("team_id",),
    "player": ("player_id",),
    "game": ("game_id",),
    "game_team": ("game_id", "side"),
    "game_player": ("game_id", "side", "role"),
}
_UPDATE_SCOPE: Final[dict[str, tuple[str, ...]]] = {
    "game_patch": ("patch_id",),
    "tournament": ("tournament_id",),
    "tournament_stage": ("stage_id",),
    "series": ("series_id",),
    "team": ("team_id", "oracle_team_id"),
    "player": ("player_id", "oracle_player_id"),
    "game": ("game_id",),
    "game_team": ("game_id", "side"),
    "game_player": ("game_team_id", "role"),
}
_UPDATE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "game_patch": ("patch_name",),
    "tournament": ("name", "region", "season"),
    "tournament_stage": ("tournament_id", "name"),
    "series": ("stage_id", "best_of"),
    "team": ("canonical_name", "display_name", "logo_file"),
    "player": ("canonical_name", "display_name", "photo_file"),
    "game": (
        "series_id",
        "stage_id",
        "patch_id",
        "game_number",
        "scheduled_at",
        "started_at",
        "ended_at",
        "winner_team_id",
    ),
    "game_team": ("team_id", "confirmation_status"),
    "game_player": (
        "player_id",
        "champion_id",
        "confirmation_status",
    ),
}
_PRESERVE_ON_NULL: Final[dict[str, frozenset[str]]] = {
    "team": frozenset({"logo_file"}),
    "player": frozenset({"photo_file"}),
    "game": frozenset({"started_at", "ended_at"}),
}


@dataclass(frozen=True)
class TableApplyStats:
    """Write outcome for one target table."""

    inserted: int
    updated: int
    skipped: int


@dataclass(frozen=True)
class OracleApplyResult:
    """Transaction-local outcome in foreign-key order."""

    state: str
    per_table: dict[str, TableApplyStats]


def _plain_value(value: object) -> object:
    """Convert pandas/numpy scalars and nested mappings to stable values."""
    if isinstance(value, Mapping):
        return {
            str(key): _plain_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _required_mapping(
    summary: Mapping[str, object],
    field: str,
) -> dict[str, object]:
    value = summary.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"Oracle approval summary field '{field}' must be a mapping.")
    return {
        str(key): _plain_value(item)
        for key, item in value.items()
    }


def _summary_table_counts(
    summary: Mapping[str, object],
    field: str,
) -> dict[str, int]:
    values = _required_mapping(summary, field)
    missing = sorted(set(TARGET_TABLE_ORDER) - set(values))
    if missing:
        raise ValueError(
            f"Oracle approval summary field '{field}' is missing tables: "
            + ", ".join(missing)
        )
    result: dict[str, int] = {}
    for table in TARGET_TABLE_ORDER:
        count = int(values[table])
        if count < 0:
            raise ValueError(f"Oracle approval count cannot be negative: {field}.{table}")
        result[table] = count
    return result


def _summary_actions(
    summary: Mapping[str, object],
    transformed: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    values = _required_mapping(summary, "actions")
    missing = sorted(set(TARGET_TABLE_ORDER) - set(values))
    if missing:
        raise ValueError(
            "Oracle approval actions are missing tables: " + ", ".join(missing)
        )

    normalized: dict[str, dict[str, int]] = {}
    for table in TARGET_TABLE_ORDER:
        table_value = values[table]
        if not isinstance(table_value, Mapping):
            raise ValueError(f"Oracle approval actions for {table} must be a mapping.")
        counts = {
            field: int(table_value.get(field, -1))
            for field in _SUMMARY_ACTION_FIELDS
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"Oracle approval actions for {table} are incomplete.")
        if sum(counts.values()) != int(transformed[table]):
            raise ValueError(
                f"Oracle approval actions do not reconcile for table {table}."
            )
        normalized[table] = counts
    return normalized


def _snapshot_counts(summary: Mapping[str, object]) -> dict[str, int]:
    values = _required_mapping(summary, "target_snapshot_rows")
    missing = sorted(set(TARGET_TABLE_ORDER) - set(values))
    if missing:
        raise ValueError(
            "Oracle target snapshot is missing tables: " + ", ".join(missing)
        )
    result = {str(key): int(value) for key, value in values.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError("Oracle target snapshot counts cannot be negative.")
    return dict(sorted(result.items()))


def _semantic_summary(summary: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): _plain_value(value)
        for key, value in sorted(summary.items())
        if str(key) not in _VOLATILE_SUMMARY_FIELDS
    }


def build_approval_contract(
    approved_summary: Mapping[str, object],
) -> dict[str, object]:
    """Validate and normalize the user-approved Step 6E dry-run summary."""
    safety_checks = (
        (approved_summary.get("mode") == "DRY_RUN", "mode must be DRY_RUN"),
        (
            approved_summary.get("source_sha256_unchanged") is True,
            "source SHA-256 must be unchanged",
        ),
        (
            approved_summary.get("transaction_status")
            == "READ_ONLY_ROLLED_BACK",
            "approved transaction must be read-only and rolled back",
        ),
        (
            approved_summary.get("postgresql_changed") is False,
            "approved dry-run must not change PostgreSQL",
        ),
        (
            approved_summary.get("reconciliation_valid") is True,
            "approved structural reconciliation must be valid",
        ),
        (
            approved_summary.get("grain_reconciliation_valid") is True,
            "approved grain reconciliation must be valid",
        ),
    )
    failures = [message for passed, message in safety_checks if not passed]
    if failures:
        raise ValueError("Unsafe Oracle approval summary: " + "; ".join(failures))

    sha_before = approved_summary.get("source_sha256_before")
    sha_after = approved_summary.get("source_sha256_after")
    approved_sha = approved_summary.get("source_sha256")
    if not (
        isinstance(approved_sha, str)
        and approved_sha
        and sha_before == approved_sha
        and sha_after == approved_sha
    ):
        raise ValueError("Approved Oracle source SHA-256 fields do not agree.")

    transformed = _summary_table_counts(
        approved_summary,
        "transformed_records",
    )
    actions = _summary_actions(approved_summary, transformed)
    initial_snapshot = _snapshot_counts(approved_summary)
    predicted_snapshot = dict(initial_snapshot)
    replay_actions: dict[str, dict[str, int]] = {}
    for table in TARGET_TABLE_ORDER:
        predicted_snapshot[table] += actions[table]["inserted_expected"]
        replay_actions[table] = {
            "inserted_expected": 0,
            "updated_expected": 0,
            "skipped": transformed[table],
        }

    return {
        "semantic_summary": _semantic_summary(approved_summary),
        "initial_target_snapshot_rows": initial_snapshot,
        "initial_actions": actions,
        "predicted_post_snapshot_rows": predicted_snapshot,
        "replay_actions": replay_actions,
    }


def classify_approval_state(
    approved_summary: Mapping[str, object],
    current_summary: Mapping[str, object],
) -> str:
    """Accept only the exact approved initial state or its all-SKIP replay."""
    contract = build_approval_contract(approved_summary)
    approved_semantic = contract["semantic_summary"]
    if _semantic_summary(current_summary) != approved_semantic:
        raise ValueError(
            "Current Oracle transform semantics do not match the approved dry-run."
        )

    transformed = _summary_table_counts(current_summary, "transformed_records")
    current_actions = _summary_actions(current_summary, transformed)
    current_snapshot = _snapshot_counts(current_summary)
    if (
        current_snapshot == contract["initial_target_snapshot_rows"]
        and current_actions == contract["initial_actions"]
    ):
        return APPROVED_INITIAL_STATE
    if (
        current_snapshot == contract["predicted_post_snapshot_rows"]
        and current_actions == contract["replay_actions"]
    ):
        return ALREADY_APPLIED
    raise ValueError(
        "Current Oracle database state is neither the approved initial state "
        "nor the predicted already-applied state."
    )


def _chunks(rows: Sequence[dict[str, object]], size: int) -> Iterator[list[dict[str, object]]]:
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def _source_key(record: Mapping[str, object], keys: Sequence[str]) -> str:
    return "|".join(str(record[key]) for key in keys)


def _records_by_action(
    result: OracleDryRunResult,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    required_action_columns = {"table", "source_key", "action"}
    if not required_action_columns <= set(result.actions.columns):
        missing = sorted(required_action_columns - set(result.actions.columns))
        raise ValueError("Oracle actions are missing columns: " + ", ".join(missing))

    unknown_tables = sorted(
        set(result.actions["table"].astype(str)) - set(TARGET_TABLE_ORDER)
    )
    if unknown_tables:
        raise ValueError("Oracle actions contain unknown tables: " + ", ".join(unknown_tables))

    partitioned: dict[str, dict[str, list[dict[str, object]]]] = {}
    for table in TARGET_TABLE_ORDER:
        frame = getattr(result.records, _FRAME_ATTRIBUTES[table])
        logical_keys = _LOGICAL_KEYS[table]
        missing_keys = sorted(set(logical_keys) - set(frame.columns))
        if missing_keys:
            raise ValueError(
                f"Oracle {table} records are missing logical keys: "
                + ", ".join(missing_keys)
            )

        table_actions = result.actions.loc[
            result.actions["table"].astype(str).eq(table)
        ]
        if bool(table_actions["source_key"].astype(str).duplicated().any()):
            raise ValueError(f"Oracle {table} actions contain duplicate source keys.")
        action_by_key = {
            str(row.source_key): str(row.action)
            for row in table_actions.itertuples(index=False)
        }
        rows = [
            {
                str(key): _plain_value(value)
                for key, value in record.items()
            }
            for record in frame.to_dict(orient="records")
        ]
        record_keys = [_source_key(record, logical_keys) for record in rows]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError(f"Oracle {table} records contain duplicate logical keys.")
        if set(record_keys) != set(action_by_key):
            raise ValueError(f"Oracle {table} records and actions do not have identical keys.")

        by_action = {
            _ACTION_INSERT: [],
            _ACTION_UPDATE: [],
            _ACTION_SKIP: [],
        }
        for key, record in zip(record_keys, rows, strict=True):
            action = action_by_key[key]
            if action not in by_action:
                raise ValueError(f"Oracle {table} action is unsupported: {action}")
            by_action[action].append(record)
        partitioned[table] = by_action
    return partitioned


def _insert_rows(
    connection: Connection,
    *,
    table_name: str,
    table: Table,
    rows: Sequence[dict[str, object]],
    batch_size: int,
) -> int:
    inserted = 0
    for batch in _chunks(rows, batch_size):
        statement = (
            postgresql_insert(table)
            .values(batch)
            .on_conflict_do_nothing()
        )
        cursor = connection.execute(
            statement,
            execution_options={"preserve_rowcount": True},
        )
        if cursor.rowcount != len(batch):
            raise RuntimeError(
                f"Oracle {table_name} insert conflict or rowcount drift: "
                f"expected {len(batch)}, wrote {cursor.rowcount}."
            )
        inserted += cursor.rowcount
    return inserted


def _update_rows(
    connection: Connection,
    *,
    table_name: str,
    table: Table,
    rows: Sequence[dict[str, object]],
    batch_size: int,
) -> int:
    if not rows:
        return 0

    scope_columns = _UPDATE_SCOPE[table_name]
    update_columns = _UPDATE_COLUMNS[table_name]
    preserved = _PRESERVE_ON_NULL.get(table_name, frozenset())
    conditions = [
        table.c[column]
        == bindparam(
            f"_scope_{column}",
            type_=table.c[column].type,
        )
        for column in scope_columns
    ]
    values: dict[str, object] = {}
    for column in update_columns:
        parameter = bindparam(
            f"_value_{column}",
            type_=table.c[column].type,
        )
        values[column] = (
            func.coalesce(parameter, table.c[column])
            if column in preserved
            else parameter
        )
    statement = update(table).where(and_(*conditions)).values(values)

    updated = 0
    for batch in _chunks(rows, batch_size):
        parameters = [
            {
                **{
                    f"_scope_{column}": record[column]
                    for column in scope_columns
                },
                **{
                    f"_value_{column}": record[column]
                    for column in update_columns
                },
            }
            for record in batch
        ]
        cursor = connection.execute(statement, parameters)
        if cursor.rowcount != len(batch):
            raise RuntimeError(
                f"Oracle {table_name} update rowcount drift: "
                f"expected {len(batch)}, wrote {cursor.rowcount}."
            )
        updated += cursor.rowcount
    return updated


def _resolve_game_team_ids(
    connection: Connection,
    records: Sequence[dict[str, object]],
    *,
    batch_size: int,
) -> dict[tuple[str, str], int]:
    keys = sorted(
        {
            (str(record["game_id"]), str(record["side"]))
            for record in records
        }
    )
    resolved: dict[tuple[str, str], int] = {}
    for batch in _chunks(
        [
            {"game_id": game_id, "side": side}
            for game_id, side in keys
        ],
        batch_size,
    ):
        batch_keys = [
            (str(record["game_id"]), str(record["side"]))
            for record in batch
        ]
        statement = select(
            GameTeam.game_team_id,
            GameTeam.game_id,
            GameTeam.side,
        ).where(tuple_(GameTeam.game_id, GameTeam.side).in_(batch_keys))
        for row in connection.execute(statement):
            key = (str(row.game_id), str(row.side))
            if key in resolved:
                raise RuntimeError(
                    "Oracle game_team lookup returned a duplicate logical key: "
                    + "|".join(key)
                )
            resolved[key] = int(row.game_team_id)

    missing = [key for key in keys if key not in resolved]
    if missing:
        samples = ", ".join("|".join(key) for key in missing[:5])
        raise RuntimeError(
            "Oracle game_player parents could not be resolved: " + samples
        )
    return resolved


def _database_game_player_records(
    connection: Connection,
    records: Sequence[dict[str, object]],
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    if not records:
        return []
    parents = _resolve_game_team_ids(
        connection,
        records,
        batch_size=batch_size,
    )
    return [
        {
            "game_team_id": parents[
                (str(record["game_id"]), str(record["side"]))
            ],
            "player_id": record["player_id"],
            "role": record["role"],
            "champion_id": record["champion_id"],
            "confirmation_status": record["confirmation_status"],
        }
        for record in records
    ]


def apply_oracle_records(
    connection: Connection,
    result: OracleDryRunResult,
    batch_size: int = 1000,
) -> OracleApplyResult:
    """Apply approved records without opening or committing a transaction."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    partitioned = _records_by_action(result)
    per_table: dict[str, TableApplyStats] = {}
    has_write = False

    for table_name in TARGET_TABLE_ORDER:
        table = _TABLES[table_name]
        table_rows = partitioned[table_name]
        insert_rows = table_rows[_ACTION_INSERT]
        update_rows = table_rows[_ACTION_UPDATE]
        skip_rows = table_rows[_ACTION_SKIP]
        has_write = has_write or bool(insert_rows or update_rows)

        if table_name == "game_player":
            insert_rows = _database_game_player_records(
                connection,
                insert_rows,
                batch_size=batch_size,
            )
            update_rows = _database_game_player_records(
                connection,
                update_rows,
                batch_size=batch_size,
            )

        inserted = _insert_rows(
            connection,
            table_name=table_name,
            table=table,
            rows=insert_rows,
            batch_size=batch_size,
        )
        updated = _update_rows(
            connection,
            table_name=table_name,
            table=table,
            rows=update_rows,
            batch_size=batch_size,
        )
        per_table[table_name] = TableApplyStats(
            inserted=inserted,
            updated=updated,
            skipped=len(skip_rows),
        )

    connection.exec_driver_sql(
        "SET CONSTRAINTS fk_game_winner_participant IMMEDIATE"
    )
    return OracleApplyResult(
        state=(APPROVED_INITIAL_STATE if has_write else ALREADY_APPLIED),
        per_table=per_table,
    )
