"""Resolve định danh Oracle và ánh xạ champion theo quy tắc Bước 6C."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from match_insight.data_processing.oracle_etl import (
    PLAYER_POSITIONS,
    OracleCoreData,
)
from match_insight.data_processing.reference_common import canonical_key

SOURCE_ID: Final[str] = "SOURCE_ID"
RECOVERED_UNIQUE: Final[str] = "RECOVERED_UNIQUE"
UNRESOLVED: Final[str] = "UNRESOLVED"

TARGET_EXISTING: Final[str] = "EXISTING"
TARGET_NEW_REQUIRED: Final[str] = "NEW_TARGET_REQUIRED"
TARGET_CONFLICT: Final[str] = "TARGET_CONFLICT"
TARGET_NOT_APPLICABLE: Final[str] = "NOT_APPLICABLE"

CHAMPION_MAPPED: Final[str] = "MAPPED"
CHAMPION_UNMAPPED: Final[str] = "UNMAPPED"

_SOURCE_ID_MAX_LENGTH: Final[int] = 64

TeamContextKey = tuple[str, str, int, str | None, int, str]


@dataclass(frozen=True)
class TeamEntityReference:
    """Snapshot read-only của một team entity hiện có."""

    team_id: str
    oracle_team_id: str | None
    canonical_name: str
    display_name: str
    logo_file: str | None = None


@dataclass(frozen=True)
class PlayerEntityReference:
    """Snapshot read-only của một player entity hiện có."""

    player_id: str
    oracle_player_id: str | None
    canonical_name: str
    display_name: str
    photo_file: str | None = None


@dataclass(frozen=True)
class ChampionEntityReference:
    """Snapshot read-only của một champion Data Dragon."""

    champion_id: str
    canonical_name: str
    display_name: str
    image_file: str | None = None


@dataclass(frozen=True)
class OracleIdentityResult:
    """Kết quả resolve ở đúng grain team participation và player row."""

    team_participations: pd.DataFrame
    player_rows: pd.DataFrame


def _require_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str,
) -> None:
    """Báo lỗi rõ nếu DataFrame thiếu cột cần cho resolver."""
    actual_columns = set(dataframe.columns)
    missing_columns = [
        column
        for column in columns
        if column not in actual_columns
    ]
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing columns: "
            + ", ".join(missing_columns)
        )


def _is_missing(value: object) -> bool:
    """Kiểm tra scalar thiếu mà không gọi bool(pd.NA)."""
    if value is None:
        return True
    return bool(pd.isna(value))


def _source_id_state(
    value: object,
) -> tuple[str | None, bool]:
    """Trả source ID nguyên dạng cùng cờ invalid; không sửa hoặc tạo ID."""
    if _is_missing(value):
        return None, False

    source_id = str(value)
    if not source_id.strip():
        return None, False

    invalid = (
        source_id != source_id.strip()
        or len(source_id) > _SOURCE_ID_MAX_LENGTH
    )
    if invalid:
        return None, True

    return source_id, False


def _required_text(value: object, *, field_name: str) -> str:
    """Kiểm tra target ID/reference text bắt buộc."""
    if _is_missing(value):
        raise ValueError(f"{field_name} cannot be missing.")

    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} cannot be blank.")
    return text


def _canonical_key_or_none(value: object) -> str | None:
    """Tạo khóa tên có kiểm soát, giữ punctuation và dấu Unicode."""
    if _is_missing(value):
        return None

    text = str(value)
    if not text.strip():
        return None
    return canonical_key(text)


def _raw_text_values(series: pd.Series) -> tuple[str, ...]:
    """Giữ các giá trị nguồn khác nhau theo thứ tự xuất hiện."""
    values: list[str] = []
    seen: set[str] = set()

    for value in series.tolist():
        if _is_missing(value):
            continue
        text = str(value)
        if not text.strip() or text in seen:
            continue
        seen.add(text)
        values.append(text)

    return tuple(values)


def _single_team_context_value(
    group: pd.DataFrame,
    column: str,
    *,
    required: bool,
) -> tuple[object | None, str | None]:
    """Collapse one context field without guessing across conflicting rows."""
    values = group[column]
    missing_mask = values.map(_is_missing)
    non_missing = values.loc[~missing_mask].drop_duplicates()

    if len(non_missing) > 1:
        return None, f"CONFLICTING_{column.upper()}_CONTEXT"
    if bool(missing_mask.any()) and len(non_missing) == 1:
        return None, f"MIXED_MISSING_{column.upper()}_CONTEXT"
    if non_missing.empty:
        if required:
            return None, f"MISSING_{column.upper()}_CONTEXT"
        return None, None

    return non_missing.iloc[0], None


def _source_observation_day(value: object) -> str | None:
    """Return the exact UTC source day; never infer a validity interval/cutoff."""
    if _is_missing(value):
        return None

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None

    if timestamp.tzinfo is None:
        return None

    return timestamp.tz_convert("UTC").date().isoformat()


def _team_group_snapshot(
    gameid: object,
    side: object,
    group: pd.DataFrame,
) -> dict[str, object]:
    """Collect direct ID plus exact source context for one participation."""
    valid_source_ids: set[str] = set()
    invalid_source_id_rows = 0

    for value in group["teamid"].tolist():
        source_id, invalid = _source_id_state(value)
        if invalid:
            invalid_source_id_rows += 1
        elif source_id is not None:
            valid_source_ids.add(source_id)

    raw_teamnames = _raw_text_values(group["teamname"])
    name_keys = {
        key
        for value in raw_teamnames
        if (key := _canonical_key_or_none(value)) is not None
    }
    context_values: dict[str, object | None] = {}
    context_issues: list[str] = []

    for column, required in (
        ("league", True),
        ("year", True),
        ("split", False),
        ("playoffs", True),
        ("date", True),
    ):
        value, issue = _single_team_context_value(
            group,
            column,
            required=required,
        )
        context_values[column] = value
        if issue is not None:
            context_issues.append(issue)

    source_year: int | None = None
    if context_values["year"] is not None:
        try:
            source_year = int(context_values["year"])
        except (TypeError, ValueError):
            context_issues.append("INVALID_YEAR_CONTEXT")

    source_playoffs: int | None = None
    if context_values["playoffs"] is not None:
        try:
            source_playoffs = int(context_values["playoffs"])
        except (TypeError, ValueError):
            context_issues.append("INVALID_PLAYOFFS_CONTEXT")
        else:
            if source_playoffs not in {0, 1}:
                context_issues.append("INVALID_PLAYOFFS_CONTEXT")

    observation_day = _source_observation_day(context_values["date"])
    if context_values["date"] is not None and observation_day is None:
        context_issues.append("INVALID_DATE_CONTEXT")

    return {
        "gameid": gameid,
        "side": side,
        "source_teamname": (
            raw_teamnames[0]
            if len(raw_teamnames) == 1
            else pd.NA
        ),
        "source_teamname_candidates": raw_teamnames,
        "canonical_teamname_candidates": tuple(
            sorted(name_keys)
        ),
        "source_teamid_candidates": tuple(
            sorted(valid_source_ids)
        ),
        "invalid_source_teamid_rows": (
            invalid_source_id_rows
        ),
        "source_league": context_values["league"],
        "source_year": source_year,
        "source_split": context_values["split"],
        "source_playoffs": source_playoffs,
        "source_observation_day": observation_day,
        "team_context_issues": tuple(sorted(set(context_issues))),
    }


def _direct_team_resolution(
    snapshot: dict[str, object],
) -> tuple[str | None, str, str] | None:
    """Resolve bằng source ID trực tiếp hoặc trả None để thử name recovery."""
    invalid_rows = int(
        snapshot["invalid_source_teamid_rows"]
    )
    source_ids = tuple(
        snapshot["source_teamid_candidates"]
    )

    if invalid_rows:
        return None, UNRESOLVED, "INVALID_SOURCE_ID"
    if len(source_ids) == 1:
        return (
            str(source_ids[0]),
            SOURCE_ID,
            "SOURCE_ID_PRESENT",
        )
    if len(source_ids) > 1:
        return None, UNRESOLVED, "SOURCE_ID_CONFLICT"
    return None


def _team_context_key(
    snapshot: dict[str, object],
) -> tuple[TeamContextKey | None, str | None]:
    """Build a fail-closed name/tournament/day key for team recovery."""
    name_keys = tuple(snapshot["canonical_teamname_candidates"])
    if not name_keys:
        return None, "MISSING_TEAM_NAME"
    if len(name_keys) > 1:
        return None, "TEAM_NAME_CONFLICT"

    context_issues = tuple(snapshot["team_context_issues"])
    if context_issues:
        return None, ",".join(context_issues)

    league_key = _canonical_key_or_none(snapshot["source_league"])
    split_value = snapshot["source_split"]
    split_key = _canonical_key_or_none(split_value)
    source_year = snapshot["source_year"]
    source_playoffs = snapshot["source_playoffs"]
    observation_day = snapshot["source_observation_day"]

    if league_key is None:
        return None, "MISSING_LEAGUE_CONTEXT"
    if split_value is not None and split_key is None:
        return None, "INVALID_SPLIT_CONTEXT"
    if (
        source_year is None
        or source_playoffs is None
        or observation_day is None
    ):
        return None, "INSUFFICIENT_TEAM_CONTEXT"

    return (
        (
            str(name_keys[0]),
            league_key,
            int(source_year),
            split_key,
            int(source_playoffs),
            str(observation_day),
        ),
        None,
    )


def _team_context_evidence(
    snapshots: list[dict[str, object]],
) -> dict[TeamContextKey, set[str]]:
    """Index direct source IDs only by full, exact source context."""
    evidence: dict[TeamContextKey, set[str]] = defaultdict(set)

    for snapshot in snapshots:
        direct = _direct_team_resolution(snapshot)
        if direct is None or direct[1] != SOURCE_ID:
            continue

        context, _ = _team_context_key(snapshot)
        if context is not None:
            evidence[context].add(str(direct[0]))

    return evidence


def _recover_team_from_context(
    snapshot: dict[str, object],
    evidence: dict[TeamContextKey, set[str]],
) -> tuple[str | None, str, str, tuple[str, ...]]:
    """Recover only when the full exact context yields one source ID."""
    context, reason = _team_context_key(snapshot)
    if context is None:
        if reason is None:
            raise AssertionError("Missing team context must include a reason.")
        return None, UNRESOLVED, reason, ()

    candidates = tuple(
        sorted(evidence.get(context, set()))
    )
    if len(candidates) == 1:
        return (
            candidates[0],
            RECOVERED_UNIQUE,
            "UNIQUE_FULL_CONTEXT",
            candidates,
        )
    if not candidates:
        return (
            None,
            UNRESOLVED,
            "NO_SOURCE_ID_CANDIDATE",
            candidates,
        )
    return (
        None,
        UNRESOLVED,
        "AMBIGUOUS_SOURCE_ID_CANDIDATES",
        candidates,
    )


def _team_target_index(
    references: Sequence[TeamEntityReference],
) -> dict[str, set[str]]:
    """Lập exact Oracle team ID→target IDs; không dùng tên."""
    index: dict[str, set[str]] = defaultdict(set)

    for reference in references:
        target_id = _required_text(
            reference.team_id,
            field_name="team_id",
        )
        source_id, invalid = _source_id_state(
            reference.oracle_team_id
        )
        if invalid:
            raise ValueError(
                "Target team contains an invalid oracle_team_id."
            )
        if source_id is not None:
            index[source_id].add(target_id)

    return index


def _player_target_index(
    references: Sequence[PlayerEntityReference],
) -> dict[str, set[str]]:
    """Lập exact Oracle player ID→target IDs; không dùng tên."""
    index: dict[str, set[str]] = defaultdict(set)

    for reference in references:
        target_id = _required_text(
            reference.player_id,
            field_name="player_id",
        )
        source_id, invalid = _source_id_state(
            reference.oracle_player_id
        )
        if invalid:
            raise ValueError(
                "Target player contains an invalid oracle_player_id."
            )
        if source_id is not None:
            index[source_id].add(target_id)

    return index


def _apply_target_mapping(
    dataframe: pd.DataFrame,
    *,
    source_id_column: str,
    target_id_column: str,
    target_status_column: str,
    target_index: dict[str, set[str]],
) -> pd.DataFrame:
    """Map target bằng exact source ID và giữ conflict tường minh."""
    mapped = dataframe.copy()
    target_ids: list[str | None] = []
    statuses: list[str] = []

    for source_value in mapped[source_id_column].tolist():
        source_id, invalid = _source_id_state(source_value)
        if invalid or source_id is None:
            target_ids.append(None)
            statuses.append(TARGET_NOT_APPLICABLE)
            continue

        candidates = tuple(
            sorted(target_index.get(source_id, set()))
        )
        if len(candidates) == 1:
            target_ids.append(candidates[0])
            statuses.append(TARGET_EXISTING)
        elif not candidates:
            target_ids.append(None)
            statuses.append(TARGET_NEW_REQUIRED)
        else:
            target_ids.append(None)
            statuses.append(TARGET_CONFLICT)

    mapped[target_id_column] = pd.Series(
        target_ids,
        index=mapped.index,
        dtype="string",
    )
    mapped[target_status_column] = pd.Series(
        statuses,
        index=mapped.index,
        dtype="string",
    )
    return mapped


def resolve_team_identities(
    source_rows: pd.DataFrame,
    *,
    references: Sequence[TeamEntityReference] = (),
) -> pd.DataFrame:
    """Resolve team ở grain ``(gameid, side)`` mà không sửa input."""
    if source_rows.empty:
        raise ValueError("Oracle source rows cannot be empty.")

    _require_columns(
        source_rows,
        (
            "gameid",
            "side",
            "teamid",
            "teamname",
            "league",
            "year",
            "split",
            "playoffs",
            "date",
        ),
        frame_name="Oracle source rows",
    )

    if bool(
        source_rows[["gameid", "side"]]
        .isna()
        .any(axis=1)
        .any()
    ):
        raise ValueError(
            "Oracle team grain contains missing gameid or side."
        )

    snapshots = [
        _team_group_snapshot(gameid, side, group)
        for (gameid, side), group in source_rows.groupby(
            ["gameid", "side"],
            sort=False,
            dropna=False,
        )
    ]
    context_evidence = _team_context_evidence(snapshots)
    records: list[dict[str, object]] = []

    for snapshot in snapshots:
        direct = _direct_team_resolution(snapshot)
        candidate_ids = tuple(
            snapshot["source_teamid_candidates"]
        )
        context, _ = _team_context_key(snapshot)
        context_candidates = (
            ()
            if context is None
            else tuple(sorted(context_evidence.get(context, set())))
        )

        if direct is not None:
            if direct[1] == SOURCE_ID and len(context_candidates) > 1:
                resolved_id = None
                method = UNRESOLVED
                reason = "SOURCE_ID_CONFLICT"
                candidate_ids = context_candidates
            else:
                resolved_id, method, reason = direct
        else:
            (
                resolved_id,
                method,
                reason,
                candidate_ids,
            ) = _recover_team_from_context(
                snapshot,
                context_evidence,
            )

        records.append(
            {
                "gameid": snapshot["gameid"],
                "side": snapshot["side"],
                "source_teamname": snapshot["source_teamname"],
                "source_teamname_candidates": snapshot[
                    "source_teamname_candidates"
                ],
                "canonical_teamname_candidates": snapshot[
                    "canonical_teamname_candidates"
                ],
                "source_teamid_candidates": snapshot[
                    "source_teamid_candidates"
                ],
                "invalid_source_teamid_rows": snapshot[
                    "invalid_source_teamid_rows"
                ],
                "resolved_oracle_team_id": resolved_id,
                "team_resolution_method": method,
                "team_resolution_reason": reason,
                "team_resolution_candidates": candidate_ids,
            }
        )

    resolved = pd.DataFrame.from_records(records)
    for column in (
        "gameid",
        "side",
        "source_teamname",
        "resolved_oracle_team_id",
        "team_resolution_method",
        "team_resolution_reason",
    ):
        resolved[column] = resolved[column].astype(
            "string"
        )

    return _apply_target_mapping(
        resolved,
        source_id_column="resolved_oracle_team_id",
        target_id_column="target_team_id",
        target_status_column="team_target_status",
        target_index=_team_target_index(references),
    )


def _context_key(
    row: object,
) -> tuple[str, str, str, str, int, str | None] | None:
    """Tạo khóa player recovery từ name/team/role/source context."""
    player_name_key = _canonical_key_or_none(
        getattr(row, "playername")
    )
    team_id, invalid_team_id = _source_id_state(
        getattr(row, "resolved_oracle_team_id")
    )
    position_value = getattr(row, "position")
    league_key = _canonical_key_or_none(
        getattr(row, "league")
    )
    year_value = getattr(row, "year")
    split_key = _canonical_key_or_none(
        getattr(row, "split")
    )

    if (
        player_name_key is None
        or invalid_team_id
        or team_id is None
        or _is_missing(position_value)
        or str(position_value) not in PLAYER_POSITIONS
        or league_key is None
        or _is_missing(year_value)
    ):
        return None

    return (
        player_name_key,
        team_id,
        str(position_value),
        league_key,
        int(year_value),
        split_key,
    )


def _player_evidence(
    player_rows: pd.DataFrame,
) -> dict[
    tuple[str, str, str, str, int, str | None],
    set[str],
]:
    """Lập context→player IDs chỉ từ source IDs trực tiếp."""
    evidence: dict[
        tuple[str, str, str, str, int, str | None],
        set[str],
    ] = defaultdict(set)

    for row in player_rows.itertuples(index=False):
        source_id, invalid = _source_id_state(
            getattr(row, "playerid")
        )
        context = _context_key(row)
        if (
            source_id is not None
            and not invalid
            and context is not None
        ):
            evidence[context].add(source_id)

    return evidence


def resolve_player_identities(
    player_rows: pd.DataFrame,
    team_participations: pd.DataFrame,
    *,
    references: Sequence[PlayerEntityReference] = (),
) -> pd.DataFrame:
    """Resolve player ID bằng source ID hoặc full context duy nhất."""
    _require_columns(
        player_rows,
        (
            "gameid",
            "side",
            "playerid",
            "playername",
            "position",
            "league",
            "year",
            "split",
        ),
        frame_name="Oracle player rows",
    )
    _require_columns(
        team_participations,
        (
            "gameid",
            "side",
            "resolved_oracle_team_id",
            "team_resolution_method",
            "team_resolution_reason",
        ),
        frame_name="Oracle team resolutions",
    )

    if not player_rows.index.is_unique:
        raise ValueError(
            "Oracle player row index must be unique for traceability."
        )
    if bool(
        team_participations.duplicated(
            subset=["gameid", "side"]
        ).any()
    ):
        raise ValueError(
            "Oracle team resolutions contain duplicate game-side keys."
        )

    team_lookup = team_participations.set_index(
        ["gameid", "side"]
    )[
        [
            "resolved_oracle_team_id",
            "team_resolution_method",
            "team_resolution_reason",
        ]
    ]
    resolved = player_rows.copy().join(
        team_lookup,
        on=["gameid", "side"],
        how="left",
        validate="many_to_one",
    )
    evidence = _player_evidence(resolved)
    resolved_ids: list[str | None] = []
    methods: list[str] = []
    reasons: list[str] = []
    candidates_column: list[tuple[str, ...]] = []

    for row in resolved.itertuples(index=False):
        source_id, invalid = _source_id_state(
            getattr(row, "playerid")
        )
        context = _context_key(row)

        if invalid:
            resolved_ids.append(None)
            methods.append(UNRESOLVED)
            reasons.append("INVALID_SOURCE_ID")
            candidates_column.append(())
            continue
        if source_id is not None:
            resolved_ids.append(source_id)
            methods.append(SOURCE_ID)
            reasons.append("SOURCE_ID_PRESENT")
            candidates_column.append((source_id,))
            continue
        if _is_missing(
            getattr(row, "resolved_oracle_team_id")
        ):
            resolved_ids.append(None)
            methods.append(UNRESOLVED)
            reasons.append("TEAM_UNRESOLVED")
            candidates_column.append(())
            continue
        if context is None:
            resolved_ids.append(None)
            methods.append(UNRESOLVED)
            reasons.append("INSUFFICIENT_CONTEXT")
            candidates_column.append(())
            continue

        candidates = tuple(
            sorted(evidence.get(context, set()))
        )
        if len(candidates) == 1:
            resolved_ids.append(candidates[0])
            methods.append(RECOVERED_UNIQUE)
            reasons.append("UNIQUE_FULL_CONTEXT")
        elif not candidates:
            resolved_ids.append(None)
            methods.append(UNRESOLVED)
            reasons.append("NO_SOURCE_ID_CANDIDATE")
        else:
            resolved_ids.append(None)
            methods.append(UNRESOLVED)
            reasons.append("AMBIGUOUS_SOURCE_ID_CANDIDATES")
        candidates_column.append(candidates)

    resolved["resolved_oracle_player_id"] = pd.Series(
        resolved_ids,
        index=resolved.index,
        dtype="string",
    )
    resolved["player_resolution_method"] = pd.Series(
        methods,
        index=resolved.index,
        dtype="string",
    )
    resolved["player_resolution_reason"] = pd.Series(
        reasons,
        index=resolved.index,
        dtype="string",
    )
    resolved["player_resolution_candidates"] = (
        candidates_column
    )

    return _apply_target_mapping(
        resolved,
        source_id_column="resolved_oracle_player_id",
        target_id_column="target_player_id",
        target_status_column="player_target_status",
        target_index=_player_target_index(references),
    )


def _champion_index(
    references: Sequence[ChampionEntityReference],
) -> dict[str, set[str]]:
    """Lập normalized canonical/display name→champion IDs."""
    index: dict[str, set[str]] = defaultdict(set)
    seen_champion_ids: set[str] = set()

    for reference in references:
        champion_id = _required_text(
            reference.champion_id,
            field_name="champion_id",
        )
        if champion_id in seen_champion_ids:
            raise ValueError(
                f"Duplicate champion reference ID: {champion_id}"
            )
        seen_champion_ids.add(champion_id)

        name_keys = {
            key
            for value in (
                reference.canonical_name,
                reference.display_name,
            )
            if (key := _canonical_key_or_none(value))
            is not None
        }
        if not name_keys:
            raise ValueError(
                f"Champion reference {champion_id} has no valid name."
            )
        for name_key in name_keys:
            index[name_key].add(champion_id)

    return index


def map_oracle_champions(
    player_rows: pd.DataFrame,
    references: Sequence[ChampionEntityReference],
) -> pd.DataFrame:
    """Map Oracle champion name khi canonical/display candidate là duy nhất."""
    _require_columns(
        player_rows,
        ("champion",),
        frame_name="Oracle player rows",
    )
    champion_index = _champion_index(references)
    mapped = player_rows.copy()
    champion_keys: list[str | None] = []
    champion_ids: list[str | None] = []
    statuses: list[str] = []
    reasons: list[str] = []
    candidate_column: list[tuple[str, ...]] = []

    for source_name in mapped["champion"].tolist():
        name_key = _canonical_key_or_none(source_name)
        champion_keys.append(name_key)

        if name_key is None:
            champion_ids.append(None)
            statuses.append(CHAMPION_UNMAPPED)
            reasons.append("MISSING_SOURCE_NAME")
            candidate_column.append(())
            continue

        candidates = tuple(
            sorted(champion_index.get(name_key, set()))
        )
        if len(candidates) == 1:
            champion_ids.append(candidates[0])
            statuses.append(CHAMPION_MAPPED)
            reasons.append("UNIQUE_CANONICAL_OR_DISPLAY_NAME")
        elif not candidates:
            champion_ids.append(None)
            statuses.append(CHAMPION_UNMAPPED)
            reasons.append("NO_CHAMPION_MATCH")
        else:
            champion_ids.append(None)
            statuses.append(CHAMPION_UNMAPPED)
            reasons.append("AMBIGUOUS_CHAMPION_MATCH")
        candidate_column.append(candidates)

    mapped["oracle_champion_key"] = pd.Series(
        champion_keys,
        index=mapped.index,
        dtype="string",
    )
    mapped["champion_id"] = pd.Series(
        champion_ids,
        index=mapped.index,
        dtype="string",
    )
    mapped["champion_mapping_status"] = pd.Series(
        statuses,
        index=mapped.index,
        dtype="string",
    )
    mapped["champion_mapping_reason"] = pd.Series(
        reasons,
        index=mapped.index,
        dtype="string",
    )
    mapped["champion_mapping_candidates"] = (
        candidate_column
    )
    return mapped


def resolve_oracle_identities(
    core_data: OracleCoreData,
    *,
    team_references: Sequence[TeamEntityReference],
    player_references: Sequence[PlayerEntityReference],
    champion_references: Sequence[ChampionEntityReference],
) -> OracleIdentityResult:
    """Điều phối resolve thuần, không truy cập hoặc ghi database."""
    team_participations = resolve_team_identities(
        core_data.dataframe,
        references=team_references,
    )
    player_rows = resolve_player_identities(
        core_data.player_rows,
        team_participations,
        references=player_references,
    )
    player_rows = map_oracle_champions(
        player_rows,
        champion_references,
    )

    return OracleIdentityResult(
        team_participations=team_participations,
        player_rows=player_rows,
    )


def _value_counts(series: pd.Series) -> dict[str, int]:
    """Trả status/reason counts bằng kiểu Python và thứ tự ổn định."""
    counts = series.astype("string").value_counts().sort_index()
    return {
        str(value): int(count)
        for value, count in counts.items()
    }


def _distinct_entity_counts(
    dataframe: pd.DataFrame,
    *,
    source_id_column: str,
    target_status_column: str,
) -> tuple[int, dict[str, int]]:
    """Đếm target mapping theo distinct resolved Oracle source ID."""
    statuses_by_source: dict[str, set[str]] = defaultdict(set)

    for source_value, status_value in dataframe[
        [source_id_column, target_status_column]
    ].itertuples(index=False, name=None):
        source_id, invalid = _source_id_state(source_value)
        if invalid or source_id is None:
            continue
        statuses_by_source[source_id].add(str(status_value))

    inconsistent = {
        source_id: tuple(sorted(statuses))
        for source_id, statuses in statuses_by_source.items()
        if len(statuses) != 1
    }
    if inconsistent:
        raise ValueError(
            "Target mapping status is inconsistent for Oracle IDs: "
            f"{inconsistent}"
        )

    counts = Counter(
        next(iter(statuses))
        for statuses in statuses_by_source.values()
    )
    return len(statuses_by_source), {
        status: int(counts[status])
        for status in sorted(counts)
    }


def _unmapped_champion_summary(
    player_rows: pd.DataFrame,
) -> list[dict[str, object]]:
    """Tổng hợp unmapped champion theo tên nguồn và reason."""
    counts: Counter[tuple[str | None, str]] = Counter()
    unmapped_rows = player_rows.loc[
        player_rows["champion_mapping_status"].eq(
            CHAMPION_UNMAPPED
        )
    ]

    for source_name, reason in unmapped_rows[
        ["champion", "champion_mapping_reason"]
    ].itertuples(index=False, name=None):
        normalized_source_name = (
            None
            if _is_missing(source_name)
            else str(source_name)
        )
        counts[(normalized_source_name, str(reason))] += 1

    return [
        {
            "source_name": source_name,
            "reason": reason,
            "row_count": int(count),
        }
        for (source_name, reason), count in sorted(
            counts.items(),
            key=lambda item: (
                item[0][0] or "",
                item[0][1],
            ),
        )
    ]


def build_identity_summary(
    result: OracleIdentityResult,
) -> dict[str, object]:
    """Tạo reconciliation summary mà không hard-code benchmark."""
    team_rows = result.team_participations
    player_rows = result.player_rows
    team_entity_total, team_entity_counts = (
        _distinct_entity_counts(
            team_rows,
            source_id_column="resolved_oracle_team_id",
            target_status_column="team_target_status",
        )
    )
    player_entity_total, player_entity_counts = (
        _distinct_entity_counts(
            player_rows,
            source_id_column="resolved_oracle_player_id",
            target_status_column="player_target_status",
        )
    )

    return {
        "team_groups": len(team_rows),
        "team_resolution_counts": _value_counts(
            team_rows["team_resolution_method"]
        ),
        "team_reason_counts": _value_counts(
            team_rows["team_resolution_reason"]
        ),
        "distinct_team_source_ids_eligible": (
            team_entity_total
        ),
        "team_target_counts": team_entity_counts,
        "player_rows": len(player_rows),
        "player_resolution_counts": _value_counts(
            player_rows["player_resolution_method"]
        ),
        "player_reason_counts": _value_counts(
            player_rows["player_resolution_reason"]
        ),
        "distinct_player_source_ids_eligible": (
            player_entity_total
        ),
        "player_target_counts": player_entity_counts,
        "champion_mapping_counts": _value_counts(
            player_rows["champion_mapping_status"]
        ),
        "champion_reason_counts": _value_counts(
            player_rows["champion_mapping_reason"]
        ),
        "unmapped_champions": (
            _unmapped_champion_summary(player_rows)
        ),
    }
