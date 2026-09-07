"""Paired PRE/POST dataset và temporal split thuần, không I/O hoặc model.

OK chỉ xác nhận contract xử lý in-memory.
Nguồn cutoff, end và roster PRE phải được xác minh bên ngoài.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

import pandas as pd

from match_insight.data_processing.reference_common import validate_stable_id
from match_insight.features.post import (
    ChampionSlot,
    FinalLineup,
    HistoricalChampionGame,
    build_post_features,
)
from match_insight.features.pre import (
    ROLES,
    FeatureConfig,
    HistoricalGame,
    PlayerSlot,
    PreFeatureInputError,
    TargetGame,
    build_pre_features,
)

DATASET_VERSION = "paired-pre-post-v1"
STRICT_PROTOCOL = "STRICT_PRE"
RETROSPECTIVE_PROTOCOL = "retrospective-utc-day-minus1-v1"
RETROSPECTIVE_DATASET_VERSION = "paired-pre-post-retrospective-v1"

_PRE_GROUPS = (
    "blue_recent_form",
    "blue_side_win_rate",
    "blue_roster_continuity",
    "red_recent_form",
    "red_side_win_rate",
    "red_roster_continuity",
    "h2h_blue",
)
_PRE_FIELDS = ("value", "sample_count", "missing", "win_count")
_POST_FIELDS = (
    "champion_id",
    "games_count",
    "wins_count",
    "win_rate",
    "missing",
)

PRE_COLUMNS = tuple(
    f"{group}_{field}"
    for group in _PRE_GROUPS
    for field in _PRE_FIELDS
    if not (group.endswith("roster_continuity") and field == "win_count")
)

POST_EXTRA_COLUMNS = tuple(
    f"{side}_{role.lower()}_{field}"
    for side in ("blue", "red")
    for role in ROLES
    for field in _POST_FIELDS
)
POST_COLUMNS = PRE_COLUMNS + POST_EXTRA_COLUMNS

CATEGORICAL_COLUMNS = tuple(
    f"{side}_{role.lower()}_champion_id"
    for side in ("blue", "red")
    for role in ROLES
)

METADATA_COLUMNS = (
    "history_cutoff_at",
    "target_ended_at",
    "patch",
    "blue_team_id",
    "red_team_id",
    "blue_roster",
    "red_roster",
    "evidence_ref",
    "policy_version",
    "cutoff_verification",
    "core_cutoff_verification",
    "dataset_version",
    "pre_config_version",
    "post_config_version",
    "pre_config",
    "history_game_ids",
    "pre_feature_game_ids",
    "post_feature_game_ids",
    "history_exclusions",
    "history_counts",
)


class DatasetInputError(PreFeatureInputError):
    """Lỗi contract dataset; dùng cùng cấu trúc code/game_id của feature core."""


class CutoffVerification(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED_EXTERNALLY = "VERIFIED_EXTERNALLY"
    PROTOCOL_ASSUMED = "PROTOCOL_ASSUMED"


def protocol_contract(evaluation_protocol):
    """Return contract labels without certifying timestamp/context sources."""
    if evaluation_protocol == STRICT_PROTOCOL:
        return DATASET_VERSION, CutoffVerification.VERIFIED_EXTERNALLY
    if evaluation_protocol == RETROSPECTIVE_PROTOCOL:
        return RETROSPECTIVE_DATASET_VERSION, CutoffVerification.PROTOCOL_ASSUMED
    raise DatasetInputError(
        "E_EVALUATION_PROTOCOL_INVALID",
        "Expected STRICT_PRE or the approved retrospective protocol",
    )


@dataclass(frozen=True, slots=True)
class CutoffRecord:
    game_id: str
    history_cutoff_at: datetime | None
    evidence_ref: str
    policy_version: str
    verification: CutoffVerification


@dataclass(frozen=True, slots=True)
class DatasetTarget:
    """Roster do caller cung cấp trong ngữ cảnh PRE, không suy từ final lineup."""

    game_id: str
    blue_team_id: str
    red_team_id: str
    blue_roster: tuple[PlayerSlot, ...]
    red_roster: tuple[PlayerSlot, ...]
    patch: str
    final_lineup: FinalLineup
    winner_team_id: str
    ended_at: datetime | None


@dataclass(frozen=True, slots=True)
class DatasetExclusion:
    game_id: str
    reason: str
    detail: str
    source_game_id: str | None = None


@dataclass(frozen=True, slots=True)
class PairedDataset:
    status: Literal["OK", "EMPTY"]
    reason: str | None
    X_pre: pd.DataFrame
    X_post: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    excluded: tuple[DatasetExclusion, ...]
    unused_cutoff_game_ids: tuple[str, ...]
    categorical_columns: tuple[str, ...] = CATEGORICAL_COLUMNS


@dataclass(frozen=True, slots=True)
class SplitExclusion:
    game_id: str
    partition: str
    reason: str
    ended_at: datetime
    required_before: datetime


@dataclass(frozen=True, slots=True)
class PartitionSummary:
    partition: str
    planned_count: int
    count: int
    first_cutoff: datetime | None
    last_cutoff: datetime | None
    max_ended_at: datetime | None
    fraction_of_input: float
    fraction_of_retained: float


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    status: Literal["OK", "INSUFFICIENT"]
    reason: str | None
    train: tuple[str, ...] = ()
    validation: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    validation_boundary: datetime | None = None
    test_boundary: datetime | None = None
    excluded: tuple[SplitExclusion, ...] = ()
    summary: tuple[PartitionSummary, ...] = ()
    target_ratios: tuple[float, ...] = (0.70, 0.15, 0.15)


@dataclass(frozen=True, slots=True)
class RetrospectiveTemporalSplit(TemporalSplit):
    evaluation_protocol: str = RETROSPECTIVE_PROTOCOL


@dataclass(frozen=True, slots=True)
class _Pair:
    game_id: str
    pre_values: dict[str, object]
    post_values: dict[str, object]
    label: int
    metadata: dict[str, object]


def _records(values, expected_type, name):
    try:
        records = tuple(values)
    except TypeError:
        raise DatasetInputError(
            "E_DATASET_INPUT_INVALID",
            f"{name} must be iterable",
        ) from None

    if any(not isinstance(record, expected_type) for record in records):
        raise DatasetInputError(
            "E_DATASET_INPUT_INVALID",
            f"{name} contains an invalid record type",
        )
    return records


def _require_game_id(value: object) -> None:
    if not isinstance(value, str):
        raise DatasetInputError("E_ID_INVALID", "game_id must be a stable string ID")

    try:
        canonical = validate_stable_id(value, "game_id", 100)
    except ValueError:
        raise DatasetInputError(
            "E_ID_INVALID",
            "game_id violates the existing stable ID contract",
        ) from None

    if canonical != value:
        raise DatasetInputError(
            "E_ID_INVALID",
            "game_id must already be canonical",
        )


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _utc(value: object, code: str, game_id: str) -> datetime:
    if not isinstance(value, datetime):
        raise DatasetInputError(
            code,
            "An aware datetime is required; strings and epochs are not parsed",
            game_id,
        )

    try:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime")
        return value.astimezone(UTC)
    except (OverflowError, TypeError, ValueError):
        raise DatasetInputError(
            code,
            "Timestamp requires an explicit, valid timezone",
            game_id,
        ) from None


def _resolve_cutoff(
    game_id: str,
    records: tuple[CutoffRecord, ...],
    approved_policy_versions: frozenset[str],
    *,
    evaluation_protocol: str = STRICT_PROTOCOL,
) -> tuple[CutoffRecord, datetime]:
    _version, required_verification = protocol_contract(evaluation_protocol)
    if not records:
        raise DatasetInputError(
            "E_CUTOFF_MISSING",
            "No cutoff record was supplied",
            game_id,
        )
    if len(records) != 1:
        raise DatasetInputError(
            "E_CUTOFF_CONFLICT",
            "Exactly one cutoff record is required; duplicates are not merged",
            game_id,
        )

    record = records[0]
    if record.history_cutoff_at is None:
        raise DatasetInputError(
            "E_CUTOFF_MISSING",
            "Cutoff timestamp is missing",
            game_id,
        )

    cutoff = _utc(record.history_cutoff_at, "E_CUTOFF_INVALID", game_id)

    if not _nonempty_text(record.evidence_ref):
        raise DatasetInputError(
            "E_CUTOFF_EVIDENCE_INVALID",
            "evidence_ref must be nonempty and canonical",
            game_id,
        )
    if not _nonempty_text(record.policy_version):
        raise DatasetInputError(
            "E_CUTOFF_POLICY_INVALID",
            "policy_version must be nonempty and canonical",
            game_id,
        )
    if record.policy_version not in approved_policy_versions:
        raise DatasetInputError(
            "E_CUTOFF_POLICY_UNAPPROVED",
            "Cutoff policy is absent from the caller-approved policy set",
            game_id,
        )
    if (
        evaluation_protocol == RETROSPECTIVE_PROTOCOL
    ) != (record.policy_version == RETROSPECTIVE_PROTOCOL):
        raise DatasetInputError(
            "E_CUTOFF_PROTOCOL_MISMATCH",
            "Cutoff policy does not belong to the requested protocol",
            game_id,
        )
    if not isinstance(record.verification, CutoffVerification):
        raise DatasetInputError(
            "E_CUTOFF_STATUS_INVALID",
            "verification must be CutoffVerification",
            game_id,
        )
    if record.verification is not required_verification:
        raise DatasetInputError(
            "E_CUTOFF_UNVERIFIED",
            "Cutoff verification does not satisfy the requested protocol",
            game_id,
        )

    return record, cutoff


def _pre_stats(pre):
    return (
        ("blue_recent_form", pre.blue.recent_form),
        ("blue_side_win_rate", pre.blue.side_win_rate),
        ("blue_roster_continuity", pre.blue.roster_continuity),
        ("red_recent_form", pre.red.recent_form),
        ("red_side_win_rate", pre.red.side_win_rate),
        ("red_roster_continuity", pre.red.roster_continuity),
        ("h2h_blue", pre.h2h_blue),
    )


def _roster_identity(roster):
    """Compare role/player identities without depending on tuple order."""
    if not isinstance(roster, tuple) or any(
        not isinstance(slot, PlayerSlot)
        or not isinstance(slot.role, str)
        or not isinstance(slot.player_id, str)
        for slot in roster
    ):
        return None
    return tuple(sorted((slot.role, slot.player_id) for slot in roster))


def _champion_identity(slots):
    """Canonical comparison only; POST retains its existing lineup validator."""
    if not isinstance(slots, tuple) or any(
        not isinstance(slot, ChampionSlot)
        or any(
            not isinstance(value, str)
            for value in (
                slot.team_id, slot.side, slot.role, slot.player_id, slot.champion_id
            )
        )
        for slot in slots
    ):
        return None
    return tuple(sorted(
        (slot.team_id, slot.side, slot.role, slot.player_id, slot.champion_id)
        for slot in slots
    ))


def _require_history_target_consistency(pre, history, targets_by_id):
    """Reject a consuming pair when accepted history contradicts another target.

    Use the PRE snapshot's selection; do not introduce another temporal filter.
    Target/self and temporally excluded records are never reconciled here.
    """
    snapshot = pre.history_snapshot
    if snapshot is None:
        raise DatasetInputError(
            "E_PRE_SNAPSHOT_REQUIRED",
            "Target-history reconciliation requires the PRE history snapshot",
            pre.metadata.game_id,
        )

    # PRE has already rejected duplicate historical IDs, including identical rows.
    history_by_id = {record.game.game_id: record for record in history}
    for game in snapshot.games:
        candidates = targets_by_id.get(game.game_id, ())
        if not candidates:
            continue
        if len(candidates) != 1:
            raise DatasetInputError(
                "E_SOURCE_CONFLICT",
                "Accepted history has an ambiguous target identity",
                game.game_id,
            )

        source = candidates[0]
        conflicts = []
        if _utc(source.ended_at, "E_SOURCE_CONFLICT", game.game_id) != _utc(
            game.ended_at, "E_SOURCE_CONFLICT", game.game_id
        ):
            conflicts.append("ended_at")
        for field in ("blue_team_id", "red_team_id", "winner_team_id"):
            if getattr(source, field) != getattr(game, field):
                conflicts.append(field)
        for field in ("blue_roster", "red_roster"):
            if _roster_identity(getattr(source, field)) != _roster_identity(
                getattr(game, field)
            ):
                conflicts.append(field)

        final = source.final_lineup
        if (
            not isinstance(final, FinalLineup)
            or final.game_id != source.game_id
            or final.patch != source.patch
            or _champion_identity(final.slots)
            != _champion_identity(history_by_id[game.game_id].slots)
        ):
            conflicts.append("final_lineup")

        if conflicts:
            raise DatasetInputError(
                "E_SOURCE_CONFLICT",
                "Accepted history differs from target: " + ", ".join(conflicts),
                game.game_id,
            )


def _build_pair(
    target: DatasetTarget,
    cutoff_record: CutoffRecord,
    cutoff: datetime,
    history: tuple[HistoricalChampionGame, ...],
    champion_reference: frozenset[str],
    config: FeatureConfig,
    targets_by_id: dict[str, list[DatasetTarget]],
    dataset_version: str,
) -> _Pair:
    if target.winner_team_id not in (
        target.blue_team_id,
        target.red_team_id,
    ):
        raise DatasetInputError(
            "E_TARGET_WINNER_INVALID",
            "winner_team_id must identify BLUE or RED",
            target.game_id,
        )
    if target.ended_at is None:
        raise DatasetInputError(
            "E_LABEL_END_MISSING",
            "Target ended_at is required to check label availability",
            target.game_id,
        )

    end = _utc(target.ended_at, "E_LABEL_END_INVALID", target.game_id)
    if end <= cutoff:
        raise DatasetInputError(
            "E_LABEL_END_NOT_AFTER_CUTOFF",
            "Target ended_at must be strictly after its PRE cutoff",
            target.game_id,
        )

    context = TargetGame(
        game_id=target.game_id,
        blue_team_id=target.blue_team_id,
        red_team_id=target.red_team_id,
        blue_roster=target.blue_roster,
        red_roster=target.red_roster,
        patch=target.patch,
        history_cutoff_at=cutoff,
    )
    pre = build_pre_features(
        context,
        (record.game for record in history),
        config,
    )
    _require_history_target_consistency(pre, history, targets_by_id)
    post = build_post_features(
        pre,
        target.final_lineup,
        history,
        champion_reference,
    )

    if (
        post.pre is not pre
        or post.metadata.history_cutoff_at != pre.metadata.history_cutoff_at
        or post.accepted_game_ids != pre.accepted_game_ids
    ):
        raise DatasetInputError(
            "E_EVAL_INCOMPATIBLE",
            "POST does not preserve the paired PRE snapshot",
            target.game_id,
        )

    pre_values = {}
    for prefix, stat in _pre_stats(pre):
        for field in _PRE_FIELDS:
            if prefix.endswith("roster_continuity") and field == "win_count":
                continue
            pre_values[f"{prefix}_{field}"] = getattr(stat, field)

    # Copy the exact PRE values; do not recompute their formulas.
    post_values = dict(pre_values)
    for feature in post.player_champion:
        prefix = f"{feature.side.lower()}_{feature.role.lower()}"
        for field in _POST_FIELDS:
            post_values[f"{prefix}_{field}"] = getattr(feature, field)

    metadata = {
        "history_cutoff_at": pre.metadata.history_cutoff_at,
        "target_ended_at": end,
        "patch": context.patch,
        "blue_team_id": context.blue_team_id,
        "red_team_id": context.red_team_id,
        "blue_roster": context.blue_roster,
        "red_roster": context.red_roster,
        "evidence_ref": cutoff_record.evidence_ref,
        "policy_version": cutoff_record.policy_version,
        "cutoff_verification": cutoff_record.verification.value,
        # Preserve the feature core's own statement about source verification.
        "core_cutoff_verification": pre.metadata.cutoff_verification,
        "dataset_version": dataset_version,
        "pre_config_version": pre.metadata.config_version,
        "post_config_version": post.metadata.config_version,
        "pre_config": pre.metadata.config,
        "history_game_ids": pre.accepted_game_ids,
        "pre_feature_game_ids": tuple(
            (prefix, stat.game_ids) for prefix, stat in _pre_stats(pre)
        ),
        "post_feature_game_ids": tuple(
            (
                f"{feature.side.lower()}_{feature.role.lower()}",
                feature.game_ids,
            )
            for feature in post.player_champion
        ),
        "history_exclusions": pre.exclusions,
        "history_counts": pre.counts,
    }

    return _Pair(
        game_id=target.game_id,
        pre_values=pre_values,
        post_values=post_values,
        label=int(target.winner_team_id == target.blue_team_id),
        metadata=metadata,
    )


def build_paired_dataset(
    targets: Iterable[DatasetTarget],
    history: Iterable[HistoricalChampionGame],
    cutoffs: Iterable[CutoffRecord],
    champion_reference: frozenset[str],
    approved_policy_versions: frozenset[str],
    config: FeatureConfig = FeatureConfig(),
    *,
    evaluation_protocol: str = STRICT_PROTOCOL,
) -> PairedDataset:
    """Build paired tables without I/O, fitting or implicit cutoff creation.

    A bad identifiable target excludes one complete pair.
    Invalid outer record types or game IDs raise a batch contract error.
    Extra cutoff IDs are reported separately because they are not targets.
    A conflict with another target in accepted history excludes the consuming pair
    and reports that historical game ID as source_game_id.
    """
    dataset_version, _verification = protocol_contract(evaluation_protocol)
    targets = _records(targets, DatasetTarget, "targets")
    history = _records(history, HistoricalChampionGame, "history")
    cutoffs = _records(cutoffs, CutoffRecord, "cutoffs")

    if any(not isinstance(record.game, HistoricalGame) for record in history):
        raise DatasetInputError(
            "E_HISTORY_INPUT_INVALID",
            "Every history wrapper must contain HistoricalGame",
        )

    if (
        not isinstance(approved_policy_versions, frozenset)
        or not approved_policy_versions
        or any(not _nonempty_text(value) for value in approved_policy_versions)
    ):
        raise DatasetInputError(
            "E_CUTOFF_POLICY_SET_INVALID",
            "Caller must supply a nonempty frozenset of approved policy versions",
        )

    targets_by_id = defaultdict(list)
    cutoffs_by_id = defaultdict(list)

    for target in targets:
        _require_game_id(target.game_id)
        targets_by_id[target.game_id].append(target)
    for record in cutoffs:
        _require_game_id(record.game_id)
        cutoffs_by_id[record.game_id].append(record)

    pairs = []
    excluded = []

    for game_id in sorted(targets_by_id):
        candidates = targets_by_id[game_id]
        if len(candidates) != 1:
            excluded.append(
                DatasetExclusion(
                    game_id,
                    "E_TARGET_DUPLICATE_GAME_ID",
                    "All target records with this ID were excluded",
                )
            )
            continue

        try:
            cutoff_record, cutoff = _resolve_cutoff(
                game_id,
                tuple(cutoffs_by_id.get(game_id, ())),
                approved_policy_versions,
                evaluation_protocol=evaluation_protocol,
            )
            pair = _build_pair(
                candidates[0],
                cutoff_record,
                cutoff,
                history,
                champion_reference,
                config,
                targets_by_id,
                dataset_version,
            )
        except PreFeatureInputError as error:
            excluded.append(
                DatasetExclusion(
                    game_id=game_id,
                    reason=error.code,
                    detail=str(error),
                    source_game_id=error.game_id,
                )
            )
        else:
            # Append only after both PRE and POST have succeeded.
            pairs.append(pair)

    pairs.sort(
        key=lambda pair: (
            pair.metadata["history_cutoff_at"],
            pair.game_id,
        )
    )
    index = pd.Index(
        [pair.game_id for pair in pairs],
        name="game_id",
        dtype=object,
    )

    # Object dtype preserves None; no dataset-wide fillna or type imputation.
    X_pre = pd.DataFrame(
        [pair.pre_values for pair in pairs],
        index=index,
        columns=PRE_COLUMNS,
        dtype=object,
    )
    X_post = pd.DataFrame(
        [pair.post_values for pair in pairs],
        index=index,
        columns=POST_COLUMNS,
        dtype=object,
    )
    for column in CATEGORICAL_COLUMNS:
        # IDs were already validated by POST. This is storage, not encoding.
        X_post[column] = X_post[column].astype("string")

    labels = pd.Series(
        [pair.label for pair in pairs],
        index=index,
        name="y_blue_win",
        dtype="int8",
    )
    metadata = pd.DataFrame(
        [pair.metadata for pair in pairs],
        index=index,
        columns=METADATA_COLUMNS,
        dtype=object,
    )

    return PairedDataset(
        status="OK" if pairs else "EMPTY",
        reason=None if pairs else (
            "NO_TARGETS" if not targets else "NO_VALID_PAIRED_TARGETS"
        ),
        X_pre=X_pre,
        X_post=X_post,
        y=labels,
        metadata=metadata,
        excluded=tuple(excluded),
        unused_cutoff_game_ids=tuple(
            sorted(set(cutoffs_by_id) - set(targets_by_id))
        ),
    )


def _validate_tables(dataset: PairedDataset) -> tuple[str, ...]:
    if (
        not isinstance(dataset, PairedDataset)
        or not isinstance(dataset.X_pre, pd.DataFrame)
        or not isinstance(dataset.X_post, pd.DataFrame)
        or not isinstance(dataset.metadata, pd.DataFrame)
        or not isinstance(dataset.y, pd.Series)
    ):
        raise DatasetInputError(
            "E_PAIRED_DATASET_INVALID",
            "Expected PairedDataset containing DataFrames and a label Series",
        )

    index = dataset.X_pre.index
    if (
        not index.is_unique
        or not index.equals(dataset.X_post.index)
        or not index.equals(dataset.y.index)
        or not index.equals(dataset.metadata.index)
        or tuple(dataset.X_pre.columns) != PRE_COLUMNS
        or tuple(dataset.X_post.columns) != POST_COLUMNS
        or tuple(dataset.metadata.columns) != METADATA_COLUMNS
        or not dataset.X_post.loc[:, list(PRE_COLUMNS)].equals(dataset.X_pre)
        or dataset.y.name != "y_blue_win"
        or str(dataset.y.dtype) != "int8"
        or not dataset.y.isin((0, 1)).all()
        or dataset.categorical_columns != CATEGORICAL_COLUMNS
        or any(
            str(dataset.X_post[column].dtype) != "string"
            for column in CATEGORICAL_COLUMNS
        )
    ):
        raise DatasetInputError(
            "E_PAIRED_DATASET_INVALID",
            "PRE/POST/label/metadata alignment or feature schema is invalid",
        )

    game_ids = tuple(index)
    for game_id in game_ids:
        _require_game_id(game_id)

    excluded_ids = tuple(item.game_id for item in dataset.excluded)
    if (
        len(excluded_ids) != len(set(excluded_ids))
        or set(game_ids) & set(excluded_ids)
        or dataset.status != ("OK" if game_ids else "EMPTY")
    ):
        raise DatasetInputError(
            "E_PAIRED_DATASET_INVALID",
            "Dataset status or accepted/excluded disposition is inconsistent",
        )

    return game_ids


def temporal_split(
    dataset: PairedDataset,
    *,
    evaluation_protocol: str = STRICT_PROTOCOL,
) -> TemporalSplit:
    """Return one shared membership for PRE, POST and y.

    Boundaries depend only on cutoff groups and counts.
    Label-availability purging happens after boundary selection.
    A non-OK result must not be treated as a usable three-way split.
    """
    dataset_version, required_verification = protocol_contract(evaluation_protocol)
    split_type = (
        RetrospectiveTemporalSplit
        if evaluation_protocol == RETROSPECTIVE_PROTOCOL
        else TemporalSplit
    )
    game_ids = _validate_tables(dataset)
    if not game_ids:
        return split_type("INSUFFICIENT", "EMPTY_DATASET")

    cutoffs = {}
    ends = {}
    for game_id in game_ids:
        row = dataset.metadata.loc[game_id]
        if (
            row["dataset_version"] != dataset_version
            or (
                evaluation_protocol == RETROSPECTIVE_PROTOCOL
            ) != (row["policy_version"] == RETROSPECTIVE_PROTOCOL)
        ):
            raise DatasetInputError(
                "E_DATASET_PROTOCOL_MISMATCH",
                "Dataset row does not belong to the requested split protocol",
                game_id,
            )
        cutoffs[game_id] = _utc(
            row["history_cutoff_at"],
            "E_DATASET_CUTOFF_INVALID",
            game_id,
        )
        ends[game_id] = _utc(
            row["target_ended_at"],
            "E_DATASET_LABEL_END_INVALID",
            game_id,
        )
        if (
            row["cutoff_verification"]
            != required_verification.value
            or not _nonempty_text(row["evidence_ref"])
            or not _nonempty_text(row["policy_version"])
        ):
            raise DatasetInputError(
                "E_DATASET_CUTOFF_UNVERIFIED",
                "Dataset row lacks the verification contract for this protocol",
                game_id,
            )
        if ends[game_id] <= cutoffs[game_id]:
            raise DatasetInputError(
                "E_LABEL_END_NOT_AFTER_CUTOFF",
                "Target ended_at must be strictly after its PRE cutoff",
                game_id,
            )

    ordered = tuple(sorted(game_ids, key=lambda item: (cutoffs[item], item)))
    total = len(ordered)

    # Each interior position is between complete groups of equal cutoffs.
    positions = (
        [0]
        + [
            index
            for index in range(1, total)
            if cutoffs[ordered[index]] != cutoffs[ordered[index - 1]]
        ]
        + [total]
    )
    if len(positions) - 1 < 3:
        return split_type(
            "INSUFFICIENT",
            "FEWER_THAN_THREE_CUTOFF_GROUPS",
        )

    # Integer distances avoid floating-point ambiguity at exact ties.
    first = min(
        positions[1:-2],
        key=lambda position: (abs(100 * position - 70 * total), position),
    )
    second = min(
        (position for position in positions[1:-1] if position > first),
        key=lambda position: (abs(100 * position - 85 * total), position),
    )

    validation_boundary = cutoffs[ordered[first]]
    test_boundary = cutoffs[ordered[second]]
    planned = {
        "train": ordered[:first],
        "validation": ordered[first:second],
        "test": ordered[second:],
    }

    retained = {}
    excluded = []
    for partition, boundary, reason in (
        ("train", validation_boundary, "TRAIN_LABEL_NOT_AVAILABLE_BEFORE_VALIDATION"),
        ("validation", test_boundary, "VALIDATION_LABEL_NOT_AVAILABLE_BEFORE_TEST"),
        ("test", None, None),
    ):
        kept = []
        for game_id in planned[partition]:
            if boundary is not None and ends[game_id] >= boundary:
                excluded.append(
                    SplitExclusion(
                        game_id=game_id,
                        partition=partition,
                        reason=reason,
                        ended_at=ends[game_id],
                        required_before=boundary,
                    )
                )
            else:
                kept.append(game_id)
        retained[partition] = tuple(kept)

    retained_count = sum(len(values) for values in retained.values())
    summaries = []
    for partition in ("train", "validation", "test"):
        values = retained[partition]
        summaries.append(
            PartitionSummary(
                partition=partition,
                planned_count=len(planned[partition]),
                count=len(values),
                first_cutoff=cutoffs[values[0]] if values else None,
                last_cutoff=cutoffs[values[-1]] if values else None,
                max_ended_at=max(ends[item] for item in values) if values else None,
                fraction_of_input=len(values) / total,
                fraction_of_retained=(
                    len(values) / retained_count if retained_count else 0.0
                ),
            )
        )

    complete = all(retained[partition] for partition in retained)
    return split_type(
        status="OK" if complete else "INSUFFICIENT",
        reason=None if complete else "EMPTY_PARTITION_AFTER_LABEL_PURGE",
        train=retained["train"],
        validation=retained["validation"],
        test=retained["test"],
        validation_boundary=validation_boundary,
        test_boundary=test_boundary,
        excluded=tuple(excluded),
        summary=tuple(summaries),
    )
