"""Điều phối PRE/POST; boundary hồi cứu chỉ nạp model local, không DB hoặc fit."""

from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
import pandas as pd

from match_insight.features.post import (
    ChampionSlot,
    FinalLineup,
    HistoricalChampionGame,
    PlayerChampionFeature,
    PostFeatureResult,
    _require_pre_context,
    _require_reference,
    _require_slots,
    build_post_features,
)
from match_insight.features.pre import (
    ROLES,
    HistoricalGame,
    PreFeatureInputError,
    PreFeatureResult,
    TargetGame,
    build_pre_features,
    select_pre_history,
)
from match_insight.ml.dataset import (
    CATEGORICAL_COLUMNS,
    POST_COLUMNS,
    PRE_COLUMNS,
    RETROSPECTIVE_PROTOCOL,
    STRICT_PROTOCOL,
    CutoffRecord,
    _pre_stats,
    _resolve_cutoff,
)
from match_insight.ml.models import (
    CONTEXT_COLUMNS,
    ModelBundle,
    PairedPrediction,
    PhasePrediction,
    _check_bundle,
    _contract_protocol,
    _digest,
    _utc,
    compare_predictions,
    predict_phase,
)
from match_insight.ml.real_training import (
    CANONICAL_RETROSPECTIVE_EXPECTATION,
    MODEL_DIRECTORY,
    RETROSPECTIVE_MODEL_FILENAME,
    load_retrospective_bundle,
    retrospective_cutoff,
)


class EvaluationInputError(PreFeatureInputError):
    """Lỗi điều phối; không tạo prediction một phần."""


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    target: TargetGame
    cutoff: CutoffRecord | None
    history: tuple[HistoricalChampionGame, ...]
    champion_reference: frozenset[str]


@dataclass(frozen=True, slots=True)
class BusinessWarning:
    code: str
    team_id: str | None = None
    sample_count: int = 0
    requested_count: int = 0


@dataclass(frozen=True, slots=True)
class PreSnapshot:
    request: EvaluationRequest
    features: PreFeatureResult
    prediction: PhasePrediction
    warnings: tuple[BusinessWarning, ...]
    context_key: str
    seal: str


@dataclass(frozen=True, slots=True)
class PostSnapshot:
    pre: PreSnapshot
    lineup: FinalLineup
    features: PostFeatureResult
    prediction: PhasePrediction
    comparison: PairedPrediction
    warnings: tuple[BusinessWarning, ...]

    @property
    def delta_percentage_points(self):
        return percentage_points(self.comparison.delta_probability)


@dataclass(frozen=True, slots=True)
class EvaluationState:
    context_key: str | None = None
    champion_ids: tuple[str | None, ...] = ()
    pre: PreSnapshot | None = None
    post: PostSnapshot | None = None


def _fail(code, detail):
    raise EvaluationInputError(code, detail)


def _prepare_request(request, bundle, *, evaluation_protocol=STRICT_PROTOCOL):
    if not isinstance(request, EvaluationRequest) or not isinstance(request.target, TargetGame):
        _fail("E_PRE_INPUT_INCOMPLETE", "Expected an EvaluationRequest and TargetGame")
    _check_bundle(bundle)
    if _contract_protocol(bundle.contract) != evaluation_protocol:
        _fail("E_EVAL_INCOMPATIBLE", "Requested protocol differs from the model contract")
    _require_reference(request.champion_reference)
    owned = deepcopy(request)
    try:
        target = replace(
            owned.target,
            blue_roster=tuple(owned.target.blue_roster),
            red_roster=tuple(owned.target.red_roster),
        )
        records = tuple(owned.history)
    except TypeError:
        _fail("E_PRE_INPUT_INCOMPLETE", "Roster and history must be iterable")

    if any(
        not isinstance(record, HistoricalChampionGame) or not isinstance(record.game, HistoricalGame)
        for record in records
    ):
        _fail("E_HISTORY_INPUT_INVALID", "Expected HistoricalChampionGame records")

    try:
        history = tuple(
            replace(
                record,
                game=replace(
                    record.game,
                    blue_roster=tuple(record.game.blue_roster),
                    red_roster=tuple(record.game.red_roster),
                ),
                slots=tuple(record.slots),
            )
            for record in records
        )
    except TypeError:
        _fail("E_HISTORY_INPUT_INVALID", "Historical roster and champion slots must be iterable")

    # Validate context with the existing PRE validator, without creating features.
    target = select_pre_history(target, ()).target
    if owned.cutoff is not None and not isinstance(owned.cutoff, CutoffRecord):
        _fail("E_CUTOFF_STATUS_INVALID", "Expected CutoffRecord")
    if owned.cutoff is not None and owned.cutoff.game_id != target.game_id:
        _fail("E_EVAL_INCOMPATIBLE", "Cutoff belongs to another game")

    cutoff_record, cutoff = _resolve_cutoff(
        target.game_id,
        () if owned.cutoff is None else (owned.cutoff,),
        frozenset(bundle.contract.cutoff_policy_versions),
        evaluation_protocol=evaluation_protocol,
    )
    if cutoff != target.history_cutoff_at:
        _fail("E_EVAL_INCOMPATIBLE", "Target and cutoff record disagree")

    return EvaluationRequest(
        target=target,
        cutoff=replace(cutoff_record, history_cutoff_at=cutoff),
        history=history,
        champion_reference=owned.champion_reference,
    )


def _request_key(request, bundle):
    # Reference is serialized explicitly because the model serializer rejects sets.
    # Target record in history is excluded by 7G before timestamp validation;
    # mask it to avoid spurious E_MODEL_TIMESTAMP_INVALID on discarded records.
    target_id = request.target.game_id
    history_entries = tuple(
        (record.game.game_id, "TARGET_GAME")
        if record.game.game_id == target_id
        else record
        for record in request.history
    )
    return _digest(
        (
            request.target,
            request.cutoff,
            history_entries,
            tuple(sorted(request.champion_reference)),
            bundle.identity,
            bundle.fit_signature,
            bundle.contract,
            bundle.feature_schema_version,
            bundle.preprocessing_version,
            bundle.library_versions,
        )
    )


def request_key(request, bundle, *, evaluation_protocol=STRICT_PROTOCOL):
    """Validate and identify the complete PRE foundation."""
    normalized = _prepare_request(request, bundle, evaluation_protocol=evaluation_protocol)
    return _request_key(normalized, bundle)


def _pre_values(pre):
    return {
        f"{prefix}_{field}": getattr(stat, field)
        for prefix, stat in _pre_stats(pre)
        for field in ("value", "sample_count", "missing", "win_count")
        if f"{prefix}_{field}" in PRE_COLUMNS
    }


def _prediction_inputs(request, pre, bundle, post=None):
    target = pre.history_snapshot.target
    index = pd.Index([target.game_id], name="game_id", dtype=object)
    values = _pre_values(pre)
    metadata = {
        "history_cutoff_at": pre.metadata.history_cutoff_at,
        "patch": target.patch,
        "blue_team_id": target.blue_team_id,
        "red_team_id": target.red_team_id,
        "blue_roster": target.blue_roster,
        "red_roster": target.red_roster,
        "evidence_ref": request.cutoff.evidence_ref,
        "policy_version": request.cutoff.policy_version,
        "cutoff_verification": request.cutoff.verification.value,
        "core_cutoff_verification": pre.metadata.cutoff_verification,
        "dataset_version": bundle.contract.dataset_version,
        "pre_config_version": pre.metadata.config_version,
        # Expected schema only; this does not manufacture a POST snapshot.
        "post_config_version": bundle.contract.post_config_version,
        "pre_config": pre.metadata.config,
        "history_game_ids": pre.accepted_game_ids,
        "pre_feature_game_ids": tuple(
            (prefix, stat.game_ids) for prefix, stat in _pre_stats(pre)
        ),
        "history_exclusions": pre.exclusions,
        "history_counts": pre.counts,
    }
    columns = PRE_COLUMNS
    if post is not None:
        columns = POST_COLUMNS
        for feature in post.player_champion:
            prefix = f"{feature.side.lower()}_{feature.role.lower()}"
            for field in ("champion_id", "games_count", "wins_count", "win_rate", "missing"):
                values[f"{prefix}_{field}"] = getattr(feature, field)

    frame = pd.DataFrame([values], index=index, columns=columns, dtype=object)
    if post is not None:
        for column in CATEGORICAL_COLUMNS:
            frame[column] = frame[column].astype("string")
    context = pd.DataFrame(
        [metadata],
        index=index,
        columns=CONTEXT_COLUMNS,
        dtype=object,
    )
    return frame, context


def _pre_warnings(pre):
    warnings = []
    requested = pre.metadata.config.recent_form_games
    for team in (pre.blue, pre.red):
        count = team.recent_form.sample_count
        if count < requested:
            warnings.append(
                BusinessWarning("W_TEAM_HISTORY_SMALL", team.team_id, count, requested)
            )
    if pre.h2h_blue.missing:
        warnings.append(BusinessWarning("W_H2H_MISSING"))
    return tuple(warnings)


def _seal(snapshot):
    return _digest(
        (
            snapshot.context_key,
            snapshot.features,
            snapshot.prediction,
            snapshot.warnings,
        )
    )


def _require_snapshot(pre, bundle, *, evaluation_protocol=STRICT_PROTOCOL):
    if not isinstance(pre, PreSnapshot):
        _fail("E_PRE_SNAPSHOT_REQUIRED", "Create PRE before creating POST")
    try:
        expected_key = request_key(pre.request, bundle, evaluation_protocol=evaluation_protocol)
    except PreFeatureInputError:
        _fail("E_EVAL_INCOMPATIBLE", "The saved PRE foundation is no longer compatible")
    if pre.context_key != expected_key or pre.seal != _seal(pre):
        _fail("E_EVAL_INCOMPATIBLE", "PRE context, history, prediction or model changed")
    if pre.prediction.phase != "PRE":
        _fail("E_EVAL_INCOMPATIBLE", "Snapshot does not contain a PRE prediction")


def create_pre(
    request: EvaluationRequest,
    bundle: ModelBundle,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
):
    """Create PRE without target champions, winner or target end."""
    normalized = _prepare_request(request, bundle, evaluation_protocol=evaluation_protocol)
    features = build_pre_features(
        normalized.target,
        tuple(record.game for record in normalized.history),
        bundle.contract.pre_config,
    )
    frame, metadata = _prediction_inputs(normalized, features, bundle)
    prediction = predict_phase(bundle, "PRE", frame, metadata)[0]
    snapshot = PreSnapshot(
        request=normalized,
        features=features,
        prediction=prediction,
        warnings=_pre_warnings(features),
        context_key=_request_key(normalized, bundle),
        seal="",
    )
    return replace(snapshot, seal=_seal(snapshot))


def _require_retrospective_cutoff(request, target_started_at):
    """Check the explicit protocol cutoff without changing the supplied context."""
    if not isinstance(request, EvaluationRequest) or not isinstance(request.target, TargetGame):
        _fail("E_PRE_INPUT_INCOMPLETE", "Expected an EvaluationRequest and TargetGame")
    if request.cutoff is None:
        _fail("E_CUTOFF_MISSING", "An explicit retrospective cutoff record is required")
    if not isinstance(request.cutoff, CutoffRecord):
        _fail("E_CUTOFF_STATUS_INVALID", "Expected CutoffRecord")
    if request.cutoff.game_id != request.target.game_id:
        _fail("E_EVAL_INCOMPATIBLE", "Cutoff belongs to another game")
    _record, cutoff = _resolve_cutoff(
        request.target.game_id,
        (request.cutoff,),
        frozenset({RETROSPECTIVE_PROTOCOL}),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    if (
        cutoff != retrospective_cutoff(_utc(target_started_at))
        or _utc(request.target.history_cutoff_at) != cutoff
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Explicit cutoff differs from the retrospective formula")


def evaluate_retrospective_pre(
    request: EvaluationRequest,
    *,
    target_started_at,
    model_path=MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME,
) -> PreSnapshot:
    """Use the canonical retrospective model; return the existing PRE snapshot.

    The supplied start only checks the approved cutoff formula. Neither this
    timestamp nor the final context is certified as verified PRE evidence here.
    No cutoff is manufactured or changed, and no target champion is required.
    """
    _require_retrospective_cutoff(request, target_started_at)
    bundle = load_retrospective_bundle(
        model_path,
        expected=CANONICAL_RETROSPECTIVE_EXPECTATION,
    )
    return create_pre(request, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL)


def lineup_from_ids(pre, champion_ids):
    """Attach caller-selected champion identities to the saved PRE role mapping."""
    if not isinstance(pre, PreSnapshot):
        _fail("E_PRE_SNAPSHOT_REQUIRED", "Create PRE before entering the final lineup")
    ids = tuple(champion_ids)
    if len(ids) != 10 or any(
        not isinstance(value, str) or not value for value in ids
    ):
        _fail("E_LINEUP_INCOMPLETE", "Select all ten final champions")

    target = pre.request.target
    slots = []
    for side, team_id, roster in (
        ("BLUE", target.blue_team_id, target.blue_roster),
        ("RED", target.red_team_id, target.red_roster),
    ):
        by_role = {player.role: player.player_id for player in roster}
        for role in ROLES:
            slots.append(
                ChampionSlot(
                    team_id=team_id,
                    side=side,
                    role=role,
                    player_id=by_role[role],
                    champion_id=ids[len(slots)],
                )
            )
    return FinalLineup(target.game_id, target.patch, tuple(slots))


def validate_final_lineup(pre, lineup, bundle, *, evaluation_protocol=STRICT_PROTOCOL):
    """Validation for UI gating; use the same slot validator as step 7H."""
    _require_snapshot(pre, bundle, evaluation_protocol=evaluation_protocol)
    if not isinstance(lineup, FinalLineup):
        _fail("E_LINEUP_INCOMPLETE", "Expected FinalLineup")
    target = pre.request.target
    if lineup.game_id != target.game_id or lineup.patch != target.patch:
        _fail("E_EVAL_INCOMPATIBLE", "Final lineup belongs to another context")
    _require_reference(pre.request.champion_reference)
    slots = _require_slots(lineup.slots, target, pre.request.champion_reference)
    return replace(lineup, slots=slots)


def create_post(pre, lineup, bundle, *, evaluation_protocol=STRICT_PROTOCOL):
    """Create POST against the saved PRE; never recompute its probability."""
    lineup = validate_final_lineup(
        pre, lineup, bundle, evaluation_protocol=evaluation_protocol,
    )
    features = build_post_features(
        pre.features,
        lineup,
        pre.request.history,
        pre.request.champion_reference,
    )
    frame, metadata = _prediction_inputs(pre.request, pre.features, bundle, features)
    prediction = predict_phase(bundle, "POST", frame, metadata)[0]
    comparison = compare_predictions(pre.prediction, prediction)
    missing_pairs = sum(feature.missing for feature in features.player_champion)
    warnings = pre.warnings
    if missing_pairs:
        warnings += (
            BusinessWarning("W_PAIR_HISTORY_MISSING", sample_count=missing_pairs),
        )
    return PostSnapshot(pre, lineup, features, prediction, comparison, warnings)


def evaluate_retrospective_post(
    pre: PreSnapshot,
    lineup: FinalLineup,
    *,
    target_started_at,
    model_path=MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME,
) -> PostSnapshot:
    """Add POST to the supplied PRE using its sealed history and canonical model."""
    if not isinstance(pre, PreSnapshot):
        _fail("E_PRE_SNAPSHOT_REQUIRED", "Create PRE before creating POST")
    _require_retrospective_cutoff(pre.request, target_started_at)
    bundle = load_retrospective_bundle(
        model_path,
        expected=CANONICAL_RETROSPECTIVE_EXPECTATION,
    )
    return create_post(pre, lineup, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL)


def compare_retrospective_evaluations(pre, post, bundle):
    """Present the saved comparison from the canonical retrospective runtime.

    Inputs are service-created snapshots and an already loaded bundle. Artifact
    hashing and snapshot sealing remain at their existing creation boundaries;
    this function performs no I/O, hashing, inference or feature construction.
    """
    if not isinstance(pre, PreSnapshot) or not isinstance(post, PostSnapshot):
        _fail("E_EVAL_INCOMPATIBLE", "Expected saved PRE and POST snapshots")
    _check_bundle(bundle)
    expected = CANONICAL_RETROSPECTIVE_EXPECTATION
    if (
        _contract_protocol(bundle.contract) != RETROSPECTIVE_PROTOCOL
        or bundle.identity != expected.identity
        or bundle.family != expected.family
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Comparison requires the canonical retrospective runtime")
    if (
        not isinstance(post.pre, PreSnapshot)
        or post.pre != pre
        or not isinstance(post.features, PostFeatureResult)
        or post.features.pre != pre.features
        or not isinstance(pre.request, EvaluationRequest)
        or not isinstance(pre.request.cutoff, CutoffRecord)
    ):
        _fail("E_EVAL_INCOMPATIBLE", "POST does not preserve the supplied PRE foundation")

    target = _require_pre_context(pre.features)
    if pre.request.target != target or pre.request.cutoff.game_id != target.game_id:
        _fail("E_EVAL_INCOMPATIBLE", "Saved request differs from the PRE context")
    history = pre.request.history
    if not isinstance(history, tuple) or any(
        not isinstance(record, HistoricalChampionGame)
        or not isinstance(record.game, HistoricalGame)
        for record in history
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Expected the saved professional history")
    by_game_id = {record.game.game_id: record.game for record in history}
    if (
        len(by_game_id) != len(history)
        or len(history) != pre.features.counts.input_games
        or set(by_game_id) != set(pre.features.accepted_game_ids).union(
            exclusion.game_id for exclusion in pre.features.exclusions
        )
        or any(
            by_game_id.get(game.game_id) != game
            for game in pre.features.history_snapshot.games
        )
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Saved history differs from the PRE selection")
    _record, cutoff = _resolve_cutoff(
        target.game_id,
        (pre.request.cutoff,),
        frozenset(bundle.contract.cutoff_policy_versions),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    if (
        _utc(target.history_cutoff_at) != cutoff
        or pre.features.metadata.config != bundle.contract.pre_config
        or pre.features.metadata.config_version != bundle.contract.pre_config_version
        or post.features.metadata.config_version != bundle.contract.post_config_version
        or post.features.metadata.pre_config_version != pre.features.metadata.config_version
        or post.features.metadata.game_id != target.game_id
        or post.features.metadata.patch != target.patch
        or _utc(post.features.metadata.history_cutoff_at) != cutoff
        or post.features.accepted_game_ids != pre.features.accepted_game_ids
        or post.features.exclusions != pre.features.exclusions
        or post.features.counts != pre.features.counts
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Feature context, history or model contract changed")
    if not isinstance(post.lineup, FinalLineup):
        _fail("E_LINEUP_INCOMPLETE", "Expected FinalLineup")
    if post.lineup.game_id != target.game_id or post.lineup.patch != target.patch:
        _fail("E_EVAL_INCOMPATIBLE", "Final lineup belongs to another context")
    _require_reference(pre.request.champion_reference)
    slots = _require_slots(post.lineup.slots, target, pre.request.champion_reference)
    pairs = post.features.player_champion
    if not isinstance(pairs, tuple) or any(
        not isinstance(pair, PlayerChampionFeature) for pair in pairs
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Expected saved player-champion features")
    identity_fields = ("team_id", "side", "role", "player_id", "champion_id")
    if sorted(tuple(getattr(slot, field) for field in identity_fields) for slot in slots) != sorted(
        tuple(getattr(pair, field) for field in identity_fields) for pair in pairs
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Saved POST features differ from the final lineup")

    # Recompute only to check invariants. The saved comparison remains the output.
    checked = compare_predictions(pre.prediction, post.prediction)
    if (
        checked.game_id != target.game_id
        or checked.history_cutoff_at != cutoff
        or checked.policy_version != RETROSPECTIVE_PROTOCOL
        or checked.model_bundle_version != bundle.identity.bundle_version
        or pre.prediction.bundle_signature != bundle.fit_signature
        or checked.feature_schema_version != bundle.feature_schema_version
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Predictions differ from the saved context or runtime")
    comparison = post.comparison
    if not isinstance(comparison, PairedPrediction):
        _fail("E_EVAL_INCOMPATIBLE", "POST requires its saved comparison")
    numeric_fields = ("p_blue_win_pre", "p_blue_win_post", "delta_probability")
    for field in numeric_fields:
        value = getattr(comparison, field)
        lower = -1 if field == "delta_probability" else 0
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not lower <= value <= 1
        ):
            _fail("E_MODEL_PROBABILITY_INVALID", "Saved comparison contains an invalid probability")
    if (
        replace(comparison, **{field: getattr(checked, field) for field in numeric_fields})
        != checked
        # Match the existing model probability validation convention; no new epsilon.
        or not np.allclose(
            [getattr(comparison, field) for field in numeric_fields],
            [getattr(checked, field) for field in numeric_fields],
        )
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Saved comparison disagrees with PRE and POST")

    warnings = (*pre.warnings, *post.warnings)
    if any(not isinstance(warning, BusinessWarning) for warning in warnings):
        _fail("E_EVAL_INCOMPATIBLE", "Expected saved business warnings")
    warnings = sorted(
        set(warnings),
        key=lambda warning: (
            warning.code,
            warning.team_id is not None,
            warning.team_id or "",
            warning.sample_count,
            warning.requested_count,
        ),
    )
    return {
        "game_id": comparison.game_id,
        "history_cutoff_at": _utc(comparison.history_cutoff_at).isoformat(),
        "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "pre": {
            "evaluation_type": "PRE",
            "blue_win_probability": float(comparison.p_blue_win_pre),
            "red_win_probability": float(1 - comparison.p_blue_win_pre),
        },
        "post": {
            "evaluation_type": "POST",
            "blue_win_probability": float(comparison.p_blue_win_post),
            "red_win_probability": float(1 - comparison.p_blue_win_post),
        },
        "comparison": {
            "blue_probability_delta": float(comparison.delta_probability),
            "red_probability_delta": float(-comparison.delta_probability),
        },
        "model": {
            "family": bundle.family,
            "dataset_id": bundle.identity.dataset_id,
            "split_id": bundle.identity.split_id,
            "bundle_version": bundle.identity.bundle_version,
            "feature_schema_version": bundle.feature_schema_version,
            "preprocessing_version": bundle.preprocessing_version,
        },
        "warnings": [
            {
                "code": warning.code,
                "team_id": warning.team_id,
                "sample_count": warning.sample_count,
                "requested_count": warning.requested_count,
            }
            for warning in warnings
        ],
    }


def refresh_state(state, request, bundle, champion_ids):
    """Invalidate stale results before rendering any probability."""
    if not isinstance(state, EvaluationState):
        _fail("E_EVAL_INCOMPATIBLE", "Expected EvaluationState")
    key = request_key(request, bundle)
    champions = tuple(champion_ids)
    if key != state.context_key:
        return EvaluationState(context_key=key, champion_ids=champions)
    if champions != state.champion_ids:
        return replace(state, champion_ids=champions, post=None)
    return state


def percentage_points(delta_probability):
    if (
        isinstance(delta_probability, bool)
        or not isinstance(delta_probability, Real)
        or not np.isfinite(delta_probability)
        or not -1 <= delta_probability <= 1
    ):
        _fail("E_MODEL_PROBABILITY_INVALID", "Invalid probability difference")
    return float(delta_probability) * 100
