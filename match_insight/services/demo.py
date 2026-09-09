"""Retrospective UI application boundary; legacy synthetic helpers serve tests only."""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from match_insight.database.demo_catalog import DemoCatalog, read_demo_catalog
from match_insight.database.real_snapshot import RealSnapshot, fail, read_snapshot
from match_insight.features.post import ChampionSlot, FinalLineup, HistoricalChampionGame
from match_insight.features.pre import (
    ROLES,
    HistoricalGame,
    PlayerSlot,
    PreFeatureInputError,
    TargetGame,
    select_pre_history,
)
from match_insight.ml.dataset import (
    RETROSPECTIVE_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetTarget,
    _pre_stats,
    build_paired_dataset,
    temporal_split,
)
from match_insight.ml.models import (
    INTERACTIVE_INFERENCE_POLICY,
    ModelBundle,
    ModelConfig,
    ModelIdentity,
    RuntimeInferenceContext,
    _check_bundle,
    _digest,
    _utc,
    fit_model_bundle,
    interactive_cutoff,
    validate_runtime_context,
)
from match_insight.ml.real_training import (
    CANONICAL_RETROSPECTIVE_EXPECTATION,
    MODEL_DIRECTORY,
    RETROSPECTIVE_INPUT,
    RETROSPECTIVE_MODEL_FILENAME,
    TIME_EVIDENCE_GRADE,
    _json_document,
    _require_pre_evidence,
    _retrospective_header,
    _time_iso,
    _time_ms,
    load_retrospective_bundle,
    retrospective_cutoff,
)
from match_insight.ml.real_training import (
    _cutoff_record as _evidence_cutoff_record,
)
from match_insight.services.evaluation import (
    EvaluationRequest,
    EvaluationState,
    PostSnapshot,
    PreSnapshot,
    _require_snapshot,
    compare_interactive_evaluations,
    compare_retrospective_evaluations,
    evaluate_interactive_post,
    evaluate_interactive_pre,
    evaluate_retrospective_post,
    evaluate_retrospective_pre,
    lineup_from_ids,
    request_key,
    validate_final_lineup,
)

RETROSPECTIVE_NOTICE = "Mô phỏng hồi cứu trên dữ liệu thật"
# Reviewed input bytes from the canonical training run. Hashes identify content,
# not source authenticity; the protocol/context remain assumed/reconstructed.
CANONICAL_DEMO_INPUT_SHA256 = "5f35e2784ff3fe83c8c70402a9878af5313cbc780b5aaf4e5317fa9553e630ff"
CANONICAL_DEMO_SNAPSHOT_SHA256 = "344a5d43b8775c4e6afa61f6e5ccbadeb5e098bcedc53843835b4289b5ecf48d"


@dataclass(frozen=True, slots=True)
class RetrospectiveTarget:
    game_id: str
    target: TargetGame
    cutoff: CutoffRecord
    target_started_at: datetime


@dataclass(frozen=True, slots=True)
class RetrospectiveCase:
    game_id: str
    request: EvaluationRequest
    target_started_at: datetime


@dataclass(frozen=True, slots=True)
class RetrospectiveDemoRuntime:
    bundle: ModelBundle
    catalog: DemoCatalog
    targets: tuple[RetrospectiveTarget, ...]
    history: tuple[HistoricalChampionGame, ...]
    champion_reference: frozenset[str]
    snapshot_sha256: str
    evidence_sha256: str
    model_path: Path
    snapshot_read_at: datetime
    data_kind: str = "REAL_RETROSPECTIVE_SIMULATION"
    evaluation_protocol: str = RETROSPECTIVE_PROTOCOL


def _read_pinned_input(path):
    try:
        data = Path(path).read_bytes()
    except (OSError, TypeError, ValueError):
        fail("E_CUTOFF_MISSING")
    if hashlib.sha256(data).hexdigest() != CANONICAL_DEMO_INPUT_SHA256:
        fail("E_EVIDENCE_HASH_MISMATCH")
    document = _json_document(data)
    _retrospective_header(document)
    if document["snapshot_sha256"] != CANONICAL_DEMO_SNAPSHOT_SHA256:
        fail("E_EVIDENCE_SNAPSHOT_MISMATCH")
    return document


def _proof_times(proof):
    """Check stored proof consistency without re-opening the archived payloads."""
    if not isinstance(proof, dict) or proof.get("grade") != TIME_EVIDENCE_GRADE:
        fail("E_TIME_PROOF_CONFLICT")
    start = _time_ms(proof.get("start_timestamp_ms"))
    end = _time_ms(proof.get("end_timestamp_ms"))
    if (
        _time_iso(proof.get("started_at")) != start
        or _time_iso(proof.get("ended_at")) != end
    ):
        fail("E_TIME_SOURCE_CONFLICT")
    if start >= end:
        fail("E_TIME_ORDER_INVALID")
    if _time_iso(proof.get("fetched_at_utc")) < end:
        fail("E_TIME_RETRIEVAL_CONFLICT")
    return start, end


def _read_runtime_data():
    """One explicit initialization; no connection or credential enters session state."""
    from sqlalchemy import create_engine

    from match_insight.config import Settings

    keys = ("DATABASE_URL", "WIKI_USERNAME_MATCH_INSIGHT", "WIKI_PASSWORD_MATCH_INSIGHT")
    previous = {key: os.environ.get(key) for key in keys}
    engine = None
    try:
        engine = create_engine(Settings().database_url)
        return read_snapshot(engine), read_demo_catalog(engine)
    except PreFeatureInputError:
        raise
    except Exception:
        fail("E_SOURCE_CONFLICT")
    finally:
        try:
            if engine is not None:
                engine.dispose()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _validated_history_sources(snapshot, document):
    """Validate the pinned historical source, without selecting an analysis target."""
    if (
        not isinstance(snapshot, RealSnapshot)
        or snapshot.sha256 != CANONICAL_DEMO_SNAPSHOT_SHA256
        or snapshot.sha256 != document["snapshot_sha256"]
    ):
        fail("E_EVIDENCE_SNAPSHOT_MISMATCH")
    games = {}
    for item in snapshot.games:
        if item.game.game_id in games:
            fail("E_DB_DUPLICATE_ID")
        games[item.game.game_id] = item
    sources = []
    for raw in document["records"]:
        game_id = raw["game_id"]
        # A pinned evidence record absent from the pinned DB snapshot is a conflict.
        if game_id not in games:
            fail("E_CUTOFF_UNKNOWN_GAME", game_id)
        item = games[game_id]
        start, end = _proof_times(document["time_proofs"][game_id])
        for value, expected in ((item.started_at, start), (item.game.ended_at, end)):
            if value is not None and _utc(value) != expected:
                fail("E_TIME_SOURCE_CONFLICT", game_id)
        _require_pre_evidence(item, raw, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
        if raw["pre_context_evidence_ref"] != f"snapshot:{snapshot.sha256}:{game_id}":
            fail("E_PRE_CONTEXT_MISMATCH", game_id)
        cutoff = _evidence_cutoff_record(raw)
        if cutoff.history_cutoff_at != retrospective_cutoff(start):
            fail("E_CUTOFF_INVALID", game_id)
        sources.append((item, cutoff, start, end,
                        _time_iso(document["time_proofs"][game_id]["fetched_at_utc"])))
    return tuple(sources)


def _runtime_targets(snapshot, document):
    targets, history = [], []
    for item, cutoff, start, end, _fetched_at in _validated_history_sources(snapshot, document):
        game = item.game
        game_id = game.game_id
        target = TargetGame(
            game_id=game_id,
            blue_team_id=game.blue_team_id,
            red_team_id=game.red_team_id,
            blue_roster=game.blue_roster,
            red_roster=game.red_roster,
            patch=item.final_lineup.patch,
            history_cutoff_at=cutoff.history_cutoff_at,
        )
        target = select_pre_history(target, ()).target
        targets.append(RetrospectiveTarget(game_id, target, cutoff, start))
        history.append(
            HistoricalChampionGame(replace(game, ended_at=end), item.final_lineup.slots)
        )
    if not targets:
        fail("E_NO_ELIGIBLE_TARGETS")
    targets.sort(key=lambda item: (item.target_started_at, item.game_id))
    history.sort(key=lambda item: item.game.game_id)
    return tuple(targets), tuple(history)


def load_retrospective_demo(
    *, input_path=RETROSPECTIVE_INPUT,
    model_path=MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME,
):
    """Load reviewed content and read DB once; never fit, save or build a dataset."""
    document = _read_pinned_input(input_path)
    bundle = load_retrospective_bundle(
        model_path, expected=CANONICAL_RETROSPECTIVE_EXPECTATION,
    )
    snapshot, catalog = _read_runtime_data()
    targets, history = _runtime_targets(snapshot, document)
    teams = {row.identity for row in catalog.teams}
    players = {row.identity for row in catalog.players}
    displayed_games = {row.game_id for row in catalog.games}
    for item in targets:
        target = item.target
        if (
            item.game_id not in displayed_games
            or not {target.blue_team_id, target.red_team_id} <= teams
            or not {
                slot.player_id for slot in target.blue_roster + target.red_roster
            } <= players
        ):
            fail("E_PRE_INPUT_INCOMPLETE", item.game_id)
    return RetrospectiveDemoRuntime(
        bundle=bundle,
        catalog=catalog,
        targets=targets,
        history=history,
        champion_reference=snapshot.champion_reference,
        snapshot_sha256=snapshot.sha256,
        evidence_sha256=CANONICAL_DEMO_INPUT_SHA256,
        model_path=Path(model_path),
        snapshot_read_at=snapshot.captured_at,
    )


def get_retrospective_case(runtime, game_id):
    matches = tuple(item for item in runtime.targets if item.game_id == game_id)
    if len(matches) != 1:
        fail("E_PRE_INPUT_INCOMPLETE", game_id)
    item = matches[0]
    selection = select_pre_history(item.target, (record.game for record in runtime.history))
    by_id = {record.game.game_id: record for record in runtime.history}
    history = tuple(by_id[game.game_id] for game in selection.games)
    return RetrospectiveCase(
        game_id,
        EvaluationRequest(item.target, item.cutoff, history, runtime.champion_reference),
        item.target_started_at,
    )


def refresh_retrospective_state(state, case, bundle, *, champion_ids=None):
    key = request_key(case.request, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    if not isinstance(state, EvaluationState) or state.context_key != key:
        return EvaluationState(context_key=key)
    if champion_ids is not None:
        champions = tuple(champion_ids)
        if champions != state.champion_ids:
            # Comparison belongs to PostSnapshot, so clearing POST clears it too.
            return replace(state, champion_ids=champions, post=None)
    return state


def evaluate_demo_pre(runtime, game_id):
    case = get_retrospective_case(runtime, game_id)
    pre = evaluate_retrospective_pre(
        case.request,
        target_started_at=case.target_started_at,
        model_path=runtime.model_path,
    )
    if pre.context_key != request_key(
        case.request, runtime.bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    ):
        fail("E_EVAL_INCOMPATIBLE", game_id)
    return pre


def _require_runtime_pre(runtime, pre):
    """Bind a saved PRE to its retained input; never rebuild PRE or select new history."""
    if not isinstance(pre, PreSnapshot):
        fail("E_PRE_SNAPSHOT_REQUIRED")
    if (
        not isinstance(runtime, RetrospectiveDemoRuntime)
        or runtime.data_kind != "REAL_RETROSPECTIVE_SIMULATION"
        or runtime.evaluation_protocol != RETROSPECTIVE_PROTOCOL
        or runtime.snapshot_sha256 != CANONICAL_DEMO_SNAPSHOT_SHA256
        or runtime.evidence_sha256 != CANONICAL_DEMO_INPUT_SHA256
        or runtime.bundle.identity != CANONICAL_RETROSPECTIVE_EXPECTATION.identity
        or runtime.bundle.family != CANONICAL_RETROSPECTIVE_EXPECTATION.family
    ):
        fail("E_EVAL_INCOMPATIBLE")
    game_id = pre.request.target.game_id
    matches = tuple(item for item in runtime.targets if item.game_id == game_id)
    if len(matches) != 1:
        fail("E_EVAL_INCOMPATIBLE", game_id)
    target = matches[0]
    if (
        pre.request.target != target.target
        or pre.request.cutoff != target.cutoff
        or pre.request.champion_reference != runtime.champion_reference
        or target.cutoff.history_cutoff_at != retrospective_cutoff(target.target_started_at)
    ):
        fail("E_EVAL_INCOMPATIBLE", game_id)
    approved_history = {}
    for record in runtime.history:
        if record.game.game_id in approved_history:
            fail("E_HISTORY_DUPLICATE_GAME_ID")
        approved_history[record.game.game_id] = record
    if any(approved_history.get(record.game.game_id) != record for record in pre.request.history):
        fail("E_PRE_HISTORY_MISMATCH", game_id)
    return target


def validate_demo_lineup(runtime, pre, champion_ids):
    _require_runtime_pre(runtime, pre)
    try:
        lineup = lineup_from_ids(pre, champion_ids)
    except TypeError:
        fail("E_LINEUP_INCOMPLETE")
    return validate_final_lineup(
        pre, lineup, runtime.bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )


def evaluate_demo_post(runtime, pre, champion_ids):
    target = _require_runtime_pre(runtime, pre)
    lineup = validate_demo_lineup(runtime, pre, champion_ids)
    post = evaluate_retrospective_post(
        pre,
        lineup,
        target_started_at=target.target_started_at,
        model_path=runtime.model_path,
    )
    if post.pre is not pre:
        fail("E_EVAL_INCOMPATIBLE", target.game_id)
    return post


def phase_probabilities(snapshot):
    if not isinstance(snapshot, (PreSnapshot, PostSnapshot)):
        fail("E_PRE_SNAPSHOT_REQUIRED")
    blue = float(snapshot.prediction.p_blue_win)
    if not isfinite(blue) or not 0 <= blue <= 1:
        fail("E_MODEL_PROBABILITY_INVALID")
    return blue, 1.0 - blue


def compare_demo_evaluations(runtime, pre, post):
    """Derive presentation from the current saved pair without I/O or inference."""
    _require_runtime_pre(runtime, pre)
    return compare_retrospective_evaluations(pre, post, runtime.bundle)


@dataclass(frozen=True, slots=True)
class AnalysisInput:
    """User context for an analysis, never a reconstructed database game."""

    blue_team_id: str
    red_team_id: str
    blue_roster: tuple[PlayerSlot, ...]
    red_roster: tuple[PlayerSlot, ...]
    patch: str
    user_confirmed: bool
    analysis_id: str = field(default_factory=lambda: f"analysis:{uuid4()}")
    context_source: str = "USER_PROVIDED"
    pre_draft_verification: str = "NOT_VERIFIED"
    context_label: str | None = None
    planned_start_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AnalysisRuntime:
    """Pinned history and loaded model retained in one session; no connection."""

    bundle: ModelBundle
    catalog: DemoCatalog
    history: tuple[HistoricalChampionGame, ...]
    champion_reference: frozenset[str]
    proof_fetched_at: tuple[tuple[str, datetime], ...]
    history_patches: tuple[tuple[str, str], ...]
    snapshot_sha256: str
    evidence_sha256: str
    history_pool_sha256: str
    snapshot_read_at: datetime
    runtime_ready_at: datetime
    source_game_count: int


@dataclass(frozen=True, slots=True)
class RosterSuggestion:
    team_id: str
    roster: tuple[PlayerSlot, ...]
    source_game_id: str
    source_ended_at: datetime
    history_cutoff_at: datetime
    source: str = "HISTORICAL_ROSTER_NOT_CURRENT_VERIFIED"


@dataclass(frozen=True, slots=True)
class SelectionCoverage:
    """Read-only presentation index, retained only in its owning runtime session.

    Counts describe distinct linked games before this cutoff, not feature windows
    or confidence. No name, namespace or media-based identity mapping is applied.
    """

    history_cutoff_at: datetime
    snapshot_sha256: str
    evidence_sha256: str
    history_pool_sha256: str
    accepted_game_ids: tuple[str, ...]
    team_counts: Mapping[str, int]
    player_counts: Mapping[str, int]
    roster_suggestions: Mapping[str, RosterSuggestion]
    _runtime: AnalysisRuntime = field(repr=False, compare=False)
    _pre: PreSnapshot | None = field(default=None, repr=False, compare=False)


def _utc_now():
    return datetime.now(UTC)


def _pool_digest(history, fetched, patches):
    return _digest((history, fetched, patches))


def _require_analysis_runtime(runtime):
    if not isinstance(runtime, AnalysisRuntime) or not isinstance(runtime.catalog, DemoCatalog):
        fail("E_EVAL_INCOMPATIBLE")
    _check_bundle(runtime.bundle)
    if (
        runtime.bundle.identity != CANONICAL_RETROSPECTIVE_EXPECTATION.identity
        or runtime.bundle.family != CANONICAL_RETROSPECTIVE_EXPECTATION.family
        or runtime.snapshot_sha256 != CANONICAL_DEMO_SNAPSHOT_SHA256
        or runtime.evidence_sha256 != CANONICAL_DEMO_INPUT_SHA256
    ):
        fail("E_EVAL_INCOMPATIBLE")
    if (
        not isinstance(runtime.history, tuple)
        or any(not isinstance(record, HistoricalChampionGame) for record in runtime.history)
        or not isinstance(runtime.proof_fetched_at, tuple)
        or not isinstance(runtime.history_patches, tuple)
    ):
        fail("E_HISTORY_INPUT_INVALID")
    ids = tuple(record.game.game_id for record in runtime.history)
    if (
        ids != tuple(sorted(set(ids)))
        or tuple(game_id for game_id, _ in runtime.proof_fetched_at) != ids
        or tuple(game_id for game_id, _ in runtime.history_patches) != ids
        or _pool_digest(runtime.history, runtime.proof_fetched_at, runtime.history_patches)
        != runtime.history_pool_sha256
    ):
        fail("E_PRE_HISTORY_MISMATCH")
    ready = _utc(runtime.runtime_ready_at)
    if _utc(runtime.snapshot_read_at) > ready:
        fail("E_TIME_RETRIEVAL_CONFLICT")
    for record, (_game_id, fetched_at) in zip(runtime.history, runtime.proof_fetched_at, strict=True):
        if not _utc(record.game.ended_at) <= _utc(fetched_at) <= ready:
            fail("E_TIME_RETRIEVAL_CONFLICT", record.game.game_id)
    return runtime


def load_analysis_runtime(
    *, input_path=RETROSPECTIVE_INPUT,
    model_path=MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME,
):
    """Explicit read-only initialization; source records are history, not target options."""
    document = _read_pinned_input(input_path)
    bundle = load_retrospective_bundle(
        model_path, expected=CANONICAL_RETROSPECTIVE_EXPECTATION,
    )
    snapshot, catalog = _read_runtime_data()
    sources = _validated_history_sources(snapshot, document)
    sources = sorted(sources, key=lambda source: source[0].game.game_id)
    history = tuple(HistoricalChampionGame(
        replace(item.game, ended_at=end), item.final_lineup.slots,
    ) for item, _cutoff, _start, end, _fetched in sources)
    fetched = tuple((item.game.game_id, fetched_at)
                    for item, _cutoff, _start, _end, fetched_at in sources)
    patches = tuple((item.game.game_id, item.final_lineup.patch)
                    for item, _cutoff, _start, _end, _fetched in sources)
    teams = {row.identity for row in catalog.teams}
    players = {row.identity for row in catalog.players}
    if (
        not snapshot.champion_reference <= {row.identity for row in catalog.champions}
        or any(
            not {record.game.blue_team_id, record.game.red_team_id} <= teams
            or not {slot.player_id for slot in record.game.blue_roster + record.game.red_roster}
            <= players
            for record in history
        )
    ):
        fail("E_PRE_INPUT_INCOMPLETE")
    runtime = AnalysisRuntime(
        bundle, catalog, history, snapshot.champion_reference, fetched, patches,
        snapshot.sha256, CANONICAL_DEMO_INPUT_SHA256, _pool_digest(history, fetched, patches),
        _utc(snapshot.captured_at), _utc(_utc_now()), len(snapshot.games),
    )
    return _require_analysis_runtime(runtime)


def _analysis_target(runtime, selection, cutoff):
    if (
        not isinstance(selection, AnalysisInput)
        or selection.user_confirmed is not True
        or selection.context_source != "USER_PROVIDED"
        or selection.pre_draft_verification != "NOT_VERIFIED"
        or not isinstance(selection.analysis_id, str)
        or not selection.analysis_id.startswith("analysis:")
    ):
        fail("E_PRE_INPUT_INCOMPLETE")
    try:
        blue_roster, red_roster = tuple(selection.blue_roster), tuple(selection.red_roster)
    except TypeError:
        fail("E_PRE_INPUT_INCOMPLETE", selection.analysis_id)
    target = select_pre_history(TargetGame(
        selection.analysis_id, selection.blue_team_id, selection.red_team_id,
        blue_roster, red_roster, selection.patch, cutoff,
    ), ()).target
    if (
        not {target.blue_team_id, target.red_team_id} <= {row.identity for row in runtime.catalog.teams}
        or not {slot.player_id for slot in target.blue_roster + target.red_roster}
        <= {row.identity for row in runtime.catalog.players}
        or target.game_id in {record.game.game_id for record in runtime.history}
        or (selection.context_label is not None and (
            not isinstance(selection.context_label, str) or not selection.context_label.strip()
        ))
    ):
        fail("E_PRE_INPUT_INCOMPLETE", target.game_id)
    if selection.planned_start_at is not None:
        _utc(selection.planned_start_at)
    return target


def evaluate_analysis_pre(runtime, selection):
    """Create a current analysis; the public API has no clock/cutoff override."""
    _require_analysis_runtime(runtime)
    created = _utc(_utc_now())
    cutoff = interactive_cutoff(created)
    target = _analysis_target(runtime, selection, cutoff)
    context = validate_runtime_context(RuntimeInferenceContext(
        analysis_id=target.game_id,
        pre_created_at=created,
        history_cutoff_at=cutoff,
        runtime_ready_at=runtime.runtime_ready_at,
        snapshot_read_at=runtime.snapshot_read_at,
        snapshot_sha256=runtime.snapshot_sha256,
        evidence_sha256=runtime.evidence_sha256,
        history_pool_sha256=runtime.history_pool_sha256,
        history_pool_game_ids=tuple(record.game.game_id for record in runtime.history),
        history_available_at=max((value for _, value in runtime.proof_fetched_at),
                                 default=runtime.runtime_ready_at),
        context_source=selection.context_source,
        user_confirmed=selection.user_confirmed,
        pre_draft_verification=selection.pre_draft_verification,
        context_label=selection.context_label,
        planned_start_at=selection.planned_start_at,
    ))
    request = EvaluationRequest(
        target, None, runtime.history, runtime.champion_reference, runtime_context=context,
    )
    return evaluate_interactive_pre(request, runtime.bundle)


def _require_analysis_pre(runtime, pre):
    _require_analysis_runtime(runtime)
    _require_snapshot(pre, runtime.bundle, inference_policy=INTERACTIVE_INFERENCE_POLICY)
    context = validate_runtime_context(pre.request.runtime_context)
    if (
        context.snapshot_sha256 != runtime.snapshot_sha256
        or context.evidence_sha256 != runtime.evidence_sha256
        or context.history_pool_sha256 != runtime.history_pool_sha256
        or context.snapshot_read_at != runtime.snapshot_read_at
        or context.runtime_ready_at != runtime.runtime_ready_at
        or context.history_available_at != max(
            (_utc(value) for _, value in runtime.proof_fetched_at),
            default=runtime.runtime_ready_at,
        )
        or pre.request.history != runtime.history
        or pre.request.champion_reference != runtime.champion_reference
    ):
        fail("E_EVAL_INCOMPATIBLE")
    return context


def validate_analysis_lineup(runtime, pre, champion_ids):
    _require_analysis_pre(runtime, pre)
    try:
        lineup = lineup_from_ids(pre, champion_ids)
    except TypeError:
        fail("E_LINEUP_INCOMPLETE")
    return validate_final_lineup(
        pre, lineup, runtime.bundle, inference_policy=INTERACTIVE_INFERENCE_POLICY,
    )


def evaluate_analysis_post(runtime, pre, champion_ids):
    lineup = validate_analysis_lineup(runtime, pre, champion_ids)
    return evaluate_interactive_post(pre, lineup, runtime.bundle)


def analysis_history_summary(runtime, pre, post=None):
    """Report existing core selections; never invent trace IDs or select feature windows."""
    context = _require_analysis_pre(runtime, pre)
    games = {record.game.game_id: record.game for record in pre.request.history}
    before_cutoff = pre.features.accepted_game_ids
    pre_ids = {game_id for _name, stat in _pre_stats(pre.features) for game_id in stat.game_ids}
    post_ids = set()
    if post is not None:
        # Also checks the saved POST foundation before reading its trace.
        compare_interactive_evaluations(pre, post, runtime.bundle)
        post_ids = {game_id for stat in post.features.player_champion for game_id in stat.game_ids}

    def coverage(ids):
        ids = tuple(sorted(ids))
        latest = max((_utc(games[game_id].ended_at) for game_id in ids), default=None)
        return {
            "game_count": len(ids), "game_ids": list(ids),
            "latest_ended_at": None if latest is None else latest.isoformat(),
            "age_days_at_pre": None if latest is None else (context.pre_created_at - latest).days,
        }

    teams = []
    for team in (pre.features.blue, pre.features.red):
        relevant = tuple(game_id for game_id in before_cutoff if team.team_id in (
            games[game_id].blue_team_id, games[game_id].red_team_id,
        ))
        used = pre_ids.intersection(relevant)
        teams.append({
            "team_id": team.team_id, "side": team.side,
            "before_cutoff": coverage(relevant), "pre_used": coverage(used),
            "recent_form_sample_count": team.recent_form.sample_count,
            "side_win_rate_sample_count": team.side_win_rate.sample_count,
            "roster_continuity_sample_count": team.roster_continuity.sample_count,
        })
    return {
        "source_game_count": runtime.source_game_count,
        "proof_pool": coverage(games), "before_cutoff": coverage(before_cutoff),
        "pre_used": coverage(pre_ids), "post_pair_used": coverage(post_ids),
        "teams": teams,
        "patch_scope_status": "NOT_VERIFIED",
        "patch_observed_in_history": pre.request.target.patch in {
            patch for game_id, patch in runtime.history_patches if game_id in before_cutoff
        },
        "coverage_notice": "Chưa ghi nhận nghĩa là chưa có trong kho lịch sử đang dùng.",
    }


def compare_analysis_evaluations(runtime, pre, post):
    _require_analysis_pre(runtime, pre)
    result = compare_interactive_evaluations(pre, post, runtime.bundle)
    result["history_coverage"] = analysis_history_summary(runtime, pre, post)
    return result


def _analysis_history_before_cutoff(runtime, cutoff):
    # Presentation suggestion from one complete, proven game; no feature selection
    # is fabricated and no historical context is repurposed as an analysis target.
    games = sorted((record.game for record in runtime.history
                    if _utc(record.game.ended_at) < cutoff), key=lambda game: game.game_id)
    games.sort(key=lambda game: _utc(game.ended_at), reverse=True)
    return tuple(games)


def _build_selection_coverage(runtime, cutoff, pre):
    """One pass over the accepted proof cohort; never scan rosters per catalog row."""
    games = _analysis_history_before_cutoff(runtime, cutoff)
    team_games, player_games, suggestions = {}, {}, {}
    for game in games:
        for team_id, roster in (
            (game.blue_team_id, game.blue_roster), (game.red_team_id, game.red_roster),
        ):
            team_games.setdefault(team_id, set()).add(game.game_id)
            if team_id not in suggestions:
                suggestions[team_id] = RosterSuggestion(
                    team_id, roster, game.game_id, game.ended_at, cutoff,
                )
            for slot in roster:
                player_games.setdefault(slot.player_id, set()).add(game.game_id)
    return SelectionCoverage(
        cutoff, runtime.snapshot_sha256, runtime.evidence_sha256, runtime.history_pool_sha256,
        tuple(game.game_id for game in games),
        MappingProxyType({row.identity: len(team_games.get(row.identity, ()))
                          for row in runtime.catalog.teams}),
        MappingProxyType({row.identity: len(player_games.get(row.identity, ()))
                          for row in runtime.catalog.players}),
        MappingProxyType(suggestions), runtime, pre,
    )


def _selection_matches_runtime(coverage, runtime):
    return (
        isinstance(coverage, SelectionCoverage)
        and coverage._runtime is runtime
        and coverage.snapshot_sha256 == runtime.snapshot_sha256
        and coverage.evidence_sha256 == runtime.evidence_sha256
        and coverage.history_pool_sha256 == runtime.history_pool_sha256
    )


def selection_coverage(runtime, *, pre=None, previous=None):
    """Reuse a session index only for this retained runtime and the exact cutoff.

    A saved PRE freezes the preview to its own cutoff without reading the clock.
    Otherwise the application clock determines a preview cutoff; this preview is
    never a substitute for the independent clock capture when creating PRE.
    """
    if not isinstance(runtime, AnalysisRuntime):
        fail("E_EVAL_INCOMPATIBLE")
    reusable = _selection_matches_runtime(previous, runtime)
    if pre is not None:
        if reusable and previous._pre is pre:
            return previous
        cutoff = _require_analysis_pre(runtime, pre).history_cutoff_at
    else:
        preview_time = _utc(_utc_now())
        if preview_time < _utc(runtime.runtime_ready_at):
            fail("E_TIME_RETRIEVAL_CONFLICT")
        cutoff = interactive_cutoff(preview_time)
    if reusable and previous.history_cutoff_at == cutoff:
        return previous if previous._pre is pre else replace(previous, _pre=pre)
    if pre is None:
        _require_analysis_runtime(runtime)
    return _build_selection_coverage(runtime, cutoff, pre)


def selection_options(runtime, coverage, *, entity_kind, linked_only=True):
    """Return stable IDs without changing the catalog or any current selection."""
    if not _selection_matches_runtime(coverage, runtime):
        fail("E_EVAL_INCOMPATIBLE")
    if entity_kind == "team":
        entities, counts = runtime.catalog.teams, coverage.team_counts
    elif entity_kind == "player":
        entities, counts = runtime.catalog.players, coverage.player_counts
    else:
        fail("E_PRE_INPUT_INCOMPLETE")
    return tuple(row.identity for row in entities
                 if not linked_only or counts[row.identity] > 0)


def suggest_analysis_roster(runtime, team_id, *, coverage=None):
    """One historical roster for user review; this is not verified current membership."""
    if coverage is None:
        coverage = selection_coverage(runtime)
    if not _selection_matches_runtime(coverage, runtime):
        fail("E_EVAL_INCOMPATIBLE")
    if team_id not in coverage.team_counts:
        fail("E_PRE_INPUT_INCOMPLETE")
    return coverage.roster_suggestions.get(team_id)


def refresh_analysis_state(state, selection, runtime, *, champion_ids=None):
    """Invalidate before render; a rerun never captures a new analysis time."""
    if not isinstance(state, EvaluationState) or state.pre is None:
        return EvaluationState()
    try:
        context = _require_analysis_pre(runtime, state.pre)
        target = _analysis_target(runtime, selection, context.history_cutoff_at)
        if (
            target != state.pre.request.target
            or selection.context_label != context.context_label
            or selection.planned_start_at != context.planned_start_at
        ):
            return EvaluationState()
    except (PreFeatureInputError, TypeError):
        return EvaluationState()
    if champion_ids is not None and tuple(champion_ids) != state.champion_ids:
        return replace(state, champion_ids=tuple(champion_ids), post=None)
    return state


DEMO_NOTICE = (
    "Dữ liệu tổng hợp — minh họa chức năng, chưa đánh giá "
    "chất lượng dự đoán trên esports thật."
)

DEMO_POLICY = "synthetic-7k-explicit-cutoff-v1"
BASE = datetime(2025, 7, 1, 12, tzinfo=UTC)
TEAM_NAMES = (
    ("A", "Sao Bắc"),
    ("B", "Sóng Biển"),
    ("C", "Mặt Trời"),
    ("D", "Mây Trắng"),
)


@dataclass(frozen=True, slots=True)
class DemoChampion:
    champion_id: str
    name: str
    image_file: str


@dataclass(frozen=True, slots=True)
class DemoCase:
    game_id: str
    title: str
    request: EvaluationRequest


# Explicit ID/name mapping for this small demo catalog.
# Images refer only to existing local assets; no runtime download is performed.
CHAMPIONS = (
    DemoChampion("266", "Aatrox", "assets/champions/266-aatrox.png"),
    DemoChampion("2", "Olaf", "assets/champions/2-olaf.png"),
    DemoChampion("103", "Ahri", "assets/champions/103-ahri.png"),
    DemoChampion("22", "Ashe", "assets/champions/22-ashe.png"),
    DemoChampion("12", "Alistar", "assets/champions/12-alistar.png"),
    DemoChampion("86", "Garen", "assets/champions/86-garen.png"),
    DemoChampion("5", "Xin Zhao", "assets/champions/5-xinzhao.png"),
    DemoChampion("1", "Annie", "assets/champions/1-annie.png"),
    DemoChampion("4", "Twisted Fate", "assets/champions/4-twistedfate.png"),
    DemoChampion("3", "Galio", "assets/champions/3-galio.png"),
    DemoChampion("99", "Lux", "assets/champions/99-lux.png"),
    DemoChampion("13", "Ryze", "assets/champions/13-ryze.png"),
)

CHAMPION_REFERENCE = frozenset(champion.champion_id for champion in CHAMPIONS)


def _roster(team):
    return tuple(
        PlayerSlot(role, f"{team.lower()}{number}")
        for number, role in enumerate(ROLES, start=1)
    )


def _historical_slots(blue, red):
    team_champions = {
        "A": tuple(champion.champion_id for champion in CHAMPIONS[:5]),
        "B": tuple(champion.champion_id for champion in CHAMPIONS[5:10]),
    }
    return tuple(
        ChampionSlot(
            team_id=team,
            side=side,
            role=player.role,
            player_id=player.player_id,
            champion_id=champion_id,
        )
        for side, team in (("BLUE", blue), ("RED", red))
        for player, champion_id in zip(_roster(team), team_champions[team], strict=True)
    )


def _history():
    return tuple(
        HistoricalChampionGame(
            HistoricalGame(
                game_id=game_id,
                blue_team_id=blue,
                red_team_id=red,
                blue_roster=_roster(blue),
                red_roster=_roster(red),
                winner_team_id=winner,
                ended_at=end,
            ),
            _historical_slots(blue, red),
        )
        for game_id, blue, red, winner, end in (
            ("synthetic-h1", "A", "B", "A", datetime(2025, 6, 28, 12, tzinfo=UTC)),
            ("synthetic-h2", "B", "A", "B", datetime(2025, 6, 29, 12, tzinfo=UTC)),
        )
    )


def _cutoff_record(game_id, cutoff):
    # This status is an explicit assertion about a synthetic fixture only.
    # It is never assigned by the service and never certifies production data.
    return CutoffRecord(
        game_id=game_id,
        history_cutoff_at=cutoff,
        evidence_ref=f"synthetic://7k/explicit-fixture/{game_id}",
        policy_version=DEMO_POLICY,
        verification=CutoffVerification.VERIFIED_EXTERNALLY,
    )


def make_demo_cases():
    history = _history()
    cases = []
    for game_id, title, blue, red, cutoff in (
        (
            "demo-1",
            "Ván 1 — Sao Bắc gặp Sóng Biển",
            "A",
            "B",
            datetime(2025, 8, 1, 12, tzinfo=UTC),
        ),
        (
            "demo-2",
            "Ván 2 — Mặt Trời gặp Mây Trắng, chưa có lịch sử đội",
            "C",
            "D",
            datetime(2025, 8, 2, 12, tzinfo=UTC),
        ),
    ):
        target = TargetGame(
            game_id=game_id,
            blue_team_id=blue,
            red_team_id=red,
            blue_roster=_roster(blue),
            red_roster=_roster(red),
            patch="15.01",
            history_cutoff_at=cutoff,
        )
        cases.append(
            DemoCase(
                game_id,
                title,
                EvaluationRequest(
                    target=target,
                    cutoff=_cutoff_record(game_id, cutoff),
                    history=history,
                    champion_reference=CHAMPION_REFERENCE,
                ),
            )
        )
    return tuple(cases)


def build_demo_model():
    """Explicit initialization only; return a newly fitted, session-owned bundle."""
    targets = []
    cutoffs = []
    for number in range(1, 21):
        game_id = f"synthetic-train-{number:02d}"
        cutoff = BASE + timedelta(days=number - 1)
        targets.append(
            DatasetTarget(
                game_id=game_id,
                blue_team_id="A",
                red_team_id="B",
                blue_roster=_roster("A"),
                red_roster=_roster("B"),
                patch="15.01",
                final_lineup=FinalLineup(game_id, "15.01", _historical_slots("A", "B")),
                winner_team_id="A" if number <= 9 or number in (15, 18, 20) else "B",
                ended_at=cutoff + timedelta(hours=1),
            )
        )
        cutoffs.append(_cutoff_record(game_id, cutoff))

    dataset = build_paired_dataset(
        targets=tuple(targets),
        history=_history(),
        cutoffs=tuple(cutoffs),
        champion_reference=CHAMPION_REFERENCE,
        approved_policy_versions=frozenset({DEMO_POLICY}),
    )
    split = temporal_split(dataset)
    return fit_model_bundle(
        dataset,
        split,
        ModelIdentity(
            dataset_id="synthetic-7k-dataset-v1",
            split_id="synthetic-7k-temporal-split-v1",
            bundle_version="synthetic-7k-demo-model-v1",
        ),
        ModelConfig(forest_trees=8),
    )
