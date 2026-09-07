"""Synthetic service/state tests; no database, network or production data."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from math import isfinite, nextafter

import pandas as pd
import pytest

import match_insight.ml.models as models
import match_insight.ml.real_training as runner
import match_insight.services.evaluation as evaluation
from match_insight.features.post import FinalLineup
from match_insight.features.pre import PreFeatureInputError
from match_insight.ml.dataset import (
    RETROSPECTIVE_DATASET_VERSION,
    RETROSPECTIVE_PROTOCOL,
    STRICT_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetTarget,
    build_paired_dataset,
    temporal_split,
)
from match_insight.services.demo import CHAMPIONS, build_demo_model, make_demo_cases
from match_insight.services.evaluation import (
    BusinessWarning,
    EvaluationInputError,
    EvaluationState,
    PreSnapshot,
    compare_retrospective_evaluations,
    create_post,
    create_pre,
    evaluate_retrospective_post,
    evaluate_retrospective_pre,
    lineup_from_ids,
    percentage_points,
    refresh_state,
)

CHAMPION_IDS = tuple(champion.champion_id for champion in CHAMPIONS[:10])


@pytest.fixture(scope="module")
def bundle():
    return build_demo_model()


@pytest.fixture
def eval_request():
    return make_demo_cases()[0].request


def test_pre_needs_no_final_champions_and_full_flow_uses_real_core(monkeypatch, eval_request, bundle):
    calls = []
    original = evaluation.predict_phase

    def record_prediction(model, phase, features, metadata):
        calls.append(phase)
        if phase == "PRE":
            assert not any(column.endswith("champion_id") for column in features.columns)
            assert "target_ended_at" not in metadata.columns
            assert "winner_team_id" not in metadata.columns
        return original(model, phase, features, metadata)

    monkeypatch.setattr(evaluation, "predict_phase", record_prediction)
    pre = create_pre(eval_request, bundle)
    before = deepcopy(pre)
    assert calls == ["PRE"]
    assert pre.features.blue.recent_form.value == 0.5
    assert pre.features.blue.recent_form.sample_count == 2

    post = create_post(pre, lineup_from_ids(pre, CHAMPION_IDS), bundle)
    assert calls == ["PRE", "POST"]
    assert pre == before
    assert post.pre is pre
    assert post.features.pre is pre.features
    assert post.comparison.p_blue_win_pre == pre.prediction.p_blue_win
    assert post.comparison.p_blue_win_post == post.prediction.p_blue_win
    assert post.prediction.history_cutoff_at == pre.prediction.history_cutoff_at
    assert post.prediction.context_signature == pre.prediction.context_signature
    assert post.prediction.pre_feature_signature == pre.prediction.pre_feature_signature

    for feature in post.features.player_champion:
        assert feature.games_count == 2
        assert feature.wins_count == 1
        assert feature.win_rate == 0.5
        assert not feature.missing

    assert post.comparison.delta_probability == pytest.approx(
        post.prediction.p_blue_win - pre.prediction.p_blue_win
    )
    assert post.delta_percentage_points == pytest.approx(
        post.comparison.delta_probability * 100
    )


def test_post_requires_saved_pre(eval_request, bundle):
    pre = create_pre(eval_request, bundle)
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    with pytest.raises(EvaluationInputError, match="E_PRE_SNAPSHOT_REQUIRED"):
        create_post(None, lineup, bundle)


@pytest.mark.parametrize("case", ["missing", "unverified", "different_game", "different_time"])
def test_cutoff_is_never_inferred_or_promoted(case, eval_request, bundle):
    record = eval_request.cutoff
    if case == "missing":
        record = None
    elif case == "unverified":
        record = replace(record, verification=CutoffVerification.UNVERIFIED)
    elif case == "different_game":
        record = replace(record, game_id="other-game")
    else:
        record = replace(
            record,
            history_cutoff_at=record.history_cutoff_at + timedelta(seconds=1),
        )

    changed = replace(eval_request, cutoff=record)
    before = deepcopy(changed)
    with pytest.raises(PreFeatureInputError):
        create_pre(changed, bundle)
    assert changed == before


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate", "wrong_player", "wrong_game"])
def test_invalid_lineup_returns_no_new_prediction(case, monkeypatch, eval_request, bundle):
    pre = create_pre(eval_request, bundle)
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    slots = list(lineup.slots)
    if case == "missing":
        lineup = replace(lineup, slots=lineup.slots[:-1])
    elif case == "wrong_game":
        lineup = replace(lineup, game_id="other-game")
    else:
        value = {
            "unknown": "999999",
            "duplicate": slots[1].champion_id,
            "wrong_player": "different-player",
        }[case]
        field = "player_id" if case == "wrong_player" else "champion_id"
        slots[0] = replace(slots[0], **{field: value})
        lineup = replace(lineup, slots=tuple(slots))

    def unexpected_prediction(*args, **kwargs):
        pytest.fail("Invalid lineup reached prediction")

    monkeypatch.setattr(evaluation, "predict_phase", unexpected_prediction)
    with pytest.raises(PreFeatureInputError):
        create_post(pre, lineup, bundle)


@pytest.mark.parametrize("case", ["context", "cutoff", "history_champions", "prediction", "bundle"])
def test_changed_snapshot_foundation_is_rejected(case, eval_request, bundle):
    pre = create_pre(eval_request, bundle)
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    model = bundle
    changed = pre

    if case == "context":
        changed_request = replace(eval_request, target=replace(eval_request.target, patch="15.02"))
        changed = replace(pre, request=changed_request)
    elif case == "cutoff":
        changed_request = replace(
            eval_request,
            cutoff=replace(eval_request.cutoff, evidence_ref="synthetic://different-source"),
        )
        changed = replace(pre, request=changed_request)
    elif case == "history_champions":
        record = eval_request.history[0]
        slots = list(record.slots)
        slots[0] = replace(slots[0], champion_id="99")
        changed_request = replace(
            eval_request,
            history=(replace(record, slots=tuple(slots)),) + eval_request.history[1:],
        )
        changed = replace(pre, request=changed_request)
    elif case == "prediction":
        changed = replace(
            pre,
            prediction=replace(pre.prediction, p_blue_win=0.123),
        )
    else:
        model = replace(
            bundle,
            identity=replace(bundle.identity, bundle_version="different-bundle"),
        )

    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_post(changed, lineup, model)


def test_empty_history_preserves_business_missing_values(eval_request, bundle):
    pre = create_pre(replace(eval_request, history=()), bundle)
    post = create_post(pre, lineup_from_ids(pre, CHAMPION_IDS), bundle)

    assert pre.features.accepted_game_ids == ()
    assert pre.features.blue.recent_form.value is None
    assert pre.features.blue.recent_form.sample_count == 0
    assert pre.features.blue.recent_form.missing
    assert "W_H2H_MISSING" in {warning.code for warning in pre.warnings}

    for feature in post.features.player_champion:
        assert feature.games_count == feature.wins_count == 0
        assert feature.win_rate is None
        assert feature.missing

    pair_warning = next(
        warning for warning in post.warnings if warning.code == "W_PAIR_HISTORY_MISSING"
    )
    assert pair_warning.sample_count == 10


@pytest.mark.parametrize("case", ["target", "roster", "patch", "cutoff", "bundle"])
def test_context_changes_invalidate_both_snapshots(case, eval_request, bundle):
    pre = create_pre(eval_request, bundle)
    post = create_post(pre, lineup_from_ids(pre, CHAMPION_IDS), bundle)
    state = refresh_state(EvaluationState(), eval_request, bundle, CHAMPION_IDS)
    state = replace(state, pre=pre, post=post)

    changed_request = eval_request
    model = bundle

    if case == "target":
        changed_request = make_demo_cases()[1].request
    elif case == "roster":
        roster = list(eval_request.target.blue_roster)
        roster[0] = replace(roster[0], player_id="a-new")
        changed_request = replace(
            eval_request,
            target=replace(eval_request.target, blue_roster=tuple(roster)),
        )
    elif case == "patch":
        changed_request = replace(eval_request, target=replace(eval_request.target, patch="15.02"))
    elif case == "cutoff":
        cutoff = eval_request.target.history_cutoff_at + timedelta(minutes=1)
        changed_request = replace(
            eval_request,
            target=replace(eval_request.target, history_cutoff_at=cutoff),
            cutoff=replace(eval_request.cutoff, history_cutoff_at=cutoff),
        )
    else:
        model = replace(
            bundle,
            identity=replace(bundle.identity, bundle_version="another-demo-model"),
        )

    changed = refresh_state(state, changed_request, model, CHAMPION_IDS)
    assert changed.pre is None
    assert changed.post is None
    assert state.pre is pre
    assert state.post is post


def test_champion_change_preserves_pre_and_invalidates_only_post(eval_request, bundle):
    pre = create_pre(eval_request, bundle)
    post = create_post(pre, lineup_from_ids(pre, CHAMPION_IDS), bundle)
    state = refresh_state(EvaluationState(), eval_request, bundle, CHAMPION_IDS)
    state = replace(state, pre=pre, post=post)

    assert refresh_state(state, eval_request, bundle, CHAMPION_IDS) is state
    changed = refresh_state(state, eval_request, bundle, ("99",) + CHAMPION_IDS[1:])
    assert changed.pre is pre
    assert changed.post is None
    assert state.post is post


def test_percentage_point_conversion():
    assert percentage_points(0.60 - 0.55) == pytest.approx(5.0)
    assert percentage_points(0.55 - 0.60) == pytest.approx(-5.0)
    assert percentage_points(0.0) == 0.0


def test_target_game_in_history_edge_cases(eval_request, bundle):
    target = eval_request.target
    naive_game = replace(
        eval_request.history[0].game,
        game_id=target.game_id,
        ended_at=datetime(2025, 7, 1, 12, 0),  # naive datetime (tzinfo=None)
    )
    naive_record = replace(eval_request.history[0], game=naive_game)
    req_with_target = replace(
        eval_request,
        history=eval_request.history + (naive_record,),
    )

    # 1. request_key, PRE and POST succeed despite naive ended_at on target game in history
    key_with_target = evaluation.request_key(req_with_target, bundle)
    assert isinstance(key_with_target, str) and len(key_with_target) == 64

    pre_with_target = create_pre(req_with_target, bundle)
    assert pre_with_target.context_key == key_with_target
    assert any(
        exclusion.game_id == target.game_id and exclusion.reason == "TARGET_GAME"
        for exclusion in pre_with_target.features.exclusions
    )

    lineup = lineup_from_ids(pre_with_target, CHAMPION_IDS)
    post_with_target = create_post(pre_with_target, lineup, bundle)
    assert post_with_target.pre is pre_with_target

    # 2. Mutating ended_at / winner of target record in history does not change fingerprint or features
    different_target_game = replace(
        naive_game,
        winner_team_id=target.red_team_id,
        ended_at=datetime(2025, 7, 2, 18, 30),
    )
    different_target_record = replace(
        eval_request.history[0],
        game=different_target_game,
    )
    req_different_target = replace(
        eval_request,
        history=eval_request.history + (different_target_record,),
    )

    key_different_target = evaluation.request_key(req_different_target, bundle)
    assert key_different_target == key_with_target

    pre_different_target = create_pre(req_different_target, bundle)
    assert pre_different_target.features.blue == pre_with_target.features.blue
    assert pre_different_target.features.red == pre_with_target.features.red
    assert pre_different_target.features.h2h_blue == pre_with_target.features.h2h_blue
    assert pre_different_target.prediction.p_blue_win == pre_with_target.prediction.p_blue_win

    # 3. Compared to history without target record: accepted IDs, feature values and probabilities match
    pre_without = create_pre(eval_request, bundle)
    assert pre_with_target.features.accepted_game_ids == pre_without.features.accepted_game_ids
    assert pre_with_target.features.blue == pre_without.features.blue
    assert pre_with_target.features.red == pre_without.features.red
    assert pre_with_target.features.h2h_blue == pre_without.features.h2h_blue
    assert pre_with_target.prediction.p_blue_win == pre_without.prediction.p_blue_win
    # Do NOT assert entire snapshot equality; input count and exclusions reflect the extra record
    assert pre_with_target.features.counts.input_games == pre_without.features.counts.input_games + 1
    assert pre_with_target.features.counts.excluded_games == pre_without.features.counts.excluded_games + 1

    # 4. Non-target record in history with naive ended_at raises E_TIMESTAMP_INVALID
    non_target_naive_game = replace(
        eval_request.history[0].game,
        game_id="non-target-naive-game",
        ended_at=datetime(2025, 7, 1, 12, 0),
    )
    non_target_record = replace(eval_request.history[0], game=non_target_naive_game)
    req_non_target = replace(
        eval_request,
        history=eval_request.history + (non_target_record,),
    )
    with pytest.raises(PreFeatureInputError) as exc_non_target:
        create_pre(req_non_target, bundle)
    assert exc_non_target.value.code == "E_TIMESTAMP_INVALID"

    # 5. Duplicate target records in history raise E_HISTORY_DUPLICATE_GAME_ID
    req_duplicate = replace(
        eval_request,
        history=eval_request.history + (naive_record, different_target_record),
    )
    with pytest.raises(PreFeatureInputError) as exc_duplicate:
        create_pre(req_duplicate, bundle)
    assert exc_duplicate.value.code == "E_HISTORY_DUPLICATE_GAME_ID"


@pytest.fixture(scope="module")
def retrospective_bundle():
    """Fit a small synthetic protocol fixture with its final identity, not a relabelled model."""
    request = make_demo_cases()[0].request
    targets = []
    cutoffs = []
    for number in range(20):
        game_id = f"synthetic-8c-{number:02d}"
        cutoff = datetime(2025, 7, 1, tzinfo=UTC) + timedelta(days=3 * number)
        targets.append(
            DatasetTarget(
                game_id=game_id,
                blue_team_id=request.target.blue_team_id,
                red_team_id=request.target.red_team_id,
                blue_roster=request.target.blue_roster,
                red_roster=request.target.red_roster,
                patch=request.target.patch,
                final_lineup=FinalLineup(
                    game_id, request.target.patch, request.history[0].slots
                ),
                winner_team_id="A" if number % 2 else "B",
                ended_at=cutoff + timedelta(days=1, hours=12),
            )
        )
        cutoffs.append(
            CutoffRecord(
                game_id,
                cutoff,
                f"synthetic://8c/protocol-fixture/{game_id}",
                RETROSPECTIVE_PROTOCOL,
                CutoffVerification.PROTOCOL_ASSUMED,
            )
        )
    dataset = build_paired_dataset(
        tuple(targets),
        request.history,
        tuple(cutoffs),
        request.champion_reference,
        approved_policy_versions=frozenset({RETROSPECTIVE_PROTOCOL}),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    split = temporal_split(dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    return models.fit_model_bundle(
        dataset,
        split,
        models.ModelIdentity(
            "retrospective:postgresql:synthetic-8c",
            "retrospective:split:synthetic-8c",
            "retrospective:7l:synthetic-8c",
        ),
        models.ModelConfig(forest_trees=8),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )


@pytest.fixture
def retrospective_request(eval_request):
    cutoff = datetime(2025, 8, 1, tzinfo=UTC)
    return replace(
        eval_request,
        target=replace(eval_request.target, history_cutoff_at=cutoff),
        cutoff=replace(
            eval_request.cutoff,
            history_cutoff_at=cutoff,
            evidence_ref="synthetic://8c/assumed-final-context",
            policy_version=RETROSPECTIVE_PROTOCOL,
            verification=CutoffVerification.PROTOCOL_ASSUMED,
        ),
    )


def _install_canonical_loader_spy(monkeypatch, retrospective_bundle):
    calls = []
    canonical = runner.CANONICAL_RETROSPECTIVE_EXPECTATION

    def load(path, *, expected):
        assert expected is canonical
        calls.append(path)
        return retrospective_bundle

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", load)
    return calls


def _forbid_runtime_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("PRE inference must not fit, fit_transform or save")

    for owner in (models, runner):
        monkeypatch.setattr(owner, "fit_model_bundle", forbidden)
        monkeypatch.setattr(owner, "save_bundle", forbidden)
    monkeypatch.setattr(models.joblib, "dump", forbidden)
    for cls in (
        models.Pipeline,
        models.ColumnTransformer,
        models.DummyClassifier,
        models.LogisticRegression,
        models.RandomForestClassifier,
        models.SimpleImputer,
        models.StandardScaler,
        models.OneHotEncoder,
    ):
        for method in ("fit", "fit_transform"):
            if hasattr(cls, method):
                monkeypatch.setattr(cls, method, forbidden)


def test_retrospective_pre_boundary_preserves_contract_without_champions(
    monkeypatch, retrospective_request, retrospective_bundle
):
    calls = _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    _forbid_runtime_training(monkeypatch)
    before = deepcopy(retrospective_request)
    original_predict = evaluation.predict_phase
    contexts = []

    def inspect_prediction(bundle, phase, frame, context):
        assert phase == "PRE"
        assert tuple(frame.columns) == models.PRE_COLUMNS
        assert not any(column.endswith("champion_id") for column in frame.columns)
        assert "winner_team_id" not in context.columns
        assert "target_ended_at" not in context.columns
        contexts.append(context.copy(deep=True))
        return original_predict(bundle, phase, frame, context)

    monkeypatch.setattr(evaluation, "predict_phase", inspect_prediction)
    started = datetime(2025, 8, 2, 12, tzinfo=UTC)
    first = evaluate_retrospective_pre(retrospective_request, target_started_at=started)
    second = evaluate_retrospective_pre(retrospective_request, target_started_at=started)

    assert isinstance(first, PreSnapshot)
    assert isinstance(first.prediction, models.PhasePrediction)
    assert first == second
    assert len(calls) == 2
    assert all(
        path == runner.MODEL_DIRECTORY / runner.RETROSPECTIVE_MODEL_FILENAME
        for path in calls
    )
    assert retrospective_request == before
    assert first.request.cutoff.verification is CutoffVerification.PROTOCOL_ASSUMED
    assert first.features.metadata.cutoff_verification == "NOT_ASSESSED"
    assert first.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert first.prediction.model_bundle_version == retrospective_bundle.identity.bundle_version
    assert first.prediction.history_cutoff_at == retrospective_request.cutoff.history_cutoff_at
    assert first.features.blue.recent_form.sample_count == 2
    assert first.features.blue.recent_form.value == 0.5
    assert first.features.accepted_game_ids == ("synthetic-h2", "synthetic-h1")
    probability = first.prediction.p_blue_win
    assert isfinite(probability) and 0 <= probability <= 1
    assert probability + (1 - probability) == pytest.approx(1)
    for context in contexts:
        row = context.loc[first.prediction.game_id]
        assert row["dataset_version"] == RETROSPECTIVE_DATASET_VERSION
        assert row["cutoff_verification"] == "PROTOCOL_ASSUMED"
        assert row["core_cutoff_verification"] == "NOT_ASSESSED"
        assert row["evidence_ref"] == before.cutoff.evidence_ref


@pytest.mark.parametrize(
    ("start", "cutoff"),
    [
        (datetime(2025, 8, 2, 23, 59, tzinfo=UTC), datetime(2025, 8, 1, tzinfo=UTC)),
        (datetime(2025, 8, 3, 6, 59, tzinfo=timezone(timedelta(hours=7))),
         datetime(2025, 8, 1, tzinfo=UTC)),
        (datetime(2026, 1, 1, tzinfo=UTC), datetime(2025, 12, 31, tzinfo=UTC)),
        (datetime(2025, 12, 31, 19, tzinfo=timezone(timedelta(hours=-5))),
         datetime(2025, 12, 31, tzinfo=UTC)),
    ],
)
def test_retrospective_pre_cutoff_uses_utc_day_and_year_boundary(
    monkeypatch, retrospective_request, retrospective_bundle, start, cutoff
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    offset = timezone(timedelta(hours=7))
    request = replace(
        retrospective_request,
        target=replace(retrospective_request.target, history_cutoff_at=cutoff.astimezone(offset)),
        cutoff=replace(retrospective_request.cutoff, history_cutoff_at=cutoff),
    )
    snapshot = evaluate_retrospective_pre(request, target_started_at=start)
    equivalent = evaluate_retrospective_pre(
        request, target_started_at=start.astimezone(UTC)
    )
    assert snapshot == equivalent
    assert snapshot.prediction.history_cutoff_at == cutoff
    assert snapshot.prediction.history_cutoff_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("missing", "E_CUTOFF_MISSING"),
        ("wrong_record_type", "E_CUTOFF_STATUS_INVALID"),
        ("missing_time", "E_CUTOFF_MISSING"),
        ("naive_cutoff", "E_CUTOFF_INVALID"),
        ("unverified", "E_CUTOFF_UNVERIFIED"),
        ("verified_externally", "E_CUTOFF_UNVERIFIED"),
        ("wrong_policy", "E_CUTOFF_POLICY_UNAPPROVED"),
        ("empty_evidence", "E_CUTOFF_EVIDENCE_INVALID"),
        ("wrong_game", "E_EVAL_INCOMPATIBLE"),
        ("formula", "E_EVAL_INCOMPATIBLE"),
        ("target_cutoff", "E_EVAL_INCOMPATIBLE"),
    ],
)
def test_retrospective_pre_invalid_cutoff_fails_before_loading(
    monkeypatch, retrospective_request, case, code
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid cutoff reached model loading or feature construction")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", unexpected)
    monkeypatch.setattr(evaluation, "build_pre_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    request = retrospective_request
    record = request.cutoff
    if case == "missing":
        record = None
    elif case == "wrong_record_type":
        record = {}
    elif case == "missing_time":
        record = replace(record, history_cutoff_at=None)
    elif case == "naive_cutoff":
        record = replace(record, history_cutoff_at=record.history_cutoff_at.replace(tzinfo=None))
    elif case in {"unverified", "verified_externally"}:
        status = (
            CutoffVerification.UNVERIFIED
            if case == "unverified"
            else CutoffVerification.VERIFIED_EXTERNALLY
        )
        record = replace(record, verification=status)
    elif case == "wrong_policy":
        record = replace(record, policy_version="synthetic-unapproved-policy")
    elif case == "empty_evidence":
        record = replace(record, evidence_ref="")
    elif case == "wrong_game":
        record = replace(record, game_id="different-target")
    elif case == "formula":
        cutoff = record.history_cutoff_at + timedelta(microseconds=1)
        record = replace(record, history_cutoff_at=cutoff)
        request = replace(request, target=replace(request.target, history_cutoff_at=cutoff))
    else:
        request = replace(
            request,
            target=replace(
                request.target,
                history_cutoff_at=record.history_cutoff_at + timedelta(seconds=1),
            ),
        )
    request = replace(request, cutoff=record)
    before = deepcopy(request)
    with pytest.raises(PreFeatureInputError, match=code):
        evaluate_retrospective_pre(
            request, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )
    assert request == before


@pytest.mark.parametrize(
    "started", [None, datetime(2025, 8, 2, 12), "2025-08-02T12:00:00Z", pd.NaT]
)
def test_retrospective_pre_invalid_start_is_rejected_before_loading(
    monkeypatch, retrospective_request, started
):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid start reached model loading")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", unexpected)
    with pytest.raises(PreFeatureInputError, match="E_MODEL_TIMESTAMP_INVALID"):
        evaluate_retrospective_pre(retrospective_request, target_started_at=started)


@pytest.mark.parametrize("case", ["request_type", "target_type", "duplicate_team", "short_roster"])
def test_retrospective_pre_invalid_context_never_reaches_features(
    monkeypatch, retrospective_request, retrospective_bundle, case
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    request = retrospective_request
    if case == "request_type":
        request = None
    elif case == "target_type":
        request = replace(request, target={})
    elif case == "duplicate_team":
        request = replace(
            request, target=replace(request.target, red_team_id=request.target.blue_team_id)
        )
    else:
        request = replace(
            request, target=replace(request.target, blue_roster=request.target.blue_roster[:-1])
        )

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid context reached feature construction or prediction")

    monkeypatch.setattr(evaluation, "build_pre_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    with pytest.raises(PreFeatureInputError):
        evaluate_retrospective_pre(
            request, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )


def test_pre_service_keeps_explicit_protocol_isolation(
    monkeypatch, eval_request, bundle, retrospective_request, retrospective_bundle
):
    strict = create_pre(eval_request, bundle)
    explicit_strict = create_pre(eval_request, bundle, evaluation_protocol=STRICT_PROTOCOL)
    assert strict == explicit_strict
    retrospective = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    assert retrospective.prediction.policy_version == RETROSPECTIVE_PROTOCOL

    def unexpected(*args, **kwargs):
        pytest.fail("Incompatible protocol reached feature construction")

    monkeypatch.setattr(evaluation, "build_pre_features", unexpected)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_pre(retrospective_request, retrospective_bundle)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_pre(eval_request, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    with pytest.raises(PreFeatureInputError):
        create_pre(eval_request, bundle, evaluation_protocol="other-protocol")
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_post(
            retrospective,
            lineup_from_ids(retrospective, CHAMPION_IDS),
            retrospective_bundle,
        )
    with pytest.raises(TypeError, match="evaluation_protocol"):
        evaluate_retrospective_pre(
            retrospective_request,
            target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
            evaluation_protocol=STRICT_PROTOCOL,
        )


def test_retrospective_pre_loader_failure_is_preserved_without_fallback(
    monkeypatch, retrospective_request
):
    failure = PreFeatureInputError("E_REAL_MODEL_METADATA_MISSING", "synthetic missing pair")

    def fail_loader(path, *, expected):
        assert expected is runner.CANONICAL_RETROSPECTIVE_EXPECTATION
        raise failure

    def unexpected(*args, **kwargs):
        pytest.fail("Failed canonical loading reached feature construction or prediction")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", fail_loader)
    monkeypatch.setattr(evaluation, "build_pre_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    with pytest.raises(PreFeatureInputError) as caught:
        evaluate_retrospective_pre(
            retrospective_request, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )
    assert caught.value is failure


def test_retrospective_pre_history_filter_still_excludes_target_missing_and_future(
    monkeypatch, retrospective_request, retrospective_bundle
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    start = datetime(2025, 8, 2, 12, tzinfo=UTC)
    original = evaluate_retrospective_pre(retrospective_request, target_started_at=start)
    source = retrospective_request.history[0]
    cutoff = retrospective_request.cutoff.history_cutoff_at
    added = tuple(
        replace(
            source,
            game=replace(source.game, game_id=game_id, ended_at=end, winner_team_id="B"),
        )
        for game_id, end in (
            (retrospective_request.target.game_id, datetime(2025, 1, 1)),
            ("synthetic-equal", cutoff),
            ("synthetic-future", cutoff + timedelta(seconds=1)),
            ("synthetic-missing", None),
        )
    )
    request = replace(retrospective_request, history=retrospective_request.history + added)
    before = deepcopy(request)
    extended = evaluate_retrospective_pre(request, target_started_at=start)
    reversed_snapshot = evaluate_retrospective_pre(
        replace(request, history=tuple(reversed(request.history))), target_started_at=start
    )
    assert request == before
    for snapshot in (extended, reversed_snapshot):
        assert snapshot.features.blue == original.features.blue
        assert snapshot.features.red == original.features.red
        assert snapshot.features.h2h_blue == original.features.h2h_blue
        assert snapshot.features.accepted_game_ids == original.features.accepted_game_ids
        assert snapshot.prediction.p_blue_win == original.prediction.p_blue_win
        assert snapshot.features.counts.input_games == 6
        assert snapshot.features.counts.accepted_games == 2
        assert snapshot.features.counts.excluded_games == 4
        assert {item.game_id: item.reason for item in snapshot.features.exclusions} == {
            request.target.game_id: "TARGET_GAME",
            "synthetic-equal": "END_NOT_BEFORE_CUTOFF",
            "synthetic-future": "END_NOT_BEFORE_CUTOFF",
            "synthetic-missing": "MISSING_ENDED_AT",
        }
    changed_target = replace(
        added[0], game=replace(added[0].game, winner_team_id="A", ended_at=None)
    )
    changed = evaluate_retrospective_pre(
        replace(request, history=retrospective_request.history + (changed_target,) + added[1:]),
        target_started_at=start,
    )
    assert changed.features == extended.features
    assert changed.prediction == extended.prediction
    assert changed.context_key == extended.context_key
    assert changed.seal == extended.seal


def test_retrospective_pre_no_history_keeps_missing_counts_and_warnings(
    monkeypatch, retrospective_request, retrospective_bundle
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    snapshot = evaluate_retrospective_pre(
        replace(retrospective_request, history=()),
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
    )
    assert snapshot.features.accepted_game_ids == ()
    for team in (snapshot.features.blue, snapshot.features.red):
        assert team.recent_form.value is None
        assert team.recent_form.sample_count == 0
        assert team.recent_form.missing
        assert team.side_win_rate.value is None
        assert team.side_win_rate.sample_count == 0
        assert team.side_win_rate.missing
    assert snapshot.features.h2h_blue.value is None
    assert snapshot.features.h2h_blue.sample_count == 0
    assert snapshot.features.h2h_blue.missing
    assert {warning.code for warning in snapshot.warnings} == {
        "W_TEAM_HISTORY_SMALL", "W_H2H_MISSING"
    }
    assert isfinite(snapshot.prediction.p_blue_win)


@pytest.mark.parametrize("excluded_history", [False, True], ids=["base-history", "excluded-history"])
def test_retrospective_pre_real_loader_to_prediction_integration(
    tmp_path, monkeypatch, retrospective_request, retrospective_bundle, excluded_history
):
    """Test-only artifact and expectation; never require or relabel the canonical model."""
    if excluded_history:
        source = retrospective_request.history[0]
        cutoff = retrospective_request.cutoff.history_cutoff_at
        excluded = tuple(
            replace(
                source,
                game=replace(source.game, game_id=game_id, ended_at=end, winner_team_id="B"),
                slots=(),
            )
            for game_id, end in (
                (retrospective_request.target.game_id, datetime(2025, 1, 1)),
                ("synthetic-8f-equal", cutoff),
                ("synthetic-8f-future", cutoff + timedelta(seconds=1)),
                ("synthetic-8f-missing", None),
            )
        )
        retrospective_request = replace(
            retrospective_request, history=retrospective_request.history + excluded
        )
    path = tmp_path / "synthetic-8c.joblib"
    models.save_bundle(retrospective_bundle, path)
    model_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "artifact_schema": "real-model-artifact-v1",
        "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "model_sha256": model_hash,
        "bundle": models._plain(runner._bundle_metadata(retrospective_bundle)),
        "run": {
            "status": "TRAINED",
            "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
            "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
            "dataset_id": retrospective_bundle.identity.dataset_id,
            "split_id": retrospective_bundle.identity.split_id,
            "model_bundle_version": retrospective_bundle.identity.bundle_version,
            "selected_family": retrospective_bundle.family,
        },
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, allow_nan=False), encoding="utf-8"
    )
    expectation = runner.ModelArtifactExpectation(
        retrospective_bundle.identity, model_hash, retrospective_bundle.family
    )
    monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION", expectation)
    actual_loader = runner.load_retrospective_bundle
    boundary_loads = []

    def observed_loader(model_path, *, expected):
        assert model_path == path
        assert expected is expectation
        boundary_loads.append(model_path)
        return actual_loader(model_path, expected=expected)

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", observed_loader)
    before = {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in tmp_path.iterdir()
    }
    _forbid_runtime_training(monkeypatch)
    runtime_bundle = runner.load_retrospective_bundle(path, expected=expectation)
    request_before = deepcopy(retrospective_request)
    first = evaluate_retrospective_pre(
        retrospective_request,
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
        model_path=path,
    )
    second = evaluate_retrospective_pre(
        retrospective_request,
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
        model_path=path,
    )
    assert first == second
    assert first.features.blue.recent_form.value == 0.5
    assert first.features.blue.recent_form.sample_count == 2
    assert first.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert first.prediction.model_bundle_version == retrospective_bundle.identity.bundle_version
    assert isfinite(first.prediction.p_blue_win) and 0 <= first.prediction.p_blue_win <= 1
    assert retrospective_request == request_before
    assert {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in tmp_path.iterdir()
    } == before

    # Extend the same temporary artifact integration through POST, without rebuilding PRE.
    _forbid_pre_rebuild(monkeypatch)
    pre_before = deepcopy(first)
    lineup = lineup_from_ids(first, CHAMPION_IDS)
    post = evaluate_retrospective_post(
        first,
        lineup,
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
        model_path=path,
    )
    repeated_post = evaluate_retrospective_post(
        first,
        lineup,
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
        model_path=path,
    )
    assert post == repeated_post
    assert post.pre is first
    assert first == pre_before
    assert post.features.pre is first.features
    assert post.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert post.prediction.history_cutoff_at == first.prediction.history_cutoff_at
    assert post.prediction.model_bundle_version == first.prediction.model_bundle_version
    assert isfinite(post.prediction.p_blue_win) and 0 <= post.prediction.p_blue_win <= 1
    assert post.comparison.delta_probability == pytest.approx(
        post.prediction.p_blue_win - first.prediction.p_blue_win
    )
    assert all(
        feature.games_count == 2 and feature.wins_count == 1 and feature.win_rate == 0.5
        for feature in post.features.player_champion
    )
    assert {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in tmp_path.iterdir()
    } == before

    # Complete the vertical slice with genuine saved snapshots and the pinned loaded bundle.
    pair_before = deepcopy((first, post))
    _forbid_comparison_work(monkeypatch)
    result = compare_retrospective_evaluations(first, post, runtime_bundle)
    repeated_result = compare_retrospective_evaluations(first, repeated_post, runtime_bundle)
    assert result == repeated_result
    assert json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False)) == result
    assert set(result) == {
        "game_id", "history_cutoff_at", "data_kind", "evaluation_protocol",
        "pre", "post", "comparison", "model", "warnings",
    }
    assert result["game_id"] == retrospective_request.target.game_id
    assert result["history_cutoff_at"] == retrospective_request.cutoff.history_cutoff_at.isoformat()
    assert result["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert result["evaluation_protocol"] == RETROSPECTIVE_PROTOCOL
    assert result["pre"] == {
        "evaluation_type": "PRE",
        "blue_win_probability": first.prediction.p_blue_win,
        "red_win_probability": 1 - first.prediction.p_blue_win,
    }
    assert result["post"] == {
        "evaluation_type": "POST",
        "blue_win_probability": post.prediction.p_blue_win,
        "red_win_probability": 1 - post.prediction.p_blue_win,
    }
    for phase in ("pre", "post"):
        blue = result[phase]["blue_win_probability"]
        red = result[phase]["red_win_probability"]
        assert isfinite(blue) and 0 <= blue <= 1
        assert isfinite(red) and 0 <= red <= 1
        assert blue + red == pytest.approx(1)
    assert result["comparison"]["blue_probability_delta"] == pytest.approx(
        result["post"]["blue_win_probability"] - result["pre"]["blue_win_probability"]
    )
    assert result["comparison"]["red_probability_delta"] == pytest.approx(
        result["post"]["red_win_probability"] - result["pre"]["red_win_probability"]
    )
    assert result["model"] == {
        "family": runtime_bundle.family,
        "dataset_id": expectation.identity.dataset_id,
        "split_id": expectation.identity.split_id,
        "bundle_version": expectation.identity.bundle_version,
        "feature_schema_version": runtime_bundle.feature_schema_version,
        "preprocessing_version": runtime_bundle.preprocessing_version,
    }
    assert result["warnings"] == [
        {
            "code": "W_TEAM_HISTORY_SMALL",
            "team_id": team,
            "sample_count": 2,
            "requested_count": runtime_bundle.contract.pre_config.recent_form_games,
        }
        for team in ("A", "B")
    ]
    assert first.features.accepted_game_ids == ("synthetic-h2", "synthetic-h1")
    assert post.features.accepted_game_ids == first.features.accepted_game_ids
    assert first.features.counts.accepted_games == 2
    if excluded_history:
        assert first.features.counts.input_games == 6
        assert first.features.counts.excluded_games == 4
        assert first.features.counts.missing_end_games == 1
        assert {item.game_id: item.reason for item in first.features.exclusions} == {
            retrospective_request.target.game_id: "TARGET_GAME",
            "synthetic-8f-equal": "END_NOT_BEFORE_CUTOFF",
            "synthetic-8f-future": "END_NOT_BEFORE_CUTOFF",
            "synthetic-8f-missing": "MISSING_ENDED_AT",
        }
    else:
        assert first.features.counts.input_games == 2
        assert first.features.counts.excluded_games == 0
        assert first.features.counts.missing_end_games == 0
    assert post.features.exclusions == first.features.exclusions
    assert post.features.counts == first.features.counts
    assert post.pre is first
    assert post.features.pre is first.features
    assert (first, post) == pair_before
    assert len(boundary_loads) == 4
    assert first.request.cutoff.verification is CutoffVerification.PROTOCOL_ASSUMED
    assert first.features.metadata.cutoff_verification == "NOT_ASSESSED"
    assert post.features.metadata.cutoff_verification == "NOT_ASSESSED"
    assert first.request.history == request_before.history
    assert retrospective_request == request_before
    assert {
        file.name: (file.read_bytes(), file.stat().st_mtime_ns)
        for file in tmp_path.iterdir()
    } == before


def _forbid_pre_rebuild(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("POST must reuse the saved PRE rather than rebuild it")

    monkeypatch.setattr(evaluation, "create_pre", forbidden)
    monkeypatch.setattr(evaluation, "build_pre_features", forbidden)


def test_retrospective_post_reuses_exact_pre_and_fitted_pipeline(
    monkeypatch, retrospective_request, retrospective_bundle
):
    calls = _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    start = datetime(2025, 8, 2, 12, tzinfo=UTC)
    pre = evaluate_retrospective_pre(retrospective_request, target_started_at=start)
    before = deepcopy(pre)
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    lineup_before = deepcopy(lineup)
    pre_frame, pre_context = evaluation._prediction_inputs(pre.request, pre.features, retrospective_bundle)
    original_predict = evaluation.predict_phase
    original_build = evaluation.build_post_features
    prediction_calls = []
    feature_calls = []

    def inspect_post_features(pre_features, final_lineup, history, reference):
        assert pre_features is pre.features
        assert history is pre.request.history
        assert reference is pre.request.champion_reference
        feature_calls.append(final_lineup)
        return original_build(pre_features, final_lineup, history, reference)

    def inspect_post_prediction(bundle, phase, frame, context):
        assert phase == "POST"
        assert tuple(frame.columns) == models.POST_COLUMNS
        assert len(frame.columns) == 76
        assert tuple(frame.columns[:26]) == models.PRE_COLUMNS
        pd.testing.assert_frame_equal(frame.loc[:, list(models.PRE_COLUMNS)], pre_frame)
        pd.testing.assert_frame_equal(context, pre_context)
        assert "winner_team_id" not in frame.columns
        assert "target_ended_at" not in context.columns
        prediction_calls.append(phase)
        return original_predict(bundle, phase, frame, context)

    _forbid_runtime_training(monkeypatch)
    _forbid_pre_rebuild(monkeypatch)
    monkeypatch.setattr(evaluation, "build_post_features", inspect_post_features)
    monkeypatch.setattr(evaluation, "predict_phase", inspect_post_prediction)
    post = evaluate_retrospective_post(pre, lineup, target_started_at=start)
    repeated = evaluate_retrospective_post(
        pre,
        replace(lineup, slots=tuple(reversed(lineup.slots))),
        target_started_at=start.astimezone(timezone(timedelta(hours=7))),
    )

    assert post == repeated
    assert pre == before
    assert lineup == lineup_before
    assert post.pre is pre
    assert post.features.pre is pre.features
    assert len(calls) == 3
    assert len(feature_calls) == 2
    assert prediction_calls == ["POST", "POST"]
    assert post.prediction.context_signature == pre.prediction.context_signature
    assert post.prediction.pre_feature_signature == pre.prediction.pre_feature_signature
    assert post.prediction.history_cutoff_at == pre.prediction.history_cutoff_at
    assert post.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert post.pre.request.cutoff.verification is CutoffVerification.PROTOCOL_ASSUMED
    assert post.features.metadata.cutoff_verification == "NOT_ASSESSED"
    assert post.prediction.model_bundle_version == pre.prediction.model_bundle_version
    assert post.warnings == pre.warnings
    assert isfinite(post.prediction.p_blue_win) and 0 <= post.prediction.p_blue_win <= 1
    for feature in post.features.player_champion:
        assert feature.games_count == 2
        assert feature.wins_count == 1
        assert feature.win_rate == 0.5
        assert not feature.missing
        assert feature.game_ids == pre.features.accepted_game_ids
    assert post.comparison.p_blue_win_pre == pre.prediction.p_blue_win
    assert post.comparison.delta_probability == pytest.approx(
        post.prediction.p_blue_win - pre.prediction.p_blue_win
    )


@pytest.mark.parametrize("missing_pre", [None, {}])
def test_retrospective_post_requires_saved_pre_before_loading(monkeypatch, missing_pre):
    def unexpected(*args, **kwargs):
        pytest.fail("Missing PRE reached model loading")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", unexpected)
    with pytest.raises(PreFeatureInputError, match="E_PRE_SNAPSHOT_REQUIRED"):
        evaluate_retrospective_post(
            missing_pre, None, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("different_start_day", "E_EVAL_INCOMPATIBLE"),
        ("cutoff_microsecond", "E_EVAL_INCOMPATIBLE"),
        ("naive_start", "E_MODEL_TIMESTAMP_INVALID"),
        ("missing_start", "E_MODEL_TIMESTAMP_INVALID"),
        ("missing_cutoff", "E_CUTOFF_MISSING"),
    ],
)
def test_retrospective_post_cutoff_is_checked_before_loading(
    monkeypatch, retrospective_request, retrospective_bundle, case, code
):
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    start = datetime(2025, 8, 2, 12, tzinfo=UTC)
    if case == "different_start_day":
        start += timedelta(days=1)
    elif case == "cutoff_microsecond":
        cutoff = pre.request.cutoff.history_cutoff_at + timedelta(microseconds=1)
        request = replace(
            pre.request,
            target=replace(pre.request.target, history_cutoff_at=cutoff),
            cutoff=replace(pre.request.cutoff, history_cutoff_at=cutoff),
        )
        pre = replace(pre, request=request)
    elif case == "naive_start":
        start = start.replace(tzinfo=None)
    elif case == "missing_start":
        start = None
    else:
        pre = replace(pre, request=replace(pre.request, cutoff=None))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid retrospective foundation reached model loading or POST building")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", unexpected)
    monkeypatch.setattr(evaluation, "build_post_features", unexpected)
    with pytest.raises(PreFeatureInputError, match=code):
        evaluate_retrospective_post(pre, lineup, target_started_at=start)


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("missing_slot", "E_LINEUP_INCOMPLETE"),
        ("missing_champion", "E_CHAMPION_ID_INVALID"),
        ("unknown_champion", "E_CHAMPION_UNKNOWN"),
        ("duplicate_champion", "E_SOURCE_CONFLICT"),
        ("duplicate_role", "E_SOURCE_CONFLICT"),
        ("duplicate_player", "E_SOURCE_CONFLICT"),
        ("wrong_player", "E_EVAL_INCOMPATIBLE"),
        ("wrong_team", "E_EVAL_INCOMPATIBLE"),
        ("wrong_game", "E_EVAL_INCOMPATIBLE"),
        ("wrong_patch", "E_EVAL_INCOMPATIBLE"),
    ],
)
def test_retrospective_post_bad_lineup_returns_no_prediction(
    monkeypatch, retrospective_request, retrospective_bundle, case, code
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    before = deepcopy(pre)
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    slots = list(lineup.slots)
    if case == "missing_slot":
        lineup = replace(lineup, slots=lineup.slots[:-1])
    elif case == "wrong_game":
        lineup = replace(lineup, game_id="other-target")
    elif case == "wrong_patch":
        lineup = replace(lineup, patch="15.02")
    else:
        field, value = {
            "missing_champion": ("champion_id", None),
            "unknown_champion": ("champion_id", "999999"),
            "duplicate_champion": ("champion_id", slots[1].champion_id),
            "duplicate_role": ("role", slots[1].role),
            "duplicate_player": ("player_id", slots[1].player_id),
            "wrong_player": ("player_id", "different-player"),
            "wrong_team": ("team_id", "another-team"),
        }[case]
        slots[0] = replace(slots[0], **{field: value})
        lineup = replace(lineup, slots=tuple(slots))

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid final lineup reached POST building or prediction")

    monkeypatch.setattr(evaluation, "build_post_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    with pytest.raises(PreFeatureInputError, match=code):
        evaluate_retrospective_post(
            pre, lineup, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )
    assert pre == before


@pytest.mark.parametrize("case", ["context", "history", "prediction", "bundle", "schema"])
def test_retrospective_post_changed_foundation_rejected_before_prediction(
    monkeypatch, retrospective_request, retrospective_bundle, case
):
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    bundle = retrospective_bundle
    if case == "context":
        pre = replace(
            pre,
            request=replace(pre.request, target=replace(pre.request.target, patch="15.02")),
        )
    elif case == "history":
        record = pre.request.history[0]
        record = replace(record, game=replace(record.game, winner_team_id="B"))
        pre = replace(
            pre, request=replace(pre.request, history=(record,) + pre.request.history[1:])
        )
    elif case == "prediction":
        pre = replace(pre, prediction=replace(pre.prediction, p_blue_win=0.123))
    elif case == "bundle":
        bundle = replace(
            bundle,
            identity=replace(bundle.identity, bundle_version="retrospective:7l:different-model"),
        )
    else:
        bundle = replace(bundle, feature_schema_version="unsupported-feature-schema")
    _install_canonical_loader_spy(monkeypatch, bundle)

    def unexpected(*args, **kwargs):
        pytest.fail("An incompatible foundation reached POST building or prediction")

    monkeypatch.setattr(evaluation, "build_post_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        evaluate_retrospective_post(
            pre, lineup, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )


def test_retrospective_post_failed_canonical_load_has_no_fallback(
    monkeypatch, retrospective_request, retrospective_bundle
):
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    failure = PreFeatureInputError("E_REAL_MODEL_HASH_MISMATCH", "synthetic wrong model hash")

    def fail_loader(path, *, expected):
        assert expected is runner.CANONICAL_RETROSPECTIVE_EXPECTATION
        raise failure

    def unexpected(*args, **kwargs):
        pytest.fail("Failed canonical loading reached POST building or prediction")

    monkeypatch.setattr(evaluation, "load_retrospective_bundle", fail_loader)
    monkeypatch.setattr(evaluation, "build_post_features", unexpected)
    monkeypatch.setattr(evaluation, "predict_phase", unexpected)
    with pytest.raises(PreFeatureInputError) as caught:
        evaluate_retrospective_post(
            pre, lineup, target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC)
        )
    assert caught.value is failure


def test_post_service_keeps_strict_default_and_fixed_retrospective_boundary(
    monkeypatch, eval_request, bundle, retrospective_request, retrospective_bundle
):
    strict_pre = create_pre(eval_request, bundle)
    strict_lineup = lineup_from_ids(strict_pre, CHAMPION_IDS)
    assert create_post(strict_pre, strict_lineup, bundle) == create_post(
        strict_pre, strict_lineup, bundle, evaluation_protocol=STRICT_PROTOCOL
    )
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    lineup = lineup_from_ids(pre, CHAMPION_IDS)
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    retrospective = create_post(
        pre, lineup, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    assert retrospective.prediction.policy_version == RETROSPECTIVE_PROTOCOL

    def unexpected(*args, **kwargs):
        pytest.fail("Mixed protocol reached POST feature construction")

    monkeypatch.setattr(evaluation, "build_post_features", unexpected)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_post(pre, lineup, retrospective_bundle)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_post(
            strict_pre, strict_lineup, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
        )
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        create_post(pre, lineup, retrospective_bundle, evaluation_protocol="other-protocol")
    with pytest.raises(TypeError, match="evaluation_protocol"):
        evaluate_retrospective_post(
            pre,
            lineup,
            target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
            evaluation_protocol=STRICT_PROTOCOL,
        )


def test_retrospective_post_uses_only_history_frozen_in_pre(
    monkeypatch, retrospective_request, retrospective_bundle
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    start = datetime(2025, 8, 2, 12, tzinfo=UTC)
    cutoff = retrospective_request.cutoff.history_cutoff_at
    baseline_pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    baseline_post = evaluate_retrospective_post(
        baseline_pre, lineup_from_ids(baseline_pre, CHAMPION_IDS), target_started_at=start
    )
    source = retrospective_request.history[0]
    extras = tuple(
        replace(
            source,
            game=replace(source.game, game_id=game_id, ended_at=end, winner_team_id="B"),
            slots=(),
        )
        for game_id, end in (
            (retrospective_request.target.game_id, datetime(2025, 1, 1)),
            ("synthetic-post-equal", cutoff),
            ("synthetic-post-future", cutoff + timedelta(seconds=1)),
        )
    )
    pre = create_pre(
        replace(retrospective_request, history=retrospective_request.history + extras),
        retrospective_bundle,
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    post = evaluate_retrospective_post(pre, lineup_from_ids(pre, CHAMPION_IDS), target_started_at=start)
    assert post.pre is pre
    assert post.features.player_champion == baseline_post.features.player_champion
    assert post.prediction.p_blue_win == baseline_post.prediction.p_blue_win
    assert post.features.accepted_game_ids == baseline_pre.features.accepted_game_ids
    assert post.features.counts.input_games == 5
    assert post.features.counts.excluded_games == 3
    assert post.features.exclusions == pre.features.exclusions
    with pytest.raises(TypeError, match="history"):
        evaluate_retrospective_post(
            pre, post.lineup, target_started_at=start, history=retrospective_request.history
        )


def test_retrospective_post_champion_change_preserves_pre_and_reports_missing_pair(
    monkeypatch, retrospective_request, retrospective_bundle
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    before = deepcopy(pre)
    start = datetime(2025, 8, 2, 12, tzinfo=UTC)
    full = evaluate_retrospective_post(pre, lineup_from_ids(pre, CHAMPION_IDS), target_started_at=start)
    changed = evaluate_retrospective_post(
        pre, lineup_from_ids(pre, ("99",) + CHAMPION_IDS[1:]), target_started_at=start
    )
    assert pre == before
    assert changed.pre is full.pre is pre
    assert changed.features.player_champion[1:] == full.features.player_champion[1:]
    missing = changed.features.player_champion[0]
    assert missing.champion_id == "99"
    assert missing.games_count == missing.wins_count == 0
    assert missing.win_rate is None
    assert missing.missing
    assert missing.game_ids == ()
    warning = next(item for item in changed.warnings if item.code == "W_PAIR_HISTORY_MISSING")
    assert warning.sample_count == 1
    assert changed.comparison.p_blue_win_pre == full.comparison.p_blue_win_pre


def test_retrospective_post_empty_history_keeps_all_pair_missing_flags(
    monkeypatch, retrospective_request, retrospective_bundle
):
    _install_canonical_loader_spy(monkeypatch, retrospective_bundle)
    pre = create_pre(
        replace(retrospective_request, history=()),
        retrospective_bundle,
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    post = evaluate_retrospective_post(
        pre,
        lineup_from_ids(pre, CHAMPION_IDS),
        target_started_at=datetime(2025, 8, 2, 12, tzinfo=UTC),
    )
    assert post.pre is pre
    for feature in post.features.player_champion:
        assert feature.games_count == feature.wins_count == 0
        assert feature.win_rate is None
        assert feature.missing
    warning = next(item for item in post.warnings if item.code == "W_PAIR_HISTORY_MISSING")
    assert warning.sample_count == 10
    assert post.features.counts.input_games == 0
    assert isfinite(post.prediction.p_blue_win) and 0 <= post.prediction.p_blue_win <= 1


@pytest.fixture
def retrospective_comparison_inputs(monkeypatch, retrospective_request, retrospective_bundle):
    """Real synthetic PRE/POST first; comparison never trains or changes model identity."""
    pre = create_pre(
        retrospective_request, retrospective_bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    )
    post = create_post(
        pre,
        lineup_from_ids(pre, CHAMPION_IDS),
        retrospective_bundle,
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    expectation = replace(
        runner.CANONICAL_RETROSPECTIVE_EXPECTATION,
        identity=retrospective_bundle.identity,
        family=retrospective_bundle.family,
    )
    monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION", expectation)
    return pre, post, retrospective_bundle


def _forbid_comparison_work(monkeypatch):
    _forbid_runtime_training(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("Comparison must not hash, load, build features or predict")

    for name in (
        "_digest", "_seal", "request_key", "_require_snapshot",
        "load_retrospective_bundle", "create_pre", "create_post",
        "build_pre_features", "build_post_features", "predict_phase", "select_pre_history",
    ):
        monkeypatch.setattr(evaluation, name, forbidden)
    for name in ("_digest", "load_bundle", "predict_phase", "predict_paired", "_blue_probability"):
        monkeypatch.setattr(models, name, forbidden)
    monkeypatch.setattr(models.joblib, "load", forbidden)
    monkeypatch.setattr(runner, "load_retrospective_bundle", forbidden)
    monkeypatch.setattr(runner, "_file_sha256", forbidden)
    monkeypatch.setattr(models.Pipeline, "predict_proba", forbidden)


def _hand_probability_pair(pre, post):
    """Arithmetic-only example; .55/.60 are deliberately not claimed as fitted predictions."""
    pre = replace(pre, prediction=replace(pre.prediction, p_blue_win=0.55))
    post_prediction = replace(post.prediction, p_blue_win=0.60)
    post = replace(
        post,
        pre=pre,
        prediction=post_prediction,
        comparison=models.compare_predictions(pre.prediction, post_prediction),
    )
    return pre, post


def test_retrospective_comparison_real_service_integration_is_pure_and_serializable(
    monkeypatch, retrospective_comparison_inputs
):
    pre, post, bundle = retrospective_comparison_inputs
    before = deepcopy((pre, post))
    _forbid_comparison_work(monkeypatch)
    result = compare_retrospective_evaluations(pre, post, bundle)
    assert result == compare_retrospective_evaluations(pre, post, bundle)
    assert (pre, post) == before
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["game_id"] == pre.prediction.game_id
    assert result["history_cutoff_at"] == pre.prediction.history_cutoff_at.isoformat()
    assert result["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert result["evaluation_protocol"] == RETROSPECTIVE_PROTOCOL
    assert result["pre"] == {
        "evaluation_type": "PRE",
        "blue_win_probability": post.comparison.p_blue_win_pre,
        "red_win_probability": 1 - post.comparison.p_blue_win_pre,
    }
    assert result["post"] == {
        "evaluation_type": "POST",
        "blue_win_probability": post.comparison.p_blue_win_post,
        "red_win_probability": 1 - post.comparison.p_blue_win_post,
    }
    assert result["comparison"] == {
        "blue_probability_delta": post.comparison.delta_probability,
        "red_probability_delta": -post.comparison.delta_probability,
    }
    assert result["model"] == {
        "family": bundle.family,
        "dataset_id": bundle.identity.dataset_id,
        "split_id": bundle.identity.split_id,
        "bundle_version": bundle.identity.bundle_version,
        "feature_schema_version": bundle.feature_schema_version,
        "preprocessing_version": bundle.preprocessing_version,
    }
    for phase in ("pre", "post"):
        assert isfinite(result[phase]["blue_win_probability"])
        assert 0 <= result[phase]["blue_win_probability"] <= 1
        assert (
            result[phase]["blue_win_probability"] + result[phase]["red_win_probability"]
        ) == pytest.approx(1)
    copied = deepcopy(pre)
    assert copied is not pre and copied == pre
    assert compare_retrospective_evaluations(copied, post, bundle) == result


def test_retrospective_comparison_hand_arithmetic_and_cached_roundoff_are_preserved(
    monkeypatch, retrospective_comparison_inputs
):
    pre, post, bundle = retrospective_comparison_inputs
    pre, post = _hand_probability_pair(pre, post)
    _forbid_comparison_work(monkeypatch)
    result = compare_retrospective_evaluations(pre, post, bundle)
    assert result["pre"]["blue_win_probability"] == 0.55
    assert result["post"]["blue_win_probability"] == 0.60
    assert result["comparison"]["blue_probability_delta"] == pytest.approx(0.05)
    assert result["comparison"]["red_probability_delta"] == pytest.approx(-0.05)
    assert percentage_points(result["comparison"]["blue_probability_delta"]) == pytest.approx(5)
    cached = replace(
        post.comparison,
        p_blue_win_pre=nextafter(post.comparison.p_blue_win_pre, 1.0),
        p_blue_win_post=nextafter(post.comparison.p_blue_win_post, 1.0),
        delta_probability=nextafter(post.comparison.delta_probability, 1.0),
    )
    rounded_post = replace(post, comparison=cached)
    rounded = compare_retrospective_evaluations(pre, rounded_post, bundle)
    assert rounded["pre"]["blue_win_probability"] == cached.p_blue_win_pre
    assert rounded["post"]["blue_win_probability"] == cached.p_blue_win_post
    assert rounded["comparison"]["blue_probability_delta"] == cached.delta_probability
    assert rounded["comparison"]["red_probability_delta"] == -cached.delta_probability


@pytest.mark.parametrize("field", ["p_blue_win_pre", "p_blue_win_post", "delta_probability"])
def test_retrospective_comparison_rejects_material_cached_numeric_disagreement(
    monkeypatch, retrospective_comparison_inputs, field
):
    pre, post, bundle = retrospective_comparison_inputs
    pre, post = _hand_probability_pair(pre, post)
    post = replace(post, comparison=replace(post.comparison, **{field: 0.2}))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize(
    "value", [None, "0.5", True, float("nan"), float("inf"), -0.1, 1.1]
)
@pytest.mark.parametrize("field", ["p_blue_win_pre", "p_blue_win_post"])
def test_retrospective_comparison_rejects_invalid_cached_probability(
    monkeypatch, retrospective_comparison_inputs, field, value
):
    pre, post, bundle = retrospective_comparison_inputs
    post = replace(post, comparison=replace(post.comparison, **{field: value}))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_MODEL_PROBABILITY_INVALID"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize("value", [None, "0", True, float("nan"), float("inf"), -1.1, 1.1])
def test_retrospective_comparison_rejects_invalid_cached_delta(
    monkeypatch, retrospective_comparison_inputs, value
):
    pre, post, bundle = retrospective_comparison_inputs
    post = replace(post, comparison=replace(post.comparison, delta_probability=value))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_MODEL_PROBABILITY_INVALID"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize(
    ("phase", "value"),
    [("PRE", float("nan")), ("PRE", True), ("POST", float("inf")), ("POST", 1.1)],
)
def test_retrospective_comparison_validates_phase_probabilities_too(
    monkeypatch, retrospective_comparison_inputs, phase, value
):
    pre, post, bundle = retrospective_comparison_inputs
    if phase == "PRE":
        pre = replace(pre, prediction=replace(pre.prediction, p_blue_win=value))
        post = replace(post, pre=pre)
    else:
        post = replace(post, prediction=replace(post.prediction, p_blue_win=value))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_MODEL_PROBABILITY_INVALID"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize("case", ["pre", "post", "comparison"])
def test_retrospective_comparison_rejects_missing_objects(
    monkeypatch, retrospective_comparison_inputs, case
):
    pre, post, bundle = retrospective_comparison_inputs
    if case == "pre":
        pre = None
    elif case == "post":
        post = None
    else:
        post = replace(post, comparison=None)
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("game_id", "another-target"),
        ("history_cutoff_at", datetime(2025, 8, 2, tzinfo=UTC)),
        ("model_bundle_version", "retrospective:7l:another-model"),
        ("feature_schema_version", "unsupported-schema"),
        ("policy_version", STRICT_PROTOCOL),
        ("context_signature", "different-context-signature"),
    ],
)
def test_retrospective_comparison_rejects_cached_metadata_disagreement(
    monkeypatch, retrospective_comparison_inputs, field, value
):
    pre, post, bundle = retrospective_comparison_inputs
    post = replace(post, comparison=replace(post.comparison, **{field: value}))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize(
    "case", ["target", "cutoff", "patch", "roster", "history", "protocol", "post_pre"]
)
def test_retrospective_comparison_rejects_inconsistent_snapshot_foundation(
    monkeypatch, retrospective_comparison_inputs, case
):
    pre, post, bundle = retrospective_comparison_inputs
    request = pre.request
    if case == "target":
        request = replace(request, target=replace(request.target, game_id="another-target"))
    elif case == "cutoff":
        cutoff = request.cutoff.history_cutoff_at + timedelta(seconds=1)
        request = replace(
            request,
            target=replace(request.target, history_cutoff_at=cutoff),
            cutoff=replace(request.cutoff, history_cutoff_at=cutoff),
        )
    elif case == "patch":
        request = replace(request, target=replace(request.target, patch="15.02"))
    elif case == "roster":
        roster = list(request.target.blue_roster)
        roster[0] = replace(roster[0], player_id="new-player")
        request = replace(request, target=replace(request.target, blue_roster=tuple(roster)))
    elif case == "history":
        record = request.history[0]
        record = replace(record, game=replace(record.game, winner_team_id="B"))
        request = replace(request, history=(record,) + request.history[1:])
    elif case == "protocol":
        request = replace(
            request,
            cutoff=replace(request.cutoff, verification=CutoffVerification.VERIFIED_EXTERNALLY),
        )
    else:
        post = replace(post, pre=replace(pre, context_key="different-pre-context"))
    if case != "post_pre":
        pre = replace(pre, request=request)
        # Changing both references must not bypass comparison's structural validation.
        post = replace(post, pre=pre)
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError):
        compare_retrospective_evaluations(pre, post, bundle)


@pytest.mark.parametrize(
    "case", ["dataset", "split", "version", "family", "fit_signature", "protocol"]
)
def test_retrospective_comparison_binds_to_expected_model_contract(
    monkeypatch, retrospective_comparison_inputs, case
):
    pre, post, bundle = retrospective_comparison_inputs
    if case in {"dataset", "split", "version"}:
        field = {"dataset": "dataset_id", "split": "split_id", "version": "bundle_version"}[case]
        bundle = replace(bundle, identity=replace(bundle.identity, **{field: "retrospective:other"}))
    elif case == "family":
        expectation = replace(evaluation.CANONICAL_RETROSPECTIVE_EXPECTATION, family="random_forest")
        if expectation.family == bundle.family:
            expectation = replace(expectation, family="logistic_regression")
        monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION", expectation)
    elif case == "fit_signature":
        bundle = replace(bundle, fit_signature="different-fitted-signature")
    else:
        bundle = replace(bundle, contract=replace(bundle.contract, dataset_version="paired-pre-post-v1"))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError):
        compare_retrospective_evaluations(pre, post, bundle)


def test_retrospective_comparison_aggregates_warning_fields_deterministically(
    monkeypatch, retrospective_comparison_inputs
):
    pre, post, bundle = retrospective_comparison_inputs
    first = BusinessWarning("W_TEAM_HISTORY_SMALL", "A", 2, 10)
    distinct_count = BusinessWarning("W_TEAM_HISTORY_SMALL", "A", 3, 10)
    missing_pair = BusinessWarning("W_PAIR_HISTORY_MISSING", sample_count=1)
    pre = replace(pre, warnings=(first, distinct_count, first))
    post = replace(post, pre=pre, warnings=(missing_pair, first, missing_pair))
    before = deepcopy((pre, post))
    _forbid_comparison_work(monkeypatch)
    result = compare_retrospective_evaluations(pre, post, bundle)
    assert result["warnings"] == [
        {"code": "W_PAIR_HISTORY_MISSING", "team_id": None, "sample_count": 1, "requested_count": 0},
        {"code": "W_TEAM_HISTORY_SMALL", "team_id": "A", "sample_count": 2, "requested_count": 10},
        {"code": "W_TEAM_HISTORY_SMALL", "team_id": "A", "sample_count": 3, "requested_count": 10},
    ]
    reordered_pre = replace(pre, warnings=tuple(reversed(pre.warnings)))
    reordered_post = replace(post, pre=reordered_pre, warnings=tuple(reversed(post.warnings)))
    assert compare_retrospective_evaluations(reordered_pre, reordered_post, bundle) == result
    assert (pre, post) == before
    result["warnings"][0]["sample_count"] = 99
    assert post.warnings[0].sample_count == 1


def test_retrospective_comparison_rejects_strict_demo_without_changing_it(
    monkeypatch, eval_request, bundle, retrospective_bundle
):
    pre = create_pre(eval_request, bundle)
    post = create_post(pre, lineup_from_ids(pre, CHAMPION_IDS), bundle)
    before = deepcopy((pre, post))
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError):
        compare_retrospective_evaluations(pre, post, retrospective_bundle)
    with pytest.raises(PreFeatureInputError):
        compare_retrospective_evaluations(pre, post, bundle)
    assert (pre, post) == before


@pytest.mark.parametrize("case", ["different_lineup", "missing_saved_pair", "changed_saved_identity"])
def test_retrospective_comparison_binds_saved_pair_identities_to_lineup(
    monkeypatch, retrospective_comparison_inputs, case
):
    pre, post, bundle = retrospective_comparison_inputs
    if case == "different_lineup":
        # Lux is valid and distinct, but the saved POST still describes the original champion.
        lineup = lineup_from_ids(pre, ("99",) + CHAMPION_IDS[1:])
        post = replace(post, lineup=lineup)
    elif case == "missing_saved_pair":
        post = replace(
            post,
            features=replace(post.features, player_champion=post.features.player_champion[:-1]),
        )
    else:
        pairs = post.features.player_champion
        changed = replace(pairs[0], champion_id="99")
        post = replace(
            post,
            features=replace(post.features, player_champion=(changed,) + pairs[1:]),
        )
    _forbid_comparison_work(monkeypatch)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_retrospective_evaluations(pre, post, bundle)
