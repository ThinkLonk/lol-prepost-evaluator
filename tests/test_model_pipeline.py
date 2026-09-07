"""Synthetic model tests only; no production temporal evidence, DB or network."""

import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.base import clone

import match_insight.ml.models as models
import match_insight.ml.real_training as runner
from match_insight.features.post import ChampionSlot, FinalLineup, HistoricalChampionGame
from match_insight.features.pre import ROLES, HistoricalGame, PlayerSlot, PreFeatureInputError
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
from match_insight.ml.models import (
    CONTEXT_COLUMNS,
    FAMILIES,
    NUMERIC_PRE_COLUMNS,
    ModelConfig,
    ModelIdentity,
    ModelInputError,
    ProbabilityMetrics,
    SelectionPolicy,
    ValidationScore,
    choose_family,
    compare_predictions,
    evaluate_probabilities,
    evaluate_test,
    fit_model_bundle,
    load_bundle,
    predict_paired,
    predict_phase,
    save_bundle,
)

BASE = datetime(2025, 7, 1, 12, tzinfo=UTC)
POLICY = "synthetic-7j-v1"
IDENTITY = ModelIdentity("synthetic-dataset-7j", "synthetic-split-7j", "synthetic-bundle-7j")
CONFIG = ModelConfig(forest_trees=8)
REFERENCE = frozenset(str(value) for value in range(1, 12))


def roster(team):
    return tuple(
        PlayerSlot(role, f"{team.lower()}{number}")
        for number, role in enumerate(ROLES, start=1)
    )


def slots(blue="A", red="B", unknown=False):
    result = tuple(
        ChampionSlot(
            team_id=team,
            side=side,
            role=role,
            player_id=f"{team.lower()}{number}",
            champion_id=str(number + (0 if team == "A" else 5)),
        )
        for side, team in (("BLUE", blue), ("RED", red))
        for number, role in enumerate(ROLES, start=1)
    )
    if unknown:
        result = (replace(result[0], champion_id="11"),) + result[1:]
    return result


def make_dataset(
    unknown_holdout=False,
    extra_holdout_history=False,
    late_history=False,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
):
    history = []
    for number, (blue, red, winner) in enumerate(
        (("A", "B", "A"), ("B", "A", "B")),
        start=1,
    ):
        end = BASE - timedelta(days=3 - number)
        if late_history:
            end = BASE + timedelta(days=13, hours=number)
        history.append(
            HistoricalChampionGame(
                HistoricalGame(
                    game_id=f"h{number}",
                    blue_team_id=blue,
                    red_team_id=red,
                    blue_roster=roster(blue),
                    red_roster=roster(red),
                    winner_team_id=winner,
                    ended_at=end,
                ),
                slots(blue, red),
            )
        )
    if extra_holdout_history:
        history.append(
            HistoricalChampionGame(
                replace(
                    history[0].game,
                    game_id="h3",
                    ended_at=BASE + timedelta(days=13, hours=3),
                ),
                slots(),
            )
        )

    targets = []
    cutoffs = []
    retrospective = evaluation_protocol == RETROSPECTIVE_PROTOCOL
    cutoff_policy = RETROSPECTIVE_PROTOCOL if retrospective else POLICY
    cutoff_verification = (
        CutoffVerification.PROTOCOL_ASSUMED
        if retrospective
        else CutoffVerification.VERIFIED_EXTERNALLY
    )
    for number in range(1, 21):
        game_id = f"g{number:02d}"
        cutoff = BASE + timedelta(days=number - 1)
        if retrospective:
            cutoff = cutoff.replace(hour=0)
        targets.append(
            DatasetTarget(
                game_id=game_id,
                blue_team_id="A",
                red_team_id="B",
                blue_roster=roster("A"),
                red_roster=roster("B"),
                patch="15.01",
                final_lineup=FinalLineup(
                    game_id,
                    "15.01",
                    slots(unknown=unknown_holdout and number >= 15),
                ),
                winner_team_id="A" if number <= 9 or number in (15, 18, 20) else "B",
                ended_at=cutoff + timedelta(hours=1),
            )
        )
        cutoffs.append(
            CutoffRecord(
                game_id=game_id,
                history_cutoff_at=cutoff,
                evidence_ref=f"synthetic://7j/{game_id}",
                policy_version=cutoff_policy,
                # Fixture assertion only; this does not certify a real cutoff source.
                verification=cutoff_verification,
            )
        )

    return build_paired_dataset(
        targets=tuple(targets),
        history=tuple(history),
        cutoffs=tuple(cutoffs),
        champion_reference=REFERENCE,
        approved_policy_versions=frozenset({cutoff_policy}),
        evaluation_protocol=evaluation_protocol,
    )


@pytest.fixture(scope="module")
def fitted():
    dataset = make_dataset()
    split = temporal_split(dataset)
    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)
    return dataset, split, bundle


def capture_pipelines(monkeypatch):
    captured = []
    original = models._make_pipeline

    def capture(phase, family, config):
        pipeline = original(phase, family, config)
        captured.append((phase, family, pipeline))
        return pipeline

    monkeypatch.setattr(models, "_make_pipeline", capture)
    return captured


def preprocessing_state(pipeline):
    transformer = pipeline.named_steps["features"]
    numeric = transformer.named_transformers_["numeric"]
    state = {
        "imputer": numeric.named_steps["imputer"].statistics_.copy(),
        "mean": numeric.named_steps["scaler"].mean_.copy(),
        "scale": numeric.named_steps["scaler"].scale_.copy(),
    }
    if "champions" in transformer.named_transformers_:
        for number, categories in enumerate(
            transformer.named_transformers_["champions"].categories_
        ):
            state[f"category_{number}"] = categories.copy()
    return state


def test_all_three_families_fit_independent_pre_post_and_baseline_is_train_prior(monkeypatch):
    captured = capture_pipelines(monkeypatch)
    dataset = make_dataset()
    split = temporal_split(dataset)
    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)

    assert len(captured) == 6
    assert len({id(pipeline) for _, _, pipeline in captured}) == 6
    assert {score.family for score in bundle.validation_scores} == set(FAMILIES)
    assert len(split.train) == 14
    assert int(dataset.y.loc[list(split.train)].sum()) == 9

    baseline = {}
    for phase, family, pipeline in captured:
        raw = dataset.X_pre if phase == "PRE" else dataset.X_post
        frame = models._model_frame(raw.loc[list(split.validation)], phase)
        probabilities = models._blue_probability(pipeline, frame)
        assert np.isfinite(probabilities).all()
        assert ((0 <= probabilities) & (probabilities <= 1)).all()
        assert set(pipeline.named_steps["model"].classes_) == {0, 1}
        if family == "baseline":
            np.testing.assert_allclose(probabilities, 9 / 14)
            baseline[phase] = probabilities

    np.testing.assert_array_equal(baseline["POST"] - baseline["PRE"], 0)


def test_integration_from_real_7i_to_locked_test_predictions(fitted):
    dataset, split, bundle = fitted
    assert dataset.X_pre.at["g01", "blue_recent_form_value"] == 0.5
    assert dataset.X_post.at["g01", "blue_top_games_count"] == 2

    result = evaluate_test(bundle, dataset)

    assert [row.game_id for row in result.predictions] == list(split.test)
    assert result.pre.sample_count == result.post.sample_count == 3
    assert bundle.split == split
    for row in result.predictions:
        assert row.delta_probability == pytest.approx(
            row.p_blue_win_post - row.p_blue_win_pre
        )
        assert 0 <= row.p_blue_win_pre <= 1
        assert 0 <= row.p_blue_win_post <= 1
        assert row.model_bundle_version == IDENTITY.bundle_version
        assert row.history_cutoff_at == dataset.metadata.at[row.game_id, "history_cutoff_at"]


def test_holdout_changes_do_not_change_any_fitted_preprocessing(monkeypatch):
    captured = capture_pipelines(monkeypatch)
    original = make_dataset()
    changed = make_dataset(unknown_holdout=True, extra_holdout_history=True)
    split = temporal_split(original)

    assert_frame_equal(
        original.X_pre.loc[list(split.train)],
        changed.X_pre.loc[list(split.train)],
    )
    assert_frame_equal(
        original.X_post.loc[list(split.train)],
        changed.X_post.loc[list(split.train)],
    )

    fit_model_bundle(original, split, IDENTITY, CONFIG)
    first = [(phase, family, preprocessing_state(pipe)) for phase, family, pipe in captured]
    captured.clear()
    fit_model_bundle(changed, temporal_split(changed), IDENTITY, CONFIG)
    second = [(phase, family, preprocessing_state(pipe)) for phase, family, pipe in captured]

    for (phase_a, family_a, state_a), (phase_b, family_b, state_b) in zip(
        first, second, strict=True
    ):
        assert (phase_a, family_a) == (phase_b, family_b)
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            np.testing.assert_array_equal(state_a[key], state_b[key])


def test_unseen_champion_uses_fitted_unknown_category_policy():
    dataset = make_dataset(unknown_holdout=True)
    bundle = fit_model_bundle(dataset, temporal_split(dataset), IDENTITY, CONFIG)

    encoder = bundle.post_pipeline.named_steps["features"].named_transformers_["champions"]
    assert "11" not in encoder.categories_[0]
    assert encoder.handle_unknown == "ignore"
    result = evaluate_test(bundle, dataset)
    assert len(result.predictions) == 3


def test_all_missing_train_columns_are_retained_with_internal_zero_fallback():
    dataset = make_dataset(late_history=True)
    before = deepcopy(dataset)
    split = temporal_split(dataset)
    assert all(
        value is None
        for value in dataset.X_pre.loc[list(split.train), "blue_recent_form_value"]
    )

    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)
    transformer = bundle.pre_pipeline.named_steps["features"]
    imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    position = NUMERIC_PRE_COLUMNS.index("blue_recent_form_value")

    assert imputer.statistics_[position] == 0
    assert len(imputer.statistics_) == len(NUMERIC_PRE_COLUMNS)
    assert "numeric__blue_recent_form_value" in transformer.get_feature_names_out()
    evaluate_test(bundle, dataset)
    assert_frame_equal(dataset.X_pre, before.X_pre)
    assert_frame_equal(dataset.X_post, before.X_post)


@pytest.mark.parametrize(
    "corruption",
    ["post_order", "label_order", "missing_column", "duplicate_column", "label_feature", "pre_drift"],
)
def test_alignment_and_exact_feature_schema_are_enforced(corruption):
    dataset = make_dataset()
    split = temporal_split(dataset)
    if corruption == "post_order":
        dataset = replace(dataset, X_post=dataset.X_post.iloc[::-1].copy())
    elif corruption == "label_order":
        dataset = replace(dataset, y=dataset.y.iloc[::-1].copy())
    elif corruption == "missing_column":
        dataset = replace(dataset, X_pre=dataset.X_pre.iloc[:, 1:].copy())
    elif corruption == "duplicate_column":
        duplicated = pd.concat([dataset.X_pre, dataset.X_pre.iloc[:, :1]], axis=1)
        dataset = replace(dataset, X_pre=duplicated)
    else:
        post = dataset.X_post.copy(deep=True)
        if corruption == "label_feature":
            post["y_blue_win"] = dataset.y
        else:
            post.at["g01", "blue_recent_form_value"] = 0.99
        dataset = replace(dataset, X_post=post)

    with pytest.raises(ModelInputError):
        fit_model_bundle(dataset, split, IDENTITY, CONFIG)


@pytest.mark.parametrize("value", ["0.5", float("inf"), True])
def test_invalid_numeric_values_are_not_coerced_into_missing(value):
    dataset = make_dataset()
    split = temporal_split(dataset)
    pre = dataset.X_pre.copy(deep=True)
    post = dataset.X_post.copy(deep=True)
    pre.at["g01", "blue_recent_form_value"] = value
    post.at["g01", "blue_recent_form_value"] = value

    with pytest.raises(ModelInputError, match="E_MODEL_VALUE_INVALID"):
        fit_model_bundle(replace(dataset, X_pre=pre, X_post=post), split, IDENTITY, CONFIG)


def test_single_class_train_is_explicitly_rejected():
    dataset = make_dataset()
    split = temporal_split(dataset)
    labels = dataset.y.copy()
    labels.loc[list(split.train)] = 1

    with pytest.raises(ModelInputError, match="E_TRAIN_SINGLE_CLASS"):
        fit_model_bundle(replace(dataset, y=labels), split, IDENTITY, CONFIG)


def test_single_class_validation_keeps_other_metrics_and_reports_missing_auc():
    dataset = make_dataset()
    split = temporal_split(dataset)
    labels = dataset.y.copy()
    labels.loc[list(split.validation)] = 1

    bundle = fit_model_bundle(replace(dataset, y=labels), split, IDENTITY, CONFIG)

    for score in bundle.validation_scores:
        for metrics in (score.pre, score.post):
            assert metrics.roc_auc is None
            assert metrics.roc_auc_reason == "EVAL_SINGLE_CLASS"
            assert metrics.sample_count == 3
            assert np.isfinite(metrics.brier_score)
            assert np.isfinite(metrics.log_loss)
            assert sum(item.sample_count for item in metrics.calibration) == 3


def score(family, brier, loss):
    metrics = ProbabilityMetrics(brier, loss, None, "SYNTHETIC_SELECTION", 3, ())
    return ValidationScore(family, metrics, metrics)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            ((0.2, 0.6), (0.2, 0.6), (0.2, 0.6)),
            "baseline",
        ),
        (
            ((0.2, 0.6), (0.19, 0.59), (0.1895, 0.58)),
            "logistic_regression",
        ),
        (
            ((0.2, 0.6), (0.19, 0.59), (0.188, 0.58)),
            "random_forest",
        ),
        (
            ((0.2, 0.6), (0.2, 0.59), (0.21, 0.58)),
            "logistic_regression",
        ),
    ],
)
def test_reviewable_validation_selection_policy(values, expected):
    scores = tuple(
        score(family, brier, loss)
        for family, (brier, loss) in zip(FAMILIES, values, strict=True)
    )
    assert choose_family(scores, SelectionPolicy()) == expected
    assert choose_family(tuple(reversed(scores)), SelectionPolicy()) == expected


def test_selection_uses_mean_pre_post_brier():
    baseline = score("baseline", 0.2, 0.6)
    logistic = score("logistic_regression", 0.19, 0.6)
    logistic = replace(
        logistic,
        pre=replace(logistic.pre, brier_score=0.1),
        post=replace(logistic.post, brier_score=0.3),
    )
    forest = score("random_forest", 0.21, 0.6)
    assert choose_family((baseline, logistic, forest)) == "baseline"


def test_test_metrics_are_only_computed_after_selection(monkeypatch):
    dataset = make_dataset()
    split = temporal_split(dataset)
    seen = []
    original_evaluate = models.evaluate_probabilities
    original_choose = models.choose_family

    def record_evaluation(labels, probabilities, bins=10):
        seen.append(tuple(labels.index))
        return original_evaluate(labels, probabilities, bins)

    def select_from_validation(validation_scores, policy):
        assert seen == [split.validation] * 6
        return original_choose(validation_scores, policy)

    monkeypatch.setattr(models, "evaluate_probabilities", record_evaluation)
    monkeypatch.setattr(models, "choose_family", select_from_validation)

    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)
    assert seen == [split.validation] * 6

    evaluate_test(bundle, dataset)
    assert seen == [split.validation] * 6 + [split.test] * 2


def test_changing_test_labels_cannot_change_the_selected_model(fitted):
    dataset, split, original = fitted
    labels = dataset.y.copy()
    labels.loc[list(split.test)] = 1 - labels.loc[list(split.test)]
    changed = replace(dataset, y=labels)

    updated = fit_model_bundle(changed, split, IDENTITY, CONFIG)

    assert updated.family == original.family
    assert updated.validation_scores == original.validation_scores
    assert updated.fit_signature == original.fit_signature
    assert updated.dataset_signature != original.dataset_signature
    ids = list(split.test)
    left = predict_paired(
        original,
        dataset.X_pre.loc[ids],
        dataset.X_post.loc[ids],
        dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)],
    )
    right = predict_paired(
        updated,
        changed.X_pre.loc[ids],
        changed.X_post.loc[ids],
        changed.metadata.loc[ids, list(CONTEXT_COLUMNS)],
    )
    assert left == right
    with pytest.raises(ModelInputError, match="E_MODEL_DATASET_IDENTITY"):
        evaluate_test(original, changed)


def test_save_load_preserves_predictions(tmp_path, fitted):
    dataset, _split, bundle = fitted
    path = tmp_path / "synthetic-bundle.joblib"
    before = evaluate_test(bundle, dataset)

    save_bundle(bundle, path)
    restored = load_bundle(path)
    after = evaluate_test(restored, dataset)

    assert before == after
    assert restored.identity == bundle.identity
    assert restored.library_versions == bundle.library_versions
    assert restored.estimator_parameters == bundle.estimator_parameters
    with pytest.raises(ModelInputError, match="E_MODEL_ARTIFACT_EXISTS"):
        save_bundle(bundle, path)


@pytest.mark.parametrize(
    "field",
    [
        "game_id",
        "history_cutoff_at",
        "model_bundle_version",
        "bundle_signature",
        "feature_schema_version",
        "policy_version",
        "context_signature",
        "pre_feature_signature",
    ],
)
def test_comparison_rejects_incompatible_predictions(field, fitted):
    dataset, _split, bundle = fitted
    ids = ["g18"]
    metadata = dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)]
    pre = predict_phase(bundle, "PRE", dataset.X_pre.loc[ids], metadata)[0]
    post = predict_phase(bundle, "POST", dataset.X_post.loc[ids], metadata)[0]
    changed = (
        post.history_cutoff_at + timedelta(seconds=1)
        if field == "history_cutoff_at"
        else "different"
    )

    with pytest.raises(ModelInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_predictions(pre, replace(post, **{field: changed}))


def test_context_change_is_detected_from_real_metadata(fitted):
    dataset, _split, bundle = fitted
    ids = ["g18"]
    metadata = dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)].copy(deep=True)
    pre = predict_phase(bundle, "PRE", dataset.X_pre.loc[ids], metadata)[0]
    metadata.at["g18", "patch"] = "15.02"
    post = predict_phase(bundle, "POST", dataset.X_post.loc[ids], metadata)[0]

    with pytest.raises(ModelInputError, match="E_EVAL_INCOMPATIBLE"):
        compare_predictions(pre, post)


@pytest.mark.parametrize("case", ["duplicate", "omitted", "unknown", "cross_boundary"])
def test_forged_split_is_rejected_before_fit(case):
    dataset = make_dataset()
    split = temporal_split(dataset)
    if case == "duplicate":
        changed = replace(split, train=split.train + (split.train[0],))
    elif case == "omitted":
        changed = replace(split, train=split.train[:-1])
    elif case == "unknown":
        changed = replace(split, train=split.train[:-1] + ("unknown",))
    else:
        changed = replace(
            split,
            train=split.train[:-1] + (split.validation[0],),
            validation=(split.train[-1],) + split.validation[1:],
        )
    with pytest.raises(ModelInputError, match="E_MODEL_SPLIT_INVALID"):
        fit_model_bundle(dataset, changed, IDENTITY, CONFIG)


def test_valid_purged_split_keeps_original_boundary():
    dataset = make_dataset()
    metadata = dataset.metadata.copy(deep=True)
    metadata.at["g15", "target_ended_at"] = metadata.at["g18", "history_cutoff_at"]
    dataset = replace(dataset, metadata=metadata)
    split = temporal_split(dataset)

    assert split.status == "OK"
    assert split.validation == ("g16", "g17")
    assert split.validation_boundary == metadata.at["g15", "history_cutoff_at"]

    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)
    assert bundle.split.validation_boundary == split.validation_boundary
    assert evaluate_test(bundle, dataset).pre.sample_count == 3


def test_probability_lookup_uses_class_identity_not_column_position():
    fake = SimpleNamespace(
        named_steps={"model": SimpleNamespace(classes_=np.array([1, 0]))},
        predict_proba=lambda frame: np.tile([0.8, 0.2], (len(frame), 1)),
    )
    actual = models._blue_probability(fake, pd.DataFrame({"x": [1, 2]}))
    np.testing.assert_allclose(actual, [0.8, 0.8])


def test_calibration_counts_probability_endpoints_and_metrics_match_hand_calculation():
    perfect = evaluate_probabilities([0, 1], [0.0, 1.0], bins=10)
    assert perfect.brier_score == 0
    assert perfect.roc_auc == 1
    assert perfect.calibration[0].sample_count == 1
    assert perfect.calibration[-1].sample_count == 1
    assert sum(item.sample_count for item in perfect.calibration) == 2

    baseline = evaluate_probabilities([1, 0, 1], [9 / 14] * 3, bins=10)
    assert baseline.brier_score == pytest.approx(131 / 588)
    assert baseline.roc_auc == 0.5
    assert baseline.calibration[6].sample_count == 3
    assert baseline.calibration[6].mean_probability == pytest.approx(9 / 14)
    assert baseline.calibration[6].observed_blue_win_rate == pytest.approx(2 / 3)


def test_fit_prediction_and_evaluation_do_not_mutate_dataset():
    dataset = make_dataset()
    before = deepcopy(dataset)
    split = temporal_split(dataset)

    bundle = fit_model_bundle(dataset, split, IDENTITY, CONFIG)
    evaluate_test(bundle, dataset)

    assert_frame_equal(dataset.X_pre, before.X_pre)
    assert_frame_equal(dataset.X_post, before.X_post)
    assert_frame_equal(dataset.metadata, before.metadata)
    assert_series_equal(dataset.y, before.y)
    assert dataset.excluded == before.excluded


@pytest.fixture(scope="module")
def retrospective_fitted():
    dataset = make_dataset(evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    split = temporal_split(dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    identity = ModelIdentity(
        "synthetic-retrospective-dataset",
        "synthetic-retrospective-split",
        "synthetic-retrospective-bundle",
    )
    bundle = fit_model_bundle(
        dataset,
        split,
        identity,
        CONFIG,
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    return dataset, split, bundle


@pytest.mark.parametrize(
    ("dataset_protocol", "split_protocol", "fit_protocol"),
    (
        (RETROSPECTIVE_PROTOCOL, RETROSPECTIVE_PROTOCOL, STRICT_PROTOCOL),
        (STRICT_PROTOCOL, STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL),
        (RETROSPECTIVE_PROTOCOL, STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL),
        (STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL, STRICT_PROTOCOL),
    ),
)
def test_fit_rejects_protocol_mismatch_before_estimator_creation(
    monkeypatch, dataset_protocol, split_protocol, fit_protocol
):
    dataset = make_dataset(evaluation_protocol=dataset_protocol)
    split_dataset = make_dataset(evaluation_protocol=split_protocol)
    split = temporal_split(split_dataset, evaluation_protocol=split_protocol)

    def forbidden_pipeline(*args, **kwargs):
        pytest.fail("Protocol mismatch must fail before estimator creation")

    monkeypatch.setattr(models, "_make_pipeline", forbidden_pipeline)
    with pytest.raises(ModelInputError, match="E_MODEL_SCHEMA|E_MODEL_SPLIT_INVALID"):
        fit_model_bundle(
            dataset,
            split,
            IDENTITY,
            CONFIG,
            evaluation_protocol=fit_protocol,
        )


def test_default_fit_rejects_retrospective_dataset(retrospective_fitted):
    dataset, split, _bundle = retrospective_fitted
    with pytest.raises(ModelInputError, match="E_MODEL_SCHEMA"):
        fit_model_bundle(dataset, split, IDENTITY, CONFIG)


@pytest.mark.parametrize("use_retrospective_bundle", (False, True))
def test_predict_and_evaluate_reject_cross_protocol_context(
    fitted, retrospective_fitted, use_retrospective_bundle
):
    strict_dataset, _strict_split, strict_bundle = fitted
    retro_dataset, _retro_split, retro_bundle = retrospective_fitted
    if use_retrospective_bundle:
        bundle, other = retro_bundle, strict_dataset
    else:
        bundle, other = strict_bundle, retro_dataset
    with pytest.raises(ModelInputError, match="E_MODEL_SCHEMA"):
        predict_paired(
            bundle,
            other.X_pre,
            other.X_post,
            other.metadata.loc[:, list(CONTEXT_COLUMNS)],
        )
    with pytest.raises(ModelInputError, match="E_MODEL_SCHEMA"):
        evaluate_test(bundle, other)


def test_retrospective_save_load_preserves_predictions_and_contract(
    tmp_path, retrospective_fitted
):
    dataset, split, bundle = retrospective_fitted
    before = evaluate_test(bundle, dataset)
    path = tmp_path / "synthetic-retrospective.joblib"
    save_bundle(bundle, path)
    restored = load_bundle(path)
    after = evaluate_test(restored, dataset)
    assert before == after
    assert restored.contract == bundle.contract
    assert restored.contract.dataset_version == RETROSPECTIVE_DATASET_VERSION
    assert restored.contract.cutoff_policy_versions == (RETROSPECTIVE_PROTOCOL,)
    assert restored.split == split
    assert restored.identity == bundle.identity
    assert tuple(row.game_id for row in after.predictions) == split.test
    for prediction in after.predictions:
        assert prediction.policy_version == RETROSPECTIVE_PROTOCOL
        assert 0 <= prediction.p_blue_win_pre <= 1
        assert 0 <= prediction.p_blue_win_post <= 1
        assert prediction.delta_probability == pytest.approx(
            prediction.p_blue_win_post - prediction.p_blue_win_pre
        )


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("cutoff_verification", "VERIFIED_EXTERNALLY"),
        ("cutoff_verification", "UNVERIFIED"),
        ("policy_version", POLICY),
    ),
)
def test_retrospective_prediction_rejects_relabelled_metadata(
    retrospective_fitted, column, value
):
    dataset, _split, bundle = retrospective_fitted
    metadata = dataset.metadata.loc[:, list(CONTEXT_COLUMNS)].copy(deep=True)
    metadata.at["g01", column] = value
    with pytest.raises(ModelInputError, match="E_MODEL_SCHEMA"):
        predict_paired(bundle, dataset.X_pre, dataset.X_post, metadata)


@pytest.fixture(scope="module")
def artifact_bundle_factory():
    """Fit tiny synthetic fixtures with their final identity, never relabel a bundle."""
    cached = {}

    def get(evaluation_protocol):
        if evaluation_protocol not in cached:
            prefix = "real" if evaluation_protocol == STRICT_PROTOCOL else "retrospective"
            identity = ModelIdentity(
                f"{prefix}:postgresql:synthetic-8b",
                f"{prefix}:split:synthetic-8b",
                f"{prefix}:7l:synthetic-8b",
            )
            dataset = make_dataset(evaluation_protocol=evaluation_protocol)
            split = temporal_split(dataset, evaluation_protocol=evaluation_protocol)
            bundle = fit_model_bundle(
                dataset, split, identity, CONFIG, evaluation_protocol=evaluation_protocol
            )
            cached[evaluation_protocol] = dataset, split, bundle
        return cached[evaluation_protocol]

    return get


def artifact_loader(evaluation_protocol):
    if evaluation_protocol == STRICT_PROTOCOL:
        return runner.load_real_bundle
    return runner.load_retrospective_bundle


def write_artifact_pair(directory, bundle):
    """Write only caller-owned tmp_path artifacts; no canonical files are required."""
    path = directory / "synthetic-8b.joblib"
    save_bundle(bundle, path)
    protocol = models._contract_protocol(bundle.contract)
    data_kind = (
        "REAL_POSTGRESQL"
        if protocol == STRICT_PROTOCOL
        else "REAL_RETROSPECTIVE_SIMULATION"
    )
    metadata = {
        "artifact_schema": "real-model-artifact-v1",
        "data_kind": data_kind,
        "evaluation_protocol": protocol,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bundle": models._plain(runner._bundle_metadata(bundle)),
        "run": {
            "status": "TRAINED",
            "data_kind": data_kind,
            "evaluation_protocol": protocol,
            "dataset_id": bundle.identity.dataset_id,
            "split_id": bundle.identity.split_id,
            "model_bundle_version": bundle.identity.bundle_version,
            "selected_family": bundle.family,
        },
    }
    write_artifact_metadata(path, metadata)
    return path, metadata


def write_artifact_metadata(path, metadata):
    path.with_suffix(".json").write_text(
        json.dumps(metadata, allow_nan=False), encoding="utf-8"
    )


def artifact_expectation(bundle, metadata):
    return runner.ModelArtifactExpectation(
        identity=bundle.identity,
        model_sha256=metadata["model_sha256"],
        family=bundle.family,
    )


def guard_artifact_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Artifact loading and prediction must not fit or save anything")

    for owner in (models, runner, sys.modules[__name__]):
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


def artifact_inventory(directory):
    return {
        path.relative_to(directory).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("protocol", (STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL))
def test_artifact_loader_valid_roundtrip_and_prediction(
    tmp_path, monkeypatch, artifact_bundle_factory, protocol
):
    dataset, split, bundle = artifact_bundle_factory(protocol)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    expected = artifact_expectation(bundle, metadata)
    before = artifact_inventory(tmp_path)
    guard_artifact_runtime(monkeypatch)

    for expectation in (None, expected):
        restored = artifact_loader(protocol)(path, expected=expectation)
        assert restored.identity == bundle.identity
        assert restored.family == bundle.family
        assert restored.contract == bundle.contract
        assert restored.split == split
        result = evaluate_test(restored, dataset)
        assert result == evaluate_test(bundle, dataset)
        for row in result.predictions:
            assert 0 <= row.p_blue_win_pre <= 1
            assert 0 <= row.p_blue_win_post <= 1
            assert row.delta_probability == pytest.approx(
                row.p_blue_win_post - row.p_blue_win_pre
            )
    assert artifact_inventory(tmp_path) == before


_ARTIFACT_FIELD_MISSING = object()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_schema", _ARTIFACT_FIELD_MISSING),
        ("artifact_schema", "unsupported-v2"),
        ("data_kind", _ARTIFACT_FIELD_MISSING),
        ("data_kind", "SYNTHETIC"),
        ("evaluation_protocol", _ARTIFACT_FIELD_MISSING),
        ("evaluation_protocol", STRICT_PROTOCOL),
        ("model_sha256", "not-a-sha256"),
        ("bundle", None),
        ("bundle.family", _ARTIFACT_FIELD_MISSING),
        ("bundle.family", "unsupported"),
        ("bundle.feature_schema_version", _ARTIFACT_FIELD_MISSING),
        ("bundle.feature_schema_version", "unsupported"),
        ("bundle.preprocessing_version", _ARTIFACT_FIELD_MISSING),
        ("bundle.preprocessing_version", "unsupported"),
        ("bundle.pre_columns", _ARTIFACT_FIELD_MISSING),
        ("bundle.pre_columns", list(models.PRE_COLUMNS[:-1])),
        ("bundle.pre_columns", list(reversed(models.PRE_COLUMNS))),
        ("bundle.pre_columns", list(models.PRE_COLUMNS) + [models.PRE_COLUMNS[0]]),
        ("bundle.post_columns", _ARTIFACT_FIELD_MISSING),
        ("bundle.post_columns", list(models.POST_COLUMNS[:-1])),
        ("bundle.post_columns", list(reversed(models.POST_COLUMNS))),
        ("bundle.post_columns", list(models.POST_COLUMNS) + [models.POST_COLUMNS[0]]),
        ("bundle.identity", None),
        ("bundle.identity.dataset_id", "synthetic-wrong-namespace"),
        ("bundle.library_versions", _ARTIFACT_FIELD_MISSING),
        (
            "bundle.library_versions",
            [
                [name, "0.0-invalid" if name == "scikit-learn" else version]
                for name, version in models._versions()
            ],
        ),
        ("bundle.contract", None),
        ("bundle.contract.dataset_version", "unsupported"),
        ("bundle.contract.cutoff_policy_versions", [POLICY]),
        ("bundle.split.evaluation_protocol", STRICT_PROTOCOL),
        ("run", _ARTIFACT_FIELD_MISSING),
        ("run", []),
        ("run.status", "DRY_RUN_READY"),
        ("run.data_kind", "REAL_POSTGRESQL"),
        ("run.evaluation_protocol", _ARTIFACT_FIELD_MISSING),
        ("run.evaluation_protocol", STRICT_PROTOCOL),
        ("run.dataset_id", "retrospective:postgresql:different"),
        ("run.split_id", "retrospective:split:different"),
        ("run.model_bundle_version", _ARTIFACT_FIELD_MISSING),
        ("run.model_bundle_version", "retrospective:7l:different"),
        ("run.selected_family", _ARTIFACT_FIELD_MISSING),
        ("run.selected_family", "unsupported"),
    ],
)
def test_artifact_loader_metadata_rejected_before_deserialization(
    tmp_path, monkeypatch, artifact_bundle_factory, field, value
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    container = metadata
    parts = field.split(".")
    for part in parts[:-1]:
        container = container[part]
    if value is _ARTIFACT_FIELD_MISSING:
        del container[parts[-1]]
    else:
        container[parts[-1]] = value
    write_artifact_metadata(path, metadata)
    guard_artifact_runtime(monkeypatch)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Invalid sidecar metadata must fail before deserialization")

    monkeypatch.setattr(runner, "load_bundle", forbidden_load)
    with pytest.raises(PreFeatureInputError, match="E_REAL_MODEL_METADATA_INVALID"):
        runner.load_retrospective_bundle(path)


@pytest.mark.parametrize(
    "corruption", ("metadata_hash", "expected_hash", "dataset_id", "split_id", "version", "family")
)
def test_artifact_loader_hash_and_expected_identity_rejected(
    tmp_path, monkeypatch, artifact_bundle_factory, corruption
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    expected = artifact_expectation(bundle, metadata)
    code = "E_REAL_MODEL_IDENTITY_MISMATCH"
    if corruption == "metadata_hash":
        metadata["model_sha256"] = "0" * 64
        write_artifact_metadata(path, metadata)
        expected = None
        code = "E_REAL_MODEL_HASH_MISMATCH"
    elif corruption == "expected_hash":
        expected = replace(expected, model_sha256="0" * 64)
        code = "E_REAL_MODEL_HASH_MISMATCH"
    elif corruption == "family":
        family = next(value for value in FAMILIES if value != bundle.family)
        expected = replace(expected, family=family)
    else:
        field = "bundle_version" if corruption == "version" else corruption
        identity = replace(
            bundle.identity, **{field: getattr(bundle.identity, field) + "-different"}
        )
        expected = replace(expected, identity=identity)
    guard_artifact_runtime(monkeypatch)

    def forbidden_deserialization(*args, **kwargs):
        pytest.fail("Hash and expected identity mismatches must fail before deserialization")

    monkeypatch.setattr(models.joblib, "load", forbidden_deserialization)
    with pytest.raises(PreFeatureInputError, match=code):
        runner.load_retrospective_bundle(path, expected=expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected", {}),
        ("identity", {}),
        ("identity.dataset_id", 12),
        ("identity.split_id", ""),
        ("identity.bundle_version", None),
        ("model_sha256", 12),
        ("model_sha256", "not-a-sha256"),
        ("family", []),
        ("family", "unsupported"),
    ),
)
def test_artifact_loader_rejects_malformed_expectation(
    tmp_path, monkeypatch, artifact_bundle_factory, field, value
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    expected = artifact_expectation(bundle, metadata)
    if field == "expected":
        expected = value
    elif field.startswith("identity."):
        identity = replace(bundle.identity, **{field.split(".")[1]: value})
        expected = replace(expected, identity=identity)
    else:
        expected = replace(expected, **{field: value})
    guard_artifact_runtime(monkeypatch)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Malformed expectations must fail before deserialization")

    monkeypatch.setattr(runner, "load_bundle", forbidden_load)
    with pytest.raises(PreFeatureInputError, match="E_REAL_MODEL_IDENTITY_MISMATCH"):
        runner.load_retrospective_bundle(path, expected=expected)


@pytest.mark.parametrize(
    ("corruption", "code"),
    (
        ("missing_metadata", "E_REAL_MODEL_METADATA_MISSING"),
        ("corrupt_metadata", "E_REAL_MODEL_METADATA_INVALID"),
        ("missing_model", "E_MODEL_BUNDLE_INVALID"),
        ("corrupt_model", "E_MODEL_BUNDLE_INVALID"),
    ),
)
def test_artifact_loader_missing_or_corrupt_pair(
    tmp_path, monkeypatch, artifact_bundle_factory, corruption, code
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    if corruption == "missing_metadata":
        path.with_suffix(".json").unlink()
    elif corruption == "corrupt_metadata":
        path.with_suffix(".json").write_bytes(b"{not-valid-json")
    elif corruption == "missing_model":
        path.unlink()
    else:
        path.write_bytes(b"not-a-joblib-model")
        metadata["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_artifact_metadata(path, metadata)
    before = artifact_inventory(tmp_path)
    guard_artifact_runtime(monkeypatch)

    with pytest.raises(PreFeatureInputError, match=code) as caught:
        runner.load_retrospective_bundle(path)
    if corruption in ("missing_model", "corrupt_model"):
        assert isinstance(caught.value, ModelInputError)
    assert artifact_inventory(tmp_path) == before


@pytest.mark.parametrize("phase", ("pre", "post"))
@pytest.mark.parametrize(
    "corruption",
    (
        "none", "wrong_pipeline", "unfitted_features", "unfitted_model",
        "wrong_features", "wrong_estimator", "column_order", "classes", "duplicate_step",
    ),
)
def test_artifact_loader_rejects_invalid_fitted_pipeline(
    tmp_path, monkeypatch, artifact_bundle_factory, phase, corruption
):
    _dataset, _split, original = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, original)
    bundle = deepcopy(original)
    field = f"{phase}_pipeline"
    pipeline = getattr(bundle, field)
    if corruption in ("none", "wrong_pipeline"):
        pipeline = None if corruption == "none" else SimpleNamespace()
        bundle = replace(bundle, **{field: pipeline})
    elif corruption == "unfitted_features":
        pipeline.set_params(features=clone(pipeline.named_steps["features"]))
    elif corruption == "unfitted_model":
        pipeline.set_params(model=clone(pipeline.named_steps["model"]))
    elif corruption == "wrong_features":
        pipeline.set_params(features=models.StandardScaler())
    elif corruption == "wrong_estimator":
        pipeline.set_params(model=models.StandardScaler())
    elif corruption == "column_order":
        transformer = pipeline.named_steps["features"]
        transformer.feature_names_in_ = transformer.feature_names_in_[::-1].copy()
    elif corruption == "duplicate_step":
        pipeline.steps.insert(1, pipeline.steps[0])
    else:
        pipeline.named_steps["model"].classes_ = np.array([0, 2])
    # Deliberately serialize an invalid fixture without weakening save_bundle's gate.
    models.joblib.dump(bundle, path, compress=3)
    metadata["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_artifact_metadata(path, metadata)
    guard_artifact_runtime(monkeypatch)

    with pytest.raises(ModelInputError, match="E_MODEL_BUNDLE_INVALID|E_MODEL_CLASSES_INVALID"):
        runner.load_retrospective_bundle(path)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_pre_column", "reordered_pre_columns", "missing_post_column",
        "duplicate_post_column", "mismatched_games", "changed_pre_foundation",
        "invalid_context", "nonfinite_feature",
    ),
)
def test_artifact_loader_prediction_rejects_invalid_input(
    tmp_path, monkeypatch, artifact_bundle_factory, corruption
):
    dataset, split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, _metadata = write_artifact_pair(tmp_path, bundle)
    ids = list(split.test)
    pre = dataset.X_pre.loc[ids].copy(deep=True)
    post = dataset.X_post.loc[ids].copy(deep=True)
    context = dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)].copy(deep=True)
    if corruption == "missing_pre_column":
        pre = pre.iloc[:, 1:]
    elif corruption == "reordered_pre_columns":
        pre = pre.iloc[:, ::-1]
    elif corruption == "missing_post_column":
        post = post.iloc[:, :-1]
    elif corruption == "duplicate_post_column":
        post = pd.concat([post, post.iloc[:, -1:]], axis=1)
    elif corruption == "mismatched_games":
        post = post.iloc[::-1]
    elif corruption == "changed_pre_foundation":
        post.at[ids[0], "blue_recent_form_value"] = 0.99
    elif corruption == "invalid_context":
        context = context.iloc[:, 1:]
    else:
        pre.at[ids[0], "blue_recent_form_value"] = float("inf")
        post.at[ids[0], "blue_recent_form_value"] = float("inf")
    guard_artifact_runtime(monkeypatch)
    restored = runner.load_retrospective_bundle(path)

    with pytest.raises(
        ModelInputError, match="E_MODEL_ALIGNMENT|E_MODEL_SCHEMA|E_MODEL_VALUE_INVALID"
    ):
        predict_paired(restored, pre, post, context)


def test_artifact_loader_strict_rejects_retrospective(
    tmp_path, monkeypatch, artifact_bundle_factory
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, _metadata = write_artifact_pair(tmp_path, bundle)
    guard_artifact_runtime(monkeypatch)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Strict loader must reject a retrospective sidecar before deserialization")

    monkeypatch.setattr(runner, "load_bundle", forbidden_load)
    with pytest.raises(PreFeatureInputError, match="E_REAL_MODEL_METADATA_INVALID"):
        runner.load_real_bundle(path)


@pytest.mark.parametrize("protocol", (STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL))
def test_artifact_loader_never_refits_or_rewrites_artifacts(
    tmp_path, monkeypatch, artifact_bundle_factory, protocol
):
    dataset, split, bundle = artifact_bundle_factory(protocol)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    expected = artifact_expectation(bundle, metadata)
    before = artifact_inventory(tmp_path)
    ids = list(split.test)
    pre = dataset.X_pre.loc[ids].copy(deep=True)
    post = dataset.X_post.loc[ids].copy(deep=True)
    context = dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)].copy(deep=True)
    pre_before, post_before, context_before = deepcopy((pre, post, context))
    guard_artifact_runtime(monkeypatch)

    restored = artifact_loader(protocol)(path, expected=expected)
    states = [
        preprocessing_state(pipe)
        for pipe in (restored.pre_pipeline, restored.post_pipeline)
    ]
    predictions = predict_paired(restored, pre, post, context)
    assert predictions == predict_paired(restored, pre, post, context)
    for pipeline, state in zip(
        (restored.pre_pipeline, restored.post_pipeline), states, strict=True
    ):
        after = preprocessing_state(pipeline)
        for key in state:
            np.testing.assert_array_equal(after[key], state[key])
    assert_frame_equal(pre, pre_before)
    assert_frame_equal(post, post_before)
    assert_frame_equal(context, context_before)
    assert artifact_inventory(tmp_path) == before


def test_artifact_loader_deserializes_the_verified_buffer_without_reopening(
    tmp_path, monkeypatch, artifact_bundle_factory
):
    _dataset, _split, bundle = artifact_bundle_factory(RETROSPECTIVE_PROTOCOL)
    path, metadata = write_artifact_pair(tmp_path, bundle)
    saved_bytes = path.read_bytes()
    original_open = Path.open
    original_load = models.joblib.load
    reads = []

    def read_once(source, *args, **kwargs):
        if source == path:
            reads.append(source)
            assert len(reads) == 1, "The verified model path must not be reopened"
        return original_open(source, *args, **kwargs)

    def load_verified_buffer(source, *args, **kwargs):
        assert isinstance(source, BytesIO)
        assert source.getvalue() == saved_bytes
        return original_load(source, *args, **kwargs)

    guard_artifact_runtime(monkeypatch)
    monkeypatch.setattr(Path, "open", read_once)
    monkeypatch.setattr(models.joblib, "load", load_verified_buffer)
    restored = load_bundle(path, expected_sha256=metadata["model_sha256"])
    assert restored.identity == bundle.identity
    assert len(reads) == 1
