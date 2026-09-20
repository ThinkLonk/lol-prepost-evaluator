"""Genuine PRE/POST services and three fitted pipelines; synthetic setup only."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from match_insight.features.pre import PreFeatureInputError
from match_insight.ml import models
from match_insight.ml.dataset import POST_COLUMNS, PRE_COLUMNS, RETROSPECTIVE_PROTOCOL
from match_insight.services import demo, model_comparison


def _forbid_fit_and_model_save(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not fit, refit or write a model")

    monkeypatch.setattr(models, "fit_model_bundle", forbidden)
    monkeypatch.setattr(models, "fit_model_candidates", forbidden)
    monkeypatch.setattr(models, "save_bundle", forbidden)
    monkeypatch.setattr(models.joblib, "dump", forbidden)
    for cls in (
        models.Pipeline, models.ColumnTransformer, models.DummyClassifier,
        models.LogisticRegression, models.RandomForestClassifier, models.SimpleImputer,
        models.StandardScaler, models.OneHotEncoder,
    ):
        for method in ("fit", "fit_transform"):
            if hasattr(cls, method):
                monkeypatch.setattr(cls, method, forbidden)


def _pair(environment):
    pre = demo.evaluate_analysis_pre(environment.runtime, environment.selection)
    post = demo.evaluate_analysis_post(environment.runtime, pre, environment.champion_ids)
    return pre, post


def test_three_models_share_real_foundation_without_repredicting_canonical_or_pre(
    monkeypatch, comparison_environment,
):
    env = comparison_environment
    _forbid_fit_and_model_save(monkeypatch)
    pre, post = _pair(env)
    original = model_comparison.predict_phase
    calls = []

    def predict(bundle, phase, frame, metadata, *, inference_policy):
        assert bundle is not env.canonical_bundle
        assert inference_policy == models.INTERACTIVE_INFERENCE_POLICY
        calls.append((bundle.family, phase, frame.copy(deep=True), metadata.copy(deep=True)))
        return original(bundle, phase, frame, metadata, inference_policy=inference_policy)

    monkeypatch.setattr(model_comparison, "predict_phase", predict)
    # Once the canonical snapshots exist, this service may never infer from LR again.
    def forbidden(*args, **kwargs):
        pytest.fail("Canonical predictions must be reused")

    for pipeline in (env.canonical_bundle.pre_pipeline, env.canonical_bundle.post_pipeline):
        monkeypatch.setattr(pipeline, "predict_proba", forbidden)
    before = deepcopy((pre, post))
    pre_result = model_comparison.predict_pre_comparison(pre, env.canonical_bundle, env.models)
    saved_pre = deepcopy(pre_result)
    result = model_comparison.predict_post_comparison(
        pre, post, env.canonical_bundle, env.models, pre_result,
    )
    assert pre_result == saved_pre
    assert (pre, post) == before
    assert [(family, phase) for family, phase, _, _ in calls] == [
        ("baseline", "PRE"), ("random_forest", "PRE"),
        ("baseline", "POST"), ("random_forest", "POST"),
    ]
    assert all(tuple(frame.columns) == (PRE_COLUMNS if phase == "PRE" else POST_COLUMNS)
               for _family, phase, frame, _metadata in calls)
    assert calls[0][2].equals(calls[1][2])
    assert calls[2][2].equals(calls[3][2])
    assert calls[2][2].loc[:, list(PRE_COLUMNS)].equals(calls[0][2])
    assert all(metadata.equals(calls[0][3]) for _, _, _, metadata in calls)
    assert result["analysis_id"] == env.selection.analysis_id
    assert result["inference_mode"] == "INTERACTIVE_ANALYSIS"
    assert result["runtime_context"]["context_source"] == "USER_PROVIDED"
    assert result["runtime_context"]["pre_draft_verification"] == "NOT_VERIFIED"
    assert result["pre_document_seal"] == pre_result["seal"]
    assert result["history_cutoff_at"] == pre.prediction.history_cutoff_at.isoformat()
    assert [row["family"] for row in result["models"]] == list(models.FAMILIES)
    assert result["models"][1]["pre"]["blue_win_probability"] == pre.prediction.p_blue_win
    assert result["models"][1]["post"]["blue_win_probability"] == post.prediction.p_blue_win
    train_rate = float(env.dataset.y.loc[list(env.split.train)].mean())
    assert result["models"][0]["pre"]["blue_win_probability"] == train_rate
    assert result["models"][0]["comparison"]["blue_probability_delta"] == 0
    for before_row, row in zip(pre_result["models"], result["models"], strict=True):
        assert row["pre"] == before_row["pre"] and row["model"] == before_row["model"]
        assert row["model"]["evaluation_protocol"] == RETROSPECTIVE_PROTOCOL
        assert row["model"]["dataset_id"] == env.canonical_bundle.identity.dataset_id
        assert row["model"]["split_id"] == env.canonical_bundle.identity.split_id
        for phase in ("pre", "post"):
            assert 0 <= row[phase]["blue_win_probability"] <= 1
            assert row[phase]["blue_win_probability"] + row[phase]["red_win_probability"] == 1
        delta = row["post"]["blue_win_probability"] - row["pre"]["blue_win_probability"]
        assert row["comparison"] == {
            "blue_probability_delta": delta, "red_probability_delta": -delta,
        }
    json.dumps(result, allow_nan=False)
    assert model_comparison.predict_post_comparison(
        pre, post, env.canonical_bundle, env.models, pre_result,
    ) == result


@pytest.mark.parametrize("tamper", [
    "probability", "phase", "analysis", "cutoff", "context", "models_order", "identity",
    "model_set", "extra", "missing",
])
def test_changed_saved_pre_document_is_rejected_before_post_inference(
    monkeypatch, comparison_environment, tamper,
):
    env = comparison_environment
    _forbid_fit_and_model_save(monkeypatch)
    pre, post = _pair(env)
    document = model_comparison.predict_pre_comparison(pre, env.canonical_bundle, env.models)
    if tamper == "probability":
        document["models"][0]["pre"]["blue_win_probability"] = 0.12345
    elif tamper == "phase":
        document["phase"] = "POST"
    elif tamper == "analysis":
        document["analysis_id"] = "analysis:other"
    elif tamper == "cutoff":
        document["history_cutoff_at"] = "2024-01-01T00:00:00+00:00"
    elif tamper == "context":
        document["runtime_context"]["context_source"] = "RECONSTRUCTED_POSTGAME"
    elif tamper == "models_order":
        document["models"].reverse()
    elif tamper == "identity":
        document["models"][0]["model"]["bundle_version"] = "other"
    elif tamper == "model_set":
        document["model_set_id"] = "other"
    elif tamper == "extra":
        document["ignored"] = True
    else:
        del document["models"][0]["pre"]

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid retained PRE must fail before POST inference")

    monkeypatch.setattr(model_comparison, "predict_phase", forbidden)
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        model_comparison.predict_post_comparison(
            pre, post, env.canonical_bundle, env.models, document,
        )


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_invalid_probability_is_rejected_even_when_document_digest_is_recomputed(
    monkeypatch, comparison_environment, probability,
):
    env = comparison_environment
    _forbid_fit_and_model_save(monkeypatch)
    pre, post = _pair(env)
    document = model_comparison.predict_pre_comparison(pre, env.canonical_bundle, env.models)
    document["models"][0]["pre"]["blue_win_probability"] = probability
    try:
        document["seal"] = models._digest({k: v for k, v in document.items() if k != "seal"})
    except (ValueError, PreFeatureInputError):
        pass
    with pytest.raises(PreFeatureInputError):
        model_comparison.predict_post_comparison(
            pre, post, env.canonical_bundle, env.models, document,
        )


def test_other_pre_or_model_set_cannot_reuse_saved_comparison(monkeypatch, comparison_environment):
    env = comparison_environment
    _forbid_fit_and_model_save(monkeypatch)
    pre, post = _pair(env)
    document = model_comparison.predict_pre_comparison(pre, env.canonical_bundle, env.models)
    other = demo.evaluate_analysis_pre(
        env.runtime, replace(env.selection, analysis_id="analysis:another-request"),
    )
    with pytest.raises(PreFeatureInputError):
        model_comparison.predict_post_comparison(
            other, post, env.canonical_bundle, env.models, document,
        )
    with pytest.raises(PreFeatureInputError):
        model_comparison.predict_pre_comparison(
            pre, env.canonical_bundle, replace(env.models, model_set_id="modified"),
        )


def test_model_loader_boundary_delegates_without_replacing_canonical(monkeypatch):
    canonical, loaded, calls = object(), object(), []

    def load(bundle, *, manifest_path):
        calls.append((bundle, manifest_path))
        return loaded

    monkeypatch.setattr(model_comparison, "load_comparison_models", load)
    assert model_comparison.load_models(canonical, manifest_path="reviewed.json") is loaded
    assert calls == [(canonical, "reviewed.json")]


def test_interactive_service_does_not_accept_retroactive_context(monkeypatch, comparison_environment):
    env = comparison_environment
    _forbid_fit_and_model_save(monkeypatch)
    pre, _ = _pair(env)
    changed = replace(pre, request=replace(pre.request, runtime_context=None))
    with pytest.raises(PreFeatureInputError):
        model_comparison.predict_pre_comparison(changed, env.canonical_bundle, env.models)
