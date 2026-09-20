"""Three fitted models on one genuine interactive PRE/POST foundation.

Supplemental predictions are immutable JSON documents for presentation/persistence;
they never replace the canonical evaluation snapshots or their training identity.
"""

import json
from copy import deepcopy
from dataclasses import replace

from match_insight.ml.comparison import (
    DEFAULT_COMPARISON_MANIFEST,
    load_comparison_models,
    validate_comparison_models,
)
from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL
from match_insight.ml.models import (
    INTERACTIVE_INFERENCE_POLICY,
    PhasePrediction,
    _digest,
    _plain,
    compare_predictions,
    predict_phase,
)
from match_insight.services.evaluation import (
    EvaluationInputError,
    _prediction_inputs,
    _require_snapshot,
    compare_interactive_evaluations,
)

SCHEMA_VERSION = "interactive-model-comparison-v1"
MODEL_LABELS = {
    "baseline": "Baseline",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def load_models(canonical_bundle, *, manifest_path=DEFAULT_COMPARISON_MANIFEST):
    """Load the explicitly supplied three-model manifest once at application init."""
    return load_comparison_models(canonical_bundle, manifest_path=manifest_path)


def _fail(detail):
    raise EvaluationInputError("E_EVAL_INCOMPATIBLE", detail)


def _model_metadata(bundle):
    return {
        "family": bundle.family,
        "dataset_id": bundle.identity.dataset_id,
        "split_id": bundle.identity.split_id,
        "bundle_version": bundle.identity.bundle_version,
        "fit_signature": bundle.fit_signature,
        "feature_schema_version": bundle.feature_schema_version,
        "preprocessing_version": bundle.preprocessing_version,
        "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
    }


def _base_document(pre, model_set, phase):
    context = pre.request.runtime_context
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "analysis_id": context.analysis_id,
        "inference_mode": context.inference_mode,
        "inference_policy": context.policy_version,
        "context_key": pre.context_key,
        "pre_seal": pre.seal,
        "history_cutoff_at": context.history_cutoff_at.isoformat(),
        "runtime_context": _plain(context),
        "model_set_id": model_set.model_set_id,
    }


def _phase_document(prediction):
    # Reuse the core probability/domain checks, without invoking inference.
    if not isinstance(prediction, PhasePrediction) or prediction.phase not in ("PRE", "POST"):
        _fail("Expected a model phase prediction")
    compare_predictions(
        replace(prediction, phase="PRE"), replace(prediction, phase="POST"),
    )
    return {
        "blue_win_probability": float(prediction.p_blue_win),
        "red_win_probability": float(1 - prediction.p_blue_win),
        "prediction": _plain(prediction),
    }


def _require_prediction(prediction, pre, bundle, phase):
    if not isinstance(prediction, PhasePrediction):
        _fail("Expected a saved model prediction")
    expected = replace(
        pre.prediction, phase=phase, p_blue_win=prediction.p_blue_win,
        model_bundle_version=bundle.identity.bundle_version,
        bundle_signature=bundle.fit_signature,
        feature_schema_version=bundle.feature_schema_version,
    )
    if prediction != expected:
        _fail("Comparison prediction differs from the shared PRE context")
    return _phase_document(prediction)


def _seal(document):
    # As with the evaluation seal, this binds content; it is not source attestation.
    json.dumps(document, allow_nan=False)
    return {**document, "seal": _digest(document)}


def _checked_pre_document(document, pre, canonical_bundle, model_set):
    """Verify the retained PRE document without recomputing any PRE probability."""
    if not isinstance(document, dict):
        _fail("Create three-model PRE predictions before POST")
    try:
        body = {key: value for key, value in document.items() if key != "seal"}
        if document.get("seal") != _digest(body):
            _fail("The retained PRE comparison document changed")
        rows = document["models"]
        if not isinstance(rows, list) or len(rows) != len(model_set.bundles):
            _fail("Expected exactly three ordered PRE model predictions")
        expected_rows, predictions = [], []
        for row, bundle in zip(rows, model_set.bundles, strict=True):
            probability = row["pre"]["blue_win_probability"]
            prediction = replace(
                pre.prediction, p_blue_win=probability,
                model_bundle_version=bundle.identity.bundle_version,
                bundle_signature=bundle.fit_signature,
                feature_schema_version=bundle.feature_schema_version,
            )
            if bundle is canonical_bundle and prediction != pre.prediction:
                _fail("Canonical PRE probability must remain unchanged")
            expected_rows.append({
                "family": bundle.family,
                "label": MODEL_LABELS[bundle.family],
                "model": _model_metadata(bundle),
                "pre": _require_prediction(prediction, pre, bundle, "PRE"),
            })
            predictions.append(prediction)
        expected = _seal({**_base_document(pre, model_set, "PRE"), "models": expected_rows})
        if document != expected:
            _fail("Retained PRE predictions differ from context or model identity")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if isinstance(error, EvaluationInputError):
            raise
        _fail("Invalid retained PRE comparison document")
    return tuple(predictions)


def predict_pre_comparison(pre, canonical_bundle, model_set):
    """Reuse canonical PRE; predict supplemental families from its exact feature frame."""
    validate_comparison_models(model_set, canonical_bundle)
    _require_snapshot(pre, canonical_bundle, inference_policy=INTERACTIVE_INFERENCE_POLICY)
    frame, metadata = _prediction_inputs(pre.request, pre.features, canonical_bundle)
    rows = []
    for bundle in model_set.bundles:
        prediction = pre.prediction if bundle is canonical_bundle else predict_phase(
            bundle, "PRE", frame, metadata, inference_policy=INTERACTIVE_INFERENCE_POLICY,
        )[0]
        rows.append({
            "family": bundle.family,
            "label": MODEL_LABELS[bundle.family],
            "model": _model_metadata(bundle),
            "pre": _require_prediction(prediction, pre, bundle, "PRE"),
        })
    return _seal({**_base_document(pre, model_set, "PRE"), "models": rows})


def predict_post_comparison(pre, post, canonical_bundle, model_set, pre_comparison):
    """Predict POST against retained per-model PRE values; never run PRE inference here."""
    validate_comparison_models(model_set, canonical_bundle)
    compare_interactive_evaluations(pre, post, canonical_bundle)
    pre_predictions = _checked_pre_document(pre_comparison, pre, canonical_bundle, model_set)
    frame, metadata = _prediction_inputs(pre.request, pre.features, canonical_bundle, post.features)
    rows = []
    for bundle, saved_pre, saved_row in zip(
        model_set.bundles, pre_predictions, pre_comparison["models"], strict=True,
    ):
        prediction = post.prediction if bundle is canonical_bundle else predict_phase(
            bundle, "POST", frame, metadata, inference_policy=INTERACTIVE_INFERENCE_POLICY,
        )[0]
        phase = _require_prediction(prediction, pre, bundle, "POST")
        compared = compare_predictions(saved_pre, prediction)
        rows.append({
            **deepcopy(saved_row), "post": phase,
            "comparison": {
                "blue_probability_delta": float(compared.delta_probability),
                "red_probability_delta": float(-compared.delta_probability),
            },
        })
    return _seal({
        **_base_document(pre, model_set, "POST"),
        "pre_document_seal": pre_comparison["seal"],
        "post_snapshot_seal": _digest((
            pre.seal, post.lineup, post.features, post.prediction, post.comparison, post.warnings,
        )),
        "models": rows,
    })
