"""Three-family supplemental artifacts, with synthetic fitting only in fixture setup."""

import hashlib
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from match_insight.ml import comparison, models
from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL
from scripts import prepare_comparison_models as cli


def _publish(environment, tmp_path):
    return comparison.publish_comparison_models(
        environment.models, environment.canonical_bundle, tmp_path / "manifest.json",
    )


def _edit_manifest(path, change):
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")


def _forbidden(*args, **kwargs):
    pytest.fail("Runtime loading/prediction must not fit or save")


def test_candidates_preserve_canonical_and_original_training_recipe(comparison_training_fixture):
    env = comparison_training_fixture
    result = comparison.validate_comparison_models(env.models, env.canonical_bundle)
    assert result is env.models
    assert tuple(row.family for row in result.bundles) == models.FAMILIES
    assert result.bundles[1] is env.canonical_bundle
    assert len({row.identity.bundle_version for row in result.bundles}) == 3
    for row in result.bundles:
        assert row.identity.dataset_id == env.canonical_bundle.identity.dataset_id
        assert row.identity.split_id == env.canonical_bundle.identity.split_id
        assert row.config == env.canonical_bundle.config
        assert row.selection_policy == env.canonical_bundle.selection_policy
        assert row.split == env.split
        assert row.contract == env.canonical_bundle.contract
        assert row.pre_columns == models.PRE_COLUMNS and len(row.pre_columns) == 26
        assert row.post_columns == models.POST_COLUMNS and len(row.post_columns) == 76
        assert row.preprocessing_version == models.PREPROCESSING_VERSION


def test_existing_selected_api_still_uses_same_selection_policy(
    monkeypatch, comparison_training_fixture,
):
    env = comparison_training_fixture
    monkeypatch.setattr(models, "fit_model_candidates", lambda *args, **kwargs: env.models.bundles)
    result = models.fit_model_bundle(
        env.dataset, env.split, env.canonical_bundle.identity,
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    assert result.family == models.choose_family(
        env.canonical_bundle.validation_scores, env.canonical_bundle.selection_policy,
    )
    assert result is next(row for row in env.models.bundles if row.family == result.family)


def test_save_load_predict_no_fit_or_resave(monkeypatch, comparison_training_fixture, tmp_path):
    env = comparison_training_fixture
    path = _publish(env, tmp_path)
    monkeypatch.setattr(Pipeline, "fit", _forbidden)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", _forbidden)
    monkeypatch.setattr(models, "save_bundle", _forbidden)
    monkeypatch.setattr(comparison, "save_bundle", _forbidden)
    loaded = comparison.load_comparison_models(env.canonical_bundle, path)
    assert loaded.model_set_id == env.models.model_set_id
    assert loaded.bundles[1] is env.canonical_bundle
    assert {file.name for file in tmp_path.iterdir()} == {
        "manifest.json", "baseline_pre_post.joblib", "random_forest_pre_post.joblib",
    }
    ids = list(env.split.validation)
    metadata = env.dataset.metadata.loc[ids, list(models.CONTEXT_COLUMNS)]
    expected_prior = float(env.dataset.y.loc[list(env.split.train)].mean())
    for original, bundle in zip(env.models.bundles, loaded.bundles, strict=True):
        for phase in ("PRE", "POST"):
            frame = getattr(env.dataset, f"X_{phase.lower()}").loc[ids]
            expected = models.predict_phase(original, phase, frame, metadata)
            actual = models.predict_phase(bundle, phase, frame, metadata)
            assert actual == expected == models.predict_phase(bundle, phase, frame, metadata)
            for prediction in actual:
                assert math.isfinite(prediction.p_blue_win) and 0 <= prediction.p_blue_win <= 1
                if bundle.family == "baseline":
                    assert prediction.p_blue_win == expected_prior
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["canonical_selected_family"] == "logistic_regression"
    assert [row["role"] for row in document["candidates"]] == ["SUPPLEMENTAL_CANDIDATE"] * 2
    assert "test_metrics" not in document


@pytest.mark.parametrize("field,value", [
    ("artifact_schema", "wrong"), ("data_kind", "SYNTHETIC"),
    ("evaluation_protocol", "STRICT_PRE"), ("canonical_selected_family", "random_forest"),
    ("model_set_id", "comparison:wrong"),
])
def test_manifest_contract_rejected(comparison_training_fixture, tmp_path, field, value):
    env = comparison_training_fixture
    path = _publish(env, tmp_path)
    _edit_manifest(path, lambda document: document.__setitem__(field, value))
    with pytest.raises(models.ModelInputError):
        comparison.load_comparison_models(env.canonical_bundle, path)


@pytest.mark.parametrize("field", [
    "pre_columns", "post_columns", "preprocessing_version", "library_versions", "identity",
])
def test_candidate_sidecar_mismatch_rejected(comparison_training_fixture, tmp_path, field):
    env = comparison_training_fixture
    path = _publish(env, tmp_path)

    def change(document):
        saved = document["candidates"][0]["bundle"]
        saved[field] = list(reversed(saved[field])) if field.endswith("columns") else "wrong"

    _edit_manifest(path, change)
    with pytest.raises(models.ModelInputError):
        comparison.load_comparison_models(env.canonical_bundle, path)


@pytest.mark.parametrize("damage", ["hash", "bytes", "missing", "null_hash", "non_hex_hash"])
def test_bad_artifact_rejected_before_deserialization(
    monkeypatch, comparison_training_fixture, tmp_path, damage,
):
    env = comparison_training_fixture
    path = _publish(env, tmp_path)
    artifact = tmp_path / "baseline_pre_post.joblib"
    if damage == "hash":
        _edit_manifest(path, lambda row: row["candidates"][0].__setitem__("sha256", "0" * 64))
    elif damage == "null_hash":
        _edit_manifest(path, lambda row: row["candidates"][0].__setitem__("sha256", None))
    elif damage == "non_hex_hash":
        _edit_manifest(path, lambda row: row["candidates"][0].__setitem__("sha256", "g" * 64))
    elif damage == "bytes":
        artifact.write_bytes(b"corrupt")
    else:
        artifact.unlink()
    monkeypatch.setattr(models.joblib, "load", _forbidden)
    with pytest.raises(models.ModelInputError):
        comparison.load_comparison_models(env.canonical_bundle, path)


@pytest.mark.parametrize("change", [
    lambda row: row["candidates"].reverse(),
    lambda row: row["candidates"].append(row["candidates"][0]),
    lambda row: row["candidates"][0].__setitem__("filename", "../baseline_pre_post.joblib"),
    lambda row: row["canonical_bundle"]["identity"].__setitem__("split_id", "wrong"),
])
def test_family_order_duplicates_path_and_canonical_pin_rejected(
    comparison_training_fixture, tmp_path, change,
):
    env = comparison_training_fixture
    path = _publish(env, tmp_path)
    _edit_manifest(path, change)
    with pytest.raises(models.ModelInputError):
        comparison.load_comparison_models(env.canonical_bundle, path)


def test_missing_manifest_does_not_fallback(comparison_training_fixture, tmp_path):
    with pytest.raises(models.ModelInputError):
        comparison.load_comparison_models(
            comparison_training_fixture.canonical_bundle, tmp_path / "manifest.json",
        )


def test_publish_never_overwrites_existing_artifacts(comparison_training_fixture, tmp_path):
    env = comparison_training_fixture
    _publish(env, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(models.ModelInputError, match="overwrite"):
        _publish(env, tmp_path)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("change", [
    lambda env: replace(env.models, bundles=tuple(reversed(env.models.bundles))),
    lambda env: replace(env.models, model_set_id="comparison:wrong"),
    lambda env: replace(env.models, bundles=(
        replace(env.models.bundles[0], identity=env.canonical_bundle.identity),
        env.canonical_bundle, env.models.bundles[2],
    )),
    lambda env: replace(env.models, bundles=(
        env.models.bundles[0], replace(env.canonical_bundle), env.models.bundles[2],
    )),
])
def test_runtime_cannot_mix_or_relabel_bundle_set(comparison_training_fixture, change):
    env = comparison_training_fixture
    with pytest.raises(models.ModelInputError):
        comparison.validate_comparison_models(change(env), env.canonical_bundle)


def test_changed_dataset_rejected_before_any_fit(monkeypatch, comparison_training_fixture):
    env = comparison_training_fixture
    monkeypatch.setattr(comparison, "fit_model_candidates", _forbidden)
    changed = replace(env.dataset, y=1 - env.dataset.y)
    with pytest.raises(models.ModelInputError):
        comparison.fit_comparison_models(changed, env.split, env.canonical_bundle)


def test_cli_requires_reviewed_identity_before_io(monkeypatch):
    monkeypatch.setattr(cli, "load_evidence", _forbidden)
    monkeypatch.setattr(cli, "_runtime_snapshot", _forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--apply", "--expected-dataset-id", "wrong", "--expected-split-id", "wrong"])
    assert error.value.code == 2


def test_cli_dry_run_never_fits_or_publishes(
    monkeypatch, comparison_training_fixture, tmp_path, capsys,
):
    env = comparison_training_fixture
    evidence = tmp_path / "pinned-input.json"
    data = b'{"source":"synthetic test only"}'
    evidence.write_bytes(data)
    monkeypatch.setattr(cli, "RETROSPECTIVE_INPUT", evidence)
    monkeypatch.setattr(cli, "CANONICAL_DEMO_INPUT_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(cli, "load_retrospective_bundle", lambda *args, **kwargs: env.canonical_bundle)
    monkeypatch.setattr(cli, "load_evidence", lambda *args, **kwargs: object())
    reads = []

    def snapshot():
        reads.append(1)
        return SimpleNamespace(sha256=cli.CANONICAL_DEMO_SNAPSHOT_SHA256)

    monkeypatch.setattr(cli, "_runtime_snapshot", snapshot)
    monkeypatch.setattr(cli, "prepare_run", lambda *args, **kwargs: SimpleNamespace(
        dataset=env.dataset, split=env.split, report={
            "status": "READY", "dataset_id": env.canonical_bundle.identity.dataset_id,
            "split_id": env.canonical_bundle.identity.split_id,
            "partition_counts": {name: len(getattr(env.split, name))
                                 for name in ("train", "validation", "test")},
        },
    ))
    monkeypatch.setattr(cli, "fit_comparison_models", _forbidden)
    monkeypatch.setattr(cli, "publish_comparison_models", _forbidden)
    output = tmp_path / "not-created"
    assert cli.main(["--output-directory", str(output)]) == 0
    assert reads == [1] and not output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "DRY_RUN_READY"
    assert report["test_evaluation"] == "NOT_RUN"


def test_candidate_identities_rejected_before_fit(monkeypatch, comparison_training_fixture):
    env = comparison_training_fixture
    monkeypatch.setattr(Pipeline, "fit", _forbidden)
    with pytest.raises(models.ModelInputError, match="identity"):
        models.fit_model_candidates(
            env.dataset, env.split, env.canonical_bundle.identity,
            evaluation_protocol=RETROSPECTIVE_PROTOCOL, candidate_identities={},
        )
