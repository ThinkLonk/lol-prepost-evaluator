"""Synthetic import reconstruction only; no PostgreSQL or canonical local artifact."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from test_evaluation_service import retrospective_bundle as retrospective_bundle

import match_insight.ml.models as models
import match_insight.services.evaluation as evaluation
import match_insight.services.import_evaluations as importer
import scripts.import_evaluations as cli
from match_insight.database.real_snapshot import DatabaseGame
from match_insight.features.post import FinalLineup
from match_insight.features.pre import HistoricalGame, PreFeatureInputError
from match_insight.ml.dataset import CutoffRecord, CutoffVerification
from match_insight.services.demo import make_demo_cases
from match_insight.services.evaluation import create_post, create_pre


@pytest.fixture
def import_fixture(retrospective_bundle, monkeypatch):
    request = make_demo_cases()[0].request
    target = request.target
    cutoff = datetime(2025, 8, 1, tzinfo=UTC)
    game = HistoricalGame(
        target.game_id, target.blue_team_id, target.red_team_id,
        target.blue_roster, target.red_roster, "A", cutoff + timedelta(days=1, hours=12),
    )
    lineup = FinalLineup(target.game_id, target.patch, request.history[0].slots)
    source = (
        DatabaseGame(game, lineup, cutoff + timedelta(days=1, hours=11)),
        CutoffRecord(target.game_id, cutoff, "synthetic://import/cutoff",
                     importer.RETROSPECTIVE_PROTOCOL, CutoffVerification.PROTOCOL_ASSUMED),
        cutoff + timedelta(days=1, hours=11), game.ended_at,
        cutoff + timedelta(days=2),
    )
    pre, post = importer._reconstruct_pair(
        source, request.history, request.champion_reference, retrospective_bundle,
    )
    expected = models._plain(post.comparison)
    snapshot = SimpleNamespace(champion_reference=request.champion_reference, sha256="a" * 64)
    document = {"time_proofs": {target.game_id: {"source": "synthetic://import/proof"}}}
    prepared = (
        retrospective_bundle, snapshot, document, {target.game_id: expected},
        {target.game_id: {"source": "synthetic://import/context"}},
        {target.game_id: source}, request.history,
    )
    monkeypatch.setattr(importer, "_prepare_import", lambda *args: prepared)
    monkeypatch.setattr(importer, "_database_counts", lambda *args: {
        "evaluations": 0, "warnings": 0,
    })

    def forbidden(*args, **kwargs):
        pytest.fail("Import runtime must not fit or save a model")

    monkeypatch.setattr(models, "fit_model_bundle", forbidden)
    monkeypatch.setattr(models, "save_bundle", forbidden)
    return prepared, pre, post


def test_actual_core_reconstruction_preserves_context_features_and_predictions(import_fixture):
    prepared, pre, post = import_fixture
    assert post.pre is pre
    assert post.features.pre is pre.features
    assert pre.prediction.context_signature == post.prediction.context_signature
    assert pre.prediction.pre_feature_signature == post.prediction.pre_feature_signature
    assert post.comparison.policy_version == importer.RETROSPECTIVE_PROTOCOL
    importer._check_prediction(post, prepared[3][pre.request.target.game_id])
    assert json.loads(json.dumps(models._plain(post.comparison), allow_nan=False))
    legacy_pre = create_pre(pre.request, prepared[0],
                            evaluation_protocol=importer.RETROSPECTIVE_PROTOCOL)
    legacy_post = create_post(legacy_pre, post.lineup, prepared[0],
                              evaluation_protocol=importer.RETROSPECTIVE_PROTOCOL)
    assert pre == legacy_pre
    assert post == legacy_post


def test_dry_run_reconstructs_without_constructing_a_writer(import_fixture, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run constructed a persistence writer")

    monkeypatch.setattr(importer, "EvaluationPersistence", forbidden)
    report = importer.import_retrospective_evaluations(object())
    assert report["status"] == "DRY_RUN_READY"
    assert report["validated_pairs"] == report["expected_pairs"] == 1
    assert report["created_evaluations"] == 0
    assert report["rejected"] == []


def test_apply_keeps_original_prediction_provenance_and_is_idempotent(import_fixture, monkeypatch):
    saved = {}
    observed = []

    class Store:
        def __init__(self, engine):
            pass

        def save_pair(self, pre, post, bundle, *, provenance):
            observed.append(deepcopy(provenance))
            key = pre.context_key
            created = key not in saved
            saved[key] = provenance
            return tuple(SimpleNamespace(evaluation_id=index, created=created, is_active=True)
                         for index in (1, 2))

    monkeypatch.setattr(importer, "EvaluationPersistence", Store)
    first = importer.import_retrospective_evaluations(object(), apply=True)
    second = importer.import_retrospective_evaluations(object(), apply=True)
    assert first["created_evaluations"] == 2
    assert second["created_evaluations"] == 0
    assert second["existing_evaluations"] == 2
    assert observed[0] == observed[1]
    assert observed[0]["reconstructed_from_existing_sources"] is True
    assert observed[0]["original_inference_time_status"] == "NOT_RECORDED"
    assert "inferred_at" not in observed[0]


@pytest.mark.parametrize("field", ["p_blue_win_pre", "context_signature", "history_cutoff_at"])
def test_reconstruction_mismatch_is_rejected_before_pair_write(import_fixture, monkeypatch, field):
    prepared, pre, _post = import_fixture
    wanted = prepared[3][pre.request.target.game_id]
    wanted[field] = {
        "p_blue_win_pre": 0.99,
        "context_signature": "b" * 64,
        "history_cutoff_at": "2025-08-02T00:00:00+00:00",
    }[field]

    class Store:
        def __init__(self, engine):
            pass

        def save_pair(self, *args, **kwargs):
            pytest.fail("Mismatched pair reached persistence")

    monkeypatch.setattr(importer, "EvaluationPersistence", Store)
    report = importer.import_retrospective_evaluations(object(), apply=True)
    assert report["status"] == "PARTIAL_IMPORT"
    assert report["created_evaluations"] == 0
    assert report["rejected"] == [{"game_id": pre.request.target.game_id,
                                   "reason": "E_EVAL_INCOMPATIBLE"}]


def test_equivalent_timezone_is_compared_as_instant(import_fixture):
    prepared, pre, post = import_fixture
    expected = deepcopy(prepared[3][pre.request.target.game_id])
    expected["history_cutoff_at"] = "2025-08-01T07:00:00+07:00"
    importer._check_prediction(post, expected)


def test_source_hash_and_duplicate_json_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    path.write_text('{"run": {}, "run": {}}', encoding="utf-8")
    with pytest.raises(PreFeatureInputError, match="E_EVIDENCE_HASH_MISMATCH"):
        importer._read_source(path)
    monkeypatch.setattr(importer, "CANONICAL_METADATA_SHA256",
                        hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(PreFeatureInputError, match="E_EVIDENCE_JSON_INVALID"):
        importer._read_source(path)


def test_duplicate_and_mismatched_membership_reject(import_fixture):
    prepared, pre, _post = import_fixture
    row = prepared[3][pre.request.target.game_id]
    with pytest.raises(PreFeatureInputError, match="E_SOURCE_CONFLICT"):
        importer._index_rows([row, dict(row)], code="E_MODEL_METADATA_INVALID")
    with pytest.raises(PreFeatureInputError, match="E_MODEL_ALIGNMENT"):
        importer._source_predictions({"run": {"test_predictions": [row]}}, prepared[0])


def test_changed_probability_outside_tolerance_rejects(import_fixture):
    prepared, pre, post = import_fixture
    expected = deepcopy(prepared[3][pre.request.target.game_id])
    expected["p_blue_win_post"] += 1e-8
    with pytest.raises(PreFeatureInputError, match="E_EVAL_INCOMPATIBLE"):
        importer._check_prediction(post, expected)
    changed = replace(post, comparison=replace(post.comparison, delta_probability=float("inf")))
    with pytest.raises(PreFeatureInputError):
        importer._check_prediction(changed, prepared[3][pre.request.target.game_id])


@pytest.mark.parametrize("corruption", [None, "features", "warning", "parent"])
def test_retry_checks_full_stored_packet_and_warning_content(import_fixture, monkeypatch, corruption):
    prepared, pre, post = import_fixture
    bundle, snapshot, document, predictions, _provenance, _sources, _history = prepared
    game_id = pre.request.target.game_id
    target = pre.request.target
    provenance = importer._provenance(
        game_id, 0, predictions[game_id], {"pre_context_sha256": models._digest((
            game_id, target.patch, ("BLUE", target.blue_team_id, target.blue_roster),
            ("RED", target.red_team_id, target.red_roster),
        ))}, document, snapshot, importer.DEFAULT_METADATA,
    )
    monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION",
                        SimpleNamespace(identity=bundle.identity, family=bundle.family))
    store = importer.EvaluationPersistence(object())
    rows = []
    warnings = []
    for index, phase in enumerate((pre, post), start=1):
        packet, _history, actual_warnings = store._packet(
            phase, bundle, "RETROSPECTIVE_IMPORT", provenance,
        )
        packet.update(evaluation_id=index, is_active=True,
                      pre_evaluation_id=None if index == 1 else 1)
        rows.append(packet)
        warnings.extend({**warning, "evaluation_id": index, "position": position}
                        for position, warning in enumerate(actual_warnings))
    if corruption == "features":
        rows[0]["input_snapshot"]["feature_values"][0] = 0.123
    elif corruption == "warning":
        assert warnings
        warnings[0]["message"] = "invented warning"
    elif corruption == "parent":
        rows[1]["pre_evaluation_id"] = 999
    if corruption is None:
        saved, count = importer._check_stored_pair(
            rows, warnings, predictions[game_id], provenance, bundle, rows[0]["history_sha256"],
        )
        assert [item.evaluation_id for item in saved] == [1, 2]
        assert count == len(warnings)
        assert all(not item.created for item in saved)
    else:
        with pytest.raises(PreFeatureInputError, match="E_STORAGE_CONFLICT"):
            importer._check_stored_pair(
                rows, warnings, predictions[game_id], provenance, bundle,
                rows[0]["history_sha256"],
            )


def test_cli_disposes_engine_and_sanitizes_failures(monkeypatch, capsys):
    import sqlalchemy

    import match_insight.config as config

    disposed = []
    engine = SimpleNamespace(dispose=lambda: disposed.append(True))
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *args: engine)
    monkeypatch.setattr(config, "Settings", lambda: SimpleNamespace(database_url="synthetic"))

    def broken(*args, **kwargs):
        raise RuntimeError("DATABASE_URL=secret-must-not-appear")

    monkeypatch.setattr(cli, "import_retrospective_evaluations", broken)
    assert cli.main([]) == 2
    assert disposed == [True]
    output = capsys.readouterr().out
    assert "secret-must-not-appear" not in output
    assert json.loads(output)["reason"] == "E_EVALUATION_IMPORT_FAILED"


@pytest.mark.parametrize("workers", [0, 9, True, 1.5, "2"])
def test_invalid_worker_count_rejects_before_preflight(monkeypatch, workers):
    monkeypatch.setattr(importer, "_prepare_import",
                        lambda *args: pytest.fail("Invalid workers reached source/DB preflight"))
    with pytest.raises(PreFeatureInputError, match="E_PRE_INPUT_INCOMPLETE"):
        importer.import_retrospective_evaluations(object(), workers=workers)


def test_spawned_dry_run_uses_real_core_without_a_database_writer(import_fixture):
    # The child receives only a fitted synthetic fixture and immutable inputs;
    # its dry-run initializer never creates an engine or persistence service.
    sequential = importer.import_retrospective_evaluations(object())
    parallel = importer.import_retrospective_evaluations(object(), workers=2)
    for field in ("status", "validated_pairs", "expected_pairs", "source_warning_count",
                  "created_evaluations", "rejected"):
        assert parallel[field] == sequential[field]
    assert parallel["workers"] == 2


def test_parallel_submission_is_bounded_and_uses_spawn(monkeypatch):
    observed = {"pending": 0, "maximum": 0, "shutdown": False}

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            observed["pending"] -= 1
            return {"position": self.task[0], "game_id": self.task[1], "reason": None}

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 2
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initializer"] is importer._worker_initialize
            assert kwargs["initargs"] == (None, "synthetic-prepared-input")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            observed["shutdown"] = True

        def submit(self, function, task):
            assert function is importer._worker_pair
            observed["pending"] += 1
            observed["maximum"] = max(observed["maximum"], observed["pending"])
            return Future(task)

    def completed(pending, **kwargs):
        assert kwargs["return_when"] == importer.FIRST_COMPLETED
        return {next(reversed(pending))}, set()

    monkeypatch.setattr(importer, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(importer, "wait", completed)
    tasks = [(index, f"synthetic-{index}", {}) for index in range(6)]
    results = list(importer._parallel_results(tasks, engine_url=None,
                                              prepared="synthetic-prepared-input", workers=2))
    assert sorted(row["position"] for row in results) == list(range(6))
    assert observed == {"pending": 0, "maximum": 2, "shutdown": True}


def test_worker_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(importer, "_WORKER_STATE", (None, None))

    def broken(*args):
        raise RuntimeError("postgresql://secret-password")

    monkeypatch.setattr(importer, "_run_pair", broken)
    result = importer._worker_pair((0, "synthetic-id", {}))
    assert result == {"position": 0, "game_id": "synthetic-id",
                      "reason": "E_EVALUATION_IMPORT_FAILED"}
    assert "secret" not in json.dumps(result)
