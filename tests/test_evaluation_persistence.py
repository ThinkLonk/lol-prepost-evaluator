"""Synthetic storage tests; PostgreSQL cases require an explicit opt-in scratch schema."""

import importlib.util
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from test_demo_service import _analysis_input
from test_demo_service import analysis_environment as analysis_environment
from test_demo_service import demo_environment as demo_environment
from test_demo_service import synthetic_demo_bundle as synthetic_demo_bundle

from match_insight.database.base import Base
from match_insight.database.evaluations import EvaluationRepository, SavedEvaluation, StorageError
from match_insight.database.models import (
    AnalysisSession,
    Champion,
    Evaluation,
    EvaluationHistory,
    EvaluationWarning,
    Game,
    Player,
    Team,
)
from match_insight.features.pre import PreFeatureInputError
from match_insight.ml import models
from match_insight.ml.dataset import POST_COLUMNS, PRE_COLUMNS, RETROSPECTIVE_PROTOCOL
from match_insight.services import demo, evaluation, persistence
from match_insight.services.persistence import EvaluationPersistence


class ForbiddenEngine:
    def begin(self):
        pytest.fail("Pure persistence preparation must not open a database transaction")

    def connect(self):
        pytest.fail("Pure persistence preparation must not connect to PostgreSQL")


class RecordingRepository:
    def __init__(self):
        self.calls = []

    def save(self, packet, history, warnings, *, pre_evaluation_id=None):
        self.calls.append(deepcopy((packet, history, warnings, pre_evaluation_id)))
        return SavedEvaluation(len(self.calls), True, True)

    def save_pair(self, pre, post, history, pre_warnings, post_warnings):
        self.calls.append(deepcopy((pre, post, history, pre_warnings, post_warnings)))
        return SavedEvaluation(1, True, True), SavedEvaluation(2, True, True)


@pytest.fixture
def interactive_pair(analysis_environment):
    env, runtime, clock = analysis_environment
    selection = _analysis_input()
    pre = demo.evaluate_analysis_pre(runtime, selection)
    champion_ids = tuple(champion.champion_id for champion in demo.CHAMPIONS[:10])
    post = demo.evaluate_analysis_post(runtime, pre, champion_ids)
    return SimpleNamespace(
        env=env, runtime=runtime, clock=clock, selection=selection,
        pre=pre, post=post, champion_ids=champion_ids, bundle=runtime.bundle,
    )


def _forbid_prediction(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Persistence must not predict, train or reconstruct features")

    for owner in (demo, evaluation, models):
        for name in ("predict_phase", "build_pre_features", "build_post_features",
                     "fit_model_bundle", "save_bundle"):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)


def test_interactive_packet_preserves_exact_snapshots_and_full_history(
    interactive_pair, monkeypatch,
):
    pair = interactive_pair
    before = deepcopy((pair.pre, pair.post, pair.bundle.identity, pair.bundle.contract))
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    provenance = {"source": "synthetic://persistence/test-only"}
    packets = [store._packet(snapshot, pair.bundle, "INTERACTIVE", provenance)
               for snapshot in (pair.pre, pair.post)]
    pre, history, pre_warnings = packets[0]
    post, post_history, post_warnings = packets[1]
    context = pair.pre.request.runtime_context

    assert pre["game_id"] is None and post["game_id"] is None
    assert pre["analysis_id"] == post["analysis_id"] == pair.selection.analysis_id
    assert pre["analysis_id"] not in pair.env.document["time_proofs"]
    assert pre["origin"] == post["origin"] == "INTERACTIVE"
    assert pre["inference_mode"] == post["inference_mode"] == "INTERACTIVE_ANALYSIS"
    assert pre["inferred_at"] == context.pre_created_at
    assert post["inferred_at"] is None
    assert post["provenance"]["original_inference_time_status"] == "NOT_RECORDED"
    assert pre["history_cutoff_at"] == post["history_cutoff_at"] == context.history_cutoff_at
    assert pre["data_version"] == post["data_version"] == pair.bundle.identity.dataset_id
    assert pre["model_version"] == post["model_version"] == pair.bundle.identity.bundle_version
    assert pre["context_key"] == post["context_key"] == pair.pre.context_key
    assert pre["history_sha256"] == post["history_sha256"] == persistence._history_hash(history)
    assert history == post_history
    assert history["games"] == persistence._json_value(pair.pre.request.history)
    assert history["champion_reference"] == sorted(pair.pre.request.champion_reference)
    assert len(history["games"]) == len(pair.pre.request.history)
    for packet, snapshot, columns in ((pre, pair.pre, PRE_COLUMNS), (post, pair.post, POST_COLUMNS)):
        document = packet["input_snapshot"]
        assert json.loads(json.dumps(document, allow_nan=False)) == document
        assert tuple(document["feature_columns"]) == columns
        assert len(document["feature_values"]) == len(columns)
        assert document["prediction"] == persistence._json_value(snapshot.prediction)
        assert document["runtime_context"]["context_source"] == "USER_PROVIDED"
        assert document["runtime_context"]["pre_draft_verification"] == "NOT_VERIFIED"
        assert document["model"]["identity"] == persistence._json_value(pair.bundle.identity)
        assert document["pre_seal"] == pair.pre.seal
        assert packet["blue_win_probability"] == snapshot.prediction.p_blue_win
    assert pre["input_snapshot"]["lineup"] is None
    assert pre["input_snapshot"]["comparison"] is None
    assert post["input_snapshot"]["lineup"] == persistence._json_value(pair.post.lineup)
    assert post["input_snapshot"]["comparison"] == evaluation.compare_interactive_evaluations(
        pair.pre, pair.post, pair.bundle,
    )
    assert [row["details"] for row in pre_warnings] == persistence._json_value(pair.pre.warnings)
    assert [row["details"] for row in post_warnings] == persistence._json_value(pair.post.warnings)
    assert packets[0] == store._packet(pair.pre, pair.bundle, "INTERACTIVE", provenance)
    assert pre["idempotency_key"] != post["idempotency_key"]
    assert (pair.pre, pair.post, pair.bundle.identity, pair.bundle.contract) == before
    assert provenance == {"source": "synthetic://persistence/test-only"}


def test_public_save_calls_use_exact_saved_pre_parent(interactive_pair, monkeypatch):
    pair = interactive_pair
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    recording = RecordingRepository()
    store.repository = recording
    pre = store.save_pre(pair.pre, pair.bundle)
    post = store.save_post(pair.post, pair.bundle, pre.evaluation_id)
    assert pre.evaluation_id == 1 and post.evaluation_id == 2
    assert recording.calls[0][3] is None
    assert recording.calls[1][3] == pre.evaluation_id
    assert recording.calls[0][0]["context_key"] == recording.calls[1][0]["context_key"]
    assert recording.calls[0][1] == recording.calls[1][1]


def test_post_operation_identity_is_separate_from_immutable_model_snapshot(
    interactive_pair, monkeypatch,
):
    pair = interactive_pair
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    recording = RecordingRepository()
    store.repository = recording
    first, second = str(uuid4()), str(uuid4())
    provenance = {"source": "synthetic://post-operation"}
    for operation_id in (None, first, first, second):
        store.save_post(pair.post, pair.bundle, 1, provenance=provenance, operation_id=operation_id)
    legacy, a, retry, b = [call[0] for call in recording.calls]
    assert "post_operation_id" not in legacy["input_snapshot"]["provenance"]
    assert a == retry
    assert len({packet["idempotency_key"] for packet in (legacy, a, b)}) == 3
    for packet, identity in ((a, first), (b, second)):
        assert packet["provenance"]["post_operation_id"] == identity
        assert packet["input_snapshot"]["provenance"]["post_operation_id"] == identity
        assert packet["provenance"]["input_snapshot_sha256"] == persistence._history_hash(
            packet["input_snapshot"],
        )
        for field in ("features", "prediction", "comparison", "runtime_context", "model", "lineup"):
            assert packet["input_snapshot"][field] == legacy["input_snapshot"][field]
    assert provenance == {"source": "synthetic://post-operation"}


@pytest.mark.parametrize("operation_id", ["", "not-a-uuid", False, 123,
                                        "00000000-0000-0000-0000-000000000000"])
def test_invalid_post_operation_id_is_rejected_before_storage(interactive_pair, operation_id):
    store = EvaluationPersistence(ForbiddenEngine())
    store.repository = RecordingRepository()
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.save_post(interactive_pair.post, interactive_pair.bundle, 1, operation_id=operation_id)
    assert store.repository.calls == []


def test_conflicting_post_operation_provenance_is_rejected(interactive_pair):
    store = EvaluationPersistence(ForbiddenEngine())
    store.repository = RecordingRepository()
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.save_post(interactive_pair.post, interactive_pair.bundle, 1,
                        operation_id=str(uuid4()), provenance={"post_operation_id": str(uuid4())})
    assert store.repository.calls == []


def test_retrospective_import_keeps_unknown_original_inference_time(demo_environment, monkeypatch):
    env = demo_environment
    target_id = env.runtime.targets[0].game_id
    pre = demo.evaluate_demo_pre(env.runtime, target_id)
    ids = tuple(champion.champion_id for champion in demo.CHAMPIONS[:10])
    post = demo.evaluate_demo_post(env.runtime, pre, ids)
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    recording = RecordingRepository()
    store.repository = recording
    provenance = {"import_kind": "synthetic-fixture", "artifact_sha256": "a" * 64}
    saved = store.save_pair(pre, post, env.runtime.bundle, provenance=provenance)
    assert tuple(item.evaluation_id for item in saved) == (1, 2)
    a, b, history, _wa, _wb = recording.calls[0]
    for packet in (a, b):
        assert packet["origin"] == "RETROSPECTIVE_IMPORT"
        assert packet["inference_mode"] == "REAL_RETROSPECTIVE_SIMULATION"
        assert packet["analysis_id"] is None and packet["game_id"] == target_id
        assert packet["inferred_at"] is None
        assert packet["provenance"]["original_inference_time_status"] == "NOT_RECORDED"
        assert packet["input_snapshot"]["runtime_context"] is None
        assert packet["input_snapshot"]["cutoff_record"]["policy_version"] == RETROSPECTIVE_PROTOCOL
    assert a["history_sha256"] == b["history_sha256"] == persistence._history_hash(history)


def test_retrospective_pair_validates_once_and_preserves_complete_packets(
    demo_environment, monkeypatch,
):
    env = demo_environment
    pre = demo.evaluate_demo_pre(env.runtime, env.runtime.targets[0].game_id)
    post = demo.evaluate_demo_post(
        env.runtime, pre, tuple(champion.champion_id for champion in demo.CHAMPIONS[:10]),
    )
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    recording = RecordingRepository()
    store.repository = recording
    provenance = {"import_kind": "synthetic-fixture", "artifact_sha256": "a" * 64}
    expected_pre = store._packet(pre, env.runtime.bundle, "RETROSPECTIVE_IMPORT", provenance)
    expected_post = store._packet(post, env.runtime.bundle, "RETROSPECTIVE_IMPORT", provenance)
    calls = {"pre": [], "comparison": []}
    original_require = persistence._require_snapshot
    original_compare = persistence.compare_retrospective_evaluations

    def require(*args, **kwargs):
        calls["pre"].append((args, kwargs))
        return original_require(*args, **kwargs)

    def compare(*args, **kwargs):
        calls["comparison"].append((args, kwargs))
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(persistence, "_require_snapshot", require)
    monkeypatch.setattr(persistence, "compare_retrospective_evaluations", compare)
    store.save_pair(pre, post, env.runtime.bundle, provenance=provenance)
    a, b, history, warnings_a, warnings_b = recording.calls[0]
    assert (a, history, warnings_a) == expected_pre
    assert (b, history, warnings_b) == expected_post
    assert len(calls["pre"]) == len(calls["comparison"]) == 1
    assert calls["pre"][0][1] == {"evaluation_protocol": RETROSPECTIVE_PROTOCOL}
    for actual, expected in ((a, expected_pre[0]), (b, expected_post[0])):
        assert json.dumps(persistence._json_value(actual), sort_keys=True, allow_nan=False) == json.dumps(
            persistence._json_value(expected), sort_keys=True, allow_nan=False,
        )


@pytest.mark.parametrize("change", ("pre_seal", "post_parent", "post_probability", "post_lineup"))
def test_retrospective_pair_rejects_tampering_before_repository(
    demo_environment, monkeypatch, change,
):
    env = demo_environment
    pre = demo.evaluate_demo_pre(env.runtime, env.runtime.targets[0].game_id)
    post = demo.evaluate_demo_post(
        env.runtime, pre, tuple(champion.champion_id for champion in demo.CHAMPIONS[:10]),
    )
    if change == "pre_seal":
        pre = replace(pre, seal="0" * 64)
        post = replace(post, pre=pre)
    elif change == "post_parent":
        post = replace(post, pre=replace(pre, context_key="0" * 64))
    elif change == "post_probability":
        post = replace(post, comparison=replace(post.comparison, delta_probability=0.99))
    else:
        post = replace(post, lineup=replace(post.lineup, game_id="different-target"))
    _forbid_prediction(monkeypatch)
    store = EvaluationPersistence(ForbiddenEngine())
    recording = RecordingRepository()
    store.repository = recording
    with pytest.raises(PreFeatureInputError):
        store.save_pair(pre, post, env.runtime.bundle, provenance={"source": "synthetic://tampered"})
    assert recording.calls == []


@pytest.mark.parametrize("change", ["seal", "origin", "cutoff", "post_comparison"])
def test_packet_rejects_forged_or_mixed_snapshots_before_storage(interactive_pair, change):
    pair = interactive_pair
    store = EvaluationPersistence(ForbiddenEngine())
    snapshot, origin = pair.pre, "INTERACTIVE"
    if change == "seal":
        snapshot = replace(pair.pre, seal="0" * 64)
    elif change == "origin":
        origin = "RETROSPECTIVE_IMPORT"
    elif change == "cutoff":
        snapshot = replace(pair.pre, request=replace(pair.pre.request, target=replace(
            pair.pre.request.target,
            history_cutoff_at=pair.pre.prediction.history_cutoff_at + timedelta(seconds=1),
        )))
    else:
        snapshot = replace(pair.post, comparison=replace(pair.post.comparison, delta_probability=0.99))
    with pytest.raises(PreFeatureInputError):
        store._packet(snapshot, pair.bundle, origin, {})


def test_warning_mapping_keeps_missing_zero_counts_and_rejects_unknown_code():
    warnings = (
        evaluation.BusinessWarning("W_TEAM_HISTORY_SMALL", "team-one", 0, 10),
        evaluation.BusinessWarning("W_H2H_MISSING", None, 0, 10),
        evaluation.BusinessWarning("W_PAIR_HISTORY_MISSING", None, 10, 0),
    )
    rows = persistence.warning_rows(warnings)
    assert [row["details"] for row in rows] == persistence._json_value(warnings)
    assert "0/10" in rows[0]["message"]
    assert "10/10" in rows[2]["message"]
    with pytest.raises(StorageError):
        persistence.warning_rows((evaluation.BusinessWarning("UNKNOWN"),))


@pytest.fixture(scope="module")
def storage_database():
    if os.environ.get("MATCH_INSIGHT_STORAGE_DB_TESTS") != "1":
        pytest.skip("PostgreSQL scratch-schema tests require MATCH_INSIGHT_STORAGE_DB_TESTS=1")

    # Importing project configuration is deliberately inside the opt-in boundary.
    env_keys = ("DATABASE_URL", "WIKI_USERNAME_MATCH_INSIGHT", "WIKI_PASSWORD_MATCH_INSIGHT")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    schema = f"evaluation_test_{uuid4().hex}"
    assert re.fullmatch(r"evaluation_test_[0-9a-f]{32}", schema)
    admin = engine = None
    created = False
    try:
        from match_insight.database.engine import engine as project_engine

        admin = sa.create_engine(project_engine.url, poolclass=NullPool)
        engine = sa.create_engine(
            project_engine.url, poolclass=NullPool,
            connect_args={"options": f"-csearch_path={schema}"},
        )
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            created = True
        with engine.begin() as connection:
            assert connection.execute(sa.text("SELECT current_schema()")).scalar_one() == schema
            Base.metadata.create_all(connection)
            path = (Path(__file__).resolve().parents[1] / "alembic" / "versions"
                    / "c6e31a9d4b72_persist_interactive_evaluations.py")
            spec = importlib.util.spec_from_file_location("evaluation_storage_test_migration", path)
            assert spec is not None and spec.loader is not None
            migration = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration)
            with Operations.context(MigrationContext.configure(connection)):
                migration._create_guards()
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        try:
            if created:
                # Only the unique schema owned by this fixture can be removed.
                assert re.fullmatch(r"evaluation_test_[0-9a-f]{32}", schema)
                with admin.begin() as connection:
                    connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            if admin is not None:
                admin.dispose()
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@pytest.fixture
def db_storage(storage_database, interactive_pair):
    pair = interactive_pair
    with storage_database.begin() as connection:
        for table, entities, identity_key in (
            (Team, pair.env.catalog.teams, "team_id"),
            (Player, pair.env.catalog.players, "player_id"),
            (Champion, pair.env.catalog.champions, "champion_id"),
        ):
            for entity in entities:
                connection.execute(pg_insert(table).values(**{
                    identity_key: entity.identity,
                    "canonical_name": entity.name, "display_name": entity.name,
                }).on_conflict_do_nothing(index_elements=[identity_key]))
    return EvaluationPersistence(storage_database), pair


def _table_counts(engine):
    with engine.connect() as connection:
        return {table.__tablename__: connection.execute(
            sa.select(sa.func.count()).select_from(table)).scalar_one()
            for table in (Evaluation, EvaluationWarning, EvaluationHistory, AnalysisSession, Game)}


def test_postgresql_pair_readback_is_complete_idempotent_and_does_not_create_games(db_storage):
    store, pair = db_storage
    before = _table_counts(store.repository.engine)
    a = store.save_pre(pair.pre, pair.bundle)
    b = store.save_post(pair.post, pair.bundle, a.evaluation_id)
    first_pre, first_post = store.read(a.evaluation_id), store.read(b.evaluation_id)
    replay_pre = store.save_pre(pair.pre, pair.bundle)
    replay_post = store.save_post(pair.post, pair.bundle, a.evaluation_id)
    assert not replay_pre.created and not replay_post.created
    assert replay_pre.evaluation_id == a.evaluation_id and replay_post.evaluation_id == b.evaluation_id
    assert store.read(a.evaluation_id) == first_pre
    assert store.read(b.evaluation_id) == first_post
    assert first_pre["paired_post_ids"] == [b.evaluation_id]
    assert first_post["pre_evaluation_id"] == a.evaluation_id
    assert first_pre["game_id"] is None and first_post["game_id"] is None
    assert first_pre["analysis_id"] == pair.selection.analysis_id
    assert first_pre["inferred_at"] == pair.pre.request.runtime_context.pre_created_at.isoformat()
    assert first_post["inferred_at"] is None
    assert first_pre["history_payload"] == persistence._history_document(pair.pre)
    assert first_pre["history_payload"] == first_post["history_payload"]
    assert first_pre["blue_win_probability"] == pair.pre.prediction.p_blue_win
    assert first_post["blue_win_probability"] == pair.post.prediction.p_blue_win
    assert first_pre["warnings"] == persistence.warning_rows(pair.pre.warnings)
    assert first_post["warnings"] == persistence.warning_rows(pair.post.warnings)
    assert json.loads(json.dumps(first_pre, allow_nan=False)) == first_pre
    assert json.loads(json.dumps(first_post, allow_nan=False)) == first_post
    after = _table_counts(store.repository.engine)
    assert after["evaluation"] == before["evaluation"] + 2
    assert after["game"] == before["game"] == 0


def test_postgresql_warning_failure_rolls_back_evaluation_and_history(db_storage):
    store, pair = db_storage
    packet, history, warnings = store._packet(pair.pre, pair.bundle, "INTERACTIVE", {})
    assert warnings
    warnings = deepcopy(warnings)
    warnings[0]["severity"] = "too-long-for-the-database-column"
    before = _table_counts(store.repository.engine)
    with pytest.raises(StorageError, match="E_STORAGE_WRITE"):
        store.repository.save(packet, history, warnings)
    assert _table_counts(store.repository.engine) == before


def test_postgresql_pair_failure_rolls_back_both_phases_and_warnings(db_storage):
    store, pair = db_storage
    pre, history, pre_warnings = store._packet(pair.pre, pair.bundle, "INTERACTIVE", {})
    post, _, post_warnings = store._packet(pair.post, pair.bundle, "INTERACTIVE", {})
    post = {**post, "blue_win_probability": 2.0}
    before = _table_counts(store.repository.engine)
    with pytest.raises(StorageError, match="E_STORAGE_WRITE"):
        store.repository.save_pair(pre, post, history, pre_warnings, post_warnings)
    assert _table_counts(store.repository.engine) == before


def test_postgresql_failed_post_keeps_previous_active_pair(db_storage):
    store, pair = db_storage
    a = store.save_pre(pair.pre, pair.bundle)
    b = store.save_post(pair.post, pair.bundle, a.evaluation_id)
    before_pre, before_post = store.read(a.evaluation_id), store.read(b.evaluation_id)
    post, history, warnings = store._packet(pair.post, pair.bundle, "INTERACTIVE", {})
    post = {**post, "idempotency_key": "f" * 64, "blue_win_probability": 2.0}
    with pytest.raises(StorageError, match="E_STORAGE_WRITE"):
        store.repository.save(post, history, warnings, pre_evaluation_id=a.evaluation_id)
    assert store.read(a.evaluation_id) == before_pre
    assert store.read(b.evaluation_id) == before_post


@pytest.mark.parametrize("field,value", [
    ("context_key", "e" * 64), ("model_version", "different-bundle"),
    ("data_version", "different-dataset"),
])
def test_postgresql_incompatible_post_cannot_attach_to_saved_pre(db_storage, field, value):
    store, pair = db_storage
    a = store.save_pre(pair.pre, pair.bundle)
    before = _table_counts(store.repository.engine)
    packet, history, warnings = store._packet(pair.post, pair.bundle, "INTERACTIVE", {})
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.repository.save({**packet, field: value}, history, warnings,
                              pre_evaluation_id=a.evaluation_id)
    assert _table_counts(store.repository.engine) == before
    assert store.read(a.evaluation_id)["is_active"]


def test_postgresql_invalidation_and_new_pre_preserve_older_versions(db_storage):
    store, pair = db_storage
    a = store.save_pre(pair.pre, pair.bundle)
    b = store.save_post(pair.post, pair.bundle, a.evaluation_id)
    old_pre, old_post = store.read(a.evaluation_id), store.read(b.evaluation_id)
    store.invalidate(pair.selection.analysis_id, pair.bundle.identity.bundle_version, post_only=True)
    assert store.read(a.evaluation_id)["is_active"]
    assert not store.read(b.evaluation_id)["is_active"]
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.save_post(pair.post, pair.bundle, a.evaluation_id)
    pair.clock["now"] += timedelta(minutes=1)
    selection = replace(pair.selection, context_label="User confirmed a changed context")
    new_pre = demo.evaluate_analysis_pre(pair.runtime, selection)
    c = store.save_pre(new_pre, pair.bundle)
    assert c.evaluation_id != a.evaluation_id
    assert store.read(c.evaluation_id)["is_active"]
    assert not store.read(a.evaluation_id)["is_active"]
    assert store.read(a.evaluation_id)["input_snapshot"] == old_pre["input_snapshot"]
    assert store.read(b.evaluation_id)["input_snapshot"] == old_post["input_snapshot"]
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.save_pre(pair.pre, pair.bundle)
    store.invalidate(pair.selection.analysis_id, pair.bundle.identity.bundle_version)
    assert not store.read(c.evaluation_id)["is_active"]


def test_postgresql_return_to_prior_lineup_creates_new_version_and_retry_is_idempotent(db_storage):
    store, pair = db_storage
    pre = store.save_pre(pair.pre, pair.bundle)
    operation_a, operation_b, operation_new_a = (str(uuid4()) for _ in range(3))
    first = store.save_post(pair.post, pair.bundle, pre.evaluation_id, operation_id=operation_a)
    first_record = store.read(first.evaluation_id)
    store.invalidate(pair.selection.analysis_id, pair.bundle.identity.bundle_version, post_only=True)
    changed = demo.evaluate_analysis_post(pair.runtime, pair.pre, ("99",) + pair.champion_ids[1:])
    second = store.save_post(changed, pair.bundle, pre.evaluation_id, operation_id=operation_b)
    store.invalidate(pair.selection.analysis_id, pair.bundle.identity.bundle_version, post_only=True)
    third = store.save_post(pair.post, pair.bundle, pre.evaluation_id, operation_id=operation_new_a)
    before_retry = _table_counts(store.repository.engine)
    retry = store.save_post(pair.post, pair.bundle, pre.evaluation_id, operation_id=operation_new_a)
    assert retry.evaluation_id == third.evaluation_id and not retry.created and retry.is_active
    assert _table_counts(store.repository.engine) == before_retry
    assert len({first.evaluation_id, second.evaluation_id, third.evaluation_id}) == 3
    assert store.read(first.evaluation_id)["input_snapshot"] == first_record["input_snapshot"]
    assert store.read(first.evaluation_id)["warnings"] == first_record["warnings"]
    assert not store.read(first.evaluation_id)["is_active"]
    assert not store.read(second.evaluation_id)["is_active"]
    assert store.read(third.evaluation_id)["is_active"]
    assert store.read(pre.evaluation_id)["is_active"]
    assert store.read(third.evaluation_id)["pre_evaluation_id"] == pre.evaluation_id
    assert store.read(third.evaluation_id)["blue_win_probability"] == first_record["blue_win_probability"]
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.save_post(pair.post, pair.bundle, pre.evaluation_id, operation_id=operation_a)
    assert store.read(third.evaluation_id)["is_active"]
    with store.repository.engine.connect() as connection:
        active_posts = connection.execute(sa.select(sa.func.count()).select_from(Evaluation).where(
            Evaluation.analysis_id == pair.selection.analysis_id,
            Evaluation.model_version == pair.bundle.identity.bundle_version,
            Evaluation.evaluation_type == "POST", Evaluation.is_active.is_(True),
        )).scalar_one()
    assert active_posts == 1


@pytest.mark.parametrize("table,column", [
    ("evaluation", "input_snapshot"), ("evaluation_history", "payload"),
    ("evaluation_warning", "message"), ("analysis_session", "created_at"),
])
def test_postgresql_immutable_guards_reject_payload_changes_and_deletes(db_storage, table, column):
    store, pair = db_storage
    saved = store.save_pre(pair.pre, pair.bundle)
    record = store.read(saved.evaluation_id)
    predicates = {
        "evaluation": ("evaluation_id = :identity", saved.evaluation_id),
        "evaluation_history": ("history_sha256 = :identity", record["history_sha256"]),
        "evaluation_warning": ("evaluation_id = :identity", saved.evaluation_id),
        "analysis_session": ("analysis_id = :identity", pair.selection.analysis_id),
    }
    predicate, identity = predicates[table]
    replacements = {"input_snapshot": "'{}'::jsonb", "payload": "'{}'::jsonb",
                    "message": "'changed'", "created_at": "created_at + interval '1 second'"}
    for sql in (f"UPDATE {table} SET {column} = {replacements[column]} WHERE {predicate}",
                f"DELETE FROM {table} WHERE {predicate}"):
        with pytest.raises(SQLAlchemyError):
            with store.repository.engine.begin() as connection:
                connection.execute(sa.text(sql), {"identity": identity})
    assert store.read(saved.evaluation_id) == record


def test_postgresql_guard_rejects_inactive_parent_with_active_post(db_storage):
    store, pair = db_storage
    a = store.save_pre(pair.pre, pair.bundle)
    b = store.save_post(pair.post, pair.bundle, a.evaluation_id)
    with pytest.raises(SQLAlchemyError):
        with store.repository.engine.begin() as connection:
            connection.execute(sa.update(Evaluation).where(
                Evaluation.evaluation_id == a.evaluation_id).values(is_active=False))
    assert store.read(a.evaluation_id)["is_active"]
    assert store.read(b.evaluation_id)["is_active"]


def test_postgresql_same_request_race_creates_one_evaluation(db_storage):
    store, pair = db_storage
    packet, history, warnings = store._packet(pair.pre, pair.bundle, "INTERACTIVE", {})

    def save():
        return EvaluationRepository(store.repository.engine).save(packet, history, warnings)

    before = _table_counts(store.repository.engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _value: save(), range(2)))
    assert results[0].evaluation_id == results[1].evaluation_id
    assert sorted(item.created for item in results) == [False, True]
    assert _table_counts(store.repository.engine)["evaluation"] == before["evaluation"] + 1


def test_postgresql_replay_conflict_does_not_mutate_saved_record(db_storage):
    store, pair = db_storage
    saved = store.save_pre(pair.pre, pair.bundle)
    before = store.read(saved.evaluation_id)
    packet, history, warnings = store._packet(pair.pre, pair.bundle, "INTERACTIVE", {})
    with pytest.raises(StorageError, match="E_STORAGE_CONFLICT"):
        store.repository.save({**packet, "blue_win_probability": 0.123456789}, history, warnings)
    assert store.read(saved.evaluation_id) == before
