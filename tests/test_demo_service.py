"""Synthetic integration fixtures only; never require PostgreSQL or canonical files."""

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from match_insight.database import demo_catalog
from match_insight.database.real_snapshot import DatabaseGame, RealSnapshot, pre_context_sha256
from match_insight.features.post import FinalLineup
from match_insight.features.pre import ROLES, HistoricalGame, PlayerSlot, PreFeatureInputError
from match_insight.ml import models, real_training
from match_insight.ml.dataset import (
    RETROSPECTIVE_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetTarget,
    build_paired_dataset,
    temporal_split,
)
from match_insight.services import demo, evaluation


@pytest.fixture(scope="module")
def synthetic_demo_bundle():
    """Fit once during test setup using explicitly synthetic retrospective samples."""
    request = demo.make_demo_cases()[0].request
    targets, cutoffs = [], []
    for number in range(20):
        game_id = f"synthetic-9b-{number:02d}"
        cutoff = datetime(2025, 7, 1, tzinfo=UTC) + timedelta(days=3 * number)
        targets.append(DatasetTarget(
            game_id, "A", "B", request.target.blue_roster, request.target.red_roster,
            "15.01", FinalLineup(game_id, "15.01", request.history[0].slots),
            "A" if number % 2 else "B", cutoff + timedelta(days=1, hours=12),
        ))
        cutoffs.append(CutoffRecord(
            game_id, cutoff, f"synthetic://9b/{game_id}", RETROSPECTIVE_PROTOCOL,
            CutoffVerification.PROTOCOL_ASSUMED,
        ))
    dataset = build_paired_dataset(
        tuple(targets), request.history, tuple(cutoffs), request.champion_reference,
        approved_policy_versions=frozenset({RETROSPECTIVE_PROTOCOL}),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    split = temporal_split(dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    return models.fit_model_bundle(
        dataset, split, models.ModelIdentity(
            "retrospective:postgresql:synthetic-9b",
            "retrospective:split:synthetic-9b", "retrospective:7l:synthetic-9b",
        ), models.ModelConfig(forest_trees=8), evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )


def _milliseconds(value):
    return (value - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(milliseconds=1)


def _synthetic_input():
    """DB timestamps deliberately NULL; only pinned proof timestamps are authoritative."""
    games, proofs = [], {}
    history = demo.make_demo_cases()[0].request.history
    contexts = [
        (record.game, record.slots, record.game.ended_at - timedelta(hours=1),
         record.game.ended_at)
        for record in history
    ]
    for case in demo.make_demo_cases():
        target = case.request.target
        start = target.history_cutoff_at + timedelta(days=1)
        slots = tuple(
            replace(slot, team_id=team, player_id=player.player_id)
            for side, team, roster, original in (
                ("BLUE", target.blue_team_id, target.blue_roster, history[0].slots[:5]),
                ("RED", target.red_team_id, target.red_roster, history[0].slots[5:]),
            )
            for slot, player in zip(original, roster, strict=True)
        )
        contexts.append((
            HistoricalGame(target.game_id, target.blue_team_id, target.red_team_id,
                           target.blue_roster, target.red_roster, target.blue_team_id, None),
            slots, start, start + timedelta(hours=1),
        ))
    for game, slots, start, end in contexts:
        games.append(DatabaseGame(
            replace(game, ended_at=None), FinalLineup(game.game_id, "15.01", slots), None,
        ))
        proofs[game.game_id] = {
            "grade": real_training.TIME_EVIDENCE_GRADE,
            "start_timestamp_ms": _milliseconds(start),
            "end_timestamp_ms": _milliseconds(end),
            "started_at": start.isoformat(), "ended_at": end.isoformat(),
            "fetched_at_utc": datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
            "revision_status": "MISSING",
        }
    snapshot = RealSnapshot(
        "postgresql:synthetic-fixture", datetime(2026, 1, 1, tzinfo=UTC),
        "a" * 64, tuple(games), demo.CHAMPION_REFERENCE,
    )
    document = {
        "schema_version": real_training.RETROSPECTIVE_SCHEMA,
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "snapshot_sha256": snapshot.sha256,
        "approval_ref": "synthetic://9b/test-only-approval",
        "ended_at_provenance": {
            "evidence_ref": "local-v5-summary-and-twelve-exports",
            "verification": real_training.TIME_EVIDENCE_GRADE,
        },
        "source_files": [], "time_proofs": proofs, "records": [], "exclusions": [],
    }
    for item in snapshot.games:
        game_id = item.game.game_id
        start = real_training._time_ms(proofs[game_id]["start_timestamp_ms"])
        document["records"].append({
            "game_id": game_id,
            "history_cutoff_at": real_training.retrospective_cutoff(start).isoformat(),
            "evidence_ref": f"synthetic://9b/{game_id}",
            "policy_version": RETROSPECTIVE_PROTOCOL,
            "verification": "PROTOCOL_ASSUMED",
            "pre_context_sha256": pre_context_sha256(item),
            "pre_context_evidence_ref": f"snapshot:{snapshot.sha256}:{game_id}",
            "pre_context_verification": "RECONSTRUCTED_POSTGAME",
        })
    return snapshot, document


def _catalog_rows(snapshot):
    team_ids = sorted({team for item in snapshot.games
                       for team in (item.game.blue_team_id, item.game.red_team_id)})
    player_ids = sorted({slot.player_id for item in snapshot.games
                         for slot in item.game.blue_roster + item.game.red_roster})
    return {
        "tournament": [{"tournament_id": "t", "name": "Giải fixture"}],
        "tournament_stage": [{"stage_id": "s", "tournament_id": "t", "name": "Vòng fixture"}],
        "series": [{"series_id": "series", "stage_id": "s"}],
        "game": [{"game_id": item.game.game_id, "series_id": "series", "stage_id": None,
                  "game_number": number} for number, item in enumerate(snapshot.games, start=1)],
        "team": [{"team_id": identity, "display_name": f"Đội {identity}", "logo_file": None}
                 for identity in team_ids],
        "player": [{"player_id": identity, "display_name": f"Người chơi {identity}",
                    "photo_file": None} for identity in player_ids],
        "champion": [{"champion_id": champion.champion_id, "display_name": champion.name,
                      "image_file": None} for champion in demo.CHAMPIONS],
    }


@pytest.fixture
def demo_environment(monkeypatch, tmp_path, synthetic_demo_bundle):
    snapshot, document = _synthetic_input()
    rows = _catalog_rows(snapshot)
    catalog = demo_catalog.map_demo_catalog(rows)
    input_path, model_path = tmp_path / "synthetic-input.json", tmp_path / "synthetic-model.joblib"

    def write_input(doc):
        data = json.dumps(doc, allow_nan=False).encode("utf-8")
        input_path.write_bytes(data)
        # Test-only expectation: never edit or depend on production artifacts.
        monkeypatch.setattr(demo, "CANONICAL_DEMO_INPUT_SHA256", hashlib.sha256(data).hexdigest())

    write_input(document)
    monkeypatch.setattr(demo, "CANONICAL_DEMO_SNAPSHOT_SHA256", snapshot.sha256)
    models.save_bundle(synthetic_demo_bundle, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    identity = synthetic_demo_bundle.identity
    metadata = {
        "artifact_schema": "real-model-artifact-v1",
        "data_kind": "REAL_RETROSPECTIVE_SIMULATION", "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "model_sha256": model_hash, "bundle": models._plain(
            real_training._bundle_metadata(synthetic_demo_bundle)),
        "run": {
            "status": "TRAINED", "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
            "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
            "dataset_id": identity.dataset_id, "split_id": identity.split_id,
            "model_bundle_version": identity.bundle_version,
            "selected_family": synthetic_demo_bundle.family,
        },
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    expected = real_training.ModelArtifactExpectation(identity, model_hash, synthetic_demo_bundle.family)
    monkeypatch.setattr(demo, "CANONICAL_RETROSPECTIVE_EXPECTATION", expected)
    monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION", expected)
    calls = {"db": 0, "loader": 0}
    original_load = real_training.load_retrospective_bundle

    def read_data():
        calls["db"] += 1
        return snapshot, catalog

    def load_model(path, *, expected):
        assert expected is demo.CANONICAL_RETROSPECTIVE_EXPECTATION
        calls["loader"] += 1
        return original_load(path, expected=expected)

    monkeypatch.setattr(demo, "_read_runtime_data", read_data)
    monkeypatch.setattr(demo, "load_retrospective_bundle", load_model)
    monkeypatch.setattr(evaluation, "load_retrospective_bundle", load_model)
    runtime = demo.load_retrospective_demo(input_path=input_path, model_path=model_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime must not fit/save/train/build a dataset/read the temporal archive")

    for owner in (models, real_training, demo):
        for name in ("fit_model_bundle", "save_bundle", "prepare_run", "build_paired_dataset",
                     "temporal_split", "load_time_archive", "build_demo_model"):
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, forbidden)
    for cls in (models.Pipeline, models.ColumnTransformer, models.SimpleImputer,
                models.StandardScaler, models.OneHotEncoder, models.LogisticRegression,
                models.RandomForestClassifier, models.DummyClassifier):
        for name in ("fit", "fit_transform"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, forbidden)
    monkeypatch.setattr(models.joblib, "dump", forbidden)
    calls.update(db=0, loader=0)
    return SimpleNamespace(
        runtime=runtime, snapshot=snapshot, document=document, rows=rows, catalog=catalog,
        input_path=input_path, model_path=model_path, calls=calls, write_input=write_input,
    )


def _load(env):
    return demo.load_retrospective_demo(input_path=env.input_path, model_path=env.model_path)


def test_pinned_initialization_and_real_pre_use_proof_times_with_null_db(demo_environment):
    env = demo_environment
    runtime = _load(env)
    before = deepcopy(runtime.history)
    case = demo.get_retrospective_case(runtime, "demo-1")
    assert all(item.started_at is None and item.game.ended_at is None for item in env.snapshot.games)
    proof = env.document["time_proofs"]["demo-1"]
    assert case.target_started_at == real_training._time_ms(proof["start_timestamp_ms"])
    assert tuple(row.game.game_id for row in case.request.history) == ("synthetic-h2", "synthetic-h1")
    assert all(row.game.ended_at < case.request.target.history_cutoff_at for row in case.request.history)
    first = demo.evaluate_demo_pre(runtime, "demo-1")
    second = demo.evaluate_demo_pre(runtime, "demo-1")
    assert first == second
    assert first.features.blue.recent_form.value == 0.5
    assert first.features.blue.recent_form.sample_count == 2
    assert first.request.cutoff.verification is CutoffVerification.PROTOCOL_ASSUMED
    assert first.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert first.features.metadata.cutoff_verification == "NOT_ASSESSED"
    blue, red = demo.phase_probabilities(first)
    assert 0 <= blue <= 1 and 0 <= red <= 1 and blue + red == pytest.approx(1)
    assert env.calls == {"db": 1, "loader": 3}
    assert runtime.history == before


@pytest.mark.parametrize("change", ["missing", "bytes", "schema", "protocol", "duplicate"])
def test_invalid_input_rejected_before_model_or_db(demo_environment, change):
    env = demo_environment
    doc = deepcopy(env.document)
    if change == "missing":
        env.input_path.unlink()
    elif change == "bytes":
        env.input_path.write_bytes(b"{}")
    else:
        if change == "duplicate":
            doc["records"].append(deepcopy(doc["records"][0]))
        else:
            doc["schema_version" if change == "schema" else "evaluation_protocol"] = "wrong"
        env.write_input(doc)
    with pytest.raises(PreFeatureInputError):
        _load(env)
    assert env.calls == {"db": 0, "loader": 0}


@pytest.mark.parametrize("change", ["snapshot", "context", "cutoff", "unverified", "proof_missing",
                                    "milliseconds", "naive", "end_order", "proof_grade"])
def test_evidence_inconsistency_fails_closed(demo_environment, change):
    env = demo_environment
    doc = deepcopy(env.document)
    row = next(row for row in doc["records"] if row["game_id"] == "demo-1")
    proof = doc["time_proofs"]["demo-1"]
    if change == "snapshot":
        doc["snapshot_sha256"] = "b" * 64
    elif change == "context":
        row["pre_context_sha256"] = "b" * 64
    elif change == "cutoff":
        row["history_cutoff_at"] = "2025-08-01T01:00:00+00:00"
    elif change == "unverified":
        row["verification"] = "UNVERIFIED"
    elif change == "proof_missing":
        del doc["time_proofs"]["demo-1"]
    elif change == "milliseconds":
        proof["start_timestamp_ms"] += 1
    elif change == "naive":
        proof["started_at"] = "2025-08-02T12:00:00"
    elif change == "end_order":
        proof["end_timestamp_ms"] = proof["start_timestamp_ms"]
        proof["ended_at"] = proof["started_at"]
    else:
        proof["grade"] = "VERIFIED_EXTERNALLY"
    env.write_input(doc)
    with pytest.raises(PreFeatureInputError):
        _load(env)


@pytest.mark.parametrize("change", ["hash", "duplicate", "unknown", "start", "end"])
def test_snapshot_conflicts_rejected(demo_environment, monkeypatch, change):
    env = demo_environment
    snapshot = env.snapshot
    item = snapshot.games[-2]
    if change == "hash":
        snapshot = replace(snapshot, sha256="b" * 64)
    elif change == "duplicate":
        snapshot = replace(snapshot, games=snapshot.games + (item,))
    elif change == "unknown":
        snapshot = replace(snapshot, games=snapshot.games[:-1])
    else:
        invalid = datetime(2024, 1, 1, tzinfo=UTC)
        altered = (replace(item, started_at=invalid) if change == "start"
                   else replace(item, game=replace(item.game, ended_at=invalid)))
        snapshot = replace(snapshot, games=snapshot.games[:-2] + (altered, snapshot.games[-1]))
    monkeypatch.setattr(demo, "_read_runtime_data", lambda: (snapshot, env.catalog))
    with pytest.raises(PreFeatureInputError):
        _load(env)


def test_matching_db_timestamps_and_offsets_do_not_change_proof_authority(demo_environment, monkeypatch):
    env = demo_environment
    offset = timezone(timedelta(hours=7))
    games = []
    for item in env.snapshot.games:
        proof = env.document["time_proofs"][item.game.game_id]
        games.append(replace(item,
            started_at=real_training._time_ms(proof["start_timestamp_ms"]).astimezone(offset),
            game=replace(item.game, ended_at=real_training._time_ms(proof["end_timestamp_ms"]))
        ))
    snapshot = replace(env.snapshot, games=tuple(games))
    monkeypatch.setattr(demo, "_read_runtime_data", lambda: (snapshot, env.catalog))
    runtime = _load(env)
    assert runtime.targets == env.runtime.targets
    assert runtime.history == env.runtime.history


def test_runtime_history_filters_target_equal_future_and_unproven_db_games(demo_environment):
    env = demo_environment
    case = demo.get_retrospective_case(env.runtime, "demo-1")
    source = env.runtime.history[0]
    cutoff = case.request.target.history_cutoff_at
    extras = tuple(replace(source, game=replace(source.game, game_id=name, ended_at=end))
                   for name, end in (("equal", cutoff), ("future", cutoff + timedelta(seconds=1))))
    changed = replace(env.runtime, history=env.runtime.history + extras)
    assert demo.get_retrospective_case(changed, "demo-1").request == case.request
    extra_db = replace(env.snapshot.games[0], game=replace(env.snapshot.games[0].game,
                       game_id="db-no-proof", ended_at=cutoff - timedelta(days=1)))
    snapshot = replace(env.snapshot, games=env.snapshot.games + (extra_db,))
    _targets, history = demo._runtime_targets(snapshot, env.document)
    assert "db-no-proof" not in {row.game.game_id for row in history}


def test_bad_model_identity_is_rejected_before_db(demo_environment):
    env = demo_environment
    path = env.model_path.with_suffix(".json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["bundle"]["identity"]["dataset_id"] = "retrospective:postgresql:other"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PreFeatureInputError):
        _load(env)
    assert env.calls["db"] == 0


def test_target_selection_fixed_context_and_state_invalidation(demo_environment):
    runtime = demo_environment.runtime
    first = demo.get_retrospective_case(runtime, "demo-1")
    second = demo.get_retrospective_case(runtime, "demo-2")
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    state = evaluation.EvaluationState(pre.context_key, pre=pre)
    assert demo.refresh_retrospective_state(state, first, runtime.bundle) is state
    changed = demo.refresh_retrospective_state(state, second, runtime.bundle)
    assert changed.pre is None and changed.post is None
    with pytest.raises(PreFeatureInputError):
        demo.get_retrospective_case(runtime, "invented-match")
    result = demo.evaluate_demo_pre(runtime, "demo-2")
    assert result.features.blue.recent_form.value is None
    assert result.features.blue.recent_form.missing
    assert result.features.blue.recent_form.sample_count == 0


@pytest.mark.parametrize("failure", [None, "snapshot", "catalog"])
def test_owned_engine_disposed_and_runtime_environment_restored(monkeypatch, failure):
    import sqlalchemy

    from match_insight.config import Settings

    events = []
    engine = SimpleNamespace(dispose=lambda: events.append("dispose"))
    before = {key: os.environ.get(key) for key in
              ("DATABASE_URL", "WIKI_USERNAME_MATCH_INSIGHT", "WIKI_PASSWORD_MATCH_INSIGHT")}

    def settings_init(self):
        self.database_url = "synthetic://not-a-database"
        os.environ["WIKI_USERNAME_MATCH_INSIGHT"] = "synthetic-fixture-only"

    def read(name):
        def perform(_engine):
            assert _engine is engine
            events.append(name)
            if failure == name:
                raise RuntimeError("Transport detail must not escape application boundary")
            return name
        return perform

    monkeypatch.setattr(Settings, "__init__", settings_init)
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url: engine)
    monkeypatch.setattr(demo, "read_snapshot", read("snapshot"))
    monkeypatch.setattr(demo, "read_demo_catalog", read("catalog"))
    if failure:
        with pytest.raises(PreFeatureInputError) as error:
            demo._read_runtime_data()
        assert error.value.code == "E_SOURCE_CONFLICT"
        assert "Transport detail" not in str(error.value)
    else:
        assert demo._read_runtime_data() == ("snapshot", "catalog")
    assert events[-1] == "dispose"
    assert {key: os.environ.get(key) for key in before} == before


def test_catalog_is_presentation_only_and_missing_media_is_allowed():
    snapshot, _document = _synthetic_input()
    rows = _catalog_rows(snapshot)
    before = deepcopy(rows)
    catalog = demo_catalog.map_demo_catalog(rows)
    assert all(entity.media_file is None for entity in catalog.teams + catalog.players)
    assert catalog.teams[0].name == "Đội A"
    assert catalog.games[0].tournament_name == "Giải fixture"
    assert not hasattr(catalog.games[0], "started_at")
    assert rows == before
    reordered = {key: list(reversed(value)) for key, value in rows.items()}
    assert demo_catalog.map_demo_catalog(reordered) == catalog


@pytest.mark.parametrize("change", ["duplicate", "parent", "both_parents"])
def test_catalog_duplicate_or_conflicting_identity_is_rejected(change):
    snapshot, _document = _synthetic_input()
    rows = _catalog_rows(snapshot)
    if change == "duplicate":
        rows["team"].append(dict(rows["team"][0]))
    elif change == "parent":
        rows["game"][0]["series_id"] = "unknown"
    else:
        rows["game"][0]["stage_id"] = "s"
    with pytest.raises(PreFeatureInputError):
        demo_catalog.map_demo_catalog(rows)


@pytest.mark.parametrize("failure", [None, "query", "not_readonly", "mapping"])
def test_catalog_readonly_transaction_and_cleanup(failure):
    snapshot, _document = _synthetic_input()
    rows = _catalog_rows(snapshot)
    if failure == "mapping":
        rows["team"].append(dict(rows["team"][0]))
    events = []

    class Connection:
        def __enter__(self):
            events.append("connect")
            return self

        def __exit__(self, *args):
            events.append("close")

        def execution_options(self, **kwargs):
            assert kwargs == {"isolation_level": "REPEATABLE READ"}
            return self

        def begin(self):
            return SimpleNamespace(rollback=lambda: events.append("rollback"))

        def exec_driver_sql(self, sql):
            events.append(sql)
            assert sql in {"SET TRANSACTION READ ONLY", "SHOW transaction_read_only"}
            return SimpleNamespace(scalar_one=lambda: "off" if failure == "not_readonly" else "on")

        def execute(self, statement):
            assert "SHOW transaction_read_only" in events
            if failure == "query":
                raise RuntimeError("test query failure")
            name = statement.get_final_froms()[0].name
            return SimpleNamespace(mappings=lambda: SimpleNamespace(all=lambda: rows[name]))

    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"), connect=Connection)
    if failure:
        with pytest.raises((RuntimeError, PreFeatureInputError)):
            demo_catalog.read_demo_catalog(engine)
    else:
        assert demo_catalog.read_demo_catalog(engine) == demo_catalog.map_demo_catalog(rows)
    assert events[-2:] == ["rollback", "close"]


def _post_ids():
    return tuple(champion.champion_id for champion in demo.CHAMPIONS[:10])


def test_post_real_pipeline_preserves_pre_and_uses_retained_proof(demo_environment, monkeypatch):
    env = demo_environment
    runtime = env.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    before = deepcopy(pre)
    original_post = evaluation.evaluate_retrospective_post
    original_validate = evaluation.validate_final_lineup
    observed = []

    def post_boundary(saved_pre, lineup, *, target_started_at, model_path):
        assert saved_pre is pre
        assert target_started_at == next(
            item.target_started_at for item in runtime.targets if item.game_id == "demo-1"
        )
        observed.append("post")
        return original_post(saved_pre, lineup, target_started_at=target_started_at,
                             model_path=model_path)

    def validate(saved_pre, lineup, bundle, *, evaluation_protocol):
        assert saved_pre is pre
        assert evaluation_protocol == RETROSPECTIVE_PROTOCOL
        observed.append("validate")
        return original_validate(saved_pre, lineup, bundle, evaluation_protocol=evaluation_protocol)

    def forbidden(*args, **kwargs):
        pytest.fail("POST must not read DB, rebuild PRE or reselect application history")

    monkeypatch.setattr(demo, "evaluate_retrospective_post", post_boundary)
    monkeypatch.setattr(demo, "validate_final_lineup", validate)
    for name in ("_read_runtime_data", "get_retrospective_case", "evaluate_demo_pre",
                 "evaluate_retrospective_pre", "select_pre_history"):
        monkeypatch.setattr(demo, name, forbidden)
    monkeypatch.setattr(evaluation, "create_pre", forbidden)
    first = demo.evaluate_demo_post(runtime, pre, _post_ids())
    second = demo.evaluate_demo_post(runtime, pre, _post_ids())
    assert first == second
    assert first.pre is pre and pre == before
    assert observed == ["validate", "post", "validate", "post"]
    assert env.calls["db"] == 0
    assert first.prediction.history_cutoff_at == pre.prediction.history_cutoff_at
    assert first.prediction.policy_version == RETROSPECTIVE_PROTOCOL
    assert first.prediction.model_bundle_version == pre.prediction.model_bundle_version
    assert first.features.pre == pre.features
    for pair in first.features.player_champion:
        assert pair.games_count == 2 and pair.wins_count == 1
        assert pair.win_rate == 0.5 and not pair.missing
    blue, red = demo.phase_probabilities(first)
    assert 0 <= blue <= 1 and 0 <= red <= 1
    assert blue + red == pytest.approx(1)
    assert first.comparison.delta_probability == pytest.approx(
        first.prediction.p_blue_win - pre.prediction.p_blue_win
    )


@pytest.mark.parametrize("ids", [None, 42, (), (None,) * 10, _post_ids()[:9],
                                 _post_ids()[:-1] + ("not-a-champion",),
                                 _post_ids()[:-1] + (_post_ids()[0],)])
def test_invalid_post_lineup_rejected_before_prediction(demo_environment, monkeypatch, ids):
    runtime = demo_environment.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid lineup must not reach POST boundary/prediction")

    monkeypatch.setattr(demo, "evaluate_retrospective_post", forbidden)
    with pytest.raises(PreFeatureInputError):
        demo.validate_demo_lineup(runtime, pre, ids)
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_demo_post(runtime, pre, ids)


def test_post_requires_saved_pre(demo_environment):
    with pytest.raises(PreFeatureInputError) as error:
        demo.evaluate_demo_post(demo_environment.runtime, None, _post_ids())
    assert error.value.code == "E_PRE_SNAPSHOT_REQUIRED"


@pytest.mark.parametrize("change", ["target", "cutoff", "proof_start", "bundle", "snapshot",
                                    "evidence", "protocol", "history", "missing_target"])
def test_post_rejects_incompatible_runtime(demo_environment, change):
    runtime = demo_environment.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    index = next(i for i, target in enumerate(runtime.targets) if target.game_id == "demo-1")
    target = runtime.targets[index]
    if change in {"target", "cutoff", "proof_start"}:
        if change == "target":
            target = replace(target, target=replace(target.target, patch="15.02"))
        elif change == "cutoff":
            target = replace(target, cutoff=replace(target.cutoff,
                history_cutoff_at=target.cutoff.history_cutoff_at + timedelta(hours=1)))
        else:
            target = replace(target, target_started_at=target.target_started_at + timedelta(days=1))
        runtime = replace(runtime, targets=runtime.targets[:index] + (target,) + runtime.targets[index + 1:])
    elif change == "bundle":
        runtime = replace(runtime, bundle=replace(runtime.bundle, identity=replace(
            runtime.bundle.identity, bundle_version="retrospective:7l:other")))
    elif change in {"snapshot", "evidence"}:
        runtime = replace(runtime, **{f"{change}_sha256": "b" * 64})
    elif change == "protocol":
        runtime = replace(runtime, evaluation_protocol="strict")
    elif change == "history":
        runtime = replace(runtime, history=tuple(
            row for row in runtime.history if row.game.game_id != "synthetic-h1"
        ))
    else:
        runtime = replace(runtime, targets=tuple(
            row for row in runtime.targets if row.game_id != "demo-1"
        ))
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_demo_post(runtime, pre, _post_ids())


def test_champion_change_invalidates_post_comparison_and_preserves_pre(demo_environment):
    runtime = demo_environment.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    post = demo.evaluate_demo_post(runtime, pre, _post_ids())
    case = demo.get_retrospective_case(runtime, "demo-1")
    state = evaluation.EvaluationState(pre.context_key, _post_ids(), pre, post)
    assert demo.refresh_retrospective_state(
        state, case, runtime.bundle, champion_ids=_post_ids()
    ) is state
    changed_ids = ("99",) + _post_ids()[1:]
    changed = demo.refresh_retrospective_state(
        state, case, runtime.bundle, champion_ids=changed_ids
    )
    assert changed.pre is pre and changed.post is None
    new_post = demo.evaluate_demo_post(runtime, pre, changed_ids)
    assert new_post.pre is pre
    for old, new in zip(post.features.player_champion, new_post.features.player_champion, strict=True):
        if new.side == "BLUE" and new.role == "TOP":
            assert new.games_count == 0 and new.wins_count == 0
            assert new.win_rate is None and new.missing
        else:
            assert new == old
    reset = demo.refresh_retrospective_state(
        changed, demo.get_retrospective_case(runtime, "demo-2"), runtime.bundle
    )
    assert reset.pre is None and reset.post is None and reset.champion_ids == ()


def test_post_does_not_include_history_added_after_pre(demo_environment):
    runtime = demo_environment.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    baseline = demo.evaluate_demo_post(runtime, pre, _post_ids())
    source = runtime.history[0]
    later = replace(source, game=replace(source.game, game_id="after-pre",
                    ended_at=pre.prediction.history_cutoff_at + timedelta(hours=1)))
    changed_runtime = replace(runtime, history=runtime.history + (later,))
    result = demo.evaluate_demo_post(changed_runtime, pre, _post_ids())
    assert result == baseline
    assert result.pre is pre
    assert "after-pre" not in result.features.accepted_game_ids


def test_comparison_preserves_unified_output_without_runtime_side_effects(
    demo_environment, monkeypatch,
):
    env = demo_environment
    runtime = env.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    post = demo.evaluate_demo_post(runtime, pre, ("99",) + _post_ids()[1:])
    before = deepcopy((pre, post, runtime.history))
    expected = evaluation.compare_retrospective_evaluations(pre, post, runtime.bundle)
    original_compare = evaluation.compare_retrospective_evaluations
    returned = []

    def compare(saved_pre, saved_post, bundle):
        assert saved_pre is pre and saved_post is post and bundle is runtime.bundle
        result = original_compare(saved_pre, saved_post, bundle)
        returned.append(result)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Comparison must not read/hash/load/fit/save/predict or rebuild features")

    monkeypatch.setattr(demo, "compare_retrospective_evaluations", compare)
    for owner, names in (
        (demo, ("_read_runtime_data", "_read_pinned_input", "load_retrospective_bundle",
                "get_retrospective_case", "select_pre_history", "evaluate_demo_pre",
                "evaluate_demo_post", "validate_demo_lineup")),
        (evaluation, ("load_retrospective_bundle", "create_pre", "create_post", "predict_phase",
                      "_digest", "build_pre_features", "build_post_features")),
        (models, ("_digest", "predict_phase")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    env.calls.update(db=0, loader=0)
    result = demo.compare_demo_evaluations(runtime, pre, post)
    again = demo.compare_demo_evaluations(runtime, pre, post)
    assert result is returned[0] and again is returned[1]
    assert result == again == expected
    assert (pre, post, runtime.history) == before
    assert env.calls == {"db": 0, "loader": 0}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["game_id"] == pre.request.target.game_id
    assert result["history_cutoff_at"] == pre.prediction.history_cutoff_at.isoformat()
    assert result["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert result["evaluation_protocol"] == RETROSPECTIVE_PROTOCOL
    assert result["model"]["bundle_version"] == runtime.bundle.identity.bundle_version
    for phase in ("pre", "post"):
        values = result[phase]
        assert 0 <= values["blue_win_probability"] <= 1
        assert 0 <= values["red_win_probability"] <= 1
        assert values["blue_win_probability"] + values["red_win_probability"] == pytest.approx(1)
    delta = result["comparison"]
    assert delta["blue_probability_delta"] == pytest.approx(
        result["post"]["blue_win_probability"] - result["pre"]["blue_win_probability"]
    )
    assert delta["red_probability_delta"] == pytest.approx(-delta["blue_probability_delta"])
    assert delta["red_probability_delta"] == pytest.approx(
        result["post"]["red_win_probability"] - result["pre"]["red_win_probability"]
    )
    assert any(warning["code"] == "W_PAIR_HISTORY_MISSING" for warning in result["warnings"])
    assert result["warnings"] == expected["warnings"]


@pytest.mark.parametrize("change", [
    "missing_pre", "missing_post", "target", "cutoff", "bundle", "protocol", "saved_delta",
])
def test_comparison_rejects_missing_or_incompatible_current_pair(demo_environment, change):
    runtime = demo_environment.runtime
    pre = demo.evaluate_demo_pre(runtime, "demo-1")
    post = demo.evaluate_demo_post(runtime, pre, _post_ids())
    if change == "missing_pre":
        pre = None
    elif change == "missing_post":
        post = None
    elif change == "target":
        pre = demo.evaluate_demo_pre(runtime, "demo-2")
    elif change == "cutoff":
        pre = replace(pre, request=replace(pre.request, cutoff=replace(
            pre.request.cutoff,
            history_cutoff_at=pre.request.cutoff.history_cutoff_at + timedelta(hours=1),
        )))
    elif change == "bundle":
        runtime = replace(runtime, bundle=replace(runtime.bundle, identity=replace(
            runtime.bundle.identity, bundle_version="retrospective:7l:other",
        )))
    elif change == "protocol":
        runtime = replace(runtime, evaluation_protocol="strict")
    else:
        # A valid-domain delta that disagrees with the saved predictions must fail.
        delta = post.comparison.delta_probability
        changed = 0.25 if abs(delta - 0.25) > 0.1 else -0.25
        post = replace(post, comparison=replace(post.comparison, delta_probability=changed))
    with pytest.raises(PreFeatureInputError):
        demo.compare_demo_evaluations(runtime, pre, post)


def _analysis_input():
    # Independent user choices. These two valid catalog teams have never met.
    return demo.AnalysisInput(
        "A", "D",
        tuple(PlayerSlot(role, f"a{number}") for number, role in enumerate(ROLES, 1)),
        tuple(PlayerSlot(role, f"d{number}") for number, role in enumerate(ROLES, 1)),
        "15.01", True, context_label="User-selected functional test",
    )


@pytest.fixture
def analysis_environment(demo_environment, monkeypatch):
    env = demo_environment
    clock = {"now": datetime(2026, 1, 3, 12, tzinfo=UTC), "reads": 0}

    def now():
        clock["reads"] += 1
        return clock["now"]

    monkeypatch.setattr(demo, "_utc_now", now)
    runtime = demo.load_analysis_runtime(input_path=env.input_path, model_path=env.model_path)
    assert env.calls == {"db": 1, "loader": 1}
    assert clock["reads"] == 1
    return env, runtime, clock


def test_interactive_analysis_absent_from_db_and_evidence_full_integration(
    analysis_environment, monkeypatch,
):
    env, runtime, clock = analysis_environment
    selection = _analysis_input()
    assert selection.analysis_id not in {item.game.game_id for item in env.snapshot.games}
    assert selection.analysis_id not in env.document["time_proofs"]
    assert not hasattr(selection, "winner_team_id")
    assert not hasattr(selection, "ended_at")
    assert not hasattr(selection, "champion_ids")
    before_files = {p.name: p.read_bytes() for p in env.model_path.parent.iterdir() if p.is_file()}
    before = deepcopy((runtime.history, runtime.bundle.identity, runtime.bundle.contract))

    def forbidden(*args, **kwargs):
        pytest.fail("Analysis runtime must not look up historical targets or read DB/model again")

    for name in ("get_retrospective_case", "_runtime_targets", "evaluate_demo_pre",
                 "_read_runtime_data", "_read_pinned_input", "load_retrospective_bundle"):
        monkeypatch.setattr(demo, name, forbidden)
    pre = demo.evaluate_analysis_pre(runtime, selection)
    assert clock["reads"] == 2  # One init + exactly one PRE creation clock read.
    context = pre.request.runtime_context
    assert context.pre_created_at == clock["now"]
    assert context.history_cutoff_at == datetime(2026, 1, 2, tzinfo=UTC)
    assert context.context_source == "USER_PROVIDED"
    assert context.user_confirmed is True and context.pre_draft_verification == "NOT_VERIFIED"
    assert pre.request.cutoff is None
    assert pre.features.h2h_blue.value is None
    assert pre.features.h2h_blue.sample_count == 0 and pre.features.h2h_blue.missing
    assert any(warning.code == "W_H2H_MISSING" for warning in pre.warnings)
    assert pre.request.target.game_id == selection.analysis_id

    clock["now"] += timedelta(days=2)
    post = demo.evaluate_analysis_post(runtime, pre, _post_ids())
    result = demo.compare_analysis_evaluations(runtime, pre, post)
    assert post.pre is pre
    assert post.prediction.history_cutoff_at == context.history_cutoff_at
    assert clock["reads"] == 2
    assert result["inference_mode"] == "INTERACTIVE_ANALYSIS"
    assert result["analysis_id"] == selection.analysis_id
    assert result["model"]["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert result["model"]["evaluation_protocol"] == RETROSPECTIVE_PROTOCOL
    assert result["model"]["dataset_id"] == runtime.bundle.identity.dataset_id
    assert result["model"]["split_id"] == runtime.bundle.identity.split_id
    assert result["model"]["bundle_version"] == runtime.bundle.identity.bundle_version
    assert "data_kind" not in result and "evaluation_protocol" not in result
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert demo.evaluate_analysis_post(runtime, pre, _post_ids()) == post
    for phase in ("pre", "post"):
        assert 0 <= result[phase]["blue_win_probability"] <= 1
        assert result[phase]["red_win_probability"] == pytest.approx(
            1 - result[phase]["blue_win_probability"]
        )
    assert result["comparison"]["blue_probability_delta"] == pytest.approx(
        result["post"]["blue_win_probability"] - result["pre"]["blue_win_probability"]
    )
    assert result["comparison"]["red_probability_delta"] == pytest.approx(
        -result["comparison"]["blue_probability_delta"]
    )
    coverage = result["history_coverage"]
    assert coverage["proof_pool"]["game_count"] == len(runtime.history)
    assert set(coverage["pre_used"]["game_ids"]) <= set(pre.features.accepted_game_ids)
    assert coverage["patch_scope_status"] == "NOT_VERIFIED"
    assert coverage["patch_observed_in_history"]
    assert (runtime.history, runtime.bundle.identity, runtime.bundle.contract) == before
    assert env.calls == {"db": 1, "loader": 1}
    assert before_files == {
        p.name: p.read_bytes() for p in env.model_path.parent.iterdir() if p.is_file()
    }


@pytest.mark.parametrize("field,value", [
    ("user_confirmed", False), ("user_confirmed", 1), ("user_confirmed", "true"),
    ("context_source", "RECONSTRUCTED_POSTGAME"),
    ("pre_draft_verification", "VERIFIED_EXTERNALLY"),
    ("blue_team_id", "unknown-team"), ("blue_team_id", "D"),
    ("analysis_id", "LOLTMNT03_179647"), ("patch", ""),
    ("planned_start_at", datetime(2026, 1, 3)),
])
def test_interactive_user_context_invalid_is_rejected(analysis_environment, field, value):
    _env, runtime, _clock = analysis_environment
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_analysis_pre(runtime, replace(_analysis_input(), **{field: value}))


def test_interactive_planned_start_has_no_cutoff_authority(analysis_environment):
    _env, runtime, clock = analysis_environment
    selection = _analysis_input()
    planned = replace(selection, planned_start_at=datetime(2025, 1, 1, tzinfo=UTC))
    first = demo.evaluate_analysis_pre(runtime, selection)
    second = demo.evaluate_analysis_pre(runtime, planned)
    assert first.prediction.history_cutoff_at == second.prediction.history_cutoff_at
    assert first.prediction.p_blue_win == second.prediction.p_blue_win
    assert first.context_key != second.context_key
    assert second.request.runtime_context.pre_created_at == clock["now"]
    with pytest.raises(TypeError):
        demo.evaluate_analysis_pre(runtime, selection, pre_created_at=planned.planned_start_at)


def test_interactive_clock_backwards_is_rejected(analysis_environment):
    _env, runtime, clock = analysis_environment
    clock["now"] = runtime.runtime_ready_at - timedelta(microseconds=1)
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_analysis_pre(runtime, _analysis_input())


def test_interactive_application_binds_exact_proof_availability(analysis_environment):
    _env, runtime, _clock = analysis_environment
    pre = demo.evaluate_analysis_pre(runtime, _analysis_input())
    # A real service-created snapshot can assert a plausible but wrong retrieval
    # instant. The application must still bind it to the actual retained proofs.
    claimed = replace(pre.request.runtime_context,
                      history_available_at=pre.request.runtime_context.history_available_at
                      - timedelta(seconds=1))
    other = evaluation.evaluate_interactive_pre(
        replace(pre.request, runtime_context=claimed), runtime.bundle,
    )
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_analysis_post(runtime, other, _post_ids())


@pytest.mark.parametrize("kind", ["snapshot", "evidence", "history", "availability", "bundle"])
def test_interactive_runtime_mutation_is_rejected(analysis_environment, kind):
    _env, runtime, _clock = analysis_environment
    pre = demo.evaluate_analysis_pre(runtime, _analysis_input())
    if kind == "snapshot":
        changed = replace(runtime, snapshot_sha256="b" * 64)
    elif kind == "evidence":
        changed = replace(runtime, evidence_sha256="b" * 64)
    elif kind == "history":
        changed = replace(runtime, history=runtime.history[:-1])
    elif kind == "availability":
        changed = replace(runtime, runtime_ready_at=runtime.runtime_ready_at - timedelta(days=1000))
    else:
        changed = replace(runtime, bundle=replace(runtime.bundle, identity=replace(
            runtime.bundle.identity, bundle_version="other",
        )))
    with pytest.raises(PreFeatureInputError):
        demo.evaluate_analysis_post(changed, pre, _post_ids())


@pytest.mark.parametrize("change", ["teams", "role", "patch", "planned", "analysis", "confirmation"])
def test_interactive_context_state_invalidates_both(analysis_environment, change):
    _env, runtime, clock = analysis_environment
    selection = _analysis_input()
    pre = demo.evaluate_analysis_pre(runtime, selection)
    post = demo.evaluate_analysis_post(runtime, pre, _post_ids())
    state = evaluation.EvaluationState(pre.context_key, _post_ids(), pre, post)
    if change == "teams":
        changed = replace(selection, blue_team_id="D", red_team_id="A",
                          blue_roster=selection.red_roster, red_roster=selection.blue_roster)
    elif change == "role":
        roster = selection.blue_roster
        changed = replace(selection, blue_roster=(
            replace(roster[0], player_id=roster[1].player_id),
            replace(roster[1], player_id=roster[0].player_id), *roster[2:],
        ))
    elif change == "patch":
        changed = replace(selection, patch="26.01")
    elif change == "planned":
        changed = replace(selection, planned_start_at=clock["now"] + timedelta(hours=1))
    elif change == "analysis":
        changed = replace(selection, analysis_id="analysis:new-session")
    else:
        changed = replace(selection, user_confirmed=False)
    result = demo.refresh_analysis_state(state, changed, runtime)
    assert result.pre is None and result.post is None


def test_interactive_champion_state_keeps_pre_without_new_clock_read(analysis_environment):
    _env, runtime, clock = analysis_environment
    selection = _analysis_input()
    pre = demo.evaluate_analysis_pre(runtime, selection)
    post = demo.evaluate_analysis_post(runtime, pre, _post_ids())
    state = evaluation.EvaluationState(pre.context_key, _post_ids(), pre, post)
    assert demo.refresh_analysis_state(state, selection, runtime) is state
    changed = demo.refresh_analysis_state(state, selection, runtime, champion_ids=(None,) * 10)
    assert changed.pre is pre and changed.post is None
    assert clock["reads"] == 2


def test_interactive_roster_suggestion_is_one_historical_roster(analysis_environment):
    _env, runtime, clock = analysis_environment
    suggestion = demo.suggest_analysis_roster(runtime, "A")
    assert suggestion is not None
    source = next(record.game for record in runtime.history
                  if record.game.game_id == suggestion.source_game_id)
    expected = source.blue_roster if source.blue_team_id == "A" else source.red_roster
    assert suggestion.roster == expected
    assert suggestion.source_ended_at == source.ended_at < suggestion.history_cutoff_at
    assert suggestion.source == "HISTORICAL_ROSTER_NOT_CURRENT_VERIFIED"
    eligible = [record.game for record in runtime.history if "A" in (
        record.game.blue_team_id, record.game.red_team_id,
    ) and record.game.ended_at < demo.interactive_cutoff(clock["now"])]
    assert suggestion.source_ended_at == max(game.ended_at for game in eligible)


@pytest.mark.parametrize("end,accepted", [
    (datetime(2026, 1, 1, 23, 59, tzinfo=UTC), True),
    (datetime(2026, 1, 2, tzinfo=UTC), False),
    (datetime(2026, 1, 2, 1, tzinfo=UTC), False),
])
def test_interactive_proof_pool_and_cutoff_selection_are_distinct(
    demo_environment, monkeypatch, end, accepted,
):
    env = demo_environment
    doc = deepcopy(env.document)
    game_id = "synthetic-h1"
    proof = doc["time_proofs"][game_id]
    start = end - timedelta(hours=1)
    proof.update(start_timestamp_ms=_milliseconds(start), end_timestamp_ms=_milliseconds(end),
                 started_at=start.isoformat(), ended_at=end.isoformat(),
                 fetched_at_utc=datetime(2026, 1, 2, 12, tzinfo=UTC).isoformat())
    for raw in doc["records"]:
        if raw["game_id"] == game_id:
            raw["history_cutoff_at"] = real_training.retrospective_cutoff(start).isoformat()
    env.write_input(doc)
    monkeypatch.setattr(demo, "_utc_now", lambda: datetime(2026, 1, 3, 12, tzinfo=UTC))
    runtime = demo.load_analysis_runtime(input_path=env.input_path, model_path=env.model_path)
    pre = demo.evaluate_analysis_pre(runtime, _analysis_input())
    assert game_id in pre.request.runtime_context.history_pool_game_ids
    assert (game_id in pre.features.accepted_game_ids) is accepted
    if not accepted:
        assert any(row.game_id == game_id and row.reason == "END_NOT_BEFORE_CUTOFF"
                   for row in pre.features.exclusions)
    summary = demo.analysis_history_summary(runtime, pre)
    assert summary["proof_pool"]["game_count"] == len(runtime.history)
    assert summary["before_cutoff"]["game_count"] == pre.features.counts.accepted_games


def test_interactive_missing_proof_excludes_source_not_user_target(demo_environment, monkeypatch):
    env = demo_environment
    doc = deepcopy(env.document)
    del doc["time_proofs"]["synthetic-h1"]
    doc["records"] = [row for row in doc["records"] if row["game_id"] != "synthetic-h1"]
    env.write_input(doc)
    monkeypatch.setattr(demo, "_utc_now", lambda: datetime(2026, 1, 3, 12, tzinfo=UTC))
    runtime = demo.load_analysis_runtime(input_path=env.input_path, model_path=env.model_path)
    assert "synthetic-h1" not in {record.game.game_id for record in runtime.history}
    pre = demo.evaluate_analysis_pre(runtime, _analysis_input())
    assert "synthetic-h1" not in pre.features.accepted_game_ids


@pytest.mark.parametrize("change", ["future_fetch", "missing_fetch", "conflicting_ms", "duplicate"])
def test_interactive_bad_provenance_rejected_before_pre(demo_environment, monkeypatch, change):
    env = demo_environment
    doc = deepcopy(env.document)
    proof = doc["time_proofs"]["synthetic-h1"]
    if change == "future_fetch":
        proof["fetched_at_utc"] = "2027-01-01T00:00:00+00:00"
    elif change == "missing_fetch":
        del proof["fetched_at_utc"]
    elif change == "conflicting_ms":
        proof["end_timestamp_ms"] += 1000
    else:
        doc["records"].append(deepcopy(doc["records"][0]))
    env.write_input(doc)
    monkeypatch.setattr(demo, "_utc_now", lambda: datetime(2026, 1, 3, 12, tzinfo=UTC))
    with pytest.raises(PreFeatureInputError):
        demo.load_analysis_runtime(input_path=env.input_path, model_path=env.model_path)


@pytest.fixture
def selection_environment(analysis_environment):
    """Synthetic namespaces deliberately oppose a prefix-based eligibility rule."""
    env, runtime, clock = analysis_environment
    rows = deepcopy(env.rows)
    for row in rows["team"]:
        if row["team_id"] == "A":
            row.update(team_id="lp_team_linked", display_name="Same team", logo_file="same-logo.png")
    rows["team"].append({
        "team_id": "oe:team:unlinked", "display_name": "Same team", "logo_file": "same-logo.png",
    })
    for row in rows["player"]:
        if row["player_id"] == "a1":
            row.update(player_id="lp_player_linked", display_name="Same player",
                       photo_file="same-photo.png")
    rows["player"].append({
        "player_id": "oe:player:unlinked", "display_name": "Same player",
        "photo_file": "same-photo.png",
    })
    catalog = demo_catalog.map_demo_catalog(rows)

    def team(identity):
        return "lp_team_linked" if identity == "A" else identity

    def player(identity):
        return "lp_player_linked" if identity == "a1" else identity

    history = tuple(replace(
        record,
        game=replace(record.game,
                     blue_team_id=team(record.game.blue_team_id),
                     red_team_id=team(record.game.red_team_id),
                     winner_team_id=team(record.game.winner_team_id),
                     blue_roster=tuple(replace(slot, player_id=player(slot.player_id))
                                       for slot in record.game.blue_roster),
                     red_roster=tuple(replace(slot, player_id=player(slot.player_id))
                                      for slot in record.game.red_roster)),
        slots=tuple(replace(slot, team_id=team(slot.team_id), player_id=player(slot.player_id))
                    for slot in record.slots),
    ) for record in runtime.history)
    runtime = replace(runtime, catalog=catalog, history=history,
                      history_pool_sha256=demo._pool_digest(
                          history, runtime.proof_fetched_at, runtime.history_patches,
                      ))
    selection = _analysis_input()
    selection = replace(selection, blue_team_id="lp_team_linked", blue_roster=tuple(
        replace(slot, player_id=player(slot.player_id)) for slot in selection.blue_roster
    ))
    return env, runtime, clock, selection


def test_selection_coverage_uses_exact_ids_not_namespace_name_or_media(selection_environment):
    env, runtime, clock, _selection = selection_environment
    before = deepcopy(runtime.history)
    coverage = demo.selection_coverage(runtime)
    assert coverage.history_cutoff_at == demo.interactive_cutoff(clock["now"])
    assert coverage.history_pool_sha256 == runtime.history_pool_sha256
    assert len(coverage.accepted_game_ids) == 4
    # A participates in three distinct games, not fifteen player rows.
    assert coverage.team_counts["lp_team_linked"] == 3
    assert coverage.player_counts["lp_player_linked"] == 3
    assert coverage.team_counts["oe:team:unlinked"] == 0
    assert coverage.player_counts["oe:player:unlinked"] == 0
    for kind, linked, unlinked in (
        ("team", "lp_team_linked", "oe:team:unlinked"),
        ("player", "lp_player_linked", "oe:player:unlinked"),
    ):
        options = demo.selection_options(runtime, coverage, entity_kind=kind)
        assert linked in options and unlinked not in options
        all_options = demo.selection_options(runtime, coverage, entity_kind=kind, linked_only=False)
        assert linked in all_options and unlinked in all_options
        assert len(all_options) == len(set(all_options))
    with pytest.raises(TypeError):
        coverage.team_counts["lp_team_linked"] = 999
    assert runtime.history == before
    assert env.calls == {"db": 1, "loader": 1}


def test_selection_coverage_excludes_equal_future_and_keeps_one_complete_roster(
    selection_environment,
):
    _env, runtime, clock, _selection = selection_environment
    cutoff = demo.interactive_cutoff(clock["now"])
    ends = {
        "demo-1": (cutoff - timedelta(minutes=1)).astimezone(timezone(timedelta(hours=7))),
        "synthetic-h1": cutoff,
        "synthetic-h2": cutoff + timedelta(minutes=1),
    }
    history = tuple(replace(record, game=replace(
        record.game, ended_at=ends.get(record.game.game_id, record.game.ended_at),
    )) for record in runtime.history)
    fetched = tuple((game_id, clock["now"]) for game_id, _value in runtime.proof_fetched_at)
    runtime = replace(runtime, history=history, proof_fetched_at=fetched,
                      history_pool_sha256=demo._pool_digest(history, fetched, runtime.history_patches))
    coverage = demo.selection_coverage(runtime)
    assert coverage.team_counts["lp_team_linked"] == 1
    assert coverage.player_counts["lp_player_linked"] == 1
    assert "synthetic-h1" not in coverage.accepted_game_ids
    assert "synthetic-h2" not in coverage.accepted_game_ids
    suggestion = demo.suggest_analysis_roster(runtime, "lp_team_linked", coverage=coverage)
    assert suggestion.source_game_id == "demo-1"
    assert suggestion.source_ended_at == cutoff - timedelta(minutes=1)
    source = next(record.game for record in history if record.game.game_id == "demo-1")
    assert suggestion.roster == source.blue_roster
    assert suggestion.roster[0].player_id == "lp_player_linked"
    assert all(slot.player_id != "oe:player:unlinked" for slot in suggestion.roster)
    assert suggestion.source == "HISTORICAL_ROSTER_NOT_CURRENT_VERIFIED"


@pytest.mark.parametrize("change", ["duplicate", "conflict", "digest"])
def test_selection_coverage_rejects_changed_or_duplicate_history(selection_environment, change):
    _env, runtime, _clock, _selection = selection_environment
    previous = demo.selection_coverage(runtime)
    record = runtime.history[0]
    if change == "conflict":
        other_winner = (record.game.red_team_id if record.game.winner_team_id == record.game.blue_team_id
                        else record.game.blue_team_id)
        record = replace(record, game=replace(record.game, winner_team_id=other_winner))
    if change == "digest":
        changed = replace(runtime, history_pool_sha256="0" * 64)
    else:
        history = (*runtime.history, record)
        changed = replace(runtime, history=history, history_pool_sha256=demo._pool_digest(
            history, runtime.proof_fetched_at, runtime.history_patches,
        ))
    with pytest.raises(PreFeatureInputError):
        demo.selection_coverage(changed, previous=previous)


def test_selection_coverage_cache_binds_runtime_cutoff_and_saved_pre(
    selection_environment, monkeypatch,
):
    env, runtime, clock, selection = selection_environment
    builds = []
    original_build = demo._build_selection_coverage

    def build(*args):
        builds.append(1)
        return original_build(*args)

    monkeypatch.setattr(demo, "_build_selection_coverage", build)
    preview = demo.selection_coverage(runtime)
    assert demo.selection_coverage(runtime, previous=preview) is preview
    pre = demo.evaluate_analysis_pre(runtime, selection)
    locked = demo.selection_coverage(runtime, pre=pre, previous=preview)
    assert locked.team_counts is preview.team_counts
    assert locked.roster_suggestions is preview.roster_suggestions
    saved_clock_reads = clock["reads"]
    clock["now"] += timedelta(days=2)
    assert demo.selection_coverage(runtime, pre=pre, previous=locked) is locked
    assert clock["reads"] == saved_clock_reads
    assert locked.history_cutoff_at == pre.prediction.history_cutoff_at
    assert len(builds) == 1
    newer = demo.selection_coverage(runtime, previous=locked)
    assert newer.history_cutoff_at == demo.interactive_cutoff(clock["now"])
    assert newer.history_cutoff_at != locked.history_cutoff_at
    assert len(builds) == 2
    # Even equal content in another session must not retain the old runtime binding.
    replacement = replace(runtime)
    rebound = demo.selection_coverage(replacement, previous=newer)
    assert rebound is not newer and len(builds) == 3
    with pytest.raises(PreFeatureInputError):
        demo.selection_options(replacement, newer, entity_kind="team")
    assert env.calls == {"db": 1, "loader": 1}


def test_manual_roster_without_team_history_keeps_player_ids_and_pair_history(selection_environment):
    _env, runtime, _clock, selection = selection_environment
    coverage = demo.selection_coverage(runtime)
    assert demo.suggest_analysis_roster(runtime, "oe:team:unlinked", coverage=coverage) is None
    # User may choose catalog players independently; no inferred current membership.
    manual = replace(selection, blue_team_id="oe:team:unlinked")
    pre = demo.evaluate_analysis_pre(runtime, manual)
    post = demo.evaluate_analysis_post(runtime, pre, _post_ids())
    assert pre.features.blue.recent_form.missing
    assert pre.features.blue.recent_form.sample_count == 0
    assert pre.features.h2h_blue.missing
    pair = next(stat for stat in post.features.player_champion
                if stat.player_id == "lp_player_linked")
    assert pair.games_count == 3 and pair.wins_count == 2
    assert pair.win_rate == pytest.approx(2 / 3)
    assert not pair.missing


def test_same_name_player_id_change_invalidates_context_and_changes_exact_pair_history(
    selection_environment,
):
    _env, runtime, _clock, selection = selection_environment
    pre = demo.evaluate_analysis_pre(runtime, selection)
    post = demo.evaluate_analysis_post(runtime, pre, _post_ids())
    state = evaluation.EvaluationState(pre.context_key, _post_ids(), pre, post)
    changed = replace(selection, blue_roster=(
        replace(selection.blue_roster[0], player_id="oe:player:unlinked"), *selection.blue_roster[1:],
    ))
    refreshed = demo.refresh_analysis_state(state, changed, runtime)
    assert refreshed.pre is None and refreshed.post is None
    replacement_pre = demo.evaluate_analysis_pre(runtime, changed)
    replacement_post = demo.evaluate_analysis_post(runtime, replacement_pre, _post_ids())
    pair = next(stat for stat in replacement_post.features.player_champion
                if stat.player_id == "oe:player:unlinked")
    assert pair.games_count == 0 and pair.wins_count == 0
    assert pair.win_rate is None and pair.missing
    assert replacement_pre.context_key != pre.context_key
