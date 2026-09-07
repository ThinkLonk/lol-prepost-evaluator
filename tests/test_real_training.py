"""Synthetic integration fixtures only; no production DB/network or temporal evidence."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import match_insight.ml.real_training as runner
import scripts.train_real_models as cli
from match_insight.database.real_snapshot import (
    SELECTS,
    map_snapshot,
    pre_context_sha256,
    read_snapshot,
)
from match_insight.features.pre import ROLES, PreFeatureInputError
from match_insight.ml.models import (
    CONTEXT_COLUMNS,
    ModelConfig,
    SelectionPolicy,
    predict_paired,
)

BASE = datetime(2025, 7, 1, 12, tzinfo=UTC)
POLICY = "test-fixture-explicit-pre-policy"
POLICIES = frozenset({POLICY})


def make_rows(count=20):
    rows = {model.__tablename__: [] for model, _, _ in SELECTS}
    rows["tournament"] = [{"tournament_id": "oracle:tournament:lck:2025"}]
    rows["tournament_stage"] = [
        {
            "stage_id": "oracle:stage:lck:2025:spring:0",
            "tournament_id": "oracle:tournament:lck:2025",
        }
    ]
    rows["game_patch"] = [{"patch_id": "patch-key", "patch_name": "15.01"}]
    rows["team"] = [{"team_id": team} for team in ("A", "B")]
    rows["player"] = [
        {"player_id": f"{team.lower()}{number}"}
        for team in ("A", "B")
        for number in range(1, 6)
    ]
    rows["champion"] = [{"champion_id": str(number)} for number in range(1, 11)]

    for index in range(count):
        game_id = f"g{index:02d}"
        rows["game"].append(
            {
                "game_id": game_id,
                "series_id": None,
                "stage_id": "oracle:stage:lck:2025:spring:0",
                "patch_id": "patch-key",
                "started_at": BASE + timedelta(days=index, hours=1),
                "ended_at": BASE + timedelta(days=index, hours=2),
                "winner_team_id": "A" if index % 2 == 0 else "B",
            }
        )
        for side_index, (side, team) in enumerate((("BLUE", "A"), ("RED", "B"))):
            parent = index * 2 + side_index + 1
            rows["game_team"].append(
                {
                    "game_team_id": parent,
                    "game_id": game_id,
                    "team_id": team,
                    "side": side,
                }
            )
            for number, role in enumerate(ROLES, start=1):
                slot_number = side_index * 5 + number
                rows["game_player"].append(
                    {
                        "game_player_id": index * 10 + slot_number,
                        "game_team_id": parent,
                        "player_id": f"{team.lower()}{number}",
                        "role": role,
                        "champion_id": str(slot_number),
                    }
                )
    return rows


def make_evidence(snapshot):
    # Test-only external assertions; these are not production source approvals.
    document = {
        "schema_version": runner.INPUT_SCHEMA,
        "snapshot_sha256": snapshot.sha256,
        "ended_at_provenance": {
            "evidence_ref": "test-fixture://explicit-end",
            "verification": "VERIFIED_EXTERNALLY",
        },
        "records": [
            {
                "game_id": item.game.game_id,
                "history_cutoff_at": (BASE + timedelta(days=index)).isoformat(),
                "evidence_ref": f"test-fixture://cutoff/{item.game.game_id}",
                "policy_version": POLICY,
                "verification": "VERIFIED_EXTERNALLY",
                "pre_context_sha256": pre_context_sha256(item),
                "pre_context_evidence_ref": f"test-fixture://pre/{item.game.game_id}",
                "pre_context_verification": "VERIFIED_EXTERNALLY",
            }
            for index, item in enumerate(snapshot.games)
        ],
    }
    data = json.dumps(document).encode("utf-8")
    return runner.EvidenceArtifact(
        "LOADED", runner.hashlib.sha256(data).hexdigest(), document
    )


@pytest.fixture
def snapshot():
    return map_snapshot(make_rows(), BASE, "test-fixture:database")


def forbid_fit_or_save(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Blocked/dry-run input reached fitting or persistence")

    monkeypatch.setattr(runner, "fit_model_bundle", forbidden)
    monkeypatch.setattr(runner, "save_bundle", forbidden)


@pytest.mark.parametrize(
    "source_ref",
    (None, "", " ", " padded ", "x" * 101),
)
def test_invalid_source_reference_is_rejected(source_ref):
    with pytest.raises(PreFeatureInputError) as caught:
        map_snapshot(make_rows(1), BASE, source_ref)

    assert caught.value.code == "E_DB_SOURCE_REF_INVALID"


def test_mapping_is_canonical_and_does_not_promote_pre_context():
    rows = make_rows(2)
    before = deepcopy(rows)
    first = map_snapshot(rows, BASE, "test-fixture:database")
    reversed_rows = {key: list(reversed(values)) for key, values in rows.items()}
    second = map_snapshot(
        reversed_rows, BASE + timedelta(hours=1), "test-fixture:database"
    )

    assert rows == before
    assert first.sha256 == second.sha256
    assert first.games == second.games
    assert first.captured_at != second.captured_at
    assert first.pre_context_verification == "UNVERIFIED"
    assert first.games[0].game.blue_team_id == "A"
    assert first.games[0].game.blue_roster[0].player_id == "a1"
    assert first.games[0].final_lineup.patch == "15.01"
    assert first.games[0].final_lineup.slots[-1].champion_id == "10"


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("duplicate", "E_DB_DUPLICATE_ID"),
        ("side", "E_DB_GAME_SIDES"),
        ("parent", "E_DB_PARENT"),
        ("winner", "E_DB_WINNER"),
        ("naive", "E_TIMESTAMP_INVALID"),
        ("partial", "E_DB_TIME_PAIR"),
        ("ordering", "E_DB_TIME_ORDER"),
        ("role", "E_LINEUP_INVALID"),
        ("champion", "E_DB_PARENT"),
        ("db_id", "E_DB_ID_INVALID"),
    ),
)
def test_invalid_snapshot_is_rejected(case, reason):
    rows = make_rows(1)
    if case == "duplicate":
        rows["game"].append(dict(rows["game"][0]))
    elif case == "side":
        rows["game_team"][1]["side"] = "BLUE"
    elif case == "parent":
        rows["game_player"][0]["game_team_id"] = 999
    elif case == "winner":
        rows["game"][0]["winner_team_id"] = "OTHER"
    elif case == "naive":
        rows["game"][0]["ended_at"] = datetime(2025, 7, 1, 14)
    elif case == "partial":
        rows["game"][0]["started_at"] = None
    elif case == "ordering":
        rows["game"][0]["ended_at"] = rows["game"][0]["started_at"]
    elif case == "role":
        rows["game_player"].pop()
    elif case == "champion":
        rows["game_player"][0]["champion_id"] = "999"
    else:
        rows["tournament"][0]["tournament_id"] = "   "

    with pytest.raises(PreFeatureInputError) as caught:
        map_snapshot(rows, BASE)
    assert caught.value.code == reason


class FakeResult:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def mappings(self):
        return self

    def all(self):
        return self.value


class FakeConnection:
    def __init__(self, rows, failure=None):
        self.rows = rows
        self.failure = failure
        self.events = []
        self.closed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def execution_options(self, **kwargs):
        assert kwargs == {"isolation_level": "REPEATABLE READ"}
        return self

    def begin(self):
        return self

    def rollback(self):
        self.rolled_back = True

    def exec_driver_sql(self, statement):
        self.events.append(statement)
        if self.failure == statement:
            raise RuntimeError("test-only database failure")
        if statement == "SHOW transaction_read_only":
            return FakeResult("off" if self.failure == "readonly" else "on")
        if statement == "SELECT transaction_timestamp()":
            return FakeResult(BASE)
        return FakeResult(None)

    def execute(self, statement):
        if self.failure == "select":
            raise RuntimeError("test-only SELECT failure")
        table = statement.get_final_froms()[0].name
        self.events.append(f"SELECT {table}")
        return FakeResult(self.rows[table])


def fake_engine(connection):
    return SimpleNamespace(
        dialect=SimpleNamespace(name="postgresql"),
        connect=lambda: connection,
    )


@pytest.mark.parametrize(
    "failure",
    (None, "SET TRANSACTION READ ONLY", "SHOW transaction_read_only", "readonly", "select", "map"),
)
def test_reader_checks_readonly_and_always_releases_connection(failure):
    rows = make_rows(1)
    if failure == "map":
        rows["game"].append(dict(rows["game"][0]))
    connection = FakeConnection(rows, failure)

    if failure is None:
        result = read_snapshot(fake_engine(connection))
        assert len(result.games) == 1
        assert connection.events[:3] == [
            "SET TRANSACTION READ ONLY",
            "SHOW transaction_read_only",
            "SELECT transaction_timestamp()",
        ]
    else:
        with pytest.raises((PreFeatureInputError, RuntimeError)):
            read_snapshot(fake_engine(connection))

    assert connection.rolled_back
    assert connection.closed
    if failure == "readonly":
        assert not any(event.startswith("SELECT") for event in connection.events)


def test_missing_end_is_preserved_and_excluded_by_existing_core(monkeypatch):
    rows = make_rows()
    rows["game"][0]["started_at"] = None
    rows["game"][0]["ended_at"] = None
    snapshot = map_snapshot(rows, BASE)
    forbid_fit_or_save(monkeypatch)
    plan = runner.prepare_run(snapshot, make_evidence(snapshot), POLICIES)

    assert snapshot.history[0].game.ended_at is None
    assert plan.report["historical_games_with_ended_at"] == 19
    assert plan.report["missing_end_game_ids"] == ("g00",)
    assert plan.report["paired_samples"] == 19
    assert any(
        item["game_id"] == "g00" and item["reason"] == "E_LABEL_END_MISSING"
        for item in plan.report["exclusions"]
    )
    assert plan.dataset.metadata.at["g01", "history_counts"].missing_end_games == 1


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("missing", "E_CUTOFF_MISSING"),
        ("unverified", "E_CUTOFF_UNVERIFIED"),
        ("duplicate", "E_CUTOFF_CONFLICT"),
        ("context_unverified", "E_PRE_CONTEXT_UNVERIFIED"),
        ("context_mismatch", "E_PRE_CONTEXT_MISMATCH"),
    ),
)
def test_ineligible_targets_never_fit_or_save(case, reason, monkeypatch, tmp_path):
    snapshot = map_snapshot(make_rows(1), BASE)
    evidence = make_evidence(snapshot)
    records = evidence.document["records"]
    if case == "missing":
        records.clear()
    elif case == "unverified":
        records[0]["verification"] = "UNVERIFIED"
    elif case == "duplicate":
        records.append(dict(records[0]))
    elif case == "context_unverified":
        records[0]["pre_context_verification"] = "UNVERIFIED"
    else:
        records[0]["pre_context_sha256"] = "0" * 64

    forbid_fit_or_save(monkeypatch)
    report = runner.execute_run(
        snapshot, evidence, POLICIES, train=True, output_directory=tmp_path
    )
    assert report["status"] == "NO_ELIGIBLE_TARGETS"
    assert report["paired_samples"] == 0
    assert report["exclusions"][0]["reason"] == reason
    assert list(tmp_path.iterdir()) == []


def test_unknown_exact_cutoff_id_blocks_the_run(snapshot, monkeypatch):
    evidence = make_evidence(snapshot)
    evidence.document["records"][0]["game_id"] = "unknown-exact-id"
    forbid_fit_or_save(monkeypatch)
    report = runner.execute_run(snapshot, evidence, POLICIES)
    assert report["status"] == "BLOCKED"
    assert {
        "reason": "E_CUTOFF_UNKNOWN_GAME_ID",
        "game_id": "unknown-exact-id",
    } in report["blockers"]


def test_dry_run_calls_real_dataset_split_without_fitting(snapshot, monkeypatch, tmp_path):
    forbid_fit_or_save(monkeypatch)
    report = runner.execute_run(
        snapshot, make_evidence(snapshot), POLICIES, output_directory=tmp_path
    )
    assert report["status"] == "DRY_RUN_READY"
    assert report["paired_samples"] == 20
    assert report["partition_counts"] == {"train": 14, "validation": 3, "test": 3}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("case", ("snapshot_hash", "end_evidence", "policy"))
def test_unapproved_global_evidence_cannot_train(case, snapshot, monkeypatch):
    evidence = make_evidence(snapshot)
    policies = POLICIES
    if case == "snapshot_hash":
        evidence.document["snapshot_sha256"] = "0" * 64
    elif case == "end_evidence":
        evidence.document["ended_at_provenance"]["verification"] = "UNVERIFIED"
    else:
        policies = frozenset()

    forbid_fit_or_save(monkeypatch)
    report = runner.execute_run(snapshot, evidence, policies, train=True)
    assert report["status"] in ("BLOCKED", "NO_ELIGIBLE_TARGETS")


def test_empty_and_insufficient_datasets_do_not_fit(monkeypatch):
    forbid_fit_or_save(monkeypatch)
    for count, expected in ((0, "NO_ELIGIBLE_TARGETS"), (2, "INSUFFICIENT_SPLIT")):
        snapshot = map_snapshot(make_rows(count), BASE)
        report = runner.execute_run(
            snapshot, make_evidence(snapshot), POLICIES, train=True
        )
        assert report["status"] == expected


def test_evidence_file_missing_and_duplicate_json_keys(tmp_path):
    path = tmp_path / "inputs.json"
    assert runner.load_evidence(path).status == "MISSING"
    assert not path.exists()

    path.write_text('{"records": [], "records": []}', encoding="utf-8")
    with pytest.raises(PreFeatureInputError):
        runner.load_evidence(path)


def test_train_preserves_core_policy_and_persistence(snapshot, monkeypatch, tmp_path):
    evidence = make_evidence(snapshot)
    plan = runner.prepare_run(snapshot, evidence, POLICIES)
    events = []
    original_fit = runner.fit_model_bundle
    original_evaluate = runner.evaluate_test
    original_save = runner.save_bundle

    def fit(dataset, split, identity, *, config, policy):
        events.append("fit")
        assert split == plan.split
        assert config == ModelConfig()
        assert policy == SelectionPolicy()
        return original_fit(dataset, split, identity, config=config, policy=policy)

    def evaluate(bundle, dataset):
        events.append("evaluate_test")
        return original_evaluate(bundle, dataset)

    def save(bundle, path):
        events.append("save")
        return original_save(bundle, path)

    monkeypatch.setattr(runner, "fit_model_bundle", fit)
    monkeypatch.setattr(runner, "evaluate_test", evaluate)
    monkeypatch.setattr(runner, "save_bundle", save)

    report = runner.execute_run(
        snapshot,
        evidence,
        POLICIES,
        train=True,
        expected_dataset_id=plan.report["dataset_id"],
        expected_split_id=plan.report["split_id"],
        output_directory=tmp_path,
    )
    assert report["status"] == "TRAINED"
    assert events == ["fit", "evaluate_test", "save"]
    assert {path.name for path in tmp_path.iterdir()} == {
        runner.MODEL_FILENAME,
        runner.METADATA_FILENAME,
    }

    loaded = runner.load_real_bundle(tmp_path / runner.MODEL_FILENAME)
    predictions = predict_paired(
        loaded,
        plan.dataset.X_pre,
        plan.dataset.X_post,
        plan.dataset.metadata.loc[:, list(CONTEXT_COLUMNS)],
    )
    assert len(predictions) == 20
    assert all(0 <= item.p_blue_win_pre <= 1 for item in predictions)
    assert all(0 <= item.p_blue_win_post <= 1 for item in predictions)

    metadata_path = tmp_path / runner.METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["data_kind"] = "SYNTHETIC"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PreFeatureInputError):
        runner.load_real_bundle(tmp_path / runner.MODEL_FILENAME)


def test_training_requires_matching_dry_run_identity(snapshot, monkeypatch, tmp_path):
    forbid_fit_or_save(monkeypatch)
    report = runner.execute_run(
        snapshot,
        make_evidence(snapshot),
        POLICIES,
        train=True,
        expected_dataset_id="different",
        expected_split_id="different",
        output_directory=tmp_path,
    )
    assert report["reason"] == "E_DRY_RUN_ID_MISMATCH"
    assert list(tmp_path.iterdir()) == []


def test_existing_artifact_is_not_overwritten_or_refitted(snapshot, monkeypatch, tmp_path):
    evidence = make_evidence(snapshot)
    plan = runner.prepare_run(snapshot, evidence, POLICIES)
    existing = tmp_path / runner.MODEL_FILENAME
    existing.write_bytes(b"existing-test-artifact")
    forbid_fit_or_save(monkeypatch)

    with pytest.raises(PreFeatureInputError) as caught:
        runner.execute_run(
            snapshot,
            evidence,
            POLICIES,
            train=True,
            expected_dataset_id=plan.report["dataset_id"],
            expected_split_id=plan.report["split_id"],
            output_directory=tmp_path,
        )
    assert caught.value.code == "E_MODEL_ARTIFACT_EXISTS"
    assert existing.read_bytes() == b"existing-test-artifact"


def test_cli_missing_input_has_no_synthetic_fallback(monkeypatch, tmp_path, capsys):
    def forbidden_snapshot():
        pytest.fail("Missing input must be rejected before DB access")

    monkeypatch.setattr(cli, "_runtime_snapshot", forbidden_snapshot)
    forbid_fit_or_save(monkeypatch)
    code = cli.main(["--cutoffs", str(tmp_path / "missing.json")])
    output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert output["status"] == "NO_ELIGIBLE_TARGETS"
    assert output["reason"] == "E_CUTOFF_MISSING"
    assert output["paired_samples"] == 0
    assert "db_source_games" not in output
    assert list(tmp_path.iterdir()) == []


def test_map_snapshot_accepts_namespaced_oracle_ids():
    rows = make_rows(1)
    rows["team"][0]["team_id"] = "oe:team:84bc703e28859788770611d94cf02ac"
    rows["game_team"][0]["team_id"] = "oe:team:84bc703e28859788770611d94cf02ac"
    rows["game"][0]["winner_team_id"] = "oe:team:84bc703e28859788770611d94cf02ac"
    rows["player"][0]["player_id"] = "oe:player:c659697694306de62d978569b84c344"
    rows["game_player"][0]["player_id"] = "oe:player:c659697694306de62d978569b84c344"

    snapshot = map_snapshot(rows, BASE)
    assert snapshot.games[0].game.blue_team_id == "oe:team:84bc703e28859788770611d94cf02ac"
    assert snapshot.games[0].game.blue_roster[0].player_id == "oe:player:c659697694306de62d978569b84c344"


def write_time_fixture(tmp_path, count=20):
    """Tiny synthetic files matching the inspected archive schema."""
    source = tmp_path / "time-source"
    source.mkdir()
    rows = make_rows(count)
    identity_map = {
        row["game_id"]: f"LOLTMNT01_{1000 + number}"
        for number, row in enumerate(rows["game"])
    }
    for row in rows["game"]:
        row["game_id"] = identity_map[row["game_id"]]
    for row in rows["game_team"]:
        row["game_id"] = identity_map[row["game_id"]]
    snapshot = map_snapshot(rows, BASE)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    exported = []
    summary = []
    for item in snapshot.games:
        game_id = item.game.game_id
        start_ms = (item.started_at - epoch) // timedelta(milliseconds=1)
        end_ms = (item.game.ended_at - epoch) // timedelta(milliseconds=1)
        record = {
            "game_id": game_id,
            "status": "SUCCESS",
            "start_timestamp_ms": start_ms,
            "end_timestamp_ms": end_ms,
            "started_at": item.started_at.isoformat(),
            "ended_at": item.game.ended_at.isoformat(),
        }
        summary.append(dict(record))
        exported.append(
            {
                **record,
                "fetched_at_utc": (BASE + timedelta(days=60)).isoformat(),
                "raw_payload": {
                    "platformId": "LOLTMNT01",
                    "gameId": int(game_id.split("_")[1]),
                    "gameStartTimestamp": start_ms,
                    "gameEndTimestamp": end_ms,
                },
            }
        )
    (source / runner.TIME_SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    for number, name in enumerate(runner.TIME_EXPORTS):
        (source / name).write_text(
            json.dumps(exported if number == 0 else []),
            encoding="utf-8",
        )
    return snapshot, source


def retrospective_evidence(snapshot, source):
    return runner.make_retrospective_evidence(
        snapshot,
        runner.load_time_archive(source),
        approval_ref="synthetic://review/retrospective-protocol",
    )


def change_json(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("start", "expected"),
    (
        ("2025-07-10T12:00:00+00:00", "2025-07-09T00:00:00+00:00"),
        ("2025-07-10T19:00:00+07:00", "2025-07-09T00:00:00+00:00"),
        ("2025-01-01T00:00:00+00:00", "2024-12-31T00:00:00+00:00"),
        ("2025-01-01T01:00:00+07:00", "2024-12-30T00:00:00+00:00"),
        ("2024-03-01T12:00:00+00:00", "2024-02-29T00:00:00+00:00"),
    ),
)
def test_retrospective_cutoff_utc_boundaries(start, expected):
    actual = runner.retrospective_cutoff(datetime.fromisoformat(start))
    assert actual == datetime.fromisoformat(expected)


@pytest.mark.parametrize("value", (None, "2025-07-01", datetime(2025, 7, 1)))
def test_retrospective_cutoff_rejects_missing_or_naive_start(value):
    with pytest.raises(PreFeatureInputError):
        runner.retrospective_cutoff(value)


def test_missing_revision_uses_own_archive_fetch_time(tmp_path):
    snapshot, source = write_time_fixture(tmp_path, 2)
    evidence = retrospective_evidence(snapshot, source)
    proofs = evidence.document["time_proofs"]
    assert len(proofs) == 2
    assert all(proof["revision_status"] == "MISSING" for proof in proofs.values())
    assert all(proof["grade"] == runner.TIME_EVIDENCE_GRADE for proof in proofs.values())
    assert "VERIFIED_EXTERNALLY" not in json.dumps(evidence.document)
    assert all("fetched_at_utc" in proof for proof in proofs.values())
    assert len(evidence.document["source_files"]) == 13


@pytest.mark.parametrize(
    ("case", "reason"),
    (
        ("missing_fetch", "E_TIME_TIMESTAMP_INVALID"),
        ("naive_fetch", "E_TIME_TIMESTAMP_INVALID"),
        ("float_ms", "E_TIME_MILLISECONDS_INVALID"),
        ("bool_ms", "E_TIME_MILLISECONDS_INVALID"),
        ("payload_time", "E_TIME_SOURCE_CONFLICT"),
        ("summary_time", "E_TIME_SOURCE_CONFLICT"),
        ("platform", "E_TIME_IDENTITY_CONFLICT"),
        ("metadata_identity", "E_TIME_IDENTITY_CONFLICT"),
        ("revision_partial", "E_TIME_REVISION_INVALID"),
        ("revision_conflict", "E_TIME_REVISION_CONFLICT"),
        ("duplicate_raw", "E_TIME_DUPLICATE_GAME_ID"),
        ("duplicate_summary", "E_TIME_DUPLICATE_GAME_ID"),
    ),
)
def test_time_provenance_errors_exclude_only_affected_game(tmp_path, case, reason):
    snapshot, source = write_time_fixture(tmp_path, 2)
    raw_path = source / runner.TIME_EXPORTS[0]
    summary_path = source / runner.TIME_SUMMARY

    def edit_raw(data):
        first = data[0]
        if case == "missing_fetch":
            first.pop("fetched_at_utc")
        elif case == "naive_fetch":
            first["fetched_at_utc"] = "2026-01-01T12:00:00"
        elif case == "float_ms":
            first["start_timestamp_ms"] = float(first["start_timestamp_ms"])
        elif case == "bool_ms":
            first["raw_payload"]["gameStartTimestamp"] = True
        elif case == "payload_time":
            first["raw_payload"]["gameEndTimestamp"] += 1
        elif case == "platform":
            first["raw_payload"]["platformId"] = "LOLTMNT02"
        elif case == "metadata_identity":
            first["raw_payload"]["metadata"] = {"matchId": "LOLTMNT01_9999"}
        elif case == "revision_partial":
            first["revision_id"] = 10
        elif case == "revision_conflict":
            first["revision_id"] = 10
            first["revision_timestamp_utc"] = "2025-07-01T15:00:00Z"
        elif case == "duplicate_raw":
            data.append(deepcopy(first))

    def edit_summary(data):
        if case == "summary_time":
            data[0]["end_timestamp_ms"] += 1
        elif case == "duplicate_summary":
            data.append(deepcopy(data[0]))
        elif case == "revision_conflict":
            data[0]["revision_id"] = 11
            data[0]["revision_timestamp_utc"] = "2025-07-01T15:00:00Z"

    change_json(raw_path, edit_raw)
    change_json(summary_path, edit_summary)
    proofs, excluded, _sources = runner.resolve_time_proofs(
        snapshot, runner.load_time_archive(source)
    )
    first_id = snapshot.games[0].game.game_id
    second_id = snapshot.games[1].game.game_id
    assert set(proofs) == {second_id}
    assert excluded[0]["game_id"] == first_id
    assert reason in excluded[0].get("diagnostics", (excluded[0]["reason"],))


def test_snapshot_comparison_preserves_milliseconds_and_timezone(tmp_path):
    snapshot, source = write_time_fixture(tmp_path, 2)
    archive = runner.load_time_archive(source)
    item = snapshot.games[0]
    offset = datetime.fromisoformat("2025-07-01T20:00:00+07:00").tzinfo
    equivalent = replace(
        item,
        started_at=item.started_at.astimezone(offset),
        game=replace(item.game, ended_at=item.game.ended_at.astimezone(offset)),
    )
    same = replace(snapshot, games=(equivalent, snapshot.games[1]))
    assert runner.resolve_time_proofs(same, archive)[0] == runner.resolve_time_proofs(
        snapshot, archive
    )[0]

    changed = replace(
        item,
        game=replace(item.game, ended_at=item.game.ended_at + timedelta(microseconds=1)),
    )
    conflicting = replace(snapshot, games=(changed, snapshot.games[1]))
    proofs, excluded, _sources = runner.resolve_time_proofs(conflicting, archive)
    assert item.game.game_id not in proofs
    assert excluded[0]["reason"] == "E_TIME_SNAPSHOT_CONFLICT"


@pytest.mark.parametrize("body", ('[{},]', '[{}] {}', '[{"game_id":', '{}'))
def test_time_reader_rejects_malformed_array(tmp_path, body):
    path = tmp_path / "bad.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(PreFeatureInputError):
        list(runner._iter_time_rows(path))


@pytest.mark.parametrize("case", ("missing", "schema", "protocol"))
def test_cli_rejects_input_before_db(tmp_path, monkeypatch, capsys, case):
    path = tmp_path / "input.json"
    if case != "missing":
        path.write_text(
            json.dumps(
                {
                    "schema_version": "wrong" if case == "schema" else runner.INPUT_SCHEMA,
                    "evaluation_protocol": runner.RETROSPECTIVE_PROTOCOL,
                    "records": [],
                }
            ),
            encoding="utf-8",
        )

    def forbidden():
        pytest.fail("Invalid input must not open DB")

    monkeypatch.setattr(cli, "_runtime_snapshot", forbidden)
    assert cli.main(["--cutoffs", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] in (
        "BLOCKED", "NO_ELIGIBLE_TARGETS"
    )


@pytest.mark.parametrize("case", ("cutoff", "snapshot", "duplicate", "proof", "context"))
def test_modified_retrospective_input_never_fits(tmp_path, monkeypatch, case):
    snapshot, source = write_time_fixture(tmp_path)
    evidence = retrospective_evidence(snapshot, source)
    record = evidence.document["records"][0]
    if case == "cutoff":
        record["history_cutoff_at"] = (
            datetime.fromisoformat(record["history_cutoff_at"]) + timedelta(hours=1)
        ).isoformat()
    elif case == "snapshot":
        evidence.document["snapshot_sha256"] = "0" * 64
    elif case == "duplicate":
        evidence.document["records"].append(deepcopy(record))
    elif case == "proof":
        evidence.document["time_proofs"][record["game_id"]]["end_timestamp_ms"] += 1
    else:
        record["pre_context_sha256"] = "0" * 64
    forbid_fit_or_save(monkeypatch)
    with pytest.raises(PreFeatureInputError):
        runner.execute_run(
            snapshot,
            evidence,
            frozenset({runner.RETROSPECTIVE_PROTOCOL}),
            train=True,
            output_directory=tmp_path,
            evaluation_protocol=runner.RETROSPECTIVE_PROTOCOL,
        )


def test_create_input_is_explicit_and_never_overwrites(tmp_path):
    snapshot, source = write_time_fixture(tmp_path)
    archive = runner.load_time_archive(source)
    path = tmp_path / "retrospective_pre_inputs.json"
    report = runner.write_retrospective_inputs(
        snapshot, archive, path, approval_ref="synthetic://approved-protocol"
    )
    before = path.read_bytes()
    assert report["status"] == "INPUT_CREATED"
    loaded = runner.load_evidence(
        path,
        evaluation_protocol=runner.RETROSPECTIVE_PROTOCOL,
        source_directory=source,
    )
    assert loaded.document["evaluation_protocol"] == runner.RETROSPECTIVE_PROTOCOL
    with pytest.raises(PreFeatureInputError):
        runner.load_evidence(path)
    with pytest.raises(PreFeatureInputError, match="E_INPUT_ARTIFACT_EXISTS"):
        runner.write_retrospective_inputs(
            snapshot, archive, path, approval_ref="synthetic://approved-protocol"
        )
    assert path.read_bytes() == before


def test_no_time_proofs_do_not_create_empty_input(tmp_path):
    snapshot, source = write_time_fixture(tmp_path, 1)
    change_json(source / runner.TIME_EXPORTS[0], lambda rows: rows.clear())
    path = tmp_path / "retrospective_pre_inputs.json"
    with pytest.raises(PreFeatureInputError, match="E_NO_ELIGIBLE_TARGETS"):
        runner.write_retrospective_inputs(
            snapshot,
            runner.load_time_archive(source),
            path,
            approval_ref="synthetic://approved-protocol",
        )
    assert not path.exists()


def test_retrospective_dry_run_is_pure_and_uses_reviewed_identity(tmp_path, monkeypatch):
    snapshot, source = write_time_fixture(tmp_path)
    evidence = retrospective_evidence(snapshot, source)
    before = deepcopy(evidence.document)
    forbid_fit_or_save(monkeypatch)
    policies = frozenset({runner.RETROSPECTIVE_PROTOCOL})
    report = runner.execute_run(
        snapshot, evidence, policies, evaluation_protocol=runner.RETROSPECTIVE_PROTOCOL
    )
    assert report["status"] == "DRY_RUN_READY"
    assert report["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert report["targets_with_verified_pre_context"] == 0
    assert evidence.document == before
    assert not (tmp_path / runner.RETROSPECTIVE_MODEL_FILENAME).exists()

    refused = runner.execute_run(
        snapshot,
        evidence,
        policies,
        train=True,
        expected_dataset_id="unreviewed",
        expected_split_id=report["split_id"],
        output_directory=tmp_path,
        evaluation_protocol=runner.RETROSPECTIVE_PROTOCOL,
    )
    assert refused["reason"] == "E_DRY_RUN_ID_MISMATCH"


def test_retrospective_training_persistence_keeps_its_contract(tmp_path):
    snapshot, source = write_time_fixture(tmp_path)
    evidence = retrospective_evidence(snapshot, source)
    protocol = runner.RETROSPECTIVE_PROTOCOL
    policies = frozenset({protocol})
    dry = runner.execute_run(snapshot, evidence, policies, evaluation_protocol=protocol)
    trained = runner.execute_run(
        snapshot,
        evidence,
        policies,
        train=True,
        expected_dataset_id=dry["dataset_id"],
        expected_split_id=dry["split_id"],
        output_directory=tmp_path,
        evaluation_protocol=protocol,
    )
    assert trained["status"] == "TRAINED"
    path = tmp_path / runner.RETROSPECTIVE_MODEL_FILENAME
    loaded = runner.load_retrospective_bundle(path)
    assert loaded.contract.cutoff_policy_versions == (protocol,)
    assert loaded.split.evaluation_protocol == protocol
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["data_kind"] == "REAL_RETROSPECTIVE_SIMULATION"
    assert metadata["evaluation_protocol"] == protocol
    assert metadata["run"]["targets_with_verified_pre_context"] == 0
    assert "partition_label_counts" in metadata["run"]
    assert metadata["run"]["partition_label_counts"]["train"]["total"] == len(
        loaded.split.train
    )
    with pytest.raises(PreFeatureInputError):
        runner.load_real_bundle(path)


def test_partition_label_counts_consistency(tmp_path):
    snapshot, source = write_time_fixture(tmp_path)
    evidence = retrospective_evidence(snapshot, source)
    protocol = runner.RETROSPECTIVE_PROTOCOL
    policies = frozenset({protocol})
    report = runner.execute_run(
        snapshot, evidence, policies, evaluation_protocol=protocol
    )
    assert report["status"] == "DRY_RUN_READY"
    counts = report["partition_label_counts"]
    split = report["split"]

    assert counts["train"]["total"] == len(split.train)
    assert counts["validation"]["total"] == len(split.validation)
    assert counts["test"]["total"] == len(split.test)

    for name in ("train", "validation", "test"):
        row = counts[name]
        assert row["blue_win_1"] + row["red_win_0"] == row["total"]
        if row["total"] > 0:
            assert 0.0 <= row["blue_win_rate"] <= 1.0
            assert row["blue_win_rate"] == pytest.approx(row["blue_win_1"] / row["total"])
        else:
            assert row["blue_win_rate"] is None
