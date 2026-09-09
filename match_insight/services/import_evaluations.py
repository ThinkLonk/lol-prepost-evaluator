"""Import existing canonical test predictions with reconstructed, checked snapshots.

This boundary never trains, changes artifacts or fabricates the original inference
time. A completed pair is committed atomically by EvaluationPersistence.
"""

import hashlib
import math
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import fields, replace
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pandas as pd
from sqlalchemy import func, select

from match_insight.database.models import Evaluation, EvaluationHistory, EvaluationWarning
from match_insight.database.real_snapshot import fail, read_snapshot
from match_insight.features.post import HistoricalChampionGame, build_post_features
from match_insight.features.pre import PreFeatureInputError, TargetGame, build_pre_features
from match_insight.ml.dataset import POST_COLUMNS, PRE_COLUMNS, RETROSPECTIVE_PROTOCOL
from match_insight.ml.models import (
    PairedPrediction,
    _digest,
    _model_frame,
    _plain,
    compare_predictions,
    predict_phase,
)
from match_insight.ml.real_training import (
    CANONICAL_RETROSPECTIVE_EXPECTATION,
    HASH_PATTERN,
    MODEL_DIRECTORY,
    RETROSPECTIVE_INPUT,
    RETROSPECTIVE_METADATA_FILENAME,
    _json_document,
    _time_iso,
    load_retrospective_bundle,
)
from match_insight.services.demo import (
    CANONICAL_DEMO_INPUT_SHA256,
    _read_pinned_input,
    _validated_history_sources,
)
from match_insight.services.evaluation import (
    BusinessWarning,
    EvaluationRequest,
    PostSnapshot,
    PreSnapshot,
    _pre_warnings,
    _prediction_inputs,
    _request_key,
    _require_retrospective_cutoff,
    _seal,
)
from match_insight.services.persistence import (
    EvaluationPersistence,
    SavedEvaluation,
    _history_hash,
    warning_rows,
)

CANONICAL_METADATA_SHA256 = "10d8684ed28c591934ca8d441a08b325c5dbee328e5afb1e2c8947a47b9530cd"
DEFAULT_METADATA = MODEL_DIRECTORY / RETROSPECTIVE_METADATA_FILENAME
_NUMBERS = ("p_blue_win_pre", "p_blue_win_post", "delta_probability")
_TOLERANCE = 1e-12
_WORKER_STATE = None


def _read_source(path):
    try:
        body = Path(path).read_bytes()
    except (OSError, TypeError, ValueError):
        fail("E_REAL_MODEL_METADATA_INVALID")
    if hashlib.sha256(body).hexdigest() != CANONICAL_METADATA_SHA256:
        fail("E_EVIDENCE_HASH_MISMATCH")
    return _json_document(body)


def _index_rows(rows, *, code):
    if not isinstance(rows, list) or not rows:
        fail(code)
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("game_id"), str):
            fail(code)
        if row["game_id"] in result:
            fail("E_SOURCE_CONFLICT", row["game_id"])
        result[row["game_id"]] = row
    return result


def _source_predictions(metadata, bundle):
    run = metadata.get("run", {})
    predictions = _index_rows(run.get("test_predictions"), code="E_MODEL_METADATA_INVALID")
    if tuple(predictions) != tuple(bundle.split.test):
        fail("E_MODEL_ALIGNMENT")
    expected_fields = {field.name for field in fields(PairedPrediction)}
    for game_id, row in predictions.items():
        if set(row) != expected_fields:
            fail("E_MODEL_METADATA_INVALID", game_id)
        if (
            row["model_bundle_version"] != bundle.identity.bundle_version
            or row["feature_schema_version"] != bundle.feature_schema_version
            or row["policy_version"] != RETROSPECTIVE_PROTOCOL
            or not isinstance(row["context_signature"], str)
            or HASH_PATTERN.fullmatch(row["context_signature"]) is None
        ):
            fail("E_EVAL_INCOMPATIBLE", game_id)
        _time_iso(row["history_cutoff_at"])
        if any(
            type(row[key]) not in (int, float) or not math.isfinite(row[key])
            for key in _NUMBERS
        ):
            fail("E_MODEL_VALUE_INVALID", game_id)
        if any(not 0 <= row[key] <= 1 for key in _NUMBERS[:2]) or not math.isclose(
            row["delta_probability"], row["p_blue_win_post"] - row["p_blue_win_pre"],
            rel_tol=0, abs_tol=_TOLERANCE,
        ):
            fail("E_MODEL_VALUE_INVALID", game_id)
    return predictions


def _prepare_import(engine, metadata_path, input_path):
    metadata = _read_source(metadata_path)
    bundle = load_retrospective_bundle(
        Path(metadata_path).with_suffix(".joblib"), expected=CANONICAL_RETROSPECTIVE_EXPECTATION,
    )
    predictions = _source_predictions(metadata, bundle)
    provenance = _index_rows(
        metadata["run"].get("target_provenance"), code="E_MODEL_METADATA_INVALID",
    )
    document = _read_pinned_input(input_path)
    if (
        metadata["run"].get("evidence_sha256") != CANONICAL_DEMO_INPUT_SHA256
        or metadata["run"].get("snapshot_sha256") != document["snapshot_sha256"]
    ):
        fail("E_EVIDENCE_SNAPSHOT_MISMATCH")
    snapshot = read_snapshot(engine)
    sources = _validated_history_sources(snapshot, document)
    by_id = {source[0].game.game_id: source for source in sources}
    raw_by_id = {row["game_id"]: row for row in document["records"]}
    for game_id, prediction in predictions.items():
        if game_id not in by_id or game_id not in provenance:
            fail("E_CUTOFF_UNKNOWN_GAME", game_id)
        cutoff = by_id[game_id][1]
        saved = provenance[game_id]
        raw = raw_by_id[game_id]
        if (
            _time_iso(prediction["history_cutoff_at"]) != cutoff.history_cutoff_at
            or _time_iso(saved.get("history_cutoff_at")) != cutoff.history_cutoff_at
            or any(saved.get(key) != raw[key] for key in (
                "evidence_ref", "policy_version", "pre_context_sha256",
                "pre_context_evidence_ref",
            ))
        ):
            fail("E_PRE_CONTEXT_MISMATCH", game_id)
    history = tuple(
        HistoricalChampionGame(replace(item.game, ended_at=end), item.final_lineup.slots)
        for item, _cutoff, _start, end, _fetched_at in sources
    )
    return bundle, snapshot, document, predictions, provenance, by_id, history


def _reconstruct_pair(source, history, reference, bundle):
    item, cutoff, start, _end, _fetched_at = source
    game = item.game
    target = TargetGame(
        game.game_id, game.blue_team_id, game.red_team_id, game.blue_roster,
        game.red_roster, item.final_lineup.patch, cutoff.history_cutoff_at,
    )
    # Keep the full accepted cohort, including excluded/equal/future/target rows.
    # Filtering it here would change the canonical history diagnostics/signature.
    request = EvaluationRequest(target, cutoff, history, reference)
    _require_retrospective_cutoff(request, start)
    # The DB/proof batch has already validated and frozen this shared source.
    # Rebuild actual core outputs without deepcopying 7,552 games for each phase;
    # persistence still validates the resulting request key, seal and comparison.
    features = build_pre_features(target, (row.game for row in history), bundle.contract.pre_config)
    request = replace(request, target=features.history_snapshot.target)
    frame, metadata = _prediction_inputs(request, features, bundle)
    prediction = predict_phase(bundle, "PRE", frame, metadata)[0]
    pre = PreSnapshot(request, features, prediction, _pre_warnings(features),
                      _request_key(request, bundle), "")
    pre = replace(pre, seal=_seal(pre))
    post_features = build_post_features(features, item.final_lineup, history, reference)
    frame, metadata = _prediction_inputs(request, features, bundle, post_features)
    prediction = predict_phase(bundle, "POST", frame, metadata)[0]
    missing_pairs = sum(feature.missing for feature in post_features.player_champion)
    warnings = pre.warnings
    if missing_pairs:
        warnings += (BusinessWarning("W_PAIR_HISTORY_MISSING", sample_count=missing_pairs),)
    post = PostSnapshot(pre, item.final_lineup, post_features, prediction,
                        compare_predictions(pre.prediction, prediction), warnings)
    return pre, post


def _check_prediction(post, expected):
    actual = _plain(post.comparison)
    for key in expected:
        if key in _NUMBERS:
            same = math.isclose(actual[key], expected[key], rel_tol=0, abs_tol=_TOLERANCE)
        elif key == "history_cutoff_at":
            same = _time_iso(actual[key]) == _time_iso(expected[key])
        else:
            same = actual[key] == expected[key]
        if not same:
            fail("E_EVAL_INCOMPATIBLE", expected["game_id"])


def _database_counts(engine, bundle):
    with engine.connect() as connection:
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            if connection.exec_driver_sql("SHOW transaction_read_only").scalar_one() != "on":
                fail("E_DB_NOT_READ_ONLY")
            condition = (
                Evaluation.origin == "RETROSPECTIVE_IMPORT",
                Evaluation.model_version == bundle.identity.bundle_version,
            )
            evaluations = connection.execute(
                select(func.count()).select_from(Evaluation).where(*condition),
            ).scalar_one()
            warnings = connection.execute(
                select(func.count()).select_from(EvaluationWarning)
                .join(Evaluation, Evaluation.evaluation_id == EvaluationWarning.evaluation_id)
                .where(*condition),
            ).scalar_one()
    return {"evaluations": int(evaluations), "warnings": int(warnings)}


def _provenance(game_id, position, expected, target_provenance, document, snapshot, metadata_path):
    return {
        "source_file": str(Path(metadata_path).name),
        "source_file_sha256": CANONICAL_METADATA_SHA256,
        "source_prediction_index": position,
        "source_prediction": expected,
        "source_target_provenance": target_provenance,
        "time_proof": document["time_proofs"][game_id],
        "pinned_input_sha256": CANONICAL_DEMO_INPUT_SHA256,
        "snapshot_sha256": snapshot.sha256,
        "reconstructed_from_existing_sources": True,
        "original_inference_time_status": "NOT_RECORDED",
    }


def _check_stored_pair(rows, warnings, expected, provenance, bundle, expected_history_hash):
    """Recheck immutable persisted content on retry without rebuilding or predicting."""
    game_id = expected["game_id"]
    by_phase = {row["evaluation_type"]: row for row in rows}
    if len(rows) != 2 or set(by_phase) != {"PRE", "POST"}:
        fail("E_STORAGE_CONFLICT", game_id)
    pre = by_phase["PRE"]
    post = by_phase["POST"]
    if pre["pre_evaluation_id"] is not None or post["pre_evaluation_id"] != pre["evaluation_id"]:
        fail("E_STORAGE_CONFLICT", game_id)
    pre_values = None
    for phase in ("PRE", "POST"):
        row = by_phase[phase]
        document = row["input_snapshot"]
        stored_provenance = row["provenance"]
        expected_probability = expected[f"p_blue_win_{phase.lower()}"]
        if (
            not isinstance(document, dict) or not isinstance(stored_provenance, dict)
            or document.get("schema_version") != "evaluation-input-v1"
            or stored_provenance != {**provenance, "input_snapshot_sha256": _history_hash(document)}
            or document.get("provenance") != provenance
            or row["history_sha256"] != expected_history_hash
            or document.get("history_sha256") != expected_history_hash
            or row["game_id"] != game_id or row["analysis_id"] is not None
            or row["inference_mode"] != "REAL_RETROSPECTIVE_SIMULATION"
            or row["origin"] != "RETROSPECTIVE_IMPORT" or row["inferred_at"] is not None
            or row["data_version"] != bundle.identity.dataset_id
            or row["model_version"] != bundle.identity.bundle_version
            or not row["is_active"]
            or row["history_cutoff_at"] != _time_iso(expected["history_cutoff_at"])
            or document.get("runtime_context") is not None
            or document.get("pre_context_key") != row["context_key"]
            or not math.isclose(row["blue_win_probability"], expected_probability,
                                rel_tol=0, abs_tol=_TOLERANCE)
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        target = document.get("target", {})
        if (
            target.get("game_id") != game_id
            or _time_iso(target.get("history_cutoff_at"))
            != _time_iso(expected["history_cutoff_at"])
            or _digest((game_id, target.get("patch"),
                        ("BLUE", target.get("blue_team_id"), target.get("blue_roster")),
                        ("RED", target.get("red_team_id"), target.get("red_roster"))))
            != provenance["source_target_provenance"]["pre_context_sha256"]
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        model = document.get("model", {})
        if (
            model.get("identity") != _plain(bundle.identity)
            or model.get("contract") != _plain(bundle.contract)
            or model.get("fit_signature") != bundle.fit_signature
            or model.get("family") != bundle.family
            or model.get("feature_schema_version") != bundle.feature_schema_version
            or model.get("preprocessing_version") != bundle.preprocessing_version
            or model.get("library_versions") != dict(bundle.library_versions)
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        prediction = document.get("prediction", {})
        if (
            prediction.get("game_id") != game_id or prediction.get("phase") != phase
            or prediction.get("context_signature") != expected["context_signature"]
            or prediction.get("policy_version") != expected["policy_version"]
            or prediction.get("model_bundle_version") != expected["model_bundle_version"]
            or prediction.get("bundle_signature") != bundle.fit_signature
            or prediction.get("feature_schema_version") != expected["feature_schema_version"]
            or _time_iso(prediction.get("history_cutoff_at"))
            != _time_iso(expected["history_cutoff_at"])
            or prediction.get("p_blue_win") != row["blue_win_probability"]
            or _digest((game_id, document.get("feature_context"))) != expected["context_signature"]
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        if row["idempotency_key"] != _digest((
            "RETROSPECTIVE_IMPORT", document.get("pre_seal"), prediction,
            document.get("lineup") if phase == "POST" else None,
        )):
            fail("E_STORAGE_CONFLICT", game_id)
        columns = PRE_COLUMNS if phase == "PRE" else POST_COLUMNS
        if document.get("feature_columns") != list(columns):
            fail("E_STORAGE_CONFLICT", game_id)
        frame = pd.DataFrame([document.get("feature_values")], columns=columns, dtype=object)
        normalized = _model_frame(frame, phase)
        if prediction.get("pre_feature_signature") != _digest(
            normalized.loc[0, list(PRE_COLUMNS)].to_dict(),
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        if phase == "PRE":
            pre_values = document["feature_values"]
        elif (
            document["feature_values"][:len(PRE_COLUMNS)] != pre_values
            or document.get("pre_prediction") != pre["input_snapshot"].get("prediction")
            or document.get("pre_seal") != pre["input_snapshot"].get("pre_seal")
        ):
            fail("E_STORAGE_CONFLICT", game_id)
        stored_warnings = sorted(
            (warning for warning in warnings if warning["evaluation_id"] == row["evaluation_id"]),
            key=lambda warning: warning["position"],
        )
        actual_warnings = warning_rows(tuple(BusinessWarning(**item)
                                            for item in document.get("warnings", [])))
        if [warning["position"] for warning in stored_warnings] != list(range(len(actual_warnings))):
            fail("E_STORAGE_CONFLICT", game_id)
        for actual, stored in zip(actual_warnings, stored_warnings, strict=True):
            if any(stored[key] != value for key, value in actual.items()):
                fail("E_STORAGE_CONFLICT", game_id)
    return tuple(SavedEvaluation(by_phase[phase]["evaluation_id"], False, True)
                 for phase in ("PRE", "POST")), len(warnings)


def _verified_existing_pairs(
    engine, bundle, snapshot, document, predictions, provenance, history, metadata_path,
):
    expected_history_hash = _history_hash(_plain({
        "schema_version": "evaluation-history-v1", "games": history,
        "champion_reference": sorted(snapshot.champion_reference),
    }))
    positions = {game_id: position for position, game_id in enumerate(predictions)}
    verified = {}
    with engine.connect() as connection:
        with connection.begin():
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            if connection.exec_driver_sql("SHOW transaction_read_only").scalar_one() != "on":
                fail("E_DB_NOT_READ_ONLY")
            existing_ids = connection.execute(select(Evaluation.game_id).where(
                Evaluation.origin == "RETROSPECTIVE_IMPORT",
                Evaluation.model_version == bundle.identity.bundle_version,
            ).distinct()).scalars().all()
            if not set(existing_ids) <= set(predictions):
                fail("E_STORAGE_CONFLICT")
            if existing_ids:
                payload = connection.execute(select(EvaluationHistory.payload).where(
                    EvaluationHistory.history_sha256 == expected_history_hash,
                )).scalar_one_or_none()
                if payload is None or _history_hash(payload) != expected_history_hash:
                    fail("E_STORAGE_CONFLICT")
            for game_id in existing_ids:
                rows = connection.execute(select(Evaluation.__table__).where(
                    Evaluation.game_id == game_id,
                    Evaluation.origin == "RETROSPECTIVE_IMPORT",
                    Evaluation.model_version == bundle.identity.bundle_version,
                )).mappings().all()
                warnings = connection.execute(select(EvaluationWarning.__table__).where(
                    EvaluationWarning.evaluation_id.in_([row["evaluation_id"] for row in rows]),
                )).mappings().all()
                wanted = _provenance(game_id, positions[game_id], predictions[game_id],
                                     provenance[game_id], document, snapshot, metadata_path)
                verified[game_id] = _check_stored_pair(
                    rows, warnings, predictions[game_id], wanted, bundle, expected_history_hash,
                )
    return verified


def _run_pair(position, game_id, expected, prepared, store):
    bundle, snapshot, document, provenance, sources, history, metadata_path = prepared
    try:
        pre, post = _reconstruct_pair(
            sources[game_id], history, snapshot.champion_reference, bundle,
        )
        _check_prediction(post, expected)
        saved = None
        if store is not None:
            saved = store.save_pair(pre, post, bundle, provenance=_provenance(
                game_id, position, expected, provenance[game_id], document,
                snapshot, metadata_path,
            ))
        return {
            "game_id": game_id, "position": position, "saved": saved,
            "warning_count": len(pre.warnings) + len(post.warnings), "reason": None,
        }
    except PreFeatureInputError as error:
        return {"game_id": game_id, "position": position, "reason": error.code}


def _worker_initialize(engine_url, prepared):
    """One child-local engine and validated immutable source copy per spawned process."""
    global _WORKER_STATE
    store = None
    if engine_url is not None:
        from sqlalchemy import create_engine

        engine = create_engine(engine_url)
        Finalize(None, engine.dispose, exitpriority=10)
        store = EvaluationPersistence(engine)
    _WORKER_STATE = (prepared, store)


def _worker_pair(task):
    if _WORKER_STATE is None:
        fail("E_EVALUATION_IMPORT_FAILED")
    try:
        return _run_pair(*task, *_WORKER_STATE)
    except Exception:
        # Never transport driver exceptions containing credentials to the parent.
        return {"position": task[0], "game_id": task[1],
                "reason": "E_EVALUATION_IMPORT_FAILED"}


def _parallel_results(tasks, *, engine_url, prepared, workers):
    """Bound both process count and in-flight work; no shared DB connections/files."""
    tasks = iter(tasks)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
        initializer=_worker_initialize, initargs=(engine_url, prepared),
    ) as executor:
        pending = {}
        for _ in range(workers):
            task = next(tasks, None)
            if task is None:
                break
            pending[executor.submit(_worker_pair, task)] = task
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda value: pending[value][0]):
                task = pending.pop(future)
                try:
                    yield future.result()
                except Exception:
                    yield {"position": task[0], "game_id": task[1],
                           "reason": "E_EVALUATION_IMPORT_FAILED"}
                task = next(tasks, None)
                if task is not None:
                    pending[executor.submit(_worker_pair, task)] = task


def import_retrospective_evaluations(
    engine, *, apply=False, metadata_path=DEFAULT_METADATA, input_path=RETROSPECTIVE_INPUT,
    progress=None, workers=1,
):
    """Reconstruct qualified pairs; dry-run is default and performs no persistence."""
    if type(apply) is not bool:
        fail("E_PRE_INPUT_INCOMPLETE")
    if progress is not None and not callable(progress):
        fail("E_PRE_INPUT_INCOMPLETE")
    if type(workers) is not int or not 1 <= workers <= 8:
        fail("E_PRE_INPUT_INCOMPLETE")
    started = monotonic()
    bundle, snapshot, document, predictions, provenance, sources, history = _prepare_import(
        engine, metadata_path, input_path,
    )
    before = _database_counts(engine, bundle)
    existing = _verified_existing_pairs(
        engine, bundle, snapshot, document, predictions, provenance, history, metadata_path,
    ) if before["evaluations"] else {}
    store = EvaluationPersistence(engine) if apply and workers == 1 else None
    report = {
        "status": "IMPORTED" if apply else "DRY_RUN_READY",
        "source_sha256": CANONICAL_METADATA_SHA256,
        "dataset_id": bundle.identity.dataset_id,
        "split_id": bundle.identity.split_id,
        "bundle_version": bundle.identity.bundle_version,
        "expected_pairs": len(predictions), "validated_pairs": 0,
        "verified_existing_pairs": len(existing),
        "created_evaluations": 0, "existing_evaluations": 0,
        "source_warning_count": 0, "rejected": [], "sample_ids": [],
        "before": before, "workers": workers,
    }
    prepared = (
        bundle, SimpleNamespace(sha256=snapshot.sha256,
                                champion_reference=snapshot.champion_reference),
        document, provenance, sources, history, metadata_path,
    )
    tasks = [(position, game_id, expected)
             for position, (game_id, expected) in enumerate(predictions.items())
             if game_id not in existing]

    def results():
        for position, game_id in enumerate(predictions):
            if game_id in existing:
                saved, warning_count = existing[game_id]
                yield {"position": position, "game_id": game_id, "saved": saved,
                       "warning_count": warning_count, "reason": None}
        if workers == 1:
            for task in tasks:
                yield _run_pair(*task, prepared, store)
        elif tasks:
            yield from _parallel_results(tasks, engine_url=engine.url if apply else None,
                                         prepared=prepared, workers=workers)

    samples = []
    for processed, result in enumerate(results(), start=1):
        game_id = result["game_id"]
        if result["reason"] is None:
            saved = result["saved"]
            report["validated_pairs"] += 1
            report["source_warning_count"] += result["warning_count"]
            if saved is not None:
                report["created_evaluations"] += sum(item.created for item in saved)
                report["existing_evaluations"] += sum(not item.created for item in saved)
                samples.append((result["position"], {
                    "game_id": game_id, "pre_evaluation_id": saved[0].evaluation_id,
                    "post_evaluation_id": saved[1].evaluation_id,
                }))
        else:
            report["rejected"].append({"game_id": game_id, "reason": result["reason"]})
        if progress is not None and (processed == 1 or processed % 25 == 0):
            progress({
                "processed_pairs": processed, "total_pairs": len(predictions),
                "created_evaluations": report["created_evaluations"],
                "rejected_pairs": len(report["rejected"]),
                "elapsed_seconds": round(monotonic() - started, 2),
            })
    report["sample_ids"] = [row for _position, row in sorted(samples)[:3]]
    positions = {game_id: position for position, game_id in enumerate(predictions)}
    report["rejected"].sort(key=lambda row: positions[row["game_id"]])
    report["after"] = _database_counts(engine, bundle)
    report["elapsed_seconds"] = round(monotonic() - started, 2)
    if report["rejected"]:
        report["status"] = "PARTIAL_IMPORT" if apply else "BLOCKED"
    return report
