"""Real-data integration gates; existing feature/split/model algorithms are reused."""

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from match_insight.database.real_snapshot import (
    RealSnapshot,
    fail,
    pre_context_sha256,
)
from match_insight.features.pre import (
    FeatureConfig,
    PreFeatureInputError,
    _require_id,
)
from match_insight.ml.dataset import (
    POST_COLUMNS,
    PRE_COLUMNS,
    RETROSPECTIVE_PROTOCOL,
    STRICT_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetTarget,
    _resolve_cutoff,
    build_paired_dataset,
    protocol_contract,
    temporal_split,
)
from match_insight.ml.models import (
    FAMILIES,
    FEATURE_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    FeatureContract,
    ModelConfig,
    ModelIdentity,
    ModelInputError,
    SelectionPolicy,
    _contract_protocol,
    _digest,
    _plain,
    _versions,
    evaluate_test,
    fit_model_bundle,
    load_bundle,
    save_bundle,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/reference/real_pre_inputs.json"
MODEL_DIRECTORY = ROOT / "artifacts/models"
MODEL_FILENAME = "real_pre_post.joblib"
METADATA_FILENAME = "real_pre_post.json"
INPUT_SCHEMA = "real-pre-input-v1"

ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})"
)
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
BUNDLE_FIELDS = (
    "identity",
    "family",
    "config",
    "selection_policy",
    "validation_scores",
    "contract",
    "split",
    "dataset_signature",
    "fit_signature",
    "estimator_parameters",
    "library_versions",
    "pre_columns",
    "post_columns",
    "feature_schema_version",
    "preprocessing_version",
)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    status: str
    sha256: str | None
    document: dict
    time_archive: object = None


@dataclass(frozen=True, slots=True)
class RunPlan:
    report: dict
    dataset: object = None
    split: object = None


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


RETROSPECTIVE_INPUT = ROOT / "data/reference/retrospective_pre_inputs.json"
RETROSPECTIVE_SCHEMA = "retrospective-pre-input-v1"
RETROSPECTIVE_MODEL_FILENAME = "retrospective_pre_post.joblib"
RETROSPECTIVE_METADATA_FILENAME = "retrospective_pre_post.json"


@dataclass(frozen=True, slots=True)
class ModelArtifactExpectation:
    """Caller-owned artifact pin; independent of the generic loader's default."""

    identity: ModelIdentity
    model_sha256: str
    family: str


CANONICAL_RETROSPECTIVE_EXPECTATION = ModelArtifactExpectation(
    identity=ModelIdentity(
        dataset_id=(
            "retrospective:postgresql:"
            "05f268bf0295dfdc79456e14f1a5ea89a79bdadc0713fb1752aa71afce9032b4"
        ),
        split_id=(
            "retrospective:split:"
            "fe4467dca8420e39346e9888fb1e99fed32b4ea75c2001cfb4c5d78cd21180d4"
        ),
        bundle_version=(
            "retrospective:7l:"
            "c1c82ad44fbe40ada209e6bb3a1d1a741819afdfa5f8152c92bc34b991144613"
        ),
    ),
    model_sha256="f83f8cb9190e126462d8f4fc1417c8b5e9edf5a2213c8a3d2d56811f1a31ae9b",
    family="logistic_regression",
)

TIME_SUMMARY = "oracle_2025_riot_v5_timestamps_summary_11565_games.json"
TIME_EXPORTS = tuple(
    f"oracle_2025_riot_v5_raw_part_{number}_of_12.json"
    for number in range(1, 13)
)
TIME_SOURCE_DIRECTORY = ROOT / "data/raw/leaguepedia/oracle_riot_temporal_v5"
TIME_EVIDENCE_GRADE = "LOCAL_ARCHIVED_V5_CONSISTENT_WITH_SNAPSHOT"


def _run_format(evaluation_protocol):
    protocol_contract(evaluation_protocol)
    if evaluation_protocol == STRICT_PROTOCOL:
        return MODEL_FILENAME, METADATA_FILENAME, "REAL_POSTGRESQL", "real"
    return (
        RETROSPECTIVE_MODEL_FILENAME,
        RETROSPECTIVE_METADATA_FILENAME,
        "REAL_RETROSPECTIVE_SIMULATION",
        "retrospective",
    )


def _time_iso(value):
    if not isinstance(value, str) or ISO_TIMESTAMP.fullmatch(value) is None:
        fail("E_TIME_TIMESTAMP_INVALID")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        fail("E_TIME_TIMESTAMP_INVALID")
    if result.utcoffset() is None:
        fail("E_TIME_TIMESTAMP_INVALID")
    return result.astimezone(UTC)


def _time_ms(value):
    if type(value) is not int or value < 0:
        fail("E_TIME_MILLISECONDS_INVALID")
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)
    except (OverflowError, ValueError):
        fail("E_TIME_MILLISECONDS_INVALID")


def retrospective_cutoff(started_at):
    if not isinstance(started_at, datetime) or started_at.utcoffset() is None:
        fail("E_TIME_TIMESTAMP_INVALID")
    start = started_at.astimezone(UTC)
    try:
        return start.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    except OverflowError:
        fail("E_CUTOFF_INVALID")


def _time_row_id(row):
    if not isinstance(row, dict):
        fail("E_TIME_SOURCE_SCHEMA")
    game_id = row.get("game_id")
    _require_id(game_id, "game_id", 100)
    return game_id


def _time_fields(row):
    if row.get("status") != "SUCCESS":
        fail("E_TIME_SOURCE_NOT_SUCCESS")
    start_ms = row.get("start_timestamp_ms")
    end_ms = row.get("end_timestamp_ms")
    start, end = _time_ms(start_ms), _time_ms(end_ms)
    if (
        _time_iso(row.get("started_at")) != start
        or _time_iso(row.get("ended_at")) != end
    ):
        fail("E_TIME_SOURCE_CONFLICT")
    if start >= end:
        fail("E_TIME_ORDER_INVALID")
    return start_ms, end_ms


def _time_revision(row):
    revision_id = row.get("revision_id")
    timestamp = row.get("revision_timestamp_utc")
    if revision_id is None and timestamp is None:
        return None
    if type(revision_id) is not int or revision_id <= 0 or timestamp is None:
        fail("E_TIME_REVISION_INVALID")
    return revision_id, _time_iso(timestamp).isoformat()


def _time_payload_identity(row, payload, game_id):
    platform = payload.get("platformId")
    number = payload.get("gameId")
    if (
        not _text(platform)
        or type(number) is not int
        or number <= 0
        or f"{platform}_{number}" != game_id
    ):
        fail("E_TIME_IDENTITY_CONFLICT")
    for container in (row, payload):
        for field in (
            "matchId",
            "game_id",
            "RiotPlatformGameId",
            "requested_riot_platform_game_id",
        ):
            if field in container and container[field] != game_id:
                fail("E_TIME_IDENTITY_CONFLICT")
        if "metadata" in container:
            metadata = container["metadata"]
            if not isinstance(metadata, dict):
                fail("E_TIME_IDENTITY_CONFLICT")
            if "matchId" in metadata and metadata["matchId"] != game_id:
                fail("E_TIME_IDENTITY_CONFLICT")


def _iter_time_rows(path):
    """Stream one top-level JSON array without retaining all payloads."""
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda value: fail("E_EVIDENCE_JSON_NONFINITE"),
    )
    try:
        with path.open("r", encoding="utf-8") as stream:
            buffer = ""
            exhausted = False

            def fill():
                nonlocal buffer, exhausted
                chunk = stream.read(65536)
                buffer += chunk
                exhausted = not chunk

            def ready():
                nonlocal buffer
                buffer = buffer.lstrip()
                while not buffer and not exhausted:
                    fill()
                    buffer = buffer.lstrip()

            ready()
            if not buffer.startswith("["):
                fail("E_TIME_SOURCE_SCHEMA")
            buffer = buffer[1:]
            ready()
            if buffer.startswith("]"):
                buffer = buffer[1:]
            else:
                while True:
                    while True:
                        try:
                            row, end = decoder.raw_decode(buffer)
                            break
                        except json.JSONDecodeError:
                            if exhausted:
                                fail("E_EVIDENCE_JSON_INVALID")
                            fill()
                    buffer = buffer[end:]
                    yield row
                    ready()
                    if buffer.startswith("]"):
                        buffer = buffer[1:]
                        break
                    if not buffer.startswith(","):
                        fail("E_EVIDENCE_JSON_INVALID")
                    buffer = buffer[1:]
                    ready()
                    if not buffer or buffer.startswith("]"):
                        fail("E_EVIDENCE_JSON_INVALID")
            ready()
            if buffer:
                fail("E_EVIDENCE_JSON_INVALID")
    except (OSError, UnicodeError):
        fail("E_TIME_SOURCE_UNREADABLE")


def _time_candidate(row, summary, export_source, index, summary_source):
    game_id = _time_row_id(row)
    if summary is None:
        fail("E_TIME_SUMMARY_MISSING")
    start_ms, end_ms = _time_fields(row)
    if (start_ms, end_ms) != summary["timestamps"]:
        fail("E_TIME_SOURCE_CONFLICT")
    payload = row.get("raw_payload")
    if not isinstance(payload, dict):
        fail("E_TIME_PAYLOAD_MISSING")
    _time_payload_identity(row, payload, game_id)
    payload_start = payload.get("gameStartTimestamp")
    payload_end = payload.get("gameEndTimestamp")
    _time_ms(payload_start)
    _time_ms(payload_end)
    if (payload_start, payload_end) != (start_ms, end_ms):
        fail("E_TIME_SOURCE_CONFLICT")

    fetched = _time_iso(row.get("fetched_at_utc"))
    if fetched < _time_ms(end_ms):
        fail("E_TIME_RETRIEVAL_CONFLICT")
    revision = _time_revision(row)
    if (
        revision is not None
        and summary["revision"] is not None
        and revision != summary["revision"]
    ):
        fail("E_TIME_REVISION_CONFLICT")

    proof = {
        "grade": TIME_EVIDENCE_GRADE,
        "start_timestamp_ms": start_ms,
        "end_timestamp_ms": end_ms,
        "started_at": _time_ms(start_ms).isoformat(),
        "ended_at": _time_ms(end_ms).isoformat(),
        "fetched_at_utc": fetched.isoformat(),
        "revision_status": "MISSING" if revision is None else "PRESENT",
        "raw_payload_sha256": _digest(payload),
        "raw_payload_hash_encoding": "models._digest canonical JSON",
        "export": {**export_source, "index": index},
        "summary": {**summary_source, "index": summary["index"]},
    }
    if revision is not None:
        proof["revision_id"], proof["revision_timestamp_utc"] = revision
    return proof


def _time_exclusions(errors):
    result = []
    for game_id, reasons in sorted(errors.items()):
        ordered = tuple(sorted(reasons))
        reason = (
            "E_TIME_DUPLICATE_GAME_ID"
            if "E_TIME_DUPLICATE_GAME_ID" in reasons
            else ordered[0]
        )
        result.append({"game_id": game_id, "reason": reason, "diagnostics": ordered})
    return tuple(result)


def load_time_archive(source_directory=None):
    """Read fixed archive sources before opening DB; keep compact proofs only."""
    directory = Path(source_directory or TIME_SOURCE_DIRECTORY)
    paths = tuple(directory / name for name in (TIME_SUMMARY, *TIME_EXPORTS))
    try:
        if any(not path.is_file() for path in paths):
            fail("E_TIME_SOURCE_MISSING")
        if any(path.resolve().parent != directory.resolve() for path in paths):
            fail("E_TIME_SOURCE_PATH_INVALID")
        sources = tuple(
            {"path": path.name, "sha256": _file_sha256(path)} for path in paths
        )
        summaries = {}
        summary_seen = set()
        raw_seen = set()
        errors = defaultdict(set)
        candidates = {}

        for index, row in enumerate(_iter_time_rows(paths[0])):
            game_id = _time_row_id(row)
            if game_id in summary_seen:
                errors[game_id].add("E_TIME_DUPLICATE_GAME_ID")
                continue
            summary_seen.add(game_id)
            try:
                summaries[game_id] = {
                    "index": index,
                    "timestamps": _time_fields(row),
                    "revision": _time_revision(row),
                }
            except PreFeatureInputError as error:
                errors[game_id].add(error.code)
        if _file_sha256(paths[0]) != sources[0]["sha256"]:
            fail("E_TIME_SOURCE_CHANGED")

        for path, source in zip(paths[1:], sources[1:], strict=True):
            for index, row in enumerate(_iter_time_rows(path)):
                game_id = _time_row_id(row)
                if game_id in raw_seen:
                    errors[game_id].add("E_TIME_DUPLICATE_GAME_ID")
                    continue
                raw_seen.add(game_id)
                try:
                    candidates[game_id] = _time_candidate(
                        row, summaries.get(game_id), source, index, sources[0]
                    )
                except PreFeatureInputError as error:
                    errors[game_id].add(error.code)
            if _file_sha256(path) != source["sha256"]:
                fail("E_TIME_SOURCE_CHANGED")
        for game_id in summary_seen - raw_seen:
            errors[game_id].add("E_TIME_RAW_MISSING")
        for game_id in errors:
            candidates.pop(game_id, None)
        return candidates, _time_exclusions(errors), sources
    except OSError:
        fail("E_TIME_SOURCE_UNREADABLE")


def resolve_time_proofs(snapshot, archive):
    candidates, archive_exclusions, sources = archive
    rejected = {item["game_id"]: item for item in archive_exclusions}
    proofs = {}
    exclusions = []
    seen = set()
    for item in sorted(snapshot.games, key=lambda value: value.game.game_id):
        game_id = item.game.game_id
        if game_id in seen:
            fail("E_TIME_SNAPSHOT_DUPLICATE", game_id)
        seen.add(game_id)
        if game_id in rejected:
            exclusions.append(deepcopy(rejected[game_id]))
            continue
        proof = candidates.get(game_id)
        if proof is None:
            exclusions.append({"game_id": game_id, "reason": "E_TIME_ARCHIVE_MISSING"})
            continue
        start, end = item.started_at, item.game.ended_at
        if start is None or end is None:
            exclusions.append({"game_id": game_id, "reason": "E_TIME_SNAPSHOT_MISSING"})
            continue
        if (
            not isinstance(start, datetime)
            or not isinstance(end, datetime)
            or start.utcoffset() is None
            or end.utcoffset() is None
        ):
            exclusions.append({"game_id": game_id, "reason": "E_TIME_SNAPSHOT_INVALID"})
            continue
        if (
            start.astimezone(UTC) != _time_ms(proof["start_timestamp_ms"])
            or end.astimezone(UTC) != _time_ms(proof["end_timestamp_ms"])
        ):
            exclusions.append({"game_id": game_id, "reason": "E_TIME_SNAPSHOT_CONFLICT"})
            continue
        proofs[game_id] = deepcopy(proof)
    return proofs, tuple(exclusions), sources


def _retrospective_document(snapshot, archive, approval_ref):
    if not _text(approval_ref):
        fail("E_PROTOCOL_APPROVAL_REFERENCE_MISSING")
    proofs, exclusions, sources = resolve_time_proofs(snapshot, archive)
    records = []
    for item in snapshot.games:
        game_id = item.game.game_id
        if game_id not in proofs:
            continue
        records.append(
            {
                "game_id": game_id,
                "history_cutoff_at": retrospective_cutoff(item.started_at).isoformat(),
                "evidence_ref": f"protocol:{RETROSPECTIVE_PROTOCOL}:{game_id}",
                "policy_version": RETROSPECTIVE_PROTOCOL,
                "verification": "PROTOCOL_ASSUMED",
                "pre_context_sha256": pre_context_sha256(item),
                "pre_context_evidence_ref": f"snapshot:{snapshot.sha256}:{game_id}",
                "pre_context_verification": "RECONSTRUCTED_POSTGAME",
            }
        )
    return {
        "schema_version": RETROSPECTIVE_SCHEMA,
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "snapshot_sha256": snapshot.sha256,
        "approval_ref": approval_ref,
        "ended_at_provenance": {
            "evidence_ref": "local-v5-summary-and-twelve-exports",
            "verification": TIME_EVIDENCE_GRADE,
        },
        "source_files": sources,
        "time_proofs": proofs,
        "records": sorted(records, key=lambda row: row["game_id"]),
        "exclusions": exclusions,
    }


def _retrospective_header(document):
    required = {
        "schema_version",
        "evaluation_protocol",
        "snapshot_sha256",
        "approval_ref",
        "ended_at_provenance",
        "source_files",
        "time_proofs",
        "records",
        "exclusions",
    }
    if (
        not isinstance(document, dict)
        or set(document) != required
        or document["schema_version"] != RETROSPECTIVE_SCHEMA
        or document["evaluation_protocol"] != RETROSPECTIVE_PROTOCOL
        or not _text(document["approval_ref"])
        or not isinstance(document["snapshot_sha256"], str)
        or HASH_PATTERN.fullmatch(document["snapshot_sha256"]) is None
        or not isinstance(document["records"], list)
        or not document["records"]
        or not isinstance(document["time_proofs"], dict)
        or not isinstance(document["source_files"], (list, tuple))
        or not isinstance(document["exclusions"], (list, tuple))
        or document["ended_at_provenance"]
        != {
            "evidence_ref": "local-v5-summary-and-twelve-exports",
            "verification": TIME_EVIDENCE_GRADE,
        }
    ):
        fail("E_RETROSPECTIVE_INPUT_SCHEMA")
    seen = set()
    record_fields = {
        "game_id",
        "history_cutoff_at",
        "evidence_ref",
        "policy_version",
        "verification",
        "pre_context_sha256",
        "pre_context_evidence_ref",
        "pre_context_verification",
    }
    for raw in document["records"]:
        if not isinstance(raw, dict) or set(raw) != record_fields:
            fail("E_RETROSPECTIVE_RECORD_SCHEMA")
        game_id = raw["game_id"]
        _require_id(game_id, "game_id", 100)
        if game_id in seen:
            fail("E_CUTOFF_CONFLICT", game_id)
        seen.add(game_id)
        _resolve_cutoff(
            game_id,
            (_cutoff_record(raw),),
            frozenset({RETROSPECTIVE_PROTOCOL}),
            evaluation_protocol=RETROSPECTIVE_PROTOCOL,
        )
        if (
            raw["pre_context_verification"] != "RECONSTRUCTED_POSTGAME"
            or not _text(raw["pre_context_evidence_ref"])
            or not isinstance(raw["pre_context_sha256"], str)
            or HASH_PATTERN.fullmatch(raw["pre_context_sha256"]) is None
        ):
            fail("E_PRE_CONTEXT_UNVERIFIED", game_id)
    if set(document["time_proofs"]) != seen:
        fail("E_TIME_PROOF_CONFLICT")


def _retrospective_archive_check(document, archive):
    candidates, _excluded, sources = archive
    if _digest(document["source_files"]) != _digest(sources):
        fail("E_TIME_SOURCE_CHANGED")
    for game_id, proof in document["time_proofs"].items():
        if game_id not in candidates or _digest(proof) != _digest(candidates[game_id]):
            fail("E_TIME_PROOF_CONFLICT", game_id)


def make_retrospective_evidence(snapshot, archive, *, approval_ref):
    document = _retrospective_document(snapshot, archive, approval_ref)
    if not document["records"]:
        fail("E_NO_ELIGIBLE_TARGETS")
    _retrospective_header(document)
    data = json.dumps(_plain(document), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return EvidenceArtifact(
        "LOADED", hashlib.sha256(data).hexdigest(), document, deepcopy(archive)
    )


def write_retrospective_inputs(
    snapshot,
    archive,
    path=RETROSPECTIVE_INPUT,
    *,
    approval_ref,
):
    path = Path(path)
    if path.exists():
        fail("E_INPUT_ARTIFACT_EXISTS")
    if not path.parent.is_dir():
        fail("E_INPUT_DIRECTORY_MISSING")
    evidence = make_retrospective_evidence(snapshot, archive, approval_ref=approval_ref)
    data = (
        json.dumps(_plain(evidence.document), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(data)
    except FileExistsError:
        fail("E_INPUT_ARTIFACT_EXISTS")
    except OSError:
        if created:
            path.unlink(missing_ok=True)
        fail("E_INPUT_WRITE_FAILED")
    return {
        "status": "INPUT_CREATED",
        "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "snapshot_sha256": snapshot.sha256,
        "snapshot_read_at": snapshot.captured_at,
        "target_records": len(evidence.document["records"]),
        "exclusions": evidence.document["exclusions"],
    }


def _validate_retrospective_snapshot(snapshot, evidence, approved_policy_versions):
    document = evidence.document
    _retrospective_header(document)
    if RETROSPECTIVE_PROTOCOL not in approved_policy_versions:
        fail("E_CUTOFF_POLICY_UNAPPROVED")
    if document["snapshot_sha256"] != snapshot.sha256:
        fail("E_EVIDENCE_SNAPSHOT_MISMATCH")
    if evidence.time_archive is None:
        fail("E_TIME_ARCHIVE_MISSING")
    _retrospective_archive_check(document, evidence.time_archive)
    expected = _retrospective_document(
        snapshot, evidence.time_archive, document["approval_ref"]
    )
    for field in ("source_files", "time_proofs", "exclusions"):
        if _digest(document[field]) != _digest(expected[field]):
            fail("E_TIME_PROOF_CONFLICT")
    actual_records = {row["game_id"]: row for row in document["records"]}
    expected_records = {row["game_id"]: row for row in expected["records"]}
    if set(actual_records) != set(expected_records):
        fail("E_RETROSPECTIVE_COHORT_MISMATCH")
    for game_id, wanted in expected_records.items():
        actual = actual_records[game_id]
        if _time_iso(actual["history_cutoff_at"]) != _time_iso(wanted["history_cutoff_at"]):
            fail("E_CUTOFF_FORMULA_MISMATCH", game_id)
        if actual["pre_context_sha256"] != wanted["pre_context_sha256"]:
            fail("E_PRE_CONTEXT_MISMATCH", game_id)
        for field in set(wanted) - {"history_cutoff_at", "pre_context_sha256"}:
            if actual[field] != wanted[field]:
                fail("E_RETROSPECTIVE_RECORD_CONFLICT", game_id)
    return set(expected["time_proofs"]), expected["exclusions"]


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("E_EVIDENCE_DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _json_document(data):
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: fail("E_EVIDENCE_JSON_NONFINITE"),
        )
    except (UnicodeError, ValueError):
        fail("E_EVIDENCE_JSON_INVALID")


def load_evidence(
    path=DEFAULT_INPUT,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
    source_directory=None,
):
    """Validate input files/schema/protocol before opening PostgreSQL."""
    protocol_contract(evaluation_protocol)
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return EvidenceArtifact("MISSING", None, {})
    except OSError:
        fail("E_EVIDENCE_FILE_UNREADABLE")
    document = _json_document(data)
    archive = None
    if evaluation_protocol == STRICT_PROTOCOL:
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != INPUT_SCHEMA
            or not isinstance(document.get("records"), list)
            or document.get("evaluation_protocol", STRICT_PROTOCOL) != STRICT_PROTOCOL
        ):
            fail("E_EVIDENCE_SCHEMA")
    else:
        _retrospective_header(document)
        archive = load_time_archive(source_directory)
        _retrospective_archive_check(document, archive)
    return EvidenceArtifact("LOADED", hashlib.sha256(data).hexdigest(), document, archive)


def _cutoff_record(raw):
    value = raw.get("history_cutoff_at")
    if value is not None:
        if not isinstance(value, str) or ISO_TIMESTAMP.fullmatch(value) is None:
            fail("E_CUTOFF_INVALID", raw["game_id"])
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            fail("E_CUTOFF_INVALID", raw["game_id"])
    try:
        verification = CutoffVerification(raw.get("verification"))
    except (TypeError, ValueError):
        fail("E_CUTOFF_STATUS_INVALID", raw["game_id"])

    return CutoffRecord(
        game_id=raw["game_id"],
        history_cutoff_at=value,
        evidence_ref=raw.get("evidence_ref"),
        policy_version=raw.get("policy_version"),
        verification=verification,
    )


def _verified(evidence):
    return (
        isinstance(evidence, dict)
        and evidence.get("verification") == "VERIFIED_EXTERNALLY"
        and _text(evidence.get("evidence_ref"))
    )


def _require_pre_evidence(item, raw, evaluation_protocol=STRICT_PROTOCOL):
    fields = (
        "pre_context_sha256",
        "pre_context_evidence_ref",
        "pre_context_verification",
    )
    if any(raw.get(field) is None for field in fields):
        fail("E_PRE_CONTEXT_MISSING", item.game.game_id)
    required = (
        "RECONSTRUCTED_POSTGAME"
        if evaluation_protocol == RETROSPECTIVE_PROTOCOL
        else "VERIFIED_EXTERNALLY"
    )
    if raw["pre_context_verification"] != required:
        fail("E_PRE_CONTEXT_UNVERIFIED", item.game.game_id)
    if not _text(raw["pre_context_evidence_ref"]):
        fail("E_PRE_CONTEXT_EVIDENCE_INVALID", item.game.game_id)
    if raw["pre_context_sha256"] != pre_context_sha256(item):
        fail("E_PRE_CONTEXT_MISMATCH", item.game.game_id)


def _dataset_target(item):
    game = item.game
    return DatasetTarget(
        game_id=game.game_id,
        blue_team_id=game.blue_team_id,
        red_team_id=game.red_team_id,
        blue_roster=game.blue_roster,
        red_roster=game.red_roster,
        patch=item.final_lineup.patch,
        final_lineup=item.final_lineup,
        winner_team_id=game.winner_team_id,
        ended_at=game.ended_at,
    )


def prepare_run(
    snapshot,
    evidence,
    approved_policy_versions,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
):
    """Prepare only; does not fit, save or independently certify source evidence."""
    if not isinstance(snapshot, RealSnapshot):
        fail("E_REAL_SNAPSHOT_INVALID")
    if (
        not isinstance(approved_policy_versions, frozenset)
        or any(not _text(value) for value in approved_policy_versions)
    ):
        fail("E_CUTOFF_POLICY_SET_INVALID")

    protocol_contract(evaluation_protocol)
    _model_name, _metadata_name, data_kind, prefix = _run_format(evaluation_protocol)

    time_ids = None
    time_exclusions = ()
    if evaluation_protocol == RETROSPECTIVE_PROTOCOL:
        time_ids, time_exclusions = _validate_retrospective_snapshot(
            snapshot, evidence, approved_policy_versions
        )
    elif evidence.document.get("schema_version", INPUT_SCHEMA) != INPUT_SCHEMA:
        fail("E_EVIDENCE_SCHEMA")
    document = deepcopy(evidence.document)
    time_reasons = {row["game_id"]: row["reason"] for row in time_exclusions}
    history = snapshot.history
    if time_ids is not None:
        history = tuple(record for record in history if record.game.game_id in time_ids)

    games = {item.game.game_id: item for item in snapshot.games}
    grouped = defaultdict(list)
    for raw in document.get("records", []):
        if not isinstance(raw, dict):
            fail("E_EVIDENCE_RECORD_INVALID")
        _require_id(raw.get("game_id"), "game_id", 100)
        grouped[raw["game_id"]].append(raw)

    blockers = []
    for unknown_id in sorted(set(grouped) - set(games)):
        blockers.append(
            {"reason": "E_CUTOFF_UNKNOWN_GAME_ID", "game_id": unknown_id}
        )
    if document.get("snapshot_sha256") != snapshot.sha256:
        blockers.append({"reason": "E_EVIDENCE_SNAPSHOT_MISMATCH"})
    if (
        evaluation_protocol == STRICT_PROTOCOL
        and not _verified(document.get("ended_at_provenance"))
    ):
        blockers.append({"reason": "E_END_PROVENANCE_UNVERIFIED"})

    cutoff_status = Counter()
    exclusions = []
    targets = []
    cutoffs = []
    for game_id, item in sorted(games.items()):
        candidates = grouped.get(game_id, [])
        try:
            if game_id in time_reasons:
                fail(time_reasons[game_id], game_id)
            if len(candidates) > 1:
                fail("E_CUTOFF_CONFLICT", game_id)
            parsed = tuple(_cutoff_record(raw) for raw in candidates)
            cutoff, instant = _resolve_cutoff(
                game_id,
                parsed,
                approved_policy_versions,
                evaluation_protocol=evaluation_protocol,
            )
        except PreFeatureInputError as error:
            cutoff_status[error.code] += 1
            exclusions.append(
                {"stage": "INPUT", "game_id": game_id, "reason": error.code}
            )
            continue

        cutoff_status["VALID"] += 1
        try:
            _require_pre_evidence(item, candidates[0], evaluation_protocol)
            if item.started_at is not None and instant >= item.started_at:
                fail("E_CUTOFF_NOT_BEFORE_START", game_id)
        except PreFeatureInputError as error:
            exclusions.append(
                {"stage": "INPUT", "game_id": game_id, "reason": error.code}
            )
            continue

        # Strict attestation and retrospective assumptions are separate contracts.
        targets.append(_dataset_target(item))
        cutoffs.append(cutoff)

    report = {
        "status": "NO_ELIGIBLE_TARGETS",
        "data_kind": data_kind,
        "evaluation_protocol": evaluation_protocol,
        "source_ref": snapshot.source_ref,
        "snapshot_read_at": snapshot.captured_at,
        "snapshot_sha256": snapshot.sha256,
        "source_verification": "EXTERNAL_ASSERTIONS_NOT_AUTHENTICATED_BY_CODE",
        "evidence_status": evidence.status,
        "evidence_sha256": evidence.sha256,
        "evidence_document_sha256": _digest(document),
        "protocol_approval_ref": document.get("approval_ref"),
        "source_files": document.get("source_files", ()),
        "time_proofs": document.get("time_proofs", {}),
        "approved_policy_versions": tuple(sorted(approved_policy_versions)),
        "ended_at_provenance": document.get("ended_at_provenance"),
        "db_source_games": len(snapshot.games),
        "historical_games_with_ended_at": sum(
            item.game.ended_at is not None for item in snapshot.games
        ),
        "missing_end_game_ids": tuple(
            item.game.game_id
            for item in snapshot.games
            if item.game.ended_at is None
        ),
        "cutoff_status_counts": dict(sorted(cutoff_status.items())),
        "targets_with_valid_cutoff": cutoff_status["VALID"],
        "targets_with_verified_pre_context": (
            len(targets) if evaluation_protocol == STRICT_PROTOCOL else 0
        ),
        "targets_with_reconstructed_context": (
            len(targets) if evaluation_protocol == RETROSPECTIVE_PROTOCOL else 0
        ),
        "historical_games_with_accepted_time_proof": (
            len(time_ids) if time_ids is not None else None
        ),
        "paired_samples": 0,
        "dataset_status": "NOT_RUN",
        "split_status": "NOT_RUN",
        "partition_counts": {"train": 0, "validation": 0, "test": 0},
        "partition_label_counts": {
            name: {
                "total": 0,
                "blue_win_1": 0,
                "red_win_0": 0,
                "blue_win_rate": None,
            }
            for name in ("train", "validation", "test")
        },
        "blockers": blockers,
        "exclusions": exclusions,
    }
    if not targets:
        return RunPlan(report)
    if blockers:
        report["status"] = "BLOCKED"
        return RunPlan(report)

    dataset = build_paired_dataset(
        targets=tuple(targets),
        history=history,
        cutoffs=tuple(cutoffs),
        champion_reference=snapshot.champion_reference,
        approved_policy_versions=approved_policy_versions,
        config=FeatureConfig(),
        evaluation_protocol=evaluation_protocol,
    )
    report["dataset_status"] = dataset.status
    report["paired_samples"] = len(dataset.y)
    report["exclusions"].extend(
        {
            "stage": "DATASET",
            "game_id": item.game_id,
            "reason": item.reason,
            "source_game_id": item.source_game_id,
        }
        for item in dataset.excluded
    )
    if dataset.status != "OK":
        report["reason"] = dataset.reason
        return RunPlan(report, dataset)

    split = temporal_split(dataset, evaluation_protocol=evaluation_protocol)
    report["split_status"] = split.status
    report["split"] = split
    report["partition_counts"] = {
        name: len(getattr(split, name))
        for name in ("train", "validation", "test")
    }
    partition_label_counts = {}
    for name in ("train", "validation", "test"):
        members = list(getattr(split, name))
        if members:
            labels = dataset.y.loc[members].to_numpy()
            blue_wins = int((labels == 1).sum())
            red_wins = int((labels == 0).sum())
            if blue_wins + red_wins != len(members):
                fail("E_LABEL_DOMAIN_INVALID")
            partition_label_counts[name] = {
                "total": len(members),
                "blue_win_1": blue_wins,
                "red_win_0": red_wins,
                "blue_win_rate": blue_wins / len(members),
            }
        else:
            partition_label_counts[name] = {
                "total": 0,
                "blue_win_1": 0,
                "red_win_0": 0,
                "blue_win_rate": None,
            }
    report["partition_label_counts"] = partition_label_counts
    report["exclusions"].extend(
        {
            "stage": "SPLIT",
            "game_id": item.game_id,
            "reason": item.reason,
            "partition": item.partition,
        }
        for item in split.excluded
    )
    report["dataset_id"] = f"{prefix}:postgresql:" + _digest(
        (
            snapshot.sha256,
            evidence.sha256,
            _digest(document),
            evaluation_protocol,
            tuple(sorted(approved_policy_versions)),
            FeatureConfig(),
            FEATURE_SCHEMA_VERSION,
        )
    )
    report["split_id"] = f"{prefix}:split:" + _digest(split)
    report["target_provenance"] = tuple(
        {
            "game_id": game_id,
            "history_cutoff_at": dataset.metadata.at[game_id, "history_cutoff_at"],
            "evidence_ref": dataset.metadata.at[game_id, "evidence_ref"],
            "policy_version": dataset.metadata.at[game_id, "policy_version"],
            "pre_context_sha256": grouped[game_id][0]["pre_context_sha256"],
            "pre_context_evidence_ref": grouped[game_id][0]["pre_context_evidence_ref"],
        }
        for game_id in dataset.y.index
    )
    if split.status != "OK":
        report.update(status="INSUFFICIENT_SPLIT", reason=split.reason)
    elif set(dataset.y.loc[list(split.train)].tolist()) != {0, 1}:
        report.update(status="BLOCKED", reason="E_TRAIN_SINGLE_CLASS")
    else:
        report["status"] = "READY"
    return RunPlan(report, dataset, split)


def _output_paths(directory, evaluation_protocol=STRICT_PROTOCOL):
    directory = Path(directory)
    if not directory.is_dir():
        fail("E_MODEL_OUTPUT_DIRECTORY_MISSING")
    model_name, metadata_name, _kind, _prefix = _run_format(evaluation_protocol)
    paths = (directory / model_name, directory / metadata_name)
    if any(path.exists() for path in paths):
        fail("E_MODEL_ARTIFACT_EXISTS")
    return paths


def _bundle_metadata(bundle):
    return {field: getattr(bundle, field) for field in BUNDLE_FIELDS}


def _publish(bundle, report, directory):
    evaluation_protocol = report["evaluation_protocol"]
    model_name, metadata_name, data_kind, _prefix = _run_format(evaluation_protocol)
    destinations = _output_paths(directory, evaluation_protocol)
    published = []
    with TemporaryDirectory(prefix=".real-training-", dir=directory) as temporary:
        temporary = Path(temporary)
        model_path = temporary / model_name
        metadata_path = temporary / metadata_name
        save_bundle(bundle, model_path)
        metadata = {
            "artifact_schema": "real-model-artifact-v1",
            "data_kind": data_kind,
            "evaluation_protocol": evaluation_protocol,
            "model_sha256": _file_sha256(model_path),
            "bundle": _bundle_metadata(bundle),
            "run": report,
        }
        metadata_path.write_text(
            json.dumps(_plain(metadata), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        try:
            # Same-filesystem hard links publish without overwriting existing files.
            for source, destination in zip(
                (model_path, metadata_path), destinations, strict=True
            ):
                os.link(source, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                destination.unlink()
            raise
    return destinations


def execute_run(
    snapshot,
    evidence,
    approved_policy_versions,
    *,
    train=False,
    expected_dataset_id=None,
    expected_split_id=None,
    output_directory=MODEL_DIRECTORY,
    evaluation_protocol=STRICT_PROTOCOL,
):
    plan = prepare_run(
        snapshot,
        evidence,
        approved_policy_versions,
        evaluation_protocol=evaluation_protocol,
    )
    report = deepcopy(plan.report)
    report["mode"] = "TRAIN" if train else "DRY_RUN"
    if report["status"] != "READY":
        return report
    if not train:
        report["status"] = "DRY_RUN_READY"
        return report

    if (
        expected_dataset_id != report["dataset_id"]
        or expected_split_id != report["split_id"]
    ):
        report.update(status="BLOCKED", reason="E_DRY_RUN_ID_MISMATCH")
        return report

    _output_paths(output_directory, evaluation_protocol)
    _model_name, _metadata_name, _kind, prefix = _run_format(evaluation_protocol)
    config = ModelConfig()
    policy = SelectionPolicy()
    identity = ModelIdentity(
        dataset_id=report["dataset_id"],
        split_id=report["split_id"],
        bundle_version=f"{prefix}:7l:" + _digest(
            (report["dataset_id"], report["split_id"], config, policy)
        ),
    )
    fit_options = {"config": config, "policy": policy}
    if evaluation_protocol == RETROSPECTIVE_PROTOCOL:
        fit_options["evaluation_protocol"] = evaluation_protocol
    bundle = fit_model_bundle(plan.dataset, plan.split, identity, **fit_options)
    test_result = evaluate_test(bundle, plan.dataset)
    report.update(
        status="TRAINED",
        selected_family=bundle.family,
        model_bundle_version=bundle.identity.bundle_version,
        validation_scores=bundle.validation_scores,
        test_metrics={"pre": test_result.pre, "post": test_result.post},
        test_predictions=test_result.predictions,
    )
    paths = _publish(bundle, report, output_directory)
    report["artifacts"] = tuple(str(path) for path in paths)
    return report


def _validate_model_metadata(metadata, evaluation_protocol):
    """Check the sidecar before any model bytes are deserialized."""
    _model_name, _metadata_name, data_kind, prefix = _run_format(evaluation_protocol)
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifact_schema") != "real-model-artifact-v1"
        or metadata.get("data_kind") != data_kind
        or metadata.get("evaluation_protocol") != evaluation_protocol
        or not isinstance(metadata.get("model_sha256"), str)
        or HASH_PATTERN.fullmatch(metadata["model_sha256"]) is None
    ):
        fail("E_REAL_MODEL_METADATA_INVALID")

    saved = metadata.get("bundle")
    run = metadata.get("run")
    if (
        not isinstance(saved, dict)
        or set(saved) != set(BUNDLE_FIELDS)
        or not isinstance(run, dict)
        or saved.get("family") not in FAMILIES
        or saved.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or saved.get("preprocessing_version") != PREPROCESSING_VERSION
        or saved.get("pre_columns") != list(PRE_COLUMNS)
        or saved.get("post_columns") != list(POST_COLUMNS)
        or saved.get("library_versions") != _plain(_versions())
        or not isinstance(saved.get("split"), dict)
        or saved["split"].get("evaluation_protocol", STRICT_PROTOCOL) != evaluation_protocol
    ):
        fail("E_REAL_MODEL_METADATA_INVALID")
    identity = saved.get("identity")
    prefixes = {
        "dataset_id": f"{prefix}:postgresql:",
        "split_id": f"{prefix}:split:",
        "bundle_version": f"{prefix}:7l:",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != set(prefixes)
        or any(
            not _text(identity.get(field))
            or not identity[field].startswith(start)
            or len(identity[field]) == len(start)
            for field, start in prefixes.items()
        )
        or run.get("status") != "TRAINED"
        or run.get("data_kind") != data_kind
        or run.get("evaluation_protocol") != evaluation_protocol
        or run.get("dataset_id") != identity["dataset_id"]
        or run.get("split_id") != identity["split_id"]
        or run.get("model_bundle_version") != identity["bundle_version"]
        or run.get("selected_family") != saved["family"]
    ):
        fail("E_REAL_MODEL_METADATA_INVALID")

    try:
        raw_contract = saved["contract"]
        raw_config = raw_contract["pre_config"]
        if set(raw_config) != {"recent_form_games", "side_win_rate_games", "head_to_head_games"}:
            fail("E_REAL_MODEL_METADATA_INVALID")
        config = FeatureConfig(**raw_config)
        if any(type(value) is not int or value <= 0 for value in raw_config.values()):
            fail("E_REAL_MODEL_METADATA_INVALID")
        raw_policies = raw_contract["cutoff_policy_versions"]
        if not isinstance(raw_policies, list):
            fail("E_REAL_MODEL_METADATA_INVALID")
        contract = FeatureContract(
            **{
                **raw_contract,
                "pre_config": config,
                "cutoff_policy_versions": tuple(raw_policies),
            }
        )
        expected_pre_version = (
            f"pre-v1-r{config.recent_form_games}-s{config.side_win_rate_games}"
            f"-h{config.head_to_head_games}-roster-latest1"
        )
        if (
            _contract_protocol(contract) != evaluation_protocol
            or contract.pre_config_version != expected_pre_version
            or contract.post_config_version != "post-v1-player-champion-all-pre-history"
        ):
            fail("E_REAL_MODEL_METADATA_INVALID")
    except (KeyError, TypeError, ValueError, PreFeatureInputError):
        fail("E_REAL_MODEL_METADATA_INVALID")


def _load_run_bundle(path, evaluation_protocol, *, expected=None):
    if expected is not None and (
        not isinstance(expected, ModelArtifactExpectation)
        or not isinstance(expected.identity, ModelIdentity)
        or any(
            not _text(value)
            for value in (
                expected.identity.dataset_id,
                expected.identity.split_id,
                expected.identity.bundle_version,
            )
        )
        or not isinstance(expected.model_sha256, str)
        or HASH_PATTERN.fullmatch(expected.model_sha256) is None
        or expected.family not in FAMILIES
    ):
        fail("E_REAL_MODEL_IDENTITY_MISMATCH")
    try:
        path = Path(path)
        metadata_bytes = path.with_suffix(".json").read_bytes()
    except (OSError, TypeError, ValueError):
        fail("E_REAL_MODEL_METADATA_MISSING")
    try:
        metadata = _json_document(metadata_bytes)
    except PreFeatureInputError:
        fail("E_REAL_MODEL_METADATA_INVALID")
    _validate_model_metadata(metadata, evaluation_protocol)
    if expected is not None:
        if (
            metadata["bundle"]["identity"] != _plain(expected.identity)
            or metadata["bundle"]["family"] != expected.family
        ):
            fail("E_REAL_MODEL_IDENTITY_MISMATCH")
        if metadata["model_sha256"] != expected.model_sha256:
            fail("E_REAL_MODEL_HASH_MISMATCH")
    try:
        actual_sha256 = _file_sha256(path)
    except OSError:
        raise ModelInputError("E_MODEL_BUNDLE_INVALID", "Cannot read model artifact") from None
    if actual_sha256 != metadata["model_sha256"]:
        fail("E_REAL_MODEL_HASH_MISMATCH")
    bundle = load_bundle(path, expected_sha256=metadata["model_sha256"])
    expected_dataset_version, _verification = protocol_contract(evaluation_protocol)
    if (
        bundle.contract.dataset_version != expected_dataset_version
        or _plain(_bundle_metadata(bundle)) != metadata.get("bundle")
    ):
        fail("E_REAL_MODEL_IDENTITY_MISMATCH")
    return bundle


def load_real_bundle(path=MODEL_DIRECTORY / MODEL_FILENAME, *, expected=None):
    """Strict loader: never accepts a retrospective model."""
    return _load_run_bundle(path, STRICT_PROTOCOL, expected=expected)


def load_retrospective_bundle(
    path=MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME, *, expected=None,
):
    """Explicit loader for locally trusted retrospective model files."""
    return _load_run_bundle(path, RETROSPECTIVE_PROTOCOL, expected=expected)
