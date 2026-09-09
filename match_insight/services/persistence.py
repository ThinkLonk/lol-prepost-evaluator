"""Persist validated business snapshots without rebuilding features or predicting."""

import hashlib
import json
from dataclasses import fields
from uuid import UUID

from match_insight.database.evaluations import EvaluationRepository, SavedEvaluation, StorageError
from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL, STRICT_PROTOCOL
from match_insight.ml.models import INTERACTIVE_INFERENCE_POLICY, _digest, _plain
from match_insight.services.evaluation import (
    _prediction_inputs,
    _require_snapshot,
    compare_interactive_evaluations,
    compare_retrospective_evaluations,
)

__all__ = ["EvaluationPersistence", "SavedEvaluation", "StorageError", "default_store"]


def _json_value(value):
    # Reject nonfinite output rather than allowing PostgreSQL/nonstandard JSON coercion.
    return json.loads(json.dumps(_plain(value), allow_nan=False, ensure_ascii=False))


def _history_document(pre):
    return _json_value({
        "schema_version": "evaluation-history-v1",
        "games": pre.request.history,
        "champion_reference": sorted(pre.request.champion_reference),
    })


def _history_hash(document):
    def jsonb_numbers(value):
        # PostgreSQL JSONB numeric does not retain the sign of IEEE negative zero.
        # Hash equal numeric values equally without changing any stored probability.
        if isinstance(value, float) and value == 0:
            return 0.0
        if isinstance(value, dict):
            return {key: jsonb_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [jsonb_numbers(item) for item in value]
        return value

    return hashlib.sha256(json.dumps(jsonb_numbers(document), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def warning_rows(warnings):
    """Persist only actual core warnings with their exact IDs and counts."""
    result = []
    for warning in warnings:
        detail = _json_value(warning)
        if warning.code == "W_TEAM_HISTORY_SMALL":
            message = (f"Đội {warning.team_id}: {warning.sample_count}/{warning.requested_count} "
                       "ván trong cửa sổ phong độ yêu cầu.")
            group = "TEAM_HISTORY"
        elif warning.code == "W_H2H_MISSING":
            message = "Chưa ghi nhận đối đầu trong kho lịch sử đang dùng."
            group = "HEAD_TO_HEAD"
        elif warning.code == "W_PAIR_HISTORY_MISSING":
            message = (f"Chưa ghi nhận lịch sử của {warning.sample_count}/10 cặp "
                       "tuyển thủ–tướng trong kho lịch sử đang dùng.")
            group = "PLAYER_CHAMPION"
        else:
            raise StorageError("E_STORAGE_CONFLICT", "Chưa có mapping cho mã cảnh báo thực tế.")
        result.append({"warning_code": warning.code, "warning_group": group,
                       "message": message, "severity": "WARNING", "details": detail})
    return result


def _feature_document(result):
    # Histories are stored once as real payloads in evaluation_history; feature-selected
    # game IDs, counts and exclusions remain here, including the exact PRE foundation.
    return {field.name: _json_value(getattr(result, field.name)) for field in fields(result)
            if field.name not in {"pre", "history_snapshot"}}


class EvaluationPersistence:
    def __init__(self, engine):
        self.repository = EvaluationRepository(engine)
        self._history_cache = None

    def _history(self, pre):
        key = (pre.request.history, pre.request.champion_reference)
        if self._history_cache is not None and self._history_cache[0] == key:
            return self._history_cache[1:]
        document = _history_document(pre)
        digest = _history_hash(document)
        self._history_cache = (key, document, digest)
        return document, digest

    def _packet(self, snapshot, bundle, origin, provenance, *, operation_id=None):
        is_post = snapshot.prediction.phase == "POST"
        if operation_id is not None:
            try:
                valid_operation = (
                    isinstance(operation_id, str) and str(UUID(operation_id)) == operation_id
                    and UUID(operation_id).int != 0
                )
            except (ValueError, TypeError, AttributeError):
                valid_operation = False
            if not valid_operation or not is_post or origin != "INTERACTIVE":
                raise StorageError("E_STORAGE_CONFLICT", "Expected a canonical POST operation UUID")
        pre = snapshot.pre if is_post else snapshot
        interactive = pre.request.runtime_context is not None
        if interactive != (origin == "INTERACTIVE"):
            raise StorageError("E_STORAGE_CONFLICT", "Nguồn nhập không khớp context đánh giá.")
        protocol = RETROSPECTIVE_PROTOCOL if origin == "RETROSPECTIVE_IMPORT" else STRICT_PROTOCOL
        _require_snapshot(pre, bundle, evaluation_protocol=protocol,
                          inference_policy=INTERACTIVE_INFERENCE_POLICY if interactive else None)
        comparison = None
        if is_post:
            if interactive:
                comparison = compare_interactive_evaluations(pre, snapshot, bundle)
            else:
                comparison = compare_retrospective_evaluations(pre, snapshot, bundle)
        return self._serialize_packet(
            snapshot, bundle, origin, provenance, comparison, operation_id=operation_id,
        )

    def _serialize_packet(self, snapshot, bundle, origin, provenance, comparison, *, operation_id=None):
        """Serialize after the calling save boundary has validated the complete snapshot."""
        is_post = snapshot.prediction.phase == "POST"
        pre = snapshot.pre if is_post else snapshot
        interactive = pre.request.runtime_context is not None
        history, history_hash = self._history(pre)
        frame, context_frame = _prediction_inputs(
            pre.request, pre.features, bundle, snapshot.features if is_post else None,
        )
        context = pre.request.runtime_context
        # The import/reconstruction clock is never substituted for original inference time.
        inferred_at = context.pre_created_at if interactive and not is_post else None
        provenance = _json_value(provenance or {})
        provenance["original_inference_time_status"] = (
            "APPLICATION_PRE_CLOCK" if inferred_at is not None else "NOT_RECORDED"
        )
        if operation_id is not None:
            if provenance.get("post_operation_id", operation_id) != operation_id:
                raise StorageError("E_STORAGE_CONFLICT", "POST operation provenance conflicts")
            provenance["post_operation_id"] = operation_id
        model = _json_value({
            "family": bundle.family, "identity": bundle.identity, "contract": bundle.contract,
            "feature_schema_version": bundle.feature_schema_version,
            "preprocessing_version": bundle.preprocessing_version,
            "library_versions": dict(bundle.library_versions), "fit_signature": bundle.fit_signature,
        })
        document = {
            "schema_version": "evaluation-input-v1",
            "target": _json_value(pre.request.target),
            "cutoff_record": _json_value(pre.request.cutoff),
            "runtime_context": _json_value(context),
            "history_sha256": history_hash,
            "history_storage": "evaluation_history.payload (complete immutable input)",
            "champion_reference": sorted(pre.request.champion_reference),
            "features": _feature_document(snapshot.features),
            "pre_features": _feature_document(pre.features),
            "feature_columns": list(frame.columns),
            "feature_values": _json_value(frame.iloc[0].tolist()),
            "feature_context": _json_value(context_frame.iloc[0].to_dict()),
            "lineup": _json_value(snapshot.lineup) if is_post else None,
            "comparison": _json_value(comparison),
            "prediction": _json_value(snapshot.prediction),
            "warnings": _json_value(snapshot.warnings),
            "pre_prediction": _json_value(pre.prediction),
            "pre_context_key": pre.context_key, "pre_seal": pre.seal,
            "model": model, "provenance": provenance,
        }
        document = _json_value(document)
        stored_provenance = {**provenance, "input_snapshot_sha256": _history_hash(document)}
        packet = {
            "game_id": None if interactive else pre.request.target.game_id,
            "analysis_id": pre.request.target.game_id if interactive else None,
            "evaluation_type": snapshot.prediction.phase,
            "blue_win_probability": float(snapshot.prediction.p_blue_win),
            "history_cutoff_at": snapshot.prediction.history_cutoff_at,
            "data_version": bundle.identity.dataset_id,
            "model_version": bundle.identity.bundle_version,
            "input_snapshot": document, "context_key": pre.context_key,
            "history_sha256": history_hash,
            "inference_mode": "INTERACTIVE_ANALYSIS" if interactive
            else "REAL_RETROSPECTIVE_SIMULATION",
            "origin": origin, "inferred_at": inferred_at, "provenance": stored_provenance,
            "idempotency_key": _digest((origin, pre.seal, snapshot.prediction,
                                        snapshot.lineup if is_post else None)),
        }
        if operation_id is not None:
            packet["idempotency_key"] = _digest((packet["idempotency_key"], operation_id))
        return packet, history, warning_rows(snapshot.warnings)

    def save_pre(self, pre, bundle, *, provenance=None):
        if pre.prediction.phase != "PRE":
            raise StorageError("E_STORAGE_CONFLICT", "Expected PRE snapshot")
        packet, history, warnings = self._packet(pre, bundle, "INTERACTIVE", provenance)
        saved = self.repository.save(packet, history, warnings)
        if not saved.is_active:
            raise StorageError("E_STORAGE_CONFLICT", "Đánh giá đã mất hiệu lực; xác nhận context mới.")
        return saved

    def save_post(self, post, bundle, pre_evaluation_id, *, provenance=None, operation_id=None):
        if post.prediction.phase != "POST":
            raise StorageError("E_STORAGE_CONFLICT", "Expected POST snapshot")
        packet, history, warnings = self._packet(
            post, bundle, "INTERACTIVE", provenance, operation_id=operation_id,
        )
        saved = self.repository.save(packet, history, warnings, pre_evaluation_id=pre_evaluation_id)
        if not saved.is_active:
            raise StorageError("E_STORAGE_CONFLICT", "POST đã mất hiệu lực; chọn lại đội hình.")
        return saved

    def save_pair(self, pre, post, bundle, *, provenance):
        if pre != post.pre:
            raise StorageError("E_STORAGE_CONFLICT", "POST does not reference the supplied PRE")
        if pre.request.runtime_context is not None:
            raise StorageError("E_STORAGE_CONFLICT", "Nguồn nhập không khớp context đánh giá.")
        _require_snapshot(pre, bundle, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
        comparison = compare_retrospective_evaluations(pre, post, bundle)
        a, history, warnings_a = self._serialize_packet(
            pre, bundle, "RETROSPECTIVE_IMPORT", provenance, None,
        )
        b, _, warnings_b = self._serialize_packet(
            post, bundle, "RETROSPECTIVE_IMPORT", provenance, comparison,
        )
        return self.repository.save_pair(a, b, history, warnings_a, warnings_b)

    def invalidate(self, analysis_id, model_version, *, post_only=False):
        self.repository.invalidate(analysis_id, model_version, post_only=post_only)

    def read(self, evaluation_id):
        result = self.repository.read(evaluation_id)
        if _history_hash(result["history_payload"]) != result["history_sha256"]:
            raise StorageError("E_STORAGE_READ", "Lịch sử đã lưu không khớp nội dung bất biến.")
        if result["input_snapshot"]["prediction"]["p_blue_win"] != result["blue_win_probability"]:
            raise StorageError("E_STORAGE_READ", "Xác suất đọc lại không khớp snapshot.")
        if _history_hash(result["input_snapshot"]) != result["provenance"]["input_snapshot_sha256"]:
            raise StorageError("E_STORAGE_READ", "Snapshot đã lưu không khớp toàn bộ nội dung.")
        return result


def default_store():
    """Lazy engine import; no database dependency in feature/model code or module import."""
    from match_insight.database.engine import engine

    return EvaluationPersistence(engine)
