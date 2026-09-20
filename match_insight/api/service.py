"""Process-wide read-only resources and isolated, retry-safe browser sessions.

Run one API worker: live snapshots belong to that process. PostgreSQL remains the
durable source for saved evaluations, including after API restarts.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from threading import RLock
from time import monotonic, perf_counter
from uuid import uuid4

from match_insight.ml.models import _plain
from match_insight.services import demo, evaluation, model_comparison, persistence


class ApiError(Exception):
    def __init__(self, code, message, status=409):
        self.code, self.message, self.status = code, message, status


def media_url(reference, kind):
    if not isinstance(reference, str):
        return None
    path = PureWindowsPath(reference)
    if (path.drive or path.root or len(path.parts) != 3
            or path.parts[:2] != ("assets", kind) or ".." in path.parts
            or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}):
        return None
    from urllib.parse import quote

    return f"/api/media/{kind}/{quote(path.name)}"


def _entity(entity, kind, counts=None):
    return {
        "id": entity.identity, "name": entity.name,
        "image": media_url(entity.media_file, kind),
        "history_count": None if counts is None else counts.get(entity.identity, 0),
    }


def _coverage_view(summary):
    if isinstance(summary, dict):
        return {key: _coverage_view(value) for key, value in summary.items() if key != "game_ids"}
    if isinstance(summary, list):
        return [_coverage_view(value) for value in summary]
    return summary


def _model_view(document):
    rows = []
    for row in document["models"]:
        result = {"family": row["family"], "label": row["label"]}
        for phase in ("pre", "post"):
            if phase in row:
                result[phase] = {f"{side}_win_probability": row[phase][f"{side}_win_probability"]
                                 for side in ("blue", "red")}
        if "comparison" in row:
            result["delta"] = {
                side: evaluation.percentage_points(row["comparison"][f"{side}_probability_delta"])
                for side in ("blue", "red")
            }
        rows.append(result)
    return rows


def _stat(stat):
    return {key: _plain(getattr(stat, key))
            for key in ("value", "sample_count", "win_count", "missing")}


def _warnings(warnings, names):
    rows = persistence.warning_rows(warnings)
    for row in rows:
        team_id = row["details"].get("team_id")
        if team_id:
            row["message"] = row["message"].replace(team_id, names.get(team_id, team_id))
    return rows


@dataclass
class Session:
    token: str = field(default_factory=lambda: str(uuid4()))
    touched: float = field(default_factory=monotonic)
    lock: RLock = field(default_factory=RLock)
    pre: object = None
    post: object = None
    pre_id: int | None = None
    post_id: int | None = None
    pre_document: dict | None = None
    pre_view: dict | None = None
    post_view: dict | None = None
    pre_request: dict | None = None
    post_request: dict | None = None
    pending_pre: object = None
    pending_post: object = None
    retired_operations: set = field(default_factory=set)


class WebAnalysisService:
    def __init__(self, *, runtime_loader=None, models_loader=None, store_factory=None,
                 session_ttl=14400, max_sessions=128):
        self.runtime_loader = runtime_loader or demo.load_analysis_runtime
        self.models_loader = models_loader or model_comparison.load_models
        self.store_factory = store_factory or persistence.default_store
        self.session_ttl, self.max_sessions = session_ttl, max_sessions
        self._resource_lock = RLock()
        self._sessions_lock = RLock()
        self._sessions = {}
        self.runtime = self.models = self.coverage = None
        self.load_ms = None

    def resources(self):
        with self._resource_lock:
            if self.runtime is None:
                started = perf_counter()
                runtime = self.runtime_loader()
                models = self.models_loader(runtime.bundle)
                self.runtime, self.models = runtime, models
                self.load_ms = round((perf_counter() - started) * 1000, 1)
            return self.runtime, self.models

    def bootstrap(self):
        runtime, models = self.resources()
        with self._resource_lock:
            self.coverage = demo.selection_coverage(runtime, previous=self.coverage)
            coverage = self.coverage
        return {
            "teams": [_entity(row, "teams", coverage.team_counts) for row in runtime.catalog.teams],
            "players": [_entity(row, "players", coverage.player_counts)
                        for row in runtime.catalog.players],
            "champions": [_entity(row, "champions") for row in runtime.catalog.champions
                          if row.identity in runtime.champion_reference],
            "rosters": {team: {
                "slots": _plain(suggestion.roster),
                "source_ended_at": suggestion.source_ended_at.isoformat(),
            } for team, suggestion in coverage.roster_suggestions.items()},
            "roles": list(demo.ROLES),
            "metrics": {
                "partition": "validation", "count": len(runtime.bundle.split.validation),
                "selected_family": runtime.bundle.family,
                "scores": _plain(runtime.bundle.validation_scores),
                "dataset_id": runtime.bundle.identity.dataset_id,
                "split_id": runtime.bundle.identity.split_id,
                "model_set_id": models.model_set_id,
            },
            "history_latest": max((record.game.ended_at for record in runtime.history),
                                  default=runtime.snapshot_read_at).isoformat(),
            "load_ms": self.load_ms,
        }

    def create_session(self):
        with self._sessions_lock:
            now = monotonic()
            expired = [key for key, value in self._sessions.items()
                       if now - value.touched > self.session_ttl]
            for key in expired:
                del self._sessions[key]
            if len(self._sessions) >= self.max_sessions:
                raise ApiError("E_SESSION_LIMIT", "Máy chủ đang bận. Hãy thử lại sau.", 503)
            session = Session()
            self._sessions[session.token] = session
            return {"session_id": session.token, "pre": None, "post": None}

    def session(self, token):
        with self._sessions_lock:
            session = self._sessions.get(token)
            if session is None or monotonic() - session.touched > self.session_ttl:
                raise ApiError("E_SESSION_EXPIRED",
                               "Phiên đã hết hạn hoặc máy chủ đã khởi động lại. Tạo phiên mới; "
                               "các kết quả đã lưu vẫn có thể tra cứu bằng mã.", 404)
            session.touched = monotonic()
            return session

    def state(self, token):
        session = self.session(token)
        with session.lock:
            return {"session_id": token, "pre": deepcopy(session.pre_view),
                    "post": deepcopy(session.post_view),
                    "pending_pre": deepcopy(session.pending_pre[0]) if session.pending_pre else None,
                    "pending_post": deepcopy(session.pending_post[0]) if session.pending_post else None}

    def _provenance(self, runtime, pre, post, comparison, summary):
        target = pre.request.target
        player_ids = {slot.player_id for slot in target.blue_roster + target.red_roster}
        champion_ids = set() if post is None else {slot.champion_id for slot in post.lineup.slots}
        return {
            "team_names": {row.identity: row.name for row in runtime.catalog.teams
                           if row.identity in (target.blue_team_id, target.red_team_id)},
            "player_names": {row.identity: row.name for row in runtime.catalog.players
                             if row.identity in player_ids},
            "champion_names": {row.identity: row.name for row in runtime.catalog.champions
                               if row.identity in champion_ids},
            "history_coverage": summary, "model_comparison": comparison,
        }

    def _view(self, runtime, pre, post, provenance):
        target = pre.request.target
        teams = {row.identity: _entity(row, "teams") for row in runtime.catalog.teams}
        players = {row.identity: _entity(row, "players") for row in runtime.catalog.players}
        champions = {row.identity: _entity(row, "champions") for row in runtime.catalog.champions}
        slots = {} if post is None else {(row.side, row.role): row for row in post.lineup.slots}
        roster = {}
        for side in ("blue", "red"):
            roster[side] = [{
                "role": row.role, "player": players[row.player_id],
                "champion": (champions[slots[(side.upper(), row.role)].champion_id]
                             if (side.upper(), row.role) in slots else None),
            } for row in getattr(target, f"{side}_roster")]
        statistics = {}
        for side in ("blue", "red"):
            features = getattr(pre.features, side)
            statistics[side] = {name: _stat(getattr(features, name))
                                for name in ("recent_form", "side_win_rate", "roster_continuity")}
        statistics["head_to_head"] = _stat(pre.features.h2h_blue)
        pairs = [] if post is None else _plain(post.features.player_champion)
        for pair in pairs:
            pair.pop("game_ids", None)
        return {
            "phase": "POST" if post else "PRE", "active": True,
            "teams": {"blue": teams[target.blue_team_id], "red": teams[target.red_team_id]},
            "roster": roster, "patch": target.patch,
            "context_label": pre.request.runtime_context.context_label,
            "history_cutoff_at": pre.prediction.history_cutoff_at.isoformat(),
            "models": _model_view(provenance["model_comparison"]),
            "coverage": _coverage_view(provenance["history_coverage"]),
            "statistics": statistics, "player_champion": pairs,
            "warnings": _warnings(pre.warnings if post is None else post.warnings,
                                  provenance["team_names"]),
            "metadata": {"dataset_id": runtime.bundle.identity.dataset_id,
                         "model_set_id": provenance["model_comparison"]["model_set_id"],
                         "analysis_id": target.game_id},
        }

    def create_pre(self, token, request):
        session = self.session(token)
        payload = request.model_dump(mode="json")
        with session.lock:
            if session.pre_view is not None:
                if session.pre_request == payload:
                    return deepcopy(session.pre_view)
                raise ApiError("E_CONTEXT_LOCKED", "Chỉnh sửa bối cảnh trước khi tạo PRE mới.")
            if payload["operation_id"] in session.retired_operations:
                raise ApiError("E_OPERATION_RETIRED", "Yêu cầu cũ đã mất hiệu lực. Tạo yêu cầu mới.")
            if session.pending_pre is not None and session.pending_pre[0] != payload:
                raise ApiError("E_PENDING_SAVE", "Có PRE chưa lưu xong. Thử lại hoặc chỉnh sửa bối cảnh.")
            runtime, models = self.resources()
            if session.pending_pre is None:
                started = perf_counter()
                pre = demo.evaluate_analysis_pre(runtime, request.context.to_selection())
                comparison = model_comparison.predict_pre_comparison(pre, runtime.bundle, models)
                predicted = perf_counter()
                summary = demo.analysis_history_summary(runtime, pre)
                provenance = self._provenance(runtime, pre, None, comparison, summary)
                view = self._view(runtime, pre, None, provenance)
                view["timing_ms"] = {"analysis": round((predicted - started) * 1000, 1),
                                     "details": round((perf_counter() - predicted) * 1000, 1)}
                session.pending_pre = (payload, pre, comparison, provenance, view)
            _, pre, comparison, provenance, view = session.pending_pre
            started = perf_counter()
            saved = self.store_factory().save_pre(pre, runtime.bundle, provenance=provenance)
            view = deepcopy(view)
            view["timing_ms"]["save"] = round((perf_counter() - started) * 1000, 1)
            view.update(pre_evaluation_id=saved.evaluation_id, post_evaluation_id=None)
            session.pre, session.pre_id = pre, saved.evaluation_id
            session.pre_document, session.pre_view = comparison, view
            session.pre_request, session.pending_pre = payload, None
            return deepcopy(view)

    def create_post(self, token, request):
        session = self.session(token)
        payload = request.model_dump(mode="json")
        with session.lock:
            if session.pre is None or session.pre_id != request.pre_evaluation_id:
                raise ApiError("E_PRE_REQUIRED", "POST phải tham chiếu PRE hiện tại của phiên này.")
            if session.post_view is not None:
                if session.post_request == payload:
                    return deepcopy(session.post_view)
                raise ApiError("E_LINEUP_LOCKED", "Chỉnh sửa đội hình trước khi tạo POST mới.")
            if payload["operation_id"] in session.retired_operations:
                raise ApiError("E_OPERATION_RETIRED", "Yêu cầu POST cũ đã mất hiệu lực.")
            if session.pending_post is not None and session.pending_post[0] != payload:
                raise ApiError("E_PENDING_SAVE", "Có POST chưa lưu xong. Thử lại hoặc sửa đội hình.")
            runtime, models = self.resources()
            if session.pending_post is None:
                started = perf_counter()
                post = demo.evaluate_analysis_post(runtime, session.pre, tuple(request.champion_ids))
                comparison = model_comparison.predict_post_comparison(
                    session.pre, post, runtime.bundle, models, session.pre_document,
                )
                predicted = perf_counter()
                summary = demo.analysis_history_summary(runtime, session.pre, post)
                provenance = self._provenance(runtime, session.pre, post, comparison, summary)
                view = self._view(runtime, session.pre, post, provenance)
                view["timing_ms"] = {"analysis": round((predicted - started) * 1000, 1),
                                     "details": round((perf_counter() - predicted) * 1000, 1)}
                session.pending_post = (payload, post, provenance, view)
            _, post, provenance, view = session.pending_post
            started = perf_counter()
            saved = self.store_factory().save_post(
                post, runtime.bundle, session.pre_id, provenance=provenance,
                operation_id=str(request.operation_id),
            )
            view = deepcopy(view)
            view["timing_ms"]["save"] = round((perf_counter() - started) * 1000, 1)
            view.update(pre_evaluation_id=session.pre_id, post_evaluation_id=saved.evaluation_id)
            session.post, session.post_id = post, saved.evaluation_id
            session.post_view, session.post_request, session.pending_post = view, payload, None
            return deepcopy(view)

    def invalidate(self, token, *, post_only=False):
        session = self.session(token)
        with session.lock:
            pre = session.pre or (session.pending_pre[1] if session.pending_pre else None)
            if pre is not None:
                # Also reconciles an uncertain previous commit before discarding a retry.
                self.store_factory().invalidate(
                    pre.request.target.game_id, pre.prediction.model_bundle_version,
                    post_only=post_only,
                )
            for name in ("post_request", "pending_post") + (() if post_only else (
                "pre_request", "pending_pre",
            )):
                value = getattr(session, name)
                if value:
                    request = value[0] if name.startswith("pending") else value
                    session.retired_operations.add(request["operation_id"])
            session.post = session.post_id = session.post_view = session.post_request = None
            session.pending_post = None
            if not post_only:
                session.pre = session.pre_id = session.pre_view = session.pre_document = None
                session.pre_request = session.pending_pre = None
            return self.state(token)

    def read_saved(self, evaluation_id):
        record = self.store_factory().read(evaluation_id)
        document = record["input_snapshot"]
        provenance = document.get("provenance") or {}
        comparison = provenance.get("model_comparison")
        target = document["target"]
        lineup = {(row["side"].lower(), row["role"]): row
                  for row in (document.get("lineup") or {}).get("slots", [])}
        roster = {}
        for side in ("blue", "red"):
            roster[side] = []
            for row in target[f"{side}_roster"]:
                champion_id = lineup.get((side, row["role"]), {}).get("champion_id")
                roster[side].append({
                    "role": row["role"],
                    "player": {"id": row["player_id"], "name": provenance.get("player_names", {})
                               .get(row["player_id"], row["player_id"])},
                    "champion": None if champion_id is None else {
                        "id": champion_id, "name": provenance.get("champion_names", {})
                        .get(champion_id, champion_id),
                    },
                })
        pre_features = document["pre_features"]
        statistics = {side: {name: {key: pre_features[side][name][key]
                                    for key in ("value", "sample_count", "win_count", "missing")}
                            for name in ("recent_form", "side_win_rate", "roster_continuity")}
                      for side in ("blue", "red")}
        statistics["head_to_head"] = {key: pre_features["h2h_blue"][key]
                                      for key in ("value", "sample_count", "win_count", "missing")}
        # A saved result is displayed, never restored as an executable live snapshot.
        return {
            "evaluation_id": evaluation_id, "phase": record["evaluation_type"],
            "active": record["is_active"],
            "teams": {side: {"id": target[f"{side}_team_id"],
                             "name": provenance.get("team_names", {}).get(
                                 target[f"{side}_team_id"], target[f"{side}_team_id"])}
                      for side in ("blue", "red")},
            "patch": target["patch"], "history_cutoff_at": str(record["history_cutoff_at"]),
            "roster": roster, "statistics": statistics,
            "player_champion": _coverage_view(document["features"].get("player_champion", [])),
            "models": _model_view(comparison) if comparison else [{
                "family": document["model"]["family"],
                "label": model_comparison.MODEL_LABELS.get(document["model"]["family"],
                                                           document["model"]["family"]),
                record["evaluation_type"].lower(): {
                    "blue_win_probability": record["blue_win_probability"],
                    "red_win_probability": 1 - record["blue_win_probability"],
                },
            }],
            "coverage": _coverage_view(provenance.get("history_coverage")),
            "warnings": _plain(record.get("warnings", [])),
        }
