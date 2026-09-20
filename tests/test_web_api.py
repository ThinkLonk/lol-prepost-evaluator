"""HTTP integration on the same real core pipelines, with synthetic test fixtures.

Only PostgreSQL I/O is replaced. Features, model inference, temporal checks,
comparison seals and persistence serialization execute normally.
"""

from copy import deepcopy
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from match_insight.api.app import create_app
from match_insight.api.service import WebAnalysisService
from match_insight.database.evaluations import SavedEvaluation, StorageError
from match_insight.ml.models import _plain
from match_insight.services import demo, model_comparison
from match_insight.services.persistence import EvaluationPersistence


class MemoryStore:
    def __init__(self):
        self.records = {}
        self.keys = {}
        self.invalidations = []
        self.fail_after_commit = False
        self.fail_invalidation = False
        self.packets = EvaluationPersistence(None)

    def _save(self, snapshot, bundle, provenance, operation_id=None):
        packet, history, warnings = self.packets._packet(
            snapshot, bundle, "INTERACTIVE", provenance, operation_id=operation_id,
        )
        key = packet["idempotency_key"]
        existing = key in self.keys
        if not existing:
            identity = len(self.records) + 1
            self.keys[key] = identity
            self.records[identity] = {**packet, "evaluation_id": identity,
                                      "is_active": True, "history_payload": history,
                                      "warnings": warnings}
        identity = self.keys[key]
        if self.fail_after_commit:
            self.fail_after_commit = False
            raise StorageError("E_STORAGE_WRITE", "Simulated uncertain transaction response")
        return SavedEvaluation(identity, not existing, self.records[identity]["is_active"])

    def save_pre(self, snapshot, bundle, *, provenance):
        return self._save(snapshot, bundle, provenance)

    def save_post(self, snapshot, bundle, pre_evaluation_id, *, provenance, operation_id):
        assert self.records[pre_evaluation_id]["is_active"]
        assert snapshot.pre.request.target.game_id == self.records[pre_evaluation_id]["analysis_id"]
        return self._save(snapshot, bundle, provenance, operation_id)

    def invalidate(self, analysis_id, model_version, *, post_only=False):
        if self.fail_invalidation:
            raise StorageError("E_STORAGE_WRITE", "Simulated invalidation failure")
        self.invalidations.append((analysis_id, model_version, post_only))
        for record in self.records.values():
            if record["analysis_id"] == analysis_id and (
                not post_only or record["evaluation_type"] == "POST"
            ):
                record["is_active"] = False

    def read(self, identity):
        if identity not in self.records:
            raise StorageError("E_STORAGE_READ", "Not found")
        return deepcopy(self.records[identity])


@pytest.fixture
def web(comparison_environment, monkeypatch):
    env = comparison_environment
    store = MemoryStore()
    loads = []

    def runtime_loader():
        loads.append("runtime")
        return env.runtime

    def models_loader(bundle):
        assert bundle is env.canonical_bundle
        loads.append("models")
        return env.models

    def forbid_fit(*args, **kwargs):
        pytest.fail("HTTP runtime must never train")

    monkeypatch.setattr(Pipeline, "fit", forbid_fit)
    service = WebAnalysisService(runtime_loader=runtime_loader, models_loader=models_loader,
                                 store_factory=lambda: store)
    with TestClient(create_app(service)) as client:
        token = client.post("/api/sessions").json()["session_id"]
        headers = {"X-Analysis-Session": token}
        context = {key: _plain(getattr(env.selection, key)) for key in (
            "blue_team_id", "red_team_id", "blue_roster", "red_roster", "patch",
            "context_label", "user_confirmed",
        )}
        yield client, headers, {"operation_id": str(uuid4()), "context": context}, service, store, env, loads


def _pre(web):
    client, headers, payload, *_ = web
    response = client.post("/api/pre", headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _post_payload(pre, env):
    return {"operation_id": str(uuid4()), "pre_evaluation_id": pre["pre_evaluation_id"],
            "champion_ids": list(env.champion_ids)}


def test_full_http_flow_equals_existing_three_model_service(web):
    client, headers, payload, service, store, env, loads = web
    for _ in range(2):
        bootstrap = client.get("/api/bootstrap")
        assert bootstrap.status_code == 200
        assert bootstrap.headers.get("content-encoding") == "gzip"
        assert len(bootstrap.json()["metrics"]["scores"]) == 3
    assert loads == ["runtime", "models"]
    pre = _pre(web)
    saved_pre = deepcopy(store.records[pre["pre_evaluation_id"]])
    session = service.session(headers["X-Analysis-Session"])
    direct = model_comparison.predict_pre_comparison(session.pre, env.canonical_bundle, env.models)
    assert [row["pre"] for row in pre["models"]] == [
        {key: row["pre"][key] for key in ("blue_win_probability", "red_win_probability")}
        for row in direct["models"]
    ]
    request = _post_payload(pre, env)
    response = client.post("/api/post", headers=headers, json=request)
    assert response.status_code == 200, response.text
    post = response.json()
    direct_post = model_comparison.predict_post_comparison(
        session.pre, session.post, env.canonical_bundle, env.models, direct,
    )
    for displayed, source in zip(post["models"], direct_post["models"], strict=True):
        assert displayed["post"]["blue_win_probability"] == source["post"]["blue_win_probability"]
        assert displayed["delta"]["blue"] == pytest.approx(
            100 * source["comparison"]["blue_probability_delta"],
        )
    assert post["models"][0]["delta"] == {"blue": 0.0, "red": 0.0}
    assert post["history_cutoff_at"] == pre["history_cutoff_at"]
    assert post["models"][1]["pre"] == pre["models"][1]["pre"]
    assert store.records[pre["pre_evaluation_id"]] == saved_pre
    assert client.post("/api/post", headers=headers, json=request).json() == post
    assert len(store.records) == 2
    read = client.get(f"/api/evaluations/{post['post_evaluation_id']}")
    assert read.status_code == 200 and read.json()["models"] == post["models"]
    assert len(read.json()["player_champion"]) == 10
    assert read.json()["roster"]["blue"][0]["player"] == {
        "id": post["roster"]["blue"][0]["player"]["id"],
        "name": post["roster"]["blue"][0]["player"]["name"],
    }
    assert "Server-Timing" in response.headers


def test_post_requires_this_sessions_pre_and_complete_unique_lineup(web):
    client, headers, payload, service, store, env, _ = web
    body = _post_payload({"pre_evaluation_id": 1}, env)
    assert client.post("/api/post", headers=headers, json=body).status_code == 409
    pre = _pre(web)
    body["pre_evaluation_id"] = pre["pre_evaluation_id"]
    other = client.post("/api/sessions").json()["session_id"]
    assert client.post("/api/post", headers={"X-Analysis-Session": other}, json=body).status_code == 409
    assert client.post("/api/post", headers=headers, json={**body, "champion_ids": body["champion_ids"][:9]}).status_code == 422
    assert client.post("/api/post", headers=headers, json={**body, "champion_ids": [body["champion_ids"][0]] * 10}).status_code == 422
    assert len(store.records) == 1


@pytest.mark.parametrize("field,value", [("history_cutoff_at", "2030-01-01"),
                                        ("probability", 0.95), ("features", {}),
                                        ("champion_ids", ["266"] * 10)])
def test_pre_rejects_client_clocks_features_and_draft(web, field, value):
    client, headers, payload, _, store, *_ = web
    payload["context"][field] = value
    assert client.post("/api/pre", headers=headers, json=payload).status_code == 422
    assert not store.records


def test_invalidation_preserves_pre_for_draft_but_retires_old_post_requests(web):
    client, headers, payload, service, store, env, _ = web
    pre = _pre(web)
    request = _post_payload(pre, env)
    post = client.post("/api/post", headers=headers, json=request).json()
    response = client.delete("/api/post", headers=headers)
    assert response.status_code == 200
    assert response.json()["pre"] == pre and response.json()["post"] is None
    assert store.records[pre["pre_evaluation_id"]]["is_active"]
    assert not store.records[post["post_evaluation_id"]]["is_active"]
    assert client.post("/api/post", headers=headers, json=request).status_code == 409
    request["operation_id"] = str(uuid4())
    request["champion_ids"] = list(reversed(request["champion_ids"]))
    updated = client.post("/api/post", headers=headers, json=request)
    assert updated.status_code == 200, updated.text
    assert updated.json()["pre_evaluation_id"] == pre["pre_evaluation_id"]
    assert updated.json()["post_evaluation_id"] != post["post_evaluation_id"]
    cleared = client.delete("/api/pre", headers=headers).json()
    assert cleared["pre"] is None and cleared["post"] is None
    assert client.post("/api/post", headers=headers, json=request).status_code == 409
    assert client.post("/api/pre", headers=headers, json=payload).status_code == 409


def test_uncertain_pre_save_reuses_snapshot_and_operation_without_inference(web, monkeypatch):
    client, headers, payload, service, store, env, _ = web
    store.fail_after_commit = True
    response = client.post("/api/pre", headers=headers, json=payload)
    assert response.status_code == 503
    state = client.get("/api/session", headers=headers).json()
    assert state["pre"] is None and state["pending_pre"] == payload
    pending = service.session(headers["X-Analysis-Session"]).pending_pre[1]

    def forbidden(*args, **kwargs):
        pytest.fail("Retry must retain the original PRE and cutoff")

    monkeypatch.setattr(demo, "evaluate_analysis_pre", forbidden)
    restored = _pre(web)
    assert restored["history_cutoff_at"] == pending.prediction.history_cutoff_at.isoformat()
    assert len(store.records) == 1
    assert client.post("/api/pre", headers=headers, json=payload).json() == restored


def test_uncertain_post_save_reuses_snapshot_and_failed_invalidation_keeps_state(web, monkeypatch):
    client, headers, payload, service, store, env, _ = web
    pre = _pre(web)
    request = _post_payload(pre, env)
    store.fail_after_commit = True
    assert client.post("/api/post", headers=headers, json=request).status_code == 503
    assert client.get("/api/session", headers=headers).json()["pending_post"] == request

    def forbidden(*args, **kwargs):
        pytest.fail("POST retry must not perform inference")

    monkeypatch.setattr(demo, "evaluate_analysis_post", forbidden)
    post = client.post("/api/post", headers=headers, json=request)
    assert post.status_code == 200 and len(store.records) == 2
    store.fail_invalidation = True
    assert client.delete("/api/pre", headers=headers).status_code == 503
    assert client.get("/api/session", headers=headers).json()["post"] == post.json()


def test_read_saved_after_new_server_does_not_restore_live_pre(web):
    client, headers, payload, service, store, env, _ = web
    pre = _pre(web)
    restarted = WebAnalysisService(runtime_loader=lambda: env.runtime,
                                   models_loader=lambda bundle: env.models,
                                   store_factory=lambda: store)
    with TestClient(create_app(restarted)) as other:
        assert other.get("/api/session", headers=headers).status_code == 404
        saved = other.get(f"/api/evaluations/{pre['pre_evaluation_id']}")
        assert saved.status_code == 200
        assert saved.json()["models"] == pre["models"]


def test_input_errors_and_files_do_not_expose_internal_data(web):
    client, headers, payload, *_ = web
    payload["context"]["blue_team_id"] = "unknown-team"
    response = client.post("/api/pre", headers=headers, json=payload)
    assert response.status_code == 409
    assert "Traceback" not in response.text
    assert client.get("/api/media/teams/.env").status_code == 404
    assert client.get("/api/media/unknown/file.png").status_code == 404
    assert client.get("/api/does-not-exist").status_code == 404
    assert client.get("/api/health").json()["models_loaded"]
