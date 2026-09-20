"""Real three-model inference through Streamlit; only database I/O is doubled."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from match_insight.database.evaluations import SavedEvaluation, StorageError
from match_insight.features.pre import ROLES
from match_insight.ml import models
from match_insight.services import demo, model_comparison, persistence
from match_insight.ui.app import _probability_figure


class MemoryRepository:
    def __init__(self):
        self.records = {}
        self.fail_next = False

    def save(self, packet, history, warnings, pre_evaluation_id=None):
        if self.fail_next:
            self.fail_next = False
            raise StorageError("E_STORAGE_WRITE", "Fixture transaction failed")
        for identity, record in self.records.items():
            if record["idempotency_key"] == packet["idempotency_key"]:
                assert record["input_snapshot"] == packet["input_snapshot"]
                return SavedEvaluation(identity, False, record["is_active"])
        identity = len(self.records) + 1
        self.records[identity] = deepcopy({
            **packet, "evaluation_id": identity, "is_active": True,
            "pre_evaluation_id": pre_evaluation_id, "history_payload": history,
            "warnings": warnings, "created_at": "2025-09-01T12:00:00+00:00",
        })
        return SavedEvaluation(identity, True, True)

    def invalidate(self, analysis_id, model_version, *, post_only=False):
        for record in self.records.values():
            if record["analysis_id"] == analysis_id and record["model_version"] == model_version:
                if not post_only or record["evaluation_type"] == "POST":
                    record["is_active"] = False

    def read(self, evaluation_id):
        return deepcopy(self.records[evaluation_id])


@pytest.fixture
def ui_environment(comparison_environment, monkeypatch):
    env = comparison_environment
    calls = {"runtime": 0, "models": 0, "predictions": []}

    def load_runtime():
        calls["runtime"] += 1
        return env.runtime

    def load_models(bundle):
        assert bundle is env.canonical_bundle
        calls["models"] += 1
        return env.models

    original_predict = model_comparison.predict_phase

    def predict(bundle, phase, *args, **kwargs):
        calls["predictions"].append((bundle.family, phase))
        return original_predict(bundle, phase, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("No runtime fitting, model save or fallback")

    monkeypatch.setattr(demo, "load_analysis_runtime", load_runtime)
    monkeypatch.setattr(model_comparison, "load_models", load_models)
    monkeypatch.setattr(model_comparison, "predict_phase", predict)
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "fit_transform", forbidden)
    monkeypatch.setattr(models, "fit_model_bundle", forbidden)
    monkeypatch.setattr(models, "save_bundle", forbidden)
    repository = MemoryRepository()
    store = persistence.EvaluationPersistence(None)
    store.repository = repository
    monkeypatch.setattr(persistence, "default_store", lambda: store)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30,
    )
    return SimpleNamespace(env=env, app=app, repository=repository, store=store, calls=calls)


def _no_error(app):
    assert not app.exception
    assert not app.error


def _prepare(ui):
    app = ui.app.run()
    _no_error(app)
    app.selectbox("blue_team").select("A")
    app.selectbox("red_team").select("D")
    app.text_input("analysis_patch").set_value("15.01").run()
    _no_error(app)
    assert app.button("create_pre").disabled
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    _no_error(app)
    return app


def _lineup(ui):
    for (side, role), champion in zip(
        ((side, role) for side in ("BLUE", "RED") for role in ROLES),
        ui.env.champion_ids, strict=True,
    ):
        ui.app.selectbox(f"champion_{side}_{role}").select(champion)
    ui.app.run()
    _no_error(ui.app)


def _document(app, name):
    return app.session_state["presentation_cache"][name][4]


def _visible_text(app):
    return "\n".join(item.value for kind in ("markdown", "caption", "warning")
                     for item in getattr(app, kind))


def test_three_model_flow_persistence_rerun_and_invalidation(ui_environment):
    ui = ui_environment
    app = _prepare(ui)
    pre = app.session_state["evaluation_state"].pre
    pre_doc = _document(app, "models_pre")
    assert pre.request.target.game_id.startswith("analysis:")
    assert not any(record.game.game_id == pre.request.target.game_id for record in ui.env.runtime.history)
    assert len(app.get("plotly_chart")) == 3
    headings = [item.value for item in app.subheader]
    assert headings.index("So sánh dự đoán của ba mô hình") < headings.index(
        "Đội hình cuối cùng — năm vị trí mỗi đội",
    )
    assert all(app.selectbox(f"champion_{side}_{role}").value is None
               for side in ("BLUE", "RED") for role in ROLES)
    assert "2 ván" in _visible_text(app)
    assert "Liên tục lực lượng" in _visible_text(app)
    pre_id = app.session_state["stored_pre_id"]
    assert ui.store.read(pre_id)["input_snapshot"]["provenance"]["model_comparison"] == pre_doc
    _lineup(ui)
    app.button("create_post").click().run()
    _no_error(app)
    state = app.session_state["evaluation_state"]
    assert state.pre is pre and state.post.pre is pre
    post_doc = _document(app, "models_post")
    assert post_doc["history_cutoff_at"] == pre_doc["history_cutoff_at"]
    assert post_doc["runtime_context"] == pre_doc["runtime_context"]
    for row, previous in zip(post_doc["models"], pre_doc["models"], strict=True):
        assert row["pre"] == previous["pre"]
        for side in ("blue", "red"):
            delta = row["comparison"][f"{side}_probability_delta"]
            assert delta == pytest.approx(
                row["post"][f"{side}_win_probability"] - row["pre"][f"{side}_win_probability"],
            )
            assert f"{100 * delta:+.1f} điểm phần trăm" in _visible_text(app)
    assert len(app.get("plotly_chart")) == 3
    headings = [item.value for item in app.subheader]
    assert headings.index("POST — đội hình đã xác nhận") < headings.index(
        "So sánh dự đoán của ba mô hình",
    )
    json.dumps(post_doc, allow_nan=False)
    post_id = app.session_state["stored_post_id"]
    stored = ui.store.read(post_id)
    assert stored["pre_evaluation_id"] == pre_id
    assert stored["input_snapshot"]["provenance"]["model_comparison"] == post_doc
    assert ui.calls["predictions"] == [
        ("baseline", "PRE"), ("random_forest", "PRE"),
        ("baseline", "POST"), ("random_forest", "POST"),
    ]
    app.run()
    _no_error(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert len(ui.calls["predictions"]) == 4
    assert ui.calls["runtime"] == ui.calls["models"] == 1
    assert len(ui.repository.records) == 2

    # Switching champions preserves all three PRE values, and hides every POST/delta.
    replacement = next(item.champion_id for item in demo.CHAMPIONS
                       if item.champion_id not in ui.env.champion_ids)
    app.selectbox("champion_BLUE_TOP").select(replacement).run()
    _no_error(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert _document(app, "models_pre") is pre_doc
    assert "models_post" not in app.session_state["presentation_cache"]
    assert "Δ BLUE" not in _visible_text(app)
    assert not ui.repository.records[post_id]["is_active"]

    app.text_input("analysis_patch").set_value("15.02").run()
    _no_error(app)
    assert not app.checkbox("confirm_context").value
    assert app.session_state["evaluation_state"].pre is None
    assert not app.get("plotly_chart")
    assert not ui.repository.records[pre_id]["is_active"]
    assert len(ui.calls["predictions"]) == 4


def test_readback_three_models_does_not_predict(ui_environment):
    ui = ui_environment
    app = _prepare(ui)
    _lineup(ui)
    app.button("create_post").click().run()
    post_id = app.session_state["stored_post_id"]
    count = len(ui.calls["predictions"])
    app.text_input("analysis_context").set_value("New context").run()
    app.text_input("saved_evaluation_lookup").set_value(str(post_id)).run()
    app.button("read_saved_evaluation").click().run()
    _no_error(app)
    assert len(app.get("plotly_chart")) == 3
    assert len(ui.calls["predictions"]) == count
    assert "Logistic Regression" in _visible_text(app)
    assert "Δ BLUE" in _visible_text(app)


def test_save_retry_keeps_pre_and_comparison_without_repredicting(ui_environment):
    ui = ui_environment
    ui.repository.fail_next = True
    app = ui.app.run()
    app.selectbox("blue_team").select("A")
    app.selectbox("red_team").select("D")
    app.text_input("analysis_patch").set_value("15.01").run()
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert not app.exception and app.error
    assert not app.get("plotly_chart")
    assert not ui.repository.records
    pending = app.session_state["pending_pre"][1]
    assert len(ui.calls["predictions"]) == 2
    app.button("create_pre").click().run()
    _no_error(app)
    assert app.session_state["evaluation_state"].pre is pending
    assert len(ui.calls["predictions"]) == 2
    assert len(ui.repository.records) == 1


def test_missing_comparison_artifact_fails_closed_without_autoretry(ui_environment, monkeypatch):
    def missing(*args, **kwargs):
        raise models.ModelInputError("E_MODEL_BUNDLE_INVALID", "Missing manifest")

    monkeypatch.setattr(model_comparison, "load_models", missing)
    ui = ui_environment
    app = ui.app.run()
    assert not app.exception and app.error
    assert app.button("create_pre").disabled
    assert not app.get("plotly_chart")
    app.run()
    assert ui.calls["runtime"] == 1
    assert not ui.calls["predictions"]
    assert not ui.repository.records


@pytest.mark.parametrize("blue", [0.0, 0.001, 0.5, 1.0])
def test_plotly_common_scale_order_and_side_colors(blue):
    row = {
        "pre": {"blue_win_probability": blue, "red_win_probability": 1 - blue},
        "post": {"blue_win_probability": 0.3, "red_win_probability": 0.7},
    }
    figure = _probability_figure(row, "Blue team", "Red team")
    assert list(figure.layout.xaxis.range) == [0, 100]
    assert list(figure.layout.xaxis.ticktext) == ["0%", "25%", "50%", "75%", "100%"]
    assert figure.layout.margin.r >= 32
    assert figure.layout.barmode == "stack"
    assert figure.layout.legend.traceorder == "normal"
    assert figure.layout.legend.itemclick is False
    assert figure.layout.legend.itemdoubleclick is False
    assert list(figure.data[0].y) == ["PRE", "POST"]
    assert list(figure.data[0].x) == [100 * blue, 30]
    assert list(figure.data[1].x) == [100 * (1 - blue), 70]
    assert figure.data[0].marker.color == "#2563eb"
    assert figure.data[1].marker.color == "#ef4444"
