"""Interactive AppTest flow with isolated I/O and genuine fitted inference."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops
from streamlit.testing.v1 import AppTest
from test_demo_service import analysis_environment as analysis_environment
from test_demo_service import demo_environment as demo_environment
from test_demo_service import synthetic_demo_bundle as synthetic_demo_bundle

import match_insight.ml.models as models
import match_insight.services.evaluation as evaluation
from match_insight.database.demo_catalog import CatalogEntity
from match_insight.features.pre import ROLES, FeatureStat, PreFeatureInputError
from match_insight.services import demo
from match_insight.ui import app as demo_ui

ROOT = Path(__file__).resolve().parents[1]


class MemoryEvaluationStore:
    """Persistence boundary double; feature building and inference stay real."""

    def __init__(self):
        self.records = {}
        self.keys = {}
        self.calls = []
        self.fail_pre_after_commit = False
        self.fail_post_after_commit = False
        self.fail_invalidation = False

    def _save(self, phase, snapshot, bundle, provenance, pre_id=None, operation_id=None):
        pre = snapshot if phase == "PRE" else snapshot.pre
        key = (phase, pre.seal, None if phase == "PRE" else tuple(
            slot.champion_id for slot in snapshot.lineup.slots
        ), operation_id)
        created = key not in self.keys
        if created:
            identity = len(self.records) + 1
            self.keys[key] = identity
            document = {
                "schema_version": "test-saved-json-v1",
                "target": models._plain(pre.request.target),
                "prediction": models._plain(snapshot.prediction),
                "features": models._plain(snapshot.features),
                "lineup": None if phase == "PRE" else models._plain(snapshot.lineup),
                "comparison": None if phase == "PRE" else evaluation.compare_interactive_evaluations(
                    pre, snapshot, bundle,
                ),
                "model": {"family": bundle.family},
                "provenance": deepcopy(provenance),
                "runtime_context": models._plain(pre.request.runtime_context),
            }
            json.dumps(document, allow_nan=False)
            self.records[identity] = {
                "evaluation_id": identity, "evaluation_type": phase,
                "pre_evaluation_id": pre_id, "is_active": True,
                "created_at": "2025-08-01T00:00:00+00:00",
                "inferred_at": pre.request.runtime_context.pre_created_at.isoformat(),
                "origin": "INTERACTIVE_ANALYSIS", "game_id": None,
                "analysis_id": pre.request.target.game_id,
                "model_version": snapshot.prediction.model_bundle_version,
                "data_version": pre.request.runtime_context.snapshot_sha256,
                "input_snapshot": document,
                "warnings": [{
                    "warning_code": warning.code, "warning_group": "COVERAGE",
                    "message": f"{warning.code}: {warning.sample_count} mẫu",
                    "severity": "WARNING", "details": models._plain(warning),
                } for warning in snapshot.warnings],
                "paired_post_ids": [],
            }
            if pre_id is not None:
                self.records[pre_id]["paired_post_ids"].append(identity)
        identity = self.keys[key]
        return SimpleNamespace(
            evaluation_id=identity, created=created, is_active=self.records[identity]["is_active"],
        )

    def save_pre(self, pre, bundle, *, provenance=None):
        self.calls.append(("save_pre", pre.seal))
        saved = self._save("PRE", pre, bundle, provenance)
        if self.fail_pre_after_commit:
            self.fail_pre_after_commit = False
            raise PreFeatureInputError("E_STORAGE_WRITE", "private storage failure")
        if not saved.is_active:
            raise PreFeatureInputError("E_STORAGE_CONFLICT", "Inactive PRE")
        return saved

    def save_post(self, post, bundle, pre_evaluation_id, *, provenance=None, operation_id=None):
        self.calls.append(("save_post", pre_evaluation_id, operation_id))
        saved = self._save("POST", post, bundle, provenance, pre_evaluation_id, operation_id)
        if self.fail_post_after_commit:
            self.fail_post_after_commit = False
            raise PreFeatureInputError("E_STORAGE_WRITE", "private storage failure")
        if not saved.is_active:
            raise PreFeatureInputError("E_STORAGE_CONFLICT", "Inactive POST")
        return saved

    def invalidate(self, analysis_id, model_version, *, post_only=False):
        self.calls.append(("invalidate", analysis_id, model_version, post_only))
        if self.fail_invalidation:
            raise PreFeatureInputError("E_STORAGE_WRITE", "private storage failure")
        for record in self.records.values():
            if (
                record["analysis_id"] == analysis_id and record["model_version"] == model_version
                and (not post_only or record["evaluation_type"] == "POST")
            ):
                record["is_active"] = False

    def read(self, evaluation_id):
        self.calls.append(("read", evaluation_id))
        if evaluation_id not in self.records:
            raise PreFeatureInputError("E_STORAGE_READ", "private missing record")
        return deepcopy(self.records[evaluation_id])


@pytest.fixture(autouse=True)
def memory_store(monkeypatch):
    store = MemoryEvaluationStore()
    monkeypatch.setattr(demo_ui.persistence, "default_store", lambda: store)
    return store


def open_app():
    return AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()


def assert_notice(app):
    assert not app.exception
    assert any(item.value == demo_ui.INTERACTIVE_NOTICE for item in app.info)


def install_runtime(monkeypatch, environment):
    """Double initialization only; all evaluations use the real A services."""
    _, source_runtime, _ = environment
    calls = {"loads": [], "predictions": [], "suggestions": [], "coverage_builds": []}
    original_predict = evaluation.predict_phase
    original_suggest = demo.suggest_analysis_roster
    original_coverage = demo._build_selection_coverage

    def load():
        runtime = deepcopy(source_runtime)
        calls["loads"].append(runtime)
        return runtime

    def predict(bundle, phase, features, metadata, *, inference_policy=None):
        assert inference_policy == models.INTERACTIVE_INFERENCE_POLICY
        calls["predictions"].append(phase)
        return original_predict(
            bundle, phase, features, metadata, inference_policy=inference_policy,
        )

    def suggest(runtime, team_id, **kwargs):
        result = original_suggest(runtime, team_id, **kwargs)
        calls["suggestions"].append((team_id, result))
        return result

    def build_coverage(runtime, cutoff, pre):
        result = original_coverage(runtime, cutoff, pre)
        calls["coverage_builds"].append(result)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("The active UI must not read DB/artifacts, use benchmark targets, fit or save")

    monkeypatch.setattr(demo, "load_analysis_runtime", load)
    monkeypatch.setattr(demo, "suggest_analysis_roster", suggest)
    monkeypatch.setattr(demo, "_build_selection_coverage", build_coverage)
    for name in (
        "load_retrospective_demo", "get_retrospective_case", "evaluate_demo_pre",
        "evaluate_demo_post", "compare_demo_evaluations", "refresh_retrospective_state",
        "make_demo_cases", "build_demo_model", "fit_model_bundle",
        "_read_runtime_data", "_read_pinned_input", "load_retrospective_bundle",
    ):
        monkeypatch.setattr(demo, name, forbidden)
    monkeypatch.setattr(models, "fit_model_bundle", forbidden)
    monkeypatch.setattr(models, "save_bundle", forbidden)
    monkeypatch.setattr(models.joblib, "dump", forbidden)
    for cls in (
        models.Pipeline, models.ColumnTransformer, models.DummyClassifier,
        models.LogisticRegression, models.RandomForestClassifier, models.SimpleImputer,
        models.StandardScaler, models.OneHotEncoder,
    ):
        for method in ("fit", "fit_transform"):
            if hasattr(cls, method):
                monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(evaluation, "predict_phase", predict)
    return calls


def champion_keys():
    return tuple(f"champion_{side}_{role}" for side in ("BLUE", "RED") for role in ROLES)


def roster_keys():
    return tuple(f"roster_{side}_{role}" for side in ("BLUE", "RED") for role in ROLES)


def assert_empty_champions(app):
    assert all(app.selectbox(key).value is None for key in champion_keys())
    assert sum(
        'class="mi-empty-media"' in item.value and "width:64px;height:64px" in item.value
        for item in app.markdown
    ) == 10


def choose_context(app, *, blue="A", red="D", confirm=True):
    # Independent user-selected teams: A and D never met in the fixture.
    app.selectbox("blue_team").select(blue).run()
    app.selectbox("red_team").select(red).run()
    app.text_input("analysis_patch").input("15.01").run()
    if confirm:
        app.checkbox("confirm_context").check().run()
    assert_notice(app)


def start_pre(app):
    app.button("initialize_demo").click().run()
    choose_context(app)
    assert not app.button("create_pre").disabled
    app.button("create_pre").click().run()
    assert_notice(app)
    return app.session_state["evaluation_state"].pre


def select_lineup(app):
    # Explicit simulated user input; never read a target's historical champions.
    choices = tuple(champion.champion_id for champion in demo.CHAMPIONS[:10])
    for key, champion_id in zip(champion_keys(), choices, strict=True):
        app.selectbox(key).select(champion_id).run()
    assert_notice(app)
    return choices


def start_post(app):
    pre = start_pre(app)
    select_lineup(app)
    app.button("create_post").click().run()
    assert_notice(app)
    return pre, app.session_state["evaluation_state"].post


def comparison_tables(app):
    return [
        item.value for item in app.dataframe
        if "Thay đổi (điểm phần trăm)" in item.value.columns
    ]


def assert_no_comparison(app):
    assert not comparison_tables(app)
    assert not any(item.label == "Thông tin truy vết so sánh" for item in app.expander)


def track_comparisons(monkeypatch):
    calls = []
    original = demo.compare_analysis_evaluations

    def compare(runtime, pre, post):
        result = original(runtime, pre, post)
        calls.append((runtime, pre, post, result))
        return result

    monkeypatch.setattr(demo, "compare_analysis_evaluations", compare)
    return calls


def test_interactive_pre_uses_independent_context_and_locks_creation_time(
    monkeypatch, analysis_environment,
):
    env, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    before_io = dict(env.calls)
    app = open_app()
    assert_notice(app)
    assert any(item.value == "Phân tích ván đấu" for item in app.title)
    assert not calls["loads"] and not calls["predictions"]
    assert app.button("create_pre").disabled
    assert app.button("create_post").disabled
    assert not app.metric
    app.button("initialize_demo").click().run()
    assert_notice(app)
    assert app.selectbox("blue_team").value is None
    assert app.selectbox("red_team").value is None
    assert all(app.selectbox(key).value is None for key in roster_keys())
    assert app.text_input("analysis_patch").value == ""
    assert not app.checkbox("confirm_context").value
    assert not any(item.key == "target_game" for item in app.selectbox)
    assert not calls["suggestions"]
    choose_context(app, confirm=False)
    assert app.button("create_pre").disabled
    selection = app.session_state["analysis_input"]
    analysis_id = selection.analysis_id
    assert analysis_id.startswith("analysis:")
    assert analysis_id not in {row.game.game_id for row in env.snapshot.games}
    assert analysis_id not in env.document["time_proofs"]
    assert selection.blue_team_id == "A" and selection.red_team_id == "D"
    assert [app.selectbox(f"roster_BLUE_{role}").value for role in ROLES] == [
        f"a{index}" for index in range(1, 6)
    ]
    assert [app.selectbox(f"roster_RED_{role}").value for role in ROLES] == [
        f"d{index}" for index in range(1, 6)
    ]
    app.checkbox("confirm_context").check().run()
    reads_before = clock["reads"]
    app.button("create_pre").click().run()
    assert_notice(app)
    pre = app.session_state["evaluation_state"].pre
    context = pre.request.runtime_context
    assert pre.request.target.game_id == analysis_id
    assert pre.request.cutoff is None
    assert context.pre_created_at == clock["now"]
    assert context.history_cutoff_at == models.interactive_cutoff(clock["now"])
    # Selection preview checks the current day; actual PRE captures its own T once.
    assert clock["reads"] == reads_before + 2
    assert context.context_source == "USER_PROVIDED"
    assert context.pre_draft_verification == "NOT_VERIFIED"
    assert context.user_confirmed is True
    assert pre.features.h2h_blue.missing and pre.features.h2h_blue.value is None
    assert pre.features.h2h_blue.sample_count == 0
    assert any("W_H2H_MISSING" in item.value for item in app.warning)
    assert calls["predictions"] == ["PRE"]
    blue, red = demo.phase_probabilities(pre)
    summary = demo.analysis_history_summary(app.session_state["demo_runtime"], pre)
    used = summary["pre_used"]
    assert any(
        f"{used['game_count']} ván · mới nhất: {used['latest_ended_at']}" in item.value
        and f"{used['age_days_at_pre']} ngày" in item.value
        for item in app.info
    )
    assert any(
        f"Kho có time proof: {summary['proof_pool']['game_count']} ván" in item.value
        and f"trước cutoff: {summary['before_cutoff']['game_count']} ván" in item.value
        and f"thực dùng cho PRE: {used['game_count']} ván" in item.value
        for item in app.caption
    )
    for team in summary["teams"]:
        assert any(
            f"Đội {team['team_id']}: {team['pre_used']['game_count']} ván thực dùng" in item.value
            and str(team["pre_used"]["latest_ended_at"]) in item.value
            for item in app.caption
        )
    for _, suggestion in calls["suggestions"]:
        assert any(
            "Roster gợi ý từ lịch sử" in item.value
            and suggestion.source_ended_at.isoformat() in item.value
            and suggestion.source_game_id in item.value
            for item in app.caption
        )
    assert {item.label: item.value for item in app.metric} == {
        "Đội A thắng — PRE": f"{blue:.1%}", "Đội D thắng — PRE": f"{red:.1%}",
    }
    displayed = [item.value for item in app.markdown]
    for team in (pre.features.blue, pre.features.red):
        for label, stat, unit in (
            ("Phong độ trong kho lịch sử", team.recent_form, "ván"),
            ("Thắng khi chơi bên hiện tại", team.side_win_rate, "ván"),
            ("Liên tục lực lượng", team.roster_continuity, "ván đối chiếu"),
        ):
            value = demo_ui.MISSING_HISTORY if stat.value is None else f"{stat.value:.1%}"
            expected = f"{value} · {stat.sample_count} {unit}"
            if unit == "ván" and stat.win_count is not None:
                expected += f" · {stat.win_count} thắng"
            if stat.missing:
                expected += " · Thiếu lịch sử"
            assert f"**{label}**" in displayed
            assert expected in displayed
            displayed.remove(expected)
    assert not app.dataframe
    assert_empty_champions(app)
    assert app.button("create_post").disabled
    saved_reads = clock["reads"]
    clock["now"] += timedelta(days=2)
    app.run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["analysis_input"].analysis_id == analysis_id
    assert clock["reads"] == saved_reads
    assert len(calls["loads"]) == 1
    assert env.calls == before_io
    assert calls["predictions"] == ["PRE"]


@pytest.mark.parametrize(
    "value,count,missing,wins,continuity,expected",
    [
        (None, 0, True, 0, False,
         "Chưa ghi nhận trong kho lịch sử đang dùng · 0 ván · 0 thắng · Thiếu lịch sử"),
        (0.0, 3, False, 0, False, "0.0% · 3 ván · 0 thắng"),
        (0.5, 4, False, 2, False, "50.0% · 4 ván · 2 thắng"),
        (0.8, 1, False, None, True, "80.0% · 1 ván đối chiếu"),
        (0.0, 1, False, None, True, "0.0% · 1 ván đối chiếu"),
        (None, 0, True, None, True,
         "Chưa ghi nhận trong kho lịch sử đang dùng · 0 ván đối chiếu · Thiếu lịch sử"),
    ],
)
def test_pre_stat_display_keeps_samples_missing_and_continuity_semantics(
    value, count, missing, wins, continuity, expected,
):
    stat = FeatureStat(value, count, missing, tuple(f"g{i}" for i in range(count)), wins)
    assert demo_ui._pre_stat_text(stat, continuity=continuity) == expected
    if continuity:
        assert "thắng" not in expected
    if missing:
        assert "%" not in expected


def test_roster_suggestions_are_whole_historical_rosters_and_remain_editable(
    monkeypatch, analysis_environment,
):
    _, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    app.button("initialize_demo").click().run()
    choose_context(app)
    for side, team in (("BLUE", "A"), ("RED", "D")):
        suggestion = next(result for identity, result in calls["suggestions"] if identity == team)
        assert tuple(app.selectbox(f"roster_{side}_{role}").value for role in ROLES) == tuple(
            slot.player_id for slot in suggestion.roster
        )
        assert suggestion.source_game_id
        assert suggestion.source_ended_at < suggestion.history_cutoff_at
        assert suggestion.source == "HISTORICAL_ROSTER_NOT_CURRENT_VERIFIED"
    assert [identity for identity, _ in calls["suggestions"]] == ["A", "D"]
    original_id = app.session_state["analysis_input"].analysis_id
    app.selectbox("roster_BLUE_TOP").select("b1").run()
    assert_notice(app)
    assert app.selectbox("roster_BLUE_TOP").value == "b1"
    assert not app.checkbox("confirm_context").value
    assert app.button("create_pre").disabled
    assert app.session_state["analysis_input"].analysis_id == original_id
    app.checkbox("confirm_context").check().run()
    assert not app.button("create_pre").disabled
    app.button("create_pre").click().run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre.request.target.blue_roster[0].player_id == "b1"
    saved_reads = clock["reads"]
    app.run()
    assert clock["reads"] == saved_reads
    assert [identity for identity, _ in calls["suggestions"]] == ["A", "D"]
    assert app.selectbox("roster_BLUE_TOP").value == "b1"


def _environment_with_catalog_twins(environment):
    env, runtime, clock = environment
    team = next(item for item in runtime.catalog.teams if item.identity == "A")
    player = next(item for item in runtime.catalog.players if item.identity == "a1")
    catalog = replace(
        runtime.catalog,
        teams=runtime.catalog.teams + (replace(team, identity="oe:team:unlinked-a"),),
        players=runtime.catalog.players + (replace(player, identity="oe:player:unlinked-a1"),),
    )
    return env, replace(runtime, catalog=catalog), clock


def test_catalog_filters_preserve_pair_and_same_name_player_change_invalidates(
    monkeypatch, analysis_environment,
):
    environment = _environment_with_catalog_twins(analysis_environment)
    env, _, clock = environment
    calls = install_runtime(monkeypatch, environment)
    app = open_app()
    pre, post = start_post(app)
    before_io, reads = dict(env.calls), clock["reads"]
    runtime = app.session_state["demo_runtime"]
    coverage = app.session_state["selection_coverage"]
    team_labels = demo_ui._coverage_labels(
        {item.identity: item for item in runtime.catalog.teams}, coverage.team_counts,
    )
    player_labels = demo_ui._coverage_labels(
        {item.identity: item for item in runtime.catalog.players}, coverage.player_counts,
    )
    assert app.selectbox("team_catalog_scope").value == "linked"
    assert app.selectbox("player_catalog_scope").value == "linked"
    assert team_labels["oe:team:unlinked-a"] not in app.selectbox("blue_team").options
    assert player_labels["oe:player:unlinked-a1"] not in app.selectbox("roster_BLUE_TOP").options
    assert app.selectbox("roster_BLUE_TOP").value == "a1"
    assert len(calls["coverage_builds"]) == 1

    for key in ("team_catalog_scope", "player_catalog_scope"):
        app.selectbox(key).select("all").run()
        assert_notice(app)
        assert app.checkbox("confirm_context").value
        assert app.session_state["evaluation_state"].pre is pre
        assert app.session_state["evaluation_state"].post is post
        assert len(comparison_tables(app)) == 1
    assert team_labels["oe:team:unlinked-a"] in app.selectbox("blue_team").options
    assert player_labels["oe:player:unlinked-a1"] in app.selectbox("roster_BLUE_TOP").options
    assert team_labels["A"] != team_labels["oe:team:unlinked-a"]
    assert player_labels["a1"] != player_labels["oe:player:unlinked-a1"]
    app.run()
    assert clock["reads"] == reads
    assert len(calls["coverage_builds"]) == 1
    assert calls["predictions"] == ["PRE", "POST"]

    app.selectbox("roster_BLUE_TOP").select("oe:player:unlinked-a1").run()
    assert_notice(app)
    assert app.selectbox("roster_BLUE_TOP").value == "oe:player:unlinked-a1"
    assert not app.checkbox("confirm_context").value
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.metric
    assert_no_comparison(app)
    assert any(
        "Chưa ghi nhận lịch sử liên kết cho bản ghi này" in item.value for item in app.caption
    )
    app.selectbox("player_catalog_scope").select("linked").run()
    assert_notice(app)
    assert app.selectbox("roster_BLUE_TOP").value == "oe:player:unlinked-a1"
    assert any("Bản ghi đang chọn nằm ngoài nhóm lọc" in item.value for item in app.caption)
    assert not app.checkbox("confirm_context").value
    assert app.button("create_pre").disabled
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    assert_empty_champions(app)
    select_lineup(app)
    app.button("create_post").click().run()
    assert_notice(app)
    changed = app.session_state["evaluation_state"].post
    old_pair = next(item for item in post.features.player_champion
                    if item.side == "BLUE" and item.role == "TOP")
    new_pair = next(item for item in changed.features.player_champion
                    if item.side == "BLUE" and item.role == "TOP")
    assert old_pair.games_count > 0
    assert new_pair.player_id == "oe:player:unlinked-a1"
    assert new_pair.games_count == 0 and new_pair.win_rate is None and new_pair.missing
    assert len(calls["loads"]) == 1 and env.calls == before_io
    assert len(calls["coverage_builds"]) == 1


def test_same_name_team_change_keeps_manual_roster_path_and_never_first_selects(
    monkeypatch, analysis_environment,
):
    calls = install_runtime(monkeypatch, _environment_with_catalog_twins(analysis_environment))
    app = open_app()
    pre, _ = start_post(app)
    app.selectbox("team_catalog_scope").select("all").run()
    app.selectbox("blue_team").select("oe:team:unlinked-a").run()
    assert_notice(app)
    assert app.selectbox("blue_team").value == "oe:team:unlinked-a"
    assert all(app.selectbox(f"roster_BLUE_{role}").value is None for role in ROLES)
    assert not app.checkbox("confirm_context").value
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.metric
    assert_no_comparison(app)
    assert all(app.session_state[key] is None for key in champion_keys())
    app.selectbox("team_catalog_scope").select("linked").run()
    assert app.selectbox("blue_team").value == "oe:team:unlinked-a"
    assert any("Chưa có roster gợi ý" in item.value for item in app.caption)
    for number, role in enumerate(ROLES, 1):
        # No inferred current membership: these players have history under another team.
        app.selectbox(f"roster_BLUE_{role}").select(f"b{number}").run()
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    current = app.session_state["evaluation_state"].pre
    assert current is not pre
    assert current.request.target.blue_team_id == "oe:team:unlinked-a"
    assert tuple(slot.player_id for slot in current.request.target.blue_roster) == tuple(
        f"b{number}" for number in range(1, 6)
    )
    assert current.features.blue.recent_form.value is None
    assert current.features.blue.recent_form.sample_count == 0
    assert_empty_champions(app)
    assert len(calls["loads"]) == 1


def test_selection_coverage_reuses_session_index_and_refreshes_preview_cutoff(
    monkeypatch, analysis_environment,
):
    _, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    app.button("initialize_demo").click().run()
    choose_context(app)
    first = app.session_state["selection_coverage"]
    app.run()
    assert app.session_state["selection_coverage"] is first
    assert len(calls["coverage_builds"]) == 1
    clock["now"] += timedelta(days=1)
    app.run()
    second = app.session_state["selection_coverage"]
    assert second.history_cutoff_at == first.history_cutoff_at + timedelta(days=1)
    assert len(calls["coverage_builds"]) == 2
    assert app.selectbox("blue_team").value == "A"
    assert app.checkbox("confirm_context").value
    app.button("create_pre").click().run()
    pre = app.session_state["evaluation_state"].pre
    saved = app.session_state["selection_coverage"]
    assert saved.history_cutoff_at == pre.request.runtime_context.history_cutoff_at
    reads = clock["reads"]
    clock["now"] += timedelta(days=2)
    app.run()
    assert app.session_state["selection_coverage"] is saved
    assert clock["reads"] == reads
    app.text_input("analysis_context").input("Changed context after midnight").run()
    app.run()
    assert app.session_state["evaluation_state"].pre is None
    fresh = app.session_state["selection_coverage"]
    assert fresh.history_cutoff_at == models.interactive_cutoff(clock["now"])
    assert fresh.history_cutoff_at != saved.history_cutoff_at
    assert len(calls["coverage_builds"]) == 3
    assert len(calls["loads"]) == 1
    assert calls["predictions"] == ["PRE"]


def test_same_name_same_coverage_labels_are_distinct_without_identity_merge():
    entities = {
        identity: CatalogEntity(identity, "Tên trùng", "assets/players/shared.webp")
        for identity in ("lp_player_1", "oe:player:2")
    }
    labels = demo_ui._coverage_labels(entities, dict.fromkeys(entities, 0))
    assert len(set(labels.values())) == 2
    assert all("mã " in label and "chưa có lịch sử liên kết" in label for label in labels.values())
    assert set(labels) == set(entities)
    assert labels == demo_ui._coverage_labels(dict(reversed(tuple(entities.items()))), {})


@pytest.mark.parametrize("invalid", [
    "same_team", "duplicate_player", "blank_patch", "padded_patch", "blank_label", "unconfirmed",
])
def test_incomplete_or_conflicting_context_blocks_pre(
    monkeypatch, analysis_environment, invalid,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    app.button("initialize_demo").click().run()
    choose_context(app)
    if invalid == "same_team":
        app.selectbox("red_team").select("A").run()
    elif invalid == "duplicate_player":
        app.selectbox("roster_BLUE_JUNGLE").select("a1").run()
    elif invalid == "blank_patch":
        app.text_input("analysis_patch").input(" ").run()
    elif invalid == "padded_patch":
        app.text_input("analysis_patch").input(" 15.01").run()
    elif invalid == "blank_label":
        app.text_input("analysis_context").input(" ").run()
    else:
        app.checkbox("confirm_context").uncheck().run()
    if invalid != "unconfirmed":
        assert app.checkbox("confirm_context").disabled
    assert_notice(app)
    assert app.button("create_pre").disabled
    assert app.button("create_post").disabled
    assert app.session_state["evaluation_state"].pre is None
    assert not calls["predictions"]


def test_two_sessions_keep_separate_runtime_and_analysis_ids(monkeypatch, analysis_environment):
    calls = install_runtime(monkeypatch, analysis_environment)
    first, second = open_app(), open_app()
    first.button("initialize_demo").click().run()
    second.button("initialize_demo").click().run()
    choose_context(first)
    choose_context(second)
    first_runtime = first.session_state["demo_runtime"]
    second_runtime = second.session_state["demo_runtime"]
    assert first_runtime is not second_runtime
    assert first_runtime.bundle is not second_runtime.bundle
    assert first_runtime.bundle.pre_pipeline is not second_runtime.bundle.pre_pipeline
    assert first_runtime.bundle.post_pipeline is not second_runtime.bundle.post_pipeline
    assert first.session_state["analysis_input"].analysis_id != second.session_state[
        "analysis_input"
    ].analysis_id
    first.button("create_pre").click().run()
    assert_notice(first)
    assert first.session_state["evaluation_state"].pre is not None
    assert second.session_state["evaluation_state"].pre is None
    assert not second.metric
    assert len(calls["loads"]) == 2
    assert calls["predictions"] == ["PRE"]


def test_loading_failure_is_sanitized_and_does_not_retry_on_rerun(monkeypatch):
    calls = []

    def fail_load():
        calls.append(1)
        raise RuntimeError("DATABASE_URL=postgresql://private-user:private-password@private-host")

    monkeypatch.setattr(demo, "load_analysis_runtime", fail_load)
    app = open_app()
    app.button("initialize_demo").click().run()
    assert_notice(app)
    assert app.error
    assert all("private-" not in item.value and "DATABASE_URL" not in item.value for item in app.error)
    assert app.button("create_pre").disabled
    assert app.button("create_post").disabled
    assert not app.button("initialize_demo").disabled
    assert not app.metric
    app.run()
    assert_notice(app)
    assert calls == [1]
    assert app.session_state["evaluation_state"].pre is None


def test_pre_failure_does_not_show_stale_probabilities(monkeypatch, analysis_environment):
    install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    start_pre(app)
    assert len(app.metric) == 2

    def fail_pre(*args, **kwargs):
        raise PreFeatureInputError("E_EVIDENCE_SNAPSHOT_MISMATCH", "private evidence details")

    monkeypatch.setattr(demo, "evaluate_analysis_pre", fail_pre)
    app.text_input("analysis_context").input("Reconfirmed context").run()
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    assert app.error
    assert all("private evidence" not in item.value for item in app.error)
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.metric
    assert app.button("create_post").disabled
    assert_no_comparison(app)


def test_full_post_flow_keeps_pre_time_and_invalidates_champions_then_context(
    monkeypatch, analysis_environment,
):
    env, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    comparisons = track_comparisons(monkeypatch)
    app = open_app()
    pre = start_pre(app)
    pre_before = deepcopy(pre)
    assert_empty_champions(app)
    saved_reads = clock["reads"]
    clock["now"] += timedelta(days=2)
    choices = select_lineup(app)
    assert not app.button("create_post").disabled
    assert calls["predictions"] == ["PRE"]
    assert_no_comparison(app)
    app.button("create_post").click().run()
    assert_notice(app)
    state = app.session_state["evaluation_state"]
    post = state.post
    assert state.pre is pre and post.pre is pre and post.features.pre is pre.features
    assert pre == pre_before
    assert tuple(slot.champion_id for slot in post.lineup.slots) == choices
    assert post.prediction.history_cutoff_at == pre.request.runtime_context.history_cutoff_at
    assert clock["reads"] == saved_reads
    assert calls["predictions"] == ["PRE", "POST"]
    blue, red = demo.phase_probabilities(post)
    assert {item.label: item.value for item in app.metric if item.label.endswith("POST")} == {
        "Đội A thắng — POST": f"{blue:.1%}", "Đội D thắng — POST": f"{red:.1%}",
    }
    assert len(app.metric) == 4
    assert comparisons[-1][1] is pre and comparisons[-1][2] is post
    result = comparisons[-1][3]
    assert result["analysis_id"] == pre.request.target.game_id
    assert result["inference_mode"] == "INTERACTIVE_ANALYSIS"
    assert result["model"]["evaluation_protocol"] == models.RETROSPECTIVE_PROTOCOL
    table = comparison_tables(app)[0].copy(deep=True)
    assert table["Bên"].tolist() == ["BLUE", "RED"]
    assert "comparison" not in app.session_state
    assert any(item.label == "Chi tiết lịch sử tuyển thủ–tướng" for item in app.expander)
    detail = next(item.value for item in app.dataframe if "Tướng" in item.value.columns)
    assert len(detail) == 10
    assert detail["Bên"].tolist() == ["BLUE"] * 5 + ["RED"] * 5
    assert detail["Số ván"].tolist() == [pair.games_count for pair in post.features.player_champion]
    assert detail["Số thắng"].tolist() == [pair.wins_count for pair in post.features.player_champion]
    detail_expander = next(
        item for item in app.expander if item.label == "Chi tiết lịch sử tuyển thủ–tướng"
    )
    assert detail_expander.dataframe[0].value.equals(detail)
    assert [item.value for item in app.markdown if item.value in {
        f"**{name}**" for name in demo_ui.ROLE_NAMES.values()
    }] == [f"**{demo_ui.ROLE_NAMES[role]}**" for role in ROLES]
    for pair in post.features.player_champion:
        expected = (
            f"{pair.games_count} ván · {pair.wins_count} thắng · {pair.win_rate:.1%}"
            if pair.games_count else f"0 ván · {demo_ui.MISSING_HISTORY}"
        )
        assert any(item.value == expected for item in app.markdown)
    before_io = dict(env.calls)
    app.run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is post
    assert comparison_tables(app)[0].equals(table)
    assert comparisons[-1][3] == result
    assert clock["reads"] == saved_reads
    assert env.calls == before_io
    app.selectbox("champion_BLUE_TOP").select("99").run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert_no_comparison(app)
    assert calls["predictions"] == ["PRE", "POST"]
    app.button("create_post").click().run()
    assert_notice(app)
    changed = app.session_state["evaluation_state"].post
    assert changed.pre is pre
    assert changed.features.player_champion[0].missing
    assert changed.features.player_champion[0].games_count == 0
    assert changed.features.player_champion[0].win_rate is None
    assert clock["reads"] == saved_reads
    app.text_input("analysis_patch").input("15.02").run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.checkbox("confirm_context").value
    assert app.button("create_pre").disabled
    assert_no_comparison(app)
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    newer = app.session_state["evaluation_state"].pre
    assert newer is not pre
    assert newer.request.target.patch == "15.02"
    assert newer.request.runtime_context.pre_created_at == clock["now"]
    assert_empty_champions(app)
    assert len(calls["loads"]) == 1
    assert env.calls == before_io


@pytest.mark.parametrize("field", ["team", "roster", "label"])
def test_context_edits_clear_pre_and_confirmation(monkeypatch, analysis_environment, field):
    install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre = start_pre(app)
    app.selectbox("champion_BLUE_TOP").select("266").run()
    if field == "team":
        app.selectbox("blue_team").select("B").run()
    elif field == "roster":
        app.selectbox("roster_BLUE_TOP").select("b1").run()
    else:
        app.text_input("analysis_context").input("Another user context").run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.checkbox("confirm_context").value
    assert app.session_state["analysis_input"].analysis_id == pre.request.target.game_id
    assert not app.metric
    assert_no_comparison(app)
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    assert_empty_champions(app)


def test_partial_and_duplicate_lineups_keep_post_disabled(monkeypatch, analysis_environment):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre = start_pre(app)
    app.selectbox("champion_BLUE_TOP").select("266").run()
    assert app.button("create_post").disabled
    select_lineup(app)
    app.selectbox("champion_BLUE_JUNGLE").select("266").run()
    assert_notice(app)
    assert app.button("create_post").disabled
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert calls["predictions"] == ["PRE"]
    assert_no_comparison(app)
    app.selectbox("champion_BLUE_JUNGLE").select("2").run()
    assert not app.button("create_post").disabled


def test_ten_missing_pairs_are_visible_with_counts_not_fake_win_rates(
    monkeypatch, analysis_environment,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre = start_pre(app)
    # Each fixture player has one historical champion. Rotate explicit user choices
    # so all ten selected player/champion pairs have no sample, without changing history.
    original = tuple(item.champion_id for item in demo.CHAMPIONS[:10])
    choices = original[1:] + original[:1]
    for key, champion in zip(champion_keys(), choices, strict=True):
        app.selectbox(key).select(champion).run()
    app.button("create_post").click().run()
    assert_notice(app)
    post = app.session_state["evaluation_state"].post
    assert post.pre is pre
    assert all(
        item.games_count == item.wins_count == 0 and item.win_rate is None and item.missing
        for item in post.features.player_champion
    )
    assert any(
        "10/10 cặp tuyển thủ–tướng" in item.value
        and demo_ui.MISSING_HISTORY.lower() in item.value
        and "W_PAIR_HISTORY_MISSING" in item.value
        for item in app.warning
    )
    assert sum(
        item.value == f"0 ván · {demo_ui.MISSING_HISTORY}" for item in app.markdown
    ) == 10
    detail = next(item.value for item in app.dataframe if "Tướng" in item.value.columns)
    assert detail["Tỷ lệ thắng lịch sử"].tolist() == [demo_ui.MISSING_HISTORY] * 10
    assert detail["Số ván"].tolist() == [0] * 10
    assert len(app.metric) == 4
    assert calls["predictions"] == ["PRE", "POST"]
    # The clicked button was rendered before POST existed in that run. The next
    # rerun must disable it without running inference again or changing the saved pair.
    app.run()
    assert_notice(app)
    assert app.button("create_post").disabled
    assert app.session_state["evaluation_state"].post is post
    assert calls["predictions"] == ["PRE", "POST"]


def test_failed_post_clears_old_post_without_changing_pre(monkeypatch, analysis_environment):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre, post = start_post(app)
    assert post is not None

    def fail_post(*args, **kwargs):
        raise RuntimeError("private artifact path and database details")

    monkeypatch.setattr(demo, "evaluate_analysis_post", fail_post)
    app.selectbox("champion_BLUE_TOP").select("99").run()
    app.button("create_post").click().run()
    assert_notice(app)
    assert app.error
    assert all("private artifact" not in item.value for item in app.error)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert len(app.metric) == 2
    assert calls["predictions"] == ["PRE", "POST"]
    assert_no_comparison(app)


def test_comparison_uses_service_dictionary_and_explicit_percentage_points(
    monkeypatch, analysis_environment,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    original_compare = demo.compare_analysis_evaluations
    original_points = evaluation.percentage_points
    returned, point_inputs = [], []

    def compare(runtime, pre, post):
        # Display double only: real PRE/POST are still validated by the A boundary.
        result = deepcopy(original_compare(runtime, pre, post))
        result["pre"].update(blue_win_probability=0.55, red_win_probability=0.45)
        result["post"].update(blue_win_probability=0.60, red_win_probability=0.40)
        result["comparison"].update(blue_probability_delta=0.05, red_probability_delta=-0.05)
        result["model"].update(
            family="logistic_regression", dataset_id="retrospective:display-dataset",
            split_id="retrospective:display-split", bundle_version="retrospective:display-bundle",
            feature_schema_version="display-schema", preprocessing_version="display-preprocessing",
        )
        result["warnings"] = list(reversed(result["warnings"]))
        returned.append((result, deepcopy(result)))
        return result

    def points(value):
        point_inputs.append(value)
        return original_points(value)

    monkeypatch.setattr(demo, "compare_analysis_evaluations", compare)
    monkeypatch.setattr(evaluation, "percentage_points", points)
    app = open_app()
    start_post(app)
    assert comparison_tables(app)[0].to_dict("records") == [
        {
            "Đội": "Đội A", "Bên": "BLUE", "PRE": "55.0%", "POST": "60.0%",
            "Thay đổi (điểm phần trăm)": "+5.0 điểm phần trăm",
        },
        {
            "Đội": "Đội D", "Bên": "RED", "PRE": "45.0%", "POST": "40.0%",
            "Thay đổi (điểm phần trăm)": "-5.0 điểm phần trăm",
        },
    ]
    assert point_inputs == [0.05, -0.05]
    assert calls["predictions"] == ["PRE", "POST"]
    assert len(returned) == 1
    result, before = returned[0]
    assert result == before
    assert "comparison" not in app.session_state
    assert any(item.label == "Thông tin truy vết so sánh" for item in app.expander)
    traces = [
        json.loads(item.value) if isinstance(item.value, str) else item.value
        for item in app.json
    ]
    trace = next(item for item in traces if "inference_policy" in item)
    assert trace["model"] == result["model"]
    assert trace["inference_policy"] == result["inference_policy"]
    assert trace["analysis_id"] == result["analysis_id"]
    assert trace["runtime_context"] == {
        key: value for key, value in result["runtime_context"].items()
        if key != "history_pool_game_ids"
    }
    names = {"A": "Đội A", "D": "Đội D"}
    expected_warnings = []
    for warning in result["warnings"]:
        code = warning["code"]
        if code == "W_TEAM_HISTORY_SMALL":
            message = (
                f"{names[warning['team_id']]}: {warning['sample_count']}/"
                f"{warning['requested_count']} ván trong cửa sổ lịch sử yêu cầu."
            )
        elif code == "W_H2H_MISSING":
            message = f"Đối đầu: {demo_ui.MISSING_HISTORY.lower()} (0 ván)."
        elif code == "W_PAIR_HISTORY_MISSING":
            message = (
                f"{warning['sample_count']}/10 cặp tuyển thủ–tướng: "
                f"{demo_ui.MISSING_HISTORY.lower()}."
            )
        else:
            pytest.fail(f"Unexpected fixture warning: {code}")
        expected_warnings.append(f"{message} [{code}]")
    assert [item.value for item in app.warning][-len(expected_warnings):] == expected_warnings



def test_incompatible_comparison_removes_table_and_trace_without_stale_output(
    monkeypatch, analysis_environment,
):
    env, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre, _ = start_post(app)
    assert len(comparison_tables(app)) == 1
    before_io, saved_reads = dict(env.calls), clock["reads"]

    def fail_compare(*args, **kwargs):
        raise PreFeatureInputError("E_EVAL_INCOMPATIBLE", "private comparison evidence")

    monkeypatch.setattr(demo, "compare_analysis_evaluations", fail_compare)
    app.run()
    assert_notice(app)
    assert app.error
    assert all("private comparison" not in item.value for item in app.error)
    assert_no_comparison(app)
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert "comparison" not in app.session_state
    assert len(app.metric) == 2
    app.run()
    assert_notice(app)
    assert_no_comparison(app)
    assert app.session_state["evaluation_state"].post is None
    assert len(calls["loads"]) == 1
    assert calls["predictions"] == ["PRE", "POST"]
    assert env.calls == before_io
    assert clock["reads"] == saved_reads


def write_media_png(root, relative_path, color=(0, 90, 180), *, size=(2, 2)):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.new("RGB", size, color) as picture:
        picture.save(path)
    return path


@pytest.mark.parametrize("kind", ["teams", "players", "champions"])
def test_media_renders_exact_catalog_file_in_fixed_frame(monkeypatch, tmp_path, kind):
    relative = f"assets/{kind}/catalog-selected.png"
    path = write_media_png(tmp_path, relative, size=(6, 2))
    source_before = path.read_bytes()
    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", tmp_path)
    opened, rendered = [], []
    original_open = Image.open

    def open_image(filename, *args, **kwargs):
        opened.append((Path(filename), kwargs))
        return original_open(filename, *args, **kwargs)

    def render(frame, **kwargs):
        rendered.append((frame.copy(), kwargs))

    monkeypatch.setattr(demo_ui.Image, "open", open_image)
    monkeypatch.setattr(demo_ui.st, "image", render)
    entity = CatalogEntity("different-from-filename", "Tên hiển thị rất dài " * 8, relative)
    assert demo_ui._show_media(entity, kind) is True
    assert opened == [(path.resolve(), {"formats": ("PNG", "JPEG", "WEBP", "GIF")})]
    assert len(rendered) == 1
    frame, kwargs = rendered[0]
    assert frame.size == demo_ui.MEDIA_SIZES[kind]
    assert kwargs == {"width": frame.width}
    # A 3:1 image remains 3:1 inside the reserved frame, instead of stretching/cropping.
    with Image.new("RGB", frame.size, frame.getpixel((0, 0))[:3]) as background:
        box = ImageChops.difference(frame.convert("RGB"), background).getbbox()
    assert box is not None
    left, top, right, bottom = box
    assert (right - left) / (bottom - top) == pytest.approx(3, abs=0.08)
    assert abs(left - (frame.width - right)) <= 1
    assert abs(top - (frame.height - bottom)) <= 1
    assert path.read_bytes() == source_before


@pytest.mark.parametrize(
    "case",
    [
        "null", "nonstring", "empty", "missing", "corrupt", "bmp_payload", "outside", "wrong_kind",
        "https", "file_url", "unc", "drive", "traversal",
    ],
)
def test_media_rejects_invalid_catalog_paths_and_renders_only_placeholder(
    monkeypatch, tmp_path, case,
):
    write_media_png(tmp_path, "assets/teams/valid.png")
    write_media_png(tmp_path, "assets/players/valid.png")
    outside = write_media_png(tmp_path, "outside.png")
    corrupt = tmp_path / "assets/teams/corrupt.png"
    corrupt.write_bytes(b"this is not an image")
    bmp_payload = tmp_path / "assets/teams/disguised.png"
    with Image.new("RGB", (2, 2), (0, 90, 180)) as picture:
        picture.save(bmp_payload, format="BMP")
    values = {
        "null": None, "nonstring": 42, "empty": "",
        "missing": "assets/teams/missing.png", "corrupt": "assets/teams/corrupt.png",
        "bmp_payload": "assets/teams/disguised.png", "outside": str(outside),
        "wrong_kind": "assets/players/valid.png", "https": "https://example.invalid/logo.png",
        "file_url": "file:///private/logo.png", "unc": r"\\private-server\share\logo.png",
        "drive": r"C:\private\logo.png", "traversal": "assets/teams/../teams/valid.png",
    }
    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", tmp_path)
    opened, rendered = [], []
    original_open = Image.open

    def open_image(filename, *args, **kwargs):
        opened.append(Path(filename))
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(demo_ui.Image, "open", open_image)
    monkeypatch.setattr(
        demo_ui.st, "image", lambda frame, **kwargs: rendered.append((frame.copy(), kwargs)),
    )
    entity = CatalogEntity("entity", "Tên vẫn hiển thị", values[case])
    assert demo_ui._show_media(entity, "teams") is False
    assert len(rendered) == 1
    frame, kwargs = rendered[0]
    assert frame.size == demo_ui.MEDIA_SIZES["teams"]
    assert kwargs == {"width": frame.width}
    assert len(frame.getcolors(frame.width * frame.height)) == 1
    decoded_path = {"corrupt": corrupt, "bmp_payload": bmp_payload}.get(case)
    assert opened == ([decoded_path.resolve()] if decoded_path is not None else [])


@pytest.mark.parametrize("kind", ["teams", "players", "champions"])
def test_missing_media_keeps_same_geometry_and_render_failure_keeps_html_frame(
    monkeypatch, tmp_path, kind,
):
    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", tmp_path)
    frames, fallback = [], []
    monkeypatch.setattr(demo_ui.st, "image", lambda frame, **kwargs: frames.append(frame.copy()))
    assert demo_ui._show_media(None, kind) is False
    assert frames[0].size == demo_ui.MEDIA_SIZES[kind]

    def fail_render(*args, **kwargs):
        raise RuntimeError("private renderer details")

    monkeypatch.setattr(demo_ui.st, "image", fail_render)
    monkeypatch.setattr(
        demo_ui.st, "markdown", lambda text, **kwargs: fallback.append((text, kwargs)),
    )
    assert demo_ui._show_media(None, kind) is False
    width, height = demo_ui.MEDIA_SIZES[kind]
    assert len(fallback) == 1
    assert f"width:{width}px;height:{height}px" in fallback[0][0]
    assert "private" not in fallback[0][0]


@pytest.mark.parametrize("junction", ["kind_root", "nested"])
def test_media_rejects_resolved_junction_escape(monkeypatch, tmp_path, junction):
    project = tmp_path.resolve()
    outside = write_media_png(project, "outside/valid.png").parent
    relative = (
        "assets/teams/valid.png" if junction == "kind_root"
        else "assets/teams/nested/valid.png"
    )
    link = project / "assets/teams"
    if junction == "nested":
        link /= "nested"
    original_resolve = Path.resolve
    frames = []

    def redirected_resolve(path, *args, **kwargs):
        if path.is_relative_to(link):
            return outside / path.relative_to(link)
        return original_resolve(path, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Media outside the fixed asset boundary must not be decoded")

    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", project)
    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    monkeypatch.setattr(demo_ui.Image, "open", forbidden)
    monkeypatch.setattr(demo_ui.st, "image", lambda frame, **kwargs: frames.append(frame.copy()))
    assert demo_ui._show_media(CatalogEntity("team", "Đội", relative), "teams") is False
    assert len(frames) == 1
    assert frames[0].size == demo_ui.MEDIA_SIZES["teams"]
    assert len(frames[0].getcolors(frames[0].width * frames[0].height)) == 1


def test_media_uses_exact_catalog_paths_for_distinct_ids_with_same_name(monkeypatch, tmp_path):
    first = write_media_png(tmp_path, "assets/teams/catalog-first.png", (255, 0, 0))
    second = write_media_png(tmp_path, "assets/teams/catalog-second.png", (0, 0, 255))
    write_media_png(tmp_path, "assets/teams/Shared name.png")
    write_media_png(tmp_path, "assets/teams/first-id.png")
    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", tmp_path)
    opened = []
    original_open = Image.open

    def open_image(filename, *args, **kwargs):
        opened.append(Path(filename))
        return original_open(filename, *args, **kwargs)

    monkeypatch.setattr(demo_ui.Image, "open", open_image)
    monkeypatch.setattr(demo_ui.st, "image", lambda *args, **kwargs: None)
    assert demo_ui._show_media(
        CatalogEntity("first-id", "Shared name", "assets/teams/catalog-first.png"), "teams",
    ) is True
    assert demo_ui._show_media(
        CatalogEntity("second-id", "Shared name", "assets/teams/catalog-second.png"), "teams",
    ) is True
    assert demo_ui._show_media(CatalogEntity("first-id", "Shared name", None), "teams") is False
    assert opened == [first.resolve(), second.resolve()]


def test_fixed_labels_escape_long_catalog_names(monkeypatch):
    rendered = []
    monkeypatch.setattr(
        demo_ui.st, "markdown", lambda value, **kwargs: rendered.append(value),
    )
    name = '<script>"Name & more"</script>' + " A long display name" * 10
    demo_ui._fixed_label(name, team=True)
    demo_ui._fixed_label(name)
    assert "mi-team-label" in rendered[0]
    assert "mi-person-label" in rendered[1]
    assert all("<script>" not in value and "&lt;script&gt;" in value for value in rendered)
    assert all("&quot;" in value and "&amp;" in value for value in rendered)


def test_media_changes_preserve_real_evaluations_and_champion_visibility(
    monkeypatch, tmp_path, analysis_environment,
):
    env, _, clock = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    comparisons = track_comparisons(monkeypatch)
    monkeypatch.setattr(demo_ui, "PROJECT_ROOT", tmp_path)
    for kind in ("teams", "players", "champions"):
        write_media_png(tmp_path, f"assets/{kind}/fixture.png")
        (tmp_path / f"assets/{kind}/corrupt.png").write_bytes(b"not an image")
    outside_media = write_media_png(tmp_path, "outside.png")
    media_calls = []
    original_media = demo_ui._show_media

    def show_media(entity, kind):
        result = original_media(entity, kind)
        media_calls.append((kind, getattr(entity, "identity", None), result))
        return result

    monkeypatch.setattr(demo_ui, "_show_media", show_media)
    app = open_app()
    pre = start_pre(app)
    assert_empty_champions(app)
    assert len(app.image) == 12
    assert not any(kind == "champions" for kind, _, _ in media_calls)
    runtime = app.session_state["demo_runtime"]
    context_key = pre.context_key
    request_before = deepcopy(pre.request)
    before_io, saved_reads = dict(env.calls), clock["reads"]
    catalog = replace(runtime.catalog, **{
        kind: tuple(
            replace(entity, media_file=f"assets/{kind}/fixture.png")
            for entity in getattr(runtime.catalog, kind)
        )
        for kind in ("teams", "players", "champions")
    })
    with_media = replace(runtime, catalog=catalog)
    app.session_state["demo_runtime"] = with_media
    media_calls.clear()
    app.run()
    assert_notice(app)
    assert len(app.image) == 12
    assert not any(kind == "champions" for kind, _, _ in media_calls)
    assert app.session_state["evaluation_state"].pre is pre
    assert_empty_champions(app)
    media_calls.clear()
    app.selectbox("champion_BLUE_TOP").select("266").run()
    assert_notice(app)
    assert len(app.image) == 13
    assert [(identity, shown) for kind, identity, shown in media_calls if kind == "champions"] == [
        ("266", True),
    ]
    select_lineup(app)
    app.button("create_post").click().run()
    assert_notice(app)
    # Ten selected champion frames plus ten POST pair frames, besides 12 context frames.
    assert len(app.image) == 32
    post = app.session_state["evaluation_state"].post
    assert post.pre is pre
    comparison = deepcopy(comparisons[-1][3])
    table = comparison_tables(app)[0].copy(deep=True)
    for mode in ("none", "corrupt", "outside"):
        fallback_catalog = replace(catalog, **{
            kind: tuple(
                replace(entity, media_file=(
                    None if mode == "none" else
                    f"assets/{kind}/corrupt.png" if mode == "corrupt" else str(outside_media)
                ))
                for entity in getattr(catalog, kind)
            )
            for kind in ("teams", "players", "champions")
        })
        app.session_state["demo_runtime"] = replace(runtime, catalog=fallback_catalog)
        app.run()
        assert_notice(app)
        assert len(app.image) == 32
        state = app.session_state["evaluation_state"]
        assert state.context_key == context_key
        assert state.pre is pre and state.post is post
        assert pre.request == request_before
        assert comparisons[-1][3] == comparison
        assert comparison_tables(app)[0].equals(table)
        assert env.calls == before_io
        assert clock["reads"] == saved_reads
        assert calls["predictions"] == ["PRE", "POST"]
        assert len(calls["loads"]) == 1
        displayed = "\n".join(item.value for item in app.markdown)
        assert "Đội A" in displayed and "Đội D" in displayed
        assert all(
            f"Người chơi {slot.player_id}" in displayed
            for slot in (*pre.request.target.blue_roster, *pre.request.target.red_roster)
        )
    app.session_state["demo_runtime"] = with_media
    app.run()
    assert_notice(app)
    assert comparisons[-1][3] == comparison
    media_calls.clear()
    app.text_input("analysis_context").input("New context after media changes").run()
    assert_notice(app)
    assert len(app.image) == 12
    assert not any(kind == "champions" for kind, _, _ in media_calls)
    assert_no_comparison(app)
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    app.checkbox("confirm_context").check().run()
    app.button("create_pre").click().run()
    assert_notice(app)
    assert_empty_champions(app)
    assert len(app.image) == 12
    assert env.calls == before_io


def test_old_retrospective_session_requires_explicit_interactive_initialization(
    monkeypatch, analysis_environment,
):
    env, _, _ = analysis_environment
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    app.session_state["demo_runtime"] = deepcopy(env.runtime)
    app.session_state["target_game"] = "demo-1"
    app.session_state["champion_BLUE_TOP"] = "266"
    app.run()
    assert_notice(app)
    assert not app.button("initialize_demo").disabled
    assert app.button("create_pre").disabled
    assert app.button("create_post").disabled
    assert app.session_state["evaluation_state"].pre is None
    assert not app.metric
    assert not calls["loads"] and not calls["predictions"]
    app.button("initialize_demo").click().run()
    assert_notice(app)
    assert isinstance(app.session_state["demo_runtime"], demo.AnalysisRuntime)
    assert app.selectbox("blue_team").value is None
    assert app.selectbox("red_team").value is None
    assert len(calls["loads"]) == 1


def test_durable_pre_post_readback_in_new_session_never_recomputes(
    monkeypatch, analysis_environment, memory_store,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre, post = start_post(app)
    pre_id = app.session_state["stored_pre_id"]
    post_id = app.session_state["stored_post_id"]
    assert memory_store.records[post_id]["pre_evaluation_id"] == pre_id
    assert memory_store.records[pre_id]["paired_post_ids"] == [post_id]
    assert any(str(pre_id) in item.value for item in app.caption)
    assert any(str(post_id) in item.value for item in app.caption)
    saved_calls = list(memory_store.calls)
    app.run()
    assert memory_store.calls == saved_calls
    assert calls["predictions"] == ["PRE", "POST"]

    restarted = open_app()
    assert restarted.session_state["evaluation_state"].pre is None
    restarted.text_input("saved_evaluation_lookup").input(str(pre_id)).run()
    restarted.button("read_saved_evaluation").click().run()
    assert_notice(restarted)
    assert any(item.value == f"{pre.prediction.p_blue_win:.1%}" for item in restarted.metric)
    assert any("W_H2H_MISSING" in item.value for item in restarted.warning)
    assert any("0 ván" in item.value and "Thiếu lịch sử" in item.value for item in restarted.markdown)
    stored_pre = restarted.session_state["saved_evaluation_document"]
    context = stored_pre["input_snapshot"]["runtime_context"]
    assert context["context_source"] == "USER_PROVIDED"
    assert context["pre_draft_verification"] == "NOT_VERIFIED"
    assert context["history_cutoff_at"] == pre.request.runtime_context.history_cutoff_at.isoformat()
    assert stored_pre["warnings"] == memory_store.records[pre_id]["warnings"]
    assert restarted.button("initialize_demo").disabled is False
    assert calls["predictions"] == ["PRE", "POST"] and len(calls["loads"]) == 1

    restarted.text_input("saved_evaluation_lookup").input(str(post_id)).run()
    assert not restarted.metric
    restarted.button("read_saved_evaluation").click().run()
    assert_notice(restarted)
    assert any(item.value == f"{post.prediction.p_blue_win:.1%}" for item in restarted.metric)
    assert comparison_tables(restarted)
    assert restarted.session_state["evaluation_state"].pre is None
    json.dumps(restarted.session_state["saved_evaluation_document"], allow_nan=False)
    reads = list(memory_store.calls)
    restarted.run()
    assert memory_store.calls == reads
    assert calls["predictions"] == ["PRE", "POST"]


def test_uncertain_pre_save_retry_reuses_exact_unpublished_snapshot(
    monkeypatch, analysis_environment, memory_store,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    app.button("initialize_demo").click().run()
    choose_context(app)
    memory_store.fail_pre_after_commit = True
    app.button("create_pre").click().run()
    assert_notice(app)
    pending = app.session_state["pending_pre"][1]
    assert app.session_state["evaluation_state"].pre is None
    assert not app.metric and app.error
    assert all("private" not in item.value for item in app.error)
    assert len(memory_store.records) == 1
    app.run()
    assert calls["predictions"] == ["PRE"]
    assert len([call for call in memory_store.calls if call[0] == "save_pre"]) == 1
    app.button("create_pre").click().run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is pending
    assert app.session_state["stored_pre_id"] in memory_store.records
    assert len(memory_store.records) == 1
    assert [call[1] for call in memory_store.calls if call[0] == "save_pre"] == [pending.seal] * 2
    assert calls["predictions"] == ["PRE"]


def test_uncertain_post_save_retry_and_champion_context_invalidation(
    monkeypatch, analysis_environment, memory_store,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre = start_pre(app)
    select_lineup(app)
    memory_store.fail_post_after_commit = True
    app.button("create_post").click().run()
    assert_notice(app)
    pending = app.session_state["pending_post"][1]
    operation_id = app.session_state["pending_post"][2]
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert_no_comparison(app)
    assert len(app.metric) == 2
    app.button("create_post").click().run()
    assert app.session_state["evaluation_state"].post is pending
    assert len(memory_store.records) == 2
    assert calls["predictions"] == ["PRE", "POST"]
    assert [call[2] for call in memory_store.calls if call[0] == "save_post"] == [operation_id] * 2
    pre_id = app.session_state["stored_pre_id"]
    post_id = app.session_state["stored_post_id"]
    app.selectbox("champion_BLUE_TOP").select("99").run()
    assert_notice(app)
    assert memory_store.records[pre_id]["is_active"]
    assert not memory_store.records[post_id]["is_active"]
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    app.text_input("analysis_patch").input("15.02").run()
    assert_notice(app)
    assert not memory_store.records[pre_id]["is_active"]
    assert not app.checkbox("confirm_context").value
    assert not app.metric


def test_returning_to_prior_lineup_creates_new_active_post_submission(
    monkeypatch, analysis_environment, memory_store,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    pre, first_post = start_post(app)
    first_id = app.session_state["stored_post_id"]
    first_operation = next(call[2] for call in memory_store.calls if call[0] == "save_post")
    before = deepcopy(memory_store.records[first_id])
    original = app.selectbox("champion_BLUE_TOP").value
    app.selectbox("champion_BLUE_TOP").select("99").run()
    app.selectbox("champion_BLUE_TOP").select(original).run()
    assert app.session_state["evaluation_state"].pre is pre
    assert app.session_state["evaluation_state"].post is None
    assert not memory_store.records[first_id]["is_active"]
    app.button("create_post").click().run()
    assert_notice(app)
    state = app.session_state["evaluation_state"]
    second_id = app.session_state["stored_post_id"]
    assert second_id != first_id
    assert state.pre is pre and state.post == first_post
    assert memory_store.records[second_id]["is_active"]
    assert memory_store.records[first_id] == {**before, "is_active": False}
    second_operation = [call[2] for call in memory_store.calls if call[0] == "save_post"][-1]
    assert first_operation != second_operation
    assert calls["predictions"] == ["PRE", "POST", "POST"]
    previous_calls = list(memory_store.calls)
    app.run()
    assert memory_store.calls == previous_calls
    assert calls["predictions"] == ["PRE", "POST", "POST"]


def test_durable_invalidation_failure_hides_results_and_requires_retry(
    monkeypatch, analysis_environment, memory_store,
):
    calls = install_runtime(monkeypatch, analysis_environment)
    app = open_app()
    start_post(app)
    pre_id = app.session_state["stored_pre_id"]
    memory_store.fail_invalidation = True
    app.text_input("analysis_context").input("New context").run()
    assert_notice(app)
    assert app.session_state["evaluation_state"].pre is None
    assert app.session_state["evaluation_state"].post is None
    assert not app.metric
    assert not any(item.key == "create_pre" for item in app.button)
    assert app.session_state["pending_invalidations"]
    assert app.error and all("private" not in item.value for item in app.error)
    assert memory_store.records[pre_id]["is_active"]
    previous = list(memory_store.calls)
    app.run()
    assert memory_store.calls == previous
    memory_store.fail_invalidation = False
    app.button("retry_storage_invalidation").click().run()
    assert_notice(app)
    assert not memory_store.records[pre_id]["is_active"]
    assert not app.session_state["pending_invalidations"]
    assert app.button("create_pre").disabled
    assert not app.checkbox("confirm_context").value
    assert calls["predictions"] == ["PRE", "POST"]


def test_missing_saved_record_is_sanitized_without_runtime_initialization(memory_store):
    app = open_app()
    app.text_input("saved_evaluation_lookup").input("999").run()
    app.button("read_saved_evaluation").click().run()
    assert_notice(app)
    assert app.error and all("private" not in item.value for item in app.error)
    assert not app.metric
    assert memory_store.calls == [("read", 999)]


@pytest.mark.parametrize("identity", ("not-an-id", "0", "-1", "1.5", "9223372036854775808"))
def test_malformed_saved_id_is_rejected_before_storage_read(memory_store, identity):
    app = open_app()
    app.text_input("saved_evaluation_lookup").input(identity).run()
    app.button("read_saved_evaluation").click().run()
    assert_notice(app)
    assert app.error and not app.metric
    assert memory_store.calls == []
