"""Interactive analysis presentation; application services own time and inference."""

import hashlib
from collections import Counter
from dataclasses import asdict, replace
from html import escape
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageOps

from match_insight import SYSTEM_NAME
from match_insight.data_processing.reference_common import _IMAGE_EXTENSIONS
from match_insight.features.pre import ROLES, PlayerSlot, PreFeatureInputError
from match_insight.services import demo, evaluation, persistence
from match_insight.services.evaluation import EvaluationState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERACTIVE_NOTICE = (
    "Ước lượng từ dữ liệu lịch sử; bối cảnh và đội hình do người dùng cung cấp."
)
MISSING_HISTORY = "Chưa ghi nhận trong kho lịch sử đang dùng"
MEDIA_SIZES = {"teams": (128, 96), "players": (80, 96), "champions": (64, 64)}
ROLE_NAMES = {
    "TOP": "Đường trên", "JUNGLE": "Đi rừng", "MID": "Đường giữa",
    "BOT": "Xạ thủ", "SUPPORT": "Hỗ trợ",
}


def _empty_media_frame(kind):
    width, height = MEDIA_SIZES[kind]
    st.markdown(
        f'<div class="mi-empty-media" style="width:{width}px;height:{height}px" '
        'aria-label="Chưa có ảnh"></div>', unsafe_allow_html=True,
    )


def _show_media(entity, kind):
    """Contain optional local media in a fixed frame; never change the source file."""
    if kind not in MEDIA_SIZES:
        return False
    size = MEDIA_SIZES[kind]
    frame = Image.new("RGBA", size, (28, 31, 38, 255))
    found = False
    reference = getattr(entity, "media_file", None)
    try:
        if isinstance(reference, str):
            relative = PureWindowsPath(reference)
            if (
                not relative.drive and not relative.root and ":" not in reference
                and ".." not in relative.parts and relative.parts[:2] == ("assets", kind)
                and relative.suffix.lower() in _IMAGE_EXTENSIONS
            ):
                project = PROJECT_ROOT.resolve()
                # Do not resolve this boundary: a junction must not widen it.
                allowed = project / "assets" / kind
                path = project.joinpath(*relative.parts).resolve()
                if path.is_relative_to(allowed) and path.is_file():
                    with Image.open(path, formats=("PNG", "JPEG", "WEBP", "GIF")) as picture:
                        picture.load()
                        contained = ImageOps.contain(picture.convert("RGBA"), size)
                        frame.alpha_composite(contained, (
                            (size[0] - contained.width) // 2,
                            (size[1] - contained.height) // 2,
                        ))
                        found = True
    except Exception:
        # Unreadable/corrupt media never blocks the analysis.
        found = False
    try:
        # Empty media reserves exactly the same space as a valid image.
        st.image(frame, width=size[0])
    except Exception:
        _empty_media_frame(kind)
        return False
    return found


def _fixed_label(text, *, team=False):
    css_class = "mi-team-label" if team else "mi-person-label"
    st.markdown(
        f'<div class="{css_class}" title="{escape(text, quote=True)}">'
        f'{escape(text)}</div>', unsafe_allow_html=True,
    )


def _widget_key(side, role):
    return f"champion_{side}_{role}"


def _reset_champion_choices():
    for side in ("BLUE", "RED"):
        for role in ROLES:
            st.session_state[_widget_key(side, role)] = None


def _clear_context_result():
    _invalidate_stored()
    st.session_state["confirm_context"] = False
    st.session_state["evaluation_state"] = EvaluationState()
    for key in ("stored_pre_id", "stored_post_id", "pending_pre", "pending_post"):
        st.session_state.pop(key, None)
    _reset_champion_choices()


def _invalidate_stored(*, post_only=False):
    state = st.session_state.get("evaluation_state", EvaluationState())
    pending_pre = st.session_state.get("pending_pre")
    pre = state.pre if state.pre is not None else (
        None if pending_pre is None else pending_pre[1]
    )
    relevant = (
        st.session_state.get("stored_post_id") or st.session_state.get("pending_post")
        if post_only else
        st.session_state.get("stored_pre_id") or pending_pre
    )
    if pre is None or not relevant:
        return
    job = (pre.request.target.game_id, pre.prediction.model_bundle_version, post_only)
    try:
        persistence.default_store().invalidate(job[0], job[1], post_only=job[2])
    except Exception:
        pending = st.session_state.setdefault("pending_invalidations", [])
        if job not in pending:
            pending.append(job)


def _clear_stored_post():
    _invalidate_stored(post_only=True)
    st.session_state.pop("stored_post_id", None)
    st.session_state.pop("pending_post", None)
    state = st.session_state.get("evaluation_state")
    if isinstance(state, EvaluationState) and state.post is not None:
        st.session_state["evaluation_state"] = replace(state, post=None)


def _storage_ready():
    pending = st.session_state.get("pending_invalidations", [])
    if not pending:
        return True
    st.error("Chưa thể cập nhật trạng thái bản lưu. Kết quả cũ đã ẩn; hãy thử đồng bộ lại.")
    if st.button("Thử đồng bộ trạng thái bản lưu", key="retry_storage_invalidation"):
        remaining = []
        for analysis_id, model_version, post_only in pending:
            try:
                persistence.default_store().invalidate(
                    analysis_id, model_version, post_only=post_only,
                )
            except Exception:
                remaining.append((analysis_id, model_version, post_only))
        st.session_state["pending_invalidations"] = remaining
        return not remaining
    return False


def _save_provenance(runtime, pre, post=None):
    target = pre.request.target
    player_ids = {slot.player_id for slot in target.blue_roster + target.red_roster}
    champion_ids = set() if post is None else {slot.champion_id for slot in post.lineup.slots}
    return {
        "team_names": {
            item.identity: item.name for item in runtime.catalog.teams
            if item.identity in (target.blue_team_id, target.red_team_id)
        },
        "player_names": {
            item.identity: item.name for item in runtime.catalog.players if item.identity in player_ids
        },
        "champion_names": {
            item.identity: item.name for item in runtime.catalog.champions
            if item.identity in champion_ids
        },
        "history_coverage": demo.analysis_history_summary(runtime, pre, post),
    }


def _persist_pre(runtime, selection):
    pending = st.session_state.get("pending_pre")
    if pending is None:
        snapshot = demo.evaluate_analysis_pre(runtime, selection)
        pending = (selection, snapshot)
        st.session_state["pending_pre"] = pending
    if pending[0] != selection:
        raise PreFeatureInputError("E_STORAGE_CONFLICT", "Pending PRE context changed")
    snapshot = pending[1]
    saved = persistence.default_store().save_pre(
        snapshot, runtime.bundle, provenance=_save_provenance(runtime, snapshot),
    )
    st.session_state["stored_pre_id"] = saved.evaluation_id
    st.session_state.pop("pending_pre", None)
    return snapshot


def _persist_post(runtime, pre, choices):
    pending = st.session_state.get("pending_post")
    key = (pre.seal, tuple(choices))
    if pending is None:
        snapshot = demo.evaluate_analysis_post(runtime, pre, choices)
        pending = (key, snapshot, str(uuid4()))
        st.session_state["pending_post"] = pending
    if pending[0] != key:
        raise PreFeatureInputError("E_STORAGE_CONFLICT", "Pending POST context changed")
    snapshot = pending[1]
    saved = persistence.default_store().save_post(
        snapshot, runtime.bundle, st.session_state["stored_pre_id"],
        provenance=_save_provenance(runtime, pre, snapshot),
        operation_id=pending[2],
    )
    st.session_state["stored_post_id"] = saved.evaluation_id
    st.session_state.pop("pending_post", None)
    return snapshot


def _message(error):
    messages = {
        "E_PRE_SNAPSHOT_REQUIRED": "Hãy tạo PRE trước khi tạo POST.",
        "E_PRE_INPUT_INCOMPLETE": "Thông tin bối cảnh hoặc tuyển thủ chưa đầy đủ.",
        "E_TEAMS_INVALID": "Hãy chọn hai đội khác nhau.",
        "E_LINEUP_INVALID": "Hãy chọn đủ mười tuyển thủ khác nhau theo năm vị trí mỗi đội.",
        "E_LINEUP_INCOMPLETE": "Hãy chọn đủ 10 tướng cuối cùng.",
        "E_CHAMPION_ID_INVALID": "Có vị trí chưa chọn tướng hợp lệ.",
        "E_CHAMPION_UNKNOWN": "Có tướng nằm ngoài danh mục hiện có.",
        "E_SOURCE_CONFLICT": "Thông tin nguồn hoặc bối cảnh đang có xung đột.",
        "E_EVAL_INCOMPATIBLE": "Bối cảnh đã thay đổi. Hãy xác nhận và tạo PRE mới.",
        "E_PRE_HISTORY_MISMATCH": "Lịch sử không còn khớp PRE đã lưu.",
        "E_MODEL_LIBRARY_MISMATCH": "Model không tương thích với môi trường hiện tại.",
        "E_MODEL_BUNDLE_INVALID": "Không thể nạp model đã được duyệt.",
        "E_REAL_MODEL_METADATA_MISSING": "Chưa có đủ tệp model đã được duyệt.",
        "E_REAL_MODEL_METADATA_INVALID": "Thông tin model không hợp lệ.",
        "E_REAL_MODEL_IDENTITY_MISMATCH": "Model không khớp bản đã được duyệt.",
        "E_REAL_MODEL_HASH_MISMATCH": "Tệp model không khớp bản đã được duyệt.",
        "E_EVIDENCE_SNAPSHOT_MISMATCH": "Dữ liệu hiện tại không khớp bộ đầu vào đã được duyệt.",
        "E_EVIDENCE_HASH_MISMATCH": "Tệp đầu vào không khớp bản đã được duyệt.",
        "E_STORAGE_WRITE": "Chưa lưu được đánh giá. Hãy thử lại; kết quả chưa được công bố.",
        "E_STORAGE_READ": "Không đọc được bản đánh giá đã lưu. Hãy kiểm tra mã bản lưu.",
        "E_STORAGE_CONFLICT": "Bản lưu đang xung đột với context hoặc trạng thái đánh giá.",
    }
    if error.code.startswith("E_CUTOFF_"):
        return "Mốc lịch sử chưa đáp ứng chính sách của phiên phân tích."
    return messages.get(error.code, "Đầu vào chưa đáp ứng điều kiện tạo kết quả.")


def _attempt(action):
    try:
        return action()
    except PreFeatureInputError as error:
        st.error(_message(error))
    except Exception:
        st.error("Không thể thực hiện thao tác. Hãy kiểm tra dữ liệu và cấu hình chạy ứng dụng.")
    return None


def _rate(value):
    return MISSING_HISTORY if value is None else f"{value:.1%}"


def _pre_stat_text(stat, *, continuity=False):
    unit = "ván đối chiếu" if continuity else "ván"
    text = f"{_rate(stat.value)} · {stat.sample_count} {unit}"
    if not continuity and stat.win_count is not None:
        text += f" · {stat.win_count} thắng"
    if stat.missing:
        text += " · Thiếu lịch sử"
    return text


def _show_saved_evaluation(record):
    """Render primitive persisted JSON without restoring a live evaluation snapshot."""
    document = record["input_snapshot"]
    target = document["target"]
    provenance = document.get("provenance") or {}
    names = dict(provenance.get("team_names", {}))
    for side in ("blue", "red"):
        identity = target[f"{side}_team_id"]
        names.setdefault(identity, identity)
    st.caption(
        f"Bản lưu: {record['evaluation_id']} · {record['evaluation_type']} · "
        f"Trạng thái lúc đọc: {'Còn hiệu lực' if record['is_active'] else 'Đã vô hiệu hóa'}"
    )
    st.caption(
        f"Phiên: {record.get('analysis_id') or record.get('game_id')} · "
        f"Model: {record['model_version']}"
    )
    st.caption(
        f"Thời điểm suy luận: {record['inferred_at']}" if record.get("inferred_at")
        else "Chưa ghi nhận thời điểm suy luận gốc."
    )
    st.caption(f"Nhập/lưu lúc: {record.get('created_at')}")
    st.caption("Chỉ xem lại kết quả đã lưu; không tiếp tục POST từ bản xem lại này.")
    blue = document["prediction"]["p_blue_win"]
    for side, probability, column in zip(("blue", "red"), (blue, 1 - blue), st.columns(2), strict=True):
        identity = target[f"{side}_team_id"]
        column.metric(
            f"{names.get(identity, identity)} thắng — {record['evaluation_type']} đã lưu",
            f"{probability:.1%}",
        )
    features = document["features"]
    if record["evaluation_type"] == "PRE":
        for side, column in zip(("blue", "red"), st.columns(2), strict=True):
            team = features[side]
            with column:
                st.markdown(f"**Lịch sử của {names.get(team['team_id'], team['team_id'])}**")
                for label, field, continuity in (
                    ("Phong độ trong kho lịch sử", "recent_form", False),
                    ("Thắng khi chơi bên hiện tại", "side_win_rate", False),
                    ("Liên tục lực lượng", "roster_continuity", True),
                ):
                    st.markdown(f"**{label}**")
                    st.write(_pre_stat_text(SimpleNamespace(**team[field]), continuity=continuity))
        st.write("Đối đầu: " + _pre_stat_text(SimpleNamespace(**features["h2h_blue"])))
    else:
        player_names = provenance.get("player_names", {})
        champion_names = provenance.get("champion_names", {})
        pairs = {(item["side"], item["role"]): item for item in features["player_champion"]}
        st.caption("Tỷ lệ lịch sử của cặp tuyển thủ–tướng; không phải xác suất thắng ván.")
        for role in ROLES:
            st.markdown(f"**{ROLE_NAMES[role]}**")
            for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
                item = pairs[(side, role)]
                with column:
                    st.write(
                        f"{side} · {player_names.get(item['player_id'], item['player_id'])} · "
                        f"{champion_names.get(item['champion_id'], item['champion_id'])}"
                    )
                    st.write(
                        f"{item['games_count']} ván · {item['wins_count']} thắng · {_rate(item['win_rate'])}"
                        if item["games_count"] else f"0 ván · {MISSING_HISTORY}"
                    )
        comparison = document.get("comparison")
        if isinstance(comparison, dict) and all(key in comparison for key in ("pre", "post", "comparison")):
            _show_comparison(comparison, SimpleNamespace(**target), names)
    for warning in record["warnings"]:
        st.warning(f"{warning['message']} [{warning['warning_code']}]")
    if "history_coverage" in provenance:
        _show_coverage(provenance["history_coverage"], names)
    with st.expander("Dữ liệu và truy vết của bản lưu"):
        st.json(record)


def _saved_evaluations_view():
    with st.expander("Xem đánh giá đã lưu"):
        identity = st.text_input("Mã đánh giá đã lưu", key="saved_evaluation_lookup", max_chars=20)
        if identity != st.session_state.get("saved_evaluation_lookup_previous"):
            st.session_state.pop("saved_evaluation_document", None)
        st.session_state["saved_evaluation_lookup_previous"] = identity
        if st.button("Đọc bản lưu", key="read_saved_evaluation", disabled=not identity.strip()):
            st.session_state["saved_evaluation_document"] = _attempt(
                lambda: _read_saved_record(identity),
            )
        record = st.session_state.get("saved_evaluation_document")
        if record is not None:
            _attempt(lambda: _show_saved_evaluation(record))


def _read_saved_record(value):
    value = value.strip()
    if not value.isascii() or not value.isdecimal() or not 0 < int(value) < 2**63:
        raise PreFeatureInputError("E_STORAGE_READ", "Expected a positive stored evaluation ID")
    return persistence.default_store().read(int(value))


def _show_warnings(warnings, names):
    for warning in warnings:
        values = warning if isinstance(warning, dict) else asdict(warning)
        code = values["code"]
        team = names.get(values["team_id"], values["team_id"])
        if code == "W_TEAM_HISTORY_SMALL":
            text = (
                f"{team}: {values['sample_count']}/{values['requested_count']} ván "
                "trong cửa sổ lịch sử yêu cầu."
            )
        elif code == "W_H2H_MISSING":
            text = f"Đối đầu: {MISSING_HISTORY.lower()} (0 ván)."
        elif code == "W_PAIR_HISTORY_MISSING":
            text = (
                f"{values['sample_count']}/10 cặp tuyển thủ–tướng: "
                f"{MISSING_HISTORY.lower()}."
            )
        else:
            text = (
                f"{team or 'Hai đội'}: số mẫu {values['sample_count']}; "
                f"phạm vi yêu cầu {values['requested_count']}."
            )
        st.warning(f"{text} [{code}]")


def _entity_label(identity, entities, prompt):
    return prompt if identity is None else f"{entities[identity].name} [{identity}]"


def _coverage_labels(entities, counts):
    labels = {
        identity: (
            f"{entity.name} · {counts.get(identity, 0)} ván liên kết"
            if counts.get(identity, 0)
            else f"{entity.name} · chưa có lịch sử liên kết"
        )
        for identity, entity in entities.items()
    }
    repeated = Counter(labels.values())
    for identity, label in labels.items():
        if repeated[label] > 1:
            code = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8]
            labels[identity] = f"{label} · mã {code}"
    # A short presentation code is never an identity or a reason to merge options.
    collisions = Counter(labels.values())
    return {
        identity: f"{label} [{identity}]" if collisions[label] > 1 else label
        for identity, label in labels.items()
    }


def _catalog_choice(label, key, options, entities, labels, prompt):
    selected = st.session_state.get(key)
    choices = tuple(options)
    retained = selected is not None and selected not in choices
    if retained:
        if selected not in entities:
            raise PreFeatureInputError("E_PRE_INPUT_INCOMPLETE", "Unknown catalog selection")
        choices += (selected,)
    # Keep the exact widget value across an options/filter change.
    st.session_state[key] = selected
    identity = st.selectbox(
        label, (None,) + choices, key=key,
        format_func=lambda value: prompt if value is None else labels[value],
    )
    if retained:
        st.caption(
            "Bản ghi đang chọn nằm ngoài nhóm lọc; vẫn được giữ nguyên. "
            "Chọn bản ghi khác hoặc chuyển sang Toàn bộ danh mục nếu muốn thay đổi."
        )
    return identity


def _context_widgets(runtime, teams, players, coverage):
    """Collect stable catalog choices; service remains the authoritative validator."""
    scopes = {"linked": "Có lịch sử đã liên kết", "all": "Toàn bộ danh mục"}
    team_scope = st.selectbox(
        "Phạm vi chọn đội", tuple(scopes), key="team_catalog_scope", format_func=scopes.get,
    )
    team_options = demo.selection_options(
        runtime, coverage, entity_kind="team", linked_only=team_scope == "linked",
    )
    team_labels = _coverage_labels(teams, coverage.team_counts)
    st.caption(
        "Coverage: số ván lịch sử liên kết với bản ghi này trong phạm vi đang dùng; "
        "không phải số mẫu của mọi chỉ số hay điểm tin cậy. "
        f"Lịch sử kết thúc trước {coverage.history_cutoff_at.isoformat()}."
    )
    selected = {}
    for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
        with column:
            selected[side] = _catalog_choice(
                f"Đội {side}", f"{side.lower()}_team", team_options, teams, team_labels, "Chọn đội",
            )

    if st.session_state.get("roster_coverage") is not coverage.roster_suggestions:
        st.session_state["roster_suggestions"] = {}
        st.session_state["roster_coverage"] = coverage.roster_suggestions
    suggestions = st.session_state.setdefault("roster_suggestions", {})
    prior_teams = st.session_state.get("roster_teams", {})
    for side, team_id in selected.items():
        if team_id is not None and team_id not in suggestions:
            suggestions[team_id] = _attempt(
                lambda: demo.suggest_analysis_roster(runtime, team_id, coverage=coverage),
            )
        if prior_teams.get(side) != team_id:
            suggestion = suggestions.get(team_id)
            slots = {} if suggestion is None else {
                slot.role: slot.player_id for slot in suggestion.roster
            }
            for role in ROLES:
                st.session_state[f"roster_{side}_{role}"] = slots.get(role)
    st.session_state["roster_teams"] = selected

    for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
        with column:
            team = teams.get(selected[side])
            _fixed_label(f"{side} · {team.name if team else 'Chưa chọn đội'}", team=True)
            _show_media(team, "teams")
            if team is not None and not coverage.team_counts.get(team.identity, 0):
                st.caption(
                    "Chưa ghi nhận lịch sử liên kết cho bản ghi này "
                    "trong phạm vi dữ liệu đang dùng."
                )
            suggestion = suggestions.get(selected[side])
            if suggestion is None:
                st.caption("Chưa có roster gợi ý. Hãy chọn tuyển thủ từ danh mục.")
            else:
                st.caption(
                    "Roster gợi ý từ lịch sử · "
                    f"{suggestion.source_ended_at.isoformat()} · nguồn {suggestion.source_game_id}"
                )
            st.caption("Gợi ý không xác nhận lực lượng hiện tại; bạn có thể sửa từng vị trí.")

    player_scope = st.selectbox(
        "Phạm vi chọn tuyển thủ", tuple(scopes), key="player_catalog_scope", format_func=scopes.get,
    )
    player_options = demo.selection_options(
        runtime, coverage, entity_kind="player", linked_only=player_scope == "linked",
    )
    player_labels = _coverage_labels(players, coverage.player_counts)
    rosters = {"BLUE": [], "RED": []}
    for role in ROLES:
        for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
            with column:
                player_id = _catalog_choice(
                    f"{ROLE_NAMES[role]} · {side}", f"roster_{side}_{role}",
                    player_options, players, player_labels, "Chọn tuyển thủ",
                )
                player = players.get(player_id)
                _fixed_label(player.name if player else "Chưa chọn tuyển thủ")
                _show_media(player, "players")
                if player_id is not None:
                    rosters[side].append(PlayerSlot(role, player_id))
                    if not coverage.player_counts.get(player_id, 0):
                        st.caption(
                            "Chưa ghi nhận lịch sử liên kết cho bản ghi này "
                            "trong phạm vi dữ liệu đang dùng."
                        )

    with st.expander("Định danh đội và tuyển thủ đã chọn"):
        st.caption("Mã phân biệt chỉ phục vụ hiển thị; bản ghi giữ nguyên stable ID.")
        st.json({
            "history_pool_sha256": coverage.history_pool_sha256,
            "coverage_cutoff": coverage.history_cutoff_at.isoformat(),
            "teams": selected,
            "players": {
                side: {slot.role: slot.player_id for slot in slots}
                for side, slots in rosters.items()
            },
        })

    patch = st.text_input("Patch", key="analysis_patch", max_chars=20)
    label = st.text_input("Giải / bối cảnh ván (tùy chọn)", key="analysis_context")
    previous = st.session_state["analysis_input"]
    selection = replace(
        previous, blue_team_id=selected["BLUE"], red_team_id=selected["RED"],
        blue_roster=tuple(rosters["BLUE"]), red_roster=tuple(rosters["RED"]),
        patch=patch, context_label=label or None,
    )
    # These are form constraints only. No synthetic cutoff or core preflight is created here.
    ids = [slot.player_id for roster in rosters.values() for slot in roster]
    complete = bool(
        selected["BLUE"] and selected["RED"] and selected["BLUE"] != selected["RED"]
        and len(ids) == 10 and len(set(ids)) == 10 and patch.strip() and patch == patch.strip()
        and (not label or label.strip())
    )
    if selected["BLUE"] is not None and selected["BLUE"] == selected["RED"]:
        st.warning("Hãy chọn hai đội khác nhau.")
    if len(ids) != len(set(ids)):
        st.warning("Mỗi tuyển thủ chỉ được chọn cho một vị trí trong ván.")
    return selection, complete


def _show_coverage(summary, names):
    used = summary["pre_used"]
    latest = used["latest_ended_at"] or MISSING_HISTORY
    age = used["age_days_at_pre"]
    st.info(
        f"Lịch sử thực dùng cho PRE: {used['game_count']} ván · mới nhất: {latest}"
        + ("" if age is None else f" · cách thời điểm tạo PRE {age} ngày.")
    )
    for team in summary["teams"]:
        team_used = team["pre_used"]
        st.caption(
            f"{names[team['team_id']]}: {team_used['game_count']} ván thực dùng; "
            f"mới nhất {team_used['latest_ended_at'] or MISSING_HISTORY}; "
            f"{team['before_cutoff']['game_count']} ván trước cutoff."
        )
    st.caption(
        f"Kho có time proof: {summary['proof_pool']['game_count']} ván · "
        f"trước cutoff: {summary['before_cutoff']['game_count']} ván · "
        f"thực dùng cho PRE: {used['game_count']} ván."
    )
    st.caption(
        "Patch có trong lịch sử trước cutoff: "
        f"{'Có' if summary['patch_observed_in_history'] else 'Chưa ghi nhận'}. "
        f"Phạm vi hỗ trợ patch: {summary['patch_scope_status']}. "
        "Lịch sử được hiển thị theo ngày nguồn, không xác nhận phong độ hiện tại."
    )


def _show_pre(snapshot, names, summary):
    pre = snapshot.features
    probabilities = demo.phase_probabilities(snapshot)
    st.caption("Xác suất ước lượng của hệ thống")
    for column, team, probability in zip(
        st.columns(2), (pre.blue, pre.red), probabilities, strict=True,
    ):
        with column:
            st.metric(f"{names[team.team_id]} thắng — PRE", f"{probability:.1%}")
            st.markdown(f"**Lịch sử của {names[team.team_id]}**")
            for label, stat, continuity in (
                ("Phong độ trong kho lịch sử", team.recent_form, False),
                ("Thắng khi chơi bên hiện tại", team.side_win_rate, False),
                ("Liên tục lực lượng", team.roster_continuity, True),
            ):
                st.markdown(f"**{label}**")
                st.write(_pre_stat_text(stat, continuity=continuity))
    st.write(
        f"Đối đầu — {names[pre.blue.team_id]}: {_rate(pre.h2h_blue.value)}; "
        f"{pre.h2h_blue.sample_count} ván"
        + ("." if pre.h2h_blue.win_count is None else f" · {pre.h2h_blue.win_count} thắng.")
    )
    _show_warnings(snapshot.warnings, names)
    _show_coverage(summary, names)
    context = snapshot.request.runtime_context
    with st.expander("Thông tin truy vết PRE"):
        st.caption(f"Phiên: {context.analysis_id}")
        st.caption(f"Tạo PRE: {context.pre_created_at.isoformat()}")
        st.caption(f"Cutoff đã khóa: {context.history_cutoff_at.isoformat()}")
        st.caption(f"Lịch sử sẵn sàng trong runtime: {context.runtime_ready_at.isoformat()}")
        st.caption(f"Chính sách inference: {context.policy_version}")
        st.caption(f"Nguồn context: {context.context_source}")
        st.caption(f"Xác minh trước cấm/chọn: {context.pre_draft_verification}")


def _champion_widgets(pre, players, champions):
    target = pre.request.target
    by_side = {
        "BLUE": {slot.role: slot.player_id for slot in target.blue_roster},
        "RED": {slot.role: slot.player_id for slot in target.red_roster},
    }
    choices = {}
    for role in ROLES:
        for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
            with column:
                player = players[by_side[side][role]]
                _fixed_label(f"{side} · {ROLE_NAMES[role]} · {player.name}")
                champion_id = st.selectbox(
                    f"Tướng · {side} · {ROLE_NAMES[role]}", (None,) + tuple(champions),
                    format_func=lambda value: _entity_label(value, champions, "Chọn tướng"),
                    key=_widget_key(side, role),
                )
                choices[(side, role)] = champion_id
                if champion_id is not None:
                    _show_media(champions[champion_id], "champions")
                else:
                    _empty_media_frame("champions")
    return tuple(choices[(side, role)] for side in ("BLUE", "RED") for role in ROLES)


def _show_post(snapshot, names, players, champions):
    target = snapshot.pre.request.target
    st.caption("Xác suất ước lượng của hệ thống")
    for column, team_id, probability in zip(
        st.columns(2), (target.blue_team_id, target.red_team_id),
        demo.phase_probabilities(snapshot), strict=True,
    ):
        column.metric(f"{names[team_id]} thắng — POST", f"{probability:.1%}")
    _show_warnings(
        tuple(item for item in snapshot.warnings if item.code == "W_PAIR_HISTORY_MISSING"), names,
    )
    st.caption("Tỷ lệ lịch sử của cặp tuyển thủ–tướng; không phải xác suất thắng ván đang phân tích.")
    features = {(item.side, item.role): item for item in snapshot.features.player_champion}
    for role in ROLES:
        st.markdown(f"**{ROLE_NAMES[role]}**")
        for side, column in zip(("BLUE", "RED"), st.columns(2), strict=True):
            item = features[(side, role)]
            with column:
                _fixed_label(f"{side} · {names[item.team_id]} · {players[item.player_id].name}")
                _fixed_label(champions[item.champion_id].name)
                _show_media(champions[item.champion_id], "champions")
                st.write(
                    f"{item.games_count} ván · {item.wins_count} thắng · {_rate(item.win_rate)}"
                    if item.games_count else f"0 ván · {MISSING_HISTORY}"
                )
    with st.expander("Chi tiết lịch sử tuyển thủ–tướng"):
        st.dataframe([
            {
                "Đội": names[item.team_id], "Bên": item.side, "Vị trí": ROLE_NAMES[item.role],
                "Tuyển thủ": players[item.player_id].name, "Tướng": champions[item.champion_id].name,
                "Số ván": item.games_count, "Số thắng": item.wins_count,
                "Tỷ lệ thắng lịch sử": _rate(item.win_rate),
                "Thiếu mẫu": "Có" if item.missing else "Không",
            }
            for item in snapshot.features.player_champion
        ], hide_index=True)


def _show_comparison(comparison, target, names):
    st.subheader("So sánh PRE và POST")
    st.dataframe([
        {
            "Đội": names[team_id], "Bên": side.upper(),
            "PRE": f"{comparison['pre'][f'{side}_win_probability']:.1%}",
            "POST": f"{comparison['post'][f'{side}_win_probability']:.1%}",
            "Thay đổi (%)": (
                f"{evaluation.percentage_points(comparison['comparison'][f'{side}_probability_delta']):+.1f}"
                " %"
            ),
        }
        for side, team_id in (("blue", target.blue_team_id), ("red", target.red_team_id))
    ], hide_index=True)
    st.caption("Xác suất ước lượng của mô hình thay đổi từ PRE sang POST.")
    _show_warnings(comparison["warnings"], names)
    with st.expander("Thông tin truy vết so sánh"):
        trace = {
            key: comparison[key] for key in (
                "analysis_id", "inference_mode", "inference_policy", "model",
            ) if key in comparison
        }
        if "runtime_context" in comparison:
            trace["runtime_context"] = {
                key: value for key, value in comparison["runtime_context"].items()
                if key != "history_pool_game_ids"
            }
        else:
            trace.update({key: comparison[key] for key in (
                "game_id", "history_cutoff_at", "data_kind", "evaluation_protocol",
            ) if key in comparison})
        st.json(trace)


def _show_method(runtime):
    with st.expander("Phương pháp và nguồn mô hình"):
        st.write("Model được huấn luyện và đánh giá bằng benchmark hồi cứu.")
        st.write(
            "Context phiên này: USER_PROVIDED. Chưa xác minh thời điểm cung cấp context "
            "trước cấm/chọn (NOT_VERIFIED)."
        )
        st.caption(f"Họ mô hình: {runtime.bundle.family}")
        st.json({
            "training_contract": asdict(runtime.bundle.contract),
            "training_identity": asdict(runtime.bundle.identity),
        })


def run():
    st.set_page_config(page_title=SYSTEM_NAME, layout="wide")
    st.title("Phân tích ván đấu")
    st.info(INTERACTIVE_NOTICE)
    st.markdown(
        "<style>"
        ".mi-team-label,.mi-person-label {display:-webkit-box;-webkit-box-orient:vertical;"
        "-webkit-line-clamp:2;overflow:hidden;overflow-wrap:anywhere;line-height:1.5;}"
        ".mi-team-label {height:3em;font-size:1.25rem;font-weight:600;}"
        ".mi-person-label {height:3em;font-weight:500;}"
        "</style>", unsafe_allow_html=True,
    )
    st.session_state.setdefault("evaluation_state", EvaluationState())
    _saved_evaluations_view()
    if not _storage_ready():
        return
    runtime = st.session_state.get("demo_runtime")
    if runtime is not None and (
        not isinstance(runtime, demo.AnalysisRuntime)
        or not isinstance(st.session_state.get("analysis_input"), demo.AnalysisInput)
    ):
        # Streamlit can retain an old retrospective session when this page is upgraded.
        # Require fresh explicit initialization, never relabel its old snapshots.
        for key in (
            "demo_runtime", "analysis_input", "roster_suggestions", "roster_teams",
            "target_game", "blue_team", "red_team", "analysis_patch", "analysis_context",
            "selection_coverage", "roster_coverage", "team_catalog_scope", "player_catalog_scope",
        ):
            st.session_state.pop(key, None)
        for side in ("BLUE", "RED"):
            for role in ROLES:
                st.session_state.pop(f"roster_{side}_{role}", None)
        _clear_context_result()
        runtime = None
        st.info("Phiên giao diện cũ không tương thích. Hãy nạp dữ liệu và mô hình cho phiên mới.")
    if (
        runtime is not None and st.session_state["evaluation_state"].pre is not None
        and st.session_state.get("stored_pre_id") is None
    ):
        _clear_context_result()
        st.info("PRE của phiên trước chưa được lưu bền. Hãy xác nhận context và tạo PRE mới để lưu.")
    if st.button("Nạp dữ liệu và mô hình", key="initialize_demo", disabled=runtime is not None):
        runtime = _attempt(demo.load_analysis_runtime)
        if runtime is not None:
            st.session_state["demo_runtime"] = runtime
            st.session_state["analysis_input"] = demo.AnalysisInput("", "", (), (), "", False)
            st.session_state["roster_suggestions"] = {}
            st.session_state["roster_teams"] = {}
            st.session_state.pop("selection_coverage", None)
            st.session_state.pop("roster_coverage", None)
            _clear_context_result()
    if runtime is None:
        st.caption("Nạp dữ liệu và mô hình trước khi chọn hai đội và xác nhận context.")
        st.button("Tạo PRE", key="create_pre", disabled=True)
        st.button("Tạo POST", key="create_post", disabled=True)
        return

    teams = {item.identity: item for item in runtime.catalog.teams}
    names = {identity: item.name for identity, item in teams.items()}
    players = {item.identity: item for item in runtime.catalog.players}
    _show_method(runtime)
    coverage = _attempt(lambda: demo.selection_coverage(
        runtime, pre=st.session_state["evaluation_state"].pre,
        previous=st.session_state.get("selection_coverage"),
    ))
    if coverage is None:
        _clear_context_result()
        return
    st.session_state["selection_coverage"] = coverage
    selection, complete = _context_widgets(runtime, teams, players, coverage)
    previous = st.session_state["analysis_input"]
    # Confirmation is a UI action; all edited context must be reviewed again.
    if replace(selection, user_confirmed=False) != replace(previous, user_confirmed=False):
        _clear_context_result()
    if not _storage_ready():
        return
    selection = replace(selection, user_confirmed=st.session_state.get("confirm_context") is True)
    old_state = st.session_state["evaluation_state"]
    current_choices = tuple(
        st.session_state.get(_widget_key(side, role)) for side in ("BLUE", "RED") for role in ROLES
    )
    if old_state.pre is not None and current_choices != old_state.champion_ids:
        _clear_stored_post()
    if not _storage_ready():
        return
    state = _attempt(lambda: demo.refresh_analysis_state(
        old_state, selection, runtime, champion_ids=current_choices,
    ))
    if state is None or (old_state.pre is not None and state.pre is None):
        _clear_context_result()
        state = EvaluationState()
    if not _storage_ready():
        return
    confirmed = st.checkbox(
        "Tôi xác nhận hai đội, side, patch và tuyển thủ/vị trí do tôi cung cấp.",
        key="confirm_context", disabled=not complete,
    )
    selection = replace(selection, user_confirmed=confirmed is True and complete)
    st.session_state["analysis_input"] = selection
    st.session_state["evaluation_state"] = state
    st.subheader("PRE — trước đội hình tướng cuối cùng")
    if not complete:
        st.caption("Chọn hai đội khác nhau, đủ mười tuyển thủ khác nhau và nhập patch.")
    if st.button(
        "Tạo PRE", key="create_pre", disabled=not selection.user_confirmed or state.pre is not None,
    ):
        snapshot = _attempt(lambda: _persist_pre(runtime, selection))
        if snapshot is not None:
            st.session_state["selection_coverage"] = demo.selection_coverage(
                runtime, pre=snapshot, previous=coverage,
            )
            _reset_champion_choices()
            state = EvaluationState(context_key=snapshot.context_key, pre=snapshot)
            st.session_state["evaluation_state"] = state
    if state.pre is None:
        st.caption("Xác nhận context và tạo PRE trước khi nhập đội hình tướng cuối cùng.")
        st.button("Tạo POST", key="create_post", disabled=True)
        return

    summary = _attempt(lambda: demo.analysis_history_summary(runtime, state.pre))
    if summary is None:
        _clear_context_result()
        return
    st.caption(f"Mã PRE đã lưu: {st.session_state['stored_pre_id']}")
    _show_pre(state.pre, names, summary)
    st.subheader("Đội hình cuối cùng — năm vị trí mỗi đội")
    champions = {
        item.identity: item for item in runtime.catalog.champions
        if item.identity in runtime.champion_reference
    }
    choices = _champion_widgets(state.pre, players, champions)
    if choices != state.champion_ids:
        _clear_stored_post()
        state = replace(state, champion_ids=choices, post=None)
    if not _storage_ready():
        st.session_state["evaluation_state"] = replace(state, post=None)
        return
    lineup = None
    try:
        lineup = demo.validate_analysis_lineup(runtime, state.pre, choices)
    except PreFeatureInputError as error:
        st.caption(_message(error))
    except Exception:
        st.caption("Chưa thể xác nhận đội hình. Hãy kiểm tra lại thông tin đã chọn.")
    if lineup is None:
        _clear_stored_post()
        state = replace(state, post=None)
    if not _storage_ready():
        st.session_state["evaluation_state"] = replace(state, post=None)
        return
    st.session_state["evaluation_state"] = state
    if st.button("Tạo POST", key="create_post", disabled=lineup is None or state.post is not None):
        snapshot = _attempt(lambda: _persist_post(runtime, state.pre, choices))
        state = replace(state, post=snapshot)
        st.session_state["evaluation_state"] = state
    if state.post is not None:
        comparison = _attempt(lambda: demo.compare_analysis_evaluations(runtime, state.pre, state.post))
        if comparison is None:
            _clear_stored_post()
            st.session_state["evaluation_state"] = replace(state, post=None)
            return
        st.subheader("POST — đội hình đã xác nhận")
        st.caption(f"Mã POST đã lưu: {st.session_state['stored_post_id']}")
        _show_post(state.post, names, players, champions)
        _show_comparison(comparison, state.pre.request.target, names)
