"""Synthetic in-memory fixtures; không phải evidence thời gian của dữ liệu thật."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from match_insight.features.post import (
    ChampionSlot,
    FinalLineup,
    HistoricalChampionGame,
    PostFeatureInputError,
    build_post_features,
)
from match_insight.features.pre import (
    ROLES,
    FeatureConfig,
    HistoricalGame,
    PlayerSlot,
    PreFeatureInputError,
    TargetGame,
    build_pre_features,
)

CUTOFF = datetime(2025, 7, 10, 12, tzinfo=UTC)
CHAMPIONS = frozenset(str(number) for number in range(1, 21))

# Explicit synthetic mapping, not a production identity/name heuristic.
SYNTHETIC_CHAMPIONS = {
    f"{prefix}{number}": str(offset + number)
    for prefix, offset in (("a", 0), ("b", 5), ("x", 15))
    for number in range(1, 6)
}


def roster(team):
    return tuple(
        PlayerSlot(role, f"{team.lower()}{number}")
        for number, role in enumerate(ROLES, start=1)
    )


def target():
    return TargetGame(
        game_id="target",
        blue_team_id="A",
        red_team_id="B",
        blue_roster=roster("A"),
        red_roster=roster("B"),
        patch="15.01",
        history_cutoff_at=CUTOFF,
    )


def historical_game(game_id, day, blue, red, winner):
    return HistoricalGame(
        game_id=game_id,
        blue_team_id=blue,
        red_team_id=red,
        blue_roster=roster(blue),
        red_roster=roster(red),
        winner_team_id=winner,
        ended_at=datetime(2025, 7, day, 12, tzinfo=UTC),
    )


def champion_slots(game, overrides=None):
    assignments = SYNTHETIC_CHAMPIONS | (overrides or {})
    return tuple(
        ChampionSlot(
            team_id=team_id,
            side=side,
            role=player.role,
            player_id=player.player_id,
            champion_id=assignments[player.player_id],
        )
        for side, team_id, players in (
            ("BLUE", game.blue_team_id, game.blue_roster),
            ("RED", game.red_team_id, game.red_roster),
        )
        for player in players
    )


def synthetic_history():
    g1 = historical_game("g1", 7, "A", "B", "A")
    g2 = historical_game("g2", 8, "B", "A", "B")
    g3 = historical_game("g3", 9, "A", "X", "A")
    return (
        HistoricalChampionGame(g1, champion_slots(g1)),
        HistoricalChampionGame(g2, champion_slots(g2)),
        HistoricalChampionGame(g3, champion_slots(g3, {"a1": "11"})),
    )


def create_pre(history, config=FeatureConfig()):
    return build_pre_features(
        target(),
        (record.game for record in history),
        config,
    )


def final_lineup():
    context = target()
    return FinalLineup(
        game_id=context.game_id,
        patch=context.patch,
        slots=champion_slots(context),
    )


def position(result, side="BLUE", role="TOP"):
    return next(
        item
        for item in result.player_champion
        if item.side == side and item.role == role
    )


@pytest.fixture
def case():
    history = synthetic_history()
    return create_pre(history), final_lineup(), history


def test_pre_post_integration_matches_hand_calculation(case):
    pre, final, history = case
    before = deepcopy((pre, final, history))

    post = build_post_features(pre, final, history, CHAMPIONS)

    assert pre.blue.recent_form.value == pytest.approx(2 / 3)
    assert pre.red.recent_form.value == pytest.approx(1 / 2)
    assert post.pre is pre
    assert (pre, final, history) == before
    assert post.metadata.history_cutoff_at == CUTOFF
    assert post.metadata.history_cutoff_at.tzinfo is UTC
    assert post.metadata.pre_config_version == pre.metadata.config_version
    assert post.metadata.cutoff_verification == "NOT_ASSESSED"
    assert post.metadata.history_scope == "ALL_PRE_ACCEPTED_GAMES"
    assert post.accepted_game_ids == ("g3", "g2", "g1")

    blue_top = position(post)
    assert blue_top.player_id == "a1"
    assert blue_top.champion_id == "1"
    assert blue_top.games_count == 2
    assert blue_top.wins_count == 1
    assert blue_top.win_rate == pytest.approx(1 / 2)
    assert blue_top.missing is False
    assert blue_top.game_ids == ("g2", "g1")

    blue_jungle = position(post, role="JUNGLE")
    assert blue_jungle.games_count == 3
    assert blue_jungle.wins_count == 2
    assert blue_jungle.win_rate == pytest.approx(2 / 3)

    red_top = position(post, side="RED")
    assert red_top.player_id == "b1"
    assert red_top.champion_id == "6"
    assert red_top.games_count == 2
    assert red_top.wins_count == 1
    assert red_top.win_rate == pytest.approx(1 / 2)
    assert red_top.game_ids == ("g2", "g1")


def test_pair_history_uses_all_pre_accepted_games_not_pre_window():
    history = synthetic_history()
    pre = create_pre(history, FeatureConfig(1, 1, 1))

    post = build_post_features(pre, final_lineup(), history, CHAMPIONS)

    assert pre.blue.recent_form.sample_count == 1
    assert pre.blue.recent_form.game_ids == ("g3",)
    assert position(post).games_count == 2
    assert position(post).game_ids == ("g2", "g1")


def test_changing_champion_changes_only_corresponding_post_feature(case):
    pre, final, history = case
    initial = build_post_features(pre, final, history, CHAMPIONS)
    changed_final = replace(
        final,
        slots=(replace(final.slots[0], champion_id="11"),) + final.slots[1:],
    )

    changed = build_post_features(pre, changed_final, history, CHAMPIONS)

    assert changed.pre is initial.pre is pre
    assert changed.metadata == initial.metadata
    assert changed.player_champion[1:] == initial.player_champion[1:]
    assert position(changed).champion_id == "11"
    assert position(changed).games_count == 1
    assert position(changed).wins_count == 1
    assert position(changed).win_rate == 1.0
    assert position(changed).game_ids == ("g3",)


def test_no_pair_samples_do_not_become_zero_or_half(case):
    pre, final, history = case
    final = replace(
        final,
        slots=(replace(final.slots[0], champion_id="15"),) + final.slots[1:],
    )

    post = build_post_features(pre, final, history, CHAMPIONS)
    feature = position(post)

    assert feature.games_count == 0
    assert feature.wins_count == 0
    assert feature.win_rate is None
    assert feature.missing is True
    assert feature.game_ids == ()


def test_empty_history_returns_missing_for_all_ten_positions():
    pre = create_pre(())

    post = build_post_features(pre, final_lineup(), (), CHAMPIONS)

    assert post.pre is pre
    assert len(post.player_champion) == 10
    assert post.counts.accepted_games == 0
    for feature in post.player_champion:
        assert feature.games_count == 0
        assert feature.wins_count == 0
        assert feature.win_rate is None
        assert feature.missing is True
        assert feature.game_ids == ()


def test_zero_win_rate_with_samples_is_not_missing():
    game = historical_game("loss", 7, "A", "B", "B")
    history = (HistoricalChampionGame(game, champion_slots(game)),)
    pre = create_pre(history)

    post = build_post_features(pre, final_lineup(), history, CHAMPIONS)

    assert position(post).games_count == 1
    assert position(post).wins_count == 0
    assert position(post).win_rate == 0.0
    assert position(post).missing is False


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("champion_id", None, "E_CHAMPION_ID_INVALID"),
        ("champion_id", "999", "E_CHAMPION_UNKNOWN"),
        ("champion_id", 1, "E_CHAMPION_ID_INVALID"),
        ("team_id", "OTHER", "E_EVAL_INCOMPATIBLE"),
        ("player_id", "other1", "E_EVAL_INCOMPATIBLE"),
        ("side", "GREEN", "E_POST_LINEUP_INVALID"),
        ("role", "ADC", "E_POST_LINEUP_INVALID"),
    ],
)
def test_invalid_final_slot_is_rejected(case, field, value, code):
    pre, final, history = case
    final = replace(
        final,
        slots=(replace(final.slots[0], **{field: value}),) + final.slots[1:],
    )

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(pre, final, history, CHAMPIONS)

    assert caught.value.code == code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("champion_id", "1"),
        ("player_id", "a1"),
        ("role", "TOP"),
    ],
)
def test_duplicate_champion_player_or_side_role_is_rejected(case, field, value):
    pre, final, history = case
    slots = list(final.slots)
    slots[1] = replace(slots[1], **{field: value})

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            pre,
            replace(final, slots=tuple(slots)),
            history,
            CHAMPIONS,
        )

    assert caught.value.code == "E_SOURCE_CONFLICT"


def test_missing_lineup_entry_is_rejected(case):
    pre, final, history = case

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            pre,
            replace(final, slots=final.slots[:-1]),
            history,
            CHAMPIONS,
        )

    assert caught.value.code == "E_LINEUP_INCOMPLETE"


def test_role_swap_in_final_lineup_requires_new_pre(case):
    pre, final, history = case
    slots = list(final.slots)
    slots[0] = replace(slots[0], role="JUNGLE")
    slots[1] = replace(slots[1], role="TOP")

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            pre,
            replace(final, slots=tuple(slots)),
            history,
            CHAMPIONS,
        )

    assert caught.value.code == "E_EVAL_INCOMPATIBLE"


@pytest.mark.parametrize(
    ("field", "value"),
    [("game_id", "another-game"), ("patch", "15.02")],
)
def test_final_context_must_match_pre(case, field, value):
    pre, final, history = case

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            pre,
            replace(final, **{field: value}),
            history,
            CHAMPIONS,
        )

    assert caught.value.code == "E_EVAL_INCOMPATIBLE"


def test_pre_without_bound_snapshot_is_rejected(case):
    pre, final, history = case

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            replace(pre, history_snapshot=None),
            final,
            history,
            CHAMPIONS,
        )

    assert caught.value.code == "E_PRE_SNAPSHOT_REQUIRED"


def test_changed_pre_metadata_cutoff_is_incompatible(case):
    pre, final, history = case
    changed = replace(
        pre,
        metadata=replace(
            pre.metadata,
            history_cutoff_at=CUTOFF + timedelta(hours=1),
        ),
    )

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(changed, final, history, CHAMPIONS)

    assert caught.value.code == "E_EVAL_INCOMPATIBLE"


@pytest.mark.parametrize(
    "reference",
    [frozenset(), frozenset({""}), frozenset({1}), ["1", "2"]],
)
def test_reference_contract_is_validated(case, reference):
    pre, final, history = case

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(pre, final, history, reference)

    assert caught.value.code == "E_CHAMPION_REFERENCE_INVALID"


@pytest.mark.parametrize(
    ("seconds", "accepted"),
    [(-1, True), (0, False), (1, False)],
)
def test_shared_temporal_filter_has_strict_boundary(seconds, accepted):
    candidate = replace(
        historical_game("boundary", 7, "A", "B", "A"),
        ended_at=CUTOFF + timedelta(seconds=seconds),
    )
    history = (
        HistoricalChampionGame(candidate, champion_slots(candidate)),
    )
    pre = create_pre(history)

    post = build_post_features(pre, final_lineup(), history, CHAMPIONS)

    assert post.counts.accepted_games == int(accepted)
    assert position(post).games_count == int(accepted)
    if accepted:
        assert position(post).game_ids == ("boundary",)
    else:
        assert position(post).win_rate is None
        assert post.exclusions[0].reason == "END_NOT_BEFORE_CUTOFF"


def test_new_future_or_equal_games_do_not_change_post_features(case):
    pre, final, history = case
    baseline = build_post_features(pre, final, history, CHAMPIONS)
    extras = tuple(
        HistoricalChampionGame(
            replace(
                history[0].game,
                game_id=game_id,
                winner_team_id="B",
                ended_at=end,
            ),
            (),
        )
        for game_id, end in (
            ("equal", CUTOFF),
            ("future", CUTOFF + timedelta(hours=1)),
        )
    )

    post = build_post_features(pre, final, history + extras, CHAMPIONS)

    assert post.pre is pre
    assert post.player_champion == baseline.player_champion
    assert post.metadata == baseline.metadata
    assert post.accepted_game_ids == baseline.accepted_game_ids
    assert post.counts.excluded_games == 2
    assert {item.reason for item in post.exclusions} == {
        "END_NOT_BEFORE_CUTOFF"
    }


@pytest.mark.parametrize("winner", ["A", "B", "OUTSIDE"])
def test_target_record_is_always_excluded_before_outcome_or_champions(case, winner):
    pre, final, history = case
    baseline = build_post_features(pre, final, history, CHAMPIONS)
    target_record = HistoricalChampionGame(
        replace(
            history[0].game,
            game_id=pre.metadata.game_id,
            winner_team_id=winner,
        ),
        (),
    )

    post = build_post_features(
        pre,
        final,
        history + (target_record,),
        CHAMPIONS,
    )

    assert post.player_champion == baseline.player_champion
    assert post.pre is pre
    assert post.exclusions[0].reason == "TARGET_GAME"


def test_missing_end_is_excluded_and_counted(case):
    pre, final, history = case
    baseline = build_post_features(pre, final, history, CHAMPIONS)
    missing = HistoricalChampionGame(
        replace(history[0].game, game_id="missing", ended_at=None),
        (),
    )

    post = build_post_features(pre, final, history + (missing,), CHAMPIONS)

    assert post.player_champion == baseline.player_champion
    assert post.counts.missing_end_games == 1
    assert post.exclusions[0].game_id == "missing"
    assert post.exclusions[0].reason == "MISSING_ENDED_AT"


@pytest.mark.parametrize(
    "invalid_end",
    ["2025-07-09T12:00:00Z", 1752062400, datetime(2025, 7, 9, 12)],
)
def test_invalid_non_null_end_preserves_pre_error_semantics(case, invalid_end):
    pre, final, history = case
    invalid = HistoricalChampionGame(
        replace(history[0].game, game_id="invalid", ended_at=invalid_end),
        (),
    )

    with pytest.raises(PreFeatureInputError) as caught:
        build_post_features(pre, final, history + (invalid,), CHAMPIONS)

    assert caught.value.code == "E_TIMESTAMP_INVALID"


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_history_is_never_resolved_by_first_row(case, conflicting):
    pre, final, history = case
    duplicate = history[0]
    if conflicting:
        duplicate = replace(
            duplicate,
            game=replace(duplicate.game, winner_team_id="B"),
        )

    for records in (history + (duplicate,), (duplicate,) + history):
        with pytest.raises(PreFeatureInputError) as caught:
            build_post_features(pre, final, records, CHAMPIONS)
        assert caught.value.code == "E_HISTORY_DUPLICATE_GAME_ID"


@pytest.mark.parametrize("change", ["added", "removed", "changed"])
def test_eligible_history_cannot_drift_after_pre(case, change):
    pre, final, history = case
    if change == "added":
        extra = replace(
            history[0],
            game=replace(history[0].game, game_id="new-eligible"),
        )
        records = history + (extra,)
    elif change == "removed":
        records = history[1:]
    else:
        records = (
            replace(
                history[0],
                game=replace(history[0].game, winner_team_id="B"),
            ),
        ) + history[1:]

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(pre, final, records, CHAMPIONS)

    assert caught.value.code == "E_PRE_HISTORY_MISMATCH"


def test_accepted_history_requires_complete_champion_mapping(case):
    pre, final, history = case
    incomplete = replace(history[0], slots=history[0].slots[:-1])

    with pytest.raises(PostFeatureInputError) as caught:
        build_post_features(
            pre,
            final,
            (incomplete,) + history[1:],
            CHAMPIONS,
        )

    assert caught.value.code == "E_LINEUP_INCOMPLETE"
    assert caught.value.game_id == "g1"


def test_pair_identity_survives_team_and_role_changes():
    original = historical_game("transfer", 7, "A", "X", "A")
    changed_roles = tuple(
        replace(slot, role=ROLES[(index + 1) % len(ROLES)])
        for index, slot in enumerate(original.blue_roster)
    )
    historical = replace(
        original,
        blue_team_id="Y",
        blue_roster=changed_roles,
        winner_team_id="Y",
    )
    history = (
        HistoricalChampionGame(historical, champion_slots(historical)),
    )
    pre = create_pre(history)

    post = build_post_features(pre, final_lineup(), history, CHAMPIONS)

    assert pre.blue.recent_form.missing is True
    assert position(post).player_id == "a1"
    assert position(post).games_count == 1
    assert position(post).wins_count == 1
    assert position(post).win_rate == 1.0
    assert position(post).game_ids == ("transfer",)


def test_reversing_history_and_lineup_order_does_not_change_result(case):
    pre, final, history = case
    before = deepcopy((pre, final, history))
    baseline = build_post_features(pre, final, history, CHAMPIONS)
    reversed_history = tuple(
        replace(record, slots=tuple(reversed(record.slots)))
        for record in reversed(history)
    )

    post = build_post_features(
        pre,
        replace(final, slots=tuple(reversed(final.slots))),
        reversed_history,
        CHAMPIONS,
    )

    assert post == baseline
    assert (pre, final, history) == before


def test_equal_end_timestamps_keep_lexical_game_id_order():
    first = historical_game("zeta", 7, "A", "B", "A")
    second = replace(first, game_id="alpha", winner_team_id="B")
    history = (
        HistoricalChampionGame(first, champion_slots(first)),
        HistoricalChampionGame(second, champion_slots(second)),
    )
    pre = create_pre(history)

    forward = build_post_features(pre, final_lineup(), history, CHAMPIONS)
    backward = build_post_features(
        pre,
        final_lineup(),
        tuple(reversed(history)),
        CHAMPIONS,
    )

    assert forward == backward
    assert position(forward).game_ids == ("alpha", "zeta")
    assert position(forward).win_rate == pytest.approx(1 / 2)
