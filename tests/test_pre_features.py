"""Synthetic in-memory tests; không phải evidence temporal của dữ liệu thật."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

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


def roster(team, members=None):
    if members is None:
        members = tuple(f"{team.lower()}{i}" for i in range(1, 6))
    return tuple(
        PlayerSlot(role, player_id)
        for role, player_id in zip(ROLES, members, strict=True)
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


def game(game_id, day, blue, red, winner, blue_members=None, red_members=None):
    return HistoricalGame(
        game_id=game_id,
        blue_team_id=blue,
        red_team_id=red,
        blue_roster=roster(blue, blue_members),
        red_roster=roster(red, red_members),
        winner_team_id=winner,
        ended_at=datetime(2025, 7, day, 12, tzinfo=UTC),
    )


def example_history():
    return [
        game("g1", 6, "A", "X", "A"),
        game("g2", 7, "B", "A", "B"),
        game("g3", 8, "A", "B", "A"),
        game(
            "g4",
            9,
            "X",
            "A",
            "X",
            red_members=("a1", "a2", "a3", "a4", "a6"),
        ),
    ]


def assert_rate(stat, value, wins, ids):
    assert stat.value == pytest.approx(value)
    assert stat.win_count == wins
    assert stat.sample_count == len(ids)
    assert stat.missing is False
    assert stat.game_ids == ids


def assert_missing(stat):
    assert stat.value is None
    assert stat.sample_count == 0
    assert stat.missing is True
    assert stat.game_ids == ()


def test_hand_calculated_default_features():
    result = build_pre_features(target(), example_history())

    assert result.accepted_game_ids == ("g4", "g3", "g2", "g1")
    assert result.counts.input_games == 4
    assert result.counts.accepted_games == 4
    assert result.counts.excluded_games == 0
    assert result.counts.missing_end_games == 0

    assert_rate(result.blue.recent_form, 1 / 2, 2, ("g4", "g3", "g2", "g1"))
    assert_rate(result.red.recent_form, 1 / 2, 1, ("g3", "g2"))
    assert_rate(result.blue.side_win_rate, 1.0, 2, ("g3", "g1"))
    assert_rate(result.red.side_win_rate, 0.0, 0, ("g3",))
    assert_rate(result.h2h_blue, 1 / 2, 1, ("g3", "g2"))

    assert result.blue.roster_continuity.value == pytest.approx(4 / 5)
    assert result.blue.roster_continuity.sample_count == 1
    assert result.blue.roster_continuity.game_ids == ("g4",)
    assert result.blue.roster_continuity.missing is False

    assert result.red.roster_continuity.value == 1.0
    assert result.red.roster_continuity.sample_count == 1
    assert result.red.roster_continuity.game_ids == ("g3",)
    assert result.red.roster_continuity.missing is False

    assert result.metadata.game_id == "target"
    assert result.metadata.patch == "15.01"
    assert result.metadata.history_cutoff_at == CUTOFF
    assert result.metadata.cutoff_verification == "NOT_ASSESSED"


def test_filter_each_feature_group_before_applying_window():
    config = FeatureConfig(
        recent_form_games=3,
        side_win_rate_games=3,
        head_to_head_games=3,
    )
    result = build_pre_features(target(), example_history(), config)

    assert_rate(result.blue.recent_form, 1 / 3, 1, ("g4", "g3", "g2"))
    # g1 nằm ngoài ba game toàn cục gần nhất nhưng thuộc mẫu BLUE của A.
    assert_rate(result.blue.side_win_rate, 1.0, 2, ("g3", "g1"))
    assert_rate(result.h2h_blue, 1 / 2, 1, ("g3", "g2"))

    default = build_pre_features(target(), example_history())
    assert result.metadata.config_version != default.metadata.config_version


def test_default_windows_select_the_correct_latest_games():
    history = [
        replace(
            game(
                f"g{i:02d}",
                9,
                "A",
                "B",
                "A" if i <= 10 else "B",
            ),
            ended_at=CUTOFF - timedelta(minutes=i),
        )
        for i in range(1, 25)
    ]
    result = build_pre_features(target(), list(reversed(history)))

    latest_ten = tuple(f"g{i:02d}" for i in range(1, 11))
    latest_twenty = tuple(f"g{i:02d}" for i in range(1, 21))

    assert_rate(result.blue.recent_form, 1.0, 10, latest_ten)
    assert_rate(result.blue.side_win_rate, 1 / 2, 10, latest_twenty)
    assert_rate(result.red.side_win_rate, 1 / 2, 10, latest_twenty)
    assert_rate(result.h2h_blue, 1.0, 10, latest_ten)
    assert result.blue.roster_continuity.game_ids == ("g01",)


@pytest.mark.parametrize(
    ("seconds", "included"),
    [(-1, True), (0, False), (1, False)],
)
def test_end_must_be_strictly_before_cutoff(seconds, included):
    candidate = replace(
        game("candidate", 9, "A", "B", "B"),
        ended_at=CUTOFF + timedelta(seconds=seconds),
    )
    result = build_pre_features(
        target(),
        [game("base", 6, "A", "X", "A"), candidate],
    )

    assert ("candidate" in result.accepted_game_ids) is included
    if not included:
        assert result.exclusions[0].game_id == "candidate"
        assert result.exclusions[0].reason == "END_NOT_BEFORE_CUTOFF"


def test_missing_end_is_excluded_and_counted_separately():
    history = [
        game("valid", 6, "A", "X", "A"),
        replace(game("missing", 7, "A", "B", "A"), ended_at=None),
        game("future", 11, "A", "B", "B"),
    ]
    result = build_pre_features(target(), history)

    assert result.accepted_game_ids == ("valid",)
    assert result.counts.input_games == 3
    assert result.counts.accepted_games == 1
    assert result.counts.excluded_games == 2
    assert result.counts.missing_end_games == 1
    assert {
        item.game_id: item.reason for item in result.exclusions
    } == {
        "future": "END_NOT_BEFORE_CUTOFF",
        "missing": "MISSING_ENDED_AT",
    }


@pytest.mark.parametrize(
    "invalid_end",
    [
        "2025-07-09T12:00:00+00:00",
        1752062400000,
        datetime(2025, 7, 9, 12),
    ],
)
def test_non_null_invalid_end_is_an_error_not_missing(invalid_end):
    invalid = replace(
        game("invalid", 9, "A", "B", "A"),
        ended_at=invalid_end,
    )

    with pytest.raises(PreFeatureInputError) as caught:
        build_pre_features(target(), [invalid])

    assert caught.value.code == "E_TIMESTAMP_INVALID"
    assert caught.value.game_id == "invalid"
    assert "ended_at" in str(caught.value)


@pytest.mark.parametrize(
    ("cutoff", "code"),
    [
        (None, "E_CUTOFF_MISSING"),
        (datetime(2025, 7, 10, 12), "E_TIMESTAMP_INVALID"),
        ("2025-07-10T12:00:00+00:00", "E_TIMESTAMP_INVALID"),
    ],
)
def test_missing_naive_or_string_cutoff_is_rejected(cutoff, code):
    with pytest.raises(PreFeatureInputError) as caught:
        build_pre_features(
            replace(target(), history_cutoff_at=cutoff),
            example_history(),
        )

    assert caught.value.code == code


def test_equivalent_timezone_instants_produce_equal_results():
    offset = timezone(timedelta(hours=7))
    profile = target()
    history = example_history()
    history.append(game("equal", 10, "A", "B", "B"))

    baseline = build_pre_features(profile, history)
    shifted = build_pre_features(
        replace(profile, history_cutoff_at=CUTOFF.astimezone(offset)),
        [
            replace(item, ended_at=item.ended_at.astimezone(offset))
            for item in history
        ],
    )

    assert shifted == baseline
    assert shifted.metadata.history_cutoff_at == CUTOFF
    assert "equal" not in shifted.accepted_game_ids


@pytest.mark.parametrize("winner", ["A", "B", "OUTSIDE"])
def test_target_record_outcome_never_changes_features(winner):
    baseline = build_pre_features(target(), example_history())
    target_record = game("target", 5, "A", "B", winner)

    result = build_pre_features(
        target(),
        [target_record] + example_history(),
    )

    assert result.blue == baseline.blue
    assert result.red == baseline.red
    assert result.h2h_blue == baseline.h2h_blue
    assert result.accepted_game_ids == baseline.accepted_game_ids
    assert result.exclusions[0].game_id == "target"
    assert result.exclusions[0].reason == "TARGET_GAME"


def test_reversing_input_does_not_change_result_or_mutate_inputs():
    profile = target()
    history = example_history()
    before = deepcopy((profile, history))

    forward = build_pre_features(profile, history)
    backward = build_pre_features(profile, list(reversed(history)))

    assert forward == backward
    assert (profile, history) == before


def test_equal_end_uses_lexical_game_id_for_window_and_continuity():
    history = [
        game(
            "zeta",
            8,
            "A",
            "B",
            "B",
            blue_members=("a1", "a2", "a3", "a4", "a6"),
        ),
        game("alpha", 8, "A", "B", "A"),
    ]
    config = FeatureConfig(1, 1, 1)

    result = build_pre_features(target(), history, config)
    reversed_result = build_pre_features(
        target(),
        list(reversed(history)),
        config,
    )

    assert result == reversed_result
    assert result.accepted_game_ids == ("alpha", "zeta")
    assert_rate(result.blue.recent_form, 1.0, 1, ("alpha",))
    assert_rate(result.h2h_blue, 1.0, 1, ("alpha",))
    assert result.blue.roster_continuity.value == 1.0
    assert result.blue.roster_continuity.game_ids == ("alpha",)


def test_h2h_uses_target_blue_identity_across_historical_sides():
    history = [
        game("old", 6, "A", "B", "B"),
        game("new", 7, "B", "A", "A"),
        game("unrelated", 8, "A", "X", "A"),
    ]

    result = build_pre_features(target(), history)

    assert_rate(result.h2h_blue, 1 / 2, 1, ("new", "old"))


def test_continuity_uses_latest_game_not_an_average():
    history = [
        game("old", 6, "A", "B", "A"),
        game(
            "new",
            7,
            "A",
            "B",
            "B",
            blue_members=("a1", "a2", "a3", "a4", "a6"),
        ),
    ]

    result = build_pre_features(target(), history)

    assert result.blue.roster_continuity.value == pytest.approx(4 / 5)
    assert result.blue.roster_continuity.sample_count == 1
    assert result.blue.roster_continuity.game_ids == ("new",)


def test_role_changes_do_not_change_player_identity_continuity():
    profile = target()
    slots = profile.blue_roster
    swapped = (
        replace(slots[0], player_id=slots[1].player_id),
        replace(slots[1], player_id=slots[0].player_id),
    ) + slots[2:]

    result = build_pre_features(
        replace(profile, blue_roster=swapped),
        [game("previous", 8, "A", "B", "A")],
    )

    assert result.blue.roster_continuity.value == 1.0
    assert result.blue.roster_continuity.game_ids == ("previous",)


@pytest.mark.parametrize("case", ["empty", "unrelated", "future"])
def test_no_usable_team_history_returns_missing_features(case):
    history = {
        "empty": [],
        "unrelated": [game("unrelated", 6, "X", "Y", "X")],
        "future": [game("future", 11, "A", "B", "B")],
    }[case]

    result = build_pre_features(target(), history)

    for team in (result.blue, result.red):
        assert_missing(team.recent_form)
        assert_missing(team.side_win_rate)
        assert_missing(team.roster_continuity)
    assert_missing(result.h2h_blue)


def test_side_sample_can_be_missing_while_recent_form_exists():
    result = build_pre_features(
        target(),
        [game("a_red_only", 6, "X", "A", "A")],
    )

    assert_rate(result.blue.recent_form, 1.0, 1, ("a_red_only",))
    assert_missing(result.blue.side_win_rate)
    assert_missing(result.h2h_blue)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_ids_are_rejected_without_first_row_wins(conflicting):
    first = game("duplicate", 6, "A", "B", "A")
    second = replace(first, winner_team_id="B") if conflicting else first

    for history in ([first, second], [second, first]):
        with pytest.raises(PreFeatureInputError) as caught:
            build_pre_features(target(), history)
        assert caught.value.code == "E_HISTORY_DUPLICATE_GAME_ID"


def malformed(record, case):
    if case == "same_team":
        return replace(record, red_team_id=record.blue_team_id)
    if case == "missing_role":
        return replace(record, blue_roster=record.blue_roster[:-1])
    if case == "duplicate_role":
        slots = record.blue_roster
        return replace(
            record,
            blue_roster=(replace(slots[0], role=slots[1].role),) + slots[1:],
        )
    if case == "duplicate_player":
        slots = record.blue_roster
        return replace(
            record,
            blue_roster=(
                replace(slots[0], player_id=slots[1].player_id),
            ) + slots[1:],
        )
    if case == "cross_team_player":
        slots = record.red_roster
        return replace(
            record,
            red_roster=(
                replace(slots[0], player_id=record.blue_roster[0].player_id),
            ) + slots[1:],
        )
    raise AssertionError(f"Unknown synthetic case: {case}")


@pytest.mark.parametrize(
    ("case", "code"),
    [
        ("same_team", "E_TEAMS_INVALID"),
        ("missing_role", "E_LINEUP_INVALID"),
        ("duplicate_role", "E_LINEUP_INVALID"),
        ("duplicate_player", "E_LINEUP_INVALID"),
        ("cross_team_player", "E_LINEUP_INVALID"),
    ],
)
@pytest.mark.parametrize("location", ["target", "history"])
def test_invalid_team_or_lineup_is_rejected(case, code, location):
    profile = target()
    history = [game("history", 8, "A", "B", "A")]

    if location == "target":
        profile = malformed(profile, case)
    else:
        history = [malformed(history[0], case)]

    with pytest.raises(PreFeatureInputError) as caught:
        build_pre_features(profile, history)

    assert caught.value.code == code


def test_non_target_winner_must_belong_to_participating_teams():
    with pytest.raises(PreFeatureInputError) as caught:
        build_pre_features(
            target(),
            [game("invalid_winner", 8, "A", "B", "X")],
        )

    assert caught.value.code == "E_HISTORY_WINNER_INVALID"
    assert caught.value.game_id == "invalid_winner"


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "10"])
@pytest.mark.parametrize(
    "field",
    ["recent_form_games", "side_win_rate_games", "head_to_head_games"],
)
def test_invalid_window_values_are_rejected(field, value):
    config = replace(FeatureConfig(), **{field: value})

    with pytest.raises(PreFeatureInputError) as caught:
        build_pre_features(target(), example_history(), config)

    assert caught.value.code == "E_FEATURE_CONFIG_INVALID"


def test_future_results_and_rosters_cannot_change_current_features():
    history = example_history()
    baseline = build_pre_features(target(), history)
    additions = [
        game(
            "equal",
            10,
            "A",
            "B",
            "B",
            blue_members=("a6", "a7", "a8", "a9", "a10"),
        ),
        game(
            "future",
            11,
            "A",
            "B",
            "B",
            blue_members=("a6", "a7", "a8", "a9", "a10"),
        ),
    ]

    result = build_pre_features(target(), additions + history)

    assert result.blue == baseline.blue
    assert result.red == baseline.red
    assert result.h2h_blue == baseline.h2h_blue
    assert result.accepted_game_ids == baseline.accepted_game_ids
    assert result.counts.excluded_games == 2
