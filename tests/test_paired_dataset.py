"""Synthetic fixtures only; no production temporal evidence, files or DB."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from match_insight.features.post import (
    ChampionSlot,
    FinalLineup,
    HistoricalChampionGame,
    build_post_features,
)
from match_insight.features.pre import (
    ROLES,
    HistoricalGame,
    PlayerSlot,
    TargetGame,
    build_pre_features,
)
from match_insight.ml.dataset import (
    CATEGORICAL_COLUMNS,
    POST_COLUMNS,
    PRE_COLUMNS,
    RETROSPECTIVE_DATASET_VERSION,
    RETROSPECTIVE_PROTOCOL,
    STRICT_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetInputError,
    DatasetTarget,
    build_paired_dataset,
    temporal_split,
)

BASE = datetime(2025, 7, 1, 12, tzinfo=UTC)
POLICY = "synthetic-7i-v1"
POLICIES = frozenset({POLICY})
CHAMPIONS = frozenset(str(value) for value in range(1, 11))


def roster(team):
    return tuple(
        PlayerSlot(role, f"{team.lower()}{number}")
        for number, role in enumerate(ROLES, start=1)
    )


def slots(blue="A", red="B"):
    # Explicit fixture mapping: A uses IDs 1..5; B uses IDs 6..10.
    return tuple(
        ChampionSlot(
            team_id=team,
            side=side,
            role=role,
            player_id=f"{team.lower()}{number}",
            champion_id=str(number + (0 if team == "A" else 5)),
        )
        for side, team in (("BLUE", blue), ("RED", red))
        for number, role in enumerate(ROLES, start=1)
    )


def history_fixture():
    records = []
    for game_id, days, blue, red, winner in (
        ("h1", 2, "A", "B", "A"),
        ("h2", 1, "B", "A", "B"),
    ):
        game = HistoricalGame(
            game_id=game_id,
            blue_team_id=blue,
            red_team_id=red,
            blue_roster=roster(blue),
            red_roster=roster(red),
            winner_team_id=winner,
            ended_at=BASE - timedelta(days=days),
        )
        records.append(HistoricalChampionGame(game, slots(blue, red)))
    return tuple(records)


def input_fixture(count=20, offsets=None):
    if offsets is None:
        offsets = tuple(range(count))

    targets = []
    cutoffs = []
    for number, offset in enumerate(offsets, start=1):
        game_id = f"g{number:02d}"
        cutoff = BASE + timedelta(days=offset)
        targets.append(
            DatasetTarget(
                game_id=game_id,
                blue_team_id="A",
                red_team_id="B",
                blue_roster=roster("A"),
                red_roster=roster("B"),
                patch="15.01",
                final_lineup=FinalLineup(game_id, "15.01", slots()),
                winner_team_id="A" if number % 2 else "B",
                ended_at=cutoff + timedelta(hours=1),
            )
        )
        cutoffs.append(
            CutoffRecord(
                game_id=game_id,
                history_cutoff_at=cutoff,
                evidence_ref=f"synthetic://7i/{game_id}",
                policy_version=POLICY,
                # Synthetic assertion only; no production source is certified.
                verification=CutoffVerification.VERIFIED_EXTERNALLY,
            )
        )

    return tuple(targets), tuple(cutoffs), history_fixture()


def build(targets, cutoffs, history):
    return build_paired_dataset(
        targets=targets,
        history=history,
        cutoffs=cutoffs,
        champion_reference=CHAMPIONS,
        approved_policy_versions=POLICIES,
    )


def assert_same_dataset(left, right):
    assert left.status == right.status
    assert left.reason == right.reason
    assert left.excluded == right.excluded
    assert left.unused_cutoff_game_ids == right.unused_cutoff_game_ids
    assert left.categorical_columns == right.categorical_columns
    assert_frame_equal(left.X_pre, right.X_pre)
    assert_frame_equal(left.X_post, right.X_post)
    assert_series_equal(left.y, right.y)
    assert_frame_equal(left.metadata, right.metadata)


def test_integration_uses_real_pre_post_and_hand_calculated_features():
    targets, cutoffs, history = input_fixture(2)
    dataset = build(targets, cutoffs, history)

    context = TargetGame(
        game_id="g01",
        blue_team_id="A",
        red_team_id="B",
        blue_roster=roster("A"),
        red_roster=roster("B"),
        patch="15.01",
        history_cutoff_at=cutoffs[0].history_cutoff_at,
    )
    pre = build_pre_features(context, tuple(record.game for record in history))
    post = build_post_features(pre, targets[0].final_lineup, history, CHAMPIONS)

    assert dataset.status == "OK"
    assert list(dataset.X_pre.index) == ["g01", "g02"]
    assert dataset.X_pre.index.equals(dataset.X_post.index)
    assert dataset.X_pre.index.equals(dataset.y.index)
    assert dataset.X_pre.index.equals(dataset.metadata.index)
    assert dataset.y.tolist() == [1, 0]

    row = dataset.X_pre.loc["g01"]
    assert row["blue_recent_form_value"] == pre.blue.recent_form.value == 0.5
    assert row["blue_recent_form_sample_count"] == 2
    assert row["blue_recent_form_win_count"] == 1
    assert row["blue_side_win_rate_value"] == 1.0
    assert row["red_side_win_rate_value"] == 0.0
    assert not row["red_side_win_rate_missing"]
    assert row["h2h_blue_value"] == 0.5
    assert row["blue_roster_continuity_value"] == 1.0

    pair = post.player_champion[0]
    post_row = dataset.X_post.loc["g01"]
    assert post_row["blue_top_champion_id"] == pair.champion_id == "1"
    assert post_row["blue_top_games_count"] == pair.games_count == 2
    assert post_row["blue_top_wins_count"] == pair.wins_count == 1
    assert post_row["blue_top_win_rate"] == pair.win_rate == 0.5

    assert_frame_equal(dataset.X_post.loc[:, list(PRE_COLUMNS)], dataset.X_pre)
    metadata = dataset.metadata.loc["g01"]
    assert metadata["history_cutoff_at"] == pre.metadata.history_cutoff_at
    assert metadata["history_cutoff_at"] == post.metadata.history_cutoff_at
    assert metadata["history_cutoff_at"].tzinfo is UTC
    assert metadata["cutoff_verification"] == "VERIFIED_EXTERNALLY"
    assert metadata["core_cutoff_verification"] == "NOT_ASSESSED"
    assert metadata["history_game_ids"] == ("h2", "h1")


def test_missing_cutoff_excludes_the_complete_pair():
    targets, cutoffs, history = input_fixture(2)
    dataset = build(targets, cutoffs[1:], history)

    assert list(dataset.X_pre.index) == ["g02"]
    assert list(dataset.X_post.index) == ["g02"]
    assert dataset.y.tolist() == [0]
    assert len(dataset.excluded) == 1
    assert dataset.excluded[0].game_id == "g01"
    assert dataset.excluded[0].reason == "E_CUTOFF_MISSING"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("history_cutoff_at", None, "E_CUTOFF_MISSING"),
        ("history_cutoff_at", datetime(2025, 7, 1), "E_CUTOFF_INVALID"),
        ("history_cutoff_at", "2025-07-01T12:00:00Z", "E_CUTOFF_INVALID"),
        ("evidence_ref", "", "E_CUTOFF_EVIDENCE_INVALID"),
        ("policy_version", "", "E_CUTOFF_POLICY_INVALID"),
        ("policy_version", "not-approved", "E_CUTOFF_POLICY_UNAPPROVED"),
        ("verification", CutoffVerification.UNVERIFIED, "E_CUTOFF_UNVERIFIED"),
        ("verification", "VERIFIED_EXTERNALLY", "E_CUTOFF_STATUS_INVALID"),
    ],
)
def test_invalid_cutoff_contract_never_creates_a_sample(field, value, reason):
    targets, cutoffs, history = input_fixture(1)
    changed = replace(cutoffs[0], **{field: value})

    dataset = build(targets, (changed,), history)

    assert dataset.status == "EMPTY"
    assert dataset.X_pre.empty
    assert dataset.X_post.empty
    assert dataset.y.empty
    assert dataset.excluded[0].reason == reason


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_cutoffs_are_not_first_row_wins(conflicting):
    targets, cutoffs, history = input_fixture(2)
    duplicate = cutoffs[0]
    if conflicting:
        duplicate = replace(
            duplicate,
            history_cutoff_at=BASE + timedelta(minutes=10),
        )

    first = build(targets, cutoffs + (duplicate,), history)
    second = build(targets, (duplicate,) + tuple(reversed(cutoffs)), history)

    assert_same_dataset(first, second)
    assert list(first.X_pre.index) == ["g02"]
    assert first.excluded[0].reason == "E_CUTOFF_CONFLICT"


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_targets_are_excluded_once_without_overlap(conflicting):
    targets, cutoffs, history = input_fixture(2)
    duplicate = targets[0]
    if conflicting:
        duplicate = replace(duplicate, winner_team_id="B")

    dataset = build(targets + (duplicate,), cutoffs, history)
    reversed_dataset = build((duplicate,) + tuple(reversed(targets)), cutoffs, history)

    assert_same_dataset(dataset, reversed_dataset)
    assert list(dataset.X_pre.index) == ["g02"]
    assert len(dataset.excluded) == 1
    assert dataset.excluded[0].reason == "E_TARGET_DUPLICATE_GAME_ID"
    assert not set(dataset.X_pre.index) & {
        item.game_id for item in dataset.excluded
    }


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("pre_roster", "E_LINEUP_INVALID"),
        ("post_lineup", "E_LINEUP_INCOMPLETE"),
        ("winner", "E_TARGET_WINNER_INVALID"),
    ],
)
def test_invalid_target_never_leaves_a_partial_pair(case, reason):
    targets, cutoffs, history = input_fixture(2)
    bad = targets[0]
    if case == "pre_roster":
        bad = replace(bad, blue_roster=bad.blue_roster[:-1])
    elif case == "post_lineup":
        bad = replace(
            bad,
            final_lineup=replace(
                bad.final_lineup,
                slots=bad.final_lineup.slots[:-1],
            ),
        )
    else:
        bad = replace(bad, winner_team_id="OUTSIDE")

    dataset = build((bad, targets[1]), cutoffs, history)

    assert list(dataset.X_pre.index) == ["g02"]
    assert list(dataset.X_post.index) == ["g02"]
    assert list(dataset.y.index) == ["g02"]
    assert list(dataset.metadata.index) == ["g02"]
    assert dataset.excluded[0].reason == reason


@pytest.mark.parametrize(
    ("end", "reason"),
    [
        (None, "E_LABEL_END_MISSING"),
        (datetime(2025, 7, 2), "E_LABEL_END_INVALID"),
        (BASE, "E_LABEL_END_NOT_AFTER_CUTOFF"),
        (BASE - timedelta(seconds=1), "E_LABEL_END_NOT_AFTER_CUTOFF"),
    ],
)
def test_target_label_end_must_be_explicit_and_consistent(end, reason):
    targets, cutoffs, history = input_fixture(1)

    dataset = build((replace(targets[0], ended_at=end),), cutoffs, history)

    assert dataset.status == "EMPTY"
    assert dataset.excluded[0].reason == reason


def test_feature_schema_excludes_target_labels_and_metadata():
    dataset = build(*input_fixture(2))
    forbidden = {
        "game_id",
        "history_cutoff_at",
        "started_at",
        "ended_at",
        "target_ended_at",
        "winner_team_id",
        "result",
        "y_blue_win",
        "evidence_ref",
        "policy_version",
    }

    assert tuple(dataset.X_pre.columns) == PRE_COLUMNS
    assert tuple(dataset.X_post.columns) == POST_COLUMNS
    assert not forbidden & set(dataset.X_pre.columns)
    assert not forbidden & set(dataset.X_post.columns)
    assert dataset.categorical_columns == CATEGORICAL_COLUMNS
    for column in CATEGORICAL_COLUMNS:
        assert str(dataset.X_post[column].dtype) == "string"
        assert all(isinstance(value, str) for value in dataset.X_post[column])


def test_target_label_changes_do_not_change_features_or_boundaries():
    targets, cutoffs, history = input_fixture()
    baseline = build(targets, cutoffs, history)
    flipped = tuple(
        replace(
            target,
            winner_team_id=(
                target.red_team_id
                if target.winner_team_id == target.blue_team_id
                else target.blue_team_id
            ),
        )
        for target in targets
    )

    changed = build(flipped, cutoffs, history)

    assert_frame_equal(changed.X_pre, baseline.X_pre)
    assert_frame_equal(changed.X_post, baseline.X_post)
    assert changed.y.tolist() == [1 - value for value in baseline.y]
    assert temporal_split(changed) == temporal_split(baseline)


def test_target_record_in_history_cannot_leak_its_outcome():
    targets, cutoffs, history = input_fixture(1)
    baseline = build(targets, cutoffs, history)
    target_history = HistoricalChampionGame(
        replace(
            history[0].game,
            game_id="g01",
            winner_team_id="OUTSIDE",
        ),
        (),
    )

    changed = build(targets, cutoffs, history + (target_history,))

    assert_frame_equal(changed.X_pre, baseline.X_pre)
    assert_frame_equal(changed.X_post, baseline.X_post)
    exclusions = changed.metadata.loc["g01", "history_exclusions"]
    assert exclusions[0].game_id == "g01"
    assert exclusions[0].reason == "TARGET_GAME"


def test_empty_history_preserves_none_and_missing():
    targets, cutoffs, _history = input_fixture(1)

    dataset = build(targets, cutoffs, ())

    assert dataset.status == "OK"
    assert dataset.X_pre.loc["g01", "blue_recent_form_value"] is None
    assert dataset.X_pre.loc["g01", "blue_recent_form_sample_count"] == 0
    assert dataset.X_pre.loc["g01", "blue_recent_form_missing"]
    assert dataset.X_post.loc["g01", "blue_top_win_rate"] is None
    assert dataset.X_post.loc["g01", "blue_top_games_count"] == 0
    assert dataset.X_post.loc["g01", "blue_top_wins_count"] == 0
    assert dataset.X_post.loc["g01", "blue_top_missing"]


def test_input_order_and_timezone_representation_do_not_change_results():
    targets, cutoffs, history = input_fixture()
    baseline = build(targets, cutoffs, history)
    offset = timezone(timedelta(hours=7))
    shifted_cutoffs = tuple(
        replace(
            record,
            history_cutoff_at=record.history_cutoff_at.astimezone(offset),
        )
        for record in reversed(cutoffs)
    )
    shifted_targets = tuple(
        replace(target, ended_at=target.ended_at.astimezone(offset))
        for target in reversed(targets)
    )

    changed = build(shifted_targets, shifted_cutoffs, tuple(reversed(history)))

    assert_same_dataset(changed, baseline)
    assert temporal_split(changed) == temporal_split(baseline)


def test_twenty_games_split_exactly_fourteen_three_three():
    dataset = build(*input_fixture())

    split = temporal_split(dataset)

    assert split.status == "OK"
    assert split.train == tuple(f"g{number:02d}" for number in range(1, 15))
    assert split.validation == ("g15", "g16", "g17")
    assert split.test == ("g18", "g19", "g20")
    assert split.validation_boundary == BASE + timedelta(days=14)
    assert split.test_boundary == BASE + timedelta(days=17)
    assert [row.count for row in split.summary] == [14, 3, 3]
    assert [row.fraction_of_input for row in split.summary] == [0.70, 0.15, 0.15]
    assert not split.excluded

    memberships = (set(split.train), set(split.validation), set(split.test))
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    assert set.union(*memberships) == set(dataset.X_pre.index)

    for game_ids in (split.train, split.validation, split.test):
        ids = list(game_ids)
        assert dataset.X_pre.loc[ids].index.equals(dataset.X_post.loc[ids].index)
        assert dataset.X_pre.loc[ids].index.equals(dataset.y.loc[ids].index)
        assert_frame_equal(
            dataset.X_post.loc[ids, list(PRE_COLUMNS)],
            dataset.X_pre.loc[ids],
        )

    assert (
        split.summary[0].max_ended_at
        < split.summary[1].first_cutoff
        < split.summary[2].first_cutoff
    )
    assert split.summary[1].max_ended_at < split.summary[2].first_cutoff


def test_equal_cutoff_group_is_never_split_and_ratio_tie_chooses_earlier():
    inputs = input_fixture(offsets=(0, 1, 2, 3, 4, 5, 6, 6, 7, 8))
    dataset = build(*inputs)

    split = temporal_split(dataset)

    assert split.status == "OK"
    assert split.train == ("g01", "g02", "g03", "g04", "g05", "g06")
    assert split.validation == ("g07", "g08")
    assert split.test == ("g09", "g10")
    assert [row.count for row in split.summary] == [6, 2, 2]


@pytest.mark.parametrize("lateness", [timedelta(0), timedelta(hours=1)])
def test_late_train_and_validation_labels_are_purged_without_reassignment(lateness):
    targets, cutoffs, history = input_fixture()
    changed = list(targets)
    changed[13] = replace(
        changed[13],
        ended_at=cutoffs[14].history_cutoff_at + lateness,
    )
    changed[16] = replace(
        changed[16],
        ended_at=cutoffs[17].history_cutoff_at + lateness,
    )
    dataset = build(tuple(changed), cutoffs, history)

    split = temporal_split(dataset)

    assert split.status == "OK"
    assert [row.planned_count for row in split.summary] == [14, 3, 3]
    assert [row.count for row in split.summary] == [13, 2, 3]
    assert split.validation_boundary == cutoffs[14].history_cutoff_at
    assert split.test_boundary == cutoffs[17].history_cutoff_at
    assert split.validation == ("g15", "g16")
    assert split.test == ("g18", "g19", "g20")
    assert "g14" not in split.train + split.validation + split.test
    assert "g17" not in split.train + split.validation + split.test
    assert {item.game_id: item.reason for item in split.excluded} == {
        "g14": "TRAIN_LABEL_NOT_AVAILABLE_BEFORE_VALIDATION",
        "g17": "VALIDATION_LABEL_NOT_AVAILABLE_BEFORE_TEST",
    }
    retained = set(split.train + split.validation + split.test)
    excluded = {item.game_id for item in split.excluded}
    assert not retained & excluded
    assert retained | excluded == set(dataset.X_pre.index)


def test_boundary_stays_fixed_when_first_validation_sample_is_purged():
    targets, cutoffs, history = input_fixture()
    changed = list(targets)
    changed[14] = replace(
        changed[14],
        ended_at=cutoffs[17].history_cutoff_at,
    )

    split = temporal_split(build(tuple(changed), cutoffs, history))

    assert split.status == "OK"
    assert split.validation_boundary == cutoffs[14].history_cutoff_at
    assert split.validation == ("g16", "g17")
    assert split.summary[1].first_cutoff == cutoffs[15].history_cutoff_at


def test_purging_an_entire_partition_returns_insufficient_without_rebalancing():
    targets, cutoffs, history = input_fixture()
    changed = tuple(
        replace(target, ended_at=BASE + timedelta(days=100))
        if index < 14 else target
        for index, target in enumerate(targets)
    )

    split = temporal_split(build(changed, cutoffs, history))

    assert split.status == "INSUFFICIENT"
    assert split.reason == "EMPTY_PARTITION_AFTER_LABEL_PURGE"
    assert split.train == ()
    assert split.validation == ("g15", "g16", "g17")
    assert split.test == ("g18", "g19", "g20")
    assert len(split.excluded) == 14


@pytest.mark.parametrize("offsets", [(), (0,), (0, 1), (0, 0, 0, 0)])
def test_empty_or_insufficient_time_groups_have_explicit_status(offsets):
    dataset = build(*input_fixture(offsets=offsets))

    split = temporal_split(dataset)

    assert split.status == "INSUFFICIENT"
    assert split.reason == (
        "EMPTY_DATASET" if not offsets else "FEWER_THAN_THREE_CUTOFF_GROUPS"
    )
    assert not split.train
    assert not split.validation
    assert not split.test


def test_no_valid_targets_returns_empty_tables_without_fake_rows():
    targets, _cutoffs, history = input_fixture(2)

    dataset = build(targets, (), history)

    assert dataset.status == "EMPTY"
    assert dataset.reason == "NO_VALID_PAIRED_TARGETS"
    assert dataset.X_pre.empty
    assert dataset.X_post.empty
    assert dataset.metadata.empty
    assert dataset.y.empty
    assert len(dataset.excluded) == 2
    assert tuple(dataset.X_pre.columns) == PRE_COLUMNS
    assert tuple(dataset.X_post.columns) == POST_COLUMNS


def test_unused_cutoffs_are_reported_separately_from_target_exclusions():
    targets, cutoffs, history = input_fixture(2)
    unused = replace(cutoffs[0], game_id="not-a-target")

    dataset = build(targets, cutoffs + (unused,), history)

    assert dataset.status == "OK"
    assert not dataset.excluded
    assert dataset.unused_cutoff_game_ids == ("not-a-target",)


@pytest.mark.parametrize("corruption", ["pre_value", "label_column", "row_order"])
def test_split_rejects_broken_pair_alignment_or_feature_schema(corruption):
    dataset = build(*input_fixture())
    altered = dataset.X_post.copy(deep=True)
    if corruption == "pre_value":
        altered.loc["g01", "blue_recent_form_value"] = 0.123
    elif corruption == "label_column":
        altered["y_blue_win"] = dataset.y
    else:
        altered = altered.iloc[::-1].copy()

    with pytest.raises(DatasetInputError) as caught:
        temporal_split(replace(dataset, X_post=altered))

    assert caught.value.code == "E_PAIRED_DATASET_INVALID"


def test_build_and_split_do_not_mutate_inputs():
    inputs = input_fixture()
    original_inputs = deepcopy(inputs)

    dataset = build(*inputs)
    original_dataset = deepcopy(dataset)
    temporal_split(dataset)

    assert inputs == original_inputs
    assert_same_dataset(dataset, original_dataset)


def test_schema_contains_only_supplied_champion_strings():
    dataset = build(*input_fixture(1))

    for column in CATEGORICAL_COLUMNS:
        assert isinstance(dataset.X_post[column].dtype, pd.StringDtype)
        assert set(dataset.X_post[column]) <= CHAMPIONS


def target_history_fixture(target):
    """Represent one synthetic target as history for another synthetic target."""
    return HistoricalChampionGame(
        HistoricalGame(
            game_id=target.game_id,
            blue_team_id=target.blue_team_id,
            red_team_id=target.red_team_id,
            blue_roster=target.blue_roster,
            red_roster=target.red_roster,
            winner_team_id=target.winner_team_id,
            ended_at=target.ended_at,
        ),
        target.final_lineup.slots,
    )


def test_conflicting_history_end_excludes_only_pairs_that_consume_it():
    targets, cutoffs, history = input_fixture(offsets=(0, 1, 0))
    targets = (
        replace(targets[0], ended_at=BASE + timedelta(days=2)),
    ) + targets[1:]
    source = target_history_fixture(targets[0])
    source = replace(source, game=replace(source.game, ended_at=BASE + timedelta(hours=1)))
    inputs = (targets, cutoffs, history + (source,))
    before = deepcopy(inputs)

    dataset = build(*inputs)
    reversed_dataset = build(
        tuple(reversed(targets)),
        tuple(reversed(cutoffs)),
        tuple(reversed(inputs[2])),
    )

    assert inputs == before
    assert_same_dataset(dataset, reversed_dataset)
    assert list(dataset.X_pre.index) == ["g01", "g03"]
    assert dataset.X_pre.index.equals(dataset.X_post.index)
    assert dataset.X_pre.index.equals(dataset.y.index)
    assert dataset.X_pre.index.equals(dataset.metadata.index)
    assert len(dataset.excluded) == 1
    exclusion = dataset.excluded[0]
    assert exclusion.game_id == "g02"
    assert exclusion.reason == "E_SOURCE_CONFLICT"
    assert exclusion.source_game_id == "g01"
    assert "ended_at" in exclusion.detail
    baseline = build(targets, cutoffs, history)
    assert_frame_equal(dataset.X_pre, baseline.X_pre.loc[["g01", "g03"]])
    assert_frame_equal(dataset.X_post, baseline.X_post.loc[["g01", "g03"]])


@pytest.mark.parametrize(
    ("conflict", "detail_field"),
    [
        ("winner", "winner_team_id"),
        ("teams", "blue_team_id"),
        ("player", "blue_roster"),
        ("role", "blue_roster"),
        ("champion", "final_lineup"),
    ],
)
def test_overlapping_history_identity_conflict_rejects_complete_pair(conflict, detail_field):
    targets, cutoffs, history = input_fixture(2)
    source = target_history_fixture(targets[0])
    if conflict == "winner":
        source = replace(source, game=replace(source.game, winner_team_id="B"))
    elif conflict == "teams":
        source = replace(
            source,
            game=replace(
                source.game,
                blue_team_id="B",
                red_team_id="A",
                blue_roster=roster("B"),
                red_roster=roster("A"),
            ),
            slots=slots("B", "A"),
        )
    else:
        players = list(source.game.blue_roster)
        champions = list(source.slots)
        if conflict == "player":
            players[0] = replace(players[0], player_id="a9")
            champions[0] = replace(champions[0], player_id="a9")
        elif conflict == "role":
            players[0] = replace(players[0], role="JUNGLE")
            players[1] = replace(players[1], role="TOP")
            champions[0] = replace(champions[0], role="JUNGLE")
            champions[1] = replace(champions[1], role="TOP")
        else:
            champions[0] = replace(champions[0], champion_id="2")
            champions[1] = replace(champions[1], champion_id="1")
        source = replace(
            source,
            game=replace(source.game, blue_roster=tuple(players)),
            slots=tuple(champions),
        )

    dataset = build(targets, cutoffs, history + (source,))

    assert list(dataset.X_pre.index) == ["g01"]
    assert list(dataset.X_post.index) == ["g01"]
    assert list(dataset.y.index) == ["g01"]
    assert list(dataset.metadata.index) == ["g01"]
    assert len(dataset.excluded) == 1
    assert dataset.excluded[0].game_id == "g02"
    assert dataset.excluded[0].reason == "E_SOURCE_CONFLICT"
    assert dataset.excluded[0].source_game_id == "g01"
    assert detail_field in dataset.excluded[0].detail


def test_consistent_overlap_is_usable_with_equivalent_timezone_and_slot_order():
    targets, cutoffs, history = input_fixture(3)
    source = target_history_fixture(targets[0])
    baseline = build(targets, cutoffs, history + (source,))
    offset = timezone(timedelta(hours=7))
    equivalent = replace(
        source,
        game=replace(
            source.game,
            ended_at=source.game.ended_at.astimezone(offset),
            blue_roster=tuple(reversed(source.game.blue_roster)),
            red_roster=tuple(reversed(source.game.red_roster)),
        ),
        slots=tuple(reversed(source.slots)),
    )

    dataset = build(targets, cutoffs, history + (equivalent,))

    assert_same_dataset(dataset, baseline)
    assert dataset.status == "OK"
    assert not dataset.excluded
    assert dataset.X_pre.loc["g01", "blue_recent_form_value"] == 0.5
    assert dataset.X_pre.loc["g02", "blue_recent_form_value"] == pytest.approx(2 / 3)
    assert dataset.X_post.loc["g02", "blue_top_games_count"] == 3
    assert dataset.X_post.loc["g02", "blue_top_wins_count"] == 2
    assert dataset.X_post.loc["g02", "blue_top_win_rate"] == pytest.approx(2 / 3)
    assert dataset.metadata.loc["g02", "history_game_ids"] == ("g01", "h2", "h1")


@pytest.mark.parametrize(
    ("end", "reason"),
    [
        (None, "MISSING_ENDED_AT"),
        (BASE + timedelta(days=1), "END_NOT_BEFORE_CUTOFF"),
        (BASE + timedelta(days=2), "END_NOT_BEFORE_CUTOFF"),
    ],
)
def test_temporally_excluded_overlap_does_not_create_a_conflict(end, reason):
    targets, cutoffs, history = input_fixture(2)
    source = target_history_fixture(targets[0])
    source = replace(
        source,
        game=replace(source.game, ended_at=end, winner_team_id="B"),
        slots=(),
    )
    baseline = build(targets, cutoffs, history)

    dataset = build(targets, cutoffs, history + (source,))

    assert dataset.status == "OK"
    assert not dataset.excluded
    assert_frame_equal(dataset.X_pre, baseline.X_pre)
    assert_frame_equal(dataset.X_post, baseline.X_post)
    assert_series_equal(dataset.y, baseline.y)
    exclusions = dataset.metadata.loc["g02", "history_exclusions"]
    assert len(exclusions) == 1
    assert exclusions[0].game_id == "g01"
    assert exclusions[0].reason == reason


@pytest.mark.parametrize("conflicting", [False, True])
def test_accepted_history_cannot_select_one_of_duplicate_target_identities(conflicting):
    targets, cutoffs, history = input_fixture(2)
    source = target_history_fixture(targets[0])
    duplicate = replace(targets[0], winner_team_id="B") if conflicting else targets[0]
    history = history + (source,)

    dataset = build(targets + (duplicate,), cutoffs, history)
    reversed_dataset = build((duplicate,) + tuple(reversed(targets)), cutoffs, history)

    assert_same_dataset(dataset, reversed_dataset)
    assert dataset.status == "EMPTY"
    assert dataset.X_pre.empty
    assert dataset.X_post.empty
    assert [(item.game_id, item.reason, item.source_game_id) for item in dataset.excluded] == [
        ("g01", "E_TARGET_DUPLICATE_GAME_ID", None),
        ("g02", "E_SOURCE_CONFLICT", "g01"),
    ]


def retrospective_input_fixture(count=20):
    """Synthetic protocol assertions, not real PRE evidence."""
    targets, cutoffs, history = input_fixture(count)
    assumed = tuple(
        replace(
            record,
            history_cutoff_at=record.history_cutoff_at.replace(hour=0),
            evidence_ref=f"synthetic://retrospective/{record.game_id}",
            policy_version=RETROSPECTIVE_PROTOCOL,
            verification=CutoffVerification.PROTOCOL_ASSUMED,
        )
        for record in cutoffs
    )
    return targets, assumed, history


def build_retrospective(targets, cutoffs, history):
    return build_paired_dataset(
        targets,
        history,
        cutoffs,
        CHAMPIONS,
        frozenset({RETROSPECTIVE_PROTOCOL}),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )


def test_retrospective_pairing_order_and_split():
    inputs = retrospective_input_fixture()
    before = deepcopy(inputs)
    dataset = build_retrospective(*inputs)
    assert dataset.status == "OK"
    assert_frame_equal(dataset.X_post.loc[:, list(PRE_COLUMNS)], dataset.X_pre)
    assert dataset.X_pre.index.equals(dataset.y.index)
    assert set(dataset.metadata["dataset_version"]) == {RETROSPECTIVE_DATASET_VERSION}
    assert set(dataset.metadata["cutoff_verification"]) == {"PROTOCOL_ASSUMED"}
    assert set(dataset.metadata["core_cutoff_verification"]) == {"NOT_ASSESSED"}
    assert inputs == before

    split = temporal_split(dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    assert split.status == "OK"
    assert split.evaluation_protocol == RETROSPECTIVE_PROTOCOL
    assert split.train == tuple(f"g{number:02d}" for number in range(1, 15))
    assert split.validation == ("g15", "g16", "g17")
    assert split.test == ("g18", "g19", "g20")
    reversed_dataset = build_retrospective(
        *(tuple(reversed(values)) for values in inputs)
    )
    assert_same_dataset(dataset, reversed_dataset)
    assert temporal_split(
        reversed_dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL
    ) == split


@pytest.mark.parametrize(
    "verification",
    (CutoffVerification.PROTOCOL_ASSUMED, CutoffVerification.VERIFIED_EXTERNALLY),
)
def test_strict_builder_rejects_retrospective_policy_even_if_relabelled(verification):
    targets, cutoffs, history = retrospective_input_fixture(1)
    dataset = build_paired_dataset(
        targets,
        history,
        (replace(cutoffs[0], verification=verification),),
        CHAMPIONS,
        frozenset({RETROSPECTIVE_PROTOCOL}),
    )
    assert dataset.status == "EMPTY"
    assert dataset.excluded[0].reason == "E_CUTOFF_PROTOCOL_MISMATCH"


@pytest.mark.parametrize(
    "verification",
    (CutoffVerification.UNVERIFIED, CutoffVerification.VERIFIED_EXTERNALLY),
)
def test_retrospective_requires_assumed_status(verification):
    targets, cutoffs, history = retrospective_input_fixture(1)
    dataset = build_retrospective(
        targets,
        (replace(cutoffs[0], verification=verification),),
        history,
    )
    assert dataset.status == "EMPTY"
    assert dataset.excluded[0].reason == "E_CUTOFF_UNVERIFIED"


@pytest.mark.parametrize("conflicting", (False, True))
def test_retrospective_duplicate_cutoffs_exclude_whole_pair(conflicting):
    targets, cutoffs, history = retrospective_input_fixture(2)
    duplicate = cutoffs[0]
    if conflicting:
        duplicate = replace(
            duplicate,
            history_cutoff_at=duplicate.history_cutoff_at + timedelta(hours=1),
        )
    dataset = build_retrospective(targets, cutoffs + (duplicate,), history)
    assert tuple(dataset.X_pre.index) == ("g02",)
    assert dataset.X_pre.index.equals(dataset.X_post.index)
    assert dataset.excluded[0].reason == "E_CUTOFF_CONFLICT"


def test_split_rejects_cross_protocol_and_mixed_rows():
    strict = build(*input_fixture())
    retrospective = build_retrospective(*retrospective_input_fixture())
    assert getattr(temporal_split(strict), "evaluation_protocol", STRICT_PROTOCOL) == (
        STRICT_PROTOCOL
    )
    with pytest.raises(DatasetInputError, match="E_DATASET_PROTOCOL_MISMATCH"):
        temporal_split(retrospective)
    with pytest.raises(DatasetInputError, match="E_DATASET_PROTOCOL_MISMATCH"):
        temporal_split(strict, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    mixed_metadata = retrospective.metadata.copy(deep=True)
    mixed_metadata.at["g01", "dataset_version"] = strict.metadata.at[
        "g01", "dataset_version"
    ]
    with pytest.raises(DatasetInputError, match="E_DATASET_PROTOCOL_MISMATCH"):
        temporal_split(
            replace(retrospective, metadata=mixed_metadata),
            evaluation_protocol=RETROSPECTIVE_PROTOCOL,
        )
