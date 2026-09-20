"""Small synthetic fitted artifacts; no real database, network or canonical files."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from match_insight.database.demo_catalog import CatalogEntity, DemoCatalog
from match_insight.features.post import ChampionSlot, FinalLineup, HistoricalChampionGame
from match_insight.features.pre import ROLES, HistoricalGame, PlayerSlot
from match_insight.ml import comparison, models
from match_insight.ml.dataset import (
    RETROSPECTIVE_PROTOCOL,
    CutoffRecord,
    CutoffVerification,
    DatasetTarget,
    build_paired_dataset,
    temporal_split,
)
from match_insight.services import demo, evaluation


def _roster(team):
    return tuple(PlayerSlot(role, f"{team.lower()}{index}") for index, role in enumerate(ROLES, 1))


def _slots(blue, red):
    champions = tuple(row.champion_id for row in demo.CHAMPIONS[:10])
    return tuple(
        ChampionSlot(team, side, player.role, player.player_id, champions[offset + position])
        for side, team, offset in (("BLUE", blue, 0), ("RED", red, 5))
        for position, player in enumerate(_roster(team))
    )


@pytest.fixture(scope="session")
def comparison_training_fixture():
    history = tuple(
        HistoricalChampionGame(
            HistoricalGame(
                f"fixture-history-{index}", blue, red, _roster(blue), _roster(red), winner,
                datetime(2025, 6, 20 + index, 12, tzinfo=UTC),
            ),
            _slots(blue, red),
        )
        for index, (blue, red, winner) in enumerate(
            (("A", "B", "A"), ("B", "A", "B"), ("C", "D", "D")), 1,
        )
    )
    targets, cutoffs = [], []
    for number in range(20):
        game_id = f"fixture-training-{number:02d}"
        cutoff = datetime(2025, 7, 1, tzinfo=UTC) + timedelta(days=number)
        targets.append(DatasetTarget(
            game_id, "A", "B", _roster("A"), _roster("B"), "15.01",
            FinalLineup(game_id, "15.01", _slots("A", "B")),
            "A" if number % 3 else "B", cutoff + timedelta(hours=2),
        ))
        cutoffs.append(CutoffRecord(
            game_id, cutoff, f"synthetic://comparison-fixture/{game_id}",
            RETROSPECTIVE_PROTOCOL, CutoffVerification.PROTOCOL_ASSUMED,
        ))
    dataset = build_paired_dataset(
        targets, history, cutoffs, demo.CHAMPION_REFERENCE, frozenset({RETROSPECTIVE_PROTOCOL}),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    split = temporal_split(dataset, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
    candidates = models.fit_model_candidates(
        dataset, split,
        models.ModelIdentity("fixture-dataset", "fixture-split", "fixture-canonical-lr"),
        models.ModelConfig(forest_trees=4, forest_max_depth=2),
        evaluation_protocol=RETROSPECTIVE_PROTOCOL,
    )
    canonical = next(bundle for bundle in candidates if bundle.family == "logistic_regression")
    model_set = comparison.fit_comparison_models(dataset, split, canonical)
    return SimpleNamespace(
        dataset=dataset, split=split, canonical_bundle=canonical, models=model_set, history=history,
    )


@pytest.fixture
def comparison_environment(monkeypatch, comparison_training_fixture):
    training = deepcopy(comparison_training_fixture)
    canonical = training.canonical_bundle
    expected = SimpleNamespace(identity=canonical.identity, family=canonical.family)
    monkeypatch.setattr(demo, "CANONICAL_RETROSPECTIVE_EXPECTATION", expected)
    monkeypatch.setattr(evaluation, "CANONICAL_RETROSPECTIVE_EXPECTATION", expected)
    clock = {"now": datetime(2025, 9, 1, 12, tzinfo=UTC)}
    monkeypatch.setattr(demo, "_utc_now", lambda: clock["now"])
    history = training.history
    fetched = tuple((record.game.game_id, record.game.ended_at + timedelta(hours=1))
                    for record in history)
    patches = tuple((record.game.game_id, "15.01") for record in history)
    catalog = DemoCatalog(
        tuple(CatalogEntity(team, f"Đội {team}", None) for team in ("A", "B", "C", "D")),
        tuple(CatalogEntity(player.player_id, f"Người chơi {player.player_id}", None)
              for team in ("A", "B", "C", "D") for player in _roster(team)),
        tuple(CatalogEntity(champion.champion_id, champion.name, None)
              for champion in demo.CHAMPIONS),
        (),
    )
    runtime = demo.AnalysisRuntime(
        canonical, catalog, history, demo.CHAMPION_REFERENCE, fetched, patches,
        demo.CANONICAL_DEMO_SNAPSHOT_SHA256, demo.CANONICAL_DEMO_INPUT_SHA256,
        demo._pool_digest(history, fetched, patches),
        clock["now"] - timedelta(hours=2), clock["now"] - timedelta(hours=1), len(history),
    )
    selection = demo.AnalysisInput("A", "D", _roster("A"), _roster("D"), "15.01", True)
    return SimpleNamespace(
        runtime=runtime, selection=selection, canonical_bundle=canonical, models=training.models,
        clock=clock, champion_ids=tuple(row.champion_id for row in demo.CHAMPIONS[:10]),
        dataset=training.dataset, split=training.split,
    )
