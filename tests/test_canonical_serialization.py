"""The faster serializer must retain existing fingerprints and mutation checks."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from match_insight.ml import models
from match_insight.services import demo


def legacy_plain(value):
    """Independent reference for the serializer used by existing artifacts."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return legacy_plain(value.item())
    if isinstance(value, datetime):
        return models._utc(value).isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return legacy_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): legacy_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [legacy_plain(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Nonfinite")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    raise ValueError("Unsupported")


def test_fingerprints_match_existing_serializer_for_full_runtime_and_snapshots(comparison_environment):
    env = comparison_environment
    pre = demo.evaluate_analysis_pre(env.runtime, env.selection)
    post = demo.evaluate_analysis_post(env.runtime, pre, env.champion_ids)
    structures = [
        (env.runtime.history, env.runtime.proof_fetched_at, env.runtime.history_patches),
        (pre.features, pre.prediction, pre.warnings, pre.request.runtime_context),
        (post.lineup, post.features, post.prediction, post.comparison),
        (env.canonical_bundle.identity, env.canonical_bundle.split,
         env.canonical_bundle.validation_scores, env.canonical_bundle.contract),
        {"numpy": (np.int64(3), np.float64(0.2), np.bool_(True), np.str_("a")),
         "missing": [np.nan, pd.NA, None], "zero": [0, -0.0], "flag": True},
    ]
    for structure in structures:
        actual, expected = models._plain(structure), legacy_plain(structure)
        assert actual == expected
        body = json.dumps(expected, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
        assert models._digest(structure) == hashlib.sha256(body).hexdigest()


def test_returns_owned_containers_and_does_not_reuse_stale_digest(comparison_environment):
    history = comparison_environment.runtime.history
    original = models._digest(history)
    exported = models._plain(history)
    exported[0]["game"]["winner_team_id"] = "tampered-output"
    assert models._digest(history) == original
    changed = (replace(history[0], game=replace(history[0].game, winner_team_id="different")),)
    assert models._digest(changed + history[1:]) != original


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), object(), {"unsupported"}])
def test_still_rejects_unsupported_metadata(value):
    with pytest.raises(models.ModelInputError):
        models._plain(value)
