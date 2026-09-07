"""Offline PRE/POST models consuming the paired dataset contract from step 7I.

Fitting and selection use train/validation only. Test evaluation is separate.
Only save_bundle/load_bundle perform filesystem I/O.
Externally asserted cutoff verification is not independently certified here.
"""

import hashlib
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from io import BytesIO
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from match_insight.features.pre import (
    ROLES,
    FeatureConfig,
    PreFeatureInputError,
    TargetGame,
    select_pre_history,
)
from match_insight.ml.dataset import (
    CATEGORICAL_COLUMNS,
    DATASET_VERSION,
    METADATA_COLUMNS,
    POST_COLUMNS,
    PRE_COLUMNS,
    RETROSPECTIVE_DATASET_VERSION,
    RETROSPECTIVE_PROTOCOL,
    STRICT_PROTOCOL,
    PairedDataset,
    SplitExclusion,
    TemporalSplit,
    _validate_tables,
    protocol_contract,
)

FEATURE_SCHEMA_VERSION = "7j-pre-post-from-7i-v1"
PREPROCESSING_VERSION = "median-empty-zero-standardscale-flags-onehot-ignore-v1"
SELECTION_POLICY_VERSION = "validation-mean-brier-logloss-rf-margin-v1"
FAMILIES = ("baseline", "logistic_regression", "random_forest")

CONTEXT_COLUMNS = tuple(
    column
    for column in METADATA_COLUMNS
    if column not in {"target_ended_at", "post_feature_game_ids"}
)
NUMERIC_PRE_COLUMNS = tuple(
    column for column in PRE_COLUMNS if not column.endswith("_missing")
)
NUMERIC_POST_COLUMNS = tuple(
    column
    for column in POST_COLUMNS
    if column not in CATEGORICAL_COLUMNS and not column.endswith("_missing")
)


class ModelInputError(PreFeatureInputError):
    """Invalid model input, artifact or PRE/POST comparison."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    dataset_id: str
    split_id: str
    bundle_version: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    random_seed: int = 1729
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    forest_trees: int = 32
    forest_max_depth: int = 4
    forest_min_samples_leaf: int = 2
    calibration_bins: int = 10


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    version: str = SELECTION_POLICY_VERSION
    rf_min_brier_improvement: float = 0.001


@dataclass(frozen=True, slots=True)
class FeatureContract:
    dataset_version: str
    pre_config_version: str
    post_config_version: str
    pre_config: FeatureConfig
    cutoff_policy_versions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    sample_count: int
    mean_probability: float | None
    observed_blue_win_rate: float | None


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    brier_score: float
    log_loss: float
    roc_auc: float | None
    roc_auc_reason: str | None
    sample_count: int
    calibration: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class ValidationScore:
    family: str
    pre: ProbabilityMetrics
    post: ProbabilityMetrics


@dataclass(frozen=True, slots=True)
class ModelBundle:
    identity: ModelIdentity
    family: str
    pre_pipeline: Pipeline
    post_pipeline: Pipeline
    config: ModelConfig
    selection_policy: SelectionPolicy
    validation_scores: tuple[ValidationScore, ...]
    contract: FeatureContract
    split: TemporalSplit
    dataset_signature: str
    fit_signature: str
    estimator_parameters: tuple[tuple[str, object], ...]
    library_versions: tuple[tuple[str, str], ...]
    pre_columns: tuple[str, ...] = PRE_COLUMNS
    post_columns: tuple[str, ...] = POST_COLUMNS
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION


@dataclass(frozen=True, slots=True)
class PhasePrediction:
    game_id: str
    phase: str
    p_blue_win: float
    history_cutoff_at: datetime
    model_bundle_version: str
    bundle_signature: str
    feature_schema_version: str
    policy_version: str
    context_signature: str
    pre_feature_signature: str


@dataclass(frozen=True, slots=True)
class PairedPrediction:
    game_id: str
    p_blue_win_pre: float
    p_blue_win_post: float
    delta_probability: float
    history_cutoff_at: datetime
    model_bundle_version: str
    feature_schema_version: str
    policy_version: str
    context_signature: str


@dataclass(frozen=True, slots=True)
class TestEvaluation:
    pre: ProbabilityMetrics
    post: ProbabilityMetrics
    predictions: tuple[PairedPrediction, ...]


def _fail(code, detail):
    raise ModelInputError(code, detail)


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _utc(value):
    if (
        not isinstance(value, datetime)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        _fail("E_MODEL_TIMESTAMP_INVALID", "Expected an explicit timezone-aware datetime")
    return value.astimezone(UTC)


def _plain(value):
    """Canonical serialization for fingerprints, including object-valued metadata."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            _fail("E_MODEL_VALUE_INVALID", "Nonfinite value in fingerprint input")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    _fail("E_MODEL_METADATA_INVALID", f"Unsupported metadata type: {type(value).__name__}")


def _digest(value):
    body = json.dumps(
        _plain(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _dataset_signature(dataset):
    return _digest(
        (
            dataset.status,
            dataset.excluded,
            dataset.unused_cutoff_game_ids,
            dataset.X_pre.to_dict("split"),
            dataset.X_post.to_dict("split"),
            dataset.y.to_dict(),
            dataset.metadata.to_dict("split"),
        )
    )


def _versions():
    return (
        ("python", platform.python_version()),
        ("numpy", version("numpy")),
        ("pandas", version("pandas")),
        ("scikit-learn", version("scikit-learn")),
        ("scipy", version("scipy")),
        ("joblib", version("joblib")),
    )


def _validate_options(identity, config, policy):
    if not isinstance(identity, ModelIdentity) or any(
        not _text(value)
        for value in (identity.dataset_id, identity.split_id, identity.bundle_version)
    ):
        _fail("E_MODEL_IDENTITY_INVALID", "Supply dataset, split and bundle identities")
    if not isinstance(config, ModelConfig):
        _fail("E_MODEL_CONFIG_INVALID", "Expected ModelConfig")
    integer_values = (
        config.logistic_max_iter,
        config.forest_trees,
        config.forest_max_depth,
        config.forest_min_samples_leaf,
        config.calibration_bins,
    )
    if any(type(value) is not int or value <= 0 for value in integer_values):
        _fail("E_MODEL_CONFIG_INVALID", "Counts, depth and bins must be positive integers")
    if (
        type(config.random_seed) is not int
        or not 0 <= config.random_seed < 2**32
        or isinstance(config.logistic_c, bool)
        or not isinstance(config.logistic_c, Real)
        or not math.isfinite(config.logistic_c)
        or config.logistic_c <= 0
    ):
        _fail("E_MODEL_CONFIG_INVALID", "Invalid seed or logistic C")
    _validate_policy(policy)


def _validate_policy(policy):
    if (
        not isinstance(policy, SelectionPolicy)
        or policy.version != SELECTION_POLICY_VERSION
        or isinstance(policy.rf_min_brier_improvement, bool)
        or not isinstance(policy.rf_min_brier_improvement, Real)
        or not math.isfinite(policy.rf_min_brier_improvement)
        or not 0 <= policy.rf_min_brier_improvement <= 1
    ):
        _fail("E_MODEL_POLICY_INVALID", "Unsupported selection policy or RF margin")


def _contract_protocol(contract):
    if not isinstance(contract, FeatureContract):
        _fail("E_MODEL_SCHEMA", "Expected a fitted FeatureContract")
    if contract.dataset_version == DATASET_VERSION:
        protocol = STRICT_PROTOCOL
    elif contract.dataset_version == RETROSPECTIVE_DATASET_VERSION:
        protocol = RETROSPECTIVE_PROTOCOL
    else:
        _fail("E_MODEL_SCHEMA", "Unsupported dataset evaluation protocol")
    policies = contract.cutoff_policy_versions
    if (
        not isinstance(policies, tuple)
        or not policies
        or any(not _text(policy) for policy in policies)
        or tuple(sorted(set(policies))) != policies
        or (
            protocol == RETROSPECTIVE_PROTOCOL
            and policies != (RETROSPECTIVE_PROTOCOL,)
        )
        or (
            protocol == STRICT_PROTOCOL
            and RETROSPECTIVE_PROTOCOL in policies
        )
    ):
        _fail("E_MODEL_SCHEMA", "Cutoff policies conflict with the bundle protocol")
    return protocol


def _context_metadata(
    metadata,
    index,
    expected=None,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
):
    if expected is not None:
        evaluation_protocol = _contract_protocol(expected)
    try:
        dataset_version, required_verification = protocol_contract(evaluation_protocol)
    except PreFeatureInputError as error:
        raise ModelInputError("E_MODEL_SCHEMA", str(error)) from error

    if (
        not isinstance(metadata, pd.DataFrame)
        or tuple(metadata.columns) != CONTEXT_COLUMNS
        or not metadata.index.is_unique
        or not metadata.index.equals(index)
        or metadata.empty
    ):
        _fail("E_MODEL_ALIGNMENT", "Context metadata must exactly align with feature rows")

    normalized = metadata.copy(deep=True)
    common = None
    policies = set()

    for game_id in index:
        row = metadata.loc[game_id]
        config = row["pre_config"]
        if not isinstance(config, FeatureConfig) or any(
            type(value) is not int or value <= 0
            for value in (
                config.recent_form_games,
                config.side_win_rate_games,
                config.head_to_head_games,
            )
        ):
            _fail("E_MODEL_SCHEMA", "Invalid PRE feature configuration")

        expected_pre_version = (
            f"pre-v1-r{config.recent_form_games}"
            f"-s{config.side_win_rate_games}"
            f"-h{config.head_to_head_games}-roster-latest1"
        )
        if (
            row["dataset_version"] != dataset_version
            or row["pre_config_version"] != expected_pre_version
            or row["post_config_version"] != "post-v1-player-champion-all-pre-history"
            or row["cutoff_verification"] != required_verification.value
            or row["core_cutoff_verification"] != "NOT_ASSESSED"
            or not _text(row["evidence_ref"])
            or not _text(row["policy_version"])
            or (
                evaluation_protocol == RETROSPECTIVE_PROTOCOL
            ) != (row["policy_version"] == RETROSPECTIVE_PROTOCOL)
        ):
            _fail("E_MODEL_SCHEMA", "Unsupported feature version or incomplete cutoff contract")

        current = (
            row["dataset_version"],
            row["pre_config_version"],
            row["post_config_version"],
            config,
        )
        if common is not None and current != common:
            _fail("E_MODEL_SCHEMA", "One bundle requires one compatible feature configuration")
        common = current
        policies.add(row["policy_version"])

        # Reuse the PRE context validator; no history or features are recomputed.
        target = TargetGame(
            game_id=game_id,
            blue_team_id=row["blue_team_id"],
            red_team_id=row["red_team_id"],
            blue_roster=row["blue_roster"],
            red_roster=row["red_roster"],
            patch=row["patch"],
            history_cutoff_at=row["history_cutoff_at"],
        )
        target = select_pre_history(target, ()).target
        normalized.at[game_id, "history_cutoff_at"] = target.history_cutoff_at
        for column in ("blue_roster", "red_roster"):
            normalized.at[game_id, column] = tuple(
                sorted(getattr(target, column), key=lambda slot: ROLES.index(slot.role))
            )

    contract = FeatureContract(*common, tuple(sorted(policies)))
    if expected is not None:
        if (
            (
                contract.dataset_version,
                contract.pre_config_version,
                contract.post_config_version,
                contract.pre_config,
            )
            != (
                expected.dataset_version,
                expected.pre_config_version,
                expected.post_config_version,
                expected.pre_config,
            )
            or not policies <= set(expected.cutoff_policy_versions)
        ):
            _fail("E_MODEL_SCHEMA", "Input is incompatible with the fitted bundle")
        contract = expected
    return normalized, contract


def _model_frame(frame, phase):
    if phase not in ("PRE", "POST"):
        _fail("E_MODEL_PHASE_INVALID", "Expected PRE or POST")
    columns = PRE_COLUMNS if phase == "PRE" else POST_COLUMNS
    if (
        not isinstance(frame, pd.DataFrame)
        or tuple(frame.columns) != columns
        or not frame.index.is_unique
        or frame.empty
    ):
        _fail("E_MODEL_SCHEMA", "Feature columns/order and unique game IDs are required")

    result = pd.DataFrame(index=frame.index.copy())
    for column in columns:
        values = []
        for value in frame[column]:
            if column in CATEGORICAL_COLUMNS:
                if not _text(value):
                    _fail("E_MODEL_VALUE_INVALID", f"{column}: expected a champion string ID")
                values.append(value)
            elif column.endswith("_missing"):
                if not isinstance(value, (bool, np.bool_)):
                    _fail("E_MODEL_VALUE_INVALID", f"{column}: expected a boolean flag")
                values.append(float(value))
            elif value is None or value is pd.NA:
                values.append(np.nan)
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or math.isinf(float(value))
            ):
                _fail("E_MODEL_VALUE_INVALID", f"{column}: expected a finite number or missing")
            else:
                values.append(float(value))
        result[column] = values

    if phase == "POST":
        for row in result.loc[:, list(CATEGORICAL_COLUMNS)].itertuples(index=False, name=None):
            if len(set(row)) != 10:
                _fail("E_MODEL_VALUE_INVALID", "POST must retain ten distinct champion identities")
    return result


def _prepare_dataset(dataset, *, evaluation_protocol=STRICT_PROTOCOL):
    try:
        ids = _validate_tables(dataset)
    except PreFeatureInputError as error:
        raise ModelInputError("E_MODEL_INPUT", str(error), error.game_id) from error
    if not ids:
        _fail("E_MODEL_DATASET_EMPTY", "An EMPTY dataset cannot train or evaluate a model")

    metadata, contract = _context_metadata(
        dataset.metadata.loc[:, list(CONTEXT_COLUMNS)],
        dataset.X_pre.index,
        evaluation_protocol=evaluation_protocol,
    )
    pre = _model_frame(dataset.X_pre, "PRE")
    post = _model_frame(dataset.X_post, "POST")
    return pre, post, metadata, contract


def _validate_split(
    dataset,
    split,
    metadata,
    *,
    evaluation_protocol=STRICT_PROTOCOL,
):
    if (
        not isinstance(split, TemporalSplit)
        or split.status != "OK"
        or getattr(split, "evaluation_protocol", STRICT_PROTOCOL) != evaluation_protocol
    ):
        _fail("E_MODEL_SPLIT_INVALID", "Expected an OK TemporalSplit from step 7I")
    validation_boundary = _utc(split.validation_boundary)
    test_boundary = _utc(split.test_boundary)
    if validation_boundary >= test_boundary:
        _fail("E_MODEL_SPLIT_INVALID", "Invalid supplied temporal boundaries")

    memberships = {
        "train": tuple(split.train),
        "validation": tuple(split.validation),
        "test": tuple(split.test),
    }
    retained = tuple(game_id for values in memberships.values() for game_id in values)
    if any(not values for values in memberships.values()) or len(set(retained)) != len(retained):
        _fail("E_MODEL_SPLIT_INVALID", "Partitions must be nonempty, unique and disjoint")
    if any(not isinstance(item, SplitExclusion) for item in split.excluded):
        _fail("E_MODEL_SPLIT_INVALID", "Invalid split exclusion contract")
    excluded = {item.game_id: item for item in split.excluded}
    known = set(dataset.X_pre.index)
    if (
        len(excluded) != len(split.excluded)
        or set(retained) & set(excluded)
        or set(retained) | set(excluded) != known
    ):
        _fail("E_MODEL_SPLIT_INVALID", "Membership and exclusions must cover the dataset exactly")

    assigned = {
        game_id: partition
        for partition, values in memberships.items()
        for game_id in values
    }
    for game_id, item in excluded.items():
        if item.partition not in ("train", "validation"):
            _fail("E_MODEL_SPLIT_INVALID", "Only train/validation label purging is supported")
        assigned[game_id] = item.partition

    planned_validation = []
    for game_id, partition in assigned.items():
        cutoff = _utc(metadata.at[game_id, "history_cutoff_at"])
        end = _utc(dataset.metadata.at[game_id, "target_ended_at"])
        if end <= cutoff:
            _fail("E_MODEL_SPLIT_INVALID", "Target label end must follow its own cutoff")

        if partition == "train":
            in_range = cutoff < validation_boundary
            limit = validation_boundary
            reason = "TRAIN_LABEL_NOT_AVAILABLE_BEFORE_VALIDATION"
        elif partition == "validation":
            in_range = validation_boundary <= cutoff < test_boundary
            limit = test_boundary
            reason = "VALIDATION_LABEL_NOT_AVAILABLE_BEFORE_TEST"
            planned_validation.append(cutoff)
        else:
            in_range = cutoff >= test_boundary
            limit = None
            reason = None
        if not in_range:
            _fail("E_MODEL_SPLIT_INVALID", "Membership conflicts with supplied cutoff boundaries")

        if game_id in excluded:
            item = excluded[game_id]
            if (
                item.reason != reason
                or _utc(item.ended_at) != end
                or _utc(item.required_before) != limit
                or end < limit
            ):
                _fail("E_MODEL_SPLIT_INVALID", "Invalid label-availability exclusion")
        elif limit is not None and end >= limit:
            _fail("E_MODEL_SPLIT_INVALID", "Retained label was unavailable before the next partition")

    if (
        min(planned_validation) != validation_boundary
        or min(metadata.at[game_id, "history_cutoff_at"] for game_id in split.test)
        != test_boundary
    ):
        _fail("E_MODEL_SPLIT_INVALID", "Boundaries do not match the supplied planned partitions")

    # Sort inside existing partitions only; never choose new membership or boundaries.
    return {
        partition: tuple(
            sorted(values, key=lambda game_id: (metadata.at[game_id, "history_cutoff_at"], game_id))
        )
        for partition, values in memberships.items()
    }


def _make_pipeline(phase, family, config):
    columns = PRE_COLUMNS if phase == "PRE" else POST_COLUMNS
    numeric = NUMERIC_PRE_COLUMNS if phase == "PRE" else NUMERIC_POST_COLUMNS
    flags = tuple(column for column in columns if column.endswith("_missing"))
    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]
            ),
            list(numeric),
        ),
        ("missing_flags", "passthrough", list(flags)),
    ]
    if phase == "POST":
        transformers.append(
            (
                "champions",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_COLUMNS),
            )
        )

    if family == "baseline":
        estimator = DummyClassifier(strategy="prior", random_state=config.random_seed)
    elif family == "logistic_regression":
        estimator = LogisticRegression(
            C=config.logistic_c,
            solver="lbfgs",
            max_iter=config.logistic_max_iter,
            random_state=config.random_seed,
        )
    elif family == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=config.forest_trees,
            max_depth=config.forest_max_depth,
            min_samples_leaf=config.forest_min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=1,
        )
    else:
        _fail("E_MODEL_FAMILY_INVALID", "Unsupported model family")

    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(transformers, remainder="drop", sparse_threshold=0),
            ),
            ("model", estimator),
        ]
    )


def _blue_probability(pipeline, frame):
    classes = np.asarray(pipeline.named_steps["model"].classes_)
    positions = np.flatnonzero(classes == 1)
    if len(classes) != 2 or set(classes.tolist()) != {0, 1} or len(positions) != 1:
        _fail("E_MODEL_CLASSES_INVALID", "Estimator must expose binary classes 0 and 1")
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
    if (
        probabilities.shape != (len(frame), 2)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        _fail("E_MODEL_PROBABILITY_INVALID", "Invalid estimator probabilities")
    return probabilities[:, positions[0]]


def evaluate_probabilities(labels, probabilities, bins=10):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    if (
        type(bins) is not int
        or bins <= 0
        or y.ndim != 1
        or p.shape != y.shape
        or len(y) == 0
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(p).all()
        or (p < 0).any()
        or (p > 1).any()
    ):
        _fail("E_MODEL_METRIC_INPUT", "Expected aligned binary labels and finite probabilities")

    bin_ids = np.minimum((p * bins).astype(int), bins - 1)
    calibration = []
    for number in range(bins):
        mask = bin_ids == number
        count = int(mask.sum())
        calibration.append(
            CalibrationBin(
                lower=number / bins,
                upper=(number + 1) / bins,
                sample_count=count,
                mean_probability=float(p[mask].mean()) if count else None,
                observed_blue_win_rate=float(y[mask].mean()) if count else None,
            )
        )
    two_classes = len(np.unique(y)) == 2
    return ProbabilityMetrics(
        brier_score=float(brier_score_loss(y, p, pos_label=1)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y, p)) if two_classes else None,
        roc_auc_reason=None if two_classes else "EVAL_SINGLE_CLASS",
        sample_count=len(y),
        calibration=tuple(calibration),
    )


def choose_family(validation_scores, policy=SelectionPolicy()):
    """Select from validation summaries only; this API has no test input."""
    _validate_policy(policy)
    scores = tuple(validation_scores)
    by_family = {score.family: score for score in scores}
    if len(scores) != len(FAMILIES) or set(by_family) != set(FAMILIES):
        _fail("E_MODEL_SELECTION_INVALID", "Require one validation score per supported family")

    counts = set()
    for score in scores:
        for metrics in (score.pre, score.post):
            if (
                metrics.sample_count <= 0
                or not math.isfinite(metrics.brier_score)
                or not 0 <= metrics.brier_score <= 1
                or not math.isfinite(metrics.log_loss)
                or metrics.log_loss < 0
            ):
                _fail("E_MODEL_SELECTION_INVALID", "Invalid validation metrics")
            counts.add(metrics.sample_count)
    if len(counts) != 1:
        _fail("E_MODEL_SELECTION_INVALID", "All candidates must use the same validation cohort")

    def key(family):
        score = by_family[family]
        return (
            (score.pre.brier_score + score.post.brier_score) / 2,
            (score.pre.log_loss + score.post.log_loss) / 2,
            FAMILIES.index(family),
        )

    winner = min(FAMILIES, key=key)
    if (
        winner == "random_forest"
        and key("logistic_regression")[:2] < key("baseline")[:2]
        and key("logistic_regression")[0] - key("random_forest")[0]
        < policy.rf_min_brier_improvement
    ):
        winner = "logistic_regression"
    return winner


def fit_model_bundle(
    dataset: PairedDataset,
    split: TemporalSplit,
    identity: ModelIdentity,
    config: ModelConfig = ModelConfig(),
    policy: SelectionPolicy = SelectionPolicy(),
    *,
    evaluation_protocol: str = STRICT_PROTOCOL,
):
    """Fit on train, select on validation, and return a locked selection.

    Test features/labels are not passed to fit, preprocessing transform,
    metric evaluation or choose_family.
    """
    _validate_options(identity, config, policy)
    pre, post, metadata, contract = _prepare_dataset(
        dataset, evaluation_protocol=evaluation_protocol
    )
    membership = _validate_split(
        dataset, split, metadata, evaluation_protocol=evaluation_protocol
    )
    train_ids = list(membership["train"])
    validation_ids = list(membership["validation"])
    y_train = dataset.y.loc[train_ids]
    if set(y_train.tolist()) != {0, 1}:
        _fail("E_TRAIN_SINGLE_CLASS", "Train must contain both BLUE-win labels 0 and 1")

    candidates = {}
    scores = []
    for family in FAMILIES:
        pre_pipeline = _make_pipeline("PRE", family, config)
        post_pipeline = _make_pipeline("POST", family, config)
        pre_pipeline.fit(pre.loc[train_ids], y_train)
        post_pipeline.fit(post.loc[train_ids], y_train)
        pre_metrics = evaluate_probabilities(
            dataset.y.loc[validation_ids],
            _blue_probability(pre_pipeline, pre.loc[validation_ids]),
            config.calibration_bins,
        )
        post_metrics = evaluate_probabilities(
            dataset.y.loc[validation_ids],
            _blue_probability(post_pipeline, post.loc[validation_ids]),
            config.calibration_bins,
        )
        candidates[family] = (pre_pipeline, post_pipeline)
        scores.append(ValidationScore(family, pre_metrics, post_metrics))

    scores = tuple(scores)
    family = choose_family(scores, policy)
    pre_pipeline, post_pipeline = candidates[family]
    train_signature = _digest(
        (
            pre.loc[train_ids].to_dict("split"),
            post.loc[train_ids].to_dict("split"),
            y_train.to_dict(),
            metadata.loc[train_ids].to_dict("split"),
        )
    )
    return ModelBundle(
        identity=identity,
        family=family,
        pre_pipeline=pre_pipeline,
        post_pipeline=post_pipeline,
        config=config,
        selection_policy=policy,
        validation_scores=scores,
        contract=contract,
        split=split,
        dataset_signature=_dataset_signature(dataset),
        fit_signature=_digest((identity, family, config, policy, contract, train_signature)),
        estimator_parameters=tuple(
            sorted(pre_pipeline.named_steps["model"].get_params(deep=False).items())
        ),
        library_versions=_versions(),
    )


def _check_bundle(bundle):
    if (
        not isinstance(bundle, ModelBundle)
        or bundle.family not in FAMILIES
        or bundle.pre_columns != PRE_COLUMNS
        or bundle.post_columns != POST_COLUMNS
        or bundle.feature_schema_version != FEATURE_SCHEMA_VERSION
        or bundle.preprocessing_version != PREPROCESSING_VERSION
        or bundle.pre_pipeline is bundle.post_pipeline
    ):
        _fail("E_MODEL_BUNDLE_INVALID", "Unsupported or incompatible model bundle")
    evaluation_protocol = _contract_protocol(bundle.contract)
    if (
        not isinstance(bundle.split, TemporalSplit)
        or getattr(bundle.split, "evaluation_protocol", STRICT_PROTOCOL)
        != evaluation_protocol
    ):
        _fail("E_MODEL_BUNDLE_INVALID", "Bundle and split protocols do not match")
    _validate_options(bundle.identity, bundle.config, bundle.selection_policy)
    if bundle.library_versions != _versions():
        _fail("E_MODEL_LIBRARY_MISMATCH", "Bundle library versions differ from this environment")

    estimator_type = {
        "baseline": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
    }[bundle.family]
    for pipeline, columns in (
        (bundle.pre_pipeline, PRE_COLUMNS),
        (bundle.post_pipeline, POST_COLUMNS),
    ):
        if not isinstance(pipeline, Pipeline) or tuple(name for name, _step in pipeline.steps) != (
            "features", "model"
        ):
            _fail("E_MODEL_BUNDLE_INVALID", "Expected the fitted feature/model pipeline")
        transformer = pipeline.named_steps["features"]
        estimator = pipeline.named_steps["model"]
        if not isinstance(transformer, ColumnTransformer) or not isinstance(
            estimator, estimator_type
        ):
            _fail("E_MODEL_BUNDLE_INVALID", "Pipeline does not match the declared model family")
        try:
            check_is_fitted(transformer)
            check_is_fitted(estimator)
        except (NotFittedError, TypeError):
            _fail("E_MODEL_BUNDLE_INVALID", "Both preprocessing and estimator must be fitted")
        if tuple(getattr(transformer, "feature_names_in_", ())) != columns:
            _fail("E_MODEL_BUNDLE_INVALID", "Fitted feature columns/order differ from the schema")
        classes = np.asarray(getattr(estimator, "classes_", ()))
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            _fail("E_MODEL_CLASSES_INVALID", "Estimator must expose binary classes 0 and 1")


def predict_phase(bundle, phase, features, metadata):
    """Label-free prediction; no target ended_at is required."""
    _check_bundle(bundle)
    frame = _model_frame(features, phase)
    context, _contract = _context_metadata(metadata, frame.index, bundle.contract)
    pipeline = bundle.pre_pipeline if phase == "PRE" else bundle.post_pipeline
    probabilities = _blue_probability(pipeline, frame)
    predictions = []
    for position, game_id in enumerate(frame.index):
        row = context.loc[game_id]
        predictions.append(
            PhasePrediction(
                game_id=game_id,
                phase=phase,
                p_blue_win=float(probabilities[position]),
                history_cutoff_at=row["history_cutoff_at"],
                model_bundle_version=bundle.identity.bundle_version,
                bundle_signature=bundle.fit_signature,
                feature_schema_version=bundle.feature_schema_version,
                policy_version=row["policy_version"],
                context_signature=_digest((game_id, row.to_dict())),
                pre_feature_signature=_digest(
                    frame.loc[game_id, list(PRE_COLUMNS)].to_dict()
                ),
            )
        )
    return tuple(predictions)


def compare_predictions(pre, post):
    if (
        not isinstance(pre, PhasePrediction)
        or not isinstance(post, PhasePrediction)
        or pre.phase != "PRE"
        or post.phase != "POST"
    ):
        _fail("E_EVAL_INCOMPATIBLE", "Expected one PRE and one POST prediction")

    matching_fields = (
        "game_id",
        "history_cutoff_at",
        "model_bundle_version",
        "bundle_signature",
        "feature_schema_version",
        "policy_version",
        "context_signature",
        "pre_feature_signature",
    )
    if any(getattr(pre, field) != getattr(post, field) for field in matching_fields):
        _fail("E_EVAL_INCOMPATIBLE", "Predictions do not share the same PRE foundation")
    _utc(pre.history_cutoff_at)
    if pre.feature_schema_version != FEATURE_SCHEMA_VERSION:
        _fail("E_EVAL_INCOMPATIBLE", "Unsupported prediction feature schema")
    for probability in (pre.p_blue_win, post.p_blue_win):
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            _fail("E_MODEL_PROBABILITY_INVALID", "Prediction must be finite and within [0, 1]")

    return PairedPrediction(
        game_id=pre.game_id,
        p_blue_win_pre=pre.p_blue_win,
        p_blue_win_post=post.p_blue_win,
        delta_probability=post.p_blue_win - pre.p_blue_win,
        history_cutoff_at=pre.history_cutoff_at,
        model_bundle_version=pre.model_bundle_version,
        feature_schema_version=pre.feature_schema_version,
        policy_version=pre.policy_version,
        context_signature=pre.context_signature,
    )


def predict_paired(bundle, X_pre, X_post, metadata):
    if (
        not isinstance(X_pre, pd.DataFrame)
        or not isinstance(X_post, pd.DataFrame)
        or tuple(X_pre.columns) != PRE_COLUMNS
        or tuple(X_post.columns) != POST_COLUMNS
        or not X_pre.index.equals(X_post.index)
        or not X_post.loc[:, list(PRE_COLUMNS)].equals(X_pre)
    ):
        _fail("E_MODEL_ALIGNMENT", "POST must preserve the aligned PRE feature table")
    pre = predict_phase(bundle, "PRE", X_pre, metadata)
    post = predict_phase(bundle, "POST", X_post, metadata)
    return tuple(compare_predictions(left, right) for left, right in zip(pre, post, strict=True))


def evaluate_test(bundle, dataset):
    """Evaluate only the selected fitted bundle; never refit or reselect."""
    _check_bundle(bundle)
    evaluation_protocol = _contract_protocol(bundle.contract)
    pre, post, metadata, contract = _prepare_dataset(
        dataset, evaluation_protocol=evaluation_protocol
    )
    if contract != bundle.contract or _dataset_signature(dataset) != bundle.dataset_signature:
        _fail("E_MODEL_DATASET_IDENTITY", "Evaluation dataset differs from the locked dataset")
    membership = _validate_split(
        dataset, bundle.split, metadata, evaluation_protocol=evaluation_protocol
    )
    ids = list(membership["test"])
    predictions = predict_paired(
        bundle,
        dataset.X_pre.loc[ids],
        dataset.X_post.loc[ids],
        dataset.metadata.loc[ids, list(CONTEXT_COLUMNS)],
    )
    return TestEvaluation(
        pre=evaluate_probabilities(
            dataset.y.loc[ids],
            _blue_probability(bundle.pre_pipeline, pre.loc[ids]),
            bundle.config.calibration_bins,
        ),
        post=evaluate_probabilities(
            dataset.y.loc[ids],
            _blue_probability(bundle.post_pipeline, post.loc[ids]),
            bundle.config.calibration_bins,
        ),
        predictions=predictions,
    )


def save_bundle(bundle, path):
    """Save to a caller-selected path; do not create directories or overwrite files."""
    _check_bundle(bundle)
    target = Path(path)
    if target.exists():
        _fail("E_MODEL_ARTIFACT_EXISTS", "Refusing to overwrite an existing artifact")
    joblib.dump(bundle, target, compress=3)


def load_bundle(path, *, expected_sha256=None):
    """Load trusted local bytes, optionally hash-pinned, and check the fitted contract."""
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        _fail("E_MODEL_BUNDLE_INVALID", "Expected a lowercase SHA-256 digest")
    try:
        data = Path(path).read_bytes()
    except (OSError, TypeError, ValueError):
        raise ModelInputError("E_MODEL_BUNDLE_INVALID", "Cannot read model artifact") from None
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        _fail("E_MODEL_BUNDLE_INVALID", "Model artifact SHA-256 mismatch")
    try:
        bundle = joblib.load(BytesIO(data))
    except Exception:
        raise ModelInputError("E_MODEL_BUNDLE_INVALID", "Cannot deserialize model artifact") from None
    _check_bundle(bundle)
    return bundle
