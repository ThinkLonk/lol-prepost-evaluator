"""Supplemental fitted families for comparison with the unchanged canonical model.

Only the explicit offline preparation API fits models. Runtime loading and validation
never train, choose another winner, evaluate test, or overwrite canonical artifacts.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL
from match_insight.ml.models import (
    FAMILIES,
    ModelBundle,
    ModelIdentity,
    ModelInputError,
    _check_bundle,
    _contract_protocol,
    _dataset_signature,
    _digest,
    _plain,
    fit_model_candidates,
    load_bundle,
    save_bundle,
)
from match_insight.ml.real_training import BUNDLE_FIELDS, ROOT, _json_document

COMPARISON_SCHEMA = "model-comparison-set-v1"
DEFAULT_COMPARISON_MANIFEST = ROOT / "artifacts/models_comparison/manifest.json"
_FILES = {
    "baseline": "baseline_pre_post.joblib",
    "random_forest": "random_forest_pre_post.joblib",
}
_COMMON_FIELDS = (
    "config", "selection_policy", "contract", "split", "dataset_signature",
    "library_versions", "pre_columns", "post_columns", "feature_schema_version",
    "preprocessing_version",
)


@dataclass(frozen=True, slots=True)
class ComparisonModels:
    model_set_id: str
    bundles: tuple[ModelBundle, ...]


def _invalid(detail):
    raise ModelInputError("E_MODEL_BUNDLE_INVALID", detail)


def _metadata(bundle):
    return _plain({field: getattr(bundle, field) for field in BUNDLE_FIELDS})


def _candidate_identity(canonical, family):
    return ModelIdentity(
        canonical.identity.dataset_id,
        canonical.identity.split_id,
        "retrospective:comparison:" + family + ":" + _digest(
            (COMPARISON_SCHEMA, canonical.identity, canonical.fit_signature, family)
        ),
    )


def _same_scores(left, right):
    """Allow only numerical roundoff in a repeated, unchanged validation recipe."""
    left, right = _plain(left), _plain(right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_scores(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_scores(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left, right, rel_tol=0.0, abs_tol=1e-12,
        )
    return left == right


def _set_id(bundles):
    return "comparison:" + _digest((COMPARISON_SCHEMA, tuple(_metadata(row) for row in bundles)))


def _require_canonical(canonical):
    _check_bundle(canonical)
    if canonical.family != "logistic_regression" or (
        _contract_protocol(canonical.contract) != RETROSPECTIVE_PROTOCOL
    ):
        _invalid("Comparison requires the supplied retrospective Logistic Regression bundle")


def validate_comparison_models(models, canonical_bundle):
    """Validate relative to a canonical bundle already loaded by its pinned boundary."""
    _require_canonical(canonical_bundle)
    if (
        not isinstance(models, ComparisonModels)
        or not isinstance(models.bundles, tuple)
        or len(models.bundles) != len(FAMILIES)
        or any(not isinstance(row, ModelBundle) for row in models.bundles)
        or tuple(row.family for row in models.bundles) != FAMILIES
        or models.bundles[1] is not canonical_bundle
    ):
        _invalid("Require ordered baseline, canonical Logistic Regression, and Random Forest")
    for bundle in models.bundles:
        _check_bundle(bundle)
        if any(getattr(bundle, field) != getattr(canonical_bundle, field) for field in _COMMON_FIELDS):
            _invalid("Comparison models must use the same data, split and training recipe")
        if not _same_scores(bundle.validation_scores, canonical_bundle.validation_scores):
            _invalid("Candidate validation scores differ from the reviewed training run")
        if bundle.family != "logistic_regression" and bundle.identity != _candidate_identity(
            canonical_bundle, bundle.family,
        ):
            _invalid("Candidate identity does not match its canonical training reference")
    if (
        len({row.identity.bundle_version for row in models.bundles}) != len(FAMILIES)
        or models.model_set_id != _set_id(models.bundles)
    ):
        _invalid("Comparison set identity is inconsistent")
    return models


def fit_comparison_models(dataset, split, canonical_bundle):
    """Explicit offline replay of existing recipes; retain the original canonical LR.

    Train/validation membership, labels, seed, preprocessing and parameters are
    unchanged. Test metrics are not computed and selection is not rerun.
    """
    _require_canonical(canonical_bundle)
    if split != canonical_bundle.split or _dataset_signature(dataset) != (
        canonical_bundle.dataset_signature
    ):
        _invalid("Preparation requires the exact reviewed dataset and split")
    identities = {
        family: canonical_bundle.identity if family == "logistic_regression"
        else _candidate_identity(canonical_bundle, family)
        for family in FAMILIES
    }
    candidates = fit_model_candidates(
        dataset, split, canonical_bundle.identity, canonical_bundle.config,
        canonical_bundle.selection_policy, evaluation_protocol=RETROSPECTIVE_PROTOCOL,
        candidate_identities=identities,
    )
    repeated = candidates[1]
    for field in BUNDLE_FIELDS:
        if field == "validation_scores":
            same = _same_scores(repeated.validation_scores, canonical_bundle.validation_scores)
        else:
            same = getattr(repeated, field) == getattr(canonical_bundle, field)
        if not same:
            _invalid(f"Repeated training differs from canonical metadata: {field}")
    bundles = (candidates[0], canonical_bundle, candidates[2])
    return validate_comparison_models(ComparisonModels(_set_id(bundles), bundles), canonical_bundle)


def publish_comparison_models(models, canonical_bundle, manifest_path=DEFAULT_COMPARISON_MANIFEST):
    """Publish new supplemental files only, without overwriting any existing file."""
    validate_comparison_models(models, canonical_bundle)
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    if manifest_path.name != "manifest.json":
        _invalid("Use the dedicated comparison manifest filename")
    destinations = (manifest_path,) + tuple(directory / name for name in _FILES.values())
    if any(path.exists() for path in destinations):
        raise ModelInputError("E_MODEL_ARTIFACT_EXISTS", "Refusing to overwrite comparison artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    published = []
    with TemporaryDirectory(prefix=".comparison-", dir=directory) as temporary:
        temporary = Path(temporary)
        entries = []
        for bundle in models.bundles:
            if bundle.family == "logistic_regression":
                continue
            filename = _FILES[bundle.family]
            path = temporary / filename
            save_bundle(bundle, path)
            entries.append({
                "family": bundle.family, "role": "SUPPLEMENTAL_CANDIDATE", "filename": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bundle": _metadata(bundle),
            })
        manifest = {
            "artifact_schema": COMPARISON_SCHEMA,
            "data_kind": "REAL_RETROSPECTIVE_SIMULATION",
            "evaluation_protocol": RETROSPECTIVE_PROTOCOL,
            "model_set_id": models.model_set_id,
            "canonical_selected_family": canonical_bundle.family,
            "canonical_bundle": _metadata(canonical_bundle),
            "candidates": entries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            # Publish the manifest last; loaders never see a completed partial set.
            for destination in destinations[1:] + destinations[:1]:
                os.link(temporary / destination.name, destination)
                published.append(destination)
        except BaseException:
            for destination in published:
                destination.unlink()
            raise
    return manifest_path


def load_comparison_models(canonical_bundle, manifest_path=DEFAULT_COMPARISON_MANIFEST):
    """Load trusted local supplemental artifacts; validate hashes before deserialization."""
    _require_canonical(canonical_bundle)
    path = Path(manifest_path)
    try:
        document = _json_document(path.read_bytes())
    except FileNotFoundError:
        raise ModelInputError(
            "E_MODEL_COMPARISON_MISSING", "Supplemental comparison artifacts have not been prepared",
        ) from None
    except (OSError, TypeError, ValueError):
        _invalid("Cannot read the comparison manifest")
    if (
        not isinstance(document, dict)
        or set(document) != {
            "artifact_schema", "data_kind", "evaluation_protocol", "model_set_id",
            "canonical_selected_family", "canonical_bundle", "candidates",
        }
        or document["artifact_schema"] != COMPARISON_SCHEMA
        or document["data_kind"] != "REAL_RETROSPECTIVE_SIMULATION"
        or document["evaluation_protocol"] != RETROSPECTIVE_PROTOCOL
        or document["canonical_selected_family"] != canonical_bundle.family
        or document["canonical_bundle"] != _metadata(canonical_bundle)
        or not isinstance(document["candidates"], list)
        or len(document["candidates"]) != len(_FILES)
    ):
        _invalid("Comparison manifest does not match the canonical training contract")
    bundles = {"logistic_regression": canonical_bundle}
    for entry, family in zip(document["candidates"], _FILES, strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"family", "role", "filename", "sha256", "bundle"}
            or entry["family"] != family
            or entry["role"] != "SUPPLEMENTAL_CANDIDATE"
            or entry["filename"] != _FILES[family]
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or not isinstance(entry["bundle"], dict)
            or set(entry["bundle"]) != set(BUNDLE_FIELDS)
        ):
            _invalid("Invalid supplemental candidate entry")
        candidate_path = path.parent / entry["filename"]
        if candidate_path.is_symlink() or candidate_path.resolve().parent != path.parent.resolve():
            _invalid("Candidate artifacts must stay inside the manifest directory")
        bundle = load_bundle(candidate_path, expected_sha256=entry["sha256"])
        if _metadata(bundle) != entry["bundle"]:
            _invalid("Candidate metadata differs from the hashed artifact")
        bundles[family] = bundle
    return validate_comparison_models(
        ComparisonModels(document["model_set_id"], tuple(bundles[family] for family in FAMILIES)),
        canonical_bundle,
    )
