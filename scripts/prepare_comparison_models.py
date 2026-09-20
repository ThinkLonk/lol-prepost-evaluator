"""Explicit offline preparation of two supplemental model families; no UI fitting."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from match_insight.features.pre import PreFeatureInputError
from match_insight.ml.comparison import (
    DEFAULT_COMPARISON_MANIFEST,
    fit_comparison_models,
    load_comparison_models,
    publish_comparison_models,
)
from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL
from match_insight.ml.models import ModelInputError, _dataset_signature
from match_insight.ml.real_training import (
    CANONICAL_RETROSPECTIVE_EXPECTATION,
    MODEL_DIRECTORY,
    RETROSPECTIVE_INPUT,
    RETROSPECTIVE_MODEL_FILENAME,
    load_evidence,
    load_retrospective_bundle,
    prepare_run,
)
from match_insight.services.demo import CANONICAL_DEMO_INPUT_SHA256, CANONICAL_DEMO_SNAPSHOT_SHA256
from scripts.train_real_models import _runtime_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Explicitly fit and publish candidates")
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--expected-split-id")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_COMPARISON_MANIFEST.parent)
    args = parser.parse_args(argv)
    expected = CANONICAL_RETROSPECTIVE_EXPECTATION
    if args.apply and (
        args.expected_dataset_id != expected.identity.dataset_id
        or args.expected_split_id != expected.identity.split_id
    ):
        parser.error("--apply requires the exact reviewed canonical dataset and split IDs")
    try:
        manifest = args.output_directory / "manifest.json"
        if args.apply and args.output_directory.exists():
            if any(args.output_directory.iterdir()):
                raise ModelInputError("E_MODEL_ARTIFACT_EXISTS", "Output directory must be empty")
        data = RETROSPECTIVE_INPUT.read_bytes()
        if hashlib.sha256(data).hexdigest() != CANONICAL_DEMO_INPUT_SHA256:
            raise ModelInputError("E_EVIDENCE_HASH_MISMATCH", "Use the pinned retrospective input")
        canonical = load_retrospective_bundle(
            MODEL_DIRECTORY / RETROSPECTIVE_MODEL_FILENAME, expected=expected,
        )
        print("Checking pinned evidence and read-only PostgreSQL snapshot...", file=sys.stderr, flush=True)
        evidence = load_evidence(RETROSPECTIVE_INPUT, evaluation_protocol=RETROSPECTIVE_PROTOCOL)
        snapshot = _runtime_snapshot()
        if snapshot.sha256 != CANONICAL_DEMO_SNAPSHOT_SHA256:
            raise ModelInputError("E_EVIDENCE_SNAPSHOT_MISMATCH", "Use the pinned DB snapshot")
        print("Building the original paired dataset and temporal split...", file=sys.stderr, flush=True)
        plan = prepare_run(
            snapshot, evidence, frozenset({RETROSPECTIVE_PROTOCOL}),
            evaluation_protocol=RETROSPECTIVE_PROTOCOL,
        )
        if (
            plan.report["status"] != "READY"
            or plan.report.get("dataset_id") != canonical.identity.dataset_id
            or plan.report.get("split_id") != canonical.identity.split_id
            or plan.split != canonical.split
            or _dataset_signature(plan.dataset) != canonical.dataset_signature
        ):
            raise ModelInputError("E_MODEL_DATASET_IDENTITY", "Prepared inputs differ from canonical")
        report = {
            "status": "DRY_RUN_READY", "dataset_id": canonical.identity.dataset_id,
            "split_id": canonical.identity.split_id, "canonical_family": canonical.family,
            "canonical_bundle_version": canonical.identity.bundle_version,
            "partition_counts": plan.report["partition_counts"],
            "training_recipe": "UNCHANGED", "test_evaluation": "NOT_RUN",
        }
        if args.apply:
            print("Fitting the unchanged comparison recipes offline...", file=sys.stderr, flush=True)
            models = fit_comparison_models(plan.dataset, plan.split, canonical)
            publish_comparison_models(models, canonical, manifest)
            reloaded = load_comparison_models(canonical, manifest)
            report.update(
                status="COMPARISON_MODELS_READY", model_set_id=reloaded.model_set_id,
                manifest=str(manifest), families=[bundle.family for bundle in reloaded.bundles],
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except PreFeatureInputError as error:
        report = {"status": "BLOCKED", "reason": error.code, "detail": str(error)}
    except Exception:
        report = {"status": "BLOCKED", "reason": "E_COMPARISON_PREPARATION_FAILED"}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
