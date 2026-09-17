"""Explicit real-data CLI. Importing this module does not load DB settings."""

import argparse
import json
import os
from pathlib import Path

from match_insight.database.real_snapshot import read_snapshot
from match_insight.features.pre import PreFeatureInputError
from match_insight.ml.dataset import RETROSPECTIVE_PROTOCOL, STRICT_PROTOCOL
from match_insight.ml.models import _plain
from match_insight.ml.real_training import (
    DEFAULT_INPUT,
    RETROSPECTIVE_INPUT,
    execute_run,
    fail,
    load_evidence,
    load_time_archive,
    write_retrospective_inputs,
)

ENVIRONMENT_KEYS = (
    "DATABASE_URL",
    "WIKI_USERNAME_MATCH_INSIGHT",
    "WIKI_PASSWORD_MATCH_INSIGHT",
)


def _runtime_snapshot():
    previous = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    engine = None
    try:
        # Existing engine/config are loaded only when the user runs this CLI.
        from match_insight.database.engine import engine as project_engine

        engine = project_engine
        return read_snapshot(engine)
    finally:
        try:
            if engine is not None:
                engine.dispose()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only snapshot and model runner")
    parser.add_argument(
        "--mode",
        choices=("create-input", "dry-run", "train"),
        default="dry-run",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=(STRICT_PROTOCOL, RETROSPECTIVE_PROTOCOL),
        default=STRICT_PROTOCOL,
    )
    parser.add_argument("--cutoffs", type=Path)
    parser.add_argument("--approved-policy", action="append", default=[])
    parser.add_argument("--approval-ref")
    parser.add_argument("--expected-dataset-id")
    parser.add_argument("--expected-split-id")
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Custom output directory for model artifacts",
    )
    args = parser.parse_args(argv)

    protocol = args.evaluation_protocol
    retrospective = protocol == RETROSPECTIVE_PROTOCOL
    input_path = args.cutoffs or (
        RETROSPECTIVE_INPUT if retrospective else DEFAULT_INPUT
    )
    if args.mode == "create-input" and (
        not retrospective
        or args.cutoffs is not None
        or not args.approval_ref
        or RETROSPECTIVE_PROTOCOL not in args.approved_policy
    ):
        parser.error(
            "create-input requires the retrospective protocol, its approved policy, "
            "an approval reference and the dedicated default input path"
        )
    if args.mode == "train" and (
        not args.approved_policy
        or not args.expected_dataset_id
        or not args.expected_split_id
    ):
        parser.error("TRAIN requires an approved policy and reviewed dataset/split IDs")

    snapshot = None
    try:
        # Reject file/schema/protocol errors before loading DB configuration.
        if args.mode == "create-input":
            if input_path.exists():
                fail("E_INPUT_ARTIFACT_EXISTS")
            if not input_path.parent.is_dir():
                fail("E_INPUT_DIRECTORY_MISSING")
            archive = load_time_archive()
        else:
            evidence = load_evidence(input_path, evaluation_protocol=protocol)
            if evidence.status == "MISSING" or not evidence.document.get("records"):
                fail("E_CUTOFF_MISSING")
            if retrospective and RETROSPECTIVE_PROTOCOL not in args.approved_policy:
                fail("E_CUTOFF_POLICY_UNAPPROVED")

        snapshot = _runtime_snapshot()
        if args.mode == "create-input":
            report = write_retrospective_inputs(
                snapshot,
                archive,
                input_path,
                approval_ref=args.approval_ref,
            )
        else:
            execute_options = {
                "train": args.mode == "train",
                "expected_dataset_id": args.expected_dataset_id,
                "expected_split_id": args.expected_split_id,
                "evaluation_protocol": protocol,
            }
            if args.output_directory is not None:
                execute_options["output_directory"] = args.output_directory
            report = execute_run(
                snapshot,
                evidence,
                frozenset(args.approved_policy),
                **execute_options,
            )
    except PreFeatureInputError as error:
        status = (
            "NO_ELIGIBLE_TARGETS"
            if error.code in ("E_CUTOFF_MISSING", "E_NO_ELIGIBLE_TARGETS")
            else "BLOCKED"
        )
        report = {
            "status": status,
            "reason": error.code,
            "detail": str(error),
            "paired_samples": 0,
        }
    except Exception:
        report = {"status": "BLOCKED", "reason": "E_REAL_RUN_FAILED"}

    if snapshot is not None:
        report.setdefault("db_source_games", len(snapshot.games))
        report.setdefault(
            "historical_games_with_ended_at",
            sum(item.game.ended_at is not None for item in snapshot.games),
        )
    visible = {
        key: value
        for key, value in report.items()
        if key not in ("target_provenance", "test_predictions", "time_proofs")
    }
    print(json.dumps(_plain(visible), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] in ("INPUT_CREATED", "DRY_RUN_READY", "TRAINED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
