"""Explicit import of pinned retrospective predictions; default is read-only dry-run."""

import argparse
import json
import os

from match_insight.features.pre import PreFeatureInputError
from match_insight.services.import_evaluations import import_retrospective_evaluations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist validated PRE/POST pairs")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1,
                        help="Bounded independent processes for reconstruction/import (default: 1)")
    args = parser.parse_args(argv)
    keys = ("DATABASE_URL", "WIKI_USERNAME_MATCH_INSIGHT", "WIKI_PASSWORD_MATCH_INSIGHT")
    previous = {key: os.environ.get(key) for key in keys}
    engine = None
    try:
        from sqlalchemy import create_engine

        from match_insight.config import Settings

        engine = create_engine(Settings().database_url)
        report = import_retrospective_evaluations(
            engine, apply=args.apply, workers=args.workers,
            progress=lambda event: print(json.dumps(event, allow_nan=False), flush=True),
        )
    except PreFeatureInputError as error:
        report = {"status": "BLOCKED", "reason": error.code}
    except Exception:
        # Database exception text may contain connection details; never print it.
        report = {"status": "BLOCKED", "reason": "E_EVALUATION_IMPORT_FAILED"}
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
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] in ("DRY_RUN_READY", "IMPORTED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
