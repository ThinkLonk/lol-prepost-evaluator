"""Local media exclusions by exact entity ID; source catalog data stays intact."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

POLICY_PATH = Path("data/reference/media_exclusions.json")


def load_media_exclusions(project_root: Path, entity_kind: str) -> frozenset[str]:
    """Return explicit exclusions; invalid policy stops sync before downloading."""
    if entity_kind not in {"players", "teams"}:
        raise ValueError("Media policy supports only players and teams.")
    path = Path(os.path.abspath(project_root)) / POLICY_PATH
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Media policy path cannot use a symlink or reparse point.")
    if not path.exists():
        return frozenset()
    if not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("Media policy must be a JSON file smaller than 4 MB.")
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Media policy cannot be read; sync stopped.") from exc
    if (
        not isinstance(policy, dict)
        or type(policy.get("schema_version")) is not int
        or policy["schema_version"] != 1
        or policy.get("mode") != "exclude_exact_ids"
        or set(policy) != {"schema_version", "mode", "players", "teams"}
    ):
        raise ValueError("Invalid media policy schema; sync stopped.")
    for kind in ("players", "teams"):
        values = policy[kind]
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in values
        ):
            raise ValueError("Media policy IDs must be nonempty exact strings.")
        if len(set(values)) != len(values):
            raise ValueError("Media policy contains duplicate IDs.")
    return frozenset(policy[entity_kind])
