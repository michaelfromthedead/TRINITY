#!/usr/bin/env python3
"""Cross-reference workspace_manifest.json against actual workspace_dir contents.

Usage: python validate_phase_outputs.py <workspace_dir>
Exit 0 = every claimed output exists and is non-empty; exit 1 = mismatch.
"""

import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"INVALID: {msg}", file=sys.stderr)
    sys.exit(1)


def main(workspace_dir_str: str) -> None:
    workspace_dir = Path(workspace_dir_str)
    if not workspace_dir.is_dir():
        fail(f"workspace_dir not a directory: {workspace_dir}")

    manifest_path = workspace_dir / "workspace_manifest.json"
    if not manifest_path.exists():
        fail(f"workspace_manifest.json not found at {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        fail(f"manifest invalid JSON: {e}")

    missing_files = []
    empty_files = []
    checked = 0

    for phase in manifest.get("phases", []):
        for task in phase.get("tasks", []):
            if task.get("status") not in ("pass",):
                continue  # only validate completed tasks
            for out in task.get("outputs", []):
                rel = out.get("path")
                if not rel:
                    continue
                p = workspace_dir / rel
                checked += 1
                if not p.exists():
                    missing_files.append(rel)
                elif p.stat().st_size == 0:
                    empty_files.append(rel)

    if missing_files:
        fail(f"missing claimed outputs: {missing_files}")
    if empty_files:
        fail(f"empty claimed outputs: {empty_files}")

    print(f"VALID: {checked} outputs verified across {len(manifest.get('phases', []))} phases")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_phase_outputs.py <workspace_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
