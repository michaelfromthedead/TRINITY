#!/usr/bin/env python3
"""Validate workspace_manifest.json against the schema in ARTIFACT_CATALOG.md.

Usage: python validate_workspace_manifest.py <workspace_dir>
Exit 0 = valid; exit 1 = schema violation.
"""

import json
import sys
from pathlib import Path

REQUIRED_TOP_KEYS = {
    "schema_version",
    "workflow",
    "workflow_version",
    "engagement",
    "phases",
}
REQUIRED_ENGAGEMENT_KEYS = {
    "started_at",
    "target_library",
    "nexus_reports_dir",
    "step_source_dir",
    "workspace_dir",
}
REQUIRED_PHASE_KEYS = {"phase", "name", "status", "tasks"}
REQUIRED_TASK_KEYS = {"task_id", "status", "attempts"}
ALLOWED_PHASE_STATUS = {"complete", "in_progress", "pending", "hold"}
ALLOWED_TASK_STATUS = {"pass", "fail_retry", "fail_escalate", "skip_by_design"}


def fail(msg: str) -> None:
    print(f"INVALID: {msg}", file=sys.stderr)
    sys.exit(1)


def main(workspace_dir: str) -> None:
    manifest_path = Path(workspace_dir) / "workspace_manifest.json"
    if not manifest_path.exists():
        fail(f"workspace_manifest.json not found at {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        fail(f"workspace_manifest.json invalid JSON: {e}")

    # Top-level
    missing = REQUIRED_TOP_KEYS - set(manifest.keys())
    if missing:
        fail(f"missing required top-level keys: {sorted(missing)}")

    if manifest["schema_version"] != "1.0.0":
        fail(f"schema_version expected '1.0.0', got {manifest['schema_version']!r}")

    if manifest["workflow"] != "LANG_DEV_V2":
        fail(f"workflow expected 'LANG_DEV_V2', got {manifest['workflow']!r}")

    # Engagement
    eng = manifest.get("engagement", {})
    missing = REQUIRED_ENGAGEMENT_KEYS - set(eng.keys())
    if missing:
        fail(f"missing engagement keys: {sorted(missing)}")

    # Phases
    phases = manifest.get("phases", [])
    if not isinstance(phases, list):
        fail("phases must be a list")

    for i, p in enumerate(phases):
        missing = REQUIRED_PHASE_KEYS - set(p.keys())
        if missing:
            fail(f"phases[{i}] missing keys: {sorted(missing)}")
        if p["status"] not in ALLOWED_PHASE_STATUS:
            fail(f"phases[{i}].status invalid: {p['status']!r}")

        tasks = p.get("tasks", [])
        if not isinstance(tasks, list):
            fail(f"phases[{i}].tasks must be a list")

        for j, t in enumerate(tasks):
            missing = REQUIRED_TASK_KEYS - set(t.keys())
            if missing:
                fail(f"phases[{i}].tasks[{j}] missing keys: {sorted(missing)}")
            if t["status"] not in ALLOWED_TASK_STATUS:
                fail(f"phases[{i}].tasks[{j}].status invalid: {t['status']!r}")

    # Optional methodology_integration
    mi = manifest.get("methodology_integration")
    if mi is not None:
        if "status" not in mi:
            fail("methodology_integration present but missing 'status'")
        if mi["status"] not in {"pending", "running", "green_light", "incomplete"}:
            fail(f"methodology_integration.status invalid: {mi['status']!r}")

    print(f"VALID: {manifest_path} (workflow={manifest['workflow']} v{manifest['workflow_version']}, {len(phases)} phases)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_workspace_manifest.py <workspace_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
