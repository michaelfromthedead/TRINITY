# ORGANIZE Schema Versioning and Migration Policy

**Task:** T1.6
**Date:** 2026-04-18
**Resolves:** TBD #6 from ORGANIZE_WORKFLOW.json §`known_tbds` — "Config schema versioning — when we bump version, how do we migrate existing .organize.json files?"
**Status:** RESOLVED — policy defined; migration scripts deferred as T1.6-FOLLOWUP.

---

## 1. Scope

This document covers versioning and migration for **`.organize.json` config schema** — the file written by ORGANIZE_WORKFLOW to each target project. It does not cover:

- ORGANIZE_WORKFLOW.json itself (workflow spec versioning is tracked in its own `version` field and `changelog` array)
- Template file versioning (see ORGANIZE_TEMPLATE_FORMAT.md §3.2)
- Worker doc versioning (workers are not versioned separately; they follow the workflow version)

---

## 2. Version format

The `version` field in `.organize.json` follows **SemVer**: `MAJOR.MINOR.PATCH`.

```json
{ "version": "0.1.0" }
```

During the DRAFT period, `-DRAFT` suffix is appended: `"0.1.0-DRAFT"`. The suffix is dropped when the schema is declared stable (at `1.0.0`).

| Component | Meaning |
|---|---|
| MAJOR | Breaking change — existing `.organize.json` files cannot be loaded by the new code without migration |
| MINOR | Backward-compatible addition — new optional fields added; existing configs still valid |
| PATCH | Backward-compatible fix — description changes, constraint tightening that doesn't affect valid configs |

---

## 3. What constitutes each bump type

### 3.1 MAJOR bump (breaking — requires migration)

A MAJOR bump is required when any of the following occur:

- A **required field is added** (existing configs lack it → validation fails)
- A **field is renamed** (existing configs use old name → validation fails or misroutes)
- A **field type changes** (e.g., `runs` changes from integer to object)
- A **field is removed** (code no longer reads it; data is lost without migration)
- The **`rules` object structure changes** in a non-backward-compatible way (e.g., `kind` enum loses a value, `destination` format changes)
- The **`quarantine_log` entry structure changes** in a breaking way

Current expected MAJOR bump scenarios as the workflow matures:

- Adding `rule.continue_matching` as a required field (would break existing rules lacking it)
- Changing `quarantine_log[].destination` from `".delete" | ".archive"` to a richer object type
- Splitting `notes` into a structured field

### 3.2 MINOR bump (additive — backward-compatible)

- A **new optional field is added** (existing configs remain valid; field defaults to absent/null)
- A **new enum value is added** to `template` (existing configs using old values remain valid)
- A **new enum value is added** to `quarantine_log[].ratified_by` (e.g., adding `"rule"` for future auto-ratify)
- A **new optional field is added to rules** (e.g., `continue_matching: boolean`, optional)

Example: adding `rule.continue_matching` as an **optional** field (not required) is MINOR.

### 3.3 PATCH bump

- Updating `description` text in the JSON Schema without changing validation behavior
- Tightening a `pattern` regex that only affects invalid inputs (existing valid configs unaffected)
- Adding `examples` to the schema
- Documentation-only changes

---

## 4. QUEEN's behavior on version drift at MAINTENANCE engagement

When QUEEN loads `.organize.json` at the start of a MAINTENANCE engagement, it compares the config's `version` field against the current schema version the workflow understands.

### 4.1 Decision tree

```
Loaded config version  →  QUEEN's schema version
─────────────────────────────────────────────────────────────────

SAME version
  → Proceed normally. No version note to user.

OLDER MINOR or PATCH (same MAJOR)
  e.g., config=0.1.0, schema=0.2.0
  → LOAD AND PROCEED with compatibility warning.
  → QUEEN notes to user: "Config schema is v0.1.0; current schema is v0.2.0
    (MINOR update, backward-compatible). New optional fields will default to
    absent. Consider re-running BOOTSTRAP with mode_override to pick up new
    defaults."
  → QUEEN does NOT block MAINTENANCE. Proceeds with the loaded config.
  → When updating .organize.json at end of circuit, QUEEN writes the new
    version number and any new optional fields with their defaults.

OLDER MAJOR (MAJOR < current)
  e.g., config=0.x.y, schema=1.0.0
  → ESCALATE. Do not proceed.
  → QUEEN reports: "Config schema version mismatch (MAJOR): .organize.json
    is v<old>; current schema requires v<new>. Automatic migration not
    available. See workflows/ORGANIZE/migrations/<old>_to_<new>.py or
    ORGANIZE_SCHEMA_VERSIONING.md §6 for manual steps. MAINTENANCE halted."
  → User must migrate before MAINTENANCE can run.

NEWER version than QUEEN understands
  e.g., config=2.0.0, schema=1.0.0 (user manually upgraded or used newer Claude)
  → ESCALATE. Do not proceed.
  → QUEEN reports: "Config schema version v<new> is newer than this workflow
    version's supported schema v<current>. Update the ORGANIZE workflow to
    a version that supports v<new>, or contact the project owner.
    MAINTENANCE halted."

DRAFT suffix
  e.g., config="0.1.0-DRAFT"
  → Treat as version 0.1.0 for comparison purposes.
  → If current schema is also 0.1.0 (or 0.1.0-DRAFT): proceed normally.
  → Note to user: "Config was created with a DRAFT schema version. Current
    schema is v<current>. Proceeding." Apply normal MINOR/MAJOR rules above
    based on numeric components.
```

### 4.2 Forward-compatibility (NEWER config version)

QUEEN cannot reliably load a config whose schema is NEWER than what it understands — future fields may have changed semantics in ways QUEEN cannot predict. The safe action is ESCALATE, not attempt to parse.

### 4.3 Version stored in updated config

After a successful MAINTENANCE circuit, QUEEN writes the **current schema version** into the config's `version` field (if upgrading from an older MINOR/PATCH). This means a config "self-upgrades" to the current version after each run under a newer QUEEN.

---

## 5. Schema version bump procedure

When a developer bumps the schema version (i.e., modifies `ORGANIZE_CONFIG_SCHEMA.json`):

1. **Determine bump type** using the criteria in §3.
2. **Update `ORGANIZE_CONFIG_SCHEMA.json`:** change the schema content and update the `$id` or `description` to reflect the new version. (The schema file itself does not contain its own version number — the version is tracked in `.organize.json` files that conform to it. The changelog in this document is the schema's version history.)
3. **Update `ORGANIZE_CONFIG_EXAMPLE.json`:** update its `version` field to the new version. Ensure it still validates against the new schema.
4. **If MAJOR bump:** document the breaking changes in §7 (changelog) and create a migration spec in §6.
5. **Update ORGANIZE_WORKFLOW.json:** update its `version` field and prepend a changelog entry.
6. **Run validation:** `python3 -c "import json, jsonschema; jsonschema.validate(json.load(open('ORGANIZE_CONFIG_EXAMPLE.json')), json.load(open('ORGANIZE_CONFIG_SCHEMA.json')))"` — must pass.
7. **Commit with message:** `organize: bump config schema to vX.Y.Z — <summary of change>`

---

## 6. Migration design

### 6.1 Migration script location

Migration scripts live at:

```
workflows/ORGANIZE/migrations/<from_version>_to_<to_version>.py
```

Examples:

```
workflows/ORGANIZE/migrations/0.1.0_to_1.0.0.py
workflows/ORGANIZE/migrations/1.0.0_to_2.0.0.py
```

### 6.2 Migration script interface (specification)

Each migration script is a Python script with the following interface:

```python
#!/usr/bin/env python3
"""
Migration: .organize.json v<FROM> -> v<TO>
Breaking changes addressed:
  - <list each breaking change>
"""

import json
import sys
from pathlib import Path

FROM_VERSION = "<from>"
TO_VERSION = "<to>"


def migrate(config: dict) -> dict:
    """
    Transform a v<FROM> .organize.json dict to v<TO> format.
    Returns the migrated dict. Does not write to disk.
    Raises ValueError if input is not the expected FROM version.
    """
    ...


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_.organize.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    with open(path) as f:
        config = json.load(f)

    migrated = migrate(config)

    backup_path = path.with_suffix(".json.bak")
    path.rename(backup_path)
    print(f"Backed up original to {backup_path}")

    with open(path, "w") as f:
        json.dump(migrated, f, indent=2)
    print(f"Migrated {path} from v{FROM_VERSION} to v{TO_VERSION}")


if __name__ == "__main__":
    main()
```

**Invariants for migration scripts:**
- Migration is idempotent: running it twice on an already-migrated file must be safe (check `version` field at top of `migrate()`).
- Original file is backed up before overwrite.
- Script writes the new version string into the output dict.
- Script never deletes quarantine_log entries (append-only invariant preserved across migrations).
- Script never removes rules (append-only invariant preserved).
- Script is standalone — no external dependencies beyond Python stdlib.

### 6.3 T1.6-FOLLOWUP: migration scripts are deferred

**Migration scripts are NOT implemented in v0.1.0-DRAFT.** They are deferred because:

1. No MAJOR version bump has occurred yet — `0.1.0-DRAFT` is the only version.
2. No real `.organize.json` files exist in production that would need migration.
3. The first MAJOR bump should drive the writing of the first migration script — writing one speculatively would be premature.

**Criteria for when to implement:**
- A MAJOR schema bump is imminent or has occurred.
- At least one real project has an existing `.organize.json` that needs migrating.
- The specific breaking changes are known (so the migration can be tested against real data).

**T1.6-FOLLOWUP tag:** Any team member seeing this tag should check whether a MAJOR bump has occurred since this document was written and, if so, implement `workflows/ORGANIZE/migrations/<from>_to_<to>.py` following the interface spec in §6.2.

---

## 7. Schema version changelog

| Version | Date | Type | Summary |
|---|---|---|---|
| `0.1.0-DRAFT` | 2026-04-18 | Initial | Initial schema definition. All fields defined from ORGANIZE_WORKFLOW.json §`organize_json_schema`. Draft status — subject to revision after first live runs. |

*Future entries prepended above in reverse chronological order.*

---

## 8. Cross-reference

| Topic | Document |
|---|---|
| Schema field definitions | `ORGANIZE_CONFIG_SCHEMA.json` |
| Example config with version field | `ORGANIZE_CONFIG_EXAMPLE.json` |
| Rule format (affected by rule schema changes) | `ORGANIZE_RULE_FORMAT.md` |
| QUEEN's MAINTENANCE engagement sequence | `ORGANIZE_WORKFLOW.json §flow.maintenance` |
| TBD #6 original text | `ORGANIZE_WORKFLOW.json §known_tbds` item 6 |
| Migration script location | `workflows/ORGANIZE/migrations/` (directory created when first script is written) |

---

*End of ORGANIZE_SCHEMA_VERSIONING.md.*
