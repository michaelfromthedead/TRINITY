# test_target — Reference Library + Test Scaffolding

**Purpose:** A small reference target for METHODOLOGY_INTEGRATOR's Gate 1 (shuffle test) and Gate 3 (e2e demo). See `../spec/REFERENCE_LIBRARY.md` for the design rationale.

**Status:** SCAFFOLDED. Validators present; reference library + nexus reports are placeholders to be authored at first-engagement time (per Part 7 T-7.1's known limit on what can be pre-built without a real run).

---

## Layout

```
test_target/
├── README.md                       ← this file
├── pandas_mini/                    ← target_library (TO BE BUILT)
│   ├── __init__.py
│   ├── load.py, transform.py, aggregate.py, limit.py, sink.py
│   └── tests/test_smoke.py
├── pandas_mini_nexus/              ← nexus_reports_dir (TO BE BUILT)
│   └── {GRAVITY,GRAMMAR,VERBS,TIERS,CLASSIFICATION,GENESIS,GENERATOR,COMPRESSION}.md
├── sample_data/
│   └── small.csv                   ← e2e demo input
└── validators/                     ← workspace_manifest + decisions schema validators
    ├── validate_workspace_manifest.py
    ├── validate_decisions_schema.py
    └── validate_phase_outputs.py
```

---

## Why parts are placeholders

`pandas_mini/` and `pandas_mini_nexus/` would each be ~300-500 LOC of code + ~200KB of structured nexus reports. They are the "target library" + "upstream nexus output" inputs the workflow consumes — they are USER-OWNED, not workflow-owned.

For LANG_DEV_V2 buildout (Parts 1-9), we need:
- The **validators** (in `validators/`) — these check workspace_manifest.json, the `<library>_decisions.json` schema, and per-phase output structure. These ARE workflow-owned.
- The **sample_data/small.csv** — used by the e2e demo. Small enough to commit; ~10 rows is fine.

The reference library + nexus reports get authored during the first engagement on the test target. Until then, this directory is scaffolded but not populated.

---

## Validators (workflow-owned)

### `validators/validate_workspace_manifest.py`
Validates `workspace_dir/workspace_manifest.json` against the schema in `contracts/ARTIFACT_CATALOG.md`.

Usage: `python validators/validate_workspace_manifest.py <workspace_dir>`

### `validators/validate_decisions_schema.py`
Validates `<library>_decisions.json` against the schema in `contracts/PHASE_02_CONTRACT.md#T-02.2`.

Usage: `python validators/validate_decisions_schema.py <decisions_json_path>`

### `validators/validate_phase_outputs.py`
Cross-references `workspace_manifest.json` against actual files in workspace_dir; verifies every claimed output exists and is non-empty.

Usage: `python validators/validate_phase_outputs.py <workspace_dir>`

These validators are run as part of PHASE_QA's acceptance commands per `contracts/PHASE_<N>_CONTRACT.md`.

---

## Sample data

`sample_data/small.csv` is the canonical input for Gate 3's e2e demo. Hand-verified expected output documented in `../spec/E2E_DEMO_SPEC.md` §4-5.

---

## What's NOT here (and why)

- **No pandas_mini source code yet.** It is target-library content. Authored at first engagement.
- **No nexus reports yet.** They are upstream-pipeline output. Authored at first engagement.
- **No pre-generated `expected_workspace_manifest.json` golden file.** Only producible after a successful first run.

These are deferred to first-engagement work. Part 7's scope was workflow-owned scaffolding; reference-library content is engagement work.

---

*End of test_target README.*
