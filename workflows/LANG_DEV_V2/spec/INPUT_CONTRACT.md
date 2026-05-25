# INPUT_CONTRACT — LANG_DEV_V2 Required Inputs

**Purpose:** Define the four required workflow inputs and their validation rules. Fixes v1's defect of omitting nexus reports from the input contract.

**Authoritative source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PROJECT.md` §"Required inputs".

---

## 1. The four inputs

| Parameter | Required? | Type | Default | Purpose |
|---|---|---|---|---|
| `target_library` | yes | absolute path | none | The library/codebase being deconstructed |
| `nexus_reports_dir` | yes | absolute path | `<target>_nexus/` | Directory containing pre-generated nexus analysis reports for the target |
| `workspace_dir` | yes | absolute path | none | Where intermediate per-phase outputs land (created if absent) |
| `step_source_dir` | yes | absolute path | `LANG_DEV_STEPS/` at project root | Directory containing the 16+2 source methodology docs |

If any required input is missing or fails validation, the workflow ESCALATEs at PRESTEP. No phase work begins.

---

## 2. `target_library`

**What it is:** absolute path to a directory containing the source code of the library being deconstructed (e.g., `/home/user/repos/pandas`).

**Validation:**
- Path exists
- Path is a directory
- Path is readable
- Contains at least one source file (`.py`, `.rs`, `.jl`, etc.)

**ESCALATE on:** missing, not-a-directory, not-readable, empty.

**Workflow access mode:** read-only throughout. PHASE_EXECUTOR may read source files for analysis; never writes back.

**Out of scope:** the workflow does NOT analyze the library from scratch. Library analysis (producing nexus reports) is an upstream process. If you don't have nexus reports, you can't engage LANG_DEV_V2 — see `nexus_reports_dir` below.

---

## 3. `nexus_reports_dir`

**What it is:** absolute path to a directory containing pre-generated nexus analysis reports for the target library. Reports are typically `.md` files with structured tables/lists derived from a separate library-analysis pipeline.

**Convention:** named `<target_library_name>_nexus/`. Example: target `/path/to/pandas` → nexus at `/path/to/pandas_nexus/`. Default discovery: if `nexus_reports_dir` is unspecified, QUEEN looks for `<target_library>_nexus/` adjacent to `target_library` and uses it if found.

**Required reports (validated by name presence):**

| Report | What it contains | Used by |
|---|---|---|
| `GRAVITY.md` | Concept-weight ranking; which operations are "central" vs "peripheral" | T-01.1, T-02.1 |
| `GRAMMAR.md` | Compositional rules + port-type signals | T-02.1, T-02.2 |
| `VERBS.md` | All operations enumerated (the "verb" inventory) | T-01.1 |
| `TIERS.md` | Multi-tier organization (Tier 0 hardware → Tier 3 goal) | T-01.1, T-02.1 |
| `CLASSIFICATION.md` | Per-operation classification (UNIVERSAL/STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL) | T-01.1 |
| `GENESIS.md` | Generative-process analysis ("what created this domain?") | T-01.1 |
| `GENERATOR.md` | Concrete generator model | T-01.1 |
| `COMPRESSION.md` | Compression-ratio measurements at each tier | T-01.1 |

**Validation:**
- Directory exists, readable
- All 8 required report files present (filename match — content not validated at PRESTEP)
- Optional: any additional `*.md` reports are read by PHASE_EXECUTOR if referenced by source STEP docs

**ESCALATE on:** missing directory, missing any of the 8 required reports.

**Workflow access mode:** read-only throughout.

**RDC source quote (`LANGS_DEV_RDC/PROJECT.md` §"Required inputs"):**
> "**Nexus reports** (upstream, from prior library-analysis process) — at minimum GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION. Typical location: `<target>_nexus/` directory."

The RDC's "at minimum" is 5 reports. v2 enforces 8 because subsequent phases reference all 8 (per `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` §2.2 nexus-signal-to-schema-field mapping).

---

## 4. `workspace_dir`

**What it is:** absolute path to a directory where ALL intermediate per-phase outputs land. Created if absent.

**Convention:** sibling of `target_library` named `<target_library_name>_workspace/`. Example: target `/path/to/pandas` → workspace `/path/to/pandas_workspace/`.

**Validation:**
- Path is absolute
- Parent directory exists and is writable
- If path exists: must be a directory (not a file)
- If path exists and is non-empty: ESCALATE unless `--reuse-workspace` flag passed (defer this flag to a later v2.x; for now: must be empty or non-existent)

**ESCALATE on:** non-absolute path, parent not writable, exists-as-file, non-empty without reuse flag.

**Workflow access mode:** read+write. PHASE_EXECUTOR writes outputs here; PHASE_QA reads them for verification; METHODOLOGY_INTEGRATOR reads the full tree.

**Subdirectory layout** (created lazily by PHASE_EXECUTOR per task; full catalog in `contracts/ARTIFACT_CATALOG.md`):

```
workspace_dir/
├── workspace_manifest.json    ← single source of truth; updated after every PASS
├── STEP_01/, STEP_02/, STEP_03/        ← Phase 1 outputs
├── STEP_04/, STEP_05B/, STEP_06/, STEP_06_1/, STEP_06_2/   ← Phase 2 (note: STEP_05A's output goes to root as <library>_decisions.json)
├── STEP_07/, STEP_07_1/, STEP_08/, STEP_09/, STEP_10/, STEP_11/   ← Phase 3
├── BOSS_LEVEL_1/, BOSS_LEVEL_2/, BOSS_LEVEL_3/, BOSS_LEVEL_4/    ← Phase 4
├── METHODOLOGY_REPORT.md       ← terminal integration output
├── SHUFFLE_TEST_RESULTS.json
├── COMPRESSION_REPORT.md
└── E2E_DEMO_OUTPUT.md
```

The authoritative artifact `<library>_decisions.json` lands at `workspace_dir/<library>_decisions.json` (root level, not in a subdir) because it is consumed by every Phase 3 stage.

---

## 5. `step_source_dir`

**What it is:** absolute path to the directory containing the methodology source docs (the 16 STEP docs + `context.md` + `CREATOR_NOTES.md`).

**Default:** `LANG_DEV_STEPS/` at project root if unspecified at engagement.

**Required files** (validated by exact filename match):

```
STEP 1 - DECONSTRUCTION OPS.md
STEP 1.1 - RECOVER OPS.md
STEP 2 - DECONSTRUCTION OBJS.md
STEP 3 - DECONSTRUCTION TYPES.md
STEP 4 - ATOMICS.md
STEP 5 - DECISIONS SCHEMA.md           ← STEP_05A (per COURT #1)
STEP 5 - BAG GRAMMAR.md                ← STEP_05B (per COURT #1)
STEP 6 - RULESETS AND DEFAULTS.md
STEP 6.1 - THE_CONUNDRUM.md
STEP 6.2 - PRE LEXER.md
STEP 7 - LEXER.md
STEP 7.1 - VALIDATOR.md
STEP 8 - PARSER.md
STEP 9 - TYPER.md
STEP 10 - CLASSIFIER.md
STEP 11 - SOLVER.md
boss_level_1_executor_rules.md
boss_level_2_optimizer_rules.md
boss_level_3_error_reporter_rules.md
boss_level_4_debugger_rules.md
context.md
CREATOR_NOTES.md
```

**Total: 22 files.** v1 expected only the 16 STEP docs; v2 adds `context.md` (T-01.1 retrofit) and `CREATOR_NOTES.md` (vision read at engagement).

**Validation:**
- Directory exists, readable
- All 22 files present (exact filename match)
- All files non-empty

**ESCALATE on:** missing directory, missing any required file, any file empty.

**Workflow access mode:** read-only throughout. The methodology source docs are immutable.

**Hard rule (`LANG_DEV_V2_WORKFLOW.json` §hard_rules.no_step_doc_modification):** PHASE_EXECUTOR / PHASE_QA / METHODOLOGY_INTEGRATOR may read these files but MUST NOT modify them. Methodology corrections require editing source docs outside the workflow.

---

## 6. PRESTEP validation sequence

QUEEN performs all of the following in order at engagement. Any FAIL → ESCALATE with structured error report; no phase work begins.

```
1. Validate target_library
   ├─ exists? readable? directory? non-empty?
   └─ FAIL → ESCALATE("target_library invalid: <reason>")

2. Validate nexus_reports_dir
   ├─ derive from target_library if absent (try <target>_nexus/)
   ├─ exists? readable? directory?
   ├─ all 8 required reports present?
   └─ FAIL → ESCALATE("nexus_reports invalid: <reason>; required reports: GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION, GENESIS, GENERATOR, COMPRESSION")

3. Validate workspace_dir
   ├─ absolute path? parent writable?
   ├─ exists-and-empty OR non-existent?
   └─ FAIL → ESCALATE("workspace_dir invalid: <reason>")

4. Validate step_source_dir
   ├─ derive from default (LANG_DEV_STEPS/) if absent
   ├─ exists? readable? directory?
   ├─ all 22 required files present and non-empty?
   └─ FAIL → ESCALATE("step_source_dir invalid: <reason>; missing: <file list>")

5. Initialize workspace
   ├─ mkdir -p workspace_dir if needed
   ├─ write empty workspace_dir/workspace_manifest.json with schema version
   └─ FAIL → ESCALATE("workspace_dir initialization failed: <reason>")

6. Initialize INPROGRESS.md (project root)
   └─ prepend engagement entry with timestamp + all 4 input paths

7. Read at-engagement docs (QUEEN reads these into context):
   ├─ workflows/LANG_DEV_V2/LANG_DEV_V2_WORKFLOW.json (this workflow's full JSON)
   ├─ workflows/SHARED/WORKER_QUEEN.md
   ├─ workflows/SHARED/WORKER_PROTOCOL.md
   ├─ workflows/SHARED/WORKER.md
   ├─ workflows/LANG_DEV_V2/WORKER_PHASE_EXECUTOR.md
   ├─ workflows/LANG_DEV_V2/WORKER_PHASE_QA.md
   ├─ workflows/LANG_DEV_V2/WORKER_METHODOLOGY_INTEGRATOR.md
   ├─ step_source_dir/CREATOR_NOTES.md (vision context)
   ├─ workflows/LANG_DEV/LANGS_DEV_RDC/CLARIFICATION.md (philosophical framing)
   └─ FAIL → ESCALATE("at-engagement read failed: <doc>")

8. Report ready
   └─ "LANG_DEV_V2_WORKFLOW mode engaged. target=<path>, nexus=<path>, workspace=<path>, step_source=<path>. All inputs validated. Total phases: 4. Total tasks: 17 (16 mandatory + 1 on-demand). Ready to begin Phase 1. Awaiting confirmation."
```

QUEEN waits for explicit human confirmation before spawning Phase 1. PRESTEP cannot ESCALATE silently — every failure is reported with the specific problem.

---

## 7. Sample invocation

```
User: LANG_DEV_V2_WORKFLOW
QUEEN: [reads JSON + SHARED + V2 docs]
       "LANG_DEV_V2_WORKFLOW mode engaged. Inputs: target=<not specified>, nexus=<not specified>, workspace=<not specified>, step_source=<default LANG_DEV_STEPS/>. Ready. Provide target_library, nexus_reports_dir, and workspace_dir."

User: target=/home/user/repos/pandas, workspace=/home/user/repos/pandas_workspace
QUEEN: [auto-derives nexus_reports_dir = /home/user/repos/pandas_nexus]
       [PRESTEP validation runs]
       "PRESTEP: target=ok, nexus=ok (8/8 reports present), workspace=ok (created), step_source=ok (22/22 docs present). Workspace initialized at /home/user/repos/pandas_workspace/. INPROGRESS entry prepended. Ready to spawn Phase 1 (T-01.1 Deconstruction Ops). Confirm to proceed."

User: proceed
QUEEN: [spawns PHASE_EXECUTOR for T-01.1 with multi-doc reads: STEP 1 + context.md]
```

---

## 8. Open questions (to resolve in Part 2 or later)

1. **`--reuse-workspace` flag:** semantics for resuming a partially-complete run. Not in v2.0.0; defer to v2.1.
2. **Nexus report content validation:** PRESTEP only validates filename presence. Should we add schema validation? (deferred — would slow PRESTEP significantly)
3. **Multi-target invocation:** can one workflow run produce DSLs for multiple target libraries in a single invocation? (deferred to Part 1 Open Questions §5)
4. **Custom step_source_dir per task:** can a single workflow run mix STEP docs from different source directories (e.g., custom `STEP 7` for a specific domain)? Probably not — methodology consistency requires one coherent STEP source corpus per run.

---

*End of INPUT_CONTRACT.*
