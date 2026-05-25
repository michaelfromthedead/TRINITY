# ORGANIZE Workflow Family — Build-Out TODO

**Date:** 2026-04-18
**Author:** Michael (owner) + Claude (co-architect)
**Status:** COMPLETE — 2026-04-18
**Workflow being built:** ORGANIZE (single workflow, two modes: BOOTSTRAP + MAINTENANCE)
**Current state:** v1.0.0 IMPLEMENTED. All 6 parts executed. See FINAL_VERIFICATION.md for acceptance checklist.
**This file's role:** The SDLC-consumable TODO for the ORGANIZE buildout. Structured as 6 parts, each with numbered tasks + acceptance criteria + dependencies.

**Status legend:** `[x]` Done · `[ ]` Not started · `[~]` In progress · `[!]` Blocked · `[-]` Cut/deferred

---

## BUILDOUT COMPLETE — 2026-04-18

All 34 tasks across 6 parts executed successfully. See FINAL_VERIFICATION.md for acceptance checklist; ORGANIZE_WORKFLOW_DISSERTATION.md for architectural rationale; ORGANIZE_WORKFLOW.json bumped to v1.0.0 IMPLEMENTED.

---

## What already exists (inventory)

| File | Lines | Status |
|---|---|---|
| `workflows/ORGANIZE/ORGANIZE_WORKFLOW.json` | 414 | v0.1.0-DRAFT — mature spec, 8 TBDs documented |
| `workflows/ORGANIZE/WORKER_INSPECTOR.md` | 273 | Mature — BOOTSTRAP worker |
| `workflows/ORGANIZE/WORKER_TRIAGE.md` | 255 | Mature — MAINTENANCE worker |

**Unlike BOOK's buildout, worker docs are already in place.** This buildout focuses on infrastructure (schema, templates), protocol formalizations, exercise artifacts (test project + walkthroughs), TBD resolution, registry integration, and the dissertation-level explainer.

---

## Dependency map between parts

```
Part 1 — Schema Infrastructure
   ↓
Part 2 — Canonical Template Library        ── needs Part 1 (schema)
   ↓
Part 3 — Protocol Formalizations           ── parallel to Part 2 (independent)
   ↓
Part 4 — Test Project + E2E Walkthroughs   ── needs Parts 1, 2, 3
   ↓
Part 5 — Open Questions + Registry         ── needs Parts 1-4 (observations inform resolutions)
   ↓
Part 6 — Dissertation + Limitations + Final ── terminal; rolls up everything
```

**Critical path:** 1 → 2 → 4 → 5 → 6.
**Parallelizable:** Part 3 runs parallel to Part 2 (both need Part 1 only).

---

# PART 1: SCHEMA INFRASTRUCTURE

**Scope:** Elevate the `organize_json_schema` section from ORGANIZE_WORKFLOW.json to a standalone JSON Schema Draft 7 file. Define schemas for rules, templates (for future library promotion), and example config. Document schema versioning.
**Blocks:** Part 2 (template library needs template schema), Part 4 (walkthroughs reference validated config).
**Estimated pessimistic:** 1 focused session.

### Tasks

- [x] **T1.1** — Decide schema location. `workflows/ORGANIZE/schemas/` subdirectory vs flat in `workflows/ORGANIZE/`.
  - **Recommendation:** flat with clear naming (`ORGANIZE_CONFIG_SCHEMA.json`, `ORGANIZE_RULE_FORMAT.md`, etc.) — one-level deep matches BOOK's convention.
  - **Acceptance:** decision recorded; location established.

- [x] **T1.2** — Write `ORGANIZE_CONFIG_SCHEMA.json` — JSON Schema Draft 7 for `.organize.json`.
  - Elevate the `organize_json_schema` section from ORGANIZE_WORKFLOW.json §`organize_json_schema` into a proper Draft 7 schema file
  - Include: `version`, `template`, `created`, `last_run`, `runs`, `rules` (array with nested rule schema), `quarantine_log`, `pending_rule_candidates`, `ignore_paths`, `notes`
  - `$schema` field pointing to Draft 7
  - Required fields list
  - Validate rules array append-only via schema constraints where possible (or document constraint in notes)
  - **Acceptance:** schema file validates against Draft 7 meta-schema; documents all fields with types + descriptions; includes nested rule schema + quarantine_log entry schema.

- [x] **T1.3** — Write `ORGANIZE_RULE_FORMAT.md` — rule schema deep-dive.
  - Document the 3 rule kinds: `glob`, `regex`, `hint`
  - Glob syntax supported (standard `*`/`**`/`?`)
  - Regex anchoring + flags
  - Hint: natural-language rule interpretation semantics — how TRIAGE evaluates hint rules
  - Priority rules (higher = evaluated first; first match wins unless `continue-matching: true` flag)
  - Destination options: `IN_PLACE` | canonical path | `.delete` | `.archive`
  - Rule lifecycle: `active: true/false` semantics; why append-only matters
  - **Acceptance:** doc covers all rule kinds with worked examples; TRIAGE-interpretation procedure for `hint` kind is spec'd.

- [x] **T1.4** — Write `ORGANIZE_TEMPLATE_FORMAT.md` — template file schema (for Part 2's library).
  - Template file = a `.json` file under `workflows/ORGANIZE/templates/`
  - Fields: `name`, `version`, `applies_to` (what kind of project), `default_rules` (array), `default_ignore_paths`, `notes`, `inherits_from` (optional — for template composition)
  - How BOOTSTRAP consults the library: INSPECTOR detects signals → proposes template by name → QUEEN loads `<template_name>.json` if it exists in the library → seeds `.organize.json` with template defaults
  - Template versioning: SemVer per template file; manifest can pin
  - **Acceptance:** schema spec complete; example template file referenced (written in Part 2).

- [x] **T1.5** — Write `ORGANIZE_CONFIG_EXAMPLE.json` — sample `.organize.json` that validates against T1.2's schema.
  - Shows: a complete config with ~6-8 rules, 2 quarantine_log entries, 1 pending_rule_candidate
  - **Acceptance:** validates against ORGANIZE_CONFIG_SCHEMA.json; exercises all field types.

- [-] **T1.6** — Write `ORGANIZE_SCHEMA_VERSIONING.md` *(policy doc written; migration scripts deferred to future work — T1.6-FOLLOWUP)* — migration policy (addresses TBD §16.6).
  - SemVer for config schema
  - MAJOR bumps require explicit migration (script or documented manual steps)
  - MINOR/PATCH are backward-compatible
  - How QUEEN handles a config with older version on MAINTENANCE engagement (attempt auto-migrate for MINOR; ESCALATE for MAJOR)
  - **Acceptance:** policy documented; migration decision tree for version drift; resolves TBD #6.

**Part 1 verification:** A new collaborator can read the schema + format docs and write a valid `.organize.json` by hand. Example config validates.

---

# PART 2: CANONICAL TEMPLATE LIBRARY

**Scope:** Write 5 canonical template files under `workflows/ORGANIZE/templates/`. These are pre-composed rulesets for common project shapes. INSPECTOR references them by name when proposing a template during BOOTSTRAP. Resolves TBD §16.2 ("Template library promotion").
**Depends on:** Part 1 (T1.4 template schema).
**Blocks:** Part 4 (walkthroughs use templates).
**Estimated pessimistic:** 2 focused sessions.

### Tasks

- [x] **T2.1** — Create `workflows/ORGANIZE/templates/` directory + `TEMPLATES_INDEX.md` listing what's in it.
  - **Acceptance:** directory exists; index file lists all templates with 1-line descriptions.

### Initial template set (5 templates — minimum viable library)

- [x] **T2.2** — Write `templates/python-lib.json`.
  - **Scope:** Python library project with `src/` + `tests/` layout.
  - **Rules:** `src/**/*.py` → IN_PLACE; `tests/**/*.py` → IN_PLACE; `**/test_*.py` not in tests/ → MOVE_TO tests/; `docs/**/*.md` → IN_PLACE; `*.md` at root (not README/CHANGELOG/LICENSE) → docs/; `scripts/**/*.sh` → IN_PLACE; `tmp*`/`scratch*` → .delete; `notes[0-9]*.md` → .archive; etc.
  - **Ignore_paths:** `.venv/**`, `__pycache__/**`, `*.pyc`, `.pytest_cache/**`, `.mypy_cache/**`, `*.egg-info/**`, `dist/**`, `build/**`, `.git/**`
  - **Acceptance:** validates against TEMPLATE_FORMAT; minimum 8 rules; all rules grounded in python-lib conventions.

- [x] **T2.3** — Write `templates/rust-crate.json`.
  - **Scope:** Rust library crate with Cargo workspace.
  - **Rules:** `src/lib.rs` → IN_PLACE; `src/**/*.rs` → IN_PLACE; `tests/**/*.rs` → IN_PLACE; `benches/**/*.rs` → IN_PLACE; `examples/**/*.rs` → IN_PLACE; `target/**` → ignore; etc.
  - **Ignore_paths:** `target/**`, `Cargo.lock` (keep; just don't touch), `.git/**`
  - **Acceptance:** validates against TEMPLATE_FORMAT; reflects Cargo conventions.

- [x] **T2.4** — Write `templates/book-markdown.json`.
  - **Scope:** Manuscript project (BOOK workflow family target). Chapters, front/back matter, drafts, references.
  - **Rules:** `chapters/CH_*.md` → IN_PLACE; `front/*.md` → IN_PLACE; `back/*.md` → IN_PLACE; `drafts/**/*.md` → IN_PLACE; `BOOK_MANIFEST.json` → IN_PLACE; `STRUCTURE.md`/`STORYBOARD.md`/`MASTER.md`/`PEDAGOGY.md`/`EVALUATIONS.md`/`INVENTORY.md`/`INPROGRESS.md` → IN_PLACE; loose `*.md` at root → drafts/ or .archive (ASK_USER); `source/*.md` → IN_PLACE (RDC source)
  - **Ignore_paths:** `output/**` (LULU_PIPELINE output), `.git/**`
  - **Cross-references:** integrates with BOOK workflow output shapes
  - **Acceptance:** aligns with BOOK family's expected project layout.

- [x] **T2.5** — Write `templates/mixed-research.json`.
  - **Scope:** Code + prose + data coexisting in the same project (common for applied research).
  - **Rules:** `src/**/*` → IN_PLACE (code); `docs/**/*.md` → IN_PLACE (prose); `data/**/*` → IN_PLACE (data); `notebooks/**/*.ipynb` → IN_PLACE; `tmp*`/`scratch*` → .delete; loose top-level `.md` → docs/ or .archive (ASK_USER for judgment); deprecated `old*`/`backup*` directories → ASK_USER (likely .archive)
  - **Ignore_paths:** same Python-ish patterns + `.ipynb_checkpoints/**` + dataset sizes
  - **Acceptance:** handles the three-category mix without ambiguity on clear cases.

- [x] **T2.6** — Write `templates/knowledge-base.json`.
  - **Scope:** Topic-organized markdown project (wiki-shaped). No code expected.
  - **Rules:** `*/README.md` → IN_PLACE (per-topic README); `*/**/*.md` → IN_PLACE (topic tree); loose top-level `.md` (not README/INDEX/LICENSE) → ASK_USER (might belong in a topic or might be historical); `tmp*`/`scratch*` → .delete; `notes[0-9]*.md` → .archive (FLAG_NEW_RULE for pattern if many)
  - **Ignore_paths:** standard .git/**
  - **Acceptance:** handles the wiki pattern; ASK_USER-heavy by design (topic placement is authorial).

### Deferred templates (document in roadmap)

- [x] **T2.7** — Write `templates/TEMPLATES_ROADMAP.md` documenting deferred templates.
  - Future templates to write as needs emerge: `python-app`, `rust-app`, `polyglot`, `book-print`
  - Rationale for deferring: not core + no current project uses them + easier to add on-demand than pre-speculate
  - Promotion criteria: when a project uses a template that isn't in the library, promote by writing it
  - **Acceptance:** roadmap lists 4+ future templates with rationale; sets promotion criteria.

**Part 2 verification:** 5 template files exist + validate against TEMPLATE_FORMAT + can be referenced by INSPECTOR during BOOTSTRAP proposal.

---

# PART 3: PROTOCOL FORMALIZATIONS

**Scope:** Formalize the 6-stage wizard dialog, the parallel batching heuristic, the SHORT_RDC sub-procedure design, and edge-case handling. Each resolves a specific TBD from the spec.
**Depends on:** nothing (all Part 3 tasks are design work independent of Parts 1, 2).
**Blocks:** Part 4 (walkthroughs exercise these protocols).
**Estimated pessimistic:** 2 focused sessions.

### Tasks

- [x] **T3.1** — Write `WIZARD_PROTOCOL.md` — formalize the 6-stage BOOTSTRAP dialog.
  - Reference ORGANIZE_WORKFLOW.json §`wizard_flow`
  - For each stage (present_detected_signals → propose_template → propose_seed_rules → propose_ignore_paths → review_initial_classification → cascade_decision):
    - Exact user-facing prompt format
    - Acceptable user responses (accept / edit / reject / request more info / abort)
    - QUEEN behavior per response
    - Transition rules (when do we advance, when do we loop back)
  - Edge case: user edits a proposal section — how is the edit captured, validated, applied?
  - Edge case: user aborts mid-wizard — partial state handling (no `.organize.json` written)
  - Edge case: user requests INSPECTOR re-run with narrower focus — semantics (currently TBD in v0.1.0-DRAFT)
  - **Acceptance:** each stage has prompt template + response vocabulary + transition logic; 3+ edge cases spec'd.

- [x] **T3.2** — Write `BATCHING_HEURISTIC.md` — formalize parallel-safe batching (resolves TBD §16.4).
  - Problem: QUEEN must split loose files into batches such that each batch can be classified independently by a TRIAGE worker without cross-batch context.
  - **Heuristic (v0.1.0-DRAFT proposal):** "disjoint directory subtrees + cap per-batch file count (≤30 files per batch)"
  - Formalize:
    - Algorithm for splitting: depth-first traversal, batch accumulates files until cap hit or subtree boundary reached
    - Per-batch cap: start at 30; may tune after observation
    - Cross-batch dependencies: explicitly document that rules are evaluated per-file independently, so FLAG_NEW_RULE detection (3+ pattern matches) is per-batch; QUEEN may re-aggregate FLAG_NEW_RULE candidates across batches after wave returns
    - Root-invariant filtering: happens BEFORE batching (per ORGANIZE_WORKFLOW.json §`root_invariants.enforcement`)
    - Ignore-path filtering: happens BEFORE batching
  - **Acceptance:** algorithm pseudocode; cap value documented; cross-batch aggregation procedure for FLAG_NEW_RULE defined.

- [-] **T3.3** — Write `SHORT_RDC_SPEC.md` *(spec written with deferral decision; implementation deferred to future work — T3.3-FOLLOWUP)* — design the compressed RDC-style relevance check (resolves TBD §16.1).
  - Context: when TRIAGE is classifying a prose file and needs to decide QUARANTINE:.archive vs KEEP_IN_PLACE vs MOVE_TO:, sometimes "is this prose still relevant to the project?" is the right question and TRIAGE's own full-read judgment isn't enough.
  - Decision: does SHORT_RDC exist as a distinct sub-procedure, or does TRIAGE handle it all?
  - **Proposal:** For v0.1.0-DRAFT, TRIAGE uses full-read judgment + ASK_USER for ambiguity. SHORT_RDC is deferred as a future enhancement: when we have > N false-positive archival decisions, introduce SHORT_RDC as a separate lightweight worker that compares the prose file's concepts against the project's current MASTER.md (if RDC has been run) or against the README + top-level architecture docs.
  - **Acceptance:** decision documented; if deferred, explicit deferral with rationale; if designed now, spec includes: input, output, invocation condition, example verdicts.

- [x] **T3.4** — Write `RATIFICATION_UI_SPEC.md` — how user interacts with the proposed-moves plan.
  - QUEEN presents TRIAGE_WAVE aggregated plan to user grouped by verdict type
  - Format: per-verdict-group counts + preview of first N files + options
  - User options per group: `ratify all` / `ratify-by-file` (loop through individually) / `skip group` / `edit destination` (for MOVE_TO) / `abort circuit`
  - Terminal-friendly text format for the dialog (ORGANIZE operates in Claude conversation, which is text)
  - Escape hatch: at any prompt, `abort` stops the circuit cleanly; ratified moves so far are committed; unratified ones left alone
  - **Acceptance:** UI format documented with concrete text templates; all user actions have explicit handling.

- [x] **T3.5** — Write `EDGE_CASES.md` — document how the workflow handles unusual situations.
  - Binary files in batch → TRIAGE emits ASK_USER per spec; QUEEN propagates to user
  - Files too large for TRIAGE read budget → same, ASK_USER
  - Files matching multiple rules with same priority → first-match-wins (document as design choice); consider tie-breaking note
  - User aborts mid-circuit → state as of last commit preserved; remaining moves untouched
  - Git mv on untracked file → falls back to plain mv; logged in quarantine_log as `untracked_at_move: true` (per ORGANIZE_WORKFLOW.json §`hard_rules.git_mv_only`)
  - Circular quarantine (file in `.delete/` referenced from elsewhere) → ORGANIZE doesn't enumerate `.delete/` on subsequent runs (ignored)
  - Pre-commit hook fails on `organize:` commit → NEVER `--no-verify` (workflow hard rule); user must fix underlying issue
  - `.organize.json` manually edited by user between runs → QUEEN validates against schema on load; if invalid, ESCALATE with specific field errors
  - **Acceptance:** 8+ edge cases with explicit handling.

**Part 3 verification:** Wizard, batching, and SHORT_RDC all have formal specs. Edge-case handling is explicit.

---

# PART 4: TEST PROJECT + E2E WALKTHROUGHS

**Scope:** Create a minimal test project with deliberately-messy structure, then mental-trace the workflow through BOOTSTRAP and MAINTENANCE scenarios including edge cases.
**Depends on:** Parts 1, 2, 3 complete.
**Blocks:** Part 5 (observations inform TBD resolutions), Part 6 (dissertation cites walkthroughs).
**Estimated pessimistic:** 2 focused sessions.

### Tasks

- [x] **T4.1** — Create `workflows/ORGANIZE/test_project/` directory structure.
  - Minimal realistic project with ~15-20 files, mix of clean + messy
  - Structure should be classifiable as `mixed-research` template (to exercise complexity)
  - Include:
    - Valid canonical files: `README.md`, `src/main.py`, `src/utils/helpers.py`, `tests/test_main.py`, `docs/overview.md`
    - Messy files at root: `scratch.py`, `notes2.md`, `old_design.md`, `tmp_data.csv`, `untitled.md`
    - Cruft directories: `backup/old_code.py`, `tmp/experiment.py`
    - Mock root invariants: `.claude/` placeholder (empty), `.claude-flow/` placeholder, `.hive-mind/` placeholder, `.mcp.json` placeholder
    - Intentionally absent: `.organize.json` (so BOOTSTRAP runs cleanly first)
  - **Acceptance:** directory created; files added; realistic but small.

- [x] **T4.2** — Write `BOOTSTRAP_WALKTHROUGH.md` — mental trace of first-run.
  - User invokes `ORGANIZE_WORKFLOW` in `test_project/` directory
  - QUEEN detects no `.organize.json`; BOOTSTRAP mode
  - QUEEN spawns INSPECTOR
  - Trace INSPECTOR's Phase 1 (orientation discovery), Phase 2 (signal detection), Phase 3 (template proposal = `mixed-research`), Phase 4 (seed rules), Phase 5 (ignore_paths), Phase 6 (initial classification preview), Phase 7 (open questions)
  - QUEEN enters WIZARD_LOOP; presents each stage; user accepts/edits
  - `.organize.json` written
  - Cascade decision: user approves 1 circuit cascade
  - → feeds into T4.3 MAINTENANCE trace
  - **Acceptance:** complete end-to-end mental trace; shows INSPECTOR proposal block; shows wizard user-dialog; resulting config shown.

- [x] **T4.3** — Write `MAINTENANCE_WALKTHROUGH.md` — single-circuit mental trace.
  - Continuing from T4.2's cascade, or independently on a project with existing `.organize.json`
  - QUEEN loads config + plans circuit
  - QUEEN enumerates loose files (respecting ignore_paths + root_invariants)
  - QUEEN batches into 2-3 parallel batches (given small project size)
  - Trace TRIAGE_WAVE: 2-3 TRIAGE workers spawned in parallel; each returns verdicts
  - QUEEN aggregates; groups by verdict type
  - Presents plan to user; user ratifies
  - `git mv` executed per move; `.delete/` + `.archive/` lazily created; .gitignore updated
  - `.organize.json` updated (quarantine_log, runs counter)
  - Commit: `organize: circuit 1 of 1 — moved 5 files, quarantined 3`
  - Verdict: CLEAN_RUN
  - **Acceptance:** full circuit traced; verdicts from TRIAGE shown with realistic rationales; ratification dialog shown; final git commit shown.

- [x] **T4.4** — Write `MULTI_CIRCUIT_WALKTHROUGH.md` — N=3 circuit trace.
  - Scenario: larger test project with more cruft; user requests 3 circuits
  - First circuit: many moves, some FLAG_NEW_RULE candidates surfaced
  - User ratifies new rules before circuit 2
  - Second circuit: catches files that pattern-matched the new rules
  - Third circuit: emits NOTHING_TO_DO (project now fully organized)
  - Commit history: 3 `organize: circuit N of 3 — ...` commits + 1 metadata update commit
  - **Acceptance:** convergence demonstrated; FLAG_NEW_RULE → ratification → application cycle shown; final NOTHING_TO_DO verdict.

- [x] **T4.5** — Write `EDGE_CASE_WALKTHROUGHS.md` — unusual scenarios.
  - Scenario A: binary file in batch (TRIAGE → ASK_USER, QUEEN escalates to user)
  - Scenario B: file matches two rules with different priorities (first-match-wins)
  - Scenario C: root-invariant path (`.claude/`) slips past QUEEN's filter (shouldn't happen but defensive) → TRIAGE refuses to classify, emits ASK_USER with bug note
  - Scenario D: user aborts mid-wizard → no `.organize.json` written, no state changes
  - Scenario E: user aborts after circuit 2 of 3 → PARTIAL_RUN verdict, 2 circuits committed, 3rd not attempted
  - Scenario F: `.organize.json` manually edited, becomes invalid → QUEEN ESCALATEs with schema error messages
  - **Acceptance:** 6 edge-case scenarios with expected behavior + user-facing messages.

- [x] **T4.6** — Smoke-test: walkthroughs self-consistent.
  - Review T4.2-T4.5 for contradictions
  - Ensure verdict vocabulary, ratification format, config schema references all match Parts 1-3 specs
  - **Acceptance:** consistency report; any issues fixed in place.

**Part 4 verification:** Four walkthroughs cover BOOTSTRAP, single-circuit MAINTENANCE, multi-circuit, and 6 edge cases. A new user can understand the full workflow by reading these.

---

# PART 5: OPEN QUESTIONS + REGISTRY INTEGRATION

**Scope:** Resolve all 8 TBDs from ORGANIZE_WORKFLOW.json §`known_tbds` (some already covered by Parts 1-3; others decided here). Update CLAUDE_APPENDIX + WORKER.md master index. Run install.sh to regenerate project CLAUDE.md.
**Depends on:** Parts 1-4 (resolutions often reference work from those parts).
**Blocks:** Part 6 (dissertation cites resolved TBDs).
**Estimated pessimistic:** 1 focused session.

### Tasks

- [x] **T5.1** — Write `OPEN_QUESTIONS_RESOLVED.md` — all 8 TBDs.

  | TBD | Topic | Resolution approach |
  |---|---|---|
  | #1 | SHORT_RDC shape | RESOLVED_ELSEWHERE (T3.3) — deferred with explicit future criteria |
  | #2 | Template library promotion | RESOLVED_ELSEWHERE (Parts 1-2) — 5 templates in library + ROADMAP |
  | #3 | Rule language (glob/regex/hint) | RESOLVED (T1.3) — all three supported per `rule.kind`; TRIAGE interpretation spec'd |
  | #4 | Parallel batching heuristic | RESOLVED_ELSEWHERE (T3.2) — disjoint subtrees + ≤30 cap |
  | #5 | FLAG_NEW_RULE threshold | RESOLVED (already in spec) — 3+ occurrences of same pattern per batch; cross-batch aggregation per T3.2 |
  | #6 | Config schema versioning migration | RESOLVED_ELSEWHERE (T1.6) |
  | #7 | Blanket-approval rules | DEFERRED — v0.1.0 is conservative; revisit after N runs with observational data |
  | #8 | Re-running wizard semantics | RESOLVED — mode_override preserves rules (append-only invariant); wizard diffs proposed-vs-existing and user ratifies diff; document in WIZARD_PROTOCOL §edge-cases |

  - **Acceptance:** all 8 TBDs have explicit status + reference + rationale; DEFERRED items have criteria for future resolution.

- [x] **T5.2** — Update `workflows/CLAUDE_APPENDIX.md` with ORGANIZE workflow.
  - Add ORGANIZE_WORKFLOW row to the workflows table
  - Add engagement behavior for ORGANIZE_WORKFLOW (natural-language triggers)
  - Add exit behavior for ORGANIZE_WORKFLOW
  - Add file artifacts: `.organize.json`, `.delete/`, `.archive/`, `ORGANIZE/*.md` docs
  - Add hard rules: never-delete, git-mv-only, append-only, no-auto-trigger, no-workflow-mixing, root-invariants-untouchable
  - References: ORGANIZE_WORKFLOW.json, WORKER_INSPECTOR, WORKER_TRIAGE
  - **Acceptance:** CLAUDE_APPENDIX updated; BOOK-family + RDC + RECON + SDLC + ORGANIZE all represented (5 workflow families).

- [x] **T5.3** — Update `workflows/SHARED/WORKER.md` master index with ORGANIZE.
  - Add ORGANIZE_WORKFLOW as 5th active workflow
  - Document roles: QUEEN, INSPECTOR (BOOTSTRAP-only), TRIAGE (MAINTENANCE parallel wave)
  - Document modes: BOOTSTRAP, MAINTENANCE
  - Document verdicts: bootstrap catalog + maintenance catalog
  - **Acceptance:** master index reflects ORGANIZE at same altitude as other workflows.

- [x] **T5.4** — Run `bash workflows/install.sh appendix` to regenerate project CLAUDE.md.
  - Verify: regenerated CLAUDE.md includes ORGANIZE sections
  - Verify: BEGIN/END markers preserved
  - **Acceptance:** install.sh reports PASS; CLAUDE.md contains all 5 workflow families.

- [x] **T5.5** — Write `workflows/ORGANIZE/README.md` — entry-point doc.
  - What ORGANIZE is (1 paragraph)
  - Quickstart: "I want to organize my project"
    1. Run `ORGANIZE_WORKFLOW` in the project's root directory
    2. QUEEN detects BOOTSTRAP vs MAINTENANCE
    3. Follow the wizard (BOOTSTRAP) or review moves (MAINTENANCE)
    4. Ratify; QUEEN commits
  - Architecture: 2 modes, 2 workers, quarantine model
  - Key docs: dissertation, TODO, workflow JSON, worker docs
  - **Acceptance:** README exists; maps to authoritative docs.

**Part 5 verification:** All 8 TBDs have explicit resolution or deferral; ORGANIZE is visible in project's workflow registry (CLAUDE.md, WORKER.md).

---

# PART 6: DISSERTATION + KNOWN LIMITATIONS + FINAL

**Scope:** Write the architectural-rationale dissertation (equivalent to BOOK_WORKFLOW_DISSERTATION.md). Document known limitations from walkthroughs. Bump workflow JSON from v0.1.0-DRAFT to v1.0.0 IMPLEMENTED. Final verification.
**Depends on:** Parts 1-5 (dissertation synthesizes everything).
**Blocks:** nothing (terminal part).
**Estimated pessimistic:** 1 focused session.

### Tasks

- [x] **T6.1** — Write `workflows/ORGANIZE/ORGANIZE_WORKFLOW_DISSERTATION.md`.
  - **Inspired by:** `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` structure
  - **Sections:**
    1. Purpose and Scope
    2. Two Modes (BOOTSTRAP, MAINTENANCE) — why split
    3. Architecture: QUEEN orchestrator + INSPECTOR (BOOTSTRAP-only) + TRIAGE (MAINTENANCE parallel)
    4. The `.organize.json` Config (per-project, append-only rules, audit trail)
    5. The Quarantine Model (`.delete/` + `.archive/` — never delete, always reversible)
    6. Root Invariants (untouchable paths, defensive design)
    7. Parallel Batching (TRIAGE_WAVE semantics)
    8. The Wizard (human-in-the-loop for BOOTSTRAP)
    9. Rule Language (glob + regex + hint; priority ordering; append-only)
    10. Verdict Catalogs (BOOTSTRAP + MAINTENANCE)
    11. Relationship to Other Workflows (parallel to RDC/RECON; downstream-optional to RECON)
    12. Hard Rules (never_delete, git_mv_only, no_auto_ratify, etc.)
    13. Circuit Bounding (no loops, N-bounded per invocation)
    14. Implementation Learnings (from Parts 1-5 buildout observations)
    15. Open Questions — Resolved (reference OPEN_QUESTIONS_RESOLVED.md)
    16. Future Work
  - **Acceptance:** dissertation complete; captures architectural rationale; cross-references all other artifacts.

- [x] **T6.2** — Write `workflows/ORGANIZE/KNOWN_LIMITATIONS.md`.
  - Observations from T4 walkthroughs on where the workflow's design produces rough edges
  - Categories: Blocking (design fix needed before production), High, Medium, Low
  - Candidates:
    - Binary file handling currently ASK_USER every time; could be coalesced into a single user dialog
    - SHORT_RDC absence means TRIAGE may make weak archival decisions on prose files
    - No blanket-approval (v0.1.0 conservative by design; flagged as enhancement)
    - Template library is minimal (5 templates); projects outside those shapes use `custom` mode
    - Cross-batch FLAG_NEW_RULE aggregation adds complexity vs strictly per-batch
  - **Acceptance:** 5+ limitations with severity + recommended follow-up.

- [x] **T6.3** — Bump `ORGANIZE_WORKFLOW.json` version.
  - v0.1.0-DRAFT → v1.0.0 (drop DRAFT suffix; mark as implemented)
  - Update `changelog` with v1.0.0 entry dated today
  - Update `notes.draft_status_note` → replaced with `notes.implementation_note` pointing to dissertation + TODO
  - **Acceptance:** JSON validates; version = 1.0.0; changelog entry added.

- [x] **T6.4** — Final verification sweep.
  - All task checkboxes in this TODO marked `[x]` or explicitly `[-]` with rationale
  - All file paths referenced in ORGANIZE workflow JSON resolve
  - All worker docs' required-reading paths resolve
  - Sample `.organize.json` (ORGANIZE_CONFIG_EXAMPLE.json) still validates against schema
  - WIZARD_PROTOCOL + BATCHING_HEURISTIC + EDGE_CASES all cited where appropriate
  - Dissertation cites all major artifacts
  - **Acceptance:** sweep document confirms all 6 parts' deliverables exist + are internally consistent.

- [x] **T6.5** — Update `workflows/ORGANIZE/ORGANIZE_BUILDOUT_TODO.md` (this file) with completion status.
  - Mark all tasks complete
  - Add summary stats at bottom
  - **Acceptance:** this file reflects final state.

**Part 6 verification:** ORGANIZE workflow is production-ready. Dissertation explains the design. KNOWN_LIMITATIONS documents where v1.0.0 falls short. JSON is at v1.0.0.

---

# SUMMARY STATS

| Part | Name | Tasks | Critical Path? | Blocks |
|---|---|---|---|---|
| 1 | Schema Infrastructure | 6 | YES | 2, 4 |
| 2 | Canonical Template Library | 7 | YES | 4 |
| 3 | Protocol Formalizations | 5 | no (parallel to 2) | 4 |
| 4 | Test Project + Walkthroughs | 6 | YES | 5 |
| 5 | Open Questions + Registry | 5 | YES | 6 |
| 6 | Dissertation + Limitations + Final | 5 | YES | (terminal) |
| **Total** | **6 parts** | **34 tasks** | | |

**Estimated pessimistic effort:** ~8–12 focused sessions (smaller than BOOK's ~32-42; ORGANIZE's narrower scope + existing mature worker docs).

---

# ACCEPTANCE CRITERIA FOR "ORGANIZE WORKFLOW FAMILY COMPLETE"

1. All 34 tasks marked `[x]` or explicitly `[-]` (cut/deferred with rationale).
2. `workflows/ORGANIZE/ORGANIZE_WORKFLOW_DISSERTATION.md` exists; matches structural quality of BOOK equivalent.
3. `ORGANIZE_WORKFLOW.json` bumped v0.1.0-DRAFT → **v1.0.0 IMPLEMENTED**.
4. `workflows/CLAUDE_APPENDIX.md` lists ORGANIZE as 5th workflow family; install.sh re-run; project CLAUDE.md reflects.
5. `workflows/SHARED/WORKER.md` master index lists ORGANIZE roles (INSPECTOR, TRIAGE) under ORGANIZE_WORKFLOW.
6. `workflows/ORGANIZE/templates/` contains 5 canonical templates + ROADMAP for future.
7. `.organize.json` sample (ORGANIZE_CONFIG_EXAMPLE.json) validates against ORGANIZE_CONFIG_SCHEMA.json.
8. All 8 dissertation TBDs have explicit RESOLVED / RESOLVED_ELSEWHERE / DEFERRED status in OPEN_QUESTIONS_RESOLVED.md.
9. BOOTSTRAP + MAINTENANCE + multi-circuit + edge-case walkthroughs all present and internally consistent.
10. `workflows/ORGANIZE/README.md` exists and accurately reflects the implemented system.

---

*End of ORGANIZE Buildout TODO.*
