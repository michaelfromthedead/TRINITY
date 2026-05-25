# Swarm Worker — Master Index

**Purpose:** Entry point when QUEEN spawns a swarm worker (via `Agent` tool). Read this, then the shared protocol, then your role-specific doc for the active workflow.

**Scope:** JARVIS project has multiple workflows. Each workflow defines its own role set; this index routes to the right one.

**Canonical references:**
- `workflows/SHARED/WORKER_PROTOCOL.md` — rules every worker follows, regardless of workflow
- `workflows/SHARED/WORKER_QUEEN.md` — orchestrator role (Claude in main conversation when a workflow is active)
- `workflows/SDLC/SDLC_WORKFLOW.json` — SDLC workflow state machine
- `workflows/RDC/RDC_WORKFLOW.json` — RDC workflow state machine
- `workflows/BOOK/BOOK_TRIAGE.json` — BOOK_TRIAGE state machine
- `workflows/BOOK/BOOK_CONSOLIDATION.json` — BOOK_CONSOLIDATION state machine
- `workflows/BOOK/BOOK_COMPLETION.json` — BOOK_COMPLETION state machine
- `workflows/BOOK/BOOK_STORYBOARD.json` — BOOK_STORYBOARD state machine
- `workflows/BOOK/BOOK_EDITORIAL.json` — BOOK_EDITORIAL state machine
- `workflows/BOOK/BOOK_PRODUCTION.json` — BOOK_PRODUCTION state machine
- `workflows/ORGANIZE/ORGANIZE_WORKFLOW.json` — ORGANIZE_WORKFLOW state machine
- `workflows/APP_INT/APP_INT_WORKFLOW.json` — APP_INT_WORKFLOW state machine
- `workflows/README.md` — deployment guide, two-layer enforcement model
- `workflows/ci-python.sh`, `workflows/ci-rust.sh`, `workflows/install-hooks.sh` — CI/CD Layer 1 (language-level enforcement under your commits)

---

## Active workflows

### APP_INT_WORKFLOW — Layer 3 Application Integration

**Purpose:** Drive the three-phase integration of curated apps into Fantasy OS: P1 (template theming), P2 (source fork + behavioral unification), P3 (deep ownership: libfantasy rebuild, SQLite unification, live reload). Input spec from `apps/docs/APP-TASKS.md` and `apps/docs/PHASE-PATTERNS.md`. Source work in `src/<app_id>/`.

**Trigger phrase:** `APP_INT_WORKFLOW` — or natural-language invocation (e.g., "integrate celluloid", "start phase 1 on celluloid", "do P2 for cherrytree")

**Roles:**

| Role | Doc | Active when |
|---|---|---|
| SURVEYOR (read source; map checklist items to file:line; flag toolkit/category mismatches) | `workflows/APP_INT/WORKER_SURVEYOR.md` | Start of every phase |
| SQLITE_EVAL (full evaluation of app data stores; categorize NATIVE_SQLITE / DCONF_BRIDGE / XML_OR_OTHER / NO_DATA; produce P2/P3 action plan) | `workflows/APP_INT/WORKER_SQLITE_EVAL.md` | P1 only |
| THEME_VERIFY (universal toolkit-agnostic theme verification; covers all 11 categories; create IRIS Tera templates if needed; confirm reload) | `workflows/APP_INT/WORKER_THEME_VERIFY.md` | P1 only |
| PATCHER (apply P2 source patches; keybinds, profile awareness, dconf-sqlite, menus) | `workflows/APP_INT/WORKER_PATCHER.md` | P2 only |
| PORTER (apply P3 patches; libfantasy swap, live reload, unified search registration) | `workflows/APP_INT/WORKER_PATCHER.md` | P3 only |
| QA_CHECKLIST (per-item checklist audit; build verification; phase verdict) | `workflows/APP_INT/WORKER_QA_CHECKLIST.md` | End of every phase |

**Phase sequence:** P1 → P2 → P3 (each requires prior GREEN_LIGHT; can stop after any phase)

**4 verdicts:** `GREEN_LIGHT` / `REVISIT` / `ESCALATE` / `DEFER_PHASE`

**Key disciplines:** SURVEYOR always runs before any patch worker; toolkit/category mismatches resolved before patching; build must pass before GREEN_LIGHT; one commit per checklist item; planning docs (APP-TASKS.md, INPROGRESS.md) updated on main, not on source branch.

---

### ORGANIZE_WORKFLOW — Structural Cleanup

**Purpose:** Periodic structural cleanup of a cluttered project. Operates at filesystem-structure altitude — rearranges where files live, never what they contain. Two modes: BOOTSTRAP (first run, wizard proposes config) and MAINTENANCE (apply rules in parallel-safe batches). Two workers: INSPECTOR (BOOTSTRAP-only survey) and TRIAGE (MAINTENANCE parallel classifier). Two quarantine destinations (`.delete/`, `.archive/`); nothing is ever `rm`'d. Bounded N circuits per invocation; no daemon mode.

**Trigger phrase:** `ORGANIZE_WORKFLOW` — or natural-language invocation (e.g., "organize", "tidy project", "clean up")

**Roles:**

| Role | Doc | Active when |
|---|---|---|
| INSPECTOR (survey project; propose template + seed rules; read-only) | `workflows/ORGANIZE/WORKER_INSPECTOR.md` | BOOTSTRAP only |
| TRIAGE (parallel per-file classifier; emits verdicts per fixed catalog) | `workflows/ORGANIZE/WORKER_TRIAGE.md` | MAINTENANCE (one per batch, parallel) |

**Modes:**
- `BOOTSTRAP` — no `.organize.json` at project root; INSPECTOR surveys → wizard → QUEEN writes config → optional cascade into MAINTENANCE
- `MAINTENANCE` — `.organize.json` exists; enumerate → batch → TRIAGE_WAVE (parallel) → aggregate → ratify → `git mv` → commit

**Composite units:**
- `TRIAGE_WAVE` — one circuit's parallel TRIAGE spawn; all workers in a wave run in a single message (`run_in_background: true`); QUEEN waits for all to return before aggregation
- `WIZARD_LOOP` — BOOTSTRAP's interactive dialog; pure QUEEN+user; no agent spawning; 6 stages (signals → template → rules → ignore_paths → preview → cascade)

**Verdicts:**

BOOTSTRAP catalog:
- `CONFIG_CREATED` — wizard complete; config written; no cascade
- `CONFIG_CREATED_AND_CLEAN_RUN` — wizard complete + 1 MAINTENANCE circuit executed in the same invocation
- `ABORTED_DURING_WIZARD` — user aborted before `.organize.json` was written; no state changes

MAINTENANCE catalog:
- `CLEAN_RUN` — all N circuits completed; all ratified moves executed; config updated
- `PARTIAL_RUN` — user halted after some circuits; state as of last completed circuit committed
- `ABORTED` — user aborted before any circuit's moves executed; no filesystem changes
- `NOTHING_TO_DO` — all circuits ran; all TRIAGE verdicts were KEEP_IN_PLACE; config updated (runs counter + last_run)

**Key disciplines:**
- `never_delete` — workflow never invokes `rm`; `.delete/` is the only garbage destination and is user-managed
- `git_mv_only` — all moves use `git mv`; untracked files use plain `mv` logged as `untracked_at_move: true`
- `no_auto_ratify` — no move executes without explicit user ratification
- `append_only_rules_log` — `.organize.json` rules array + quarantine_log are append-only
- `circuit_bounded` — single invocation runs exactly N circuits; no daemon mode
- `root_invariants_untouchable` — `.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json` NEVER enumerated or moved
- `no_workflow_mixing` — ORGANIZE does not interleave with SDLC, RDC, RECON, or BOOK in the same conversation
- `full_read_or_skip` — TRIAGE reads each file fully before verdict, or emits ASK_USER with blocker

---

### RECON_WORKFLOW — Reconnaissance and Relational Analysis

**Purpose:** Given a target directory (and optionally an anchor corpus), characterize the target — either in isolation (`UNANCHORED`), in relation to an existing consolidated corpus (`ANCHORED`), or as a pure audit of an anchor's health (`AUDIT_ONLY`). RECON is **not** a prestep to RDC — it operates at parallel altitude. A single SCOUT worker produces exactly one markdown report per invocation.

**Trigger phrase:** `RECON_WORKFLOW` — or natural-language invocation (e.g., "recon X", "scout Y with anchor Z", "audit anchor W")

**Roles:**

| Role | Doc |
|---|---|
| SCOUT (orientation-first, read-only, single-report) | `workflows/RECON/WORKER_SCOUT.md` |

**Modes:**
- `UNANCHORED` — target only; first-contact characterization
- `ANCHORED` — target + anchor; relational analysis with anchor veracity check
- `AUDIT_ONLY` — anchor only; pure anchor health check

**Output:** single `.md` report in `<session_summon_dir>/RECON/`, named per mode (`RECON_<target>.md` / `RECON_<target>_vs_<anchor>.md` / `RECON_AUDIT_<anchor>.md`). Peer of anchor directories at the session-summon directory.

**Verdicts:** mode-specific catalogs. Anchor sub-verdicts (`ANCHOR_HEALTHY` / `ANCHOR_STALE` / `ANCHOR_INCOMPLETE` / `ANCHOR_DRIFTED` / `ANCHOR_CORRUPTED`) run in `ANCHORED` and `AUDIT_ONLY`.

**Key disciplines:** full-read-or-skip (never partial-scan), orientation-first via 3-tier search, read-only on target + anchor, no auto-recursion, no auto-trigger of downstream workflows.

---

### SDLC_WORKFLOW — Software Development Lifecycle

**Purpose:** Execute coding tasks from structured `PHASE_<N>_<NAME>_TODO.md` docs. 7 worker roles + QUEEN.

**Trigger phrase:** `SDLC_WORKFLOW`

**Roles:**

| Role | Doc |
|---|---|
| DEV (code only, no tests) | `workflows/SDLC/WORKER_DEV.md` |
| TESTDEV_WHITEBOX (full visibility) | `workflows/SDLC/WORKER_TESTDEV_WHITEBOX.md` |
| TESTDEV_BLACKBOX (cleanroom) | `workflows/SDLC/WORKER_TESTDEV_BLACKBOX.md` |
| JUNIOR_QA (hypercritical) | `workflows/SDLC/WORKER_QA_JUNIOR.md` |
| SENIOR_QA_SANITY (judicial filter) | `workflows/SDLC/WORKER_QA_SENIOR_SANITY.md` |
| SENIOR_QA_FINAL (verdict) | `workflows/SDLC/WORKER_QA_SENIOR_FINAL.md` |
| ARCH_DEV (changelog-only amendments) | `workflows/SDLC/WORKER_ARCH_DEV.md` |

**Composite units:**
- `TEST_UNIT` = WHITEBOX || BLACKBOX (parallel, atomic)
- `QA_UNIT` = JUNIOR → SANITY → FINAL (sequential, atomic)
- `ARCH_FLOW` = ARCH_DEV → QA_UNIT (on REWRITE)

**4 verdicts:** `GREEN_LIGHT` / `FIX` / `REWRITE` / `ESCALATE`

---

### RDC_WORKFLOW — Research and Documentation Consolidation

**Purpose:** Consolidate N chaotic markdown documents into SDLC-ready structured output (`PROJECT.md`, `PHASE_<N>_<NAME>_ARCH.md`, `PHASE_<N>_<NAME>_TODO.md`, `CLARIFICATION.md`). Upstream of SDLC.

**Trigger phrase:** `RDC_WORKFLOW`

**Roles:**

| Role | Doc |
|---|---|
| SCRIBE (temporal upsert) | `workflows/RDC/WORKER_SCRIBE.md` |
| ADVOCATE (court advocacy) | `workflows/RDC/WORKER_ADVOCATE.md` |
| TAXONOMIST (carve MASTER into output docs) | `workflows/RDC/WORKER_TAXONOMIST.md` |
| QA_COMPLETENESS (concept-loss hunter) | `workflows/RDC/WORKER_QA_COMPLETENESS.md` |
| QA_COHERENCE (structural auditor) | `workflows/RDC/WORKER_QA_COHERENCE.md` |

**Composite units:**
- `COURT_UNIT` = 4 ADVOCATEs (2 per side, parallel) + QUEEN ruling (conditional on conflicts)
- `QA_UNIT` = QA_COMPLETENESS → QA_COHERENCE (sequential, atomic)

**4 verdicts:** `GREEN_LIGHT` / `SCRIBE_REVISIT` / `RETAXONOMIZE` / `ESCALATE`

**Pipeline:** RDC output → SDLC input. Manual handoff.

---

### BOOK_WORKFLOW — Book Writing Cycle (6-workflow family)

**Purpose:** Transform raw manuscript material into print-ready books. Handles the full pipeline from chaotic source folders through consolidation, logical structuring (storyboarding), editorial review, and production packaging. Operates on authored prose rather than source code.

**Architecture reference:** `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` (v1.0.0 IMPLEMENTED)

**Family pipeline (manual handoff at each stage):**

```
BOOK_TRIAGE → BOOK_CONSOLIDATION → BOOK_STORYBOARD → BOOK_EDITORIAL → BOOK_PRODUCTION
                    ↑ mixed-state manuscripts enter via BOOK_COMPLETION
```

---

#### BOOK_TRIAGE

**Trigger phrase:** `BOOK_TRIAGE`
**Nature:** QUEEN-only, single-pass. No worker spawning. No QA loop.
**Purpose:** Classify input folder, produce `BOOK_MANIFEST.json`, recommend pipeline entry point.

**Roles:** QUEEN only (no spawned workers)

---

#### BOOK_CONSOLIDATION

**Trigger phrase:** `BOOK_CONSOLIDATION`
**Purpose:** Transform a chaotic folder of manuscript docs into structured chapter files + STRUCTURE.md. RDC fork with COMPOSITOR replacing TAXONOMIST.

**Roles:**

| Role | Doc |
|---|---|
| SCRIBE (temporal upsert into MASTER.md) | `workflows/BOOK/WORKER_SCRIBE.md` |
| ADVOCATE (court advocacy) | `workflows/BOOK/WORKER_ADVOCATE.md` |
| COMPOSITOR (carve MASTER → chapter files + STRUCTURE.md) | `workflows/BOOK/WORKER_COMPOSITOR.md` |
| QA_COMPLETENESS (concept-loss hunter) | `workflows/BOOK/WORKER_QA_COMPLETENESS.md` |
| QA_COHERENCE (manuscript structural auditor) | `workflows/BOOK/WORKER_QA_COHERENCE.md` |

**Composite units:**
- `COURT_UNIT` = 4 ADVOCATEs (2 per side, parallel) + QUEEN ruling (conditional on conflicts)
- `QA_UNIT` = QA_COMPLETENESS → QA_COHERENCE (sequential, atomic)

**4 verdicts:** `GREEN_LIGHT` / `SCRIBE_REVISIT` / `RECOMPOSE` / `ESCALATE`

---

#### BOOK_COMPLETION

**Trigger phrase:** `BOOK_COMPLETION`
**Purpose:** Meta-orchestrator for mixed-state manuscripts. Reads per-chapter state from BOOK_MANIFEST, produces a routing plan, invokes per-chapter subsets of existing workflows, and invokes DRAFTER for chapters with insufficient existing material.

**Roles:**

| Role | Doc |
|---|---|
| DRAFTER (author prose for missing/outline/notes-only chapters under template constraint) | `workflows/BOOK/WORKER_DRAFTER.md` |

**Note:** BOOK_COMPLETION orchestrates other BOOK workflows (CONSOLIDATION, STORYBOARD, EDITORIAL) on per-chapter subsets. DRAFTER is the only role it directly spawns.

**Verdicts:** `ALL_CHAPTERS_COMPLETE` / `PARTIAL_COMPLETION` / `ESCALATE`

---

#### BOOK_STORYBOARD

**Trigger phrase:** `BOOK_STORYBOARD`
**Purpose:** Produce a voice-neutral logical/pedagogical skeleton (STORYBOARD.md) from structured chapter files.

**Roles:**

| Role | Doc |
|---|---|
| STORYBOARDER (constructive — reads full manuscript, produces STORYBOARD.md) | `workflows/BOOK/WORKER_STORYBOARDER.md` |
| QA_STORYBOARD (audits prerequisite chain, arc, completeness, accuracy, genre alignment) | `workflows/BOOK/WORKER_QA_STORYBOARD.md` |

**3 verdicts:** `GREEN_LIGHT` / `REVISE` / `ESCALATE`

---

#### BOOK_EDITORIAL

**Trigger phrase:** `BOOK_EDITORIAL`
**Purpose:** Audit and revise manuscript against declared templates (VOICE, PERSONA, STYLE, PROSE or BUNDLE). Template compatibility validated at engagement before any workers spawn.

**Roles:**

| Role | Doc |
|---|---|
| JUNIOR_VOICE (audits against VOICE template; hypercritical, high-recall) | `workflows/BOOK/WORKER_JUNIOR_VOICE.md` |
| JUNIOR_CONCEPT (audits concept consistency; not template-bound) | `workflows/BOOK/WORKER_JUNIOR_CONCEPT.md` |
| JUNIOR_STYLE (audits against STYLE + PROSE templates) | `workflows/BOOK/WORKER_JUNIOR_STYLE.md` |
| JUNIOR_FLOW (audits logical flow against STORYBOARD.md) | `workflows/BOOK/WORKER_JUNIOR_FLOW.md` |
| EDITORIAL_SYNTHESIS (cross-axis interaction detection; only worker that sees all 4 junior reports) | `workflows/BOOK/WORKER_EDITORIAL_SYNTHESIS.md` |
| SENIOR_SANITY (precision filter: real vs. overzealous; no new findings) | `workflows/BOOK/WORKER_SENIOR_SANITY.md` |
| SENIOR_FINAL (independent pass + binding verdict; QUEEN does not override) | `workflows/BOOK/WORKER_SENIOR_FINAL.md` |
| REVISION (surgical prose rewriter; touches only flagged passages) | `workflows/BOOK/WORKER_REVISION.md` |

**Composite units:**
- `JUNIOR_EDITORIAL` = JUNIOR_VOICE ‖ JUNIOR_CONCEPT ‖ JUNIOR_STYLE ‖ JUNIOR_FLOW (4 parallel; no cross-visibility)
- `EDITORIAL_PIPELINE` = JUNIOR_EDITORIAL → EDITORIAL_SYNTHESIS → SENIOR_SANITY → SENIOR_FINAL (sequential, atomic)
- `QA_UNIT` (in BOOK_EDITORIAL) = full EDITORIAL_PIPELINE (re-runs from JUNIOR on each REVISE cycle)

**3 verdicts:** `GREEN_LIGHT` / `REVISE` / `ESCALATE`

---

#### BOOK_PRODUCTION

**Trigger phrase:** `BOOK_PRODUCTION`
**Purpose:** Validate polished manuscript, generate front/back matter, produce BOOK_SPEC.json — the automation-ready intermediate consumed by LULU_PIPELINE (mechanical, not AI).

**Roles:**

| Role | Doc |
|---|---|
| FORMATTER (validates manuscript; generates front/back matter; produces BOOK_SPEC.json) | `workflows/BOOK/WORKER_FORMATTER.md` |
| QA_PRODUCTION (validates output against LULU_SPEC; checks completeness + spec compliance) | `workflows/BOOK/WORKER_QA_PRODUCTION.md` |

**3 verdicts:** `GREEN_LIGHT` / `FIX` / `ESCALATE`

---

## What every worker does first

In order:

1. **Read this file** (`workflows/SHARED/WORKER.md`) — workflow routing
2. **Read `workflows/SHARED/WORKER_PROTOCOL.md`** — non-negotiable rules, tool usage, reporting format, escalation. The contract.
3. **Read your role doc** (see tables above) — role-specific mechanics
4. **Read the workflow JSON** if you need the state machine
5. **Read task-specific inputs** named in your prompt — TODO entries, ARCH sections, source docs, prior worker outputs, etc.

---

## Hard rules (excerpted from WORKER_PROTOCOL.md)

1. **Never fabricate results.** Every number in your report must come from a command you ran. Quote the output.
2. **Never disable tests** or skip acceptance criteria to make CI pass.
3. **Never write `TODO`, `FIXME`, `HACK`, `WONTFIX`, `XXX` in committed code.**
4. **Never write a comment describing a result you didn't compute.**
5. **Never expand scope.** Stay within your task / pass / role.
6. **Never declare "done" without the acceptance command output in your report.**
7. **Stop and raise on estimate overrun.**
8. **Never skip QA** (SDLC) or **skip COURT when conflicts are flagged** (RDC).

Full list in `workflows/SHARED/WORKER_PROTOCOL.md`.

---

## What you CAN'T assume

- **You have no conversation history.** Trust only what's in files and your prompt.
- **You may or may not have ZimaBoard SSH access.** Many SDLC tasks require `root@192.168.50.142`. Verify first; if down, report BLOCKED.
- **Other workers may be running in parallel.** Commit small, don't assume exclusive file access.
- **The task / pass Do NOT list is authoritative.** Read it; obey it.
- **Your role's doc is specific.** Don't perform another role's duties.

---

## Consequences of violations

Michael's explicit rule: *"if the DEV or QA outputs COMMENTS as if they were CALCULATIONS, immediate discard and redo: do not fake results as comments."*

This extends to every role. Fabricated reports get thrown out. The swarm only works if reports are truthful.

---

## If you're blocked

1. Do not fake a result.
2. Commit any WIP with `[BLOCKED]` prefix.
3. Report the blocker with enough detail that QUEEN can resolve it.

Legitimate blockers: missing prerequisites, ZimaBoard unreachable, ambiguous task description, missing upstream context, estimate exceeded past pessimistic bound.

---

*End of index. Proceed to `workflows/SHARED/WORKER_PROTOCOL.md`.*
