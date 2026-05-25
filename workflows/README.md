# `workflows/` — Swarm Workflow + CI/CD System

**What this is:** A portable, workflow-driven AI swarm development system for Claude Code. Includes multiple self-bootstrapping workflows (SDLC, RDC, RECON, BOOK-family, ORGANIZE), shared worker protocol, polyglot CI/CD scripts, and git hooks integration.

**Intent:** drop this directory into any project, reference the workflows you want by trigger phrase, and you have a reusable swarm-development system with language-level CI enforcement. Each workflow's JSON contains its own engagement protocol (`trigger.on_engage`) — no CLAUDE.md surgery required.

---

## Directory structure

```
workflows/
├── README.md                   ← this file
├── install.sh                  ← unified installer (hooks only; workflows are self-bootstrapping)
├── install-hooks.sh            ← installs pre-commit + pre-push git hooks
├── ci-python.sh                ← Python CI pipeline (ruff + mypy + pytest + build)
├── ci-rust.sh                  ← Rust CI pipeline (cargo fmt + clippy + test + build)
│
├── SHARED/                     ← shared across all workflows
│   ├── WORKER.md               ← master index, routes workers to active workflow
│   ├── WORKER_PROTOCOL.md      ← non-negotiable contract every worker follows
│   └── WORKER_QUEEN.md         ← orchestrator role (workflow-agnostic)
│
├── SDLC/                       ← Software Development Lifecycle workflow
│   ├── SDLC_WORKFLOW.json      ← state machine
│   ├── WORKER_DEV.md           ← writes code only, no tests
│   ├── WORKER_TESTDEV_WHITEBOX.md   ← writes whitebox tests, full visibility
│   ├── WORKER_TESTDEV_BLACKBOX.md   ← writes blackbox tests, cleanroom
│   ├── WORKER_QA_JUNIOR.md     ← adversarial first filter
│   ├── WORKER_QA_SENIOR_SANITY.md   ← filters junior's findings
│   ├── WORKER_QA_SENIOR_FINAL.md    ← emits the branch verdict
│   └── WORKER_ARCH_DEV.md      ← architecture changelog amendments
│
├── RDC/                        ← Research + Documentation Consolidation workflow
│   ├── RDC_WORKFLOW.json       ← state machine with COURT mechanism
│   ├── WORKER_SCRIBE.md        ← temporal upsert worker (core of RDC)
│   ├── WORKER_ADVOCATE.md      ← court advocacy for conflict resolution
│   ├── WORKER_TAXONOMIST.md    ← carves MASTER into SDLC-ready outputs
│   ├── WORKER_QA_COMPLETENESS.md    ← concept-loss hunter
│   └── WORKER_QA_COHERENCE.md  ← structural integrity auditor
│
├── RECON/                      ← Reconnaissance + Relational Analysis workflow
│   ├── RECON_WORKFLOW.json     ← state machine (UNANCHORED / ANCHORED / AUDIT_ONLY modes)
│   └── WORKER_SCOUT.md         ← sole worker, read-only, one report per invocation
│
├── BOOK/                       ← Manuscript pipeline (v0.1.0-DRAFT specs, workers TBD)
│   ├── BOOK_WORKFLOW_DISSERTATION.md   ← design rationale
│   ├── BOOK_TRIAGE.json        ← classification + manifest
│   ├── BOOK_CONSOLIDATION.json ← chaotic folder → MASTER + chapters (RDC fork)
│   ├── BOOK_STORYBOARD.json    ← structured → pedagogical skeleton
│   ├── BOOK_EDITORIAL.json     ← storyboarded → polished (VOICE/PERSONA/STYLE/PROSE axes)
│   └── BOOK_PRODUCTION.json    ← polished → print-ready intermediate
│
└── ORGANIZE/                   ← Project-structure cleanup workflow (v0.1.0-DRAFT)
    ├── ORGANIZE_WORKFLOW.json  ← state machine (BOOTSTRAP / MAINTENANCE modes)
    ├── WORKER_INSPECTOR.md     ← BOOTSTRAP: detect project type, propose template + seed rules
    └── WORKER_TRIAGE.md        ← MAINTENANCE: per-file classifier with fixed verdict catalog
```

---

## Two-layer enforcement

This system has **two complementary enforcement layers** that work together:

### Layer 1 — CI scripts + git hooks (language-level)

**What:** `ci-python.sh`, `ci-rust.sh`, `install-hooks.sh` enforce formatting, linting, type-checking, tests, and builds at the `git commit` and `git push` boundaries. Polyglot — auto-detects Python (`pyproject.toml`) and Rust (`Cargo.toml`) and runs both if present.

**When it fires:**
- **Pre-commit:** every `git commit` triggers `quick` mode (fmt + lint on staged files of that language)
- **Pre-push:** every `git push` triggers `gate` mode (fmt + lint + typecheck + test + build, full language pipeline)

**Purpose:** catches language-level defects before the swarm's QA even looks at the code. Reduces QA_UNIT workload by filtering noise.

### Layer 2 — Swarm workflow (semantic-level)

**What:** `SDLC_WORKFLOW` / `RDC_WORKFLOW` enforce architectural, correctness, completeness, and coherence at the workflow-verdict boundary. Workers spawned per role; adversarial QA; strict protocol.

**When it fires:**
- **Whenever you type a trigger phrase** (`SDLC_WORKFLOW`, `RDC_WORKFLOW`, `RECON_WORKFLOW`, `ORGANIZE_WORKFLOW`, or a BOOK-family phrase)
- QUEEN orchestrates; workers get spawned per role; verdicts are binding

**Purpose:** catches semantic defects (wrong algorithm, missed coverage, architectural drift, concept loss) that CI can't see.

### How they interact

```
Worker commits code
    ↓
pre-commit hook: CI quick mode (fmt + lint)     ← Layer 1: language-level
    ↓ (pass or reject commit)
TEST_UNIT / QA_UNIT review (SDLC)                ← Layer 2: semantic-level
    ↓ (pass or FIX/REWRITE)
GREEN_LIGHT → QUEEN squash-merges to main
    ↓
pre-push hook: CI gate mode (full language pipeline)   ← Layer 1: final gate
    ↓
Main branch updated
```

**Hard rule:** NEVER bypass hooks with `--no-verify`. If a hook fails, fix the underlying issue. (See `SHARED/WORKER_PROTOCOL.md`.)

---

## How to deploy to a new project

The deployment model is intentionally minimal: copy `workflows/` into the project, install git hooks, and reference workflows by trigger phrase when you need them. No CLAUDE.md surgery — workflows are self-bootstrapping via their JSON `trigger.on_engage` protocol.

### Step 1 — Copy the `workflows/` directory

```bash
cp -r /path/to/source/workflows /path/to/new-project/
```

Preserve the structure (`SHARED/`, `SDLC/`, `RDC/`, `RECON/`, `BOOK/`, `ORGANIZE/`, plus the `.sh` scripts).

### Step 2 — Install git hooks

```bash
cd /path/to/new-project
bash workflows/install.sh
```

This installs `.git/hooks/pre-commit` (runs `quick` on staged Python/Rust files) and `.git/hooks/pre-push` (runs the full `gate`). Polyglot-aware — runs both languages if both manifests present.

Additional modes:
- `bash workflows/install.sh reinit` — **DESTRUCTIVE**: delete `.git/`, fresh `git init`, install hooks. Requires typing `yes` to confirm (or `--yes` flag).
- `bash workflows/install.sh --help` — full help.

The installer does **not** modify `CLAUDE.md`. It only installs hooks.

### Step 3 — Reference workflows in your CLAUDE.md (optional)

If you want discoverability, add a short section to your project's `CLAUDE.md` listing the trigger phrases you care about. A one-line pointer is enough:

```markdown
Workflows available: SDLC_WORKFLOW, RDC_WORKFLOW, RECON_WORKFLOW, ORGANIZE_WORKFLOW. See workflows/ for details.
```

This is optional — Claude can recognize trigger phrases and follow each workflow's `trigger.on_engage` protocol without prior instruction. The reference just helps you remember what's available.

### Step 4 — Provide workflow inputs

- **SDLC:** create at least one `PHASE_<N>_<NAME>_TODO.md` at project root with actionable tasks
- **RDC:** have a source directory with `.md` files to consolidate
- **RECON:** nothing required — invoke with target dir (and optional anchor) at engagement
- **BOOK family:** manuscript folder appropriate to the stage you're invoking
- **ORGANIZE:** nothing required for BOOTSTRAP; `.organize.json` created by wizard on first run

### Step 5 — Engage

Open a Claude Code session in the project. Type a trigger phrase:

- `SDLC_WORKFLOW` — execute coding tasks from structured TODOs
- `RDC_WORKFLOW` — consolidate chaotic `.md` docs into structured output
- `RECON_WORKFLOW` — characterize a directory (standalone / vs anchor / audit)
- `ORGANIZE_WORKFLOW` — organize/clean up a project's structure
- `BOOK_TRIAGE` (or other BOOK phrases) — manuscript pipeline stages

QUEEN engages, reads the workflow JSON in full (the `trigger.on_engage` section lists which shared + role docs to read next), acknowledges, and waits for you to specify the input required by that workflow.

### Updating an existing deployment

When `workflows/` itself gets updated (bug fixes, new roles, etc.):

```bash
cd /path/to/existing-project
git pull  # or however you get the new workflows/
bash workflows/install.sh  # refresh git hooks
```

Project-specific content in your `CLAUDE.md` is untouched — the installer doesn't read or modify it.

---

## Workflow summaries

### SDLC_WORKFLOW — code execution

**Trigger:** `SDLC_WORKFLOW`
**Input:** `PHASE_<N>_<NAME>_TODO.md` with TASK_IDs
**Output:** working code merged to main, TODO checkboxes toggled, INPROGRESS updated

**Pipeline per task:**

```
PRESTEP (branch, INPROGRESS init)
    ↓
DEV (code only, no tests)
    ↓
TEST_UNIT: WHITEBOX || BLACKBOX (parallel)
    ↓
QA_UNIT: JUNIOR → SANITY → FINAL (sequential)
    ↓
Verdict:
    GREEN_LIGHT → squash merge, delete task branch
    FIX         → DEV + TEST_UNIT revise → full QA re-run (max 3 cycles)
    REWRITE     → archive branch, ARCH_FLOW on main, fresh task (max 2 rewrites)
    ESCALATE    → pause, report to human
```

**Hard rules (non-exhaustive):**
- No GREEN_LIGHT without a full QA_UNIT
- QUEEN never overrides SENIOR_QA_FINAL's verdict
- BLACKBOX is cleanroom — forbidden from reading implementation
- Task branches deleted on GREEN_LIGHT; attempt branches preserved

### RDC_WORKFLOW — doc consolidation

**Trigger:** `RDC_WORKFLOW`
**Input:** directory of `.md` source docs of varying epochs
**Output:** `PROJECT.md`, `PHASE_<N>_<NAME>_ARCH.md` set, `PHASE_<N>_<NAME>_TODO.md` set, `CLARIFICATION.md`, `PEDAGOGY.md`, `MASTER.md`

**Pipeline:**

```
INVENTORY (temporal ordering, human-confirmed)
    ↓
SCRIBE_LOOP (one SCRIBE per source doc, temporal order)
    ↓
COURT (conditional — only if conflicts flagged)
    - 4 ADVOCATEs per session (2 per side, parallel)
    - QUEEN rules: SIDE_A | SIDE_B | SYNTHESIS | DEFER | REJECT_BOTH | NO_DECISION
    - Recorded to INPROGRESS.md as COURT #N entry
    ↓
TAXONOMY (carve MASTER → output docs, phases discovered not prescribed)
    ↓
QA_UNIT: QA_COMPLETENESS → QA_COHERENCE
    ↓
Verdict:
    GREEN_LIGHT     → commit outputs, announce SDLC-ready
    SCRIBE_REVISIT  → re-SCRIBE flagged docs → re-TAXONOMY → re-QA (max 3)
    RETAXONOMIZE    → re-TAXONOMY with findings → re-QA (max 3)
    ESCALATE        → pause, report
```

**Pipeline note:** RDC output → SDLC input. Manual handoff. Human reviews RDC output before typing `SDLC_WORKFLOW`.

### RECON_WORKFLOW — relational / isolation analysis

**Trigger:** `RECON_WORKFLOW` (or natural-language: "recon X", "scout Y with anchor Z", "audit anchor W")
**Input:** target directory (and/or anchor corpus name)
**Output:** exactly one markdown report in `<session_summon_dir>/RECON/` per invocation

Three modes: `UNANCHORED` (characterize target in isolation), `ANCHORED` (position target relative to existing anchor corpus), `AUDIT_ONLY` (health-check an anchor). Single SCOUT worker, read-only, full-read-or-skip discipline, 3-tier orientation discovery. Never auto-chains to other workflows.

### BOOK family — manuscript pipeline (v0.1.0-DRAFT specs)

**Trigger phrases:** `BOOK_TRIAGE`, `BOOK_CONSOLIDATION`, `BOOK_STORYBOARD`, `BOOK_EDITORIAL`, `BOOK_PRODUCTION`
**Input:** manuscript folder at the stage appropriate to each workflow
**Output:** stage-dependent — manifest, MASTER + chapters, storyboarded chapters, polished prose, print-ready intermediate

Five-stage pipeline with manual handoffs. TRIAGE classifies input; CONSOLIDATION is an RDC fork for prose; STORYBOARD/EDITORIAL/PRODUCTION refine. Worker docs not yet written (JSONs only). Forked from RDC/SDLC patterns for prose instead of code.

### ORGANIZE_WORKFLOW — project-structure cleanup (v0.1.0-DRAFT)

**Trigger:** `ORGANIZE_WORKFLOW` (or natural-language: "organize", "tidy project", "clean up project")
**Input:** the current directory (CWD)
**Output:** file moves + `.organize.json` config (created or updated)

Two modes: `BOOTSTRAP` (no `.organize.json` → INSPECTOR surveys, wizard proposes template + seed rules) and `MAINTENANCE` (config exists → parallel TRIAGE workers classify each file, user ratifies proposed moves, QUEEN executes via `git mv`). Quarantine destinations are `.delete/` (garbage) and `.archive/` (historical threads) — nothing is ever deleted. Bounded by N circuits per invocation. Root invariants (`.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json`) are untouchable.

---

## CI script usage

### Direct invocation

```bash
# Python
bash workflows/ci-python.sh fmt          # format check only
bash workflows/ci-python.sh lint         # lint only
bash workflows/ci-python.sh typecheck    # mypy
bash workflows/ci-python.sh test         # pytest
bash workflows/ci-python.sh build        # wheel + sdist
bash workflows/ci-python.sh release      # build + artifact + checksums
bash workflows/ci-python.sh quick        # fmt + lint (what pre-commit runs)
bash workflows/ci-python.sh gate         # fmt + lint + typecheck + test + build (what pre-push runs)
bash workflows/ci-python.sh full         # everything including release

# Rust (same modes, minus typecheck — clippy does it)
bash workflows/ci-rust.sh {fmt|lint|test|build|release|quick|gate|full}
```

### Pre-commit hook behavior

Staged files are scanned:
- If any `.rs` or `.toml` files staged and `Cargo.toml` exists → run `ci-rust.sh quick`
- If any `.py` files staged and `pyproject.toml`/`setup.py` exists → run `ci-python.sh quick`
- Both can run in polyglot repos
- Commit rejected on failure

### Pre-push hook behavior

Full gate per language present:
- `Cargo.toml` exists → run `ci-rust.sh gate`
- `pyproject.toml` / `setup.py` exists → run `ci-python.sh gate`
- Push rejected on failure

### Skipping (emergency only)

```bash
git commit --no-verify
git push --no-verify
```

**Workers MUST NOT skip hooks.** This is for human emergencies only (e.g., WIP commits clearly marked as such). If a hook fails during worker operation, the fix is to fix the code, not bypass.

---

## Hard rules (cross-workflow, from `SHARED/WORKER_PROTOCOL.md`)

1. No fabricated results — every number traces to a command run
2. No disabled tests to make CI pass
3. No `TODO`/`FIXME`/`HACK`/`WONTFIX`/`XXX` in committed code
4. No comments describing uncomputed results
5. No scope creep beyond the task/pass
6. No declaration of "done" without acceptance command output in report
7. No bypass of pre-commit / pre-push hooks
8. No skipping QA (SDLC) or COURT (RDC)

---

## Files produced by the system

During workflow execution, various files are created / updated at project root:

| File | Workflow | Semantics |
|---|---|---|
| `INPROGRESS.md` | both | Swarm progress log. Prepend-only except checkboxes. Dumping ground for all workflow progress + COURT transcripts. |
| `MASTER.md` | RDC | Consolidated single-file project overview. Preserved after GREEN_LIGHT for future RDC runs. |
| `PEDAGOGY.md` | RDC | Archaeological record of concept evolution. Append-only. |
| `EVALUATIONS.md` | RDC | Per-SCRIBE-pass record of what each source doc contributed. Append-only. |
| `INVENTORY.md` | RDC | Temporal manifest of source docs. Produced during INVENTORY phase. |
| `PROJECT.md` | RDC | Project-wide scope/goals/constraints. Output of TAXONOMY. |
| `PHASE_<N>_<NAME>_ARCH.md` | RDC (output) / SDLC (input) | Per-phase architectural context. |
| `PHASE_<N>_<NAME>_TODO.md` | RDC (output) / SDLC (input) | Per-phase actionable task list. |
| `CLARIFICATION.md` | RDC | Philosophical/pedagogical framing. |
| `RECON/<report>.md` | RECON | One report per RECON invocation; name encodes mode + target + anchor. |
| `.organize.json` | ORGANIZE | Per-project config: template, rules (append-only), quarantine log, pending rule candidates. |
| `.delete/`, `.archive/` | ORGANIZE | Quarantine destinations. `.delete/` is gitignored; `.archive/` is user-optional. Files mirror their original paths beneath the quarantine root. |

---

## Design notes

### Why two workflows instead of one?

They operate at different levels of abstraction. RDC produces structured documents; SDLC executes against structured documents. Trying to do both in one workflow would muddle adversarial QA semantics (different kinds of findings matter at each level).

### Why a COURT mechanism for RDC?

Source documents written across epochs genuinely contradict each other, and temporal supersession doesn't always cleanly resolve which is correct. The COURT mechanism (4 advocates, 2 per side, QUEEN as judge) makes these conflicts visible and resolves them with an audit trail in `INPROGRESS.md`. The alternative — silent resolution during SCRIBE — loses information and produces architecturally inconsistent outputs.

### Why cleanroom TESTDEV_BLACKBOX?

Independent tests. A DEV who writes their own tests writes tests matching their own mental model, including their blind spots. BLACKBOX sees only the contract (TODO spec + ARCH). WHITEBOX sees the implementation. Together they cover what each alone would miss.

### Why three QA passes (JUNIOR → SANITY → FINAL)?

Tension between precision and recall. JUNIOR is biased to flag (high recall, err toward finding issues). SANITY filters junior's findings (precision filter — drops false positives). FINAL does an independent pass and emits the binding verdict. Three passes give three independent chances to catch real bugs and two independent chances to kill false positives.

### Why does QUEEN not override SENIOR_QA_FINAL?

Verdict authority lives in the role that sees the most context (the whole task plus all prior QA output). QUEEN's job is coordination + execution, not second-guessing. If QUEEN could override, SENIOR_QA_FINAL becomes advisory and the swarm's quality gate becomes porous.

### Why is INPROGRESS prepend-only?

History is permanent. The dumping-ground design means everything that happened is recorded somewhere. Re-litigating past decisions is cheap because the decision is still visible. Truncation / editing breaks the audit trail.

### Why changelog-prepend-only ARCH updates?

Historical decisions stay readable in the context they were made in. An ARCH doc that gets edited silently loses the signal that a decision CHANGED — future readers see only the current state without realizing it evolved. Prepend-only changelogs preserve that evolution.

---

## Versioning

- `workflows/SDLC/SDLC_WORKFLOW.json` — 1.0.0
- `workflows/RDC/RDC_WORKFLOW.json` — 1.2.0 (COURT mechanism + multi-parallel cluster mode)
- `workflows/RECON/RECON_WORKFLOW.json` — 1.0.0
- `workflows/BOOK/*.json` — 0.1.0-DRAFT (specs only; workers TBD)
- `workflows/ORGANIZE/ORGANIZE_WORKFLOW.json` — 0.1.0-DRAFT

Version bumps on breaking changes to workflow state machines. Role docs version with the JSON they belong to.

---

## Credits

Designed collaboratively by Michael (owner) and Claude, with specific innovations:
- DEV / TESTDEV separation (discipline)
- 3-level QA (recall + precision + final authority)
- COURT mechanism (adversarial conflict resolution with audit trail)
- Phase discovery (not prescription) by TAXONOMIST
- CI hooks under swarm QA (two-layer enforcement)
- INPROGRESS as unified dumping ground across workflows

---

*End of workflows/ README.*
