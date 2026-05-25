# ORGANIZE Workflow Family — Entry Point

ORGANIZE is a periodic structural cleanup workflow for project directories. Given a project root, it classifies loose and misplaced files against a project-specific template and ruleset, proposes moves to the user, and executes approved moves via `git mv`. Nothing is ever deleted — files are moved to one of two quarantine destinations (`.delete/` for garbage, `.archive/` for historical content) or to their canonical template location. The workflow runs in two modes: **BOOTSTRAP** (first run, wizard proposes config) and **MAINTENANCE** (apply rules in parallel-safe batches). Each invocation is bounded by N circuits; there is no daemon mode.

---

## Architecture

```
ORGANIZE_WORKFLOW
├── BOOTSTRAP (no .organize.json at project root)
│   ├── INSPECTOR spawned (survey project → detect signals → propose template + rules)
│   ├── WIZARD_LOOP (QUEEN + user dialog: 6 stages)
│   │   ├── Stage 1: review detected signals
│   │   ├── Stage 2: confirm or change proposed template
│   │   ├── Stage 3: review + ratify seed rules
│   │   ├── Stage 4: review + ratify ignore_paths
│   │   ├── Stage 5: preview initial classification (dry run)
│   │   └── Stage 6: write config + cascade decision
│   └── (optional cascade into MAINTENANCE with circuits=1)
│
└── MAINTENANCE (.organize.json exists)
    └── Per circuit (1..N):
        ├── Enumerate project files (respecting ignore_paths + root_invariants)
        ├── Batch files (BATCH_CAP=30, disjoint subtrees)
        ├── TRIAGE_WAVE: spawn one TRIAGE worker per batch (all parallel)
        ├── Aggregate verdicts across all batches
        ├── Present proposed moves to user (grouped by verdict type)
        ├── User ratifies per group or per file
        └── Execute ratified moves via git mv → commit circuit
```

**Quarantine model:** `.delete/` (garbage; gitignored) and `.archive/` (historical; gitignored by default). Both are created lazily on first use. Original file paths are mirrored under the quarantine root for provenance.

**Root invariants:** `.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json` are never enumerated, classified, or moved by any part of this workflow.

---

## Quickstart: "I want to organize my project"

1. `cd` to your project root (the directory you want to organize).

2. Invoke `ORGANIZE_WORKFLOW` in the Claude conversation (or say "organize" / "tidy project").

3. QUEEN detects the mode automatically:
   - **First run (no `.organize.json`)** → BOOTSTRAP mode. QUEEN spawns INSPECTOR to survey the project, then enters the wizard dialog. You will answer questions about template choice, seed rules, and ignore paths. Answer each stage; type `abort` at any point to exit cleanly (no config is written).
   - **Subsequent runs (`.organize.json` exists)** → MAINTENANCE mode. QUEEN loads your config, enumerates loose files, and spawns TRIAGE workers in parallel. You review proposed moves grouped by verdict type and ratify or skip each group.

4. Review each proposal and ratify. You can:
   - Accept all moves in a group with a single response
   - Review and decide file-by-file
   - Skip a group entirely (those files are left alone this circuit)
   - Abort the circuit (moves ratified so far are committed; remaining are untouched)

5. QUEEN executes `git mv` for all ratified moves and commits each circuit: `organize: circuit N of M — moved X files, quarantined Y`.

**Safe exit:** At any prompt, typing `abort` halts cleanly. During BOOTSTRAP, no `.organize.json` is written. During MAINTENANCE, moves ratified before the abort are committed; the rest are left alone.

---

## Key Concepts

**Templates:** Pre-composed rulesets for common project shapes. INSPECTOR detects signals and proposes the best match. 5 canonical templates are in the library; `custom` mode is available for unusual projects. Templates seed the initial rules in `.organize.json`.

**Rules:** The classification logic stored in `.organize.json`. Three kinds: `glob` (path pattern), `regex` (regular expression on path), `hint` (natural-language content-based rule that TRIAGE interprets). Rules are priority-ordered; first match wins. Rules are append-only — existing rules are never deleted, only deactivated with `active: false`.

**Quarantine:** Two destinations, both reversible:
- `.delete/` — garbage-shaped files (scratch, broken, duplicate). Gitignored. User manages manually.
- `.archive/` — historically valuable but inactive files (superseded docs, old designs). Gitignored by default.

**Root invariants:** Developer tooling paths (`.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json`) are unconditionally excluded from enumeration. Not configurable; not shown in ignore_paths; enforced at the workflow level.

**Circuits:** A circuit is one full scan → triage → propose → ratify → execute pass. A single invocation runs N circuits (default 1). Circuits converge — a well-organized project reaches `NOTHING_TO_DO` in subsequent circuits.

**FLAG_NEW_RULE:** When TRIAGE observes a recurring uncovered pattern (3+ files matching no rule in a single batch, or 3+ total across batches via cross-batch aggregation), it surfaces a suggested new rule. You ratify or reject it; ratified rules are appended to `.organize.json` and apply in future runs.

---

## Key Documents

| Document | Purpose |
|---|---|
| `ORGANIZE_WORKFLOW.json` | State machine — modes, flow, roles, verdicts, hard rules |
| `WORKER_INSPECTOR.md` | BOOTSTRAP worker role — survey, signal detection, template proposal |
| `WORKER_TRIAGE.md` | MAINTENANCE worker role — per-file classification, verdict catalog |
| `ORGANIZE_CONFIG_SCHEMA.json` | JSON Schema Draft 7 for `.organize.json` |
| `ORGANIZE_RULE_FORMAT.md` | Rule syntax reference — glob/regex/hint kinds, worked examples, priority |
| `ORGANIZE_TEMPLATE_FORMAT.md` | Template file schema — fields, versioning, how INSPECTOR uses templates |
| `WIZARD_PROTOCOL.md` | BOOTSTRAP dialog spec — 6 stages, response vocabulary, edge cases |
| `RATIFICATION_UI_SPEC.md` | MAINTENANCE dialog spec — ratification UI format, user options |
| `BATCHING_HEURISTIC.md` | Parallel batching algorithm — BATCH_CAP=30, cross-batch FLAG_NEW_RULE aggregation |
| `ORGANIZE_SCHEMA_VERSIONING.md` | Config schema SemVer policy — QUEEN behavior on version drift |
| `SHORT_RDC_SPEC.md` | Compressed prose-relevance sub-procedure (deferred; future design) |
| `EDGE_CASES.md` | Unusual scenarios — binary files, rule conflicts, mid-circuit abort, invalid config |
| `OPEN_QUESTIONS_RESOLVED.md` | All 8 TBDs from v0.1.0-DRAFT — resolution status + rationale |
| `ORGANIZE_CONFIG_EXAMPLE.json` | Sample `.organize.json` validating against the schema |
| `templates/TEMPLATES_INDEX.md` | Index of the 5 canonical templates |
| `templates/TEMPLATES_ROADMAP.md` | Deferred templates + promotion criteria |
| `BOOTSTRAP_WALKTHROUGH.md` | End-to-end BOOTSTRAP mental trace |
| `MAINTENANCE_WALKTHROUGH.md` | End-to-end single-circuit MAINTENANCE trace |
| `ORGANIZE_BUILDOUT_TODO.md` | Build-out plan — 6 parts, 34 tasks, acceptance criteria |

---

## Template Library

Five canonical templates cover the most common project shapes:

| Template | Project shape |
|---|---|
| `python-lib` | Python library with `src/` + `tests/` layout and `pyproject.toml` |
| `rust-crate` | Rust library crate with `Cargo.toml`, `src/lib.rs`, `tests/`, `benches/`, `examples/` |
| `book-markdown` | Manuscript project processed by the BOOK workflow family |
| `mixed-research` | Applied research with code (`src/`), prose (`docs/`), data (`data/`), and notebooks coexisting |
| `knowledge-base` | Topic-organized markdown wiki; no code expected |

If no canonical template fits, use `custom` — INSPECTOR proposes seed rules from observed project signals and you build the ruleset from scratch via the wizard. FLAG_NEW_RULE promotes emerging patterns to explicit rules over time.

See `templates/TEMPLATES_INDEX.md` for descriptions and signal-matching criteria. See `templates/TEMPLATES_ROADMAP.md` for deferred templates (`python-app`, `rust-app`, `polyglot`, `book-print`) and their promotion criteria.

---

## Relationship to Other Workflows

**Parallel to RDC and RECON:** ORGANIZE operates at filesystem-structure altitude (where files live). RDC operates at content-consolidation altitude (merging prose into MASTER.md). RECON operates at relational-characterization altitude (writing a report about a target). They can be used in any order, independently.

**Does NOT auto-trigger other workflows:** ORGANIZE never auto-starts RDC, RECON, SDLC, or BOOK. If ORGANIZE surfaces content that needs consolidation, the user invokes RDC manually.

**Does NOT mix with active workflows:** While ORGANIZE is active, SDLC, RDC, RECON, and BOOK operations do not run in the same conversation.

**RECON as input:** A prior RECON report at the target root (in `RECON/`) is read by INSPECTOR during BOOTSTRAP as orientation input. RECON is not a required prestep — INSPECTOR works without it.

---

## Shared Infrastructure

ORGANIZE uses the same swarm infrastructure as all other workflows:

- `workflows/SHARED/WORKER_QUEEN.md` — QUEEN orchestrator role (workflow-agnostic)
- `workflows/SHARED/WORKER_PROTOCOL.md` — non-negotiable worker contract
- `workflows/SHARED/WORKER.md` — master role index (includes ORGANIZE roles)
