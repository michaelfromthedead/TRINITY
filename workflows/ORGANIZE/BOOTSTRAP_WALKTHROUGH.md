# BOOTSTRAP_WALKTHROUGH — Mental Trace of First-Run ORGANIZE

**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Target project:** `workflows/ORGANIZE/test_project/`
**Template exercised:** `mixed-research`
**Trace type:** Mental trace — no actual moves executed; all verdicts and counts are plausible given test_project layout.
**Spec references:** ORGANIZE_WORKFLOW.json, WIZARD_PROTOCOL.md, WORKER_INSPECTOR.md, ORGANIZE_CONFIG_SCHEMA.json, mixed-research.json

---

## 1. User Invokes ORGANIZE_WORKFLOW in test_project/

The user types in their Claude conversation:

```
ORGANIZE_WORKFLOW
```

(Trigger phrase per `ORGANIZE_WORKFLOW.json §trigger.phrases`.)

### 1.1 QUEEN reads workflow docs

Per `ORGANIZE_WORKFLOW.json §trigger.on_engage`, QUEEN reads:
1. `ORGANIZE_WORKFLOW.json` (full)
2. `workflows/SHARED/WORKER.md`
3. `workflows/SHARED/WORKER_PROTOCOL.md`
4. `workflows/SHARED/WORKER_QUEEN.md`
5. `workflows/ORGANIZE/WORKER_INSPECTOR.md`
6. `workflows/ORGANIZE/WORKER_TRIAGE.md`

### 1.2 QUEEN checks for `.organize.json`

QUEEN looks for `workflows/ORGANIZE/test_project/.organize.json`.

Result: **absent**. No `.organize.json` found at target root. Mode: **BOOTSTRAP**.

Per `ORGANIZE_WORKFLOW.json §root_invariants.enforcement`, QUEEN verifies root invariants:
- `.claude/` — present (`.gitkeep` placeholder)
- `.claude-flow/` — present (`.gitkeep` placeholder)
- `.hive-mind/` — present (`.gitkeep` placeholder)
- `.mcp.json` — present (empty `{}`)

### 1.3 QUEEN engagement report

QUEEN emits:

```
ORGANIZE_WORKFLOW mode engaged. BOOTSTRAP pass queued.
Config: absent. Circuits: 1 (default). Proceeding.
```

---

## 2. QUEEN Spawns INSPECTOR

### 2.1 Context packet sent to INSPECTOR

QUEEN spawns a single INSPECTOR worker with the following context packet:

```
target_dir:   /home/user/dev/USER/PROJECTS/EVAL_AI/workflows/ORGANIZE/test_project/
session_summon_dir: (same)
recon_reports: none (no /RECON/ directory found at target root)
role_docs:
  - workflows/ORGANIZE/WORKER_INSPECTOR.md
  - workflows/SHARED/WORKER_PROTOCOL.md
  - workflows/ORGANIZE/ORGANIZE_WORKFLOW.json
```

### 2.2 INSPECTOR Phase 1 — Orientation Discovery

Per `WORKER_INSPECTOR.md §4 Phase 1`, INSPECTOR reads Tier 1 orientation files first.

**Tier 1 reads (full reads):**
- `README.md` (15 lines, full read) — confirms project is "Spin Physics Research", describes src/ + docs/ + data/ + tests/ layout, Python 3.11+.
- `pyproject.toml` (17 lines, full read) — confirms `[project] name = "spin-physics"`, Python 3.11+, dependencies: numpy, scipy, h5py, dev deps: pytest, ruff, mypy. Build system: setuptools.

**Tier 2 reads:**
- `docs/` directory inspected — `docs/overview.md` (15 lines, full read). Describes LLG simulation approach, HDF5 outputs, reproduce instructions. This is an active research document.

**Files deferred:** none — all candidate orientation files are small enough to read fully.

**References harvested from README:** "docs/", "src/main.py" — no external links to follow.

### 2.3 INSPECTOR Phase 2 — Signal Detection

INSPECTOR classifies the project along all signal axes (per `WORKER_INSPECTOR.md §4 Phase 2`):

| Axis | Evidence | Signal |
|---|---|---|
| Primary language | `pyproject.toml` found; `[project]` with Python deps; `.py` files in `src/`, `tests/` | `lang:python` |
| Project kind | `src/` present; `tests/` present; `docs/` present; CSV data file at root; research context from README | `kind:mixed-research` |
| Structure quality | 5 loose files at root (`scratch.py`, `notes2.md`, `old_design.md`, `tmp_data.csv`, `untitled.md`); 2 cruft dirs (`backup/`, `tmp/`); total 7 loose-or-cruft items | `structure:middling` (5-20 loose files) |
| Orientation docs | `README.md` found and describes structure; no `CLAUDE.md`; no ARCH doc | `orientation:partial` |
| Cruft patterns | `scratch.py` (scratch* pattern), `notes2.md` (notes[0-9]*.md pattern), `tmp_data.csv` (tmp* pattern), `old_design.md` ("SUPERSEDED" marker in content), `backup/` directory, `tmp/` directory | `cruft: ["scratch.py", "notes2.md", "old_design.md", "tmp_data.csv", "untitled.md", "backup/", "tmp/"]` |
| Prior ORGANIZE artefacts | No `.organize.json`, no `.delete/`, no `.archive/` | `prior_state:clean` |
| RDC/SDLC artefacts | No `MASTER.md`, no `INPROGRESS.md`, no `PHASE_*` files | `workflow_artefacts: none` |
| Ignore-path candidates | `.venv/**` (Python), `__pycache__/**`, `*.pyc`, `.pytest_cache/**`, `.mypy_cache/**`, `.ipynb_checkpoints/**` | `ignore_seeds: [see Phase 5]` |
| Root invariants | `.claude/` present, `.claude-flow/` present, `.hive-mind/` present, `.mcp.json` present | `root_invariants: all 4 present` |

### 2.4 INSPECTOR Phase 3 — Template Proposal

Per `WORKER_INSPECTOR.md §4 Phase 3` and `ORGANIZE_TEMPLATE_FORMAT.md §5`:

INSPECTOR scores each template against observed signals:
- `mixed-research` signals: `kind:mixed-research`, `lang:python`, `structure:src_docs_data`
- Observed signals match: `kind:mixed-research` (exact), `lang:python` (exact)
- Score: 2 direct signal matches → `template_confidence: MEDIUM`
- No other template scores higher for this combination

**Proposed template:** `mixed-research`
**Rationale:** Project is a research codebase with Python code (`src/`), prose documentation (`docs/`), and data artifacts (`tmp_data.csv` at root, planned `data/` per README). The `kind:mixed-research` signal is a direct match. The template is appropriate because it is ASK_USER-heavy, matching the project's per-file judgment requirements.

### 2.5 INSPECTOR Phase 4 — Seed Rules

INSPECTOR proposes rules seeded from the `mixed-research` template (`templates/mixed-research.json`) plus project-specific observations. These will be merged by QUEEN (per `ORGANIZE_TEMPLATE_FORMAT.md §8`):

**Full proposed rules:**

```json
[
  {
    "id": "r1",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "src/**/*",
    "destination": "IN_PLACE",
    "priority": 100,
    "active": true,
    "note": "All files under src/ are at their canonical location — code area"
  },
  {
    "id": "r2",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "docs/**/*.md",
    "destination": "IN_PLACE",
    "priority": 95,
    "active": true,
    "note": "Prose documents under docs/ are at their canonical location"
  },
  {
    "id": "r3",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "tests/**/*",
    "destination": "IN_PLACE",
    "priority": 85,
    "active": true,
    "note": "Test files under tests/ are at their canonical location"
  },
  {
    "id": "r4",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "{README,CHANGELOG,LICENSE,CONTRIBUTING}.md",
    "destination": "IN_PLACE",
    "priority": 80,
    "active": true,
    "note": "Standard root-level orientation files are canonically at root"
  },
  {
    "id": "r5",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "*.py",
    "destination": "src/",
    "priority": 60,
    "active": true,
    "note": "Loose Python files at project root are likely misplaced source — MOVE_TO src/; ASK_USER if the file looks like a convention file (setup.py, conftest.py)"
  },
  {
    "id": "r6",
    "created": "2026-04-18T10:00:00Z",
    "kind": "glob",
    "pattern": "*.md",
    "destination": "IN_PLACE",
    "priority": 40,
    "active": true,
    "note": "Loose markdown at root — ASK_USER always; could be docs/, .archive, or active design note"
  },
  {
    "id": "r7",
    "created": "2026-04-18T10:00:00Z",
    "kind": "regex",
    "pattern": "^[^/]+\\.(csv|json|parquet|tsv|jsonl)$",
    "destination": "IN_PLACE",
    "priority": 40,
    "active": true,
    "note": "Loose data files at root — ASK_USER; could be data/, config, or experiment output"
  },
  {
    "id": "r8",
    "created": "2026-04-18T10:00:00Z",
    "kind": "regex",
    "pattern": "(tmp|scratch)[^/]*$",
    "destination": ".delete",
    "priority": 30,
    "active": true,
    "note": "Files matching tmp* or scratch* are temporary — QUARANTINE:.delete; ASK_USER before ratifying"
  },
  {
    "id": "r9",
    "created": "2026-04-18T10:00:00Z",
    "kind": "regex",
    "pattern": "^(old_[^/]+|backup/|v[0-9]+/)",
    "destination": "IN_PLACE",
    "priority": 25,
    "active": true,
    "note": "old_*, backup/, or v<N>/ prefixes indicate historical snapshots — ASK_USER; likely .archive"
  },
  {
    "id": "r10",
    "created": "2026-04-18T10:00:00Z",
    "kind": "hint",
    "pattern": "Markdown files that contain a 'SUPERSEDED', 'DEPRECATED', or 'OBSOLETE' marker within the first 10 lines are historical — quarantine to .archive.",
    "destination": ".archive",
    "priority": 55,
    "active": true,
    "note": "Explicit self-identification as superseded — quarantine to .archive for intellectual preservation"
  }
]
```

### 2.6 INSPECTOR Phase 5 — Ignore-Path Seeds

Per `WORKER_INSPECTOR.md §4 Phase 5`:

**Mandatory root-invariant seeds (always included, not user-configurable):**
- `.claude/**`
- `.claude-flow/**`
- `.hive-mind/**`
- `.mcp.json`

**Conditional seeds (based on Python signals):**
- `.git/**`
- `.venv/**`
- `**/__pycache__/**`
- `**/*.pyc`
- `.ipynb_checkpoints/**`
- `.pytest_cache/**`
- `.mypy_cache/**`
- `.delete/**`
- `.archive/**`

### 2.7 INSPECTOR Phase 6 — Initial Classification Preview

INSPECTOR performs a dry-run on visible loose files (per `WORKER_INSPECTOR.md §4 Phase 6`). This is a preview only — TRIAGE will re-classify during actual MAINTENANCE.

**Files classified (10 most prominent):**

| File | Would-be Verdict | Rationale |
|---|---|---|
| `scratch.py` | `QUARANTINE:.delete` | Matches rule r8 `(tmp\|scratch)[^/]*$`; content confirms scratch use ("# scratch") |
| `notes2.md` | `QUARANTINE:.archive` | Rule r9 `(old_\|backup/\|v[0-9]+/)` does not match; however, rule r10 (hint) would apply if content read — but r6 fires first at priority 40 → `ASK_USER`; note: user may want to bump r10 to priority 60 to catch this |
| `old_design.md` | `QUARANTINE:.archive` | Rule r10 (hint, priority 55): content contains "SUPERSEDED BY docs/overview.md" at line 1 — evidence confirmed |
| `tmp_data.csv` | `QUARANTINE:.delete` | Matches rule r8 `(tmp\|scratch)[^/]*$`; basename `tmp_data.csv` matches `tmp*` pattern |
| `untitled.md` | `ASK_USER` | Matches rule r6 (loose `*.md` at root, priority 40); no clear evidence for archive vs docs vs active use; title suggests authorial uncertainty |
| `backup/old_code.py` | `ASK_USER` | Rule r9 `^(old_\|backup/)` matches path prefix `backup/`; destination is IN_PLACE by rule but note says ASK_USER — likely `.archive` |
| `tmp/experiment.py` | `QUARANTINE:.delete` | Rule r8: `(tmp\|scratch)[^/]*$` matches basename `experiment.py` within `tmp/` directory; or the directory name `tmp/` itself matches; content confirms temporary nature |
| `src/main.py` | `KEEP_IN_PLACE` | Rule r1 `src/**/*` → IN_PLACE; file is already at canonical location |
| `tests/test_main.py` | `KEEP_IN_PLACE` | Rule r3 `tests/**/*` → IN_PLACE; file is already at canonical location |
| `docs/overview.md` | `KEEP_IN_PLACE` | Rule r2 `docs/**/*.md` → IN_PLACE; file is already at canonical location |

**Open questions from INSPECTOR (Phase 7):**

1. `notes2.md` — filename matches `notes[0-9]*.md` cruft pattern, but no rule specifically covers this. Should the user add a regex rule `notes\\d+\\.md → .archive`? (Rule r8 catches `scratch*` and `tmp*` but not numbered notes specifically.)
2. `untitled.md` — completely unclear purpose. Could be a personal note (→ .archive), a draft doc (→ docs/), or a worthless jot (→ .delete). Recommend ASK_USER during maintenance.
3. `backup/old_code.py` — Python file inside `backup/`. Should the `backup/` directory itself be archived wholesale, or should its contents be classified individually?

**INSPECTOR confidence:**
```
template_confidence: MEDIUM
rule_confidence: MEDIUM
notes: "kind:mixed-research matched strongly; 2 signals confirmed. Rule set is a reasonable starting point;
        notes2.md coverage is a gap — no rule catches notes[0-9]*.md pattern. User may want to add."
```

**INSPECTOR fabrication audit:** zero — every signal cites a file or observable pattern.

---

## 3. QUEEN Enters WIZARD_LOOP

QUEEN received the INSPECTOR proposal block. Now QUEEN drives the 6-stage wizard (per `WIZARD_PROTOCOL.md §4`). No agents are spawned inside the wizard — this is a pure QUEEN+user dialog.

---

### Stage 1 — `present_detected_signals` (per WIZARD_PROTOCOL §4, Stage 1)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 1 of 6: Project Signals ====

INSPECTOR surveyed: workflows/ORGANIZE/test_project/

Detected signals:
  Language:          Python (pyproject.toml found; numpy/scipy/h5py deps)
  Project kind:      mixed-research (code + prose + data coexist)
  Structure quality: middling — 7 loose/cruft items at root or in non-canonical dirs
  Orientation docs:  partial — README.md found; no CLAUDE.md, no ARCH doc
  Cruft patterns:    scratch.py, notes2.md, old_design.md, tmp_data.csv, untitled.md, backup/, tmp/
  Prior ORGANIZE:    clean — no .organize.json, no .delete/, no .archive/
  Workflow artefacts:none detected
  Root invariants:   .claude/ present, .claude-flow/ present, .hive-mind/ present, .mcp.json present

Confidence: MEDIUM — mixed-research template matched on 2 of 4 signals.

Open questions from INSPECTOR:
  1. notes2.md has no covering rule — consider adding notes[0-9]*.md → .archive rule in Stage 3
  2. untitled.md is ambiguous — will surface as ASK_USER during maintenance
  3. Should backup/ be treated as a whole or file-by-file?

────────────────────────────────────────────
Options: accept | correct <field> <value> | abort | help
>
```

**User response:** `accept`

**QUEEN behavior:** Signals confirmed. Advance to Stage 2.

---

### Stage 2 — `propose_template` (per WIZARD_PROTOCOL §4, Stage 2)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 2 of 6: Template ====

Proposed template: mixed-research
Rationale: Observed lang:python + kind:mixed-research. Project has src/ (code),
docs/ (prose), and data artifacts (tmp_data.csv at root; data/ implied by README).
The mixed-research template is ASK_USER-heavy by design — appropriate for this project
where the boundary between active experiment and orphaned scratch is per-file judgment.

Available named templates:
  python-lib       — Python library (src/ + tests/)
  python-app       — Python application (flat or src/ layout)
  rust-crate       — Rust library crate (Cargo workspace)
  rust-app         — Rust application (src/main.rs)
  book-markdown    — Manuscript project (chapters/, front-matter)
  mixed-research   — Code + prose + data coexisting
  knowledge-base   — Topic-organized markdown wiki
  custom           — No named template; you define all rules in Stage 3

────────────────────────────────────────────
Options: accept | change to <template_name> | custom | abort | help
>
```

**User response:** `accept`

**QUEEN behavior:** Template `mixed-research` confirmed. Advance to Stage 3 with `mixed-research` seed rules pre-loaded.

---

### Stage 3 — `propose_seed_rules` (per WIZARD_PROTOCOL §4, Stage 3)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 3 of 6: Seed Rules ====

Proposed rules (from template: mixed-research + INSPECTOR observations):

  ID    Priority  Kind    Pattern                               Destination     Note
  ────  ────────  ──────  ────────────────────────────────────  ──────────────  ──────────────────────────────
  r1    100       glob    src/**/*                              IN_PLACE        Code area — ORGANIZE does not reorganize within src/
  r2    95        glob    docs/**/*.md                          IN_PLACE        Prose area — docs/ is the documentation home
  r3    85        glob    tests/**/*                            IN_PLACE        Test files under tests/ are canonically placed
  r4    80        glob    {README,CHANGELOG,LICENSE}.md         IN_PLACE        Standard root-level orientation files stay at root
  r10   55        hint    Markdown with SUPERSEDED/DEPRECATED   .archive        Self-identified historical docs → .archive
  r5    60        glob    *.py                                  src/            Loose Python at root → src/ (ASK_USER if convention file)
  r6    40        glob    *.md                                  IN_PLACE        Loose root .md → ASK_USER (ambiguous in research context)
  r7    40        regex   ^[^/]+\.(csv|json|parquet|tsv|jsonl)$ IN_PLACE        Loose data files at root → ASK_USER
  r8    30        regex   (tmp|scratch)[^/]*$                   .delete         tmp* / scratch* files → QUARANTINE:.delete
  r9    25        regex   ^(old_[^/]+|backup/)                  IN_PLACE        old_* / backup/ → ASK_USER (likely .archive)

10 rules total. All active by default.

INSPECTOR note: No rule covers notes[0-9]*.md. Consider adding:
  r11 | 35 | regex | notes\\d+\\.md | .archive | Numbered note files (notes2.md etc.) → .archive

────────────────────────────────────────────
Options: accept all | review | add <json> | remove <id> | edit <id> <field> <val> | abort | help
>
```

**User response:**
```
edit r8 priority 35
```

(User wants to lower the priority of the `tmp*/scratch*` rule so the hint rule r10 evaluates first for markdown files containing SUPERSEDED markers.)

**QUEEN behavior (per WIZARD_PROTOCOL §4, Stage 3 edit handling):**
- Apply edit: r8 priority → 35.
- Validate: priority is a positive integer — valid.
- Re-present Stage 3 with updated rule.

QUEEN re-presents. User reviews the updated table and types:

```
accept all
```

**QUEEN behavior:** Rules array confirmed with 10 rules (r1–r10, r8 priority now 35). Advance to Stage 4.

---

### Stage 4 — `propose_ignore_paths` (per WIZARD_PROTOCOL §4, Stage 4)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 4 of 6: Ignore Paths ====

Files matching these patterns will never be enumerated or classified.
Root invariants (.claude/, .claude-flow/, .hive-mind/, .mcp.json) are always ignored
and are NOT shown here — they are enforced at the workflow level regardless.

Proposed ignore_paths:
  .git/**
  .venv/**
  **/__pycache__/**
  **/*.pyc
  .ipynb_checkpoints/**
  .pytest_cache/**
  .mypy_cache/**
  .delete/**
  .archive/**

────────────────────────────────────────────
Options: accept | add <pattern> | remove <pattern> | edit | abort | help
>
```

**User response:** `accept`

**QUEEN behavior:** ignore_paths confirmed. Advance to Stage 5.

---

### Stage 5 — `review_initial_classification` (per WIZARD_PROTOCOL §4, Stage 5)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 5 of 6: Initial Classification Preview ====

With the proposed rules, here is how INSPECTOR classified visible loose files.
This is a PREVIEW only — no moves happen here. TRIAGE will re-classify during actual maintenance.

  File                    Would-be Verdict           Rationale
  ──────────────────────  ─────────────────────────  ────────────────────────────────────────────────
  scratch.py              QUARANTINE:.delete          rule r8 (tmp|scratch)[^/]*$: basename matches scratch*
  notes2.md               ASK_USER                   rule r6 (*.md, priority 40): no SUPERSEDED marker found;
                                                      ambiguous — could be .archive or .delete
  old_design.md           QUARANTINE:.archive         rule r10 (hint): "SUPERSEDED BY docs/overview.md"
                                                      found at line 1
  tmp_data.csv            QUARANTINE:.delete          rule r8: basename matches tmp*
  untitled.md             ASK_USER                   rule r6: no rule matches more specifically; title
                                                      implies authorial uncertainty
  backup/old_code.py      ASK_USER                   rule r9 (backup/ prefix): destination is IN_PLACE
                                                      by rule, but note says ASK_USER → likely .archive
  tmp/experiment.py       QUARANTINE:.delete          rule r8: basename experiment.py in tmp/ dir;
                                                      full path contains tmp* pattern
  src/main.py             KEEP_IN_PLACE              rule r1 (src/**/*): canonical location
  src/utils/helpers.py    KEEP_IN_PLACE              rule r1 (src/**/*): canonical location
  tests/test_main.py      KEEP_IN_PLACE              rule r3 (tests/**/*): canonical location
  docs/overview.md        KEEP_IN_PLACE              rule r2 (docs/**/*.md): canonical location

Surprises? Use 'adjust rules' to return to Stage 3.

────────────────────────────────────────────
Options: ratify | adjust rules | abort | help
>
```

**User response:**
```
ratify
```

User notes: "I expected untitled.md to surface as ASK_USER — that's correct."

**QUEEN behavior:** Preview accepted. Advance to Stage 6.

---

### Stage 6 — `cascade_decision` (per WIZARD_PROTOCOL §4, Stage 6)

QUEEN emits:

```
==== ORGANIZE BOOTSTRAP — Stage 6 of 6: Review & Cascade ====

Ready to write .organize.json to: workflows/ORGANIZE/test_project/.organize.json

Summary:
  Template:       mixed-research
  Rules:          10 active rules (r1–r10)
  Ignore paths:   9 patterns
  Created:        2026-04-18T10:15:00Z

After writing config, would you like to run 1 MAINTENANCE circuit now?
(Recommended: yes — catches anything INSPECTOR's preview identified.)

────────────────────────────────────────────
Options: cascade | halt | abort | help
>
```

**User response:** `cascade`

**QUEEN behavior:** Write `.organize.json` with ratified content; commit; proceed to MAINTENANCE with circuits=1.

---

## 4. QUEEN Writes `.organize.json`

QUEEN writes the following config to `test_project/.organize.json`:

```json
{
  "version": "0.1.0",
  "template": "mixed-research",
  "created": "2026-04-18T10:15:00Z",
  "last_run": null,
  "runs": 0,
  "rules": [
    {
      "id": "r1",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "src/**/*",
      "destination": "IN_PLACE",
      "priority": 100,
      "active": true,
      "note": "All files under src/ are at their canonical location — code area"
    },
    {
      "id": "r2",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "docs/**/*.md",
      "destination": "IN_PLACE",
      "priority": 95,
      "active": true,
      "note": "Prose documents under docs/ are at their canonical location"
    },
    {
      "id": "r3",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "tests/**/*",
      "destination": "IN_PLACE",
      "priority": 85,
      "active": true,
      "note": "Test files under tests/ are at their canonical location"
    },
    {
      "id": "r4",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "{README,CHANGELOG,LICENSE}.md",
      "destination": "IN_PLACE",
      "priority": 80,
      "active": true,
      "note": "Standard root-level orientation files are canonically at root"
    },
    {
      "id": "r5",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "*.py",
      "destination": "src/",
      "priority": 60,
      "active": true,
      "note": "Loose Python at project root → src/; ASK_USER if convention file (setup.py, conftest.py)"
    },
    {
      "id": "r6",
      "created": "2026-04-18T10:15:00Z",
      "kind": "glob",
      "pattern": "*.md",
      "destination": "IN_PLACE",
      "priority": 40,
      "active": true,
      "note": "Loose markdown at root — ASK_USER; in mixed-research context per-file judgment required"
    },
    {
      "id": "r7",
      "created": "2026-04-18T10:15:00Z",
      "kind": "regex",
      "pattern": "^[^/]+\\.(csv|json|parquet|tsv|jsonl)$",
      "destination": "IN_PLACE",
      "priority": 40,
      "active": true,
      "note": "Loose data files at root — ASK_USER; could be data/, config, or experiment output"
    },
    {
      "id": "r8",
      "created": "2026-04-18T10:15:00Z",
      "kind": "regex",
      "pattern": "(tmp|scratch)[^/]*$",
      "destination": ".delete",
      "priority": 35,
      "active": true,
      "note": "Files matching tmp* or scratch* → QUARANTINE:.delete (priority lowered to 35 per user edit)"
    },
    {
      "id": "r9",
      "created": "2026-04-18T10:15:00Z",
      "kind": "regex",
      "pattern": "^(old_[^/]+|backup/)",
      "destination": "IN_PLACE",
      "priority": 25,
      "active": true,
      "note": "old_* and backup/ prefixed items — ASK_USER; likely .archive but user decides"
    },
    {
      "id": "r10",
      "created": "2026-04-18T10:15:00Z",
      "kind": "hint",
      "pattern": "Markdown files that contain a 'SUPERSEDED', 'DEPRECATED', or 'OBSOLETE' marker within the first 10 lines are historical — quarantine to .archive.",
      "destination": ".archive",
      "priority": 55,
      "active": true,
      "note": "Explicit self-identification as superseded → .archive for intellectual preservation"
    }
  ],
  "quarantine_log": [],
  "pending_rule_candidates": [],
  "ignore_paths": [
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".ipynb_checkpoints/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".delete/**",
    ".archive/**"
  ],
  "notes": "Spin Physics Research project. Mixed code + prose + data. Template: mixed-research. Rule r8 priority lowered to 35 (from 30) per user edit in BOOTSTRAP wizard so hint rule r10 evaluates before scratch/tmp catch-all for .md files."
}
```

### 4.1 Git commit

QUEEN commits:

```
git mv (none — config is a new file)
git add test_project/.organize.json
git commit -m "organize: bootstrap config for test_project"
```

---

## 5. Cascade into MAINTENANCE

User chose `cascade` in Stage 6. QUEEN proceeds immediately to MAINTENANCE mode with `circuits=1`.

Trace continues in `MAINTENANCE_WALKTHROUGH.md`.

---

*End of BOOTSTRAP_WALKTHROUGH.md*
