# MAINTENANCE_WALKTHROUGH — Single-Circuit Mental Trace

**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Target project:** `workflows/ORGANIZE/test_project/`
**Mode:** MAINTENANCE (cascade from BOOTSTRAP, or standalone re-invocation)
**Circuits:** 1 (default)
**Trace type:** Mental trace — no actual moves executed; counts and verdicts are plausible given test_project layout.
**Spec references:** ORGANIZE_WORKFLOW.json §flow.maintenance, BATCHING_HEURISTIC.md, RATIFICATION_UI_SPEC.md, WORKER_TRIAGE.md

---

## 1. QUEEN Loads `.organize.json`

QUEEN reads `test_project/.organize.json` (written in BOOTSTRAP_WALKTHROUGH.md §4).

**Config summary:**
- `version`: `0.1.0`
- `template`: `mixed-research`
- `created`: `2026-04-18T10:15:00Z`
- `last_run`: `null` (first MAINTENANCE run)
- `runs`: `0`
- `rules`: 10 active rules (r1–r10)
- `ignore_paths`: 9 patterns
- `quarantine_log`: empty
- `pending_rule_candidates`: empty

QUEEN reports:

```
ORGANIZE_WORKFLOW mode engaged. MAINTENANCE pass queued.
Config: test_project/.organize.json (10 rules). Circuits: 1. Proceeding.
```

---

## 2. QUEEN Plans Circuit

### 2.1 Pre-batching filters (per BATCHING_HEURISTIC.md §2)

**Filter 1 — Root-invariant filter (mandatory, silent):**
QUEEN removes from consideration unconditionally:
- `.claude/` and contents
- `.claude-flow/` and contents
- `.hive-mind/` and contents
- `.mcp.json`

These 4 items are removed before any batching logic. They never reach TRIAGE.

**Filter 2 — Ignore-path filter:**
QUEEN removes files matching `ignore_paths`:
- `.git/**` — no `.git/` directory in test_project (not a real git repo for this trace)
- `.venv/**` — no `.venv/` present
- `**/__pycache__/**` — none present
- `**/*.pyc` — none present
- `.ipynb_checkpoints/**` — none present
- `.pytest_cache/**` — none present
- `.mypy_cache/**` — none present
- `.delete/**` — no `.delete/` directory (first run)
- `.archive/**` — no `.archive/` directory (first run)

**Eligible population after both filters:**

Root level:
- `README.md`
- `pyproject.toml`
- `scratch.py`
- `notes2.md`
- `old_design.md`
- `tmp_data.csv`
- `untitled.md`
- `TEST_PROJECT_README.md`

Subdirectory files:
- `src/__init__.py`
- `src/main.py`
- `src/utils/__init__.py`
- `src/utils/helpers.py`
- `tests/__init__.py`
- `tests/test_main.py`
- `docs/overview.md`
- `backup/old_code.py`
- `tmp/experiment.py`

**Total eligible: 18 files**

### 2.2 Batching (per BATCHING_HEURISTIC.md §3)

Algorithm: depth-first traversal, alphabetical within each directory, BATCH_CAP=30.

With 18 total files and BATCH_CAP=30, all files fit in a single batch. However, per the heuristic's intent (coherent structural context), QUEEN groups by subtree to give TRIAGE meaningful context:

```
Batch A — root-level files (8 files):
  README.md, TEST_PROJECT_README.md, notes2.md, old_design.md,
  pyproject.toml, scratch.py, tmp_data.csv, untitled.md

Batch B — src/ + tests/ + docs/ (9 files):
  src/__init__.py, src/main.py, src/utils/__init__.py, src/utils/helpers.py,
  tests/__init__.py, tests/test_main.py,
  docs/overview.md

Batch C — backup/ + tmp/ (2 files):
  backup/old_code.py, tmp/experiment.py
```

Note: All 18 files fit below BATCH_CAP=30, so we could use one batch. QUEEN splits by subtree for FLAG_NEW_RULE signal coherence and to give each TRIAGE worker meaningful structural context (per BATCHING_HEURISTIC.md §3, note on traversal order). Result: 3 batches.

---

## 3. QUEEN Spawns TRIAGE_WAVE

QUEEN spawns 3 TRIAGE workers in parallel (single message, `run_in_background: true` per `ORGANIZE_WORKFLOW.json §units.TRIAGE_WAVE`).

**Context packet sent to each TRIAGE worker:**
- `batch`: their assigned file list
- `config`: full `.organize.json` contents
- `target_dir`: `workflows/ORGANIZE/test_project/`
- Role docs: `WORKER_TRIAGE.md`, `WORKER_PROTOCOL.md`, `ORGANIZE_WORKFLOW.json`

All 3 workers run in parallel. QUEEN waits for all to return before aggregation.

---

### 3.1 TRIAGE Worker A — Root-level files

**Rules loaded (sorted by priority descending):**
r1(100) → r2(95) → r3(85) → r4(80) → r5(60) → r10(55) → r6(40) → r7(40) → r8(35) → r9(25)

**File evaluations:**

**`README.md`**
- r1: `src/**/*` — no match (root file)
- r2: `docs/**/*.md` — no match
- r3: `tests/**/*` — no match
- r4: `{README,CHANGELOG,LICENSE}.md` — **match!** `README.md` matches `README.md` in the brace set
- Result: `KEEP_IN_PLACE`
- Evidence: "rule r4 (glob: {README,CHANGELOG,LICENSE}.md → IN_PLACE) matched"
- Confidence: HIGH

**`TEST_PROJECT_README.md`**
- r1–r4: no match
- r5: `*.py` — no match (not .py)
- r10 (hint, priority 55): read file — "# ORGANIZE Test Project"; no SUPERSEDED/DEPRECATED/OBSOLETE marker in first 10 lines
- r6: `*.md` — **match!** destination IN_PLACE — but rule note says ASK_USER
- Per WORKER_TRIAGE.md §5 Step 2c, the rule destination is IN_PLACE, so verdict is KEEP_IN_PLACE... but per the mixed-research template note (t8), loose root .md → ASK_USER always.
- TRIAGE emits ASK_USER: "root-level .md with no specific rule beyond generic catch-all; file content suggests it is organizational documentation for the test project rather than for the research project; placement is unusual but intentional"
- Result: `ASK_USER`
- Evidence: "matches rule r6 (*.md, priority 40); content describes test infrastructure; unclear if it should remain at root or move to docs/"
- Confidence: MEDIUM

**`notes2.md`**
- r1–r4: no match
- r5: no match (not .py)
- r10 (hint, priority 55): read file — "# Notes — 2026-01-15"; check first 10 lines for SUPERSEDED/DEPRECATED/OBSOLETE. Line 10: "These notes are superseded by the 2026-02 planning doc in docs/." — TRIAGE checks: "superseded" appears at line 10, within the first 10 lines window.
- Condition satisfied at MEDIUM confidence (the word "superseded" appears but as a sentence clause, not a formal marker — TRIAGE interprets conservatively per WORKER_TRIAGE.md §4 hint interpretation §4.3 step 4)
- Downgrade to ASK_USER: "hint rule r10 partially matches ('superseded' at line 10) but not a formal SUPERSEDED: marker; presenting to user rather than auto-archiving"
- Result: `ASK_USER`
- Options: a) QUARANTINE:.archive, b) KEEP_IN_PLACE, c) QUARANTINE:.delete
- Evidence: "file contains 'superseded' at line 10 in sentence context; not a formal SUPERSEDED: marker; TRIAGE conservative per hint discipline"
- Confidence: MEDIUM

**`old_design.md`**
- r1–r4: no match
- r5: no match
- r10 (hint, priority 55): read file — first line: "SUPERSEDED BY docs/overview.md — kept for reference only." — formal SUPERSEDED marker at line 1
- Condition satisfied at HIGH confidence
- Result: `QUARANTINE:.archive`
- Evidence: "file contains 'SUPERSEDED BY docs/overview.md' at line 1 — formal superseded marker"
- Confidence: HIGH

**`pyproject.toml`**
- r1–r9: no match (toml extension not covered by any rule)
- No rule matches → no MOVE_TO prescribed, no pattern → fallback per WORKER_TRIAGE.md §5 Step 2d
- File is a standard Python project manifest at root — clearly well-placed by convention
- TRIAGE emits KEEP_IN_PLACE: "pyproject.toml is a standard Python project configuration file; canonically at project root by Python packaging convention; no rule covers it because it is obviously IN_PLACE"
- Result: `KEEP_IN_PLACE`
- Evidence: "no rule matches; file is pyproject.toml — Python packaging convention mandates root placement; absence of any applicable move rule is itself evidence of canonical placement"
- Confidence: HIGH

**`scratch.py`**
- r1: `src/**/*` — no match (root file)
- r5: `*.py` (priority 60) — **match!** destination `src/`
  - But r8 `(tmp|scratch)[^/]*$` (priority 35) has lower priority; r5 fires first
  - r5 prescribes MOVE_TO:src/ — however, TRIAGE reads the file: content confirms "# scratch — quick test". TRIAGE notes the rule's caveat: "ASK_USER if the file looks like a one-off experiment"
  - TRIAGE considers: this IS an experiment file, not source code
  - Per `WORKER_TRIAGE.md §4, honest-ambiguity check`: two plausible destinations (src/ per r5, .delete per r8) exist
- Result: `ASK_USER`
- Options: a) MOVE_TO src/ (rule r5), b) QUARANTINE:.delete (rule r8 — scratch* pattern), c) KEEP_IN_PLACE
- Evidence: "rule r5 (*.py → src/) fired first; however file content confirms scratch use ('# scratch — quick test of noise term formula'); QUARANTINE:.delete is the more appropriate verdict but requires user confirmation per mixed-research template conservative approach"
- Confidence: MEDIUM

**`tmp_data.csv`**
- r1–r5: no match (.csv, not .py or matching src/**)
- r10 (hint): not markdown — hint rule explicitly targets markdown files → skip
- r6: `*.md` — no match (.csv)
- r7: `^[^/]+\.(csv|json|parquet|tsv|jsonl)$` (priority 40) — **match!** `tmp_data.csv` is at root and ends in `.csv`
- Rule r7 destination is IN_PLACE, note says ASK_USER
- But also: r8 `(tmp|scratch)[^/]*$` (priority 35) fires after r7 with lower priority
- First match wins: r7 (priority 40) fires; destination IN_PLACE → KEEP_IN_PLACE... but with ASK_USER note
- TRIAGE: file content shows "run_id,temperature,lattice_n,tau_relaxation" — clearly a data file; basename starts with `tmp_` suggesting temporary
- Honest ambiguity between KEEP_IN_PLACE and QUARANTINE:.delete
- Result: `ASK_USER`
- Options: a) MOVE_TO data/ (should probably be in data/ per mixed-research template), b) QUARANTINE:.delete (tmp prefix suggests temporary), c) KEEP_IN_PLACE
- Evidence: "rule r7 (regex: loose data files at root → ASK_USER) matched; basename 'tmp_data.csv' contains 'tmp_' prefix suggesting temporary; content is a 4-row relaxation time CSV — small dataset, possibly staging data for data/ directory"
- Confidence: MEDIUM

**`untitled.md`**
- r1–r4: no match
- r10 (hint, priority 55): read — "# (untitled)" + content: personal reading notes about Sachdev. No SUPERSEDED marker.
- r6: `*.md` (priority 40) — match; destination IN_PLACE; but note says ASK_USER
- Content: personal reading notes, unclear scope
- Result: `ASK_USER`
- Options: a) QUARANTINE:.archive, b) MOVE_TO docs/, c) KEEP_IN_PLACE, d) QUARANTINE:.delete
- Evidence: "file titled '(untitled)' — authorial uncertainty implied; content is personal reading notes ('quick thoughts from a reading session') that may or may not belong in the research project; per mixed-research template conservative approach, per-file judgment required"
- Confidence: HIGH (HIGH confidence that ASK_USER is the right verdict, not HIGH confidence in any specific destination)

**Batch A summary:**
- KEEP_IN_PLACE: 2 (README.md, pyproject.toml)
- QUARANTINE:.archive: 1 (old_design.md)
- ASK_USER: 5 (TEST_PROJECT_README.md, notes2.md, scratch.py, tmp_data.csv, untitled.md)

---

### 3.2 TRIAGE Worker B — src/ + tests/ + docs/

**File evaluations:**

**`src/__init__.py`** — rule r1 (`src/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r1 matched: src/**/* → IN_PLACE". Confidence: HIGH.

**`src/main.py`** — rule r1 (`src/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r1 matched: src/**/* → IN_PLACE; file is the project's main entry point". Confidence: HIGH.

**`src/utils/__init__.py`** — rule r1 (`src/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r1 matched". Confidence: HIGH.

**`src/utils/helpers.py`** — rule r1 (`src/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r1 matched: utility module correctly placed under src/utils/". Confidence: HIGH.

**`tests/__init__.py`** — rule r3 (`tests/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r3 matched: tests/**/* → IN_PLACE". Confidence: HIGH.

**`tests/test_main.py`** — rule r3 (`tests/**/*`) → `KEEP_IN_PLACE`. Evidence: "rule r3 matched; file contains 'import pytest' and 'from src.main import run' — confirmed test file". Confidence: HIGH.

**`docs/overview.md`** — rule r2 (`docs/**/*.md`) → `KEEP_IN_PLACE`. Evidence: "rule r2 matched: docs/**/*.md → IN_PLACE; file is an active project overview". Confidence: HIGH.

**Batch B summary:**
- KEEP_IN_PLACE: 7 (all files)
- No ASK_USER, no QUARANTINE, no MOVE_TO, no FLAG_NEW_RULE

---

### 3.3 TRIAGE Worker C — backup/ + tmp/

**`backup/old_code.py`**
- r1: `src/**/*` — no match (path starts with `backup/`)
- r9: `^(old_[^/]+|backup/)` (priority 25) — **match!** path starts with `backup/`
- Rule r9 destination is IN_PLACE; note says "ASK_USER — likely .archive"
- TRIAGE reads: "# backup — old MC engine (archived 2026-02-03)"; content clearly archived
- Honest ambiguity: rule says IN_PLACE but note overrides; content confirms historical value
- Result: `QUARANTINE:.archive`
- Evidence: "rule r9 matched (path prefix backup/); file content: '# backup — old MC engine (archived 2026-02-03)' — explicit archival notation at line 1; content is a class that was abandoned in favour of LLG. Historical value confirmed; archival is appropriate"
- Confidence: HIGH

**`tmp/experiment.py`**
- r1: no match
- r5: `*.py` (priority 60) — match! destination src/
  - But file is in `tmp/` directory — clearly temporary context; r5 has caveat "ASK_USER if looks like one-off experiment"
  - TRIAGE reads content: "# tmp — one-off experiment to test noise scaling"; confirmed experiment file
  - r8: `(tmp|scratch)[^/]*$` (priority 35) would match `tmp/experiment.py` because parent dir is `tmp`
  - First match is r5 (priority 60); but content analysis overrides: per honest-ambiguity discipline
- Result: `QUARANTINE:.delete`
- Evidence: "rule r5 (*.py → src/) fired first; however file content: '# tmp — one-off experiment to test noise scaling / Created 2026-03-10; results pasted into lab notebook; can delete.' — explicit 'can delete' annotation and temporary context confirmed; overriding r5 → QUARANTINE:.delete"
- Confidence: HIGH

**Batch C summary:**
- QUARANTINE:.archive: 1 (backup/old_code.py)
- QUARANTINE:.delete: 1 (tmp/experiment.py)

---

## 4. QUEEN Aggregates + Presents Ratification Dialog

All 3 TRIAGE workers have returned. QUEEN aggregates across batches:

**Cross-batch FLAG_NEW_RULE aggregation (per BATCHING_HEURISTIC.md §6):**
No FLAG_NEW_RULE verdicts were emitted in any batch. Skip aggregation.

**Final tally:**
- KEEP_IN_PLACE: 9 (README.md, pyproject.toml, src/\*, tests/\*, docs/overview.md)
- QUARANTINE:.archive: 2 (old_design.md, backup/old_code.py)
- QUARANTINE:.delete: 1 (tmp/experiment.py)
- ASK_USER: 5 (TEST_PROJECT_README.md, notes2.md, scratch.py, tmp_data.csv, untitled.md)
- MOVE_TO: 0
- FLAG_NEW_RULE: 0

**Wait — scratch.py was ASK_USER in Batch A.** For the ratification dialog, QUEEN will present this as an ASK_USER requiring individual decision. However the verdicts show the correct distribution for this test project.

QUEEN presents the ratification dialog (per RATIFICATION_UI_SPEC.md §3):

---

```
==== ORGANIZE MAINTENANCE — CIRCUIT 1 of 1 ====
Target: workflows/ORGANIZE/test_project/
Files enumerated: 18  |  Ignored: 4 (root invariants)  |  Classified: 18
Workers in wave: 3 TRIAGE workers

KEEP_IN_PLACE: 9 files — no action needed.
```

---

```
QUARANTINE:.archive (2 files — proposed historical preservation):
  old_design.md  (rule r10 hint: "SUPERSEDED BY docs/overview.md" at line 1)
  backup/old_code.py  (rule r9: backup/ prefix; content confirms "archived 2026-02-03")

Files will be moved to: test_project/.archive/
(Reversible — preserved indefinitely. You manage .archive/ manually.)

Options: [ratify all / review / skip / abort]
>
```

**User response:** `ratify all`

---

```
QUARANTINE:.delete (1 file — proposed garbage):
  tmp/experiment.py  (rule r5/r8: one-off experiment; file says "can delete")

Files will be moved to: test_project/.delete/
(Reversible — nothing is deleted. You manage .delete/ manually.)

Options: [ratify all / review / skip / abort]
>
```

**User response:** `ratify all`

---

```
ASK_USER (5 files need individual decisions):

  File 1 of 5: TEST_PROJECT_README.md
  TRIAGE rationale: "root-level .md; content describes test infrastructure for ORGANIZE
    workflow; unclear if it should remain at root or move to docs/"
  Options observed by TRIAGE:
    a) KEEP_IN_PLACE — leave at current location
    b) MOVE_TO docs/ — move to docs/ directory
    c) QUARANTINE:.archive — preserve historically
    d) QUARANTINE:.delete — discard
  Choose: [a / b / c / d / skip this file / abort]
  >
```

**User response:** `a`

```
  Decision recorded: TEST_PROJECT_README.md → KEEP_IN_PLACE
```

---

```
  File 2 of 5: notes2.md
  TRIAGE rationale: "hint rule r10 partially matched ('superseded' at line 10 in sentence
    context, not formal marker); numbered notes file — likely historical but needs confirmation"
  Options observed by TRIAGE:
    a) QUARANTINE:.archive — preserve as historical notes (recommended)
    b) KEEP_IN_PLACE — leave at current location
    c) QUARANTINE:.delete — discard entirely
  Choose: [a / b / c / skip this file / abort]
  >
```

**User response:** `a`

```
  Decision recorded: notes2.md → QUARANTINE:.archive
```

---

```
  File 3 of 5: scratch.py
  TRIAGE rationale: "rule r5 (*.py → src/) fired first; but file content confirms scratch use
    ('# scratch — quick test of noise term formula'); QUARANTINE:.delete is more appropriate
    but requires confirmation per mixed-research conservative approach"
  Options observed by TRIAGE:
    a) QUARANTINE:.delete — discard (scratch content confirmed) [RECOMMENDED]
    b) MOVE_TO src/ — move to src/ (rule r5 — but content confirms it's not source code)
    c) KEEP_IN_PLACE — leave at root
  Choose: [a / b / c / skip this file / abort]
  >
```

**User response:** `a`

```
  Decision recorded: scratch.py → QUARANTINE:.delete
```

---

```
  File 4 of 5: tmp_data.csv
  TRIAGE rationale: "rule r7 (loose data files at root → ASK_USER) matched; basename 'tmp_data.csv'
    contains 'tmp_' prefix suggesting temporary; content is a 4-row relaxation time CSV"
  Options observed by TRIAGE:
    a) MOVE_TO data/ — move to data/ directory (where datasets belong per template)
    b) QUARANTINE:.delete — discard (tmp prefix + small staging data)
    c) KEEP_IN_PLACE — leave at root
  Choose: [a / b / c / skip this file / abort]
  >
```

**User response:** `b`

```
  Decision recorded: tmp_data.csv → QUARANTINE:.delete
```

---

```
  File 5 of 5: untitled.md
  TRIAGE rationale: "file titled '(untitled)' — authorial uncertainty implied; content is
    personal reading notes that may or may not belong in the research project"
  Options observed by TRIAGE:
    a) QUARANTINE:.archive — preserve as personal research notes
    b) MOVE_TO docs/ — integrate into documentation
    c) KEEP_IN_PLACE — leave at root (if it's an active reference)
    d) QUARANTINE:.delete — discard
  Choose: [a / b / c / d / skip this file / abort]
  >
```

**User response:** `a`

```
  Decision recorded: untitled.md → QUARANTINE:.archive
```

---

### 4.1 Post-Ratification Summary

```
==== RATIFICATION SUMMARY — CIRCUIT 1 of 1 ====

Ready to execute:
  QUARANTINE:.delete:   3 files (tmp/experiment.py, scratch.py, tmp_data.csv)
  QUARANTINE:.archive:  4 files (old_design.md, backup/old_code.py, notes2.md, untitled.md)
  KEEP_IN_PLACE:        1 ASK_USER decision (TEST_PROJECT_README.md)
  New rules ratified:   0

Skipped / left in place: 0 files unresolved

Commit message will be:
  organize: circuit 1 of 1 — moved 0, quarantined 7, archived 4

Proceed with execution? [yes / abort]
>
```

**User response:** `yes`

---

## 5. QUEEN Executes Moves

Per `ORGANIZE_WORKFLOW.json §hard_rules.git_mv_only` and `RATIFICATION_UI_SPEC.md §5`:

### 5.1 Create quarantine directories

QUEEN lazily creates `.delete/` and `.archive/` (first use in this project):

```
mkdir test_project/.delete/
mkdir test_project/.delete/tmp/
mkdir test_project/.archive/
mkdir test_project/.archive/backup/
```

QUEEN adds both to `.gitignore` (creating it if absent):

```
echo ".delete/" >> test_project/.gitignore
echo ".archive/" >> test_project/.gitignore
```

### 5.2 Execute git mv commands (path-mirroring per `ORGANIZE_WORKFLOW.json §document_conventions.quarantine_dirs`)

**QUARANTINE:.delete moves:**

```bash
git mv test_project/tmp/experiment.py test_project/.delete/tmp/experiment.py
git mv test_project/scratch.py        test_project/.delete/scratch.py
git mv test_project/tmp_data.csv      test_project/.delete/tmp_data.csv
```

**QUARANTINE:.archive moves:**

```bash
git mv test_project/old_design.md          test_project/.archive/old_design.md
git mv test_project/backup/old_code.py     test_project/.archive/backup/old_code.py
git mv test_project/notes2.md              test_project/.archive/notes2.md
git mv test_project/untitled.md            test_project/.archive/untitled.md
```

**Path mirroring confirmed:** `backup/old_code.py` → `.archive/backup/old_code.py` (original relative path preserved under quarantine root).

---

## 6. QUEEN Appends to `.organize.json`

### 6.1 quarantine_log entries (7 new entries)

QUEEN appends quarantine_log entries for all 7 quarantined files:

```json
[
  {
    "run": 1, "timestamp": "2026-04-18T10:45:00Z",
    "file": "tmp/experiment.py", "destination": ".delete",
    "reason": "rule r5 (*.py → src/) fired but content confirms one-off experiment ('can delete' annotation at line 4); user ratified QUARANTINE:.delete",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:01Z",
    "file": "scratch.py", "destination": ".delete",
    "reason": "rule r5 (*.py → src/) fired but content confirms scratch ('# scratch — quick test of noise term formula'); user ratified QUARANTINE:.delete",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:02Z",
    "file": "tmp_data.csv", "destination": ".delete",
    "reason": "rule r7 (loose data files ASK_USER); basename 'tmp_data.csv' indicates temporary staging data; user ratified QUARANTINE:.delete",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:03Z",
    "file": "old_design.md", "destination": ".archive",
    "reason": "rule r10 hint: 'SUPERSEDED BY docs/overview.md' at line 1 — formal superseded marker confirmed; high confidence QUARANTINE:.archive",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:04Z",
    "file": "backup/old_code.py", "destination": ".archive",
    "reason": "rule r9 (backup/ prefix); content: '# backup — old MC engine (archived 2026-02-03)'; historical value confirmed; user ratified QUARANTINE:.archive",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:05Z",
    "file": "notes2.md", "destination": ".archive",
    "reason": "hint rule r10 partially matched ('superseded' in sentence context at line 10); user confirmed archival as historical notes",
    "ratified_by": "user", "untracked_at_move": false
  },
  {
    "run": 1, "timestamp": "2026-04-18T10:45:06Z",
    "file": "untitled.md", "destination": ".archive",
    "reason": "rule r6 (loose .md ASK_USER); content is personal reading notes; user elected to preserve as .archive",
    "ratified_by": "user", "untracked_at_move": false
  }
]
```

### 6.2 Config metadata updated

```json
{
  "last_run": "2026-04-18T10:45:10Z",
  "runs": 1
}
```

---

## 7. QUEEN Commits

```bash
git add test_project/.organize.json test_project/.gitignore
git commit -m "organize: circuit 1 of 1 — moved 0, quarantined 7, archived 4"
```

(Note: "moved 0" refers to MOVE_TO canonical-location moves, which were zero this circuit. The 7 quarantine moves are the main action.)

---

## 8. Verdict: CLEAN_RUN

Per `ORGANIZE_WORKFLOW.json §verdicts.maintenance_catalog.CLEAN_RUN`:
> "All N requested circuits completed. All TRIAGE verdicts processed. All ratified moves executed. `.organize.json` updated."

QUEEN emits:

```
CLEAN_RUN — Circuit 1 of 1 complete.
  Quarantined to .delete: 3 files
  Quarantined to .archive: 4 files
  KEEP_IN_PLACE: 11 files (9 auto + 2 ASK_USER decisions)
  New rules: 0

Remaining at root: README.md, TEST_PROJECT_README.md, pyproject.toml
Remaining in src/: main.py, utils/helpers.py, utils/__init__.py, __init__.py
Remaining in tests/: test_main.py, __init__.py
Remaining in docs/: overview.md

Project is now organized per current rules.
Next invocation will run MAINTENANCE (config exists).
```

---

*End of MAINTENANCE_WALKTHROUGH.md*
