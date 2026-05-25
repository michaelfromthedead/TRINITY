# MULTI_CIRCUIT_WALKTHROUGH — N=3 Circuits with FLAG_NEW_RULE

**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Target project:** `workflows/ORGANIZE/test_project/` (extended with 5 additional files)
**Mode:** MAINTENANCE (existing `.organize.json` from BOOTSTRAP_WALKTHROUGH.md)
**Circuits:** 3 (user-specified)
**Trace type:** Mental trace — no actual moves executed; all verdicts and counts are plausible.
**Spec references:** ORGANIZE_WORKFLOW.json §flow.maintenance, BATCHING_HEURISTIC.md §6, RATIFICATION_UI_SPEC.md §3.7, WORKER_TRIAGE.md §3

---

## 1. Setup — Extended Test Project

Starting from the post-BOOTSTRAP state (`.organize.json` exists, 10 rules active), the test project has 5 additional files:

**New files added to `src/`:**
- `src/run_bench.bench.py` — benchmark script for the run function
- `src/profile.bench.py` — profiling script for hotspot analysis
- `src/check.bench.py` — validation benchmark (checks correctness under load)

**New files added to `scripts/`:**
- `scripts/deploy.sh` — deployment helper script
- `scripts/backup.sh` — backup automation script

Note: `scripts/` directory is new — it did not exist in the original test project. The existing 10 rules have no rule for `scripts/` (no `scripts/**/*.sh` IN_PLACE rule and no general `**/*.sh` rule).

**Current file inventory for MAINTENANCE:**

After the single-circuit run in MAINTENANCE_WALKTHROUGH (which quarantined 7 files), the remaining files are:
- Root: `README.md`, `TEST_PROJECT_README.md`, `pyproject.toml`
- `src/`: `main.py`, `utils/helpers.py`, `utils/__init__.py`, `__init__.py`, plus 3 new `.bench.py` files
- `tests/`: `test_main.py`, `__init__.py`
- `docs/`: `overview.md`
- `scripts/`: `deploy.sh`, `backup.sh` (new directory)
- `.delete/`: `scratch.py`, `tmp_data.csv`, `tmp/experiment.py` (ignored)
- `.archive/`: `old_design.md`, `backup/old_code.py`, `notes2.md`, `untitled.md` (ignored)

**Total eligible for this walkthrough:** ~18 files (15 original survivors + 5 new, minus .delete/.archive which are in ignore_paths)

---

## 2. User Invokes 3-Circuit MAINTENANCE

User types:

```
ORGANIZE_WORKFLOW
```

QUEEN detects `.organize.json` (present). Mode: MAINTENANCE.

QUEEN asks (default N=1, but user specifies):

```
ORGANIZE_WORKFLOW mode engaged. MAINTENANCE pass queued.
Config: .organize.json (10 rules). How many circuits? [default: 1]
>
```

**User:** `3`

QUEEN confirms:

```
Config: .organize.json (10 rules). Circuits: 3. Proceeding.
```

---

## 3. Circuit 1 — FLAG_NEW_RULE Detected

### 3.1 Pre-batching filters

Root invariants filtered (silent): `.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json`
Ignore-paths filtered: `.delete/**`, `.archive/**`, `**/__pycache__/**`, etc.

**Eligible population:** 18 files across root, `src/`, `tests/`, `docs/`, `scripts/`.

### 3.2 Batching (BATCH_CAP=30)

All 18 files fit in one batch, but QUEEN splits by subtree for structural coherence:

```
Batch A — root (3 files): README.md, TEST_PROJECT_README.md, pyproject.toml
Batch B — src/ (7 files): __init__.py, main.py, utils/__init__.py, utils/helpers.py,
                           run_bench.bench.py, profile.bench.py, check.bench.py
Batch C — tests/ + docs/ + scripts/ (6 files): tests/__init__.py, tests/test_main.py,
                                                docs/overview.md,
                                                scripts/deploy.sh, scripts/backup.sh
```

Total: 3 batches. QUEEN spawns 3 TRIAGE workers in parallel.

### 3.3 TRIAGE Worker A — Root files

All 3 root files are standard. Per rules:
- `README.md` → rule r4 → `KEEP_IN_PLACE` (HIGH)
- `TEST_PROJECT_README.md` → rule r6 (*.md ASK_USER) → `ASK_USER` (MEDIUM) — same reasoning as MAINTENANCE_WALKTHROUGH §3.1
- `pyproject.toml` → no rule → `KEEP_IN_PLACE` (HIGH, by Python convention)

**Batch A summary:** KEEP_IN_PLACE: 2, ASK_USER: 1

### 3.4 TRIAGE Worker B — src/ files (including .bench.py)

**`src/__init__.py`** → rule r1 (`src/**/*`) → `KEEP_IN_PLACE` (HIGH)
**`src/main.py`** → rule r1 → `KEEP_IN_PLACE` (HIGH)
**`src/utils/__init__.py`** → rule r1 → `KEEP_IN_PLACE` (HIGH)
**`src/utils/helpers.py`** → rule r1 → `KEEP_IN_PLACE` (HIGH)

**`src/run_bench.bench.py`**:
- r1: `src/**/*` (priority 100) — **match!** Path is `src/run_bench.bench.py` which matches `src/**/*`
- r1 destination is IN_PLACE → `KEEP_IN_PLACE`
- BUT: TRIAGE reads file (full-read discipline); content would be a benchmark script. The double extension `.bench.py` is a pattern TRIAGE notices.
- Per WORKER_TRIAGE.md §5 Step 2c: r1 fires at priority 100 → `KEEP_IN_PLACE`. Rule match is unambiguous.
- However: TRIAGE notes the pattern for cross-file analysis (Step 3)
- Immediate verdict: `KEEP_IN_PLACE` (rule r1 matched)
- Evidence: "rule r1 (src/**/* → IN_PLACE) matched; note: .bench.py double extension observed — will check for FLAG_NEW_RULE pattern across batch"
- Confidence: HIGH

**`src/profile.bench.py`** → rule r1 → `KEEP_IN_PLACE` (HIGH). Pattern noted.

**`src/check.bench.py`** → rule r1 → `KEEP_IN_PLACE` (HIGH). Pattern noted.

**Post-classification cross-file analysis (WORKER_TRIAGE.md §3):**

TRIAGE re-scans: 3 files share the `.bench.py` suffix pattern (`run_bench.bench.py`, `profile.bench.py`, `check.bench.py`). All matched rule r1 (src/**/* → IN_PLACE), but all carry the `.bench.py` double-extension pattern that no explicit rule targets.

Threshold check: 3+ files matching same uncovered pattern → FLAG_NEW_RULE threshold met.

However: r1 already covered these files (IN_PLACE verdict). Per WORKER_TRIAGE.md §3, FLAG_NEW_RULE is triggered when files "fit no existing rule but a pattern is recurring." Here, r1 *did* match. TRIAGE's judgment: the files are in `src/` currently, but `.bench.py` convention suggests they belong in a `benches/` directory. The current placement under `src/` may be accidental. TRIAGE proposes a FLAG_NEW_RULE with rationale that the files would be better organized in a dedicated location.

TRIAGE upgrades the pattern observation to `FLAG_NEW_RULE`:

```
FLAG_NEW_RULE — pattern *.bench.py observed in 3 files in src/:
  src/run_bench.bench.py
  src/profile.bench.py
  src/check.bench.py

Suggested rule:
  {
    "id": "proposed-r11",
    "created": "2026-04-18T11:30:00Z",
    "kind": "glob",
    "pattern": "**/*.bench.py",
    "destination": "benches/",
    "priority": 80,
    "active": true,
    "note": "Benchmark scripts identified by .bench.py suffix — observed 3 files; propose routing to benches/ directory"
  }

Rationale: 3 files with .bench.py suffix found in src/; no dedicated benches/ directory exists.
The .bench.py double-extension convention typically indicates benchmark scripts which are
most organized in a dedicated benches/ directory, separate from production source code.
```

Note: The first file in the group (`src/run_bench.bench.py`) carries the full FLAG_NEW_RULE payload. The other two reference it per WORKER_TRIAGE.md §3: "only the FIRST file in the group carries the FLAG_NEW_RULE verdict with the full suggested_rule payload; the others reference the flag via 'see FLAG_NEW_RULE for <first-file>' in their rationale but retain ASK_USER verdict."

Updated verdicts for Batch B:
- `src/run_bench.bench.py` → `FLAG_NEW_RULE` (with suggested_rule payload)
- `src/profile.bench.py` → `ASK_USER` (see FLAG_NEW_RULE for src/run_bench.bench.py)
- `src/check.bench.py` → `ASK_USER` (see FLAG_NEW_RULE for src/run_bench.bench.py)

**Batch B summary:** KEEP_IN_PLACE: 4 (main.py, utils/\*), FLAG_NEW_RULE: 1 (covering 3 files), ASK_USER: 2 (references to the flag)

### 3.5 TRIAGE Worker C — tests/ + docs/ + scripts/

**`tests/__init__.py`** → rule r3 → `KEEP_IN_PLACE` (HIGH)
**`tests/test_main.py`** → rule r3 → `KEEP_IN_PLACE` (HIGH)
**`docs/overview.md`** → rule r2 → `KEEP_IN_PLACE` (HIGH)

**`scripts/deploy.sh`**:
- r1: `src/**/*` — no match
- r2–r10: none match `.sh` files specifically
- No rule covers `**/*.sh` → no rule matched
- TRIAGE: 2 `.sh` files in `scripts/`; threshold 3+ not met for FLAG_NEW_RULE
- `scripts/deploy.sh` — single file, no rule, `scripts/` is a plausible canonical location by convention
- Per WORKER_TRIAGE.md §5 Step 2d fallback: "ASK_USER if the file is clearly unusual and judgment is needed"
- Files in a `scripts/` directory are well-placed by convention; TRIAGE emits KEEP_IN_PLACE with LOW-rule-match note
- Result: `KEEP_IN_PLACE` with note "no explicit rule covers scripts/**/*.sh; files are in scripts/ which is a conventional location; recommend adding an IN_PLACE rule for scripts/**"
- Evidence: "no rule matched; file is in scripts/ which is a conventional location by project convention; KEEP_IN_PLACE is conservative fallback"
- Confidence: MEDIUM

**`scripts/backup.sh`** → same logic → `KEEP_IN_PLACE` (MEDIUM)

(Note: only 2 `.sh` files, below the 3+ threshold for FLAG_NEW_RULE. They surface as KEEP_IN_PLACE with recommendation to add a rule.)

**Batch C summary:** KEEP_IN_PLACE: 5 (all), no FLAG_NEW_RULE (2 .sh files below threshold)

### 3.6 QUEEN Aggregates — Circuit 1

**Cross-batch FLAG_NEW_RULE aggregation (BATCHING_HEURISTIC.md §6):**

FLAG_NEW_RULE suggestions collected:
- Batch B: `*.bench.py` pattern, 3 files, suggested rule `proposed-r11`

Single batch surfaced it — no cross-batch merge needed.

**Final circuit 1 tally:**
- KEEP_IN_PLACE: 14 (root canonical + src canonical + tests + docs + scripts)
- FLAG_NEW_RULE: 1 suggestion (covering 3 .bench.py files)
- ASK_USER: 3 (TEST_PROJECT_README.md + 2 .bench.py referencing the flag)

### 3.7 Ratification Dialog — Circuit 1

```
==== ORGANIZE MAINTENANCE — CIRCUIT 1 of 3 ====
Target: workflows/ORGANIZE/test_project/
Files enumerated: 18  |  Ignored: 4  |  Classified: 18
Workers in wave: 3 TRIAGE workers

KEEP_IN_PLACE: 14 files — no action needed.
```

```
FLAG_NEW_RULE (1 suggestion, covering 3 files total):

  Suggestion 1 of 1:
  Pattern observed: *.bench.py (3 files in src/, no existing rule)
  Suggested rule:
    id:          proposed-r11
    kind:        glob
    pattern:     **/*.bench.py
    destination: benches/
    priority:    80
    note:        "benchmark scripts identified by .bench.py suffix — route to benches/"
  Affected files:
    src/run_bench.bench.py
    src/profile.bench.py
    src/check.bench.py

  Options: [ratify rule / reject / edit <field> <value> / abort]
  >
```

**User response:** `ratify rule`

**QUEEN behavior (per RATIFICATION_UI_SPEC.md §3.7):**
1. Append `proposed-r11` to `.organize.json` rules array (as `r11`).
2. Re-classify affected files against the new rule: all 3 `.bench.py` files now match rule r11 (`**/*.bench.py` → `benches/`); current location is `src/`; destination is `benches/` → emit `MOVE_TO:benches/` for all 3.
3. Queue the MOVE_TO moves for execution.

```
FLAG_NEW_RULE ratified. Rule r11 added. Re-classifying affected files:
  src/run_bench.bench.py → MOVE_TO benches/
  src/profile.bench.py   → MOVE_TO benches/
  src/check.bench.py     → MOVE_TO benches/
```

---

```
ASK_USER (1 file needing individual decision — TEST_PROJECT_README.md):

  File 1 of 1: TEST_PROJECT_README.md
  [same rationale as MAINTENANCE_WALKTHROUGH]
  Choose: [a: KEEP_IN_PLACE / b: MOVE_TO docs/ / ...]
  >
```

**User response:** `a` (KEEP_IN_PLACE)

---

**Post-Ratification Summary — Circuit 1:**

```
==== RATIFICATION SUMMARY — CIRCUIT 1 of 3 ====

Ready to execute:
  MOVE_TO benches/:      3 files
  New rules ratified:    1 (r11: **/*.bench.py → benches/)

Skipped / left in place: TEST_PROJECT_README.md (ASK_USER → KEEP_IN_PLACE)

Commit message will be:
  organize: circuit 1 of 3 — moved 3, quarantined 0 + new rule r11 ratified

Proceed with execution? [yes / abort]
>
```

**User response:** `yes`

**QUEEN executes:**

```bash
mkdir test_project/benches/
git mv test_project/src/run_bench.bench.py test_project/benches/run_bench.bench.py
git mv test_project/src/profile.bench.py   test_project/benches/profile.bench.py
git mv test_project/src/check.bench.py     test_project/benches/check.bench.py
```

QUEEN updates `.organize.json`:
- Appends rule r11 to rules array
- Updates `last_run` to `2026-04-18T11:45:00Z`
- `runs` counter: 1 → 2 (this is the 2nd overall circuit: 1 from BOOTSTRAP_WALKTHROUGH + 1 now)

QUEEN commits:

```bash
git commit -m "organize: circuit 1 of 3 — moved 3 benches, 0 quarantined + new rule r11 ratified"
```

---

## 4. Circuit 2 — New Rule Active, .bench.py Files Gone

### 4.1 Re-enumeration

QUEEN re-scans (`circuits` loop re-runs from scratch per `ORGANIZE_WORKFLOW.json §flow.maintenance` step "QUEEN enumerates target files").

**Post-circuit-1 file inventory:**
- Root: `README.md`, `TEST_PROJECT_README.md`, `pyproject.toml`
- `src/`: `main.py`, `utils/helpers.py`, `utils/__init__.py`, `__init__.py` (bench files moved out)
- `tests/`: `test_main.py`, `__init__.py`
- `docs/`: `overview.md`
- `scripts/`: `deploy.sh`, `backup.sh`
- `benches/`: `run_bench.bench.py`, `profile.bench.py`, `check.bench.py` (moved in circuit 1)

**Total eligible:** 15 files.

### 4.2 Batching

BATCH_CAP=30. All 15 files in one logical grouping; QUEEN splits by subtree:

```
Batch A — root (3 files): README.md, TEST_PROJECT_README.md, pyproject.toml
Batch B — src/ + tests/ + docs/ (9 files)
Batch C — benches/ + scripts/ (5 files)
```

### 4.3 TRIAGE_WAVE — Circuit 2

**Batch A:** Same as circuit 1 — `README.md` KEEP_IN_PLACE, `pyproject.toml` KEEP_IN_PLACE, `TEST_PROJECT_README.md` ASK_USER.

**Batch B:** All 7 src/tests/docs files → KEEP_IN_PLACE (rules r1, r2, r3 fire). No new patterns.

**Batch C — benches/ + scripts/:**

`benches/run_bench.bench.py`:
- Rule r1 (`src/**/*`) — no match (path starts with `benches/`)
- Rule r11 (`**/*.bench.py` → `benches/`) (priority 80) — **match!** File is `benches/run_bench.bench.py`. Destination is `benches/`. Current location is `benches/`. → `KEEP_IN_PLACE`
- Evidence: "rule r11 (**/*.bench.py → benches/) matched; file is already at destination benches/"
- Confidence: HIGH

`benches/profile.bench.py` → same → `KEEP_IN_PLACE` (HIGH)
`benches/check.bench.py` → same → `KEEP_IN_PLACE` (HIGH)

`scripts/deploy.sh` → no rule matches; `KEEP_IN_PLACE` (MEDIUM, no rule but convention placement)
`scripts/backup.sh` → same → `KEEP_IN_PLACE` (MEDIUM)

**Circuit 2 tally:**
- KEEP_IN_PLACE: 14
- ASK_USER: 1 (TEST_PROJECT_README.md — same decision as before)
- FLAG_NEW_RULE: 0

### 4.4 Ratification Dialog — Circuit 2

```
==== ORGANIZE MAINTENANCE — CIRCUIT 2 of 3 ====
KEEP_IN_PLACE: 14 files — no action needed.
```

```
ASK_USER (1 file):
  File 1 of 1: TEST_PROJECT_README.md [same as before]
  >
```

**User:** `a` (KEEP_IN_PLACE)

```
==== RATIFICATION SUMMARY — CIRCUIT 2 of 3 ====
Ready to execute: 0 moves. No changes.
Commit: organize: circuit 2 of 3 — nothing to do
Proceed? [yes / abort]
>
```

**User:** `yes`

QUEEN updates `.organize.json` metadata (last_run, runs counter: 2 → 3). Commits:

```bash
git commit -m "organize: circuit 2 of 3 — nothing to do"
```

---

## 5. Circuit 3 — Full Convergence

### 5.1 Re-enumeration

Same file inventory as circuit 2 — no changes since circuit 1 executed all moves.

### 5.2 TRIAGE_WAVE — Circuit 3

All 15 files → same verdicts as circuit 2. All KEEP_IN_PLACE (14) + ASK_USER (1 for TEST_PROJECT_README.md).

Since user has consistently chosen KEEP_IN_PLACE for TEST_PROJECT_README.md, QUEEN notes this pattern. (A future enhancement might offer to add a rule to suppress this; in v0.1.0 it surfaces again each circuit.)

### 5.3 Ratification Dialog — Circuit 3

```
==== ORGANIZE MAINTENANCE — CIRCUIT 3 of 3 ====
KEEP_IN_PLACE: 14 files — no action needed.

ASK_USER (1 file): TEST_PROJECT_README.md [same]
>
```

**User:** `a` (KEEP_IN_PLACE, again)

```
==== RATIFICATION SUMMARY — CIRCUIT 3 of 3 ====
Ready to execute: 0 moves.

Commit: organize: circuit 3 of 3 — nothing to do
Proceed? [yes / abort]
>
```

**User:** `yes`

QUEEN updates metadata. Commits:

```bash
git commit -m "organize: circuit 3 of 3 — nothing to do"
```

---

## 6. Convergence Demonstrated

**Verdict: NOTHING_TO_DO** (all 3 circuits emitted no proposed moves after circuit 1's initial cleanup and rule ratification).

Per `ORGANIZE_WORKFLOW.json §verdicts.maintenance_catalog.NOTHING_TO_DO`:
> "All circuits ran; all TRIAGE verdicts were KEEP_IN_PLACE. Project is already organized per current rules."

```
NOTHING_TO_DO — Circuits 2 and 3 had no moves.
Project is fully organized per 11 active rules.
  Rules array grew: 10 → 11 (r11: **/*.bench.py → benches/ ratified in circuit 1)
  Commit count: 3 (one per circuit: circuit-1 with moves, circuit-2 no-op, circuit-3 no-op)
  + 1 metadata update commit after all circuits complete

Final run metadata:
  runs: 4 (1 from BOOTSTRAP cascade + 3 from this invocation)
  last_run: 2026-04-18T12:05:00Z
```

QUEEN emits final metadata update commit:

```bash
git commit -m "organize: update run metadata"
```

---

## 7. Final State Summary

| Item | Before | After |
|---|---|---|
| Rules | 10 | 11 (r11 added) |
| Files in `.bench.py` pattern | 0 (in wrong place at `src/`) | 3 (correctly in `benches/`) |
| `benches/` directory | did not exist | created, 3 files |
| `scripts/` | exists, no covering rule | exists, KEEP_IN_PLACE by convention |
| Circuits run | 0 | 3 |
| Commits from this session | 0 | 4 (circuit-1, circuit-2, circuit-3, metadata) |

The FLAG_NEW_RULE → ratify → apply → convergence cycle is demonstrated:
1. Circuit 1: TRIAGE observes `.bench.py` pattern, proposes rule, user ratifies, files moved
2. Circuit 2: new rule active, `.bench.py` files now KEEP_IN_PLACE — zero moves
3. Circuit 3: identical to circuit 2 — confirms convergence

---

*End of MULTI_CIRCUIT_WALKTHROUGH.md*
