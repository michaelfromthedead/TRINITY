# EDGE_CASES — Unusual Situation Handling in ORGANIZE_WORKFLOW

**Version:** v0.1.0
**Status:** Active
**Authoritative spec:** `ORGANIZE_WORKFLOW.json` §`hard_rules`, §`root_invariants`, §`destinations`
**Related:** `RATIFICATION_UI_SPEC.md`, `BATCHING_HEURISTIC.md`, `WIZARD_PROTOCOL.md`

---

## Overview

This document catalogs how ORGANIZE handles unusual or adversarial situations. Each scenario specifies: the triggering condition, the expected system behavior, and the user-facing message or outcome.

Scenarios are grounded in `ORGANIZE_WORKFLOW.json` hard rules and worker discipline. No behavior described here is invented — each traces to a specific spec section.

---

## Scenario 1 — Binary File in Batch

**Condition:** A non-text file (e.g., a compiled binary, a `.png`, a `.pkl` model file, a `.sqlite` database) appears in a TRIAGE batch.

**Trigger:** TRIAGE attempts to full-read the file and determines it is binary or unreadable as text.

**TRIAGE behavior** (per `WORKER_TRIAGE.md` §4):
> "If a file is binary or unreadable: Emit ASK_USER with rationale 'binary/unreadable; workflow cannot classify mechanically'."

TRIAGE emits:
```json
{
  "file": "/abs/path/model_weights.pkl",
  "verdict": "ASK_USER",
  "rationale": "binary/unreadable; workflow cannot classify mechanically",
  "evidence": "file is not UTF-8 decodable; binary content detected",
  "confidence": "HIGH"
}
```

**QUEEN behavior:** The ASK_USER verdict propagates through aggregation normally. QUEEN presents it during ratification:

```
ASK_USER: model_weights.pkl
TRIAGE reason: binary file — workflow cannot classify mechanically.
Options:
  a) KEEP_IN_PLACE — leave at current location
  b) MOVE_TO <path> — specify a destination (e.g., 'change to data/')
  c) QUARANTINE:.delete — discard (treat as garbage artifact)
  d) QUARANTINE:.archive — preserve historically
  e) skip this file — no action this circuit
Choose: [a / b / c / d / e / abort]
>
```

**User decides.** QUEEN executes the user's verdict. If the user chooses MOVE_TO, QUEEN performs `git mv` on the binary file (git tracks the move correctly regardless of file content).

**Note:** If many binary files appear in a project, the user may want to add their patterns to `ignore_paths` (e.g., `**/*.pkl`, `**/*.sqlite`) to exclude them from future circuits.

---

## Scenario 2 — File Too Large for TRIAGE Read Budget

**Condition:** A file's content is too large for TRIAGE to read in full within its agent budget. Common examples: a very large CSV data file, a concatenated log file, a large generated JSON.

**TRIAGE behavior** (per `WORKER_TRIAGE.md` §4):
> "Emit ASK_USER with rationale 'file too large for full read; user judgment needed'. Do not guess."

TRIAGE emits:
```json
{
  "file": "/abs/path/data/training_corpus.csv",
  "verdict": "ASK_USER",
  "rationale": "file too large for full read; user judgment needed",
  "evidence": "file exceeds TRIAGE read budget; content not evaluated",
  "confidence": "HIGH"
}
```

**QUEEN behavior:** Propagates as ASK_USER during ratification, identical to Scenario 1, with the specific reason stated.

**Typical resolutions:**
- Large data files should usually be in `data/` → user selects MOVE_TO `data/`
- Large generated artifacts should usually be ignored → user adds pattern to `ignore_paths` for next run
- Log files → user selects QUARANTINE:.delete (or adds `**/*.log` to `ignore_paths`)

**Recommendation QUEEN adds to the presentation:**
```
Tip: if this file type should always be ignored, type 'skip this file' now
and consider adding its pattern to ignore_paths during a future maintenance run.
```

---

## Scenario 3 — File Matches Multiple Rules with Same Priority

**Condition:** A file's path matches two or more active rules that have identical priority values.

**Example:** Rules `r4` and `r5` both have `priority: 50`. Rule r4 matches `*.md` (destination: `docs/`). Rule r5 matches `notes*.md` (destination: `.archive`). The file `notes_2026.md` matches both.

**TRIAGE behavior** (per `WORKER_TRIAGE.md` §5, Step 2c):
> "First matching rule wins."

**Tie-breaking:** Rules are evaluated in the order they appear in the `rules` array (array position), after sorting by priority descending. When two rules have equal priority, the one at the lower array index (earlier in the array) is evaluated first and wins.

**Design rationale:** First-match-wins with array-position tie-breaking is deterministic, predictable, and auditable. The alternative — erroring on priority collision — would be disruptive for large rule sets where coincidental priority values are likely.

**User guidance:** Rule authors should avoid priority collisions for rules that may match the same files. If two rules are intended to apply to overlapping file sets, assign them different priorities deliberately. The `note` field in each rule is the recommended place to document intended priority relationships:

```json
{
  "id": "r5",
  "priority": 55,
  "note": "Higher than r4 (50) because specific pattern notes*.md should take precedence over general *.md"
}
```

**When collision is detected:** TRIAGE does not warn about priority collisions at classification time. QUEEN may add a future validation step that warns on detected collisions in the rules array, but this is not implemented in v0.1.0.

---

## Scenario 4 — User Aborts Mid-Circuit

**Condition:** The user types `abort` during the ratification dialog for circuit N (not the final confirmation — before or after some groups have been ratified).

**QUEEN behavior** (per `ORGANIZE_WORKFLOW.json` §`flow.maintenance`, `RATIFICATION_UI_SPEC.md` §6):

1. If `abort` is typed **during group ratification** (before the post-ratification summary):
   - Moves ratified for groups already processed are committed in a single circuit commit.
   - Groups not yet presented are left untouched.
   - `.organize.json` is updated: quarantine_log entries and any ratified rules are appended.
   - Commit message: `organize: circuit <N> of <M> — partial — moved <X> files`
   - Verdict: `PARTIAL_RUN`

2. If `abort` is typed **at the post-ratification summary** (final gate before execution):
   - No moves execute.
   - No commit.
   - Verdict: `ABORTED`

**State invariants after PARTIAL_RUN:**
- Committed moves are permanent in git history.
- Unratified files are untouched — they remain at their current paths.
- `.organize.json` reflects what was committed, not what was proposed.
- The next invocation will re-scan; any unratified files that still don't match an IN_PLACE rule will surface again.

**User-facing message (PARTIAL_RUN):**
```
Ratification aborted mid-circuit.
Ratified and committed: <X> moves (circuit <N> of <M> — partial).
Remaining <Y> file decisions were not applied.
Verdict: PARTIAL_RUN
Next invocation will re-scan.
```

---

## Scenario 5 — git mv on an Untracked File

**Condition:** A file in the ratified move list is not tracked by git (e.g., it is a new file that was never staged, or the project is not fully git-managed).

**QUEEN behavior** (per `ORGANIZE_WORKFLOW.json` §`hard_rules.git_mv_only`):
> "Files that are not tracked by git use plain `mv` but are noted in the quarantine_log as untracked_at_move=true."

1. QUEEN attempts `git mv <src> <dest>`.
2. Git returns an error indicating the file is not tracked.
3. QUEEN falls back to plain `mv <src> <dest>`.
4. QUEEN records the quarantine_log entry with `untracked_at_move: true`:
   ```json
   {
     "run": <N>,
     "timestamp": "<ISO-8601>",
     "file": "test_scratch.py",
     "destination": ".delete/test_scratch.py",
     "reason": "matches rule r6 (scratch*); QUARANTINE:.delete ratified by user",
     "ratified_by": "user",
     "untracked_at_move": true
   }
   ```
5. The commit still proceeds. Git stages the new `.delete/test_scratch.py` (which is already present from the plain `mv`). The original path disappears from the working tree.

**No error is raised to the user.** The `untracked_at_move: true` flag is an audit trail entry, not a user-visible alert. If the user examines `.organize.json` they will see it.

**Note:** If the target project is not a git repository at all (no `.git/`), see Scenario 11.

---

## Scenario 6 — Circular Reference: File in `.delete/` Referenced Elsewhere

**Condition:** A file was moved to `.delete/` in a previous circuit. Another file in the project contains a reference (e.g., an import, a markdown link) to the original path of the now-quarantined file.

**ORGANIZE behavior:** ORGANIZE does not enumerate `.delete/` or `.archive/` in subsequent runs — they are in `ignore_paths` (seeded by INSPECTOR on BOOTSTRAP per `WORKER_INSPECTOR.md` §5, Ignore-Path Seeds):

```
.delete/**
.archive/**
```

**Consequence:** The quarantined file is invisible to TRIAGE. The reference from another file remains broken. ORGANIZE does not detect broken references between project files.

**Why this is correct behavior:**  
ORGANIZE operates at filesystem-structure altitude — it rearranges where files live, not what they contain (`ORGANIZE_WORKFLOW.json` §`notes.altitude_note`). Detecting or fixing broken references would require content modification, which is explicitly forbidden:

> `ORGANIZE_WORKFLOW.json` §`hard_rules.never_mutate_file_content`: "The workflow only moves files. It does not modify file contents."

**User responsibility:** If a broken reference is discovered, the user must:
1. Either restore the file from `.delete/` to its canonical location, or
2. Update the referencing file to remove the broken reference.

ORGANIZE will not auto-detect this. The quarantine_log provides an audit trail of what was moved where and when, allowing the user to trace the provenance of a missing file.

---

## Scenario 7 — Pre-Commit Hook Fails on `organize:` Commit

**Condition:** QUEEN attempts to commit after a circuit's moves are executed (`organize: circuit N of M — ...`). The pre-commit hook fires and fails (e.g., a lint rule, a formatting check on a newly moved Python file).

**QUEEN behavior** (per `ORGANIZE_WORKFLOW.json` §`hard_rules.never_no_verify`):
> "Never use `git commit --no-verify` to bypass pre-commit hooks. If a hook fails during an organize commit, fix the issue (usually by adding the changed file to the correct list) and re-commit."

1. The commit fails. Git does not create the commit.
2. QUEEN captures the verbatim hook error output.
3. QUEEN ESCALATEs to the user with the specific error:
   ```
   Pre-commit hook failed during organize commit. Cannot commit without resolving this.
   
   Hook error output:
   ─────────────────────────────────────────────
   <verbatim hook output>
   ─────────────────────────────────────────────
   
   The moves have been executed (files are at their new locations) but the commit
   was not created. Options:
   
   1. Fix the hook issue manually, then type 'retry commit'.
   2. If the hook failure is unrelated to ORGANIZE's changes, fix the underlying
      code issue first, then type 'retry commit'.
   3. Type 'abort' to leave the working tree in its current state (moves executed,
      no commit). You will need to commit manually.
   
   NEVER use --no-verify to bypass hooks.
   >
   ```

4. The user fixes the issue and types `retry commit`. QUEEN re-runs `git commit` with the same message.
5. If the retry succeeds, the circuit is complete and the next circuit (if any) proceeds.
6. If the retry fails again, QUEEN reports the new error and waits for user action.

**Why this matters:** Pre-commit hooks enforce code quality at git boundaries. Bypassing them (with `--no-verify`) would allow `organize:` commits to silently introduce lint or formatting regressions. This is especially likely when a Python file is moved from a non-linted location to `src/` where a stricter mypy or ruff configuration applies.

**Typical causes of hook failure during organize commits:**
- A moved Python file has import paths that break after relocation.
- A moved `.sh` script fails a shellcheck lint that wasn't applied at the original location.
- A new rule's `pattern` field contains a character that triggers a JSON lint hook.

---

## Scenario 8 — `.organize.json` Manually Edited, Becomes Invalid

**Condition:** The user edits `.organize.json` between ORGANIZE runs. The edited file violates the schema (e.g., a rule is missing a required field, a priority value is set to a string instead of an integer, the rules array was modified by deletion).

**QUEEN behavior** (during MAINTENANCE engagement, after loading config):

1. QUEEN validates the loaded `.organize.json` against the config schema.
2. If invalid, QUEEN ESCALATEs immediately before any TRIAGE workers are spawned:
   ```
   .organize.json failed validation. Cannot proceed.
   
   Validation errors:
     - rules[2].priority: expected integer, got string "high"
     - rules[3].destination: missing required field
   
   Fix the config file and re-invoke ORGANIZE_WORKFLOW.
   Reference: workflows/ORGANIZE/ORGANIZE_CONFIG_SCHEMA.json (or ORGANIZE_RULE_FORMAT.md for rule fields).
   
   Note: the rules array is append-only — restoring deleted rules may be required.
   ```
3. No TRIAGE workers are spawned. No filesystem changes.

**Rule deletion invariant:** If the user deleted a rule from the array (violating the append-only invariant), QUEEN reports this as a validation error:
```
  - rules array: appears to be shorter than last known version.
    Rules are append-only — deactivate via active=false, do not delete.
```

QUEEN does not attempt to auto-repair the config. The user must fix it manually.

---

## Scenario 9 — Root-Invariant Path Slips Past QUEEN's Filter

**Condition:** QUEEN's pre-batching filter fails (a QUEEN-level bug) and a root-invariant path (`.claude/settings.json`, for example) appears in a TRIAGE batch.

**This should not happen.** QUEEN's filter (`ORGANIZE_WORKFLOW.json` §`root_invariants.enforcement`) runs before any batch is formed. But the system is designed with defense-in-depth.

**TRIAGE's defensive check** (per `WORKER_TRIAGE.md` §5, Step 2a-i):
> "If the file matches any of `.claude/`, `.claude-flow/`, `.hive-mind/`, or is `.mcp.json` (or lives under those paths), emit ASK_USER with rationale 'root-invariant path reached TRIAGE; QUEEN should have filtered this — refusing to classify to avoid violating workflow invariant'."

TRIAGE emits:
```json
{
  "file": "/abs/path/.claude/settings.json",
  "verdict": "ASK_USER",
  "rationale": "root-invariant path reached TRIAGE; QUEEN should have filtered this — refusing to classify to avoid violating workflow invariant",
  "evidence": "file path begins with .claude/, which is a protected root invariant",
  "confidence": "HIGH"
}
```

**QUEEN behavior:** QUEEN receives this ASK_USER verdict during aggregation. Because the rationale explicitly mentions "QUEEN filter bug," QUEEN treats this as an ESCALATE:

```
ESCALATE — Root invariant reached TRIAGE (possible QUEEN filter bug).
File: .claude/settings.json
This file should never have been classified. It is a workflow-protected path.
No action has been taken on this file.

This indicates a bug in QUEEN's pre-batching filter. The file has been left in place.
The circuit will continue for all other files. Please report this incident.
```

The file is treated as KEEP_IN_PLACE by default. No move is queued. The incident is noted in the circuit's run log (appended to `.organize.json` notes field).

**This scenario is a critical bug signal.** If it occurs, the filter logic in QUEEN's enumeration step must be reviewed.

---

## Scenario 10 — Empty Target Directory (No Loose Files)

**Condition:** The target directory has no loose files after both pre-batching filters are applied. All files are either in ignore_paths, or all non-ignored files already match an IN_PLACE rule.

**Two sub-cases:**

**Sub-case A — All files matched by ignore_paths:**  
The eligible population is genuinely empty. No TRIAGE_WAVE is spawned.

**QUEEN behavior:**
```
NOTHING_TO_DO — no eligible files after filtering.
All files are in ignored paths.
Verdict: NOTHING_TO_DO
```
`.organize.json` updated: runs counter incremented, last_run updated. No filesystem changes.

**Sub-case B — Files present but all IN_PLACE:**  
TRIAGE_WAVE runs; all verdicts are KEEP_IN_PLACE. No moves proposed.

**QUEEN behavior:**
```
NOTHING_TO_DO — all files are already correctly placed per current rules.
Circuit <N> of <M>: 0 moves proposed.
Verdict: NOTHING_TO_DO (this circuit)
```
Same metadata update as Sub-case A.

Both sub-cases are valid terminal states. They indicate the project is well-organized per current rules. The user may want to run ORGANIZE again after making structural changes.

---

## Scenario 11 — Non-Git Directory (No `.git/`)

**Condition:** The target directory is not a git repository. `git mv` will fail for all moves.

**INSPECTOR detection (BOOTSTRAP):** INSPECTOR checks for `.git/` as part of Phase 2 signal detection. If absent, it records `prior_state: non-git` and includes in `open_questions`:
```
Target directory appears to not be a git repository (no .git/ found).
ORGANIZE will use plain mv for all moves — git history will not be preserved.
Quarantine moves will be logged with untracked_at_move=true throughout.
Proceed?
```

**QUEEN behavior (MAINTENANCE):** If the project is discovered to be non-git during MAINTENANCE (no `.git/` found at target_dir):

1. QUEEN warns the user before spawning TRIAGE:
   ```
   WARNING: No .git/ found at target root. This is not a git repository.
   All moves will use plain mv (not git mv). Git history will not be preserved.
   All moves will be logged as untracked_at_move=true in quarantine_log.
   
   Proceed? [yes / abort]
   >
   ```

2. If user confirms, all moves use plain `mv`. Every quarantine_log entry records `untracked_at_move: true`.

3. The "commit" step is skipped (nothing to commit in a non-git repo). QUEEN updates `.organize.json` and reports completion without a git commit.

**This is a degraded mode.** The `git_mv_only` hard rule (`ORGANIZE_WORKFLOW.json` §`hard_rules.git_mv_only`) exists specifically to preserve history. In a non-git environment, this protection is unavailable. The quarantine_log remains as the only audit trail.

---

## Cross-Reference

| Scenario | Spec reference |
|---|---|
| 1 — Binary file | `WORKER_TRIAGE.md` §4; `ORGANIZE_WORKFLOW.json` §`hard_rules.honest_ambiguity` |
| 2 — Too large | `WORKER_TRIAGE.md` §4; §`hard_rules.full_read_or_skip` |
| 3 — Priority collision | `WORKER_TRIAGE.md` §5, Step 2c; §`hard_rules` |
| 4 — Abort mid-circuit | `ORGANIZE_WORKFLOW.json` §`verdicts.maintenance_catalog`; `RATIFICATION_UI_SPEC.md` §6 |
| 5 — git mv untracked | `ORGANIZE_WORKFLOW.json` §`hard_rules.git_mv_only` |
| 6 — Circular reference | `ORGANIZE_WORKFLOW.json` §`hard_rules.never_mutate_file_content`; §`notes.altitude_note` |
| 7 — Hook failure | `ORGANIZE_WORKFLOW.json` §`hard_rules.never_no_verify` |
| 8 — Invalid config | `ORGANIZE_WORKFLOW.json` §`organize_json_schema.invariants` |
| 9 — Root-invariant slip | `ORGANIZE_WORKFLOW.json` §`root_invariants`; `WORKER_TRIAGE.md` §2a-i |
| 10 — Empty target | `ORGANIZE_WORKFLOW.json` §`verdicts.maintenance_catalog.NOTHING_TO_DO` |
| 11 — Non-git directory | `ORGANIZE_WORKFLOW.json` §`hard_rules.git_mv_only` |

---

*End of EDGE_CASES.md*
