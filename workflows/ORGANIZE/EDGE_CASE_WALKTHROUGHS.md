# EDGE_CASE_WALKTHROUGHS — 6 Unusual Scenarios

**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Trace type:** Mental traces — each scenario is a self-contained walkthrough of an unusual condition.
**Spec references:** EDGE_CASES.md (primary authority), RATIFICATION_UI_SPEC.md §6, WIZARD_PROTOCOL.md §6, ORGANIZE_WORKFLOW.json §root_invariants, ORGANIZE_CONFIG_SCHEMA.json

---

## Scenario 1 — Binary File in Batch

**Reference:** `EDGE_CASES.md §Scenario 1`

### Setup

The `test_project/` has an additional file: `data/model_weights.pkl` — a serialized Python model file (binary). It was created during an experiment and ended up in the `data/` directory. However, the `data/` directory has no IN_PLACE rule in the current `.organize.json` (the mixed-research template only has `data/**/*` IN_PLACE in the full template; the test project's bootstrapped config omitted it since no `data/` existed at BOOTSTRAP time).

So `model_weights.pkl` is in the eligible population.

### Scenario trace

**TRIAGE batch assignment:** `model_weights.pkl` lands in Batch A (data/ subtree).

**TRIAGE Step 2b — full-read attempt:**

TRIAGE attempts to read `model_weights.pkl` fully.

The file is binary (pickle format). TRIAGE cannot decode it as UTF-8.

Per `WORKER_TRIAGE.md §4`:
> "If a file is binary or unreadable: Emit ASK_USER with rationale 'binary/unreadable; workflow cannot classify mechanically'. Quarantine of a binary without user input is wrong-shape."

**TRIAGE verdict:**

```json
{
  "file": "/path/to/test_project/data/model_weights.pkl",
  "verdict": "ASK_USER",
  "destination": null,
  "rationale": "binary/unreadable; workflow cannot classify mechanically",
  "evidence": "file is not UTF-8 decodable; binary content detected (pickle format)",
  "confidence": "HIGH"
}
```

**QUEEN ratification dialog (per RATIFICATION_UI_SPEC.md §3.5 and EDGE_CASES.md §Scenario 1):**

```
ASK_USER: data/model_weights.pkl
TRIAGE reason: binary file — workflow cannot classify mechanically.

Options:
  a) KEEP_IN_PLACE — leave at current location (data/)
  b) MOVE_TO <path> — specify a destination (e.g., type 'change to data/models/')
  c) QUARANTINE:.delete — discard (treat as garbage artifact)
  d) QUARANTINE:.archive — preserve historically
  e) skip this file — no action this circuit
Choose: [a / b / c / d / e / abort]
>
```

**User response:** `d` (QUARANTINE:.archive — the model checkpoint has historical value as a research artifact)

**QUEEN behavior:**
1. Queue `QUARANTINE:.archive` for `model_weights.pkl`.
2. At execution: `git mv data/model_weights.pkl .archive/data/model_weights.pkl` (path mirrored).
3. Quarantine log entry:
   ```json
   {
     "run": 2, "timestamp": "2026-04-18T13:00:00Z",
     "file": "data/model_weights.pkl",
     "destination": ".archive",
     "reason": "binary file; TRIAGE could not classify mechanically; user elected QUARANTINE:.archive to preserve as historical research artifact",
     "ratified_by": "user", "untracked_at_move": false
   }
   ```

**Tip QUEEN adds:** "If this file type should always be handled the same way, consider adding `**/*.pkl` to `ignore_paths` (to exclude all pickle files from future enumeration) or adding a rule `**/*.pkl → data/models/` (to route model files to a canonical location)."

**Outcome:** Binary file handled correctly. No fabricated verdict. User decision recorded.

---

## Scenario 2 — File Matches Two Rules with Different Priorities

**Reference:** `EDGE_CASES.md §Scenario 3`

### Setup

Consider a file at project root: `notes_v2.md`. The `.organize.json` has two rules that could both match:

```
r6  | priority 40 | glob | *.md             | IN_PLACE  | Loose root .md → ASK_USER
r10 | priority 55 | hint | SUPERSEDED/DEPRECATED/OBSOLETE in first 10 lines → .archive
```

The file content of `notes_v2.md`:

```markdown
# Notes Version 2

DEPRECATED: these notes are superseded by the formal design in docs/.

Quick thoughts from early development phase...
```

### Scenario trace

**TRIAGE rule evaluation (priority descending):**

r1(100), r2(95), r3(85), r4(80), r5(60) — none match (`notes_v2.md` is not under src/, docs/, tests/, and doesn't match README/CHANGELOG/LICENSE, and is not a .py file).

**r10 fires at priority 55 (first matching rule for this file):**
- Kind: `hint`
- TRIAGE reads file fully.
- Checks first 10 lines: "DEPRECATED: these notes are superseded by the formal design in docs/." found at line 3.
- Formal DEPRECATED marker confirmed at HIGH confidence.
- Rule r10 destination: `.archive`
- First-match-wins: r10 fires → verdict: `QUARANTINE:.archive`

Rule r6 (priority 40) is evaluated AFTER r10 in priority order but **never reached** because r10 already fired. First-match-wins (per `ORGANIZE_RULE_FORMAT.md §6.1`; `EDGE_CASES.md §Scenario 3`).

**TRIAGE verdict:**

```json
{
  "file": "notes_v2.md",
  "verdict": "QUARANTINE:.archive",
  "destination": ".archive",
  "rationale": "rule r10 (hint: SUPERSEDED/DEPRECATED/OBSOLETE marker → .archive) matched at priority 55; first-match-wins; rule r6 (*.md, priority 40) never evaluated",
  "evidence": "file contains 'DEPRECATED: these notes are superseded by the formal design in docs/.' at line 3 — formal DEPRECATED marker",
  "confidence": "HIGH"
}
```

**User-facing message** (during ratification dialog):

```
QUARANTINE:.archive (1 file):
  notes_v2.md  (rule r10 hint: "DEPRECATED:" marker at line 3)

Files will be moved to: .archive/
(Reversible — preserved indefinitely.)

Options: [ratify all / review / skip / abort]
>
```

**Why this is correct:** Rule r10 (priority 55) evaluated before rule r6 (priority 40) because higher priority = evaluated first (per `ORGANIZE_RULE_FORMAT.md §6.1`). The DEPRECATED marker triggered r10 specifically, avoiding the generic ASK_USER response that r6 would have produced. This is the designed behavior: specific rules at higher priority correctly supersede broad catch-all rules at lower priority.

**Design note (from `EDGE_CASES.md §Scenario 3`):** Rule authors should assign priorities deliberately to express this ordering. In this case, r10 having priority 55 vs r6 having priority 40 is intentional: the hint rule (content-aware) should evaluate before the broad glob catch-all.

---

## Scenario 3 — Root-Invariant Path Slips Past QUEEN Filter

**Reference:** `EDGE_CASES.md §Scenario 9`

### Setup

This is a defensive scenario. It should not occur in correct operation. We trace what happens if QUEEN's pre-batching filter has a bug and `.claude/settings.json` appears in a TRIAGE batch.

### Scenario trace

**QUEEN pre-batching filter (expected):**

Per `BATCHING_HEURISTIC.md §2.1`, QUEEN removes `.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json` unconditionally before any batching.

**Simulated bug:** A hypothetical bug in QUEEN's filter causes `.claude/settings.json` to slip through and appear in Batch A.

**TRIAGE Step 2a-i — Root-invariant defensive check (highest priority):**

Per `WORKER_TRIAGE.md §5 Step 2a-i`:
> "If the file matches any of `.claude/`, `.claude-flow/`, `.hive-mind/`, or is `.mcp.json` (or lives under those paths), emit ASK_USER with rationale 'root-invariant path reached TRIAGE; QUEEN should have filtered this — refusing to classify to avoid violating workflow invariant'."

TRIAGE emits — without reading the file, without evaluating any rules:

```json
{
  "file": "/path/to/test_project/.claude/settings.json",
  "verdict": "ASK_USER",
  "destination": null,
  "rationale": "root-invariant path reached TRIAGE; QUEEN should have filtered this — refusing to classify to avoid violating workflow invariant",
  "evidence": "file path begins with .claude/, which is a protected root invariant per ORGANIZE_WORKFLOW.json §root_invariants",
  "confidence": "HIGH"
}
```

**QUEEN aggregation behavior (per `EDGE_CASES.md §Scenario 9`):**

QUEEN receives this verdict. Because the rationale explicitly mentions "QUEEN filter bug," QUEEN escalates rather than presenting this as a normal ASK_USER:

```
ESCALATE — Root invariant reached TRIAGE (possible QUEEN filter bug).
File: .claude/settings.json
This file should never have been classified. It is a workflow-protected path.
No action has been taken on this file.

This indicates a bug in QUEEN's pre-batching filter. The file has been left in place.
The circuit will continue for all other files. Please report this incident.
```

**QUEEN behavior:**
1. File is treated as KEEP_IN_PLACE by default — no move queued.
2. The incident is logged in `.organize.json`'s `notes` field.
3. The circuit continues for all other files in all batches.
4. The bug is flagged for investigation.

**Outcome:** Defense-in-depth works. Even if QUEEN's primary filter fails, TRIAGE's defensive check prevents any action on root-invariant paths. The user is informed of the anomaly.

**User-facing message:** The ESCALATE message above is presented verbatim. No classification of `.claude/settings.json` occurs, ever.

---

## Scenario 4 — User Aborts Mid-Wizard

**Reference:** `WIZARD_PROTOCOL.md §6, Edge Case 2`

### Setup

User has invoked `ORGANIZE_WORKFLOW` on a new project (no `.organize.json`). BOOTSTRAP starts. INSPECTOR runs and returns its proposal. The wizard begins.

The user proceeds through Stage 1 (signals — accept) and Stage 2 (template — accept), then reaches Stage 3 (seed rules). While reviewing the proposed rules, the user decides they need to think more carefully about which rules to include and decides to abort.

### Scenario trace

**Stage 1:** User types `accept` — signals confirmed. Advance to Stage 2.

**Stage 2:** User types `accept` — template `mixed-research` confirmed. Advance to Stage 3.

**Stage 3 — user types `abort`:**

```
==== ORGANIZE BOOTSTRAP — Stage 3 of 6: Seed Rules ====

[Rules table displayed...]

Options: accept all | review | add <json> | remove <id> | edit <id> <field> <val> | abort | help
> abort
```

**QUEEN behavior (per `WIZARD_PROTOCOL.md §6, Edge Case 2`):**

1. Immediately stop the wizard.
2. Discard all in-memory wizard state: proposed template, rules, ignore_paths, any edits applied in Stage 3.
3. Do NOT write `.organize.json`.
4. Do NOT create `.delete/` or `.archive/`.
5. Do NOT commit anything.

QUEEN emits:

```
ABORTED_DURING_WIZARD — No configuration written. Project state unchanged.
To restart, invoke ORGANIZE_WORKFLOW again.
```

Per `ORGANIZE_WORKFLOW.json §verdicts.bootstrap_catalog.ABORTED_DURING_WIZARD`:
> "User aborted before `.organize.json` was written. No state changes. Project remains unconfigured."

**Outcome:** Clean exit. The project is exactly as it was before ORGANIZE was invoked. No partial state. No file changes. The user can re-invoke `ORGANIZE_WORKFLOW` and start the wizard from Stage 1 again.

**Key invariant (from `WIZARD_PROTOCOL.md §5`):**
> "`.organize.json` is NEVER written until Stage 6 confirmation."
> "`abort` at any stage clears in-memory state and emits ABORTED_DURING_WIZARD."

The user's Stage 1 and Stage 2 decisions (signals confirmed, template chosen) are lost when abort fires — no partial config is persisted.

---

## Scenario 5 — User Aborts After Circuit 2 of 3

**Reference:** `EDGE_CASES.md §Scenario 4`, `RATIFICATION_UI_SPEC.md §6`

### Setup

Continuing from MULTI_CIRCUIT_WALKTHROUGH setup. User has invoked 3-circuit MAINTENANCE. Circuit 1 completes with moves (3 `.bench.py` files moved to `benches/`, rule r11 ratified). Circuit 2 completes with no moves. Now QUEEN is about to begin circuit 3.

But before circuit 3's TRIAGE_WAVE is spawned, the user intervenes: they type `abort` during QUEEN's inter-circuit pause.

### Scenario trace

**Inter-circuit pause (between circuits 2 and 3):**

Per `ORGANIZE_WORKFLOW.json §flow.maintenance`, after each circuit completes and before the next begins, QUEEN checks whether to continue. In v0.1.0, the circuits run in sequence without an explicit inter-circuit abort prompt — the `abort` option is available during each circuit's ratification dialog.

However, the user types `abort` at the circuit-2 post-ratification confirmation prompt:

```
==== RATIFICATION SUMMARY — CIRCUIT 2 of 3 ====
Ready to execute: 0 moves.

Commit message will be: organize: circuit 2 of 3 — nothing to do

Proceed with execution? [yes / abort]
> abort
```

**QUEEN behavior (per `RATIFICATION_UI_SPEC.md §6`, abort at post-ratification summary):**

When `abort` is typed at the post-ratification summary (after circuit 2's ratification dialog but before the confirmation `yes`):

| Abort point | Moves executed? | Commit? | Verdict |
|---|---|---|---|
| At post-ratification summary | None (this circuit) | No (this circuit) | `ABORTED` |

Note: Circuit 1 already committed (moves ratified + committed = permanent in git history). Circuit 2's no-op commit has NOT yet been made (the user aborted before confirming `yes`).

Wait — re-reading: if 0 moves are proposed, there is nothing harmful to abort from. The effect is: circuit 2's metadata update (last_run, runs counter) is NOT committed. Circuit 3 is not executed.

**QUEEN response:**

```
Ratification aborted. Circuit 2 of 3 not committed.
Circuit 1 of 3 remains committed (3 moves, rule r11 added).
Circuit 3 of 3 not executed.
Verdict: PARTIAL_RUN

State as of last committed circuit:
  - .organize.json rules: 11 (r11 added in circuit 1)
  - benches/ directory: exists with 3 .bench.py files
  - quarantine_log: unchanged from circuit 1

Next invocation will re-scan from current state.
```

Per `ORGANIZE_WORKFLOW.json §verdicts.maintenance_catalog.PARTIAL_RUN`:
> "User halted after some circuits completed but before reaching N. State as of last completed circuit is committed. Next invocation resumes from current state."

**Outcome:**
- Circuit 1 is permanent in git history (committed).
- Circuit 2 and 3 are not executed.
- `.organize.json` reflects state as of circuit 1 (runs counter = 2, last_run = circuit-1 timestamp).
- Verdict: `PARTIAL_RUN`.

---

## Scenario 6 — `.organize.json` Manually Edited, Becomes Invalid

**Reference:** `EDGE_CASES.md §Scenario 8`, `ORGANIZE_CONFIG_SCHEMA.json`

### Setup

Between ORGANIZE runs, the user manually opened `test_project/.organize.json` in a text editor to "fix" rule r3 (they wanted to change the priority). However, they made a typo: instead of changing `"priority": 85` to `"priority": 75`, they accidentally typed `"priority": "high"` (a string instead of an integer). They also accidentally deleted the `"active"` field from rule r3 entirely while editing.

The resulting rules[2] (r3, 0-indexed) now looks like:

```json
{
  "id": "r3",
  "created": "2026-04-18T10:15:00Z",
  "kind": "glob",
  "pattern": "tests/**/*",
  "destination": "IN_PLACE",
  "priority": "high",
  "note": "Test files under tests/ are at their canonical location"
}
```

Problems:
1. `priority` is a string `"high"` instead of an integer — schema violation (per `ORGANIZE_CONFIG_SCHEMA.json §definitions/rule.properties.priority.type: integer`)
2. `active` field is missing — schema violation (required field per `ORGANIZE_CONFIG_SCHEMA.json §definitions/rule.required: [..., "active"]`)

### Scenario trace

**User invokes:** `ORGANIZE_WORKFLOW`

**QUEEN engagement:** Config file found. Mode: MAINTENANCE.

**QUEEN config load + validation:**

Per `EDGE_CASES.md §Scenario 8` and `ORGANIZE_WORKFLOW.json §organize_json_schema.invariants`, QUEEN validates `.organize.json` against `ORGANIZE_CONFIG_SCHEMA.json` before spawning any TRIAGE workers.

Validation fails. QUEEN ESCALATEs immediately:

```
.organize.json failed validation. Cannot proceed.

Validation errors found in test_project/.organize.json:
  - rules[2].priority: expected integer, got string "high"
    (schema: ORGANIZE_CONFIG_SCHEMA.json §definitions/rule.properties.priority.type)
  - rules[2].active: required field missing
    (schema: ORGANIZE_CONFIG_SCHEMA.json §definitions/rule.required includes "active")

Fix the config file and re-invoke ORGANIZE_WORKFLOW.
Reference: workflows/ORGANIZE/ORGANIZE_CONFIG_SCHEMA.json (full schema)
           workflows/ORGANIZE/ORGANIZE_RULE_FORMAT.md (rule field documentation)

Note: the rules array is append-only. If you deleted a rule, you must restore it.
If you only changed field values, fix them to match the schema.
```

**QUEEN behavior:**
1. No TRIAGE workers spawned.
2. No filesystem changes.
3. `.organize.json` is NOT modified by QUEEN (user must fix it manually).

**User corrects the file:**

The user edits `.organize.json` to fix r3:

```json
{
  "id": "r3",
  "created": "2026-04-18T10:15:00Z",
  "kind": "glob",
  "pattern": "tests/**/*",
  "destination": "IN_PLACE",
  "priority": 85,
  "active": true,
  "note": "Test files under tests/ are at their canonical location"
}
```

**User re-invokes:** `ORGANIZE_WORKFLOW`

**QUEEN:** Validates config again — passes. Proceeds to MAINTENANCE circuit normally.

**Outcome:** Schema validation catches the error before any workflow logic runs. No partial state. No corrupted moves. User is given the specific field path and schema reference to fix. The ESCALATE message includes the JSON path (`rules[2].priority`) and the expected type, making the fix unambiguous.

**Key principle (from `EDGE_CASES.md §Scenario 8`):** "QUEEN does not attempt to auto-repair the config. The user must fix it manually." This preserves the append-only invariant — QUEEN never writes or deletes from the rules array outside of ratified MAINTENANCE operations.

---

*End of EDGE_CASE_WALKTHROUGHS.md*
