# RATIFICATION_UI_SPEC — User Ratification Dialog for TRIAGE_WAVE Results

**Version:** v0.1.0
**Status:** Active
**Authoritative spec:** `ORGANIZE_WORKFLOW.json` §`flow.maintenance`, §`hard_rules.no_auto_ratify`
**Used by:** QUEEN during MAINTENANCE mode, after TRIAGE_WAVE aggregation

---

## 1. Overview

After TRIAGE_WAVE returns and QUEEN aggregates all verdicts, QUEEN presents a proposed-moves plan to the user. The user ratifies (approves), reviews individually, skips, or aborts each group. Ratified moves are then executed via `git mv`. No move executes without explicit user ratification.

Per `ORGANIZE_WORKFLOW.json` §`hard_rules.no_auto_ratify`:
> "No move executes without explicit user ratification. ASK_USER and FLAG_NEW_RULE verdicts halt execution for interactive decision."

The ratification dialog is **terminal-friendly text** — ORGANIZE operates in a Claude conversation, which is a text environment.

---

## 2. Aggregation Before Presentation

Before presenting to the user, QUEEN:

1. Collects all verdict arrays from all TRIAGE workers in the wave.
2. Groups verdicts by type:
   - `KEEP_IN_PLACE` — no action needed; summarized but not presented for ratification
   - `MOVE_TO:<path>` — grouped by destination path
   - `QUARANTINE:.delete` — grouped together
   - `QUARANTINE:.archive` — grouped together
   - `ASK_USER` — listed individually (each requires a separate decision)
   - `FLAG_NEW_RULE` — consolidated via cross-batch aggregation (see `BATCHING_HEURISTIC.md` §6)
3. Applies cross-batch FLAG_NEW_RULE aggregation (see `BATCHING_HEURISTIC.md` §6).
4. Sorts groups for presentation: KEEP_IN_PLACE (shown as summary), then MOVE_TO, then QUARANTINE:.delete, then QUARANTINE:.archive, then ASK_USER, then FLAG_NEW_RULE.

KEEP_IN_PLACE verdicts are reported as a count only — they require no user action and are not presented for ratification.

---

## 3. Ratification Dialog Format

### 3.1 Circuit Header

```
==== ORGANIZE MAINTENANCE — CIRCUIT <N> of <M> ====
Target: <target_dir>
Files enumerated: <total>  |  Ignored: <ignored count>  |  Classified: <classified count>
Workers in wave: <N TRIAGE workers>

KEEP_IN_PLACE: <N> files — no action needed.
```

The KEEP_IN_PLACE summary is displayed once and requires no user input.

---

### 3.2 MOVE_TO Groups

One block per distinct destination path. If multiple rules direct files to the same destination, they are shown in the same block.

```
MOVE_TO canonical locations (<N> files, <destination>):
  src/utils/helpers.py → src/utils/helpers.py  [already at destination — rule r1: IN_PLACE]

MOVE_TO tests/ (<N> files):
  test_foo.py → tests/test_foo.py  (rule r2: **/test_*.py → tests/)
  test_bar.py → tests/test_bar.py  (rule r2)
  test_integration.py → tests/test_integration.py  (rule r2)
  ... <N-3> more  (type 'review' to see all)

Options: [ratify all / review / skip / abort]
>
```

**User options:**

| Option | Behavior |
|---|---|
| `ratify all` | Approve all moves in this group. QUEEN queues them for execution after full plan ratification. |
| `review` | Enter per-file review mode (see §3.6). |
| `skip` | Skip this group entirely. No moves in this group execute this circuit. QUEEN notes: "Skipped — these files remain at current location." |
| `abort` | Stop ratification. Moves ratified so far are committed. Unratified moves left untouched. Emit `PARTIAL_RUN`. |

---

### 3.3 QUARANTINE:.delete Block

```
QUARANTINE:.delete (<N> files — proposed garbage):
  scratch.py  (rule r6: scratch*)
  tmp_old.md  (rule r5: tmp*)
  untitled.txt  (rule r5: tmp* — matched by prefix; TRIAGE: "no imports, no references, orphaned")
  ... <N-3> more  (type 'review' to see all)

Files will be moved to: <target_dir>/.delete/
(Reversible — nothing is deleted. You manage .delete/ manually.)

Options: [ratify all / review / skip / abort]
>
```

**Note on .delete/ creation:** QUEEN creates `.delete/` lazily if it does not exist and at least one QUARANTINE:.delete verdict is ratified. It is added to `.gitignore` automatically on first creation.

---

### 3.4 QUARANTINE:.archive Block

```
QUARANTINE:.archive (<N> files — proposed historical preservation):
  notes2.md  (rule r7: notes[0-9]*.md)
  old_design.md  (TRIAGE: "file contains 'SUPERSEDED BY architecture_v2.md' at line 1")
  ... <N-2> more

Files will be moved to: <target_dir>/.archive/
(Reversible — preserved indefinitely. You manage .archive/ manually.)

Options: [ratify all / review / skip / abort]
>
```

---

### 3.5 ASK_USER Block

ASK_USER verdicts are listed individually. Each one requires its own decision. QUEEN presents them in sequence.

```
ASK_USER (<N> files need individual decisions):

  File 1 of <N>: my_special_notes.md
  TRIAGE rationale: "ambiguous — could be KEEP_IN_PLACE (author attachment implied by name)
    or QUARANTINE:.archive (content is historical meeting notes). No rule matched."
  Options observed by TRIAGE:
    a) KEEP_IN_PLACE — keep at current location
    b) MOVE_TO docs/ — move to docs/ directory (rule r4 would cover it)
    c) QUARANTINE:.archive — move to .archive/ as historical record
    d) QUARANTINE:.delete — discard (only if you are certain it has no value)
  Choose: [a / b / c / d / skip this file / abort]
  >
```

After the user responds for each file:

```
  Decision recorded: my_special_notes.md → MOVE_TO docs/

  File 2 of <N>: experiment_2026_04.py
  TRIAGE rationale: "dated experiment script; no test coverage; not imported by any other file.
    Could be .delete (garbage) or .archive (historical experiment worth preserving)."
  ...
  >
```

**ASK_USER options:**

| Option | Behavior |
|---|---|
| `a / b / c / d` | Apply the labeled option to this file. Record decision; move to next ASK_USER file. |
| `skip this file` | Leave this file at its current location this circuit. No move queued. |
| `abort` | Stop ratification here. Execute moves ratified so far; leave the rest. Emit `PARTIAL_RUN`. |

---

### 3.6 Per-File Review Mode

When the user types `review` for a MOVE_TO, QUARANTINE:.delete, or QUARANTINE:.archive group, QUEEN enters per-file review:

```
Reviewing MOVE_TO tests/ — file 1 of <N>:

  test_foo.py → tests/test_foo.py
  Rule matched: r2 (**/test_*.py → tests/)
  TRIAGE evidence: "filename matches test_*.py pattern; file contains 'import pytest' at line 1"
  Confidence: HIGH

  Options: [keep / change to <other verdict> / skip this file / abort]
  >
```

**Per-file review options:**

| Option | Behavior |
|---|---|
| `keep` | Accept this file's proposed move. Queue it. Move to next file. |
| `change to QUARANTINE:.delete` | Override verdict to QUARANTINE:.delete. Queue the changed verdict. Move to next file. |
| `change to QUARANTINE:.archive` | Override verdict to QUARANTINE:.archive. Move to next file. |
| `change to KEEP_IN_PLACE` | Override verdict to KEEP_IN_PLACE (no move). Move to next file. |
| `change to <path>` | Override destination to `<path>`. Move is queued with the new destination. |
| `skip this file` | Leave this file at its current location. No move queued. Move to next file. |
| `abort` | Stop ratification. Execute ratified so far; leave unratified. Emit `PARTIAL_RUN`. |

After iterating all files in the group, QUEEN summarizes:

```
Review complete for MOVE_TO tests/ (<N> files):
  Ratified: <X> moves
  Skipped: <Y> files
  Overridden: <Z> files

Proceed with ratified moves? [yes / abort]
>
```

---

### 3.7 FLAG_NEW_RULE Block

```
FLAG_NEW_RULE (<N> suggestions, covering <M> files total):

  Suggestion 1 of <N>:
  Pattern observed: *.bench.py (4 files, no existing rule)
  Suggested rule:
    id:          proposed-r8
    kind:        glob
    pattern:     **/*.bench.py
    destination: benches/
    priority:    80
    note:        "benchmark scripts belong in benches/; observed 4 files matching *.bench.py"
  Affected files:
    src/run_bench.bench.py
    src/matrix_bench.bench.py
    src/parse_bench.bench.py
    tests/end_to_end_bench.bench.py
  Cross-batch: pattern observed in 2 batches (consolidated)

  Options: [ratify rule / reject / edit <field> <value> / abort]
  >
```

**FLAG_NEW_RULE options:**

| Option | Behavior |
|---|---|
| `ratify rule` | Append the suggested rule to `.organize.json`'s rules array. The 4 affected files are then re-classified against the new rule and queued for their prescribed moves (destination: `benches/`). QUEEN applies this re-classification immediately. |
| `reject` | Record the rule candidate in `pending_rule_candidates` with `status: rejected`. The affected files are left at their current location this circuit. |
| `edit <field> <value>` | Modify the suggested rule before ratifying (e.g., `edit destination scripts/`). Validate the edit against RULE_FORMAT. Re-present the suggestion with the edit applied. |
| `abort` | Stop ratification. Process state as of the last ratified item. Emit `PARTIAL_RUN`. |

When a rule is ratified, QUEEN also applies the new rule's verdict to the affected files in the current circuit's move queue — the user does not need to separately ratify the moves that result from a just-ratified rule.

---

## 4. Post-Ratification Summary

After all groups are processed, QUEEN presents a final summary before executing moves:

```
==== RATIFICATION SUMMARY — CIRCUIT <N> of <M> ====

Ready to execute:
  MOVE_TO (various):     <X> files
  QUARANTINE:.delete:    <Y> files
  QUARANTINE:.archive:   <Z> files
  New rules ratified:    <R> rules

Skipped / left in place: <S> files

Commit message will be:
  organize: circuit <N> of <M> — moved <X+Y+Z> files, <R> new rules

Proceed with execution? [yes / abort]
>
```

- `yes` → QUEEN executes all queued moves via `git mv`; writes quarantine log entries; writes ratified rules; commits.
- `abort` → No moves execute. No commit. Emit `ABORTED`. (This is the final abort point before execution.)

**This is the last user decision before filesystem changes occur.**

---

## 5. Execution and Commit

After `yes` at the post-ratification summary:

1. QUEEN creates `.delete/` and/or `.archive/` directories if they do not exist and any quarantine moves are queued.
2. QUEEN adds `.delete/` to `.gitignore` if not already present (always). `.archive/` is added to `.gitignore` by default.
3. QUEEN executes each queued move: `git mv <src> <dest>` for git-tracked files; plain `mv` for untracked files (logged as `untracked_at_move: true` in quarantine_log).
4. QUEEN appends quarantine_log entries to `.organize.json` for all QUARANTINE moves.
5. QUEEN appends ratified new rules to `.organize.json`'s rules array.
6. QUEEN appends rejected FLAG_NEW_RULE candidates to `pending_rule_candidates` with `status: rejected`.
7. QUEEN commits: `organize: circuit <N> of <M> — moved <X> files, quarantined <Y>`.

Commit happens once per circuit, after all moves for that circuit are executed.

---

## 6. Abort Semantics

`abort` can be typed at any prompt. Behavior depends on where in the dialog it occurs:

| Abort point | Moves executed? | Commit? | Verdict |
|---|---|---|---|
| During MOVE_TO / QUARANTINE / FLAG_NEW_RULE group ratification, before post-ratification summary | Only moves ratified before abort | Yes, for ratified moves | `PARTIAL_RUN` |
| At ASK_USER file N (mid-sequence) | Moves for files 1..N-1 (if ratified) | Yes, for ratified moves | `PARTIAL_RUN` |
| At per-file review, mid-group | Moves for files reviewed before abort | Yes, for ratified moves | `PARTIAL_RUN` |
| At post-ratification summary (`abort` instead of `yes`) | None | No | `ABORTED` |

**Key distinction:** `PARTIAL_RUN` means some moves were made and committed. `ABORTED` means the post-ratification summary was reached but the user declined at the final gate — no filesystem changes.

**State after PARTIAL_RUN:**  
Ratified and committed moves are permanent (in git history). Unratified groups are left untouched. `.organize.json` is updated to reflect what was committed. The next MAINTENANCE circuit will re-scan and may surface the unratified files again (they will appear as loose files if they still match no rule's IN_PLACE condition).

---

## 7. Special Cases

### 7.1 Empty circuit (all KEEP_IN_PLACE)

```
==== ORGANIZE MAINTENANCE — CIRCUIT <N> of <M> ====
KEEP_IN_PLACE: <N> files — no action needed.
No moves proposed. Project is organized per current rules.

Verdict: NOTHING_TO_DO (this circuit)
```

No ratification dialog shown. QUEEN increments runs counter, updates last_run, commits metadata update.

### 7.2 Only ASK_USER verdicts

All N files are ambiguous. No automatic move groups. QUEEN presents only the ASK_USER sequence. After the user decides each file, post-ratification summary shows only user-decided moves (or an empty plan if all were skipped).

### 7.3 Large number of files in a group (>20)

QUEEN shows the first 5 files in the group summary, then "... <N-5> more". The `review` option shows all files. For groups >50 files, QUEEN notes: "This is a large group. Consider using 'review' to verify a sample before ratifying all."

---

## 8. Cross-Reference

- `ORGANIZE_WORKFLOW.json` §`flow.maintenance` — full circuit sequence
- `ORGANIZE_WORKFLOW.json` §`hard_rules.no_auto_ratify` — ratification requirement
- `ORGANIZE_WORKFLOW.json` §`hard_rules.git_mv_only` — execution method
- `ORGANIZE_WORKFLOW.json` §`hard_rules.never_no_verify` — commit hook policy
- `ORGANIZE_WORKFLOW.json` §`verdicts.maintenance_catalog` — terminal verdicts
- `BATCHING_HEURISTIC.md` §6 — FLAG_NEW_RULE cross-batch aggregation before presentation
- `EDGE_CASES.md` §4 — user aborts mid-circuit detail
- `EDGE_CASES.md` §7 — pre-commit hook failure during organize commit

---

*End of RATIFICATION_UI_SPEC.md*
