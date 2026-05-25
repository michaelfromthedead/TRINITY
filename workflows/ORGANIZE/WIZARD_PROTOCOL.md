# WIZARD_PROTOCOL — BOOTSTRAP Dialog Formalization

**Version:** v0.1.0
**Status:** Active
**Authoritative spec:** `ORGANIZE_WORKFLOW.json` §`wizard_flow`, §`units.WIZARD_LOOP`
**Used by:** QUEEN during BOOTSTRAP mode engagement

---

## 1. Overview

The BOOTSTRAP wizard is a 6-stage interactive dialog. QUEEN drives; the user answers. It is not a circuit — it is a dialog. No agents are spawned inside the wizard. The dialog produces a ratified `.organize.json` that QUEEN then writes to disk.

**Precondition:** INSPECTOR has returned its proposal block.  
**Postcondition:** `.organize.json` written (or ABORTED_DURING_WIZARD emitted, no writes).

The wizard is atomic in the sense that `.organize.json` is not written until the user approves the final stage. Partial wizard state is never committed.

---

## 2. Wizard State Machine

```
START
  │
  ▼
[S1: present_detected_signals]
  │ accept / correct
  ▼
[S2: propose_template]
  │ accept / change / custom
  ▼
[S3: propose_seed_rules]
  │ accept all / review individually / add / remove
  ▼
[S4: propose_ignore_paths]
  │ accept / edit
  ▼
[S5: review_initial_classification]
  │ ratify / adjust rules → loop back to S3
  ▼
[S6: cascade_decision]
  │ cascade / halt
  ▼
[WRITE .organize.json]
  │
  ▼
END (CONFIG_CREATED or CONFIG_CREATED_AND_CLEAN_RUN)

At any stage: abort → ABORTED_DURING_WIZARD (no write)
At any stage: help  → show stage context + options, re-present stage
At any stage: edit <field> <value> → apply edit, re-present stage
At S5: request rule adjustment → return to S3 with adjusted rules
```

---

## 3. Response Vocabulary (All Stages)

The following responses are valid at **every** stage unless noted otherwise:

| Response | Behavior |
|---|---|
| `accept` | Accept stage as presented; advance to next stage. |
| `edit <field> <value>` | Apply a targeted edit to a specific field; re-present the current stage for confirmation. |
| `edit` (bare) | Enter per-field review mode for the current stage; QUEEN walks through each field one at a time. |
| `reject` | Signal that the proposal is wrong; QUEEN asks for rationale, then loops the current stage with revision. |
| `abort` | Exit wizard immediately. No `.organize.json` is written. Emit `ABORTED_DURING_WIZARD`. |
| `help` | Show the stage's purpose, the fields being reviewed, and the available responses. Re-present stage. |
| free-form text | QUEEN interprets as intent; if unambiguous, applies and re-presents; if ambiguous, asks: "Did you mean X or Y?" |

---

## 4. Stage Definitions

### Stage 1 — `present_detected_signals`

**Purpose:** Show the user what INSPECTOR observed about the target project. The user confirms or corrects the factual picture before any rules or templates are proposed.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 1 of 6: Project Signals ====

INSPECTOR surveyed: <target_dir>

Detected signals:
  Language:          <lang signal, e.g. Python (pyproject.toml found)>
  Project kind:      <kind signal, e.g. library (src/ + tests/ layout observed)>
  Structure quality: <e.g. middling — 9 loose files at root>
  Orientation docs:  <e.g. present — README.md, CLAUDE.md found>
  Cruft patterns:    <e.g. tmp*, scratch*, notes[0-9]*.md>
  Prior ORGANIZE:    <e.g. clean — no .organize.json, no .delete/, no .archive/>
  Workflow artefacts:<e.g. none detected>
  Root invariants:   <e.g. .claude/ present, .claude-flow/ present, .hive-mind/ absent (noted), .mcp.json present>

Confidence: <HIGH | MEDIUM | LOW> — <note if LOW>

Open questions from INSPECTOR:
  1. <question if any>

────────────────────────────────────────────
Options: accept | correct <field> <value> | abort | help
>
```

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `accept` | Signals are confirmed; advance to Stage 2. |
| `correct <field> <value>` | Apply the correction to the in-memory signal map (e.g., `correct kind application`); re-present Stage 1. |
| `reject` | Ask user to describe what is wrong; re-present with a revised read if the correction is clear. |
| `abort` | Exit wizard; emit `ABORTED_DURING_WIZARD`; no writes. |
| `help` | Explain what "detected signals" are and how INSPECTOR derived them; re-present Stage 1. |

**Transition rules:**  
- `accept` or `correct` → advance to Stage 2.  
- QUEEN does NOT advance until at least one `accept` or equivalent confirmation is received.  
- If INSPECTOR confidence was LOW and user does not resolve the ambiguity, QUEEN notes this in the config's `notes` field before proceeding.

---

### Stage 2 — `propose_template`

**Purpose:** Present the proposed project template and allow the user to accept, substitute, or opt into custom ruleset mode.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 2 of 6: Template ====

Proposed template: <template_name>
Rationale: <INSPECTOR's 2-3 sentence rationale>

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

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `accept` | Confirm template; advance to Stage 3 with template's seed rules pre-loaded. |
| `change to <name>` | Substitute the proposed template with `<name>` from the list; reload seed rules from that template's definition; re-present Stage 2 showing the new template and rationale. |
| `custom` | Enter custom mode: no named template, no seed rules pre-loaded. Stage 3 will start with an empty rules array. The template field in `.organize.json` is recorded as `custom`. |
| `reject` | Equivalent to asking for a change; QUEEN asks which template or if user wants `custom`. |
| `abort` | Exit wizard; emit `ABORTED_DURING_WIZARD`. |
| `help` | Describe each template briefly; re-present Stage 2. |

**Transition rules:**  
- `accept` or `change to <name>` → advance to Stage 3 with the confirmed template's seed rules.  
- `custom` → advance to Stage 3 with an empty rules array.  
- On `change to <name>`: if `<name>` is not in the known template list, QUEEN responds: "Unknown template '<name>'. Available templates: <list>. Use 'custom' to define your own." Re-present Stage 2.

---

### Stage 3 — `propose_seed_rules`

**Purpose:** Present the proposed rules array (derived from the selected template) and allow the user to accept, review individually, add, or remove rules.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 3 of 6: Seed Rules ====

Proposed rules (from template: <template_name>):

  ID    Priority  Kind   Pattern                          Destination     Note
  ────  ────────  ─────  ───────────────────────────────  ──────────────  ─────────────────────────────────
  r1    100       glob   src/**/*.py                      IN_PLACE        Python source lives in src/
  r2    90        glob   **/test_*.py                     tests/          Test modules belong in tests/
  r3    80        glob   docs/**/*.md                     IN_PLACE        Markdown docs live in docs/
  r4    50        glob   *.md                             docs/           Loose root .md → docs/ (excl. README, CHANGELOG, LICENSE)
  r5    30        glob   tmp*                             .delete         Scratch files
  r6    30        glob   scratch*                         .delete         Scratch files
  r7    20        glob   notes[0-9]*.md                   .archive        Numbered notes → archive

<N> rules total. All active by default.

────────────────────────────────────────────
Options:
  accept all               — use all proposed rules as-is
  review                   — walk through each rule individually
  add <json rule object>   — append a custom rule
  remove <id>              — deactivate rule <id> (marks active=false, records note)
  edit <id> <field> <val>  — modify a rule's field (validated against RULE_FORMAT)
  abort | help
>
```

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `accept all` | Confirm all proposed rules; advance to Stage 4. |
| `review` | Present each rule one at a time. Per rule: `accept | skip | edit <field> <val> | remove`. After iterating all, re-present full list with changes and ask for `accept all` or further edits. |
| `add <json>` | Parse the rule object; validate it against RULE_FORMAT. If valid, append and re-present Stage 3. If invalid, report the specific validation error and remain at Stage 3. |
| `remove <id>` | Mark the rule `active: false`; add system note "deactivated during wizard"; re-present Stage 3. |
| `edit <id> <field> <val>` | Apply the edit to the named rule. Validate the resulting rule object against RULE_FORMAT. If valid, re-present Stage 3. If invalid, report the specific error (see §6, Edge Case 1). |
| `reject` | Ask user what is wrong with the rules; apply feedback; re-present Stage 3. |
| `abort` | Exit wizard; emit `ABORTED_DURING_WIZARD`. |
| `help` | Explain rule format (kind, pattern, destination, priority); re-present Stage 3. |

**Transition rules:**  
- `accept all` → advance to Stage 4.  
- Any edit operation loops Stage 3 until `accept all` is received.  
- S5 rule adjustment (see Stage 5) sends the user back to Stage 3 with adjusted rules pre-loaded; user must re-confirm with `accept all`.

**RULE_FORMAT validation:**  
A rule is valid if:
- `id` is present and a non-empty string.
- `kind` is one of: `glob`, `regex`, `hint`.
- `pattern` is present and non-empty.
- For `glob`: standard `*`/`**`/`?` syntax; no shell-expansion characters (`$`, `!` at pattern start, etc.).
- For `regex`: the pattern must compile as a valid regular expression.
- For `hint`: the pattern is a non-empty natural-language description.
- `destination` is one of: `IN_PLACE`, `.delete`, `.archive`, or a relative path string (must not start with `/`).
- `priority` is a positive integer.
- `active` is a boolean.

---

### Stage 4 — `propose_ignore_paths`

**Purpose:** Present and confirm the ignore_paths list — file patterns ORGANIZE will never consider.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 4 of 6: Ignore Paths ====

Files matching these patterns will never be enumerated or classified.
Root invariants (.claude/, .claude-flow/, .hive-mind/, .mcp.json) are always
ignored and are NOT shown here — they are enforced at the workflow level regardless.

Proposed ignore_paths:
  .git/**
  .venv/**
  __pycache__/**
  *.pyc
  .pytest_cache/**
  .mypy_cache/**
  *.egg-info/**
  dist/**
  build/**
  .delete/**
  .archive/**
  workflows/**

────────────────────────────────────────────
Options:
  accept                   — use proposed ignore_paths
  add <pattern>            — append a glob pattern
  remove <pattern>         — remove a pattern (must not be a root invariant)
  edit                     — walk through each pattern individually
  abort | help
>
```

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `accept` | Confirm ignore_paths; advance to Stage 5. |
| `add <pattern>` | Append the pattern; re-present Stage 4. |
| `remove <pattern>` | If the pattern is NOT a root invariant, remove it; re-present Stage 4. If the pattern IS a root invariant, respond: "Root invariant paths cannot be removed — they are workflow-enforced. Try removing a different pattern." Re-present Stage 4. |
| `edit` | Walk through each pattern one at a time. Per pattern: `keep | remove`. After iterating, re-present full list. |
| `abort` | Exit wizard; emit `ABORTED_DURING_WIZARD`. |
| `help` | Explain what ignore_paths does, note the root-invariants distinction, re-present Stage 4. |

**Transition rules:**  
- `accept` → advance to Stage 5.  
- Root invariants are never shown in the ignore_paths list and cannot be removed via this stage. They are implicit, mandatory, and silent.

---

### Stage 5 — `review_initial_classification`

**Purpose:** Show the user INSPECTOR's dry-run preview of what would happen to loose files if MAINTENANCE ran right now with the proposed rules. Allows the user to catch unexpected verdicts before committing to any config.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 5 of 6: Initial Classification Preview ====

With the proposed rules, here is how INSPECTOR classified visible loose files.
This is a PREVIEW only — no moves happen here. TRIAGE will re-classify during actual maintenance.

  File                              Would-be Verdict          Rationale
  ────────────────────────────────  ────────────────────────  ──────────────────────────────────
  scratch.py                        QUARANTINE:.delete        matches rule r6 (scratch*)
  notes2.md                         QUARANTINE:.archive       matches rule r7 (notes[0-9]*.md)
  tmp_data.csv                      QUARANTINE:.delete        matches rule r5 (tmp*)
  old_design.md                     ASK_USER                  could be .archive or docs/; ambiguous
  my_special_notes.md               ASK_USER                  author-attachment implied by name; no clear rule
  src/run_bench.bench.py            ASK_USER                  no rule covers *.bench.py; 3 similar files seen

Surprises? Use 'adjust rules' to return to Stage 3.

────────────────────────────────────────────
Options:
  ratify         — classification looks correct; proceed to Stage 6
  adjust rules   — return to Stage 3 to modify rules; classification will re-preview
  abort | help
>
```

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `ratify` | Accept preview as-is; advance to Stage 6. |
| `adjust rules` | Return to Stage 3 with current rules pre-loaded. User modifies rules; Stage 5 re-presents updated preview after Stage 3 `accept all`. |
| `reject` | Equivalent to `adjust rules`; QUEEN asks what surprised the user and suggests edits. |
| `abort` | Exit wizard; emit `ABORTED_DURING_WIZARD`. |
| `help` | Explain that this is a preview only; TRIAGE re-classifies during maintenance; ASK_USER verdicts mean the user will be consulted at maintenance time. Re-present Stage 5. |

**Transition rules:**  
- `ratify` → advance to Stage 6.  
- `adjust rules` → return to Stage 3; Stage 5 loops after Stage 3 re-confirms.  
- If INSPECTOR produced no initial_classification (no loose files), QUEEN reports: "INSPECTOR found no loose files — nothing to preview. Project appears well-structured." Offer `ratify` to proceed directly.

---

### Stage 6 — `cascade_decision`

**Purpose:** Show a summary of the config about to be written and ask the user whether to cascade into a single MAINTENANCE circuit immediately, or to halt and let the user invoke MAINTENANCE separately.

**QUEEN prompt template:**

```
==== ORGANIZE BOOTSTRAP — Stage 6 of 6: Review & Cascade ====

Ready to write .organize.json to: <target_dir>/.organize.json

Summary:
  Template:       <template_name>
  Rules:          <N> active rules
  Ignore paths:   <N> patterns
  Created:        <ISO-8601 timestamp>

After writing config, would you like to run 1 MAINTENANCE circuit now?
(Recommended: yes — catches anything INSPECTOR's preview identified.)

────────────────────────────────────────────
Options:
  cascade    — write config + immediately run 1 MAINTENANCE circuit
  halt       — write config only; run MAINTENANCE later
  abort      — exit without writing anything
  help
>
```

**Acceptable responses:**

| Response | QUEEN behavior |
|---|---|
| `cascade` | Write `.organize.json`; commit with message `organize: bootstrap config for <target_name>`; proceed to MAINTENANCE mode with `circuits=1`; emit `CONFIG_CREATED_AND_CLEAN_RUN` at end. |
| `halt` | Write `.organize.json`; commit with message `organize: bootstrap config for <target_name>`; emit `CONFIG_CREATED`; end. |
| `abort` | Exit without writing. Emit `ABORTED_DURING_WIZARD`. No filesystem changes. |
| `help` | Explain what cascade does vs halt; re-present Stage 6. |
| `cascade N` | (Extension) Write config and run N circuits. QUEEN confirms N with user if > 3 (sanity check). |

**Transition rules:**  
- `cascade` or `halt` → write `.organize.json`.  
- `abort` → no write; `ABORTED_DURING_WIZARD`.  
- QUEEN writes `.organize.json` exactly once, after user confirms in Stage 6.

---

## 5. Wizard State Invariants

1. `.organize.json` is NEVER written until Stage 6 confirmation.
2. No moves are executed inside the wizard. All classification in Stage 5 is preview only.
3. No agents are spawned inside the wizard. The wizard is a pure QUEEN+user dialog.
4. Rule edits in Stage 3 are held in-memory only until Stage 6 writes.
5. `abort` at any stage clears in-memory state and emits `ABORTED_DURING_WIZARD`.
6. Stage 5 → Stage 3 loops are allowed but QUEEN must track iteration count. If the user loops more than 5 times, QUEEN flags: "This session has looped Stage 3-5 five times. Type 'abort' if you need to restart, or 'ratify' to accept the current state."

---

## 6. Edge Cases

### Edge Case 1 — User edits a rule's pattern; validation fails

**Scenario:** User types `edit r4 pattern **/.md` (malformed glob — missing filename component).

**QUEEN behavior:**
1. Attempt to validate the edited rule against RULE_FORMAT.
2. Detect the validation error: glob `**/.md` has no filename component after the last `/`.
3. Respond with the specific error:
   ```
   Rule edit rejected: pattern '**/.md' is not a valid glob — '**/' must be followed by a filename
   pattern or wildcard (e.g., '**/*.md'). Edit not applied.
   ```
4. Re-present Stage 3 with the original rule unchanged.

**Principle:** Validation errors are reported with the specific field and reason. The edit is not applied. The user may try again or accept the original.

---

### Edge Case 2 — User aborts mid-wizard

**Scenario:** User types `abort` during Stage 4.

**QUEEN behavior:**
1. Immediately stop the wizard.
2. Discard all in-memory wizard state (proposed template, rules, ignore_paths, any edits).
3. Do NOT write `.organize.json`.
4. Do NOT create `.delete/` or `.archive/`.
5. Do NOT commit anything.
6. Report:
   ```
   ABORTED_DURING_WIZARD — No configuration written. Project state unchanged.
   To restart, invoke ORGANIZE_WORKFLOW again.
   ```

**Principle:** Abort means clean exit. No partial state survives.

---

### Edge Case 3 — User requests INSPECTOR re-run with narrower focus

**Scenario:** User says "re-run INSPECTOR but only look at the src/ directory."

**v0.1.0 behavior (DEFERRED):** INSPECTOR re-run with narrowed focus is not implemented in v0.1.0. INSPECTOR always surveys the full target_dir.

**QUEEN response:**
```
Narrowed INSPECTOR re-runs are not supported in v0.1.0.
Workaround: type 'abort' to exit the wizard, then re-invoke ORGANIZE_WORKFLOW with
mode_override: BOOTSTRAP — this re-runs a full INSPECTOR survey.
If you want to focus on a subdirectory, adjust the ignore_paths in Stage 4
to exclude the directories you don't want classified.
```

**Future:** When implemented, semantics will be: spawn INSPECTOR with `target_dir` set to the narrowed path; merge signals into the existing proposal with user confirmation of any conflicts. Track as T3.3-FOLLOWUP equivalent for wizard.

---

### Edge Case 4 — User provides a custom template (no named template)

**Scenario:** User selects `custom` in Stage 2.

**QUEEN behavior:**
1. Record `template: custom` in the in-memory config.
2. Advance to Stage 3 with an empty rules array.
3. Stage 3 prompt notes:
   ```
   Custom mode — no template pre-loaded. Rules array is empty.
   Add rules using: add <json rule object>
   ```
4. User must add at least one rule to proceed. If user attempts `accept all` with an empty rules array, QUEEN responds:
   ```
   Rules array is empty. Add at least one rule before proceeding, or type 'abort'.
   ```
5. Once rules are added, the wizard continues normally.

---

### Edge Case 5 — User tries to remove a root-invariant from ignore_paths (Stage 4)

**Scenario:** User types `remove .claude/**`.

**QUEEN behavior:**
```
Root invariant paths cannot be removed from ignore coverage.
'.claude/' is a workflow-enforced root invariant (ORGANIZE_WORKFLOW.json §root_invariants)
and will never be enumerated or classified regardless of ignore_paths.
It does not appear in the configurable ignore_paths list because it is not user-configurable.
Try removing a different pattern.
```

Re-present Stage 4 unchanged.

---

### Edge Case 6 — Re-running wizard when `.organize.json` already exists

**Scenario:** User invokes BOOTSTRAP with `mode_override: BOOTSTRAP` when `.organize.json` already exists.

**v0.1.0 behavior:**  
QUEEN detects the existing config and presents the following before entering the wizard:

```
.organize.json already exists (version: <ver>, <N> rules, last run: <timestamp>).
You are re-running BOOTSTRAP with mode_override.

Existing rules are PRESERVED (append-only invariant applies).
The wizard will propose additional rules; you ratify the diff.
Existing rules cannot be deleted — only deactivated (active=false).

Proceed? [yes / abort]
>
```

If `yes`: QUEEN runs the wizard with the existing config pre-loaded. Stage 3 shows existing rules (read-only) plus newly proposed rules. Stage 6 writes only the new/changed fields. The `template` field may be updated only via re-BOOTSTRAP with user confirmation.

**Authoritative source:** ORGANIZE_WORKFLOW.json §`known_tbds` item 8 — exact re-run semantics; this is the v0.1.0 formalization.

---

## 7. Free-Form Input Interpretation

When the user types text that is not a recognized keyword:

1. QUEEN reads the text as intent.
2. If the intent maps unambiguously to a response (e.g., "looks good to me" → `accept`; "add .DS_Store to ignores" → `add .DS_Store`), apply it and re-present.
3. If ambiguous, QUEEN asks:
   ```
   I interpreted that as: <interpretation>.
   Did you mean: [yes / no / <alternative>]?
   ```
4. If the text cannot be interpreted as a wizard action, QUEEN responds:
   ```
   I didn't understand that input. Type 'help' for available options.
   ```

Free-form is a convenience, not a contract. The recognized vocabulary is always authoritative.

---

## 8. Cross-Reference

- `ORGANIZE_WORKFLOW.json` §`wizard_flow` — authoritative stage list
- `ORGANIZE_WORKFLOW.json` §`units.WIZARD_LOOP` — "not a circuit; a dialog"
- `ORGANIZE_WORKFLOW.json` §`root_invariants` — why root invariants cannot be removed from ignore_paths
- `ORGANIZE_WORKFLOW.json` §`verdicts.bootstrap_catalog` — wizard terminal verdicts
- `WORKER_INSPECTOR.md` §5 — proposal block format that feeds wizard stages 1-5
- `BATCHING_HEURISTIC.md` — batching after wizard; cascade into maintenance

---

*End of WIZARD_PROTOCOL.md*
