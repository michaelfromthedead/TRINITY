# OPEN_QUESTIONS_RESOLVED — ORGANIZE Workflow TBD Resolution Log

**Version:** v0.1.0
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Source of TBDs:** `ORGANIZE_WORKFLOW.json` §`known_tbds`
**Scope:** Resolves all 8 TBDs from the v0.1.0-DRAFT specification.

---

## Summary

| # | Topic | Status | Resolved in |
|---|---|---|---|
| 1 | SHORT_RDC sub-procedure shape | RESOLVED_ELSEWHERE | `SHORT_RDC_SPEC.md` |
| 2 | Template library promotion | RESOLVED_ELSEWHERE | `templates/TEMPLATES_INDEX.md` + `templates/TEMPLATES_ROADMAP.md` |
| 3 | Rule language (glob vs regex vs hint) | RESOLVED_ELSEWHERE | `ORGANIZE_RULE_FORMAT.md` |
| 4 | Parallel-safe batching heuristic | RESOLVED_ELSEWHERE | `BATCHING_HEURISTIC.md` |
| 5 | FLAG_NEW_RULE threshold | RESOLVED | `ORGANIZE_WORKFLOW.json` §`known_tbds` + `WORKER_TRIAGE.md` §5 |
| 6 | Config schema versioning migration | RESOLVED_ELSEWHERE | `ORGANIZE_SCHEMA_VERSIONING.md` |
| 7 | Blanket-approval rules | DEFERRED | `ORGANIZE_WORKFLOW.json` §`notes.trust_gradient_note` |
| 8 | Re-running wizard semantics | RESOLVED_ELSEWHERE | `WIZARD_PROTOCOL.md` §edge-cases |

**Counts:** RESOLVED: 1 | RESOLVED_ELSEWHERE: 6 | DEFERRED: 1

**Follow-ups flagged:** T1.6-FOLLOWUP (migration scripts), T3.3-FOLLOWUP (SHORT_RDC implementation)

---

## Per-TBD Entries

---

### TBD #1 — SHORT_RDC Sub-Procedure Shape

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "SHORT_RDC sub-procedure shape — what exactly does the compressed RDC-style relevance check for prose files look like? For v0.1.0-DRAFT, TRIAGE uses its own full-read judgment; full RDC_WORKFLOW is a manual escape hatch."

**Resolution:**
SHORT_RDC is **DEFERRED** for v0.1.0. TRIAGE handles prose file classification through two mechanisms in the interim:

1. **Full-read judgment:** TRIAGE reads the prose file fully and uses content plus active rules to emit a verdict. For prose with clear staleness markers (e.g., "DEPRECATED", "superseded by", dated stale headers), this is sufficient.
2. **ASK_USER for ambiguity:** When TRIAGE cannot confidently distinguish KEEP_IN_PLACE from QUARANTINE:.archive, it emits `ASK_USER`. QUEEN surfaces this to the user during ratification.

**Activation criteria for future implementation (both conditions must be met):**
- **Condition A:** ≥ 10 completed MAINTENANCE runs recorded across any combination of projects
- **Condition B:** ≥ 5 prose files that received QUARANTINE:.archive verdicts have subsequently been restored by the user (moved back out of `.archive/` to their canonical location)

**Rationale for deferral:**
Adding SHORT_RDC before observing failures introduces speculative coupling to RDC_WORKFLOW infrastructure. The cost of getting a prose-archival verdict wrong is low (`.archive/` is reversible). ASK_USER is a valid interim solution. Building SHORT_RDC before seeing empirical false-positive data would be premature.

**Reference:** `SHORT_RDC_SPEC.md` §2 (decision) + §3 (activation criteria) + §4 (proposed future design)

**Status:** RESOLVED_ELSEWHERE

**Follow-up tag:** T3.3-FOLLOWUP — revisit after ≥10 maintenance runs with false-positive archival data

---

### TBD #2 — Template Library Promotion

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Template library — should common templates (python-lib, rust-crate, book-markdown, etc.) be promoted to workflows/ORGANIZE/templates/*.json for cross-project reuse? For v0.1.0-DRAFT, each project's `.organize.json` is standalone."

**Resolution:**
**RESOLVED** — 5 canonical templates have been promoted to `workflows/ORGANIZE/templates/`:

| Template | File | Project shape |
|---|---|---|
| `python-lib` | `python-lib.json` | Python library with `src/` + `tests/` layout |
| `rust-crate` | `rust-crate.json` | Rust library crate with Cargo workspace |
| `book-markdown` | `book-markdown.json` | Manuscript project (BOOK workflow family target) |
| `mixed-research` | `mixed-research.json` | Code + prose + data coexisting (applied research) |
| `knowledge-base` | `knowledge-base.json` | Topic-organized markdown wiki |

4 additional templates are **deferred** with explicit promotion criteria:
- `python-app` — deferred: too much variation across app shapes; needs concrete project validation
- `rust-app` — deferred: nearly identical to `rust-crate`; no current project needs differentiation
- `polyglot` — deferred: requires template variable support not yet in the rule schema
- `book-print` — deferred: no current LaTeX/Typst project; pure speculation

**Promotion criteria:** a real project that cannot be adequately served by any existing canonical template or `custom` mode, with stable rules validated against at least one actual project.

**Rationale:** Promoting 5 templates covers the project shapes currently in active use. The `custom` mode with FLAG_NEW_RULE handles unusual shapes. Over-specifying deferred templates before seeing real-world usage would produce templates that fit no actual project well.

**Reference:** `templates/TEMPLATES_INDEX.md` (5 canonical templates) + `templates/TEMPLATES_ROADMAP.md` (4 deferred templates + promotion criteria)

**Status:** RESOLVED_ELSEWHERE

---

### TBD #3 — Rule Language (Glob vs Regex vs Natural-Language Hint)

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Rule language — glob vs regex vs natural-language hint. v0.1.0-DRAFT supports all three via rule.kind; observe which get used."

**Resolution:**
**RESOLVED** — all three rule kinds are supported and fully specified:

| Kind | When to use | Evaluation mechanism |
|---|---|---|
| `glob` | Path-based matching; standard `*`/`**`/`?`; simplest form; covers most structural rules | Applied to file path relative to `target_dir`; matched by TRIAGE mechanically |
| `regex` | Path-based matching with alternation, digit classes, or complex patterns; use when glob cannot express the intent cleanly | Applied to file path relative to `target_dir`; matched by TRIAGE mechanically |
| `hint` | Content-dependent classification; semantic meaning; relationship to other files | TRIAGE reads file fully, interprets the natural-language condition, evaluates evidence with conservative fallback to ASK_USER |

**TRIAGE interpretation procedure for `hint` rules:**
1. Read the file fully (full-read-or-skip applies without exception)
2. Parse the hint's condition from the natural-language `pattern` field
3. Gather evidence from the file: content, imports, references, dates, markers, tone
4. Evaluate the condition against the evidence; interpret conservatively
5. HIGH confidence → emit the verdict prescribed by `destination`
6. MEDIUM or LOW confidence → emit `ASK_USER` with hint rule text in rationale
7. Never fabricate evidence; every verdict cites a file-internal observation

**Worked examples:** See `ORGANIZE_RULE_FORMAT.md` §4.4 (Examples H1–H4)

**Design preference:** Use the simplest kind that captures the intent — prefer `glob` over `regex` over `hint`.

**Reference:** `ORGANIZE_RULE_FORMAT.md` — full rule kind documentation with worked examples for all three kinds

**Status:** RESOLVED_ELSEWHERE

---

### TBD #4 — Parallel-Safe Batching Heuristic

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Parallel-safe batching heuristic — for v0.1.0-DRAFT, QUEEN uses 'disjoint subtrees + cap per-batch file count' as the batching heuristic. Revise when we see it fail."

**Resolution:**
**RESOLVED** — v0.1.0 heuristic is: **disjoint directory subtrees + BATCH_CAP = 30 files per batch**.

**Algorithm (pseudocode):**
```
BATCH_CAP = 30
batches = []
current_batch = []
traverse(target_dir):
  files = sorted eligible files in this dir (post root-invariant + ignore-path filtering)
  for f in files:
    current_batch.append(f)
    if len(current_batch) >= BATCH_CAP:
      batches.append(current_batch); current_batch = []
  for subdir in sorted subdirs:
    traverse(subdir)  # depth-first
if current_batch: batches.append(current_batch)
```

**Pre-batching filters (mandatory, in order):**
1. Root-invariant filter (`.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json`) — silent, non-configurable
2. Ignore-path filter (`.organize.json`'s `ignore_paths` array) — user-configurable

**BATCH_CAP = 30 rationale:** Lower bound — 3+ occurrences in 30-file batch gives 10% signal ratio for FLAG_NEW_RULE; upper bound — 30 files × ~300 lines = ~9,000 lines, within comfortable TRIAGE read budget. Tunable via named constant.

**Cross-batch FLAG_NEW_RULE aggregation procedure:**
1. Collect all FLAG_NEW_RULE suggestions across all batches
2. Group by `suggested_rule.pattern` (normalized to lowercase, glob-canonical)
3. Merge occurrence lists across batches for same-pattern suggestions
4. Present consolidated single proposal to user covering all matched files across batches

**Reference:** `BATCHING_HEURISTIC.md` — full algorithm pseudocode + BATCH_CAP rationale + cross-batch aggregation + 5 edge cases

**Status:** RESOLVED_ELSEWHERE

---

### TBD #5 — FLAG_NEW_RULE Threshold

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "FLAG_NEW_RULE threshold — how many occurrences of a pattern in a batch trigger a FLAG_NEW_RULE suggestion vs just ASK_USER per file? v0.1.0-DRAFT: 3+ occurrences of same pattern across the batch triggers FLAG_NEW_RULE."

**Resolution:**
**RESOLVED** — threshold is **3+ occurrences of the same pattern within a single batch** triggers FLAG_NEW_RULE. Fewer than 3 occurrences → ASK_USER per file.

**Cross-batch extension (resolved via T3.2 / BATCHING_HEURISTIC.md):** After all TRIAGE workers return, QUEEN aggregates FLAG_NEW_RULE suggestions across batches. If the same pattern appears in multiple batches (each below the 3-occurrence threshold individually), cross-batch aggregation merges the occurrence lists. A merged pattern that reaches 3+ total occurrences across batches is presented as a consolidated FLAG_NEW_RULE proposal rather than per-file ASK_USER.

**Reference:** `ORGANIZE_WORKFLOW.json` §`known_tbds` item 5 (threshold) + `WORKER_TRIAGE.md` §5 Step 3 (per-batch detection) + `BATCHING_HEURISTIC.md` §6 (cross-batch aggregation)

**Status:** RESOLVED

---

### TBD #6 — Config Schema Versioning Migration

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Config schema versioning — when we bump version, how do we migrate existing `.organize.json` files? TBD."

**Resolution:**
**RESOLVED** — SemVer policy fully defined. QUEEN behavior on version drift at MAINTENANCE engagement is spec'd:

| Scenario | QUEEN behavior |
|---|---|
| Same version | Proceed normally |
| Older MINOR/PATCH (same MAJOR) | Load and proceed with compatibility warning; write new version at end of circuit |
| Older MAJOR | ESCALATE; do not proceed; user must migrate |
| Newer than QUEEN understands | ESCALATE; do not proceed |
| DRAFT suffix | Treat as numeric version; apply normal rules |

**Migration script policy:**
- Scripts live at `workflows/ORGANIZE/migrations/<from>_to_<to>.py`
- Interface: `migrate(config: dict) -> dict`; idempotent; backs up original before overwrite
- Invariants: never deletes rules or quarantine_log entries (append-only preserved across migrations)

**Caveat — T1.6-FOLLOWUP:** Migration scripts are NOT implemented in v0.1.0-DRAFT. No MAJOR version bump has occurred; no real `.organize.json` files exist in production requiring migration. The first MAJOR bump should drive the writing of the first migration script.

**Reference:** `ORGANIZE_SCHEMA_VERSIONING.md` §2 (SemVer policy) + §3 (bump criteria) + §4 (QUEEN decision tree) + §6 (migration script spec + deferral rationale)

**Status:** RESOLVED_ELSEWHERE

**Follow-up tag:** T1.6-FOLLOWUP — implement `workflows/ORGANIZE/migrations/<from>_to_<to>.py` when a MAJOR schema bump occurs and real `.organize.json` files need migrating

---

### TBD #7 — Blanket-Approval Rules

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Blanket-approval rules — should users be able to mark certain rule categories as auto-ratify (e.g., 'auto-approve MOVE_TO:docs/ for any .md file under 100 lines')? v0.1.0-DRAFT: everything requires ratification. Future enhancement after we see ratification patterns."

**Resolution:**
**DEFERRED** — v0.1.0 is conservative by design. All moves require explicit user ratification. No blanket-approval mechanism exists in v0.1.0.

**Rationale for deferral:**
The conservative approach is correct at this stage. We do not yet have observational data on which verdict categories are reliably correct across projects and over time. Introducing blanket-approval prematurely risks auto-executing incorrect moves without user awareness. The cost of requiring ratification is low (a few seconds per verdict group); the cost of an auto-approved incorrect move is higher (misplaced files that the user may not notice).

**Activation criteria for future implementation:**
After ≥ N MAINTENANCE runs with full observational ratification pattern data, identify verdict categories where:
1. The user has ratified all instances of that verdict type across ≥ M runs without a single rejection
2. The verdict type is mechanically determined (glob/regex rule match, not hint-based)
3. The user explicitly opts in to blanket-approval for that category (no silent auto-enablement)

The specific thresholds (N runs, M consecutive ratifications) should be determined empirically from the ratification data.

**Reference:** `ORGANIZE_WORKFLOW.json` §`hard_rules.no_auto_ratify` + §`notes.trust_gradient_note`

**Status:** DEFERRED

---

### TBD #8 — Re-Running Wizard Semantics

**Question (verbatim from ORGANIZE_WORKFLOW.json):**
> "Re-running wizard — what happens if user wants to re-run BOOTSTRAP when `.organize.json` already exists? v0.1.0-DRAFT supports mode_override; exact semantics (preserve rules? start fresh? diff?) TBD."

**Resolution:**
**RESOLVED** — `mode_override: BOOTSTRAP` forces BOOTSTRAP even when `.organize.json` exists. The append-only invariant governs existing rules: they are preserved and shown as read-only during Stage 3; the wizard proposes additional or changed rules only.

**Detailed semantics:**
1. QUEEN detects the existing config and presents a pre-wizard notice: version, rule count, last run timestamp, and warning that existing rules are PRESERVED.
2. Stage 3 shows existing rules (read-only) plus newly proposed rules. The user cannot delete existing rules — only deactivate them (mark `active: false`).
3. Stage 6 writes only new/changed fields into `.organize.json`. Existing rules array is append-only.
4. The `template` field may be updated to a new template via the re-run, with explicit user confirmation.

**Edge cases covered in WIZARD_PROTOCOL:**
- User aborts mid-wizard → no writes; no changes (clean exit invariant)
- User tries to remove a root invariant from ignore_paths → rejected with explanation
- User provides an invalid rule → validation error reported with specific field; edit not applied
- User selects `custom` template → empty rules array (but existing rules remain in the file if re-running on existing config)
- User requests narrowed INSPECTOR re-run → not supported in v0.1.0; workaround documented

**Reference:** `WIZARD_PROTOCOL.md` §6, Edge Case 6 (re-running wizard) + §5 (wizard state invariants) + §4 Stage 3 (propose_seed_rules, showing existing rules read-only)

**Status:** RESOLVED_ELSEWHERE

---

## Follow-Ups Summary

| Tag | Description | Trigger condition |
|---|---|---|
| T1.6-FOLLOWUP | Implement migration scripts in `workflows/ORGANIZE/migrations/` | When first MAJOR schema bump occurs and real `.organize.json` files need migrating |
| T3.3-FOLLOWUP | Implement SHORT_RDC sub-procedure | When ≥10 MAINTENANCE runs + ≥5 prose false-positive archival restorations are observed |

Both follow-ups are correctly DEFERRED: no empirical data exists yet to justify implementation. The tags serve as reminders to re-evaluate after the specified conditions are met.

---

*End of OPEN_QUESTIONS_RESOLVED.md.*
