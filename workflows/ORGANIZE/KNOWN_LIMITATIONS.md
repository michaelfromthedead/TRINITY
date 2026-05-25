# KNOWN_LIMITATIONS — ORGANIZE Workflow v1.0.0

**Version:** v1.0.0
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW
**Source of observations:** Parts 1-5 buildout, walkthroughs (T4.2–T4.5), edge-case analysis (T3.5, T4.5), protocol design (T3.1–T3.4), open-questions resolution (T5.1).

---

## Overview

This document catalogs known rough edges in ORGANIZE_WORKFLOW v1.0.0. These are not design failures — they are deliberate tradeoffs made in the interest of correctness, conservatism, and early-stage simplicity. Each limitation is documented with severity, description, observation context, and a recommended path forward.

Severity bands:

| Band | Meaning |
|---|---|
| **BLOCKING** | Design fix needed before production use at scale |
| **HIGH** | Significant friction or incorrect behavior in common scenarios; should be addressed in next version |
| **MEDIUM** | Noticeable friction in specific scenarios; acceptable in v1.0.0 with workarounds documented |
| **LOW** | Minor friction or aesthetic issue; acceptable by design or low user impact |

---

## Limitation 1 — Binary File Handling: ASK_USER Per File, Not Coalesced

**Severity:** MEDIUM

**Description:**  
When a project contains multiple binary files (e.g., `.pkl` model weights, `.sqlite` databases, `.png` images, compiled `.so` files), TRIAGE emits an individual `ASK_USER` verdict for each binary, one per file. QUEEN presents each ASK_USER as a separate interactive decision during the ratification dialog. A project with 15 binary files of the same type produces 15 separate ASK_USER prompts, each asking essentially the same question: "where should this `.pkl` go?"

The root cause is that TRIAGE classifies files independently within its batch. FLAG_NEW_RULE detection exists for uncovered pattern recurrence, but binary files produce `ASK_USER` (not `FLAG_NEW_RULE`) because the issue is unreadability, not an uncovered rule. There is no mechanism to coalesce ASK_USER verdicts by reason across a wave.

**Observed in:**  
`EDGE_CASE_WALKTHROUGHS.md §Scenario 1` — binary file handling walkthrough shows per-file ASK_USER dialog. `EDGE_CASES.md §Scenario 1` notes: "If many binary files appear in a project, the user may want to add their patterns to `ignore_paths`."

**Recommended follow-up:**  
Two paths forward:

1. **(Short term — v1.0.0 workaround):** After the first binary file ASK_USER prompt, if the user moves it to a specific location (e.g., `data/models/`), QUEEN can prompt: "Would you like to add a rule routing all `**/*.pkl` files to `data/models/` going forward?" This is a variant of FLAG_NEW_RULE triggered by a user ASK_USER decision rather than TRIAGE pattern detection. Not implemented in v1.0.0.

2. **(Future version):** Post-wave aggregation: QUEEN groups ASK_USER verdicts by rationale before presenting to user. Binary-unreadable verdicts for the same file extension are presented as one group: "3 `.pkl` files — binary, cannot classify. Where should these go? [Specify destination for all / decide individually]"

---

## Limitation 2 — SHORT_RDC Absence: Prose Archival Decisions Rely on TRIAGE Judgment Alone

**Severity:** HIGH (deferred by design)

**Description:**  
When TRIAGE classifies a prose `.md` file as `QUARANTINE:.archive` versus `KEEP_IN_PLACE`, it relies on its own full-read judgment: reading the file, identifying staleness markers, and assessing confidence. For files with clear explicit markers ("DEPRECATED," "SUPERSEDED BY," dated stale headers), this works well. For files that are merely outdated without explicit markers — older design documents that no longer reflect current architecture, notes from a prior technical direction, historical discussions that predate the current stack — TRIAGE may lack the context to judge relevance accurately.

Without SHORT_RDC (a compressed per-file relevance check that compares a prose file's concepts against the project's current architecture or MASTER.md), TRIAGE's archival verdicts on ambiguous prose files are weakly grounded. The fallback is `ASK_USER`, which is correct behavior but transfers the judgment burden to the user without analytical support.

**Observed in:**  
`OPEN_QUESTIONS_RESOLVED.md §TBD #1` — SHORT_RDC explicitly deferred. `SHORT_RDC_SPEC.md §2-3` — deferral rationale and activation criteria. The `mixed-research` template's ASK_USER density for loose `.md` files reflects this limitation: without SHORT_RDC, the template conservatively emits ASK_USER rather than risking incorrect archival verdicts.

**Recommended follow-up:**  
Implement SHORT_RDC when activation criteria are met:
- **Condition A:** ≥ 10 completed MAINTENANCE runs recorded across any combination of projects
- **Condition B:** ≥ 5 prose files that received `QUARANTINE:.archive` verdicts have been subsequently restored by the user

Until then, the interim design is correct: TRIAGE uses full-read judgment with `ASK_USER` as the honest ambiguity fallback. The cost of getting a prose-archival verdict wrong is low (`.archive/` is reversible); the cost of building SHORT_RDC prematurely is speculative coupling to RDC infrastructure before empirical data justifies it.

**Follow-up tag:** T3.3-FOLLOWUP

---

## Limitation 3 — No Blanket-Approval Rules: Every Move Requires Ratification

**Severity:** LOW (by design)

**Description:**  
v1.0.0 is deliberately conservative: every move proposed by TRIAGE requires explicit user ratification. There is no mechanism to mark a verdict category as auto-approve (e.g., "always auto-approve MOVE_TO:docs/ for any `.md` file that matches rule r4"). For projects with stable, well-established rules and many files to classify, this means significant user interaction time per circuit — reviewing and confirming verdict groups that the user expects to be correct every time.

The ratification overhead is acceptable in the early phases of a project's organization history when the user is still calibrating TRIAGE's behavior. It becomes friction in mature projects where the user has high confidence in the established rules.

**Observed in:**  
`OPEN_QUESTIONS_RESOLVED.md §TBD #7`. `ORGANIZE_WORKFLOW.json §notes.trust_gradient_note`: "v0.1.0-DRAFT is conservative by design: every move requires user ratification. As we accumulate data on which verdicts are reliably correct, future versions may introduce blanket-approval for high-confidence categories."

**Recommended follow-up:**  
Accept as design for v1.0.0. Revisit after empirical data establishes which verdict categories are reliably correct. Design blanket-approval as an explicit user opt-in (not silent auto-enablement) for specific rule categories where: (a) the user has ratified all instances across ≥ M runs without a single rejection, (b) the verdict is mechanically determined (glob/regex, not hint-based), and (c) the user explicitly enables it. The specific thresholds (N runs, M consecutive ratifications) should emerge from observational data.

---

## Limitation 4 — Template Library is Minimal (5 Templates)

**Severity:** MEDIUM

**Description:**  
The canonical template library contains 5 templates: `python-lib`, `rust-crate`, `book-markdown`, `mixed-research`, `knowledge-base`. Projects that don't fit any of these shapes must use `custom` mode, which starts with an empty ruleset. In `custom` mode, INSPECTOR proposes seed rules from observed project signals but has no template-level baseline to work from. The user must build the ruleset from scratch via the wizard.

For projects with well-established conventions in domains not covered by the 5 templates (e.g., a Python application with a Flask structure, a polyglot monorepo, a Rust application binary), this means more wizard time and less accurate initial seed rules. The `FLAG_NEW_RULE` mechanism will surface patterns over time, but the initial circuit produces more ASK_USER verdicts than a project with a matching canonical template would.

**Observed in:**  
`OPEN_QUESTIONS_RESOLVED.md §TBD #2`. `templates/TEMPLATES_ROADMAP.md` — 4 deferred templates identified: `python-app`, `rust-app`, `polyglot`, `book-print`.

**Recommended follow-up:**  
Promote templates on demand following the documented criteria: a real project that cannot be adequately served by any existing template or `custom` mode, with stable rules validated against at least one actual project. The TEMPLATES_ROADMAP documents the 4 nearest-term candidates. Do not pre-speculate templates before a real project needs them — templates that fit no actual project well are worse than no template.

---

## Limitation 5 — Cross-Batch FLAG_NEW_RULE Aggregation Adds QUEEN Complexity

**Severity:** LOW

**Description:**  
The post-TRIAGE_WAVE aggregation procedure requires QUEEN to collect FLAG_NEW_RULE suggestions from all batches, normalize pattern strings, merge occurrence lists across batches for the same pattern, and present consolidated proposals. This aggregation step is more complex than strictly per-batch processing would be. The complexity lives entirely in QUEEN's post-wave code path, not in TRIAGE workers.

The tradeoff is clear: per-batch-only FLAG_NEW_RULE detection would miss patterns that appear as 2+2 across two batches (each below the 3-occurrence threshold), producing 4 ASK_USER verdicts instead of 1 consolidated FLAG_NEW_RULE proposal. Cross-batch aggregation produces correct behavior at the cost of additional QUEEN logic.

**Observed in:**  
`BATCHING_HEURISTIC.md §6` — cross-batch aggregation procedure. `OPEN_QUESTIONS_RESOLVED.md §TBD #4`. The walkthrough in `MULTI_CIRCUIT_WALKTHROUGH.md §circuit-1` notes that the 3 `.bench.py` files in a single Batch B are sufficient for FLAG_NEW_RULE without needing cross-batch aggregation in that specific case.

**Recommended follow-up:**  
Accept as design. The complexity is bounded and the implementation is documented in `BATCHING_HEURISTIC.md §6`. The alternative (simpler per-batch-only FLAG_NEW_RULE) produces visibly worse user experience. No change recommended; document the tradeoff for future maintainers.

---

## Limitation 6 — Rule Deactivation via `active: false` May Confuse Users

**Severity:** LOW (by design)

**Description:**  
The append-only invariant for the `rules` array means that deactivated rules remain visible in `.organize.json` with `active: false`. A user who opens the file after several rounds of rule evolution may see a rules array with several entries marked `active: false`, each representing a rule that was superseded or corrected. For users unfamiliar with the append-only design principle, this may look like dead weight or configuration errors.

The deactivated rules are load-bearing as history: they tell the story of how the project's organizational conventions evolved. But this value is not self-evident to a user who opens the file expecting a clean list of currently active rules.

**Observed in:**  
`ORGANIZE_WORKFLOW.json §organize_json_schema.invariants`: "rules array is append-only — existing rules are never mutated or deleted; to disable a rule, set its active=false (and record a reason in notes)." `ORGANIZE_RULE_FORMAT.md §Rule Lifecycle`.

**Recommended follow-up:**  
Add user-facing documentation in the wizard and ratification UI that explicitly explains the append-only semantics: "Deactivated rules are historical records, not dead weight. They explain how the project's organizational conventions evolved." The `note` field of each deactivated rule should contain the reason for deactivation — QUEEN should prompt the user for a deactivation note when `active: false` is set via the wizard re-run.

---

## Limitation 7 — `.organize.json` Schema MAJOR Version Auto-Migration Not Implemented

**Severity:** MEDIUM

**Description:**  
`ORGANIZE_SCHEMA_VERSIONING.md` defines the SemVer policy for `.organize.json` config files and specifies that MAJOR version bumps require explicit migration. However, no migration scripts exist at `workflows/ORGANIZE/migrations/`. If a MAJOR schema bump occurs, QUEEN ESCALATEs and refuses to proceed until the user migrates the config — but there is no tool to assist with the migration. The user must perform the migration manually following the documented schema changes.

This is acceptable for v1.0.0 because no MAJOR bump has occurred and no real `.organize.json` files exist in production. When the first MAJOR bump arrives — and MAJOR bumps happen when breaking changes are required, which is a question of when, not if — the absence of migration tooling will be a friction point.

**Observed in:**  
`OPEN_QUESTIONS_RESOLVED.md §TBD #6 — Caveat (T1.6-FOLLOWUP)`: "Migration scripts are NOT implemented in v0.1.0-DRAFT. No MAJOR version bump has occurred; no real `.organize.json` files exist in production requiring migration. The first MAJOR bump should drive the writing of the first migration script."

**Recommended follow-up:**  
Track the T1.6-FOLLOWUP tag. When a MAJOR schema bump becomes necessary, implement `workflows/ORGANIZE/migrations/<from>_to_<to>.py` before the new schema version is deployed to any project with a real `.organize.json`. The migration script interface (`migrate(config: dict) -> dict`, idempotent, backs up original before overwrite, preserves append-only invariants) is specified in `ORGANIZE_SCHEMA_VERSIONING.md §6`.

**Follow-up tag:** T1.6-FOLLOWUP

---

## Limitation 8 — Circuit Re-Invocation Has No Idempotence Guarantee Across External State

**Severity:** LOW

**Description:**  
Each ORGANIZE circuit produces a git commit that captures the filesystem state as of that circuit's completion. If the filesystem changes between ORGANIZE invocations — the user edits a file, adds new files, removes files manually — the subsequent ORGANIZE invocation will observe a different filesystem state than the previous one. This is expected behavior but means that re-running ORGANIZE on what a user might think is an "unchanged" project can produce different proposals if they have modified files since the last run.

More precisely: ORGANIZE circuits are not idempotent across external filesystem state changes. They are deterministic given the same filesystem state, but the filesystem state is not frozen between invocations. This is unavoidable in a tool that operates on a live project directory.

**Observed in:**  
Implicit in the MAINTENANCE flow (`ORGANIZE_WORKFLOW.json §flow.maintenance`): each circuit "enumerates target files" at circuit start time, not at session start time. External changes between circuits in a multi-circuit session are captured. External changes between sessions are also captured in the next session.

**Recommended follow-up:**  
Document this behavior explicitly in the README and wizard interactions. Users who want strict reproducibility between invocations should avoid modifying the project directory between ORGANIZE runs. This is not a bug — it is correct behavior for a tool operating on a live project — but it should be stated clearly so users are not surprised by "unexpected" proposals on re-invocation. No code change required; documentation change only.

---

## Summary Table

| # | Limitation | Severity | Status |
|---|---|---|---|
| 1 | Binary file ASK_USER per file, not coalesced | MEDIUM | Future version — post-wave ASK_USER grouping by reason |
| 2 | SHORT_RDC absence — prose archival decisions weakly grounded | HIGH | Deferred — activation criteria defined; T3.3-FOLLOWUP |
| 3 | No blanket-approval rules — every move requires ratification | LOW | By design; revisit after observational data |
| 4 | Template library minimal (5 templates) | MEDIUM | Promote on demand per TEMPLATES_ROADMAP criteria |
| 5 | Cross-batch FLAG_NEW_RULE aggregation adds QUEEN complexity | LOW | Accept as design; tradeoff documented |
| 6 | `active: false` deactivated rules may confuse users | LOW | Documentation fix; add deactivation-note prompting |
| 7 | Schema MAJOR auto-migration not implemented | MEDIUM | T1.6-FOLLOWUP — implement when first MAJOR bump needed |
| 8 | No idempotence guarantee across external state changes | LOW | Documentation fix only |

**BLOCKING:** 0  
**HIGH:** 1  
**MEDIUM:** 3  
**LOW:** 4

No blocking limitations. ORGANIZE v1.0.0 is production-ready with the above caveats documented.

---

*End of KNOWN_LIMITATIONS.md*
