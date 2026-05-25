# ORGANIZE — The Filesystem Organization Workflow

## Periodic Structural Cleanup for Project Directories

**Version:** 1.0.0
**Author:** Michael (owner) + Claude (co-architect)
**Status:** IMPLEMENTED (as of 2026-04-18). Schema infrastructure, canonical template library, protocol formalizations, test project, four E2E walkthroughs, open-question resolutions, and registry integration all complete. This document is the authoritative design reference and implementation record for the ORGANIZE workflow.

---

## 1. Purpose and Scope

ORGANIZE is an AI swarm workflow that imposes order on cluttered project directories without destroying anything. Its mandate is narrow but consequential: classify loose and misplaced files against a project-specific template and ruleset, propose relocations to the user, and execute approved moves via `git mv`. It never deletes, never modifies file contents, and never acts without explicit user ratification. Every action is reversible.

What ORGANIZE does not do is equally important to state upfront. It does not consolidate prose — that is RDC's territory, operating at content-consolidation altitude. It does not characterize relationships between files or produce analytical reports — that is RECON's territory, operating at relational-characterization altitude. It does not modify code semantics or run tests — that is SDLC's territory. ORGANIZE operates strictly at filesystem-structure altitude: where files live, not what they contain or how they relate semantically.

The problem ORGANIZE addresses is the slow accumulation of structural disorder that every active project exhibits. Scratch files multiply at the root. Notes files accrue without being absorbed into documentation. Backup directories fill with `old_design_v2.md` and `tmp_experiment.py`. The project's ostensible structure — `src/`, `tests/`, `docs/` — remains intact, but the surrounding noise makes it harder to see. ORGANIZE periodically sweeps this noise, proposes what to do with it, and commits the result to git history so the cleanup is auditable.

ORGANIZE does not replace judgment. It externalizes the tedium of enumeration and classification while preserving the human as the final authority on every decision. This asymmetry — AI does the scanning and classification, human does the ratification — is deliberate and structural.

---

## 2. Two Modes: BOOTSTRAP vs MAINTENANCE

### Why Two Modes Instead of One

A single "run ORGANIZE" mode would conflate two fundamentally different entry states. The first time ORGANIZE touches a project, nothing is known: there are no rules, no template, no history. The workflow must discover what kind of project this is, propose structure, and negotiate that structure with the user through an interactive dialog. Call this **configuration creation**. On every subsequent run, the config exists, the rules are established, and the workflow simply applies them to whatever has accumulated since the last run. Call this **rule application**.

These two states require different workers (INSPECTOR vs TRIAGE), different interaction patterns (wizard dialog vs ratification UI), different termination semantics (CONFIG_CREATED vs CLEAN_RUN), and different mental models for the user. Collapsing them into one mode would produce a workflow that is neither the wizard nor the applicator — it would be confused about which job it is doing, and so would the user.

The split makes the entry state explicit. When QUEEN engages, the first thing it checks is the presence or absence of `.organize.json`. The result of this check fully determines which branch executes. There is no ambiguity.

### BOOTSTRAP

BOOTSTRAP is the one-time initialization pass. It fires when no `.organize.json` is found at the project root.

BOOTSTRAP's core question is: "What kind of project is this, and what are the structural conventions we're trying to maintain?" This question cannot be answered mechanically from file paths alone. It requires the user's authorial input. Is this Python library adopting `src/` layout or flat layout? Are the notebooks canonical (in `notebooks/`) or experimental detritus (should be quarantined)? Is that `backup/` directory deliberate historical preservation or accumulated laziness?

To inform these questions without requiring the user to enumerate every file, QUEEN spawns a single INSPECTOR worker. INSPECTOR surveys the project — reading manifests, orientation docs, and a sampling of files — and produces a structured proposal: detected project signals, a suggested template, seed rules, suggested ignore paths, and a preliminary classification of visible loose files. INSPECTOR does the legwork; the user makes the decisions.

QUEEN then enters the WIZARD_LOOP: a 6-stage interactive dialog presenting INSPECTOR's proposal section by section, allowing the user to ratify, edit, or reject each element. Only when Stage 6 is confirmed does QUEEN write `.organize.json`. If the user aborts at any earlier stage, no file is written, no state is changed. The wizard is a pure negotiation; `.organize.json` is the outcome.

After writing the config, BOOTSTRAP typically cascades immediately into one MAINTENANCE circuit, applying the freshly established rules. The user can halt after the config is written if they prefer to review the config before running it — the mode supports both paths.

### MAINTENANCE

MAINTENANCE is the periodic re-invocation. It fires whenever `.organize.json` exists.

MAINTENANCE's core question is: "What has accumulated since the last run, and where does each new file belong?" This is a classification problem, not a configuration problem. The rules are already established. The ignore paths are already set. QUEEN's job is to enumerate the eligible file population, split it into parallel-safe batches, spawn TRIAGE workers, aggregate their verdicts, present the proposed moves grouped by verdict type, accept ratification from the user, execute the moves via `git mv`, and commit.

MAINTENANCE is bounded by N circuits (default 1). A circuit is one full scan-triage-propose-ratify-execute-commit cycle. Multiple circuits make sense when a project has many layers of disorder — the first circuit catches the most obvious issues; subsequent circuits catch patterns that only become apparent after the first wave of moves settles.

The bounded circuit model is a deliberate anti-daemon design. ORGANIZE is not meant to run continuously or automatically. It is meant to be invoked periodically, explicitly, by the user who wants to tidy their workspace. The N-circuit bound ensures each invocation has a known cost, a definite endpoint, and a committed audit trail in git history.

---

## 3. Architecture: QUEEN + INSPECTOR + TRIAGE

### The Orchestrator: QUEEN

QUEEN is Claude in the main conversation during ORGANIZE_WORKFLOW engagement. This is the same architectural role as in RDC, SDLC, and BOOK — not a spawned worker, but the session itself acting as orchestrator. QUEEN reads all required docs at engagement, detects mode, spawns workers, runs interactive dialogs, executes approved moves, writes and updates `.organize.json`, and commits.

The choice to make QUEEN the only actor that writes to the filesystem is not incidental. It creates a single point of control for all side effects. Workers (INSPECTOR, TRIAGE) are read-only. They observe and classify. QUEEN decides and acts — but only on what the user has ratified. This structure means every filesystem modification traces back to a specific user decision in a specific ratification dialog, which traces back to a specific worker verdict, which traces back to a specific file read and rule evaluation. The audit chain is complete.

### The Discovery Worker: INSPECTOR

INSPECTOR is a BOOTSTRAP-only worker. It is spawned once per BOOTSTRAP invocation.

INSPECTOR's stance is that of a surveyor. It enters the project cold, with no prior state, and attempts to characterize what it sees: what kind of project signals are present (language manifests, test frameworks, documentation structure, version control state), what template the project most closely resembles, what rules would capture the existing structure, what paths should be ignored, and what the visible loose files look like at first pass.

INSPECTOR does not classify files definitively. Its Phase 6 output is a preliminary "here's what would happen" preview, not a binding verdict. Definitive classification is TRIAGE's job. INSPECTOR's job is to give the user enough information to make good decisions during the wizard. The distinction matters: INSPECTOR proposes, the user ratifies, TRIAGE executes.

The read-only constraint on INSPECTOR is absolute. INSPECTOR does not write anything to the target project. It does not create directories, does not touch `.organize.json`, does not stage anything for git. Its entire output is a structured proposal returned to QUEEN as an agent response.

### The Classification Worker: TRIAGE

TRIAGE is a MAINTENANCE-only worker. Multiple TRIAGE instances run in parallel during a TRIAGE_WAVE.

TRIAGE's stance is that of a per-file classifier. It receives a batch of file paths, the current ruleset from `.organize.json`, and the project template name. For each file in its batch, it reads the file fully (full-read-or-skip, inherited from SCOUT discipline), evaluates the rules in priority order, and emits one verdict from the fixed catalog: KEEP_IN_PLACE, MOVE_TO, QUARANTINE:.delete, QUARANTINE:.archive, ASK_USER, or FLAG_NEW_RULE.

### The Critical Distinction: Proposer vs Classifier

The deepest architectural difference between INSPECTOR and TRIAGE is their stance: INSPECTOR is a **proposer**, TRIAGE is a **classifier**.

A proposer looks at ambiguous evidence and synthesizes a recommendation. It is comfortable with uncertainty — it signals uncertainty via `open_questions` rather than forcing a verdict. Its output is a starting point for negotiation.

A classifier looks at concrete evidence (a specific file, its content, the active rules) and renders a verdict. It must be honest about ambiguity — that is what ASK_USER is for — but it cannot defer indefinitely. It must process every file in its batch and produce an output for QUEEN to aggregate.

Giving these two different cognitive stances to the same worker would produce an incoherent agent: sometimes proposing, sometimes classifying, never sure which mode it is in. Separating them into two workers gives each a clean contract. INSPECTOR is free to say "I think this is a mixed-research project but I'm not certain; here's what I observed." TRIAGE never says "I think this file is probably...maybe...sort of in the right place." TRIAGE renders verdicts.

---

## 4. The `.organize.json` Config

### Per-Project, Not Central

`.organize.json` lives at the root of the project being organized. It does not live in `workflows/ORGANIZE/`, in a global config directory, or anywhere shared across projects. This is deliberate.

Each project's organizational conventions are specific to that project. A Python library's rules are not the same as a Rust crate's. A book manuscript's rules are not the same as a knowledge base's. Even two projects of the same template type may have idiosyncratic conventions that require project-specific overrides. Centralizing the config would force a false universality.

The per-project placement also makes the config part of the project's self-description. It is committed to the project's own git history, not to some external workflow registry. Any collaborator who clones the project sees its organizational conventions. The config travels with the project.

### Append-Only: Never-Regret Semantics

The `rules` array in `.organize.json` is append-only. Existing rules are never mutated or removed. To deactivate a rule, you set `active: false` and (ideally) note why. The rule remains in the array as a historical record.

The `quarantine_log` is similarly append-only. Every move, every quarantine event, every ratified action accumulates in this array. Nothing is overwritten.

This design embodies what might be called "never-regret semantics." A rule that was written, applied, and then found problematic is not silently removed — it is explicitly deactivated with a record of the deactivation. Three months from now, when a file is mysteriously missing and you search the audit trail, the quarantine_log tells you exactly which run moved it, which rule triggered the verdict, which user ratified the move, and when. This level of traceability is only possible if the log never loses entries.

The append-only constraint is enforced at QUEEN level (never mutates existing rules during MAINTENANCE) and at schema level (ORGANIZE_CONFIG_SCHEMA.json documents the invariant in `notes`). When a user manually edits `.organize.json` in violation of this constraint — removing a rule, for example — QUEEN detects the violation on the next MAINTENANCE engagement and ESCALATEs with a specific error before any workers are spawned.

### Schema Location and Versioning

The config schema is defined in `workflows/ORGANIZE/ORGANIZE_CONFIG_SCHEMA.json`, a JSON Schema Draft 7 document. This file is the canonical reference for what constitutes a valid `.organize.json`. QUEEN validates against it at every MAINTENANCE engagement.

Schema versioning follows SemVer. MINOR and PATCH bumps are backward-compatible; QUEEN loads older-MINOR configs and proceeds with a compatibility warning. MAJOR bumps require explicit migration — QUEEN ESCALATEs and refuses to proceed until the user migrates the config. Migration scripts live at `workflows/ORGANIZE/migrations/<from>_to_<to>.py` when needed; no MAJOR bump has occurred yet.

---

## 5. The Quarantine Model

### Two Destinations, Not One

The quarantine model has two destinations — `.delete/` and `.archive/` — not one. This distinction is important and load-bearing.

`.delete/` is the garbage can. Files go there when they are scratch files, broken experiments, orphaned artifacts, accidental duplicates, or outright junk. The intent is "this is garbage; confirm then purge someday." `.delete/` is always gitignored — it does not belong in the project's history. The user is expected to periodically look at it and run `rm -rf .delete/` when comfortable.

`.archive/` is the historical record. Files go there when they are intellectually valuable but no longer active — superseded design documents, prior architecture notes, old versions of documentation that capture a thought process that is worth preserving. The intent is "this had value once; preserve it forever." `.archive/` is gitignored by default but the user may opt to commit it as an archaeological record of the project's evolution.

Collapsing these into one quarantine destination would force users to decide, at restore time, whether a file was garbage or history. That is the wrong time to make that decision. The workflow should make the semantic distinction at classification time, when the file's content is being read and a rationale is being formed. The two-destination model externalizes that distinction into filesystem structure.

### Reversibility as a Trust Invariant

The never-delete philosophy is not purely about safety — it is about user trust. A workflow that might delete your files requires a fundamentally different trust relationship than a workflow that only moves them. The cognitive overhead of "is it safe to let ORGANIZE run?" is much lower when you know that the worst-case outcome is a file in `.delete/` that you can trivially restore by moving it back.

This trust gradient matters especially during the first few invocations of ORGANIZE on a project. The user has not yet seen how TRIAGE behaves on this specific project's files. They do not know whether TRIAGE will confidently archive a file that they still need. The reversibility of the quarantine model means they can ratify the proposed moves with lower stakes — they can always undo.

As the user observes ORGANIZE's verdict quality over many runs, their trust in the workflow's classification accuracy grows. This is the foundation for the future blanket-approval mechanism (currently deferred) — accumulate enough observed ratification history to identify categories where the user never rejects TRIAGE's verdict, and offer to auto-ratify those categories in the future. The reversibility model supports this gradual trust expansion.

### Path-Mirroring Preserves Provenance

When a file is moved to a quarantine destination, its path under the quarantine root mirrors its original path. `src/scratch.py` becomes `.delete/src/scratch.py`, not `.delete/scratch.py`. `docs/old_design.md` becomes `.archive/docs/old_design.md`.

This mirroring serves a specific purpose: when a user later opens `.delete/` to review what's in there, the path structure tells them where each file came from. Without mirroring, a `.delete/` directory full of `scratch.py`, `utils.py`, `main.py` files — all with ambiguous names — would require the user to open each one to understand its provenance. With mirroring, the directory subtree reflects the original project structure, and the provenance is self-evident.

The mirroring also enables a cleaner restoration operation: if the user decides a file was quarantined incorrectly, they can `git mv .delete/src/scratch.py src/scratch.py` and it returns to exactly where it came from.

---

## 6. Root Invariants

### What They Are

Four paths at every project root are unconditionally untouchable:

- `.claude/` — Claude Code's per-project settings and configuration
- `.claude-flow/` — claude-flow's runtime state and memory
- `.hive-mind/` — hive-mind consensus and swarm state
- `.mcp.json` — MCP server configuration

These are the infrastructure of the AI tooling itself. ORGANIZE runs inside this infrastructure. A workflow that could accidentally classify, move, or archive its own infrastructure would be self-undermining at best and catastrophically destructive at worst. Imagine ORGANIZE moving `.claude/settings.json` to `.archive/` because it matched a "loose dotfile at root" rule — the next time Claude Code engages, its configuration is missing.

These paths are not user-configurable. They are not listed in `ignore_paths` in `.organize.json` (which is user-editable). They are workflow invariants, enforced at a level the user cannot override.

### Defense-in-Depth Enforcement

The enforcement of root invariants is layered across four distinct points in the workflow:

**Layer 1 — QUEEN's enumeration filter.** Before any batching occurs, QUEEN removes all root-invariant paths from the eligible file population. This is the primary defense. TRIAGE never sees these files.

**Layer 2 — INSPECTOR's automatic seeding.** During BOOTSTRAP, INSPECTOR seeds `ignore_paths` with the root invariants (even though they are enforced separately), records their presence or absence as a detected signal, and includes absence in `open_questions`. This makes the invariants visible to the user as an orientation signal.

**Layer 3 — TRIAGE's defensive check.** If a root-invariant path somehow appears in a TRIAGE batch (indicating a QUEEN filter bug), TRIAGE refuses to classify it. It emits ASK_USER with the rationale "root-invariant path reached TRIAGE; QUEEN should have filtered this — refusing to classify." This check fires before any rule evaluation, before any file read.

**Layer 4 — Wizard exclusion.** The wizard's Stage 4 (propose_ignore_paths) does not present root-invariant paths to the user for review or editing. They are implicit and silent. The user cannot remove them from protection even if they try.

This depth of enforcement exists because real-world filesystem scans are noisy. QUEEN's primary filter is correct code, but code has bugs. The defensive TRIAGE check catches the case where the primary filter fails. Having only one enforcement point would mean a single bug could expose root-invariant paths to classification; having four enforcement points means a bug in any one layer is caught by the others.

### Why Not User-Configurable

A user might reasonably ask: "What if I want to move `.hive-mind/` to a different location? Shouldn't I be able to configure ORGANIZE to allow that?"

The answer is no, and the reason is that these paths are not project preferences — they are requirements of the infrastructure that ORGANIZE itself runs on. If `.claude/` is moved by ORGANIZE, the next ORGANIZE invocation cannot load its own configuration. The invariant is self-referential: removing it breaks the workflow that would enforce removing it.

More broadly, certain decisions should not be delegated to workflow configuration. Root invariants represent a class of "this workflow cannot safely handle its own exception here" situations. Documenting that boundary explicitly — as an immutable invariant rather than a configurable preference — is more honest than pretending the workflow can handle every case.

---

## 7. Parallel Batching: TRIAGE_WAVE

### The Problem

MAINTENANCE mode must classify every eligible file in the project. A large project might have dozens or hundreds of files that need classification in a single circuit. Running TRIAGE sequentially on all of them — one worker per file, or one worker per all files — sacrifices either parallelism or context coherence.

The challenge with parallelism in file classification is that TRIAGE verdicts for one file do not normally depend on TRIAGE verdicts for another file. Whether `scratch.py` belongs in `.delete/` is independent of whether `old_design.md` belongs in `.archive/`. This independence is the condition that makes parallelism safe: if classification A does not depend on classification B, they can run simultaneously without producing inconsistent results.

The exception is FLAG_NEW_RULE detection. If three files in a batch all match the same uncovered pattern, TRIAGE notices the recurrence and suggests a new rule. This recurrence detection is inherently batch-local — TRIAGE can only observe the files in its own batch. Cross-batch FLAG_NEW_RULE patterns require a post-wave aggregation step.

### The Algorithm

QUEEN uses the following batching algorithm before spawning TRIAGE workers:

1. Apply root-invariant filter (silent, non-configurable).
2. Apply ignore-path filter (`.organize.json`'s `ignore_paths` array).
3. Traverse the remaining eligible files depth-first, alphabetical within each directory.
4. Accumulate files into the current batch. When the batch reaches BATCH_CAP (30 files), close it and start a new batch.
5. Treat directory boundaries as natural grouping points — files within the same directory cluster in the same batch for FLAG_NEW_RULE coherence, even if the batch would not yet reach BATCH_CAP.

The BATCH_CAP of 30 is calibrated to two requirements simultaneously. At the lower bound: 3+ occurrences in a 30-file batch gives a 10% signal ratio for FLAG_NEW_RULE detection, which is a meaningful recurrence signal. At the upper bound: 30 files at typical file sizes sit within a comfortable TRIAGE read budget. The constant is named and documented — it can be tuned upward when projects grow large enough that 30-file batches produce too many workers.

All TRIAGE workers in a wave run in parallel in a single message: one Agent call per batch, all with `run_in_background: true`. The wave is bounded by batch count, not time. QUEEN waits for all workers to return before proceeding to aggregation.

### Cross-Batch FLAG_NEW_RULE Aggregation

After all TRIAGE workers return, QUEEN aggregates FLAG_NEW_RULE suggestions across batches. Two batches may each have 2 occurrences of the same pattern (individually below the 3-occurrence threshold), but when merged the combined occurrence count reaches or exceeds the threshold. QUEEN normalizes pattern strings and merges occurrence lists, then presents a consolidated FLAG_NEW_RULE proposal to the user covering all matched files across batches.

This aggregation adds complexity to QUEEN's post-wave processing but is necessary for correct behavior at the workflow boundary. Without cross-batch aggregation, a pattern that appears as 2+2 across two batches of 15 would produce four ASK_USER verdicts instead of one FLAG_NEW_RULE. The user would answer the same question four times when a single rule ratification would cover all four files.

---

## 8. The Wizard: Human-in-the-Loop BOOTSTRAP

### Why Interactive

The BOOTSTRAP wizard is the most unusual design element in ORGANIZE. It is a multi-stage, interactive dialog in which QUEEN presents INSPECTOR's proposal section by section and the user ratifies, edits, or rejects each section before proceeding. It is deliberately interactive, not autonomous.

The reason is authorial ownership. The rules in `.organize.json` define what the project's structure should look like. That is an authorial decision, not a mechanical one. When INSPECTOR proposes "all loose `.md` files at the root should go to `docs/`," it is making a recommendation based on observed signals. But whether that is correct for this specific project depends on context that INSPECTOR cannot fully know: Is one of those `.md` files an important design decision that should be at the root? Is the project's documentation strategy still being figured out? Does the user prefer `docs/` to be flat or subdirectory-organized?

Forcing ORGANIZE to answer these questions autonomously would produce a config that may not match the user's actual intentions. The resulting rules would then misclassify files on every subsequent MAINTENANCE run until manually corrected. The wizard front-loads the authorial conversation, producing a config that genuinely reflects the user's intentions — which then runs correctly on every subsequent MAINTENANCE invocation.

### Why Not Autonomous

The consequences of getting the initial config wrong are not trivially reversible. A mismatched rule that routes important files to `.archive/` will quarantine them on the next run. The user may not notice for weeks. The never-delete invariant means the files are recoverable, but the friction of recovery is real.

More fundamentally: the structural decisions captured in `.organize.json` represent the project's organizational philosophy. Automating the production of that philosophy without human input would be producing a philosophy nobody agreed to. The wizard's interactive format ensures that every rule in the initial config was explicitly reviewed and ratified by the user who owns the project.

The 6-stage structure is designed to be efficient, not exhaustive. An experienced user can move through all 6 stages in a few minutes if INSPECTOR's proposal is largely correct. The wizard respects the user's time; it does not pad the dialog with unnecessary confirmations.

### Why Not Autonomous After First Run

The re-run wizard (invoked via `mode_override: BOOTSTRAP` when `.organize.json` already exists) follows the same interactive format. This is correct behavior even though a config already exists, because the re-run wizard is proposing changes to an established set of rules. The stakes of the re-run are at least as high as the initial run — existing rules govern behavior on every subsequent invocation.

The append-only invariant governs the re-run wizard: existing rules are shown read-only; the wizard can propose additional rules or deactivations, but cannot delete or mutate existing entries. This ensures that a wizard re-run never silently removes the history that the rules array has accumulated.

---

## 9. Rule Language

### Three Kinds: Mechanical Power vs Intent Expression

ORGANIZE supports three rule kinds, ordered from most mechanical to most expressive:

**`glob`** uses standard path-pattern matching (`*`, `**`, `?`). A glob rule is evaluated purely on the file's path relative to `target_dir`. No file content is read to evaluate a glob rule. This makes glob evaluation fast and deterministic. Most structural rules — "all `.py` files under `src/` stay in place," "all `scratch*` files go to `.delete/`" — are expressible as globs.

**`regex`** uses regular expressions on the file path. It handles cases where glob cannot express the pattern cleanly: alternation, digit-class patterns, complex anchoring. Regex rules are still path-only and do not require reading file content.

**`hint`** uses a natural-language condition that TRIAGE interprets by reading the file's content. A hint rule might say "if the file's first 10 lines contain a DEPRECATED or SUPERSEDED marker, route to `.archive`." Evaluating this requires reading the file, finding the marker, and assessing confidence. Hint rules are the most expressive but the most expensive — they cannot be evaluated without a full file read.

The design preference is to use the simplest kind that captures the intent: prefer `glob` over `regex` over `hint`. Hint rules should be reserved for genuinely content-dependent classifications that cannot be expressed as path patterns.

### Priority Semantics

Rules are evaluated in descending priority order: higher priority number = evaluated first. The first rule that matches determines the verdict; subsequent rules are not evaluated (first-match-wins). When two rules have identical priority, the one at the lower array index (appearing earlier in the `rules` array) is evaluated first.

Priority collisions are not errors — they are managed by the first-match-wins semantics. Rule authors should assign priorities deliberately to express intended evaluation order: specific patterns at higher priority, broad catch-alls at lower priority. A hint rule (content-aware) that identifies DEPRECATED markers should have higher priority than a glob rule that catches all `*.md` files, so the hint rule has the first opportunity to match.

### Append-Only as Evolution Enabler

The append-only constraint on the rules array is not just a safety mechanism — it is an evolution mechanism. Rules accrue over time as the project evolves and new patterns emerge. A rule added in run 1 (detecting scratch files) sits alongside a rule added in run 7 (detecting experiment notebooks) and a rule added in run 12 (detecting superseded architecture docs). The full history of the project's organizational thinking is present in the array.

When a rule needs to be superseded, the original rule gets `active: false` (with a note explaining why) and a new rule is appended. The user can read the deactivation note and understand why the original rule was retired. This is impossible if rules are mutated in place — mutation erases the prior state, losing the context of why the change was made.

---

## 10. Verdict Catalogs

### BOOTSTRAP Verdicts

The BOOTSTRAP mode emits exactly three workflow-level verdicts:

**CONFIG_CREATED** — The wizard completed successfully and `.organize.json` was written. The user chose not to cascade into a MAINTENANCE circuit immediately. The project is configured; the next invocation will run MAINTENANCE.

**CONFIG_CREATED_AND_CLEAN_RUN** — The wizard completed, the config was written, and a MAINTENANCE circuit executed successfully in the same invocation. This is the most common terminal state when the user allows the cascade.

**ABORTED_DURING_WIZARD** — The user typed `abort` at any point during the wizard before Stage 6 confirmation. No `.organize.json` was written. No filesystem changes. The project is unconfigured. Re-invoking ORGANIZE_WORKFLOW restarts the wizard from Stage 1.

### MAINTENANCE Verdicts

The MAINTENANCE mode emits four workflow-level verdicts:

**CLEAN_RUN** — All N requested circuits completed. All TRIAGE verdicts were processed. All ratified moves were executed. `.organize.json` was updated with the audit trail. The project is as organized as the current rules allow.

**PARTIAL_RUN** — The user halted after some circuits completed but before reaching N. State as of the last committed circuit is preserved in git. The next invocation will re-scan from the current filesystem state.

**ABORTED** — The user aborted before any circuit's moves were executed. No filesystem changes. `.organize.json`'s `last_run` is not updated.

**NOTHING_TO_DO** — All circuits ran, but all TRIAGE verdicts were KEEP_IN_PLACE. The project is already organized per current rules. `.organize.json`'s `runs` counter and `last_run` are updated; no other filesystem changes.

### Why Fixed Catalogs

The fixed verdict vocabulary serves two purposes that would be impossible with free-form output. First, it enables aggregation: QUEEN can group verdicts by type across a TRIAGE_WAVE because every TRIAGE worker speaks the same vocabulary. Second, it enables the ratification UI: the presentation format for "12 files QUARANTINE:.delete — review list / ratify all / skip?" is only expressible if the number of distinct verdict types is known and bounded.

Free-form verdict output would require QUEEN to parse and interpret TRIAGE's natural-language judgment, introducing the possibility of misinterpretation and making aggregation ambiguous. The fixed catalog eliminates that ambiguity by constraining what TRIAGE is allowed to say.

---

## 11. Relationship to Other Workflows

### Parallel Altitude to RDC and RECON

ORGANIZE, RDC, and RECON operate at different altitudes and can be used in any order, independently of each other:

**ORGANIZE** — filesystem-structure altitude. Moves files. Maintains project layout. Never touches file contents.

**RDC** — content-consolidation altitude. Reads prose documents, upserts their concepts into MASTER.md, produces structured engineering documents. Never moves files; works on content.

**RECON** — relational-characterization altitude. Analyzes relationships between files, components, and concepts in a codebase. Produces an analytical report. Does not move files; does not merge content.

A project might run ORGANIZE first (to clean up the layout), then RECON (to understand the structure of what remains), then RDC (to consolidate scattered documentation into MASTER). Or it might run RECON first to understand the project, then ORGANIZE to implement the structural changes RECON revealed. Or ORGANIZE alone, repeatedly, as periodic maintenance. The three are peers at the same tier, not sequential dependencies.

### Downstream-Optional from RECON

ORGANIZE has one optional upstream coupling: if a prior RECON report exists in `RECON/` at the project root, INSPECTOR reads it during BOOTSTRAP as orientation input. A RECON report characterizes the project's content and relationships more deeply than INSPECTOR's own survey can. If RECON has already identified that a project is a Python library with certain structural patterns, INSPECTOR can incorporate that finding into its template proposal rather than rediscovering it.

This coupling is optional and read-only. ORGANIZE never triggers RECON, and RECON's absence does not block ORGANIZE. INSPECTOR works without a RECON report; it just relies more heavily on its own signal detection.

### No Auto-Coupling

ORGANIZE never auto-triggers RDC, RECON, SDLC, or BOOK. If ORGANIZE surfaces files that need consolidation (loose documentation scattered around the project), the workflow surfaces that observation in INSPECTOR's `open_questions` — but the user decides whether to run RDC, and invokes it manually. This anti-coupling principle preserves the user's agency over which workflows run and when.

---

## 12. Hard Rules

The hard rules in ORGANIZE are not guidelines — they are constraints that the workflow enforces unconditionally. Each rule encodes a decision that was made deliberately and should not be overridden in the moment by convenience.

**never_delete:** The workflow never invokes `rm`, `rm -rf`, `unlink`, or any filesystem deletion. Files are moved, not deleted. `.delete/` is the garbage destination and is user-managed after the workflow exits. This rule exists because the cost of an irreversible deletion is unbounded; the cost of an extra step to purge `.delete/` is trivial.

**git_mv_only:** All moves use `git mv` to preserve git history. Untracked files fall back to plain `mv` with `untracked_at_move: true` logged. This rule exists because git history is how you understand what happened to a file over time. A file moved via plain `mv` loses its git ancestry; `git mv` preserves it.

**no_auto_ratify:** No move executes without explicit user ratification. ASK_USER and FLAG_NEW_RULE verdicts halt execution for interactive decision. This rule exists because ORGANIZE's verdicts are proposals, not commands. The user is the authority.

**append_only_rules + append_only_quarantine_log:** Existing entries in the `rules` array and `quarantine_log` are never mutated or removed. This rule exists because the append-only invariant is the foundation of the audit trail. A log that can be rewritten is not a log.

**circuit_bounded:** A single invocation runs exactly N circuits. No infinite loops, no auto-retry, no daemon mode. This rule exists because bounded cost and bounded side effects are prerequisites for user trust.

**no_auto_workflow_trigger:** ORGANIZE never auto-triggers RDC, RECON, SDLC, or any other workflow. This rule exists to preserve the user's agency over which workflows run.

**no_workflow_mixing:** While ORGANIZE is active, SDLC, RDC, RECON, and BOOK operations do not run in the same conversation. This rule exists because interleaved workflow execution creates ambiguous state and makes audit trails unreadable.

**never_no_verify:** Never use `git commit --no-verify` to bypass pre-commit hooks. If a hook fails, fix the underlying issue. This rule exists because pre-commit hooks are the project's own quality gates — bypassing them to accommodate ORGANIZE would be letting the workflow corrupt the project's quality invariants.

**root_invariants_untouchable:** `.claude/`, `.claude-flow/`, `.hive-mind/`, `.mcp.json` are never enumerated, classified, moved, or modified by any part of the workflow. Violation is a critical workflow bug.

**full_read_or_skip:** TRIAGE reads each file fully before emitting a verdict, or emits ASK_USER with a stated blocker. This rule exists because partial reads produce partial verdicts: a file that looks like garbage in its first 50 lines may contain critical information in the rest.

**honest_ambiguity:** When a file could plausibly fit multiple verdicts, TRIAGE emits ASK_USER rather than guessing. False confidence is worse than asking.

**no_fabrication:** Every verdict cites evidence — a file-internal quote, a pattern match, or an explicit absence-of-reference observation. A verdict without evidence cannot be audited and cannot be trusted.

---

## 13. Circuit Bounding

### No Loops, N-Bounded Per Invocation

The circuit model is the primary mechanism by which ORGANIZE bounds its own cost and avoids daemon-like behavior. A single ORGANIZE invocation runs exactly N circuits (default 1). N is specified at invocation time by the user; there is no upper bound the workflow enforces, but the user decides. The workflow does not interpret "more circuits" as "continue until the project is clean" — it interprets it as "run this many circuits, then stop."

This is different from how many cleanup tools work. A daemon that watches the filesystem and continuously reorganizes it would produce unpredictable behavior — files could be moved while the user is working on them. A loop that continues "until NOTHING_TO_DO" would run an unbounded number of circuits on a project that never converges (perhaps because the user's own edits keep introducing new loose files). The N-bounded model keeps both of these failure modes impossible.

### User Agency Through Explicit Re-Invocation

The bounded circuit model expresses a specific value: each cleanup is a deliberate act, not an automated background process. The user chooses when to run ORGANIZE. The user chooses how many circuits to run. The user reviews and ratifies the proposed moves. The user confirms each circuit's commit.

This explicit re-invocation model supports the observation that structural cleanup is a human editorial decision, not a mechanical optimization. A developer working in a codebase has context about why a particular file is at the root, when it will be moved, and whether the current clutter is intentional temporary state or permanent disorder. ORGANIZE cannot always distinguish between these. Requiring explicit invocation ensures the user has made a conscious decision that cleanup is appropriate before ORGANIZE acts.

### Observability Per Circuit

Each circuit produces exactly one git commit: `organize: circuit N of M — moved X files, quarantined Y`. This commit structure makes the circuit's output visible, reviewable, and reversible at the git level. A user who disagrees with the results of circuit 2 can `git revert` that commit without affecting circuit 1.

The granularity of one commit per circuit — rather than one commit per invocation — is deliberate. Lumping all circuits into one commit would make the audit trail less granular and the revert scope larger. One commit per circuit is the right granularity for the unit of work ORGANIZE performs.

---

## 14. Implementation Learnings from Parts 1-5

Building out the ORGANIZE workflow family across 6 parts produced several design insights that are worth recording explicitly.

**1. Separating INSPECTOR from TRIAGE paid off in role clarity.** The initial temptation was to use a single worker that does both BOOTSTRAP discovery and MAINTENANCE classification. The separation into INSPECTOR (proposer, BOOTSTRAP-only) and TRIAGE (classifier, MAINTENANCE-only) produced workers with clearer contracts, cleaner prompts, and less cognitive dissonance. INSPECTOR is comfortable with uncertainty and synthesis; TRIAGE must render verdicts. These are genuinely different cognitive modes. A single worker would be schizophrenic about which mode it was in.

**2. `ASK_USER` is the honest default for ambiguity; heuristics would erode user trust.** Early in the design, there was a temptation to add logic for "if the file looks like X, auto-classify as Y even without high confidence." This was correctly rejected. TRIAGE's `honest_ambiguity` hard rule — when in doubt, emit ASK_USER — produces verdicts the user can trust. A workflow that occasionally auto-classifies incorrectly based on weak heuristics produces verdicts the user must verify. The cost of verifying is higher than the cost of answering an ASK_USER prompt.

**3. `FLAG_NEW_RULE` as emergent-pattern detection is the right architecture for a growing rule library.** The alternative — requiring users to anticipate all patterns upfront during BOOTSTRAP — would produce either over-specified configs (many rules that never fire) or under-specified configs (many files that always produce ASK_USER). FLAG_NEW_RULE allows the rule library to grow organically as real patterns appear in the real project. Three occurrences of the same uncovered pattern across a batch is a robust signal that a rule is warranted.

**4. Append-only log is load-bearing for audit trail; trivial to implement, huge semantic payoff.** The decision to make both `rules` and `quarantine_log` append-only seemed almost too simple during design. In implementation, it proved to be one of the most valuable design decisions. Every file's movement history is permanently recorded. Every rule's evolution (creation, deactivation, the reasoning for changes) is permanently recorded. The cost of the append-only invariant is minimal — entries are small, disk space is cheap. The benefit — a complete, unalterable audit trail — is substantial.

**5. Root invariants needed defense-in-depth, not a single-point filter.** The initial design had QUEEN filtering root invariants at enumeration time, which is the right primary defense. But real-world filesystem scanning code has bugs. Adding the TRIAGE-level defensive check — which fires before any rule evaluation if a root-invariant path appears in a batch — created a second layer that is cheap to implement and invaluable when the primary layer fails. The walkthrough's edge-case trace (Scenario 3 in EDGE_CASE_WALKTHROUGHS.md) demonstrated that the TRIAGE defensive check behaves correctly even in the simulated bug scenario.

**6. The `mixed-research` template being ASK_USER-heavy was the right design choice.** The `mixed-research` template — for projects where code, prose, and data coexist — deliberately emits ASK_USER for several file categories where classification depends on authorial intent rather than file structure. A research notebook at the root might be a canonical work-in-progress or an experiment to quarantine; only the author knows. Forcing a rule-based verdict on genuinely ambiguous files would produce incorrect classifications. The template's ASK_USER density is a feature, not a deficiency: it reflects that `mixed-research` projects have more authorial judgment to apply than structured single-language projects.

**7. Five canonical templates is an adequate minimum library.** The 5 templates (`python-lib`, `rust-crate`, `book-markdown`, `mixed-research`, `knowledge-base`) cover the project shapes in active use. The deferred templates (`python-app`, `rust-app`, `polyglot`, `book-print`) were deferred not because they are unimportant but because no current project needs them and speculative templates tend to fit no real project well. The promotion criteria — a real project that cannot be adequately served by any existing template, with stable rules validated against at least one actual project — ensures that future promotions are grounded in demonstrated need.

---

## 15. Open Questions — Resolved

All 8 TBDs from `ORGANIZE_WORKFLOW.json §known_tbds` have been resolved or explicitly deferred with activation criteria. The full record is in `OPEN_QUESTIONS_RESOLVED.md`. A summary follows.

**TBD #1 — SHORT_RDC sub-procedure shape:** RESOLVED_ELSEWHERE, deferred for v0.1.0. TRIAGE handles prose classification via full-read judgment and ASK_USER for ambiguity. SHORT_RDC will be implemented when ≥10 MAINTENANCE runs + ≥5 prose archival false positives are observed. Reference: `SHORT_RDC_SPEC.md`.

**TBD #2 — Template library promotion:** RESOLVED_ELSEWHERE. 5 canonical templates promoted to `workflows/ORGANIZE/templates/`. 4 additional templates deferred with explicit promotion criteria. Reference: `templates/TEMPLATES_INDEX.md`, `templates/TEMPLATES_ROADMAP.md`.

**TBD #3 — Rule language (glob/regex/hint):** RESOLVED_ELSEWHERE. All three kinds are fully specified with worked examples and TRIAGE interpretation procedure for hint rules. Design preference: use the simplest kind that captures the intent. Reference: `ORGANIZE_RULE_FORMAT.md`.

**TBD #4 — Parallel-safe batching heuristic:** RESOLVED_ELSEWHERE. Algorithm: disjoint directory subtrees + BATCH_CAP=30 files per batch. Pre-batching filters apply root invariants and ignore paths. Cross-batch FLAG_NEW_RULE aggregation procedure defined. Reference: `BATCHING_HEURISTIC.md`.

**TBD #5 — FLAG_NEW_RULE threshold:** RESOLVED. Threshold is 3+ occurrences of the same pattern within a single batch. Cross-batch aggregation via BATCHING_HEURISTIC.md §6 extends this to merged occurrence counts across batches. Reference: `ORGANIZE_WORKFLOW.json §known_tbds` item 5, `WORKER_TRIAGE.md §5`.

**TBD #6 — Config schema versioning migration:** RESOLVED_ELSEWHERE. SemVer policy defined: MINOR/PATCH are backward-compatible; MAJOR requires explicit migration. QUEEN behavior on version drift at engagement is fully specified. Migration scripts deferred (T1.6-FOLLOWUP) pending first MAJOR bump. Reference: `ORGANIZE_SCHEMA_VERSIONING.md`.

**TBD #7 — Blanket-approval rules:** DEFERRED. v0.1.0 is conservative by design; all moves require explicit ratification. Activation criteria: after observational data establishes which verdict categories are reliably correct across multiple projects and runs, with explicit user opt-in. Reference: `ORGANIZE_WORKFLOW.json §notes.trust_gradient_note`.

**TBD #8 — Re-running wizard semantics:** RESOLVED_ELSEWHERE. `mode_override: BOOTSTRAP` preserves all existing rules (append-only invariant); wizard diffs proposed changes against existing config; Stage 3 shows existing rules read-only. Reference: `WIZARD_PROTOCOL.md §edge-cases`.

---

## 16. Future Work

The following areas are candidates for future development, ordered roughly by expected priority:

**Template library growth.** The 5 canonical templates are a starting point. As new project shapes appear that cannot be adequately served by the existing templates or `custom` mode, they should be promoted following the documented promotion criteria: a real project, stable rules, at least one actual project validating the template. The TEMPLATES_ROADMAP documents 4 near-term candidates.

**Blanket-approval activation.** The conservative ratification model is correct for v0.1.0. After sufficient observational data accumulates — specifically, when verdict categories emerge where the user's ratification pattern shows zero rejections across many runs — the blanket-approval mechanism should be designed and activated. The activation criteria require empirical data that does not yet exist.

**SHORT_RDC implementation.** When the prose-archival false-positive rate crosses the activation threshold (≥10 MAINTENANCE runs + ≥5 observed false-positive archival restorations), SHORT_RDC should be implemented as a compressed per-file relevance check. The proposed design in `SHORT_RDC_SPEC.md §4` provides the starting specification: SHORT_RDC compares a prose file's concepts against the project's current MASTER.md (if RDC has been run) or against the README + top-level architecture docs.

**Migration scripts for schema MAJOR bumps.** No MAJOR bump has occurred and no real `.organize.json` files require migration. When the first MAJOR bump becomes necessary, the migration script infrastructure (`workflows/ORGANIZE/migrations/<from>_to_<to>.py`) should be implemented before the bump lands in any real project. The T1.6-FOLLOWUP tag marks this item for revisiting.

**Cross-circuit convergence metrics.** As the multi-circuit model accumulates operational history, it would be useful to track convergence — how many circuits does a typical project need before reaching NOTHING_TO_DO? This data would inform better defaults for the circuit count parameter and might reveal patterns in how different project types converge.

**Binary file classification.** Currently, binary files always produce ASK_USER because TRIAGE cannot read them as text. For projects with many binary files of the same type (model weights, images, compiled artifacts), a pattern-based rule in `.organize.json` — routing `**/*.pkl` to `data/models/` or `**/*.png` to `assets/images/` — would eliminate repeated ASK_USER prompts. The template system already supports this via glob rules that match by extension; templates for binary-heavy projects should include such rules from the start.

---

*End of ORGANIZE_WORKFLOW_DISSERTATION.md*
