# BATCHING_HEURISTIC — Parallel-Safe File Batching for TRIAGE_WAVE

**Version:** v0.1.0
**Status:** Active
**Resolves:** TBD §16.4 (parallel batching heuristic)
**Authoritative spec:** `ORGANIZE_WORKFLOW.json` §`units.TRIAGE_WAVE`, §`root_invariants.enforcement`
**Used by:** QUEEN during MAINTENANCE mode, before spawning TRIAGE_WAVE

---

## 1. Problem Statement

QUEEN must split a project's loose files into batches such that:

1. Each batch is independently classifiable by a TRIAGE worker — no file in batch A depends on the classification of a file in batch B.
2. Batches are small enough that TRIAGE can full-read every file within its time budget.
3. Batches are large enough that per-batch pattern detection (FLAG_NEW_RULE requires 3+ occurrences) has a meaningful signal surface.
4. Batches respect directory boundaries so TRIAGE has coherent structural context.

The heuristic described here is **v0.1.0 — disjoint directory subtrees + per-batch file count cap (BATCH_CAP = 30)**.

---

## 2. Pre-Batching Filters (Applied BEFORE Any Batching)

Two filters run before the batching algorithm sees any file. Order is mandatory:

### 2.1 Root-Invariant Filter (highest priority)

Per `ORGANIZE_WORKFLOW.json` §`root_invariants.enforcement`:

> "QUEEN filters these out of every file enumeration BEFORE batching — TRIAGE never sees them."

The following paths are removed from consideration unconditionally, regardless of any user configuration:

```
.claude/
.claude-flow/
.hive-mind/
.mcp.json
```

These paths and everything beneath them are excluded before any other logic runs. This filter is not user-configurable and produces no log entry — it is silent and mandatory.

If a root-invariant path somehow appears in a batch (indicating a QUEEN-level bug), TRIAGE's defensive check (WORKER_TRIAGE §2a-i) catches it and emits `ASK_USER` with the bug rationale. The batching algorithm must not rely on TRIAGE's defensive check as a primary guard.

### 2.2 Ignore-Path Filter

Files matching any pattern in `.organize.json`'s `ignore_paths` array are excluded. This includes:

- Standard language build artifacts: `.venv/**`, `__pycache__/**`, `*.pyc`, `target/**`, `node_modules/**`, `dist/**`, `build/**`, `*.egg-info/**`
- Git metadata: `.git/**`
- ORGANIZE's own quarantine directories: `.delete/**`, `.archive/**`
- Workflow system: `workflows/**` (when ORGANIZE is being run on a project that contains a `workflows/` directory)
- Any project-specific patterns the user added during BOOTSTRAP or subsequent maintenance

Pattern matching uses glob semantics consistent with the rule language (`*` within a segment, `**` across segments).

**After both filters:** the remaining files are the "eligible population" for batching.

---

## 3. Algorithm — Disjoint Subtrees + BATCH_CAP

```
BATCH_CAP = 30  # maximum files per batch

batches = []
current_batch = []

def traverse(dir):
    if is_root_invariant(dir):
        return                          # pre-filter: should already be excluded
    if matches_any_ignore_path(dir):
        return                          # pre-filter: should already be excluded

    files = sorted([
        f for f in listdir(dir)
        if is_file(f)
        and not is_root_invariant(f)
        and not matches_any_ignore_path(f)
    ])

    for f in files:
        current_batch.append(f)
        if len(current_batch) >= BATCH_CAP:
            batches.append(current_batch)
            current_batch = []

    subdirs = sorted([d for d in listdir(dir) if is_dir(d)])
    for subdir in subdirs:
        traverse(subdir)                # depth-first

traverse(target_dir)

if current_batch:                       # flush remainder
    batches.append(current_batch)
```

**Traversal order:** alphabetical depth-first. Files in a directory are added before descending into subdirectories. This ensures files in the same directory cluster in the same batch, maximizing structural coherence for FLAG_NEW_RULE detection.

**Batch boundaries:** a new batch is started only when `BATCH_CAP` is reached. Batch boundaries do not align to directory boundaries — a batch may straddle two directories if the first directory had fewer than `BATCH_CAP` files. This is intentional: it keeps batches full while maintaining the depth-first traversal order.

**Result:** `batches` is a list of file lists, each of length 1..BATCH_CAP. Total file count across all batches equals the eligible population size.

---

## 4. BATCH_CAP = 30 — Rationale

The value 30 is chosen as a balance between two competing constraints:

**Lower bound (FLAG_NEW_RULE signal):**  
FLAG_NEW_RULE requires 3+ occurrences of an uncovered pattern within a single batch. With fewer than ~10 files per batch, a pattern occurring in 3 files would consume >30% of the batch, making FLAG_NEW_RULE triggers too sensitive (spurious rule proposals). With 30 files, a 3-occurrence pattern consumes 10% of the batch — a reasonable signal-to-noise ratio.

**Upper bound (TRIAGE budget):**  
TRIAGE must full-read every file in its batch before emitting verdicts (no-skim rule, WORKER_TRIAGE §4). Realistic prose files in a mixed-research or Python project average 100-300 lines. At 30 files × 300 lines, a batch is ~9,000 lines — within a single agent's comfortable read budget. At 50+ files, budget pressure increases the risk of TRIAGE skipping files and emitting forced `ASK_USER` verdicts.

**Tuning policy:** BATCH_CAP is a named constant, not a magic number. If observational data shows that 30 is consistently too small (producing too many batches with no FLAG_NEW_RULE surface) or too large (producing budget-stressed TRIAGE workers), it should be bumped in a version update with a changelog entry. The constant name `BATCH_CAP` ensures it appears in a single place.

---

## 5. TRIAGE_WAVE Execution

After batching:

1. QUEEN spawns one TRIAGE worker per batch, all in a single message (parallel execution via `run_in_background: true`).
2. Each TRIAGE worker receives: its batch file list, the full `.organize.json` contents, and `target_dir`.
3. TRIAGE workers run in parallel. QUEEN does NOT proceed to aggregation until ALL workers in the wave have returned.
4. Individual TRIAGE worker failure is handled by retry or escalation — not by abandoning the wave.

Per `ORGANIZE_WORKFLOW.json` §`units.TRIAGE_WAVE`:
> "A wave is bounded by batch count, not time. All TRIAGE workers must return before QUEEN proceeds to aggregation."

---

## 6. Cross-Batch FLAG_NEW_RULE Aggregation

FLAG_NEW_RULE detection is per-batch (TRIAGE sees only its assigned files). After all TRIAGE workers return, QUEEN performs cross-batch aggregation:

### 6.1 Aggregation procedure

1. Collect all `FLAG_NEW_RULE` suggestions across all batches.
2. Group suggestions by `suggested_rule.pattern` (normalized to lowercase, glob-canonical form).
3. For each group with the same pattern:
   - If only one batch surfaced it: present it as a single FLAG_NEW_RULE proposal with the per-batch occurrence count and file list.
   - If two or more batches surfaced the same pattern: merge the occurrence lists; present a single consolidated FLAG_NEW_RULE proposal with the combined occurrence count and merged file list. Note in rationale: "pattern observed across N batches, M total files."
4. The merged proposal is presented to the user once during ratification — ratifying the rule once covers all affected files.

### 6.2 Pattern normalization for grouping

Two patterns are considered "the same" for aggregation if:
- They are identical after lowercasing and removing leading/trailing whitespace.
- They represent the same semantic glob (e.g., `*.bench.py` and `**/*.bench.py` are different — do not conflate unless they are textually identical).

### 6.3 Rationale for cross-batch aggregation

Without aggregation, a pattern that appears 2 files per batch across 5 batches (10 total files) would never reach the 3-occurrence threshold in any single batch and would surface as ASK_USER × 10 instead of a single FLAG_NEW_RULE proposal. Cross-batch aggregation prevents this explosion while preserving the per-batch detection threshold.

---

## 7. Edge Cases

### Edge Case 1 — Zero loose files after filtering

**Scenario:** After both pre-batching filters are applied, the eligible population is empty.

**QUEEN behavior:**
1. Skip the batching algorithm entirely.
2. Skip TRIAGE_WAVE entirely.
3. Emit `NOTHING_TO_DO` verdict immediately.
4. Update `.organize.json`: increment `runs` counter, update `last_run`. No filesystem changes to project files.
5. Commit: `organize: update run metadata` (no circuit commit, as no circuit ran meaningfully).

**User-facing message:**
```
All files are in ignored paths or already match IN_PLACE rules.
No loose files found. Verdict: NOTHING_TO_DO.
```

---

### Edge Case 2 — Single batch exceeds BATCH_CAP after filtering

**Scenario:** The eligible population is 47 files, all in the project root (flat project structure). BATCH_CAP = 30.

**QUEEN behavior:**
1. The algorithm produces 2 batches: batch 1 (30 files), batch 2 (17 files).
2. QUEEN spawns 2 TRIAGE workers.
3. No special handling is needed — this is the algorithm working correctly.
4. QUEEN emits an observational warning in the run log (not a blocker):
   ```
   Note: project has a flat structure with many root-level files.
   Consider adding subdirectories to improve organization and reduce batch count.
   ```

**When BATCH_CAP feels wrong:** if a project consistently produces a large number of batches (>10 per circuit), this is a signal that BATCH_CAP may be too small for this project's scale, or that the project needs structural reorganization. This is observational, not a workflow error.

---

### Edge Case 3 — All files in a single directory subtree

**Scenario:** All eligible files are under `src/legacy/` — 45 files.

**QUEEN behavior:**
1. Traverse produces: batch 1 (30 files from `src/legacy/`), batch 2 (15 remaining files from `src/legacy/`).
2. Two TRIAGE workers run in parallel on the same logical subtree.
3. Because both batches are from the same directory, FLAG_NEW_RULE patterns observed in batch 1 may also appear in batch 2. Cross-batch aggregation (§6) handles this correctly.

---

### Edge Case 4 — File count exactly equals BATCH_CAP

**Scenario:** Eligible population is exactly 30 files.

**QUEEN behavior:** A single batch of 30 files. One TRIAGE worker spawned. No cross-batch aggregation needed.

---

### Edge Case 5 — Ignore-path filter removes a file that was in ignore_paths but was manually un-ignored

**Scenario:** User removes `.venv/**` from ignore_paths between runs.

**QUEEN behavior:** `.venv/**` files are now included in the eligible population. They enter the batching algorithm normally. TRIAGE will likely emit `FLAG_NEW_RULE` or `ASK_USER` for Python environment files (no existing rule covers them). User ratifies or re-adds `.venv/**` to ignore_paths.

This is expected behavior — ignore_paths is user-controlled and changes between runs are honored.

---

## 8. Relationship to Rule Evaluation

Batching is a logistics operation — it determines which TRIAGE worker sees which files. It does not affect rule evaluation. Each TRIAGE worker receives the full `.organize.json` including all rules. Rule priority ordering, first-match-wins, and hint interpretation are performed by TRIAGE independently for each file regardless of which batch the file is in.

---

## 9. Cross-Reference

- `ORGANIZE_WORKFLOW.json` §`units.TRIAGE_WAVE` — wave semantics
- `ORGANIZE_WORKFLOW.json` §`root_invariants.enforcement` — pre-filter mandate
- `ORGANIZE_WORKFLOW.json` §`known_tbds` item 4 — the TBD this document resolves
- `ORGANIZE_WORKFLOW.json` §`known_tbds` item 5 — FLAG_NEW_RULE threshold (3+)
- `WORKER_TRIAGE.md` §2a-i — TRIAGE's defensive root-invariant check
- `WORKER_TRIAGE.md` §3 — FLAG_NEW_RULE cross-file pattern detection (per-batch)
- `RATIFICATION_UI_SPEC.md` §3 — how FLAG_NEW_RULE proposals are presented after aggregation

---

*End of BATCHING_HEURISTIC.md*
