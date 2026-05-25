# RECOVERY_WALKTHROUGH — T-01.1.1 RECOVER ON-DEMAND SPAWN

**Scenario:** T-01.1 fails completion criteria across two attempts. T-01.1.1 RECOVER spawns on-demand, applies a recovery strategy, and unsticks T-01.1.

**Purpose:** Mental trace demonstrating on-demand spawn semantics + recovery discipline.

---

## Setup

Same engagement as HAPPY_PATH_WALKTHROUGH, but T-01.1's first PHASE_EXECUTOR misses STRUCTURAL primitives entirely (only finds UNIVERSAL atoms).

---

## T-01.1 attempt 1 (T+0:00)

QUEEN spawns PHASE_EXECUTOR for T-01.1 (first attempt).

PHASE_EXECUTOR:
- Reads STEP 1 + context.md
- Identifies 6 primitives at Tier 2 — all classified as UNIVERSAL
- Writes outputs

PHASE_QA verifies:
- Acceptance command passes (count is in [5,15]) ✓
- BUT: completion criterion #3 ("All five primitive types considered") — completion report shows STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL all empty, with no rationale per type
- Specifically, STRUCTURAL primitives (transpose, reshape, broadcast, indexing) are notably absent from a tabular-data library that supports column reorder + reshape operations
- High finding: "STRUCTURAL primitive class is empty without justification — pandas_mini supports column reorder, melt-like reshape, broadcasting in arithmetic. Per `LANGS_DEV_RDC/CLARIFICATION.md` §4, STRUCTURAL primitives are 'often missed because they cost nothing'."
- Verdict: **TASK_FAIL_RETRY**

QUEEN re-spawns PHASE_EXECUTOR for T-01.1 with `prior_retry_findings = [STRUCTURAL absence finding]`. Retry counter = 1.

---

## T-01.1 attempt 2 (T+0:15)

PHASE_EXECUTOR addresses the finding:
- Re-reads context.md §The Outlier Principle
- Looks for zero-cost reframings in pandas_mini
- Finds: column reorder (Sort by column doesn't actually transform the data, just permutes index — but there's no standalone "Reorder" primitive in pandas_mini's surface)
- Argues in completion report: "STRUCTURAL primitives genuinely absent in pandas_mini — the trimmed surface lacks standalone transpose/reshape/broadcast operations"
- Re-emits outputs with same 6 UNIVERSAL primitives + a paragraph in deconstruction_notes.md justifying STRUCTURAL absence

PHASE_QA verifies:
- Acceptance command passes ✓
- Re-checks pandas_mini's __init__.py: no transpose, no melt, no pivot, no broadcast — STRUCTURAL absence is genuine for this trimmed library
- BUT: Tier 1 primitives are missing entirely. The completion report has 6 Tier 2 primitives but zero Tier 1.
- Per `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_ARCH.md` §2.3: "Each tier compresses the previous ~2-5×. Total compression = product." A flat-Tier-2 catalog has no compression structure to verify.
- High finding: "Tier 1 (COMPUTATIONAL) is empty. tier_compression.md cannot compute Tier 1→2 ratio. Either the library has no Tier 1 primitives (unusual for any data library) or they were missed."
- Verdict: **TASK_FAIL_RETRY**

QUEEN: retry counter is now 2 (limit reached). Per workflow rules:
- T-01.1 cannot retry directly anymore
- BUT recovery is available: spawn T-01.1.1 with `recovery_mode: true`

---

## T-01.1.1 attempt 1 — Recovery (T+0:30)

QUEEN spawns PHASE_EXECUTOR with:
- `task_id: "T-01.1.1"`
- `source_docs: ["STEP 1.1 - RECOVER OPS.md"]`
- `recovery_mode: true`
- `prior_retry_findings`: T-01.1's two STEP_QA verdicts (STRUCTURAL absence + Tier 1 absence)
- T-01.1's outputs (primitives_catalog.json, deconstruction_notes.md) for reference

PHASE_EXECUTOR (recovery mode):
- Reads STEP 1.1 fully
- Reviews available strategies:
  1. Fresh re-read
  2. Compound-primitive check
  3. Cross-domain analogy
  4. Tier escalation
- Selects strategy 1 (fresh re-read) — first time stuck, simplest first
- Re-reads pandas_mini source files with explicit "what computational primitives underlie the cognitive ones?" lens
- Discovers:
  - LoadCSV at Tier 2 decomposes into Tier 1: PARSE_CSV (text → typed columns) + ALLOCATE (in-memory representation)
  - GroupBy at Tier 2 decomposes into Tier 1: HASH_PARTITION (assign rows to groups via hashing the by-column)
  - AggSum at Tier 2 decomposes into Tier 1: REDUCE (the actual aggregation primitive)
- Updates `primitives_catalog.json`: adds 3 Tier 1 primitives (PARSE_CSV, HASH_PARTITION, REDUCE)
- Writes `recovery_log.md`:
  ```
  Attempt 1 — Strategy: fresh re-read
  Trigger: Tier 1 absence flagged by T-01.1 attempt 2 PHASE_QA
  Action: Re-read pandas_mini source with "what's the computation under the cognition?" lens
  Discovered: PARSE_CSV (under LoadCSV), HASH_PARTITION (under GroupBy), REDUCE (under all Aggs)
  Updated: primitives_catalog.json — added 3 Tier 1 entries; updated tier_compression.md with Tier 1→2 ratio (3:6 = 1:2 = 2× compression at this transition)
  Outcome: T-01.1's acceptance command should now pass. Re-verifying.
  ```
- Re-runs T-01.1's acceptance command: passes ✓
- Returns completion report

PHASE_QA for T-01.1.1:
- Verifies recovery_log.md exists with one strategy named ✓
- Verifies primitives_catalog.json updated (timestamp newer than recovery start) ✓
- Re-runs T-01.1's acceptance command: ✓
- Verdict: **TASK_PASS** for T-01.1.1

QUEEN: T-01.1.1 PASS implies T-01.1 is now ready for re-QA. QUEEN spawns PHASE_QA for T-01.1 against the updated outputs:
- Tier 1 now non-empty ✓
- All 5 primitive types considered (UNIVERSAL has 6, STRUCTURAL has 0 with rationale, BRIDGE/GOAL/PHILOSOPHICAL all 0 with rationale)
- Compression measurable across Tier 1→2 transition ✓
- Verdict: **TASK_PASS**

---

## Bookkeeping after recovery

`workspace_manifest.json` T-01.1 entry:
```json
{
  "task_id": "T-01.1",
  "status": "pass",
  "attempts": 2,
  "recovery_attempts": 1,
  "recovery_log_path": "STEP_01/recovery_log.md",
  "outputs": [
    {"path": "STEP_01/primitives_catalog.json", "sha256": "..."},
    {"path": "STEP_01/tier_compression.md", "sha256": "..."},
    {"path": "STEP_01/deconstruction_notes.md", "sha256": "..."},
    {"path": "STEP_01/recovery_log.md", "sha256": "..."}
  ],
  "qa_verdict_log": [
    "TASK_FAIL_RETRY at T+0:10: STRUCTURAL primitive class empty without justification",
    "TASK_FAIL_RETRY at T+0:25: Tier 1 (COMPUTATIONAL) empty",
    "Recovery triggered at T+0:30; T-01.1.1 spawned",
    "T-01.1.1 TASK_PASS at T+0:42 (strategy: fresh re-read)",
    "T-01.1 re-QA: TASK_PASS at T+0:45"
  ]
}
```

T-01.1.1 entry:
```json
{
  "task_id": "T-01.1.1",
  "status": "pass",
  "attempts": 1,
  "strategy_applied": "fresh re-read",
  "outputs": [
    {"path": "STEP_01/recovery_log.md", "sha256": "..."},
    {"path": "STEP_01/primitives_catalog.json", "sha256": "..." }
  ]
}
```

INPROGRESS.md gets prepended:
```
## 2026-04-19 T+0:45 — LANG_DEV_V2 — T-01.1 RECOVERED via T-01.1.1

After 2 PHASE_QA failures on T-01.1 (STRUCTURAL absence; Tier 1 absence), QUEEN spawned
T-01.1.1 RECOVER at T+0:30. Strategy "fresh re-read" applied. Discovered 3 Tier 1
primitives (PARSE_CSV, HASH_PARTITION, REDUCE). T-01.1 re-QA passed. Phase 1 advances to T-01.2.
```

QUEEN advances to T-01.2 normally.

---

## What if recovery itself fails

If T-01.1.1 attempt 1 (strategy 1) does not unstick T-01.1:
- QUEEN spawns T-01.1.1 attempt 2 with strategy 2 (compound-primitive check)
- If still failing: attempt 3 with strategy 3 (cross-domain analogy)
- If still failing after 3 strategies: `TASK_FAIL_ESCALATE` for T-01.1; PHASE_HOLD on Phase 1

Each strategy is tried only once. The 4th strategy (tier escalation) is a "last resort" — used only if strategies 1-3 don't apply (e.g., this is genuinely a Tier 2 problem and the answer is to escalate to Tier 3).

Per `RECOVERY_MODEL.md` §3 — `recovery_max_attempts: 3` is a hard limit.

---

## Why this works

The recovery mechanism is shaped exactly as `STEP 1.1 - RECOVER OPS.md` prescribes:
- One strategy per spawn (avoids combinatorial chaos)
- Strategies in priority order (cheap first)
- Pattern completion drives discovery (find what's missing → look for what generates it)
- Honest escalation if all strategies exhausted

v1 had no notion of recovery — T-01.1 (its STEP_01) would simply ESCALATE after 2 failed attempts. Recovery via STEP 1.1 was orphaned in v1's flat phase model.

v2 surfaces recovery as a structured on-demand spawn with its own contract, log, and verdict trail. The `recovery_log.md` artifact preserves the recovery story for METHODOLOGY_INTEGRATOR and the human reviewer.

---

## Lesson

Recovery is not exceptional — it is anticipated. The methodology assumes deconstruction will get stuck. The workflow encodes that assumption rather than punishing it. T-01.1.1 spawning is a normal occurrence on real targets.

A LANG_DEV_V2 run with zero recovery spawns on a real library (NOT the reference) is mildly suspicious — likely the deconstruction was too cursory. A run with 1-2 recovery spawns is healthy.

---

*End of RECOVERY_WALKTHROUGH.*
