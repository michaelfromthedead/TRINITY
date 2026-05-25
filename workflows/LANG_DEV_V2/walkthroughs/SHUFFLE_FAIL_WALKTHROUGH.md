# SHUFFLE_FAIL_WALKTHROUGH — Methodology-Level Shuffle Test FAIL

**Scenario:** All 4 phases pass per-task QA (including the Solver-level shuffle test in T-03.5), but METHODOLOGY_INTEGRATOR's Gate 1 (full-pipeline shuffle test) fails on one specific bag.

**Purpose:** Demonstrate that per-task QA is necessary but NOT sufficient for METHODOLOGY_GREEN_LIGHT — the methodology-level integration test catches cross-stage bugs invisible to per-stage QA.

---

## Setup

Same engagement as HAPPY_PATH_WALKTHROUGH. Phases 1-4 all reach PHASE_GREEN_LIGHT. T-03.5's Solver-level shuffle test passed (200 hand-built ASTs, 100 iterations each, all PASS).

---

## METHODOLOGY_INTEGRATOR Gate 1 (T+4:30)

QUEEN spawns METHODOLOGY_INTEGRATOR.

INTEGRATOR loads 12 diverse bags from `spec/SHUFFLE_TEST_SPEC.md` §2:

```
bag_1:  LoadCSV('x.csv'), Filter('a > 0'), Head(5)
bag_2:  LoadCSV('x.csv')
bag_3:  LoadCSV('x.csv') → Compute('z', 'a+b') → Filter('z > 10')
bag_4:  GroupBy('cat'), Agg('sum')
bag_5:  LoadCSV('x.csv') | LoadParquet('x.parquet')
bag_6:  LoadCSV('x.csv'), Filter('a > 0')?, Head(5)
bag_7:  LoadCSV('x.csv'), GroupBy('cat'), Agg('mean'), Sort('mean'), Head(3)
bag_8:  LoadCSV('x.csv'), Compute('z', 'a+b'), Filter('z > 10')
bag_9:  (LoadCSV('x.csv') | LoadParquet('x.parquet')), Filter('a > 0')
bag_10: LoadCSV('x.csv'), Filter('a > 0'), Sort('a'), Head(5)
bag_11: GroupBy('cat'), Sum()
bag_12: LoadCSV('x.csv'), Cache?, Filter('a > 0')
```

For each bag, run pipeline 100 times with shuffle.

---

## Bug surfaces on bag_8

bag_8 = `LoadCSV('x.csv'), Compute('z', 'a+b'), Filter('z > 10')`

Original parse + classify + solve:
```
plan_original.order = ['LoadCSV', 'Compute', 'Filter']
explanations: 'Filter must come after Compute (column dependency: z)'
```

Shuffle iteration #37 (seed=37):
```
shuffled_str = "Filter('z > 10'), LoadCSV('x.csv'), Compute('z', 'a+b')"
shuffled_plan.order = ['LoadCSV', 'Filter', 'Compute']  ← WRONG
```

The Solver placed Filter BEFORE Compute, even though Filter depends on column `z` which Compute creates.

---

## Diagnosis

INTEGRATOR records the failure:

```json
{
  "bag_id": "bag_8",
  "iterations_passed": 36,
  "iterations_failed": 64,
  "first_failure": {
    "iteration": 37,
    "seed": 37,
    "shuffled_str": "Filter('z > 10'), LoadCSV('x.csv'), Compute('z', 'a+b')",
    "original_plan_order": ["LoadCSV", "Compute", "Filter"],
    "shuffled_plan_order": ["LoadCSV", "Filter", "Compute"],
    "diff": "Filter swapped with Compute"
  }
}
```

**Why per-stage QA didn't catch this:**

T-03.5's Solver-level shuffle test (run during PHASE_QA for T-03.5) used **hand-built ASTs**. The hand-built AST for this scenario probably had the column dependency edge already encoded in the AST (because the test author wrote it that way).

The methodology-level test exposes this: when the bag is shuffled at the SOURCE STRING level, the Lexer → Validator → Parser → Typer → **Classifier** pipeline runs from scratch. The Classifier (T-03.4) is what extracts the column dependency from the parameter strings.

The bug is in the Classifier:
- The Classifier's `column_analysis.py` extracts column references from atom args
- For `Compute('z', 'a+b')` it correctly identifies: produces `z`, consumes `a, b`
- For `Filter('z > 10')` it correctly identifies: consumes `z`
- BUT: the `dependency_detection.py` walks atoms in **input order** to find producer-consumer pairs. When Filter appears BEFORE Compute in the bag, the dependency detector doesn't find the producer for `z`.

The dependency detector should be order-independent: scan all atoms, build a producer-map (column → producing atom), then for each consumer, look up its consumed columns in the producer-map.

---

## INTEGRATOR's verdict

```markdown
# METHODOLOGY_REPORT (Attempt 1)

## Verdict: METHODOLOGY_INCOMPLETE

## Phase summaries
- All 4 phases: PHASE_GREEN_LIGHT

## Gate 1 — Methodology-level shuffle test
Bags: 12; iterations per bag: 100
Pass: 11/12 bags
**Fail: 1/12 bags (bag_8)**
Status: FAIL

## Findings
### Finding 1 — Column dependency detection is order-dependent
Gate: 1
What went wrong: bag_8 (Compute → Filter on column 'z') produces different plan orders depending on input bag order. Classifier's dependency detector misses the dependency when Filter appears in the bag BEFORE Compute. Solver then places them by phase + alphabetical only, breaking semantic correctness.
Likely source: workspace/STEP_10/column_analysis.py — `detect_column_dependencies` function walks atoms in input order rather than building a global producer-map first.
Remediation: Re-engage T-03.4 with directive: "Column dependency detection must be order-independent. Build producer_map = {column: producing_atom} from a full pass over all atoms, then scan consumers."

## Gate 2 — Compression
PASS (1.83:1 above 1.5 threshold for reference target)

## Gate 3 — E2E demo
PASS (canonical input ran correctly because canonical input happens to be in dependency-respecting order; the bug doesn't surface)
```

---

## QUEEN's response

`workspace_manifest.json` updated:
```json
"methodology_integration": {
  "status": "incomplete",
  "first_attempt": {
    "ran_at": "T+4:35",
    "shuffle_test": {"pass": 11, "fail": 1},
    "compression": "PASS",
    "e2e_demo": "PASS"
  }
}
```

QUEEN re-spawns METHODOLOGY_INTEGRATOR (retry 1) with `prior_findings = [Finding 1]`. INTEGRATOR re-runs all 3 gates against the SAME workspace (no fix applied yet between attempts within INTEGRATOR's retry).

Result: Gate 1 fails again (same bug). METHODOLOGY_INTEGRATOR retry exhausted.

QUEEN emits workflow verdict `ESCALATED`. Reports to human:

```
LANG_DEV_V2 ESCALATED at T+4:55.

Phases 1-4 all PHASE_GREEN_LIGHT.
METHODOLOGY_INTEGRATOR: Gate 1 FAIL on both attempts.

Specific failure: bag_8 (LoadCSV, Compute('z', 'a+b'), Filter('z > 10')) produces different
execution plans depending on input bag order. Column dependency detection in Classifier
(STEP_10/column_analysis.py) is order-dependent.

To resume:
1. Edit workspace_dir/STEP_10/column_analysis.py per finding remediation
2. Re-run T-03.4's acceptance command (workspace_dir/STEP_10/test_classifier.py)
   - You may need to add a regression test for the order-independence property
3. Re-engage METHODOLOGY_INTEGRATOR via 'continue methodology integration'

Workspace preserved at /tmp/lang_dev_v2_run_2026-04-19/.
SHUFFLE_TEST_RESULTS.json contains per-iteration failure traces.
```

---

## What this scenario teaches

**1. Solver-level tests don't substitute for methodology-level tests.**
T-03.5's shuffle test uses hand-built ASTs — the test author can inadvertently encode dependencies in the AST that the Classifier should have derived. Methodology-level testing exercises the Classifier's derivation logic on real source strings.

**2. Per-stage QA is necessary but not sufficient.**
Each Phase 3 stage's QA passed against its contract. The contracts test each stage's responsibilities in isolation. Cross-stage bugs (Classifier doesn't propagate enough info to Solver, OR Solver assumes properties Classifier doesn't guarantee) show up only at integration.

**3. Compression check and e2e demo independently passed.**
Gate 2 doesn't depend on dependency correctness — it counts primitives. Gate 3 happened to use a dependency-respecting input string. Only Gate 1's adversarial shuffling exposed the bug.

**4. v1 would have emitted CLEAN_RUN here.**
v1's only verdict was per-phase. With all phases PASS, v1 says "done." The user would discover the bug only when invoking the produced DSL with a non-canonical input order in production.

**5. Findings are actionable.**
The methodology-level finding cites the specific file (`column_analysis.py`), the specific function, the specific bug pattern (order-dependent walk), and the specific fix (build producer-map first). PHASE_QA's per-task verdicts couldn't have been this specific because they didn't have the integration trace.

---

## After human fix

Human applies fix per finding. Re-engages workflow:

```
User: continue methodology integration
QUEEN: [re-spawns METHODOLOGY_INTEGRATOR; this is NOT a retry — it's a fresh attempt after human intervention]
INTEGRATOR:
  Gate 1: 12/12 bags PASS, 1200/1200 iterations PASS (the fix worked)
  Gate 2: PASS
  Gate 3: PASS
  Verdict: METHODOLOGY_GREEN_LIGHT
QUEEN: emits CLEAN_RUN; closes workspace_manifest.json
```

Bookkeeping notes the human intervention with timestamp + fix description.

---

*End of SHUFFLE_FAIL_WALKTHROUGH.*
