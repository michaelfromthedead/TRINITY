# SHUFFLE_TEST_SPEC — Methodology-Level Acid Test

**Purpose:** Specify how METHODOLOGY_INTEGRATOR runs the shuffle test (Gate 1).

**Authoritative source:** `LANGS_DEV_RDC/CLARIFICATION.md` §6 ("The acid test: shuffle invariance"), `LANGS_DEV_RDC/PROJECT.md` §Success criteria #1.

---

## 1. What the shuffle test verifies

That the produced DSL — running end-to-end — separates SPECIFICATION (unordered, user-facing) from EXECUTION (ordered, solver-determined).

```
solve(bag_original) == solve(bag_shuffled_100_times)
```

If shuffling the input bag produces a different execution plan, input order leaked into output. That is a methodology failure.

**v1's deficit:** v1 had no methodology-level shuffle test. The Solver-level shuffle test (T-03.5) only verifies the Solver in isolation, with hand-built ASTs. The methodology-level test runs the full pipeline (Lexer → ... → Solver) on real DSL source strings.

---

## 2. Diversity criterion: 10+ bags

The bags must cover the diversity of constructions the produced DSL supports. Minimum 10, ideally 12-15.

### Required bag categories

| # | Category | Example (for pandas_mini reference target) |
|---|---|---|
| 1 | Pure sequence | `LoadCSV('x.csv'), Filter('a > 0'), Head(5)` |
| 2 | Single-atom bag | `LoadCSV('x.csv')` |
| 3 | Sequence with explicit `→` | `LoadCSV('x.csv') → Compute('z', 'a+b') → Filter('z > 10')` |
| 4 | Compound (type-forced pair) | `GroupBy('cat'), Agg('sum')` (must stay adjacent) |
| 5 | Alternative (`A | B`) | `LoadCSV('x.csv') | LoadParquet('x.parquet')` |
| 6 | Optional (`A?`) | `LoadCSV('x.csv'), Filter('a > 0')?, Head(5)` |
| 7 | Mixed: sequence + compound | `LoadCSV('x.csv'), GroupBy('cat'), Agg('mean'), Sort('mean'), Head(3)` |
| 8 | Column dependency chain | `LoadCSV('x.csv'), Compute('z', 'a+b'), Filter('z > 10')` (Filter depends on z, must come after Compute) |
| 9 | Mixed alternative + sequence | `(LoadCSV('x.csv') | LoadParquet('x.parquet')), Filter('a > 0')` |
| 10 | Fully bag (no explicit operators) | `LoadCSV('x.csv'), Filter('a > 0'), Sort('a'), Head(5)` (defaults must order correctly) |
| 11 | Implicit compound by type | `GroupBy('cat'), Sum()` (alternative form of #4) |
| 12 | Contains optional that gets excluded | `LoadCSV('x.csv'), Cache?, Filter('a > 0')` (default: exclude Cache; Filter chains directly) |

Per category, ONE bag is sufficient (12 bags total). Or multiple per category for stress (15-20 total).

### Bag construction rules

- All atoms must be valid in `<library>_decisions.json`
- All column references must be valid for `sample_data/small.csv`
- All argument types must satisfy atom signatures
- Bags must compile without errors (lexer/validator/parser/typer/classifier all PASS)

---

## 3. Iteration procedure

For each bag:

```python
plan_original = pipeline(bag_str)  # full Lexer→...→Solver

for i in range(100):
    rng = random.Random(i)         # fixed seed → reproducible
    atoms_list = parse_to_atoms(bag_str)
    rng.shuffle(atoms_list)
    shuffled_str = atoms_to_str(atoms_list)
    plan_shuffled = pipeline(shuffled_str)
    if plan_shuffled.order != plan_original.order:
        return Failure(bag_id=..., iteration=i, original=plan_original.order, shuffled=plan_shuffled.order)
```

**Note on shuffling:** comma-separated atoms in the bag are shuffled. Atoms inside grouped scopes `(...)` or compound chains `A-B` retain their positions within the group. Sequence chains `A→B` retain their order. Only top-level bag entries shuffle.

**Determinism of shuffle:** seed = iteration index. Reproducible across runs.

---

## 4. PASS / FAIL criteria

**PASS:**
- All 12+ bags
- All 100 iterations per bag
- Plan order match in every case

**FAIL:**
- ANY single iteration where `plan_shuffled.order != plan_original.order`
- Record: bag_id, iteration index, seed, both plan orders, diff

There is no partial pass. Shuffle invariance is binary.

---

## 5. Output: `SHUFFLE_TEST_RESULTS.json`

```json
{
  "gate": "shuffle_test",
  "spec_version": "1.0.0",
  "ran_at": "ISO 8601",
  "bags_tested": 12,
  "iterations_per_bag": 100,
  "total_iterations": 1200,
  "pass_count": 12,
  "fail_count": 0,
  "verdict": "PASS",
  "per_bag_results": [
    {
      "bag_id": "category_1_pure_sequence",
      "bag_str": "LoadCSV('x.csv'), Filter('a > 0'), Head(5)",
      "original_plan_order": ["LoadCSV", "Filter", "Head"],
      "iterations_passed": 100,
      "iterations_failed": 0,
      "first_failure": null
    }
  ]
}
```

**On FAIL:**
- `verdict: "FAIL"`
- `first_failure` populated for the failing bag with: `{iteration, seed, shuffled_str, shuffled_plan_order, diff}`
- Subsequent bags still tested (full report; not stop-at-first)

---

## 6. Common failure modes (diagnosed in METHODOLOGY_INTEGRATOR's findings)

| Symptom | Likely root cause | Likely Phase to fix |
|---|---|---|
| Same bag, different orders across iterations | Solver's priority key has unresolved ties (e.g., missing `name` or `args` final tiebreaker) | Phase 3 T-03.5 (Solver) |
| Atom appears in different positions due to dict iteration | Hash-based iteration somewhere (Python `set` or `dict` enumeration without sorting) | Phase 3 T-03.4 or T-03.5 |
| Optional atom included in some iterations, excluded in others | Optional resolution depends on input order | Phase 3 T-03.5 |
| Compound atoms not adjacent | Super-node collapse not applied | Phase 3 T-03.5 |
| Column dependency missed | Classifier's column extraction failed on some bag formulations | Phase 3 T-03.4 |
| Different alternative chosen across iterations | Alternative resolution depends on input position | Phase 3 T-03.5 (alternative selection strategy) |

---

## 7. Edge cases

**What if bag has only 1 atom?** Shuffle is a no-op; iteration always produces same order. PASS trivially. (Still include category #2 for sanity.)

**What if shuffle accidentally produces a re-parse that's invalid?** (E.g., shuffle moves an Agg before its GroupBy, breaking the compound.) The shuffle should preserve validity in the unordered-bag sense. If shuffled bag fails to parse/type-check: that's a TEST construction error, not a Solver failure. The bag construction (§2) must produce bags whose every shuffle is parsable. (For type-forced compounds, atoms inside `-` stay glued.)

**What if the produced DSL doesn't support some atoms in the bag?** Bag construction (§2) verifies all atoms valid in `<library>_decisions.json` before testing.

---

## 8. Relationship to T-03.5's Solver-level shuffle test

| Test | What | Where | Run by |
|---|---|---|---|
| T-03.5 Solver-level | `solve(ast) == solve(shuffled_ast)` on hand-built ASTs | `workspace_dir/STEP_11/test_solver.py` | PHASE_QA for T-03.5 |
| METHODOLOGY_INTEGRATOR Gate 1 | `pipeline(bag_str) == pipeline(shuffle(bag_str))` on real DSL source | METHODOLOGY_INTEGRATOR | METHODOLOGY_INTEGRATOR |

The Solver-level test catches Solver bugs in isolation. The methodology-level test catches integration bugs (e.g., Classifier orders ASTs differently when CST input order differs; Parser builds different CSTs for shuffled token streams).

Both must pass. T-03.5's test is necessary but not sufficient for METHODOLOGY_GREEN_LIGHT.

---

*End of SHUFFLE_TEST_SPEC.*
