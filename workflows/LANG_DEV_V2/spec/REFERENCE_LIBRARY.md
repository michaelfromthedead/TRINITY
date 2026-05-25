# REFERENCE_LIBRARY — Integration Test Target

**Purpose:** Specify the small reference library used by METHODOLOGY_INTEGRATOR for Gate 1 (shuffle test) and Gate 3 (e2e demo).

**Decision:** **A small pandas-subset wrapper library**, packaged as `workflows/LANG_DEV_V2/test_target/pandas_mini/`.

---

## Why pandas-subset

1. **RDC examples reference pandas throughout.** `LANGS_DEV_RDC/MASTER.md` and the boss_level docs use pandas as the canonical worked example. The methodology was designed against pandas as its proof-of-concept.

2. **Small enough to ship in-repo.** A trimmed wrapper (LoadCSV, Compute, Filter, Sort, Head, Tail, GroupBy, Agg, Sink) is ~300 lines of Python. Full pandas is too large.

3. **Self-contained nexus reports.** We can author the 8 nexus reports for the trimmed surface in <1MB total, committed to the repo.

4. **Realistic verb diversity.** Covers: SOURCE atoms (LoadCSV), TRANSFORM atoms (Compute, Filter, Sort), GROUP+AGG pair (GroupBy + Agg → type-forced compound), LIMIT atoms (Head, Tail), SINK atoms.

**Rejected alternatives:**
- **Full pandas:** too large; nexus reports for 200+ public methods would dominate the repo.
- **Synthetic library with no real semantics:** would pass the methodology trivially without exercising the interesting failure modes.
- **Polars or duckdb:** smaller than pandas but not what the RDC corpus targets; would require new nexus reports and break methodology citations.
- **A non-data library (e.g., requests):** the methodology has not been validated against networking/IO domains; would conflate "library choice" issues with "methodology" issues.

---

## Layout (to be built in Part 7 T-7.1)

```
workflows/LANG_DEV_V2/test_target/
├── pandas_mini/                    ← target_library (~300 LOC Python)
│   ├── __init__.py                 ← public API surface
│   ├── load.py                     ← LoadCSV
│   ├── transform.py                ← Compute, Filter, Sort
│   ├── aggregate.py                ← GroupBy, Agg
│   ├── limit.py                    ← Head, Tail
│   ├── sink.py                     ← WriteCSV
│   └── tests/
│       └── test_smoke.py           ← validates pandas_mini itself works
│
├── pandas_mini_nexus/              ← nexus_reports_dir
│   ├── GRAVITY.md
│   ├── GRAMMAR.md
│   ├── VERBS.md
│   ├── TIERS.md
│   ├── CLASSIFICATION.md
│   ├── GENESIS.md
│   ├── GENERATOR.md
│   └── COMPRESSION.md
│
├── expected_workspace_manifest.json   ← golden output for happy-path validation
└── sample_data/
    └── small.csv                   ← input data for e2e demo
```

---

## API surface (the denominator for compression check)

The nexus reports document the following public-API operations on `pandas_mini`:

| Function | Module | Verb | Tier |
|---|---|---|---|
| `read_csv(path)` | load | SOURCE | Tier 2 (cognitive: load tabular data) |
| `to_csv(df, path)` | sink | SINK | Tier 2 |
| `df.assign(**kwargs)` | transform | TRANSFORM (compute) | Tier 2 |
| `df.query(expr)` | transform | TRANSFORM (filter) | Tier 2 |
| `df.sort_values(by)` | transform | TRANSFORM (sort) | Tier 2 |
| `df.head(n)` | limit | LIMIT | Tier 2 |
| `df.tail(n)` | limit | LIMIT | Tier 2 |
| `df.groupby(by)` | aggregate | GROUP | Tier 2 |
| `grouped.sum()` | aggregate | AGGREGATE | Tier 2 |
| `grouped.mean()` | aggregate | AGGREGATE | Tier 2 |
| `grouped.count()` | aggregate | AGGREGATE | Tier 2 |

**Public API count:** 11 (the denominator for COMPRESSION_REPORT.md ratio).

**Expected primitive count after T-01.1:** ~5-7 (LoadData, Transform, Filter, Sort, Group, Aggregate, Limit, Sink). Some methods collapse to the same primitive (sum/mean/count are all AGGREGATE variants).

**Expected compression ratio:** 11 ÷ 6 ≈ 1.8:1 — **WAIT.** This is far below the 18:1 target.

### Compression-target reconciliation

**Issue:** The ~20:1 target in `LANGS_DEV_RDC/PROJECT.md` §Success criteria assumes a full library with 100+ public methods (e.g., real pandas with `.merge`, `.join`, `.pivot`, `.melt`, `.rolling`, etc.). A trimmed reference library cannot hit ratios that high because the denominator is too small.

**Resolution options:**

**Option A (proposed):** Use a different threshold for the reference library — `≥ 1.5:1` for `pandas_mini`. The 18:1 threshold remains the production target for real libraries; the reference library only validates that the methodology RUNS end-to-end, not that the methodology achieves a particular ratio on every input.

**Option B:** Expand `pandas_mini` to ~60 public methods to test the 18:1 threshold realistically. Increases scaffold complexity significantly.

**Option C:** Skip Gate 2 for reference-library runs; gate it only on production-target runs. Risks regression undetected.

**Recommendation:** **Option A.** Update Gate 2 acceptance to be parameterized by the spec doc, not hardcoded:
- For `pandas_mini`: threshold ≥ 1.5:1 (sanity check that compression occurs at all)
- For real target libraries: threshold ≥ 18:1 (the methodology's production criterion)
- COMPRESSION_REPORT.md notes which threshold applied and why

This preserves the methodology's standard while allowing the reference library to validate workflow correctness.

**Action item:** Update `spec/COMPRESSION_SPEC.md` (next file) to reflect this parameterization.

---

## How METHODOLOGY_INTEGRATOR uses this library

1. Workflow is engaged with `target_library=workflows/LANG_DEV_V2/test_target/pandas_mini` + `nexus_reports_dir=workflows/LANG_DEV_V2/test_target/pandas_mini_nexus`
2. All 4 phases run against this small target
3. METHODOLOGY_INTEGRATOR uses `sample_data/small.csv` as input for Gate 3 e2e demo
4. Generated DSL processes the CSV via the produced Lexer → ... → Executor pipeline
5. Output verified against `expected_workspace_manifest.json` golden output

---

## Reference library is NOT the target audience

The methodology is designed for full-scale libraries (real pandas, requests, NumPy). The reference library exists ONLY to:
- Validate the workflow itself runs end-to-end
- Provide a reproducible CI/integration test target
- Give Gate 1 (shuffle test) and Gate 3 (e2e demo) something concrete to operate on

Production runs of LANG_DEV_V2 should target real libraries. The reference library run is "is the workflow working?", not "is the produced DSL useful?"

---

## Not committed yet

This file SPECIFIES the reference library. The actual files (Python code, nexus reports, sample data) are produced in Part 7 T-7.1.

---

*End of REFERENCE_LIBRARY.*
