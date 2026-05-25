# Corpus Validation Report

**Generated:** {timestamp}
**GNOSTICA Version:** {version}
**Workflow:** VERIFY_MATH_WORKFLOW Phase 2 (CORPUS_VALIDATOR)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Corpus Problems** | 989 |
| **Implemented (IMPLEMENT)** | 884 |
| **Out of Scope** | 100 |
| **Already Covered** | 5 |
| **Validated** | {validated} |
| **Passed** | {passed} |
| **Failed** | {failed} |
| **Pass Rate** | {pass_rate}% |
| **Coverage** | {coverage}% |
| **Verdict** | {PASS / WARN / FAIL} |

---

## Coverage by Era

| Era | Years | Total | Implemented | Validated | Passed | Pass Rate |
|-----|-------|-------|-------------|-----------|--------|-----------|
| Classic | 1938-1965 | {t} | {i} | {v} | {p} | {r}% |
| Transitional | 1966-1985 | {t} | {i} | {v} | {p} | {r}% |
| Modern | 1986-2010 | {t} | {i} | {v} | {p} | {r}% |
| Contemporary | 2011-2025 | {t} | {i} | {v} | {p} | {r}% |
| **Total** | **1938-2025** | **989** | **884** | **{v}** | **{p}** | **{r}%** |

---

## Results by Year

| Year | Problems | Implemented | Passed | Failed | Pass Rate |
|------|----------|-------------|--------|--------|-----------|
| 2025 | 12 | {i} | {p} | {f} | {r}% |
| 2024 | 12 | {i} | {p} | {f} | {r}% |
| 2023 | 12 | {i} | {p} | {f} | {r}% |
| ... | ... | ... | ... | ... | ... |
| 1940 | 12 | {i} | {p} | {f} | {r}% |
| 1939 | 12 | {i} | {p} | {f} | {r}% |
| 1938 | 12 | {i} | {p} | {f} | {r}% |

### Years with Failures

| Year | Failed | Problems |
|------|--------|----------|
| {year} | {n} | {problem_list} |

---

## Results by Domain

| Domain | Implemented | Passed | Failed | Pass Rate |
|--------|-------------|--------|--------|-----------|
| Algebra & Polynomials | {i} | {p} | {f} | {r}% |
| Calculus & Integration | {i} | {p} | {f} | {r}% |
| Series & Sequences | {i} | {p} | {f} | {r}% |
| Linear Algebra | {i} | {p} | {f} | {r}% |
| Number Theory | {i} | {p} | {f} | {r}% |
| Analysis & Limits | {i} | {p} | {f} | {r}% |
| Combinatorics | {i} | {p} | {f} | {r}% |
| Other | {i} | {p} | {f} | {r}% |
| **Total** | **884** | **{p}** | **{f}** | **{r}%** |

---

## Results by Difficulty

| Level | Description | Implemented | Passed | Failed | Pass Rate | Target |
|-------|-------------|-------------|--------|--------|-----------|--------|
| A1 | Accessible | {i} | {p} | {f} | {r}% | 98% |
| A2 | Moderate | {i} | {p} | {f} | {r}% | 95% |
| A3 | Challenging | {i} | {p} | {f} | {r}% | 90% |
| A4 | Hard | {i} | {p} | {f} | {r}% | 85% |
| A5 | Very Hard | {i} | {p} | {f} | {r}% | 80% |
| A6 | Extreme | {i} | {p} | {f} | {r}% | 75% |
| B1 | Accessible | {i} | {p} | {f} | {r}% | 98% |
| B2 | Moderate | {i} | {p} | {f} | {r}% | 95% |
| B3 | Challenging | {i} | {p} | {f} | {r}% | 90% |
| B4 | Hard | {i} | {p} | {f} | {r}% | 85% |
| B5 | Very Hard | {i} | {p} | {f} | {r}% | 80% |
| B6 | Extreme | {i} | {p} | {f} | {r}% | 75% |

### Levels Below Target

| Level | Pass Rate | Target | Gap |
|-------|-----------|--------|-----|
| {level} | {actual}% | {target}% | -{gap}% |

---

## Failures

{IF failures > 0}

### Failure Summary

| Type | Count | Percentage |
|------|-------|------------|
| Wrong Answer | {n} | {p}% |
| Incomplete | {n} | {p}% |
| Timeout | {n} | {p}% |
| Parse Error | {n} | {p}% |
| Runtime Error | {n} | {p}% |
| **Total** | **{total}** | **100%** |

### Detailed Failures

#### Failure 1: {problem_id}

- **Year:** {year}
- **Number:** {number} (e.g., A3)
- **Domain:** {domain}
- **Test File:** `{test_file}`
- **Test Name:** `{test_name}`
- **Failure Type:** {type}
- **Expected:**
  ```
  {expected}
  ```
- **Actual:**
  ```
  {actual}
  ```
- **Error Message:**
  ```
  {error}
  ```
- **Analysis:** {root_cause}
- **Suggested Fix:** {fix}

#### Failure 2: {problem_id}
...

{ELSE}

**No failures detected.** All 884 implemented problems pass validation.

{ENDIF}

---

## Timeouts

{IF timeouts > 0}

Problems exceeding 60-second timeout:

| Problem | Duration | Status |
|---------|----------|--------|
| {problem_id} | {duration}s | TIMEOUT |

{ELSE}

**No timeouts.** All problems completed within time limit.

{ENDIF}

---

## Missing Tests

{IF missing > 0}

Problems classified as IMPLEMENT but without corresponding test:

| Problem | Year | Classification | Issue |
|---------|------|----------------|-------|
| {id} | {year} | IMPLEMENT | No test file |

{ELSE}

**All implemented problems have tests.**

{ENDIF}

---

## Out of Scope Summary

100 problems classified as OUT_OF_SCOPE:

| Reason | Count |
|--------|-------|
| Pure Existence Proofs | 25 |
| Research-Level | 28 |
| Game Theory | 8 |
| Topology & Geometry | 18 |
| Dynamical Systems | 14 |
| Measure Theory | 7 |
| **Total** | **100** |

These are intentionally not implemented — outside CAS capabilities.

---

## Performance Analysis

### Slowest Problems (Top 10)

| Rank | Problem | Duration | Domain |
|------|---------|----------|--------|
| 1 | {id} | {duration}s | {domain} |
| 2 | {id} | {duration}s | {domain} |
| 3 | {id} | {duration}s | {domain} |
| 4 | {id} | {duration}s | {domain} |
| 5 | {id} | {duration}s | {domain} |
| 6 | {id} | {duration}s | {domain} |
| 7 | {id} | {duration}s | {domain} |
| 8 | {id} | {duration}s | {domain} |
| 9 | {id} | {duration}s | {domain} |
| 10 | {id} | {duration}s | {domain} |

### Timing Summary

| Metric | Value |
|--------|-------|
| Total Duration | {total} |
| Average per Problem | {avg}s |
| Median Duration | {median}s |
| 95th Percentile | {p95}s |

---

## Phase-by-Phase Results

| Phase | Problems | Passed | Failed | Pass Rate |
|-------|----------|--------|--------|-----------|
| Phase 1-14 (Foundation) | {n} | {p} | {f} | {r}% |
| Phase 15-23 (Enhancement) | {n} | {p} | {f} | {r}% |
| Phase 24-30 (MEGA-BLITZ) | {n} | {p} | {f} | {r}% |
| Phase 31-38 (Final) | {n} | {p} | {f} | {r}% |

---

## Verdict

### {PASS / WARN / FAIL}

{IF PASS}
All 884 implemented Putnam problems pass validation.
Corpus coverage: {coverage}%. Mathematical correctness verified.
{ENDIF}

{IF WARN}
{passed}/{validated} problems pass ({pass_rate}%).
{failed} failures detected, but pass rate above threshold (95%).
Review failures before release.
{ENDIF}

{IF FAIL}
{passed}/{validated} problems pass ({pass_rate}%).
Pass rate below threshold (95%). **WORKFLOW ABORTED.**
Fix critical failures before proceeding.
{ENDIF}

---

## Recommendations

{IF failures > 0}

### Immediate Actions

1. **Fix {critical_fails} critical failures** in {domains}
2. **Investigate {timeout_count} timeouts** for performance issues
3. **Review {wrong_answer_count} wrong answers** for rule correctness

### By Domain Priority

| Domain | Failures | Priority |
|--------|----------|----------|
| {domain} | {n} | HIGH |
| {domain} | {n} | MEDIUM |

### Specific Fixes

{list of specific fixes based on failure analysis}

{ELSE}

**No recommendations.** Corpus validation passed completely.

{ENDIF}

---

## Cross-Reference

| Check | Status |
|-------|--------|
| TEST_SWEEP alignment | {ALIGNED / MISMATCH} |
| Test count matches | {match_status} |
| No orphan tests | {orphan_status} |

---

*Report generated by CORPUS_VALIDATOR worker*
*VERIFY_MATH_WORKFLOW Phase 2*
