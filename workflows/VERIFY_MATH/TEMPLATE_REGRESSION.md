# Regression Detection Report

**Generated:** {timestamp}
**GNOSTICA Version:** {version}
**Workflow:** VERIFY_MATH_WORKFLOW Phase 3 (REGRESSION_DETECTOR)
**Baseline Version:** {baseline_version}
**Baseline Date:** {baseline_date}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Baseline Exists** | {YES / NO} |
| **Test Regressions** | {count} |
| **Output Regressions** | {count} |
| **Performance Regressions** | {count} |
| **Rule Changes** | {count} |
| **New Tests** | {count} |
| **Removed Tests** | {count} |
| **Improvements** | {count} |
| **Verdict** | {PASS / WARN / FAIL} |

---

{IF no_baseline}

## First Run Notice

**No baseline found.** This is the first verification run.

- Regression detection skipped
- Current results will be used to generate initial baseline
- Baseline marked as UNVERIFIED until human review

**Action Required:** Review current results and run `--update-baseline` to establish baseline.

{ENDIF}

---

## Test Comparison

### Summary

| Category | Baseline | Current | Change |
|----------|----------|---------|--------|
| Total Tests | {b_total} | {c_total} | {diff} |
| Passed | {b_passed} | {c_passed} | {diff} |
| Failed | {b_failed} | {c_failed} | {diff} |
| Ignored | {b_ignored} | {c_ignored} | {diff} |

### Test Regressions

{IF test_regressions > 0}

Tests that passed in baseline but fail now:

| Test Name | Category | Baseline | Current | Severity |
|-----------|----------|----------|---------|----------|
| `{test_name}` | {category} | PASS | FAIL | {CRITICAL/HIGH/MEDIUM} |

#### Regression Details

##### Regression 1: {test_name}

- **File:** `{file_path}`
- **Category:** {category}
- **Baseline Result:** PASS
- **Current Result:** FAIL
- **Error:**
  ```
  {error_message}
  ```
- **Likely Cause:** {analysis}

{ELSE}

**No test regressions detected.** ✓

{ENDIF}

### New Tests

{IF new_tests > 0}

Tests in current run but not in baseline:

| Test Name | Category | Result |
|-----------|----------|--------|
| `{test_name}` | {category} | {PASS/FAIL} |

**Count:** {new_test_count} new tests added

{ELSE}

**No new tests since baseline.**

{ENDIF}

### Removed Tests

{IF removed_tests > 0}

Tests in baseline but not in current run:

| Test Name | Category | Baseline Result |
|-----------|----------|-----------------|
| `{test_name}` | {category} | {PASS/FAIL} |

**⚠️ Review Required:** Understand why tests were removed.

{ELSE}

**No tests removed since baseline.**

{ENDIF}

---

## Output Regressions

{IF output_regressions > 0}

Canonical expressions producing different results:

| ID | Input | Baseline Output | Current Output | Severity |
|----|-------|-----------------|----------------|----------|
| {id} | `{input}` | `{baseline}` | `{current}` | {sev} |

### Regression Analysis

#### Output Regression 1: {id}

- **Input:** `{input}`
- **Baseline:** `{baseline_output}`
- **Current:** `{current_output}`
- **Difference Type:** {semantic / cosmetic / structural}
- **Impact:** {assessment}

{ELSE}

**No output regressions detected.** All canonical expressions match. ✓

{ENDIF}

---

## Performance Comparison

### Summary

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Total Duration | {b_duration} | {c_duration} | {percent}% |
| Average per Test | {b_avg}ms | {c_avg}ms | {percent}% |
| 95th Percentile | {b_p95}ms | {c_p95}ms | {percent}% |

### Performance Regressions

{IF perf_regressions > 0}

Tests significantly slower than baseline (>50%):

| Test Name | Baseline | Current | Change | Severity |
|-----------|----------|---------|--------|----------|
| `{test}` | {b_time}ms | {c_time}ms | +{percent}% | {sev} |

{ELSE}

**No significant performance regressions.** ✓

{ENDIF}

### Performance Improvements

{IF perf_improvements > 0}

Tests significantly faster than baseline (>50%):

| Test Name | Baseline | Current | Change |
|-----------|----------|---------|--------|
| `{test}` | {b_time}ms | {c_time}ms | -{percent}% |

{ENDIF}

---

## Rule File Changes

{IF rule_changes > 0}

Rule files modified since baseline:

| File | Baseline Hash | Current Hash | Status |
|------|---------------|--------------|--------|
| `{file}` | `{b_hash}` | `{c_hash}` | MODIFIED |

### Changed Rules Summary

| File | Lines Changed | Rules Affected |
|------|---------------|----------------|
| `{file}` | +{added}/-{removed} | {count} |

**Note:** Rule changes may be intentional. Verify changes correlate with expected modifications.

{ELSE}

**No rule file changes since baseline.**

{ENDIF}

---

## Improvements

{IF improvements > 0}

### Test Improvements

Tests that failed in baseline but pass now:

| Test Name | Category | Baseline | Current |
|-----------|----------|----------|---------|
| `{test}` | {category} | FAIL | PASS |

**Count:** {improvement_count} tests fixed

{ELSE}

**No test improvements** (all baseline passes still pass).

{ENDIF}

---

## Identity Comparison

| Category | Baseline Pass | Current Pass | Change |
|----------|---------------|--------------|--------|
| Derivative | {b} | {c} | {diff} |
| Integral | {b} | {c} | {diff} |
| Algebraic | {b} | {c} | {diff} |
| Trigonometric | {b} | {c} | {diff} |
| Limit | {b} | {c} | {diff} |
| Series | {b} | {c} | {diff} |
| Matrix | {b} | {c} | {diff} |
| Number Theory | {b} | {c} | {diff} |
| **Total** | **{b_total}** | **{c_total}** | **{diff}** |

---

## Corpus Comparison

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Validated | {b} | {c} | {diff} |
| Passed | {b} | {c} | {diff} |
| Failed | {b} | {c} | {diff} |
| Pass Rate | {b}% | {c}% | {diff}% |

---

## Audit Score Comparison

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| Quality Score | {b}/100 | {c}/100 | {diff} |
| Critical Issues | {b} | {c} | {diff} |
| High Issues | {b} | {c} | {diff} |
| Medium Issues | {b} | {c} | {diff} |
| Low Issues | {b} | {c} | {diff} |

---

## Verdict

### {PASS / WARN / FAIL}

{IF PASS}
No regressions detected. All metrics stable or improved vs baseline.
{ENDIF}

{IF WARN}
{regression_count} regressions detected, but within acceptable thresholds.
Review regressions before release.
{ENDIF}

{IF FAIL}
{critical_regressions} critical regressions detected.
**WORKFLOW ABORTED.** Fix regressions before proceeding.

**Critical Regressions:**
{list of critical regressions}
{ENDIF}

---

## Baseline Update

{IF verdict == PASS or verdict == WARN}

### Ready for Baseline Update

Current verification is eligible for baseline update.

To update baseline:
```bash
VERIFY_MATH_WORKFLOW --update-baseline
```

This will:
1. Archive current baseline to `baseline_v{version}.json`
2. Write new baseline from current results
3. Log update in baseline history

**Warning:** Only update baseline after human review confirms all changes are intentional.

{ELSE}

### Baseline Update Blocked

Cannot update baseline with FAIL verdict. Fix regressions first.

{ENDIF}

---

## Recommendations

{IF regressions > 0}

### Investigate Regressions

1. **Test Regressions ({test_reg}):** Check recent rule changes
2. **Output Regressions ({out_reg}):** Verify output format changes
3. **Performance Regressions ({perf_reg}):** Profile slow tests

### Correlation Analysis

Check if rule changes correlate with regressions:

| Rule File | Changed | Related Regressions |
|-----------|---------|---------------------|
| `{file}` | YES | {count} |

{ELSE}

**No regressions to investigate.** Current state matches or improves on baseline.

{ENDIF}

---

*Report generated by REGRESSION_DETECTOR worker*
*VERIFY_MATH_WORKFLOW Phase 3*
