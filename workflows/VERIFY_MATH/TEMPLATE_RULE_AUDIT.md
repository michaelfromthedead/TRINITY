# Rule Audit Report

**Generated:** {timestamp}
**GNOSTICA Version:** {version}
**Workflow:** VERIFY_MATH_WORKFLOW Phase 2 (RULE_AUDITOR)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Rule Files** | {file_count} |
| **Total Rules** | {rule_count} |
| **Critical Issues** | {critical} |
| **High Issues** | {high} |
| **Medium Issues** | {medium} |
| **Low Issues** | {low} |
| **Quality Score** | {score}/100 |
| **Quality Grade** | {EXCELLENT / GOOD / ACCEPTABLE / NEEDS_IMPROVEMENT / POOR} |
| **Verdict** | {PASS / WARN / FAIL} |

---

## Rules by Tier

| Tier | Count | Percentage |
|------|-------|------------|
| Tier 1 (Basic) | {n} | {p}% |
| Tier 2 (Intermediate) | {n} | {p}% |
| Tier 3 (Advanced) | {n} | {p}% |
| Unspecified | {n} | {p}% |
| **Total** | **{total}** | **100%** |

---

## Rules by Category

| Category | Files | Rules | Percentage |
|----------|-------|-------|------------|
| Derivative | {f} | {n} | {p}% |
| Integral | {f} | {n} | {p}% |
| Polynomial/Algebra | {f} | {n} | {p}% |
| Trigonometric | {f} | {n} | {p}% |
| Matrix/Linear Algebra | {f} | {n} | {p}% |
| Limits/Analysis | {f} | {n} | {p}% |
| Series/Sequences | {f} | {n} | {p}% |
| Number Theory | {f} | {n} | {p}% |
| Core/Simplify | {f} | {n} | {p}% |
| Other | {f} | {n} | {p}% |
| **Total** | **{total_f}** | **{total_r}** | **100%** |

---

## Issues by Type

### Summary

| Issue Type | Count | Severity |
|------------|-------|----------|
| Syntax Errors | {n} | CRITICAL |
| Rule Conflicts | {n} | HIGH |
| Circular Dependencies | {n} | HIGH |
| Redundant Rules | {n} | MEDIUM |
| Tier Violations | {n} | MEDIUM |
| Pattern Shadowing | {n} | MEDIUM |
| Undefined References | {n} | MEDIUM |
| Missing Documentation | {n} | LOW |
| **Total** | **{total}** | |

---

## Critical Issues

{IF critical > 0}

### Syntax Errors

| File | Line | Issue | Rule Name |
|------|------|-------|-----------|
| `{file}` | {line} | {description} | {name} |

**Impact:** Rules with syntax errors will not parse. Engine may fail or skip these rules.

{ELSE}

**No critical issues detected.**

{ENDIF}

---

## High Issues

{IF high > 0}

### Rule Conflicts

Rules with same pattern but different results:

| Pattern | File 1 | Result 1 | File 2 | Result 2 |
|---------|--------|----------|--------|----------|
| `{pattern}` | `{file1}:{line1}` | `{result1}` | `{file2}:{line2}` | `{result2}` |

**Impact:** Conflicting rules produce inconsistent results. Actual output depends on load order.

### Circular Dependencies

| Rule A | Rule B | Cycle |
|--------|--------|-------|
| `{rule_a}` in `{file_a}` | `{rule_b}` in `{file_b}` | A → B → A |

**Impact:** Infinite loop during rule application.

{ELSE}

**No high issues detected.**

{ENDIF}

---

## Medium Issues

{IF medium > 0}

### Redundant Rules

Duplicate rules found across files:

| Pattern | Locations |
|---------|-----------|
| `{pattern}` | `{file1}:{line1}`, `{file2}:{line2}` |

### Tier Violations

Rules with incorrect tier assignment:

| Rule | Current Tier | Suggested Tier | Reason |
|------|--------------|----------------|--------|
| `{name}` in `{file}` | {current} | {suggested} | {reason} |

### Pattern Shadowing

General patterns that shadow specific ones:

| General Pattern | Specific Pattern | File |
|-----------------|------------------|------|
| `{general}` (line {g_line}) | `{specific}` (line {s_line}) | `{file}` |

### Undefined References

Rules referencing undefined functions:

| Rule | Undefined Reference | File |
|------|---------------------|------|
| `{name}` | `{undefined}` | `{file}:{line}` |

{ELSE}

**No medium issues detected.**

{ENDIF}

---

## Low Issues

{IF low > 0}

### Missing Documentation

Rules without @description:

| Rule Name | File | Line |
|-----------|------|------|
| `{name}` | `{file}` | {line} |

**Count:** {n} rules missing documentation ({percent}%)

{ELSE}

**No low issues detected.**

{ENDIF}

---

## File Analysis

### Largest Files (by rule count)

| Rank | File | Rules | Lines |
|------|------|-------|-------|
| 1 | `{file}` | {rules} | {lines} |
| 2 | `{file}` | {rules} | {lines} |
| 3 | `{file}` | {rules} | {lines} |
| 4 | `{file}` | {rules} | {lines} |
| 5 | `{file}` | {rules} | {lines} |

### Files with Most Issues

| Rank | File | Critical | High | Medium | Low | Total |
|------|------|----------|------|--------|-----|-------|
| 1 | `{file}` | {c} | {h} | {m} | {l} | {total} |
| 2 | `{file}` | {c} | {h} | {m} | {l} | {total} |
| 3 | `{file}` | {c} | {h} | {m} | {l} | {total} |

---

## Documentation Coverage

| Metric | Value |
|--------|-------|
| Rules with @name | {n} ({p}%) |
| Rules with @tier | {n} ({p}%) |
| Rules with @description | {n} ({p}%) |
| Fully documented | {n} ({p}%) |

---

## Pattern Complexity Analysis

| Complexity | Count | Example |
|------------|-------|---------|
| Simple (single term) | {n} | `Sin[0] := 0` |
| Moderate (2-3 terms) | {n} | `D[f + g, x] := D[f, x] + D[g, x]` |
| Complex (4+ terms) | {n} | Integration by parts |
| Conditional (/;) | {n} | `x^n /; n > 0` |

---

## Quality Score Breakdown

```
Starting Score: 100

Penalties:
- Critical issues: {c} × (-10) = {c_penalty}
- High issues: {h} × (-5) = {h_penalty}
- Medium issues: {m} × (-2) = {m_penalty}
- Low issues: {l} × (-1) = {l_penalty}

Total Penalties: {total_penalty}
Final Score: {score}
```

---

## Verdict

### {PASS / WARN / FAIL}

{IF PASS}
Rule audit passed. Quality score: {score}/100 ({grade}).
No critical or high issues found.
{ENDIF}

{IF WARN}
Rule audit passed with warnings. Quality score: {score}/100 ({grade}).
{high} high issues and {medium} medium issues require attention.
{ENDIF}

{IF FAIL}
Rule audit FAILED. Quality score: {score}/100 ({grade}).
{critical} critical issues must be fixed before proceeding.
{ENDIF}

---

## Recommendations

{IF issues > 0}

### Immediate (Critical)
{IF critical > 0}
1. Fix {critical} syntax errors — rules won't parse
{ENDIF}

### High Priority
{IF high > 0}
1. Resolve {conflicts} rule conflicts — causes inconsistent results
2. Break {circular} circular dependencies — causes infinite loops
{ENDIF}

### Medium Priority
{IF medium > 0}
1. Remove {redundant} redundant rules — maintenance burden
2. Fix {tier} tier violations — performance impact
3. Review {shadow} pattern shadowing — may cause unexpected behavior
{ENDIF}

### Low Priority
{IF low > 0}
1. Add documentation to {undoc} rules — improves maintainability
{ENDIF}

{ELSE}
No recommendations. Rule base is in excellent condition.
{ENDIF}

---

## Appendix: All Issues

{IF issues > 0}

### Full Issue List

| # | Severity | Type | File | Line | Description |
|---|----------|------|------|------|-------------|
| 1 | {sev} | {type} | `{file}` | {line} | {desc} |
| 2 | {sev} | {type} | `{file}` | {line} | {desc} |
| ... | ... | ... | ... | ... | ... |

{ENDIF}

---

*Report generated by RULE_AUDITOR worker*
*VERIFY_MATH_WORKFLOW Phase 2*
