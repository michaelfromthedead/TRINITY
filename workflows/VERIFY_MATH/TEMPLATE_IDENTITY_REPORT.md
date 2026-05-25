# Identity Verification Report

**Generated:** {timestamp}
**GNOSTICA Version:** {version}
**Workflow:** VERIFY_MATH_WORKFLOW Phase 2 (IDENTITY_VERIFIER)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Identities** | {total} |
| **Passed** | {passed} |
| **Failed** | {failed} |
| **Partial** | {partial} |
| **Errors** | {errors} |
| **Skipped** | {skipped} |
| **Pass Rate** | {pass_rate}% |
| **Critical Pass Rate** | {critical_rate}% |
| **Verdict** | {PASS / WARN / FAIL} |

---

## Results by Category

| Category | Total | Passed | Failed | Partial | Pass Rate |
|----------|-------|--------|--------|---------|-----------|
| Derivative | {n} | {p} | {f} | {pt} | {r}% |
| Integral | {n} | {p} | {f} | {pt} | {r}% |
| Algebraic | {n} | {p} | {f} | {pt} | {r}% |
| Trigonometric | {n} | {p} | {f} | {pt} | {r}% |
| Limit | {n} | {p} | {f} | {pt} | {r}% |
| Series | {n} | {p} | {f} | {pt} | {r}% |
| Matrix | {n} | {p} | {f} | {pt} | {r}% |
| Number Theory | {n} | {p} | {f} | {pt} | {r}% |
| **Total** | **{total}** | **{passed}** | **{failed}** | **{partial}** | **{rate}%** |

---

## Critical Identities

These are fundamental identities that MUST pass.

| ID | Name | Category | Status |
|----|------|----------|--------|
| DERIV_001 | Power Rule | Derivative | {PASS/FAIL} |
| DERIV_002 | Product Rule | Derivative | {PASS/FAIL} |
| DERIV_003 | Chain Rule | Derivative | {PASS/FAIL} |
| DERIV_004 | Quotient Rule | Derivative | {PASS/FAIL} |
| INTEG_001 | Power Integral | Integral | {PASS/FAIL} |
| INTEG_002 | Linearity | Integral | {PASS/FAIL} |
| INTEG_003 | Fundamental Theorem | Integral | {PASS/FAIL} |
| ALG_001 | Distributive Law | Algebraic | {PASS/FAIL} |
| ALG_002 | Difference of Squares | Algebraic | {PASS/FAIL} |
| ALG_003 | Binomial Square | Algebraic | {PASS/FAIL} |
| TRIG_001 | Pythagorean Identity | Trigonometric | {PASS/FAIL} |
| TRIG_002 | Double Angle Sine | Trigonometric | {PASS/FAIL} |
| TRIG_003 | Double Angle Cosine | Trigonometric | {PASS/FAIL} |

**Critical Identity Pass Rate:** {critical_passed}/{critical_total} ({critical_rate}%)

{IF any critical fails}
**⚠️ CRITICAL FAILURE: Fundamental mathematical laws violated. Workflow FAIL.**
{ENDIF}

---

## Failures

{IF failures > 0}

### Failure 1: {identity_name} ({identity_id})

- **Category:** {category}
- **LHS:** `{lhs_expression}`
- **RHS:** `{rhs_expression}`
- **Symbolic Result:** `{symbolic_result}` (expected: 0)
- **Numeric Tests:**
  | Values | LHS Result | RHS Result | Difference |
  |--------|------------|------------|------------|
  | {values} | {lhs_val} | {rhs_val} | {diff} |
- **Severity:** {CRITICAL / HIGH / MEDIUM / LOW}
- **Analysis:** {diagnosis}

### Failure 2: {identity_name} ({identity_id})
...

{ELSE}

**No failures detected.**

{ENDIF}

---

## Partial Passes

Identities that pass symbolically but fail some numeric tests.

{IF partial > 0}

| ID | Name | Symbolic | Numeric Pass | Numeric Fail | Issue |
|----|------|----------|--------------|--------------|-------|
| {id} | {name} | PASS | {n_pass} | {n_fail} | {issue} |

{ELSE}

**No partial passes.**

{ENDIF}

---

## Evaluation Errors

Identities that could not be evaluated (missing rules, syntax issues).

{IF errors > 0}

| ID | Name | Error Message |
|----|------|---------------|
| {id} | {name} | `{error}` |

{ELSE}

**No evaluation errors.**

{ENDIF}

---

## Skipped Identities

Identities skipped due to conditions or scope.

{IF skipped > 0}

| ID | Name | Reason |
|----|------|--------|
| {id} | {name} | {reason} |

{ELSE}

**No skipped identities.**

{ENDIF}

---

## Detailed Results by Category

### Derivative Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| DERIV_001 | Power Rule | {status} | |
| DERIV_002 | Product Rule | {status} | |
| DERIV_003 | Chain Rule | {status} | |
| DERIV_004 | Quotient Rule | {status} | |
| DERIV_005 | Constant Rule | {status} | |
| DERIV_006 | Sum Rule | {status} | |
| DERIV_007 | Exponential | {status} | |
| DERIV_008 | Logarithm | {status} | |
| DERIV_009 | Sin | {status} | |
| DERIV_010 | Cos | {status} | |
| ... | ... | ... | ... |

### Integral Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| INTEG_001 | Power Integral | {status} | |
| INTEG_002 | Linearity | {status} | |
| ... | ... | ... | ... |

### Algebraic Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| ALG_001 | Distributive | {status} | |
| ALG_002 | Difference of Squares | {status} | |
| ... | ... | ... | ... |

### Trigonometric Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| TRIG_001 | Pythagorean | {status} | |
| TRIG_002 | Double Angle Sin | {status} | |
| ... | ... | ... | ... |

### Limit Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| LIM_001 | Constant Limit | {status} | |
| LIM_002 | Sum Limit | {status} | |
| ... | ... | ... | ... |

### Series Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| SER_001 | Geometric Sum | {status} | |
| SER_002 | Arithmetic Sum | {status} | |
| ... | ... | ... | ... |

### Matrix Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| MAT_001 | Transpose of Transpose | {status} | |
| MAT_002 | Determinant Product | {status} | |
| ... | ... | ... | ... |

### Number Theory Identities

| ID | Name | Status | Notes |
|----|------|--------|-------|
| NT_001 | GCD Commutativity | {status} | |
| NT_002 | Euclidean Algorithm | {status} | |
| ... | ... | ... | ... |

---

## Verdict

### {PASS / WARN / FAIL}

{IF PASS}
All identities verified. Mathematical correctness confirmed for tested identities.
{ENDIF}

{IF WARN}
{failed} identities failed ({fail_rate}%), but no critical failures.
Review failures before release.
{ENDIF}

{IF FAIL}
{critical_fails} critical identities failed. Mathematical correctness NOT verified.
**WORKFLOW ABORTED.** Fix critical identity failures before proceeding.
{ENDIF}

---

## Recommendations

{IF failures > 0}
1. **Critical:** Fix {n} critical identity failures immediately
2. **High:** Review {n} high-severity failures
3. **Medium:** Track {n} medium failures for next release
{ELSE}
No immediate actions required. Identity verification passed.
{ENDIF}

---

*Report generated by IDENTITY_VERIFIER worker*
*VERIFY_MATH_WORKFLOW Phase 2*
