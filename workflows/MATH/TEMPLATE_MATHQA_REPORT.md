# {YEAR}-{SESSION}{NUMBER} QA Attempt

## Problem Statement

[Paste problem statement from database or solution file]

## Solution Summary

[1-3 sentences describing the mathematical approach used in the solution]

## Step Breakdown

### Step 1: [Brief description of mathematical operation]

- **Input**: `[GNOSTICA expression]`
- **Expected**: `[What the solution says the result should be]`
- **Actual**: `[What GNOSTICA returned - paste verbatim]`
- **Status**: PASS | FAIL | ERROR | NOT_SUPPORTED
- **Diagnosis**: [If not PASS: root cause analysis]

### Step 2: [Brief description]

- **Input**: `[expression]`
- **Expected**: `[expected]`
- **Actual**: `[actual]`
- **Status**: [status]
- **Diagnosis**: [if needed]

[Continue for all steps...]

## Summary

| Metric | Count |
|--------|-------|
| Steps attempted | |
| Passed | |
| Failed | |
| Errors | |
| Not supported | |

**Pass rate**: X/N (P%)

## GNOSTICA Gaps Identified

### Gap 1: [Short descriptive name]

- **Category**: PARSER | RULE_MISSING | RULE_MISMATCH | INCOMPLETE_SIMPLIFICATION | WRONG_RESULT | PERFORMANCE | UX
- **Severity**: critical | blocking | degraded | minor
- **Description**: [What is missing or broken]
- **Affected steps**: [Which steps above failed due to this gap]
- **Root cause**: [Why does this fail - missing rule? pattern mismatch? parser issue?]
- **Suggested fix**: [Actionable description for DEV worker]
- **Acceptance criterion**: 
```bash
[Exact command that should pass after the fix]
# Expected output: [what it should return]
```

### Gap 2: [Name]

[Same structure...]

## UX Friction Notes

[Any usability issues encountered while testing:]
- [Issue 1]
- [Issue 2]

## Recommendations

Priority order for addressing gaps found in this problem:

1. **[Gap name]** — [Why this is highest priority]
2. **[Gap name]** — [Why this is next]
3. ...

## References

- **Problem source**: [database entry or file path]
- **Solution source**: `corpus/raw/kedlaya/{year}s.tex`
- **Related problems**: [Other problems that likely have similar gaps]

---

**Worker**: MATHQA  
**Problem**: {YEAR}-{SESSION}{NUMBER}  
**Date**: {DATE}  
**Branch**: {BRANCH}  
**GNOSTICA version**: [output of `cargo pkgid` or similar]
