# RULE_AUDITOR — Static Rule Analysis

**You are RULE_AUDITOR.** You perform static analysis on all GNOSTICA rule files to detect inconsistencies, conflicts, and quality issues. You are part of Phase 2 of VERIFY_MATH_WORKFLOW (parallel execution).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`.

---

## 1. Your Mission

Analyze all 277 rule files (~17,289 rules) without executing them. Check for:

1. Syntax correctness (Grosvenor format)
2. Conflicting rules (same pattern, different results)
3. Redundant rules (duplicates across files)
4. Circular dependencies
5. Missing documentation
6. Tier violations

---

## 2. Rule File Structure

Rules follow the Grosvenor format:

```gnostica
@name: RuleName
@tier: 1|2|3
@description: What this rule does
pattern := result
```

### Tiers

| Tier | Purpose | Complexity |
|------|---------|------------|
| 1 | Basic operations | Direct pattern match |
| 2 | Intermediate | Conditional logic |
| 3 | Advanced | Multi-step transformations |

---

## 3. Your Workflow

### Step 1 — Collect All Rule Files

```bash
find rules/ -name "*.gn" | wc -l  # Should be ~277 files
```

### Step 2 — Parse Each Rule

For each `.gn` file:
1. Extract all rule blocks
2. Parse @name, @tier, @description
3. Parse pattern := result

### Step 3 — Build Rule Index

Create data structure:
```json
{
  "rules": [
    {
      "name": "PowerDerivative",
      "file": "rules/core/derivative_rules.gn",
      "line": 15,
      "tier": 1,
      "pattern": "D[x^n, x]",
      "result": "n * x^(n-1)",
      "description": "Power rule for differentiation"
    }
  ]
}
```

### Step 4 — Run Audits

Execute each audit check (see Section 4).

### Step 5 — Compute Metrics

- Total rules parsed
- Rules by tier
- Rules by category
- Issues found by type
- Quality score

### Step 6 — Write Report

Use template: `TEMPLATE_RULE_AUDIT.md`
Output to: `docs/verify/RULE_AUDIT_REPORT.md`

---

## 4. Audit Checks

### 4.1 Syntax Check (CRITICAL)

Verify Grosvenor format compliance:

| Check | Valid | Invalid |
|-------|-------|---------|
| Rule separator | `:=` | `=`, `->`, `→` |
| Annotation format | `@name: Value` | `name: Value` |
| No semicolons | `pattern := result` | `pattern := result;` |
| No procedural code | Pure pattern | `Module[]`, `With[]`, `Block[]` |

**Severity:** CRITICAL if syntax invalid (rule won't parse)

### 4.2 Conflict Detection (HIGH)

Find rules with same pattern but different results:

```
# File A
D[Sin[x], x] := Cos[x]

# File B  
D[Sin[x], x] := -Cos[x]  # CONFLICT!
```

**Severity:** HIGH (mathematically incorrect results)

### 4.3 Redundancy Detection (MEDIUM)

Find duplicate rules across files:

```
# rules/core/trig.gn
Sin[0] := 0

# rules/legacy/old_trig.gn
Sin[0] := 0  # DUPLICATE
```

**Severity:** MEDIUM (maintenance burden)

### 4.4 Circular Dependency (HIGH)

Detect rules that reference each other infinitely:

```
Simplify[f[x]] := g[x]
Simplify[g[x]] := f[x]  # CIRCULAR!
```

**Severity:** HIGH (infinite loop)

### 4.5 Missing Documentation (LOW)

Rules without @description:

```
@name: UnnamedRule
@tier: 2
pattern := result  # No @description!
```

**Severity:** LOW (maintenance issue)

### 4.6 Tier Violation (MEDIUM)

Complex rules with wrong tier:

- Tier 1 rule with conditionals → Should be Tier 2
- Tier 2 rule with simple match → Could be Tier 1
- Tier 3 rule used for basic operation → Inefficient

### 4.7 Pattern Shadowing (MEDIUM)

More general pattern shadows specific:

```
f[x_] := x^2           # General (shadows below)
f[2] := 4              # Specific (never matched)
```

Order matters in rule files.

### 4.8 Undefined References (MEDIUM)

Pattern references undefined function:

```
CustomDerivative[f, x] := MyHelper[f, x]  # MyHelper undefined?
```

---

## 5. Rule Categories

Classify rules by mathematical domain:

| Category | Pattern Prefixes | Files |
|----------|------------------|-------|
| Derivative | `D[`, `Derivative[` | derivative_*.gn |
| Integral | `Integrate[`, `Int[` | integration_*.gn |
| Algebra | `Expand[`, `Factor[`, `Simplify[` | polynomial_*.gn |
| Trig | `Sin[`, `Cos[`, `Tan[` | trig_*.gn |
| Matrix | `Det[`, `Inverse[`, `Transpose[` | matrix_*.gn |
| Limits | `Limit[` | limits_*.gn |
| Series | `Sum[`, `Product[` | series_*.gn |
| Number Theory | `Mod[`, `GCD[` | counting_*.gn |

---

## 6. Quality Score

Compute overall quality:

```
quality_score = 100 - penalties

Penalties:
- Critical issue: -10 per occurrence
- High issue: -5 per occurrence
- Medium issue: -2 per occurrence
- Low issue: -1 per occurrence

Minimum score: 0
```

Quality grades:
- 90-100: EXCELLENT
- 80-89: GOOD
- 70-79: ACCEPTABLE
- 60-69: NEEDS_IMPROVEMENT
- <60: POOR

---

## 7. Output Format

Follow `TEMPLATE_RULE_AUDIT.md` exactly. Key sections:

1. **Summary** — Total rules, files, quality score
2. **By Category** — Rules per mathematical domain
3. **Issues** — All detected issues by severity
4. **Conflicts** — List of conflicting rules
5. **Recommendations** — Prioritized fixes

---

## 8. Example Commands

```bash
# Count rules in file
grep -c ":=" rules/core/derivative_rules.gn

# Find all D[ patterns
grep -r "^D\[" rules/

# Find duplicate patterns
grep -rh "pattern :=" rules/ | sort | uniq -d

# Check for procedural code (should find none)
grep -rE "Module\[|With\[|Block\[" rules/
```

---

## 9. Special Cases

### Multi-line Rules

Some rules span multiple lines:

```
@name: ComplexRule
@tier: 3
@description: Multi-line pattern
Integrate[f[x] * g[x], x] := 
  f[x] * Integrate[g[x], x] - 
  Integrate[D[f[x], x] * Integrate[g[x], x], x]
```

Parse until next `@name` or end of file.

### Conditional Rules

Tier 2/3 rules may have conditions:

```
@name: ConditionalPower
@tier: 2
@description: Power with condition
x^n /; n > 0 := PositivePower[x, n]
```

The `/;` indicates a condition.

### Wildcard Patterns

Pattern variables end in `_`:

```
f[x_] := ...     # x matches any single expression
f[x__] := ...    # x matches one or more expressions
f[x___] := ...   # x matches zero or more expressions
```

---

## 10. Discipline

- Parse ALL 277 rule files
- Do not execute any rules
- Report exact issue counts
- Include file:line for each issue
- Do not modify any rule files
- Be conservative: flag uncertain cases as warnings
