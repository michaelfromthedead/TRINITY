# Baseline Schema Specification

**Version:** 1.0.0
**Purpose:** Define the structure of baseline files for regression detection

---

## Overview

Baseline files store a snapshot of GNOSTICA verification state at a specific version. They enable regression detection by comparing current results against known-good states.

---

## File Locations

```
baseline/
├── baseline_current.json      # Active baseline (symlink or copy)
├── baseline_v3.0.0.json       # Version-tagged baselines
├── baseline_v2.9.0.json
├── baseline_history.json      # Update history log
└── README.md                  # Baseline documentation
```

---

## JSON Schema

### Root Structure

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["version", "created", "gnostica_version", "test_results", "identity_results", "rule_audit", "corpus_results", "canonical_expressions", "rule_hashes", "performance"],
  "properties": {
    "version": { "type": "string", "description": "Baseline schema version" },
    "created": { "type": "string", "format": "date-time" },
    "gnostica_version": { "type": "string" },
    "verified_by": { "type": "string", "description": "Who verified this baseline" },
    "notes": { "type": "string" }
  }
}
```

### Complete Baseline Example

```json
{
  "version": "1.0.0",
  "created": "2026-05-20T12:00:00Z",
  "gnostica_version": "3.0.0",
  "verified_by": "VERIFY_MATH_WORKFLOW",
  "notes": "Post-Phase 38 CORPUS COMPLETE baseline",

  "test_results": {
    "total": 10459,
    "passed": 10459,
    "failed": 0,
    "ignored": 0,
    "duration_ms": 45000,
    "by_category": {
      "series": { "total": 1500, "passed": 1500, "failed": 0 },
      "integration": { "total": 1200, "passed": 1200, "failed": 0 },
      "polynomial": { "total": 1100, "passed": 1100, "failed": 0 },
      "matrix": { "total": 900, "passed": 900, "failed": 0 },
      "limits": { "total": 800, "passed": 800, "failed": 0 },
      "number_theory": { "total": 700, "passed": 700, "failed": 0 },
      "calculus": { "total": 600, "passed": 600, "failed": 0 },
      "core": { "total": 500, "passed": 500, "failed": 0 },
      "other": { "total": 3159, "passed": 3159, "failed": 0 }
    },
    "by_phase": {
      "phase_33": { "total": 200, "passed": 200 },
      "phase_34": { "total": 210, "passed": 210 },
      "phase_35": { "total": 220, "passed": 220 },
      "phase_36": { "total": 230, "passed": 230 },
      "phase_37": { "total": 240, "passed": 240 },
      "phase_38": { "total": 195, "passed": 195 }
    },
    "test_list": [
      { "name": "test_power_rule", "status": "pass", "duration_ms": 5 },
      { "name": "test_chain_rule", "status": "pass", "duration_ms": 8 }
    ]
  },

  "identity_results": {
    "total": 210,
    "passed": 210,
    "failed": 0,
    "partial": 0,
    "errors": 0,
    "skipped": 0,
    "by_category": {
      "derivative": { "total": 30, "passed": 30 },
      "integral": { "total": 40, "passed": 40 },
      "algebraic": { "total": 35, "passed": 35 },
      "trigonometric": { "total": 30, "passed": 30 },
      "limit": { "total": 20, "passed": 20 },
      "series": { "total": 25, "passed": 25 },
      "matrix": { "total": 15, "passed": 15 },
      "number_theory": { "total": 15, "passed": 15 }
    },
    "critical_identities": {
      "total": 13,
      "passed": 13,
      "list": [
        { "id": "DERIV_001", "name": "Power Rule", "status": "pass" },
        { "id": "DERIV_002", "name": "Product Rule", "status": "pass" },
        { "id": "DERIV_003", "name": "Chain Rule", "status": "pass" }
      ]
    }
  },

  "rule_audit": {
    "total_files": 277,
    "total_rules": 17289,
    "quality_score": 95,
    "quality_grade": "EXCELLENT",
    "issues": {
      "critical": 0,
      "high": 0,
      "medium": 12,
      "low": 45
    },
    "by_tier": {
      "tier_1": 8500,
      "tier_2": 6000,
      "tier_3": 2789
    },
    "by_category": {
      "derivative": { "files": 35, "rules": 2100 },
      "integral": { "files": 40, "rules": 2500 },
      "polynomial": { "files": 45, "rules": 2800 },
      "matrix": { "files": 30, "rules": 1800 },
      "limits": { "files": 25, "rules": 1500 },
      "series": { "files": 35, "rules": 2200 },
      "number_theory": { "files": 20, "rules": 1200 },
      "core": { "files": 47, "rules": 3189 }
    }
  },

  "corpus_results": {
    "total_problems": 989,
    "implemented": 884,
    "out_of_scope": 100,
    "already_covered": 5,
    "validated": 884,
    "passed": 884,
    "failed": 0,
    "pass_rate": 100.0,
    "by_era": {
      "classic": { "total": 336, "implemented": 300, "passed": 300 },
      "transitional": { "total": 240, "implemented": 215, "passed": 215 },
      "modern": { "total": 300, "implemented": 265, "passed": 265 },
      "contemporary": { "total": 180, "implemented": 155, "passed": 155 }
    },
    "by_difficulty": {
      "A1": { "implemented": 75, "passed": 75 },
      "A2": { "implemented": 73, "passed": 73 },
      "A3": { "implemented": 70, "passed": 70 },
      "A4": { "implemented": 68, "passed": 68 },
      "A5": { "implemented": 65, "passed": 65 },
      "A6": { "implemented": 60, "passed": 60 },
      "B1": { "implemented": 76, "passed": 76 },
      "B2": { "implemented": 74, "passed": 74 },
      "B3": { "implemented": 71, "passed": 71 },
      "B4": { "implemented": 69, "passed": 69 },
      "B5": { "implemented": 66, "passed": 66 },
      "B6": { "implemented": 61, "passed": 61 }
    }
  },

  "canonical_expressions": [
    {
      "id": "CAN_DERIV_001",
      "input": "D[x^2, x]",
      "output": "2*x",
      "verified": true
    },
    {
      "id": "CAN_INTEG_001",
      "input": "Integrate[x, x]",
      "output": "x^2/2",
      "verified": true
    }
  ],

  "rule_hashes": {
    "rules/core/derivative_rules.gn": "sha256:abc123...",
    "rules/core/integration_rules.gn": "sha256:def456...",
    "rules/algebra/polynomial_p38.gn": "sha256:789ghi..."
  },

  "performance": {
    "total_duration_ms": 180000,
    "test_avg_ms": 17.2,
    "test_median_ms": 8.5,
    "test_p95_ms": 45.0,
    "slowest_tests": [
      { "name": "test_complex_integral_p38", "duration_ms": 850 },
      { "name": "test_matrix_exponential", "duration_ms": 720 }
    ]
  }
}
```

---

## Field Descriptions

### test_results

| Field | Type | Description |
|-------|------|-------------|
| total | int | Total number of tests |
| passed | int | Tests that passed |
| failed | int | Tests that failed |
| ignored | int | Tests marked #[ignore] |
| duration_ms | int | Total test duration |
| by_category | object | Results grouped by mathematical domain |
| by_phase | object | Results grouped by implementation phase |
| test_list | array | Individual test results (optional, can be large) |

### identity_results

| Field | Type | Description |
|-------|------|-------------|
| total | int | Total identities tested |
| passed | int | Identities verified |
| failed | int | Identities that failed |
| partial | int | Partial passes |
| errors | int | Evaluation errors |
| skipped | int | Intentionally skipped |
| by_category | object | Results by identity category |
| critical_identities | object | Status of must-pass identities |

### rule_audit

| Field | Type | Description |
|-------|------|-------------|
| total_files | int | Number of .gn files |
| total_rules | int | Total rules across all files |
| quality_score | int | 0-100 quality score |
| quality_grade | string | EXCELLENT/GOOD/ACCEPTABLE/etc |
| issues | object | Issue counts by severity |
| by_tier | object | Rule counts by tier |
| by_category | object | Rules by mathematical domain |

### corpus_results

| Field | Type | Description |
|-------|------|-------------|
| total_problems | int | Total Putnam problems (989) |
| implemented | int | IMPLEMENT classification (884) |
| out_of_scope | int | OUT_OF_SCOPE classification (100) |
| already_covered | int | ALREADY_COVERED classification (5) |
| validated | int | Problems actually tested |
| passed | int | Problems that passed |
| failed | int | Problems that failed |
| pass_rate | float | Percentage passed |
| by_era | object | Results by historical era |
| by_difficulty | object | Results by problem level (A1-B6) |

### canonical_expressions

Array of expression snapshots:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Expression identifier |
| input | string | Input expression |
| output | string | Expected output |
| verified | bool | Whether output was verified |

### rule_hashes

Object mapping file paths to SHA-256 hashes:

```json
{
  "rules/core/file.gn": "sha256:hexstring..."
}
```

### performance

| Field | Type | Description |
|-------|------|-------------|
| total_duration_ms | int | Total verification time |
| test_avg_ms | float | Average test duration |
| test_median_ms | float | Median test duration |
| test_p95_ms | float | 95th percentile duration |
| slowest_tests | array | Top N slowest tests |

---

## Baseline Update Protocol

### When to Update

1. After VERIFIED verdict from full workflow
2. After human review confirms changes are intentional
3. Before major releases

### How to Update

```bash
# Manual update (recommended)
VERIFY_MATH_WORKFLOW --update-baseline

# This will:
# 1. Archive current baseline: baseline_v{old}.json
# 2. Generate new baseline from current results
# 3. Set baseline_current.json to new baseline
# 4. Log update in baseline_history.json
```

### Never Auto-Update

Baselines should NEVER be automatically updated. This ensures regressions are caught, not masked.

---

## Baseline History

Track all baseline updates:

```json
{
  "history": [
    {
      "version": "3.0.0",
      "created": "2026-05-20T12:00:00Z",
      "previous": "2.9.0",
      "changes": "Phase 38 CORPUS COMPLETE",
      "verified_by": "human_reviewer",
      "workflow_run": "run_20260520_001"
    }
  ]
}
```

---

## Validation

Baselines should be validated against this schema before use:

```bash
# Validate baseline JSON
npx ajv validate -s baseline_schema.json -d baseline_current.json
```

---

*Schema version: 1.0.0*
*Last updated: 2026-05-20*
