# Trinity Harness: Fixes and Improvements

## Current State (2026-06-06)

- **1522 nodes** scanned
- **351 GREEN** (23.1% coverage)
- **1171 UNTESTED**
- **1148 tests** ran successfully

### Gap Analysis

1148 tests ran but only 351 nodes marked GREEN. The test-to-code mapping loses ~70% of coverage signal.

---

## Priority Fixes

### 1. Cargo Dependency Conflict

**Problem:** Duplicate `sqlite3` links after stash merge.

```
failed to select a version for `libsqlite3-sys` which could resolve this conflict
```

**Fix:** Align sqlite dependencies across workspace crates.

---

## Improvement Ideas

### 1. Edge-Based Test Coverage (High Impact)

**Current:** Name matching only (`test_foo` → `foo`)

**Proposed:** Use call graph edges to determine coverage.

```
Test runs → parser extracts function calls → marks all called functions as covered
```

**Implementation:**
- Parse test function bodies for function calls
- Create "calls" edges from test nodes to production nodes
- When test passes, mark all reachable nodes as GREEN

**Benefit:** Would capture ~90% of actual coverage vs current ~25%.

---

### 2. Explicit Test Mappings (Medium Impact)

**Current:** Convention-only matching

**Proposed:** Allow explicit mappings in `trinity-harness.toml`:

```toml
[[test_mappings]]
test = "tests/integration/test_full_pipeline.rs"
covers = [
    "src/pipeline.rs",
    "src/executor.rs",
    "src/validator.rs"
]

[[test_mappings]]
test_pattern = "test_*_validation"
covers_pattern = "src/validation/*.rs"
```

**Benefit:** Handles integration tests and non-conventional naming.

---

### 3. Runtime Coverage Collection (High Accuracy)

**Current:** Static analysis only

**Proposed:** Instrument tests to collect actual coverage:

```rust
// Option A: Use cargo-llvm-cov
cargo llvm-cov --json > coverage.json
trinity-harness import-coverage coverage.json

// Option B: Custom instrumentation
#[trinity_harness::track]
fn production_function() { ... }
```

**Benefit:** 100% accurate coverage data.

**Tradeoff:** Requires instrumentation, slower tests.

---

### 4. Smarter Name Matching (Quick Win)

**Current patterns:**
- `test_foo` → `foo`

**Add patterns:**
- `test_foo_bar` → `foo_bar` OR `foo` OR `bar`
- `should_foo_when_bar` → `foo`
- `FooTest::test_method` → `Foo::method`
- `test_module_function` → `module::function`

**Implementation:** ~50 lines in `transitions.rs`

---

### 5. Incremental Re-scan on File Change

**Current:** Full scan required after code changes

**Proposed:** 
- Watch for file changes
- Re-parse only changed files
- Update affected edges
- Mark dependent nodes as DIRTY

**Benefit:** Sub-second state updates vs multi-second full scans.

---

### 6. Test Prioritization by Risk

**Current:** `--priority` sorts by recency

**Add factors:**
- Code complexity (cyclomatic)
- Change frequency (git history)
- Previous failure rate
- Dependency depth

```rust
risk_score = complexity * 0.3 
           + change_freq * 0.3 
           + failure_rate * 0.3 
           + depth * 0.1
```

---

### 7. Parallel Test Execution with State Updates

**Current:** Tests run, then state updates batch

**Proposed:** Stream results, update state in real-time:

```
test_a passes → immediately mark covered nodes GREEN
test_b starts → ...
test_a's nodes now show in status
```

**Benefit:** Live progress visibility, early failure detection.

---

### 8. CI Integration Mode

**Current:** CLI only

**Proposed:** GitHub Actions integration:

```yaml
- uses: trinity-harness/action@v1
  with:
    affected-only: true
    fail-on-red: true
    comment-pr: true
```

Output:
- PR comment with coverage delta
- Fail if new code is untested
- Skip tests for unchanged code

---

## Implementation Priority

| # | Feature | Impact | Effort | Priority |
|---|---------|--------|--------|----------|
| 1 | Fix Cargo conflict | Blocker | Low | NOW |
| 2 | Smarter name matching | Medium | Low | Week 1 |
| 3 | Edge-based coverage | High | Medium | Week 2 |
| 4 | Explicit mappings | Medium | Medium | Week 3 |
| 5 | Incremental re-scan | High | High | Week 4 |
| 6 | CI integration | High | Medium | Week 5 |
| 7 | Runtime coverage | High | High | Later |
| 8 | Risk prioritization | Medium | Medium | Later |

---

## Quick Wins Checklist

- [ ] Fix sqlite3 dependency conflict
- [ ] Add `test_foo_bar` → `foo_bar` matching
- [ ] Add `should_*` test pattern support
- [ ] Log unmatched test names for debugging
- [ ] Add `--verbose` flag to show matching decisions
- [ ] Export coverage report as JSON/HTML
