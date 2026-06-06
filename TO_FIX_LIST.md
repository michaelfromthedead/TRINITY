# TO_FIX_LIST — QA Audit Results

**Generated:** 2026-06-05  
**Updated:** 2026-06-06 (all items FIXED)  
**Audited:** trinity-harness, trinity-contracts  
**Tests Verified:** 1370 (1147 + 223)

---

## Source File Inventory (39 files, 10,907 lines)

### trinity-harness (31 files, 8,530 lines)

| # | File | Lines | Module |
|---|------|-------|--------|
| 1 | `src/lib.rs` | 33 | Root |
| 2 | `src/constants.rs` | 31 | Config constants |
| 3 | `src/db.rs` | 42 | Database |
| 4 | `src/ci/mod.rs` | 243 | CI workflow generation |
| 5 | `src/cli/mod.rs` | 316 | CLI commands |
| 6 | `src/daemon/mod.rs` | 286 | Daemon core |
| 7 | `src/daemon/notify.rs` | 300 | Notifications |
| 8 | `src/daemon/processor.rs` | 301 | Event processing |
| 9 | `src/daemon/watcher.rs` | 361 | File watcher |
| 10 | `src/docs/mod.rs` | 350 | Documentation |
| 11 | `src/graph/mod.rs` | 186 | Graph core |
| 12 | `src/graph/builder.rs` | 526 | Graph building |
| 13 | `src/graph/crosslang.rs` | 313 | Cross-language edges |
| 14 | `src/graph/deps.rs` | 658 | Dependency detection |
| 15 | `src/graph/edges.rs` | 41 | Edge types |
| 16 | `src/graph/nodes.rs` | 37 | Node types |
| 17 | `src/graph/testmap.rs` | 1287 | Test mapping |
| 18 | `src/parsers/mod.rs` | 116 | Parser registry |
| 19 | `src/parsers/python.rs` | 363 | Python parser |
| 20 | `src/parsers/rust.rs` | 336 | Rust parser |
| 21 | `src/parsers/wgsl.rs` | 301 | WGSL parser |
| 22 | `src/runners/mod.rs` | 38 | Runners core |
| 23 | `src/runners/baseline.rs` | 314 | Baseline recording |
| 24 | `src/runners/cargo.rs` | 352 | Cargo test runner |
| 25 | `src/runners/executor.rs` | 282 | Test executor |
| 26 | `src/runners/mapper.rs` | 244 | Result mapping |
| 27 | `src/runners/pytest.rs` | 318 | Pytest runner |
| 28 | `src/runners/transitions.rs` | 237 | State transitions |
| 29 | `src/runners/validation.rs` | 240 | Validation |
| 30 | `src/state/mod.rs` | 5 | State core |
| 31 | `src/state/machine.rs` | 73 | State machine |

### trinity-contracts (7 files, 2,100 lines)

| # | File | Lines | Module |
|---|------|-------|--------|
| 1 | `src/lib.rs` | 304 | Root |
| 2 | `src/algebra.rs` | 311 | Algebraic properties |
| 3 | `src/layout.rs` | 303 | Layout contracts |
| 4 | `src/proptest.rs` | 276 | Property testing |
| 5 | `src/rollout.rs` | 333 | Rollout tracking |
| 6 | `src/runtime.rs` | 273 | Runtime checks |
| 7 | `src/schema.rs` | 300 | Constraint schemas |

### trinity-contracts-macros (1 file, 277 lines)

| # | File | Lines | Module |
|---|------|-------|--------|
| 1 | `src/lib.rs` | 277 | Proc macros |

---

## ✅ ALL ISSUES FIXED (commit db0c64d0)

## Priority 1: Compiler Warnings (5 min) — ✅ FIXED

| File | Line | Issue |
|------|------|-------|
| `crates/trinity-harness/src/cli/mod.rs` | 8 | unused import: `CodeGraph` |
| `crates/trinity-harness/src/daemon/processor.rs` | 6 | unused import: `std::path::Path` |
| `crates/trinity-harness/src/runners/cargo.rs` | 5 | unused import: `std::collections::HashMap` |

**Fix:** Remove unused imports. ✅ DONE

---

## Priority 2: Lazy Test (5 min) — ✅ FIXED

| File | Line | Issue |
|------|------|-------|
| `crates/trinity-harness/tests/whitebox_watcher.rs` | 144 | `assert!(true)` — tests nothing |

**Current:**
```rust
#[test]
fn test_debouncer_new() {
    let debouncer = Debouncer::new(100);
    // Just verify it creates successfully
    assert!(true);
}
```

**Fix:** Replace with real assertion:
```rust
#[test]
fn test_debouncer_new() {
    let debouncer = Debouncer::new(100);
    assert_eq!(debouncer.debounce_ms(), 100);
}
```

✅ DONE — Added `debounce_ms()` getter and real assertion.

---

## Priority 3: Misleading Comment (1 min) — ✅ FIXED

| File | Line | Issue |
|------|------|-------|
| `crates/trinity-harness/src/daemon/mod.rs` | 206 | Comment says "In a real implementation" but code works via `emit_event()` |

**Fix:** Delete the misleading comment or rewrite to reflect actual behavior. ✅ DONE

---

## Priority 4: Magic Numbers — Need Config Module (30 min) — ✅ FIXED

✅ Created `constants.rs` and wired all usages:

| Value | File | Line | Suggested Constant |
|-------|------|------|--------------------|
| `600` | `runners/cargo.rs` | 33 | `DEFAULT_CARGO_TIMEOUT_SECS` |
| `1800` | `runners/pytest.rs` | 33 | `DEFAULT_PYTEST_TIMEOUT_SECS` |
| `1000` | `daemon/mod.rs` | 45 | `DEFAULT_POLL_INTERVAL_MS` |
| `100` | `daemon/mod.rs` | 46 | `DEFAULT_DEBOUNCE_MS` |
| `100` | `daemon/watcher.rs` | 45 | `DEFAULT_DEBOUNCE_MS` |
| `500` | `daemon/watcher.rs` | 46 | `DEFAULT_POLL_INTERVAL_MS` |
| `100` | `daemon/mod.rs` | 47 | `MAX_EVENTS_PER_TICK` |
| `10` | `daemon/processor.rs` | 26 | `MAX_PROPAGATION_DEPTH` |
| `1000` | `daemon/notify.rs` | 113 | `MAX_LOG_SIZE` |
| `10000` | `daemon/notify.rs` | 239 | `MAX_BUFFER_SIZE` |
| `-64000` | `db.rs` | 18 | `SQLITE_CACHE_SIZE` |

**Fix:** Create `crates/trinity-harness/src/constants.rs`:
```rust
pub const DEFAULT_CARGO_TIMEOUT_SECS: u64 = 600;
pub const DEFAULT_PYTEST_TIMEOUT_SECS: u64 = 1800;
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_DEBOUNCE_MS: u64 = 100;
pub const MAX_EVENTS_PER_TICK: usize = 100;
pub const MAX_PROPAGATION_DEPTH: usize = 10;
pub const MAX_LOG_SIZE: usize = 1000;
pub const MAX_BUFFER_SIZE: usize = 10000;
pub const SQLITE_CACHE_SIZE: i32 = -64000;
```

✅ DONE — File created and all magic numbers replaced.

---

## Priority 5: Silent Catch-All Arms (investigate) — ✅ FIXED

These `_ =>` arms may silently swallow unknown cases:

| File | Line | Pattern | Risk |
|------|------|---------|------|
| `runners/pytest.rs` | 240 | `_ => TestOutcome::Unknown` | Swallows unknown test states |
| `graph/crosslang.rs` | 163 | `_ => "?".to_string()` | Returns garbage for unknown languages |

**Fix:** Add explicit enum variants or log warnings for unknown cases. ✅ DONE — Added `eprintln!` warnings.

---

## Summary — ALL FIXED ✅

| Priority | Count | Time Est. | Status |
|----------|-------|-----------|--------|
| P1 Warnings | 3 | 5 min | ✅ |
| P2 Lazy Test | 1 | 5 min | ✅ |
| P3 Comment | 1 | 1 min | ✅ |
| P4 Magic Numbers | 11 | 30 min | ✅ |
| P5 Catch-Alls | 2 | 10 min | ✅ |
| **Total** | **18** | **~51 min** | **ALL DONE** |

---

## Not Issues (Verified OK)

- Tests are REAL (1370 verified, ran successfully)
- No `#[ignore]` tests
- No `todo!()` or `unimplemented!()` markers
- No empty test functions
- No TODO/FIXME/HACK comments
- Daemon `emit_event()` actually works (comment just misleading)
- `panic!()` in contracts runtime is intentional (contract violations)

---

## Test File Audit Checklist (66 files)

| # | File | Tests | Checked? | Need Fix? |
|---|------|-------|----------|-----------|
| 1 | blackbox_baseline.rs | 5 | [x] | [ ] |
| 2 | blackbox_builder.rs | 67 | [x] | [ ] |
| 3 | blackbox_cargo_runner.rs | 4 | [x] | [ ] |
| 4 | blackbox_ci.rs | 5 | [x] | [ ] |
| 5 | blackbox_cli.rs | 5 | [x] | [ ] |
| 6 | blackbox_coverage_report.rs | 6 | [x] | [ ] |
| 7 | blackbox_crate_structure.rs | 10 | [x] | [ ] |
| 8 | blackbox_crosslang.rs | 11 | [x] | [ ] |
| 9 | blackbox_daemon.rs | 5 | [x] | [ ] |
| 10 | blackbox_db.rs | 12 | [x] | [ ] |
| 11 | blackbox_dependencies.rs | 4 | [x] | [ ] |
| 12 | blackbox_deps.rs | 12 | [x] | [ ] |
| 13 | blackbox_docs.rs | 5 | [x] | [ ] |
| 14 | blackbox_executor.rs | 6 | [x] | [ ] |
| 15 | blackbox_inline_tests.rs | 16 | [x] | [ ] |
| 16 | blackbox_manual_mapping.rs | 8 | [x] | [ ] |
| 17 | blackbox_map_python_tests.rs | 8 | [x] | [ ] |
| 18 | blackbox_map_rust_tests.rs | 8 | [x] | [ ] |
| 19 | blackbox_module_exports.rs | 4 | [x] | [ ] |
| 20 | blackbox_notify.rs | 5 | [x] | [ ] |
| 21 | blackbox_processor.rs | 5 | [x] | [ ] |
| 22 | blackbox_pytest_runner.rs | 4 | [x] | [ ] |
| 23 | blackbox_python_parser.rs | 67 | [x] | [ ] |
| 24 | blackbox_result_mapper.rs | 4 | [x] | [ ] |
| 25 | blackbox_rust_parser.rs | 52 | [x] | [ ] |
| 26 | blackbox_schema.rs | 19 | [x] | [ ] |
| 27 | blackbox_testmap.rs | 9 | [x] | [ ] |
| 28 | blackbox_transitions.rs | 5 | [x] | [ ] |
| 29 | blackbox_unified_codeunit.rs | 59 | [x] | [ ] |
| 30 | blackbox_unmapped_tests.rs | 6 | [x] | [ ] |
| 31 | blackbox_validation.rs | 5 | [x] | [ ] |
| 32 | blackbox_validation_baseline.rs | 5 | [x] | [ ] |
| 33 | blackbox_watcher.rs | 5 | [x] | [ ] |
| 34 | blackbox_wgsl_parser.rs | 56 | [x] | [ ] |
| 35 | whitebox_baseline.rs | 15 | [x] | [ ] |
| 36 | whitebox_builder.rs | 57 | [x] | [ ] |
| 37 | whitebox_cargo_runner.rs | 16 | [x] | [ ] |
| 38 | whitebox_ci.rs | 17 | [x] | [ ] |
| 39 | whitebox_cli.rs | 12 | [x] | [ ] |
| 40 | whitebox_coverage_report.rs | 13 | [x] | [ ] |
| 41 | whitebox_crosslang.rs | 19 | [x] | [ ] |
| 42 | whitebox_daemon.rs | 15 | [x] | [ ] |
| 43 | whitebox_db.rs | 43 | [x] | [ ] |
| 44 | whitebox_deps.rs | 21 | [x] | [ ] |
| 45 | whitebox_docs.rs | 19 | [x] | [ ] |
| 46 | whitebox_executor.rs | 19 | [x] | [ ] |
| 47 | whitebox_graph.rs | 17 | [x] | [ ] |
| 48 | whitebox_inline_tests.rs | 10 | [x] | [ ] |
| 49 | whitebox_manual_mapping.rs | 14 | [x] | [ ] |
| 50 | whitebox_map_python_tests.rs | 9 | [x] | [ ] |
| 51 | whitebox_map_rust_tests.rs | 9 | [x] | [ ] |
| 52 | whitebox_notify.rs | 19 | [x] | [ ] |
| 53 | whitebox_parsers.rs | 42 | [x] | [ ] |
| 54 | whitebox_processor.rs | 14 | [x] | [ ] |
| 55 | whitebox_pytest_runner.rs | 12 | [x] | [ ] |
| 56 | whitebox_python_parser.rs | 37 | [x] | [ ] |
| 57 | whitebox_result_mapper.rs | 17 | [x] | [ ] |
| 58 | whitebox_rust_parser.rs | 50 | [x] | [ ] |
| 59 | whitebox_state.rs | 20 | [x] | [ ] |
| 60 | whitebox_testmap.rs | 15 | [x] | [ ] |
| 61 | whitebox_transitions.rs | 18 | [x] | [ ] |
| 62 | whitebox_unmapped_tests.rs | 13 | [x] | [ ] |
| 63 | whitebox_validation.rs | 14 | [x] | [ ] |
| 64 | whitebox_validation_baseline.rs | 16 | [x] | [ ] |
| 65 | whitebox_watcher.rs | 14 | [x] | [x] |
| 66 | whitebox_wgsl_parser.rs | 25 | [x] | [ ] |

**Totals:** 66 files, 1147 tests — **ALL CHECKED**

**Result:** 65 files CLEAN, 1 file needs fix (whitebox_watcher.rs:144 `assert!(true)`)

**Legend:**
- `[x]` = Checked/Needs fix
- `[ ]` = Not yet reviewed
