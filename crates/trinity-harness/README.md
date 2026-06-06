# Trinity Harness

A next-gen testing framework that treats code and tests as a unified graph, not separate artifacts.

## Core Idea

Instead of running `cargo test` blindly on 12,743 tests, the harness tracks **which code changed** and runs **only the affected tests**.

```
┌─────────────────────────────────────────────────────────────┐
│                    CODE GRAPH                                │
├─────────────────────────────────────────────────────────────┤
│   [function A] ──depends-on──▶ [struct B]                   │
│        │                            │                        │
│    tests-by                     mirrors-layout               │
│        ▼                            ▼                        │
│   [test_a.rs]                  [B.wgsl]                     │
│                                                              │
│   State: GREEN / RED / DIRTY / UNTESTED per node            │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

| Layer | Purpose |
|-------|---------|
| **Parsers** | Extract functions, structs, classes from Rust, Python, WGSL |
| **Graph** | Build dependency edges (calls, imports, mirrors-layout) |
| **Test Mapping** | Connect tests to code (by convention or explicit TOML config) |
| **State Machine** | Track each node: `Green → Dirty → Green/Red` |
| **Daemon** | Watch files, propagate staleness, trigger minimal test runs |
| **Contracts** | `#[contract]` macros → property-based tests |

## Module Structure

```
src/
├── lib.rs              # Public API
├── constants.rs        # Configuration defaults
├── db.rs               # SQLite state persistence
│
├── parsers/            # Language-specific AST extraction
│   ├── mod.rs          # Parser registry, CodeUnit
│   ├── rust.rs         # syn-based Rust parser
│   ├── python.rs       # rustpython_parser-based Python parser
│   └── wgsl.rs         # naga-based WGSL parser (with struct layouts)
│
├── graph/              # Code graph construction
│   ├── mod.rs          # CodeGraph, CodeNode, NodeId
│   ├── edges.rs        # EdgeType: Calls, Imports, Tests, MirrorsLayout
│   ├── nodes.rs        # Node operations
│   ├── builder.rs      # GraphBuilder with full_scan()
│   ├── deps.rs         # Dependency detection (Rust/Python)
│   ├── crosslang.rs    # Cross-language edges (PyO3, repr(C))
│   └── testmap.rs      # Test-to-code mapping (convention + explicit)
│
├── runners/            # Test execution
│   ├── mod.rs          # Unified runner API
│   ├── cargo.rs        # cargo test with JSON output
│   ├── pytest.rs       # pytest with JSON report
│   ├── executor.rs     # Combined test execution
│   ├── mapper.rs       # Result-to-node mapping
│   ├── transitions.rs  # State transitions (Green/Red/Dirty)
│   └── baseline.rs     # Baseline recording and validation
│
├── daemon/             # File watching and live updates
│   ├── mod.rs          # HarnessDaemon main loop
│   ├── watcher.rs      # File watcher with debouncing
│   ├── processor.rs    # Event → state transition processor
│   └── notify.rs       # Pub/sub notification service
│
├── cli/                # Command-line interface
│   └── mod.rs          # daemon, query, run-stale, update commands
│
├── ci/                 # CI/CD integration
│   └── mod.rs          # GitHub Actions workflow generation
│
└── docs/               # Documentation generation
    └── mod.rs          # API docs, usage guides
```

## Smart Staleness Propagation

When you edit a file, only affected tests run:

```
edit src/math.rs::compute()
       │
       ▼
   mark DIRTY
       │
       ├──▶ tests/test_compute.rs     (runs)
       │
       └──▶ src/renderer.rs::draw()   (depends on compute)
                    │
                    └──▶ tests/test_draw.rs   (also runs)
```

**Result:** ~20% of tests run on a typical change instead of 100%.

## Test Mapping Conventions

### Rust (blackbox)
- `tests/test_<name>.rs` → `src/<name>.rs`
- `tests/blackbox_<name>.rs` → `src/<name>.rs`

### Rust (whitebox/unit)
- `#[test] fn test_<name>()` → `fn <name>()` in same file

### Python
- `test_<name>.py` → `<name>.py`
- `TestFoo` class → `Foo` class

### Explicit (TOML)
```toml
[[mappings]]
test = "tests/integration/*.rs"
targets = ["src/core.rs", "src/utils/*.rs"]
```

## Contract Annotations

```rust
use trinity_contracts::contract;

#[contract]
#[requires(divisor != 0)]
#[ensures(*result == dividend / divisor)]
fn safe_div(dividend: i32, divisor: i32) -> i32 {
    dividend / divisor
}
```

Generates:
- Runtime `debug_assert!` checks
- Property-based tests via proptest
- JSON schema for synth integration

## CLI Commands

```bash
# Start file-watching daemon
trinity-harness daemon

# Query what needs testing
trinity-harness query needs-testing

# Run only stale tests
trinity-harness run-stale

# Update state from test results
trinity-harness update --results ./test-results.json
```

## CI Integration

Generated `.github/workflows/harness.yml`:

```yaml
- name: Query stale tests
  run: trinity-harness query needs-testing --format json > stale.json

- name: Run stale tests
  run: trinity-harness run-stale

- name: Update state
  run: trinity-harness update
```

## Dependencies

- `syn` — Rust AST parsing
- `rustpython_parser` — Python AST parsing
- `naga` — WGSL parsing with struct layout extraction
- `blake3` — Content hashing
- `notify` — File system watching
- `rusqlite` — State persistence
- `serde` / `toml` — Configuration

## Stats

- **39 source files** (10,907 lines)
- **1,378 tests** (all passing)
- **6 development phases** (Infrastructure → Graph → Mapping → Baseline → Workflow → Contracts)
