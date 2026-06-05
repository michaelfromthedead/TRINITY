# V2 Migration Guide

**Started:** 2026-06-05
**Status:** PLANNING
**Depends On:** All V2 documents

---

## Executive Summary

Migrate TRINITY from **12,743 existing tests** to the **V2 Harness** in a single coordinated effort.

| Current State | Target State |
|---------------|--------------|
| Tests scattered across `tests/` | Code graph with statechart per unit |
| No dependency tracking | Full dependency DAG |
| Run all tests on change | Run only stale tests |
| No contract specification | Contracts → synth → tests |
| No cross-language tracking | Rust ↔ Python ↔ WGSL unified |

**Migration is ONE STEP, not incremental.**

---

## Part 1: Current State Assessment

### 1.1 What Exists

```
TRINITY/
├── crates/
│   └── renderer-backend/
│       ├── src/           # Rust implementation (~130 files)
│       └── tests/         # Rust tests (~50 blackbox_*.rs files)
│
├── engine/                # Python implementation (~982 files)
│
├── tests/                 # Python tests (~929 files)
│   ├── unit/
│   ├── integration/
│   └── ...
│
└── crates/renderer-backend/shaders/  # WGSL shaders
```

### 1.2 Test Categories

| Category | Location | Count | Framework |
|----------|----------|-------|-----------|
| Rust unit | `crates/*/src/**` | ~2,000 | `#[test]` |
| Rust blackbox | `crates/*/tests/` | ~200 | `#[test]` |
| Python unit | `tests/unit/` | ~5,000 | pytest |
| Python integration | `tests/integration/` | ~3,000 | pytest |
| Python e2e | `tests/e2e/` | ~500 | pytest |
| Shader validation | N/A | ~100 | naga |
| Benchmarks | various | ~200 | criterion |
| **Total** | | **~12,743** | |

### 1.3 Test Naming Conventions

```rust
// Rust: test_ prefix or tests module
#[test]
fn test_frustum_culling() { ... }

mod tests {
    #[test]
    fn culling_excludes_outside() { ... }
}

// Blackbox: blackbox_*.rs
// tests/blackbox_frame_graph.rs → tests frame_graph module
```

```python
# Python: test_ prefix
def test_render_pipeline():
    ...

class TestFrameGraph:
    def test_compile(self):
        ...
```

### 1.4 Test → Code Mapping (Current)

Currently implicit:
- `tests/blackbox_foo.rs` → tests `src/foo.rs` (by convention)
- `tests/unit/test_bar.py` → tests `engine/bar.py` (by convention)
- No explicit `#[tests(...)]` edges

---

## Part 2: Target State

### 2.1 The V2 Harness

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              V2 HARNESS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                          CODE GRAPH                                  │  │
│   │                                                                      │  │
│   │   Every function, struct, module is a NODE                          │  │
│   │   Dependencies, test relationships are EDGES                        │  │
│   │   Each node has a STATE (green, stale, red, etc.)                   │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                          CONTRACTS                                   │  │
│   │                                                                      │  │
│   │   #[contract]                                                        │  │
│   │   #![requires(...)]  → synth schema → generated tests               │  │
│   │   #![ensures(...)]                                                   │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                          PERSISTENCE                                 │  │
│   │                                                                      │  │
│   │   brain.db (SuperSQLite)                                            │  │
│   │   - code_nodes, code_edges, code_events                             │  │
│   │   - code_state_history, code_contracts                              │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                          WORKFLOW                                    │  │
│   │                                                                      │  │
│   │   File watcher → Event processor → State transitions                │  │
│   │   CI pipeline → Test results → State updates                        │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 What Changes

| Before | After |
|--------|-------|
| Tests are standalone files | Tests are EDGES in the graph |
| Run by file path | Run by code unit state |
| No dependency awareness | Full dependency tracking |
| All tests on CI | Only stale tests on CI |
| Tests define themselves | Contracts define, tests verify |
| Manual test coverage analysis | Automatic coverage tracking |

---

## Part 3: Migration Strategy

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIGRATION PHASES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐                                                         │
│   │   PHASE 1    │  INFRASTRUCTURE                                         │
│   │              │  - Build harness crate                                  │
│   │  (1-2 days)  │  - Set up SuperSQLite                                   │
│   │              │  - Create parsers                                        │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   PHASE 2    │  INITIAL PARSE                                          │
│   │              │  - Parse all Rust/Python/WGSL                           │
│   │  (1 day)     │  - Build code graph                                     │
│   │              │  - All nodes start as UNKNOWN                           │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   PHASE 3    │  TEST MAPPING                                           │
│   │              │  - Map existing tests to nodes                          │
│   │  (2-3 days)  │  - Create "tests" edges                                 │
│   │              │  - Validate coverage                                     │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   PHASE 4    │  BASELINE RUN                                           │
│   │              │  - Run ALL tests once                                   │
│   │  (1 day)     │  - Mark passing nodes GREEN                             │
│   │              │  - Mark failing nodes RED                               │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   PHASE 5    │  WORKFLOW ACTIVATION                                    │
│   │              │  - Start file watcher                                   │
│   │  (1 day)     │  - Enable event processor                               │
│   │              │  - CI integration                                        │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   PHASE 6    │  CONTRACT ANNOTATION                                    │
│   │              │  - Add contracts to critical code                       │
│   │  (ongoing)   │  - synth schema generation                              │
│   │              │  - Property test generation                             │
│   └──────────────┘                                                         │
│                                                                             │
│   TOTAL: ~1 week for Phases 1-5, then ongoing for Phase 6                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Phase 1 — Infrastructure

### 4.1 Create Harness Crate

```bash
# Create the harness crate
mkdir -p crates/harness
cd crates/harness
cargo init --lib
```

```toml
# crates/harness/Cargo.toml

[package]
name = "trinity-harness"
version = "0.1.0"
edition = "2021"

[dependencies]
# Parsing
syn = { version = "2", features = ["full", "parsing", "visit"] }
rustpython-parser = "0.3"
naga = { version = "0.14", features = ["wgsl-in"] }
tree-sitter = "0.20"
tree-sitter-rust = "0.20"
tree-sitter-python = "0.20"

# Storage
superrusqlite = { path = "../../../SQLITE/platform/superrusqlite" }

# Hashing
blake3 = "1"

# Utilities
walkdir = "2"
notify = "6"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }

# Statecharts
superstate = { path = "../superstate" }

[dev-dependencies]
proptest = "1"
```

### 4.2 Directory Structure

```
crates/harness/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   │
│   ├── parse/                # AST parsing
│   │   ├── mod.rs
│   │   ├── rust.rs           # syn-based Rust parser
│   │   ├── python.rs         # rustpython parser
│   │   └── wgsl.rs           # naga parser
│   │
│   ├── graph/                # Code graph
│   │   ├── mod.rs
│   │   ├── node.rs
│   │   ├── edge.rs
│   │   └── query.rs
│   │
│   ├── state/                # State machine
│   │   ├── mod.rs
│   │   ├── machine.rs
│   │   └── transitions.rs
│   │
│   ├── db/                   # Persistence
│   │   ├── mod.rs
│   │   ├── schema.sql
│   │   ├── nodes.rs
│   │   ├── edges.rs
│   │   ├── events.rs
│   │   └── history.rs
│   │
│   ├── workflow/             # Event processing
│   │   ├── mod.rs
│   │   ├── watcher.rs
│   │   ├── processor.rs
│   │   └── propagation.rs
│   │
│   ├── contracts/            # Contract extraction
│   │   ├── mod.rs
│   │   ├── extract.rs
│   │   └── synth.rs
│   │
│   └── cli/                  # CLI interface
│       ├── mod.rs
│       └── commands.rs
│
└── tests/
    ├── parse_rust.rs
    ├── parse_python.rs
    └── graph_build.rs
```

### 4.3 Initialize Database

```bash
# Create the brain database
harness init --db brain.db
```

```sql
-- crates/harness/src/db/schema.sql
-- Full schema from V2_SUPERSQLITE_PERSISTENCE.md

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS code_nodes ( ... );
CREATE TABLE IF NOT EXISTS code_edges ( ... );
CREATE TABLE IF NOT EXISTS code_events ( ... );
CREATE TABLE IF NOT EXISTS code_state_history ( ... );
CREATE TABLE IF NOT EXISTS code_contracts ( ... );
CREATE TABLE IF NOT EXISTS struct_layouts ( ... );
CREATE TABLE IF NOT EXISTS test_results ( ... );
CREATE TABLE IF NOT EXISTS stream_cursors ( ... );

-- Indexes, views, triggers...
```

---

## Part 5: Phase 2 — Initial Parse

### 5.1 Parse Script

```bash
#!/bin/bash
# scripts/migrate-phase2-parse.sh

set -e

echo "=== PHASE 2: Initial Parse ==="

# Parse Rust code
echo "Parsing Rust..."
harness parse \
    --lang rust \
    --paths crates/renderer-backend/src \
    --paths crates/superstate/src \
    --paths crates/synth/src

# Parse Python code
echo "Parsing Python..."
harness parse \
    --lang python \
    --paths engine \
    --paths trinity

# Parse WGSL shaders
echo "Parsing WGSL..."
harness parse \
    --lang wgsl \
    --paths crates/renderer-backend/shaders

# Report
echo ""
echo "=== Parse Complete ==="
harness stats
```

### 5.2 Expected Output

```
=== Parse Complete ===

Language Distribution:
  Rust:   3,847 nodes (functions: 2,103, structs: 892, ...)
  Python: 8,234 nodes (functions: 5,102, classes: 1,890, ...)
  WGSL:     156 nodes (functions: 89, structs: 67)
  ─────────────────────
  Total:  12,237 code nodes

Edge Distribution:
  imports:    4,521
  calls:      8,923
  references: 3,102
  contains:   5,234
  ─────────────────────
  Total:      21,780 edges

All nodes initialized to state: UNKNOWN
```

---

## Part 6: Phase 3 — Test Mapping

### 6.1 Mapping Strategy

| Test Location | Mapping Rule |
|---------------|--------------|
| `tests/blackbox_foo.rs` | Tests `src/foo.rs` module |
| `tests/blackbox_foo_bar.rs` | Tests `src/foo/bar.rs` module |
| `test_foo()` in `src/mod.rs` | Tests functions in same module |
| `tests/unit/test_foo.py` | Tests `engine/foo.py` |
| `TestFoo` class | Tests `Foo` class |

### 6.2 Mapping Script

```rust
// crates/harness/src/migration/test_mapper.rs

pub struct TestMapper {
    db: HarnessDb,
}

impl TestMapper {
    /// Map all existing tests to code nodes
    pub fn map_all(&self) -> Result<MappingReport> {
        let mut report = MappingReport::default();
        
        // 1. Map Rust blackbox tests
        report.merge(self.map_rust_blackbox()?);
        
        // 2. Map Rust inline tests
        report.merge(self.map_rust_inline()?);
        
        // 3. Map Python tests
        report.merge(self.map_python_tests()?);
        
        Ok(report)
    }
    
    fn map_rust_blackbox(&self) -> Result<MappingReport> {
        let mut report = MappingReport::default();
        
        // Find all blackbox_*.rs files
        let test_files = glob("crates/*/tests/blackbox_*.rs")?;
        
        for test_file in test_files {
            let test_name = test_file.file_stem()
                .and_then(|s| s.to_str())
                .ok_or(Error::InvalidPath)?;
            
            // blackbox_foo_bar → foo/bar or foo_bar
            let module_name = test_name.strip_prefix("blackbox_").unwrap();
            
            // Find test functions in this file
            let test_funcs = self.find_test_functions(&test_file)?;
            
            // Find the module being tested
            let target_module = self.find_module_by_name(module_name)?;
            
            if let Some(module) = target_module {
                for test_func in test_funcs {
                    // Create edge: test_func --tests--> module
                    self.db.add_edge(
                        &test_func.node_id,
                        &module.node_id,
                        EdgeKind::Tests,
                    )?;
                    report.mapped += 1;
                }
            } else {
                report.unmapped.push(UnmappedTest {
                    test_file: test_file.clone(),
                    reason: format!("Module '{}' not found", module_name),
                });
            }
        }
        
        Ok(report)
    }
    
    fn map_rust_inline(&self) -> Result<MappingReport> {
        let mut report = MappingReport::default();
        
        // Find all #[test] functions
        let test_funcs = self.db.query(r#"
            SELECT node_id, file_path, name, parent_id
            FROM code_nodes
            WHERE language = 'rust'
              AND kind = 'rust_function'
              AND (name LIKE 'test_%' OR parent_id LIKE '%::tests')
        "#)?;
        
        for test in test_funcs {
            // Inline tests test their sibling functions
            // Find non-test functions in same module
            let siblings = self.db.query(r#"
                SELECT node_id FROM code_nodes
                WHERE parent_id = ?1
                  AND kind = 'rust_function'
                  AND name NOT LIKE 'test_%'
            "#, &[&test.parent_id])?;
            
            for sibling in siblings {
                self.db.add_edge(&test.node_id, &sibling.node_id, EdgeKind::Tests)?;
                report.mapped += 1;
            }
        }
        
        Ok(report)
    }
    
    fn map_python_tests(&self) -> Result<MappingReport> {
        let mut report = MappingReport::default();
        
        // Map test_*.py to *.py
        let test_files = glob("tests/**/test_*.py")?;
        
        for test_file in test_files {
            let test_name = test_file.file_stem()
                .and_then(|s| s.to_str())
                .ok_or(Error::InvalidPath)?;
            
            // test_foo → foo
            let module_name = test_name.strip_prefix("test_").unwrap();
            
            // Find corresponding module in engine/
            let target = self.find_python_module(module_name)?;
            
            if let Some(module) = target {
                let test_funcs = self.find_python_test_functions(&test_file)?;
                
                for test_func in test_funcs {
                    self.db.add_edge(&test_func.node_id, &module.node_id, EdgeKind::Tests)?;
                    report.mapped += 1;
                }
            } else {
                report.unmapped.push(UnmappedTest {
                    test_file: test_file.clone(),
                    reason: format!("Module '{}' not found", module_name),
                });
            }
        }
        
        Ok(report)
    }
}

#[derive(Debug, Default)]
pub struct MappingReport {
    pub mapped: usize,
    pub unmapped: Vec<UnmappedTest>,
}
```

### 6.3 Run Mapping

```bash
#!/bin/bash
# scripts/migrate-phase3-mapping.sh

set -e

echo "=== PHASE 3: Test Mapping ==="

harness map-tests

echo ""
echo "=== Mapping Report ==="
harness mapping-report

# Review unmapped tests
harness mapping-report --unmapped > unmapped_tests.txt
echo "Unmapped tests written to unmapped_tests.txt"
echo "Review and create manual mappings if needed."
```

### 6.4 Manual Mapping File

For tests that can't be auto-mapped:

```yaml
# test_mappings.yaml

manual_mappings:
  # Test file → Tested modules
  - test: tests/integration/test_full_pipeline.py
    targets:
      - engine/pipeline/compile.py
      - engine/pipeline/execute.py
      - engine/pipeline/optimize.py
    
  - test: crates/renderer-backend/tests/blackbox_gpu_integration.rs
    targets:
      - crates/renderer-backend/src/renderer.rs
      - crates/renderer-backend/src/rhi_device.rs
      - crates/renderer-backend/src/gpu_driven/mod.rs

  # Specific function mappings
  - test: tests/unit/test_math.py::test_matrix_multiply
    targets:
      - engine/math/matrix.py::multiply
```

```bash
# Apply manual mappings
harness map-tests --manual test_mappings.yaml
```

### 6.5 Expected Output

```
=== Mapping Report ===

Mapped:     11,892 test → code edges
Unmapped:      851 tests

Unmapped by category:
  - Integration tests (no single target):  423
  - Orphan tests (code deleted):            89
  - Fixture tests (test infrastructure):   201
  - Other:                                 138

Coverage by module:
  crates/renderer-backend/src/renderer.rs:    47 tests
  crates/renderer-backend/src/frame_graph/:   92 tests
  crates/renderer-backend/src/gpu_driven/:   156 tests
  engine/pipeline/:                          312 tests
  ...

Untested code (no incoming 'tests' edges):
  - src/demoscene/mod.rs (entire module)
  - src/debug/profiler.rs::Profiler::dump
  - engine/utils/deprecated.py (7 functions)
```

---

## Part 7: Phase 4 — Baseline Run

### 7.1 Run All Tests

```bash
#!/bin/bash
# scripts/migrate-phase4-baseline.sh

set -e

echo "=== PHASE 4: Baseline Test Run ==="

# Run Rust tests with JSON output
echo "Running Rust tests..."
cargo test --workspace -- --format=json 2>&1 | tee rust_tests.jsonl
harness ingest-tests rust_tests.jsonl --format cargo

# Run Python tests with JSON output
echo "Running Python tests..."
uv run pytest tests/ --json-report -q 2>&1
harness ingest-tests .report.json --format pytest

# Run shader validation
echo "Validating shaders..."
harness validate-shaders --report shader_validation.json

echo ""
echo "=== Baseline Complete ==="
harness status
```

### 7.2 State Assignment

```rust
impl BaselineRunner {
    pub fn assign_states(&self) -> Result<StateReport> {
        let mut report = StateReport::default();
        
        // For each code node
        let nodes = self.db.all_nodes()?;
        
        for node in nodes {
            // Get test results for tests that cover this node
            let test_results = self.db.get_test_results_for(&node.node_id)?;
            
            let new_state = if test_results.is_empty() {
                // No tests cover this node
                CodeState::Untouched
            } else if test_results.iter().all(|r| r.outcome == TestOutcome::Passed) {
                // All tests passed
                CodeState::TestedGreen
            } else if test_results.iter().any(|r| r.outcome == TestOutcome::Failed) {
                // At least one test failed
                CodeState::TestedRed
            } else {
                // All tests skipped
                CodeState::TestedSkipped
            };
            
            self.db.record_state_change(&node.node_id, new_state, None, Some("baseline"))?;
            
            match new_state {
                CodeState::TestedGreen => report.green += 1,
                CodeState::TestedRed => report.red += 1,
                CodeState::Untouched => report.untouched += 1,
                CodeState::TestedSkipped => report.skipped += 1,
                _ => {}
            }
        }
        
        Ok(report)
    }
}
```

### 7.3 Expected Output

```
=== Baseline Complete ===

Test Results:
  Passed:  12,102 tests
  Failed:      89 tests
  Skipped:    552 tests

Node State Distribution:
  ┌─────────────────┬────────┬──────────┐
  │ State           │ Count  │ Percent  │
  ├─────────────────┼────────┼──────────┤
  │ TESTED_GREEN    │  9,823 │   80.2%  │
  │ TESTED_RED      │    127 │    1.0%  │
  │ UNTOUCHED       │  2,143 │   17.5%  │
  │ TESTED_SKIPPED  │    144 │    1.2%  │
  └─────────────────┴────────┴──────────┘

Failing Nodes (need attention):
  - src/gpu_driven/material_table.rs::MaterialTable::lookup (3 failures)
  - engine/pipeline/optimizer.py::Optimizer::fold_constants (7 failures)
  - ...

Untested Code (consider adding tests):
  - src/demoscene/* (entire module)
  - src/debug/* (intentionally untested?)
  - ...
```

---

## Part 8: Phase 5 — Workflow Activation

### 8.1 Start Daemon

```bash
#!/bin/bash
# scripts/migrate-phase5-activate.sh

set -e

echo "=== PHASE 5: Workflow Activation ==="

# Start the harness daemon
harness daemon start \
    --watch crates/renderer-backend/src \
    --watch crates/renderer-backend/shaders \
    --watch engine \
    --db brain.db \
    --log harness.log

echo "Daemon started. PID: $(cat harness.pid)"

# Verify it's working
sleep 2
harness daemon status
```

### 8.2 CI Integration

```yaml
# .superfossil/ci.yaml (or .github/workflows/harness.yml)

name: Harness CI

on:
  push:
  pull_request:

jobs:
  harness-test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'
          
      - name: Install uv
        run: pip install uv
        
      - name: Load cached brain.db
        uses: actions/cache@v3
        with:
          path: brain.db
          key: harness-${{ hashFiles('crates/**/src/**', 'engine/**') }}
          
      - name: Parse changes
        run: |
          harness parse --changed-since HEAD~1
          
      - name: Check layouts
        run: |
          harness check-layouts || exit 1
          
      - name: Run stale tests only
        run: |
          STALE=$(harness query-tests --stale)
          if [ -n "$STALE" ]; then
            cargo test $STALE -- --format=json > rust_tests.jsonl
            harness ingest-tests rust_tests.jsonl
          fi
          
      - name: Report state
        run: |
          harness status --summary
          
      - name: Upload brain.db
        uses: actions/upload-artifact@v3
        with:
          name: brain-db
          path: brain.db
```

### 8.3 Pre-Commit Hook

```bash
#!/bin/bash
# .superfossil/hooks/pre-commit (or .git/hooks/pre-commit)

# Run harness pre-commit checks
harness pre-commit

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Pre-commit checks failed. Fix issues before committing."
    exit 1
fi

exit 0
```

---

## Part 9: Phase 6 — Contract Annotation

### 9.1 Priority Order

Annotate contracts starting with:

1. **Critical path** — Renderer hot path, GPU interfaces
2. **Cross-language** — Rust ↔ WGSL structs
3. **Complex logic** — Algorithms, state machines
4. **Public API** — Exported functions
5. **Everything else** — Gradually over time

### 9.2 Example Migration

**Before (no contracts):**

```rust
pub fn frustum_cull(objects: &[ObjectData], frustum: &Frustum) -> Vec<usize> {
    objects.iter()
        .enumerate()
        .filter(|(_, obj)| frustum.contains(&obj.aabb))
        .map(|(i, _)| i)
        .collect()
}
```

**After (with contracts):**

```rust
use trinity_contracts::*;

#[contract]
pub fn frustum_cull(objects: &[ObjectData], frustum: &Frustum) -> Vec<usize> {
    #![ensures(result.len() <= objects.len())]
    #![ensures(result.iter().all(|&i| i < objects.len()))]
    #![ensures(result.iter().all(|&i| frustum.contains(&objects[i].aabb)))]
    
    #![synth(objects: {
        len: 0..=10000,
        element: ObjectData::arbitrary()
    })]
    
    objects.iter()
        .enumerate()
        .filter(|(_, obj)| frustum.contains(&obj.aabb))
        .map(|(i, _)| i)
        .collect()
}
```

### 9.3 Layout Contracts

**Before:**

```rust
#[repr(C)]
pub struct ObjectData {
    pub transform: Mat4,
    pub aabb: AABB,
    pub mesh_index: u32,
    // ...
}
```

**After:**

```rust
use trinity_contracts::*;

#[contract]
#[layout(size = 144, align = 16)]
#[repr(C)]
pub struct ObjectData {
    #[layout(offset = 0)]
    pub transform: Mat4,
    
    #[layout(offset = 64)]
    pub aabb: AABB,
    
    #[layout(offset = 88)]
    pub mesh_index: u32,
    
    // ...
}
```

### 9.4 Contract Progress Tracking

```bash
# Check contract coverage
harness contract-coverage

# Output:
# Contract Coverage:
#   With contracts:     234 / 2,103 functions (11%)
#   With layout:         12 /    67 shared structs (18%)
#   
# Priority targets without contracts:
#   - src/renderer.rs::Renderer::render (critical path)
#   - src/gpu_driven/build_indirect.rs::* (GPU interface)
#   - src/frame_graph/compiler.rs::compile (complex)
```

---

## Part 10: Validation

### 10.1 Migration Checklist

```
□ Phase 1: Infrastructure
  □ harness crate compiles
  □ SuperSQLite connection works
  □ schema.sql applied
  □ parsers work (Rust, Python, WGSL)

□ Phase 2: Initial Parse
  □ All source files parsed
  □ Node counts match expectations
  □ Edges created (imports, calls, contains)
  □ No parse errors

□ Phase 3: Test Mapping
  □ Existing tests mapped to nodes
  □ Unmapped tests reviewed
  □ Manual mappings applied
  □ Coverage report generated

□ Phase 4: Baseline Run
  □ All tests executed
  □ Results ingested
  □ Node states assigned
  □ Failing tests identified

□ Phase 5: Workflow Activation
  □ Daemon starts successfully
  □ File changes detected
  □ Events processed
  □ State transitions work
  □ CI pipeline integrated
  □ Pre-commit hook installed

□ Phase 6: Contract Annotation
  □ Critical path annotated
  □ Shared structs have layout contracts
  □ synth schemas generated
  □ Property tests run
```

### 10.2 Smoke Tests

```bash
#!/bin/bash
# scripts/migration-smoke-test.sh

set -e

echo "=== Migration Smoke Test ==="

# 1. Change a file
echo "Test 1: File change detection"
touch crates/renderer-backend/src/renderer.rs
sleep 1
STATE=$(harness query --node "renderer.rs::*" --state)
echo "  After touch, state: $STATE"
[[ "$STATE" == *"stale"* ]] || { echo "FAIL: Should be stale"; exit 1; }
echo "  PASS"

# 2. Run tests
echo "Test 2: Test execution updates state"
cargo test renderer -- --format=json > /tmp/test.jsonl
harness ingest-tests /tmp/test.jsonl
STATE=$(harness query --node "renderer.rs::*" --state)
echo "  After tests, state: $STATE"
[[ "$STATE" == *"green"* ]] || { echo "FAIL: Should be green"; exit 1; }
echo "  PASS"

# 3. Query stale tests
echo "Test 3: Stale test query"
touch crates/renderer-backend/src/frame_graph/mod.rs
sleep 1
STALE=$(harness query-tests --stale)
echo "  Stale tests: $(echo $STALE | wc -w)"
[[ -n "$STALE" ]] || { echo "FAIL: Should have stale tests"; exit 1; }
echo "  PASS"

# 4. History query
echo "Test 4: State history"
HISTORY=$(harness history --node "renderer.rs::Renderer" --limit 5)
echo "  History entries: $(echo "$HISTORY" | wc -l)"
[[ $(echo "$HISTORY" | wc -l) -ge 2 ]] || { echo "FAIL: Should have history"; exit 1; }
echo "  PASS"

echo ""
echo "=== All Smoke Tests Passed ==="
```

### 10.3 Metrics Comparison

After migration, compare:

| Metric | Before | After |
|--------|--------|-------|
| CI run time (full) | X min | X min |
| CI run time (incremental) | X min | Y min (should be faster) |
| Test discovery time | instant | ~5 sec (parsing) |
| "What tests to run?" | Manual | Automatic |
| Cross-language bugs caught | 0 | N layout checks |

---

## Part 11: Rollback Plan

If migration fails:

```bash
#!/bin/bash
# scripts/migration-rollback.sh

echo "=== Rolling Back Migration ==="

# 1. Stop daemon
harness daemon stop || true

# 2. Remove hooks
rm -f .superfossil/hooks/pre-commit
rm -f .git/hooks/pre-commit

# 3. Archive brain.db
mv brain.db brain.db.backup.$(date +%Y%m%d)

# 4. Restore original CI
git checkout HEAD -- .superfossil/ci.yaml
git checkout HEAD -- .github/workflows/

echo "Rollback complete. Original test infrastructure restored."
```

**Keep the backup.** You can resume migration later.

---

## Appendix A: Quick Commands Reference

```bash
# Parsing
harness parse --all                    # Parse all source files
harness parse --changed-since HEAD~1   # Parse only changed files
harness parse --paths src/             # Parse specific directory

# Mapping
harness map-tests                      # Auto-map tests
harness map-tests --manual file.yaml   # Apply manual mappings
harness mapping-report                 # Show mapping status

# State
harness status                         # Overall state summary
harness status --summary               # Brief summary
harness query --state stale            # Find stale nodes
harness query --node "foo.rs::*"       # Query specific nodes
harness history --node X               # State history for node

# Testing
harness query-tests --stale            # Tests for stale code
harness ingest-tests file.json         # Ingest test results
harness check-layouts                  # Check Rust/WGSL alignment

# Contracts
harness contract-coverage              # Contract annotation status
harness verify-contracts --level 2     # Run property tests

# Daemon
harness daemon start                   # Start file watcher
harness daemon stop                    # Stop daemon
harness daemon status                  # Daemon health

# Utility
harness init --db brain.db             # Initialize database
harness stats                          # Database statistics
harness export --format json           # Export state
```

---

## Appendix B: Timeline

```
Day 1:
  □ Phase 1: Build harness crate
  □ Phase 1: Initialize database
  
Day 2:
  □ Phase 2: Parse all source files
  □ Phase 2: Verify parse results
  
Day 3-4:
  □ Phase 3: Auto-map tests
  □ Phase 3: Review unmapped
  □ Phase 3: Create manual mappings
  
Day 5:
  □ Phase 4: Run baseline tests
  □ Phase 4: Assign initial states
  □ Phase 4: Review failures
  
Day 6:
  □ Phase 5: Start daemon
  □ Phase 5: CI integration
  □ Phase 5: Smoke tests
  
Day 7:
  □ Phase 5: Documentation
  □ Phase 5: Team training
  
Ongoing:
  □ Phase 6: Contract annotation
  □ Phase 6: synth integration
  □ Phase 6: Property tests
```

---

**One week to migrate. Then the harness maintains itself.**
