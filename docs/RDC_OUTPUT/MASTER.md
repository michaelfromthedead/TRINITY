# MASTER — V2 Testing Harness

**RDC Pass:** 8 (COMPLETE)
**Last Updated:** 2026-06-05
**Status:** SCRIBE_LOOP in progress

---

## 1. Problem Statement

### 1.1 The Core Problem

**12,743+ tests exist. Test quality is unknown.**

Passing tests mean nothing if:
- They test mocks instead of real code
- Assertions are trivial (e.g., `assert!(true)`)
- They only test the happy path
- They would still pass if the code was broken
- They're testing internal implementation, not observable behavior

**Current Confidence Level:** MEDIUM — Some tests verified good (caught real bugs), but not all tests verified.

### 1.2 Evidence of Good Tests

Campaign 1 (Build Indirect GPU/CPU Parity) found a REAL BUG:
- WGSL `vec4<f32>` requires 16-byte alignment
- Rust struct had `lod_distances` at offset 104, GPU expected 112
- Result: GPU read wrong mesh indices
- Test `test_cpu_gpu_parity` caught this

**Conclusion:** GPU/CPU parity tests ARE real tests. They verify observable behavior.

### 1.3 Red Flags

1. **Suspiciously high pass rates** — 12,743 tests, 0 failures. Real codebases have flaky tests.
2. **Naming patterns** — `test_*_succeeds` without failure cases, `test_*_basic` without complex cases
3. **Mock usage** — Unknown ratio of mocks to real implementations
4. **Headless Skip Pattern** — GPU tests silently pass without GPU adapter:
   ```rust
   let harness = match Harness::new() {
       Some(h) => h,
       None => {
           eprintln!("Skipping GPU test: no adapter");
           return;  // SILENT PASS
       }
   };
   ```
   **Concern:** How many tests "pass" by skipping?

---

## 2. Adversarial Review Framework

### 2.1 Review Checklist

For each module:

**Test Reality Check:**
- [ ] Tests use real implementations, not mocks
- [ ] Tests make network/GPU/filesystem calls where appropriate
- [ ] Tests verify observable behavior, not internal state

**Assertion Quality:**
- [ ] Assertions check specific values, not just "didn't crash"
- [ ] Edge cases have explicit assertions
- [ ] Error conditions are tested with specific error types

**Mutation Testing:**
- [ ] Introduce bugs intentionally
- [ ] Verify tests catch the bugs
- [ ] If tests still pass with bugs, tests are BAD

**Coverage Analysis:**
- [ ] Critical paths have test coverage
- [ ] Error handling paths are tested
- [ ] Boundary conditions are tested

### 2.2 Module Priority Matrix

| Priority | Risk Level | Modules |
|----------|------------|---------|
| P1 | HIGH | GPU/Shader: build_indirect, object_data, frustum_cull, hiz_cull, lod_selection |
| P2 | MEDIUM | Buffer/Memory: buffer_mapping, buffer_pool, resources/buffer |
| P2 | MEDIUM | Frame Graph: mod.rs, compiler.rs, barriers.rs |
| P3 | RECENT | IK/Animation: ik_core, ik_jacobian, ik_two_bone |

---

## 3. Campaign Types

### 3.1 Mutation Testing (P1)

**Goal:** If I break this code, will a test fail?

**Mutation Categories:**

| Category | Value | Example |
|----------|-------|---------|
| Numeric | HIGH | `index * STRIDE` → `index * STRIDE + 1` |
| Data Flow | HIGH | `object.mesh_index` → `object.material_index` |
| Conditional | MEDIUM | `count > 0` → `count <= 0` |
| Return Value | MEDIUM | `is_visible()` → always true |

**Tools:** `cargo-mutants`

**Target:** 80%+ mutation score for critical paths

### 3.2 Shader Validation (P1)

**Goal:** Verify WGSL shaders match Rust struct layouts.

```rust
#[test]
fn test_shader_struct_matches_rust() {
    let wgsl = include_str!("shader.wgsl");
    // Parse WGSL, extract struct offsets
    // Compare against Rust struct offsets
    // FAIL if mismatch
}
```

**This catches the exact bug type we found** (alignment mismatch).

### 3.3 Boundary Testing (P1)

Test at edges of valid ranges:

| Boundary | Values to Test |
|----------|----------------|
| Object count | 0, 1, 2, MAX_OBJECTS, MAX+1 |
| Buffer size | 0, 1, alignment-1, alignment, MAX |
| LOD distance | exactly at threshold, epsilon below/above |
| Texture dimensions | 1x1, power-of-two, non-power-of-two, MAX |
| Draw count | 0, 1, MAX_DRAWS, overflow |

### 3.4 Negative Testing (P2)

**Goal:** Verify error handling works.

Every `Result<T, E>` should have tests for both Ok and Err paths:
```rust
#[test] fn test_create_buffer_success() { ... }
#[test] fn test_create_buffer_out_of_memory() { ... }
#[test] fn test_create_buffer_invalid_size() { ... }
```

**Audit question:** For every `?` or `.unwrap()`, is there a test?

### 3.5 State Machine Testing (P2)

Valid state transitions:
```
Buffer: UNMAPPED → MAPPING_PENDING → MAPPED_READ → UNMAPPED
Invalid: MAPPED_READ → MAPPED_WRITE (can't switch mode)

Pipeline: CREATED → BOUND → DISPATCHED → CREATED
Invalid: CREATED → DISPATCHED (must bind first)

Frame Graph: BUILDING → COMPILED → EXECUTING → BUILDING
Invalid: BUILDING → EXECUTING (must compile first)
```

### 3.6 Memory Testing (P2)

- GPU buffer leaks (created but never destroyed)
- Use-after-free (buffer destroyed, still referenced)
- Double-free
- Reference cycles in Arc<Device> relationships

**Tools:** Valgrind, AddressSanitizer, cargo-careful

### 3.7 Fuzzing (P3)

**Targets:** GLTF loaders, shader parsing, buffer data, config parsing

**Tools:** cargo-fuzz, afl.rs, honggfuzz

### 3.8 Concurrency Testing (P3)

- Multiple threads submitting to same queue
- Buffer mapping while GPU is reading
- Resource destruction while in use

**Tools:** loom, ThreadSanitizer

### 3.9 Performance Regression (P3)

Track: frame time, draw calls, GPU memory, CPU hot paths, shader compile time

Fail if regression > 10%

### 3.10 Visual Regression (P4)

Render to texture, compare against golden reference, fail if diff > threshold

**Alternative:** Compare draw command lists instead of pixels

### 3.11 API Contract Testing (P4)

```rust
#[test]
fn api_contract_object_data_size() {
    assert_eq!(std::mem::size_of::<ObjectData>(), 144);
}
```

### 3.12 Chaos Engineering (P5)

Inject: GPU device lost, allocation failures, shader timeouts

### 3.13 Compatibility Testing (P5)

Dimensions: GPU vendors, backends, driver versions, OS

---

## 4. The GPU/CPU Parity Pattern

**Gold Standard Test Pattern:**

Run same logic on CPU and GPU, compare results. Catches:
- Shader bugs
- Struct alignment issues
- Endianness problems
- Precision differences

**Every GPU module should have a parity test.**

---

## 5. Campaign Priority Matrix

| Campaign | Value | Effort | Priority |
|----------|-------|--------|----------|
| Mutation Testing | HIGH | MEDIUM | P1 |
| Shader Validation | HIGH | LOW | P1 |
| Boundary Testing | HIGH | LOW | P1 |
| Negative Testing | MEDIUM | LOW | P2 |
| State Machine | MEDIUM | MEDIUM | P2 |
| Memory Testing | HIGH | HIGH | P2 |
| Fuzzing | MEDIUM | HIGH | P3 |
| Concurrency | MEDIUM | HIGH | P3 |
| Performance Regression | MEDIUM | MEDIUM | P3 |
| Visual Regression | HIGH | HIGH | P4 |
| API Contract | LOW | LOW | P4 |
| Chaos Engineering | LOW | HIGH | P5 |
| Compatibility | HIGH | VERY HIGH | P5 |

---

## 6. Campaign Results (In Progress)

| Campaign | Target | Bugs Found | Test Quality |
|----------|--------|------------|--------------|
| #1 Build Indirect | GPU draw gen | 1 (alignment) | GOOD |
| #2 Buffer Mapping | Async readback | 2 (API misuse) | GOOD |
| #3 Frame Graph | Dead pass elim | 1 (filter bug) | GOOD |

---

## 7. Open Questions

1. How many tests skip on headless? Need count.
2. What's the actual coverage number? (cargo tarpaulin)
3. Are there property-based tests?
4. Are shaders tested, or just Rust?
5. Integration vs unit test ratio?

---

## 8. Improvement Campaigns

**Philosophy Distinction:**
- **Testing asks:** "Does it work?"
- **Improvement asks:** "Can it work BETTER?"

### 8.1 Performance Optimization (Category A)

#### Hot Path Analysis
**Goal:** Identify and optimize critical rendering path.
- Profile with `cargo flamegraph`
- Look for: >5% frame time, unexpected allocations, lock contention, cache misses

#### Allocation Elimination (P1)
**Goal:** Zero allocations in hot path.

```rust
// BAD: Allocates every frame
let commands: Vec<DrawCommand> = objects.iter().map(...).collect();

// GOOD: Reuse buffer
self.command_buffer.clear();
self.command_buffer.extend(objects.iter().map(...));
```

**Audit:** Check for `Vec::new()`, `String`, `Box::new()`, `clone()`, `collect()` in render loop.

#### Cache Optimization
**Goal:** Improve CPU cache utilization via Data-Oriented Design (SoA vs AoS).

#### Parallelization
Opportunities: frustum culling, command encoding, asset loading, shader compilation.
**Tools:** rayon, tokio, crossbeam

### 8.2 Memory Optimization (Category B)

#### Memory Footprint Reduction
- Pack booleans into bitflags
- Check for padding waste, oversized fields
- Use appropriate texture formats

#### Buffer Pooling (P2)
Reuse GPU buffers instead of create/destroy every frame.

### 8.3 Code Quality (Category C)

#### Complexity Reduction
**Targets:** Functions >100 lines, >5 parameters, nesting >4 levels, files >500 lines

#### Dead Code Elimination (P1)
**Tools:** `cargo +nightly udeps`, compiler warnings

#### Error Message Improvement (P1)
```
// BAD: "Buffer mapping failed"
// GOOD: "Buffer mapping failed: 'vertex_staging_buffer' (1024 bytes), requested MAP_READ, reason: currently mapped for writing, hint: call unmap() first"
```

#### API Ergonomics
- Avoid boolean parameters (use enums)
- Use builder pattern
- Make invalid states unrepresentable

### 8.4 Build Optimization (Category D)

#### Compile Time Reduction
- Optimize dependencies with `opt-level = 2`
- Reduce debug info
- Split large crates

#### Binary Size Reduction
```toml
[profile.release]
lto = true
codegen-units = 1
strip = true
panic = "abort"
```

### 8.5 Shader Optimization (Category E)

#### Shader Performance
- Early-out on first plane failure
- Coalesced memory access
- Avoid divergent branches

#### Shader Variant Reduction
Fewer permutations = faster compile, less memory.

### 8.6 Observability (Category F)

#### Logging
Use structured logging with levels: ERROR, WARN, INFO, DEBUG, TRACE

#### Metrics
Expose: counters, gauges, histograms for frame time, draw calls, memory

### 8.7 Dependency Optimization (Category G)

**Tools:** `cargo tree`, `cargo udeps`, `cargo outdated`, `cargo audit`

---

## 9. Improvement Priority Matrix

| Campaign | Impact | Effort | Priority |
|----------|--------|--------|----------|
| Allocation Elimination | HIGH | MEDIUM | P1 |
| Hot Path Analysis | HIGH | LOW | P1 |
| Dead Code Elimination | MEDIUM | LOW | P1 |
| Error Message Improvement | MEDIUM | LOW | P1 |
| Buffer Pooling | HIGH | MEDIUM | P2 |
| Parallelization | HIGH | HIGH | P2 |
| Shader Performance | HIGH | HIGH | P2 |
| Compile Time Reduction | MEDIUM | MEDIUM | P2 |
| Cache Optimization | MEDIUM | HIGH | P3 |
| API Ergonomics | MEDIUM | MEDIUM | P3 |
| Complexity Reduction | LOW | MEDIUM | P3 |
| Dependency Audit | LOW | LOW | P4 |
| Binary Size Reduction | LOW | LOW | P4 |
| Logging Improvement | LOW | LOW | P4 |

---

## 10. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Frame time (1K objects) | ? ms | <2 ms |
| Frame time (100K objects) | ? ms | <16 ms |
| Allocations per frame | ? | 0 |
| Build time (incremental) | ? s | <5 s |
| Binary size (release) | ? MB | <10 MB |
| GPU memory (1K objects) | ? MB | <100 MB |

---

## 11. QA ↔ Improvement Loop

| Improvement Campaign | Related QA Campaign |
|---------------------|---------------------|
| Hot Path Analysis | Performance Regression Testing |
| Allocation Elimination | Memory Testing |
| Parallelization | Concurrency Testing |
| Shader Performance | Visual Regression |
| Error Messages | Negative Testing |

**The loop:**
1. QA campaign finds issue
2. Improvement campaign fixes it
3. QA campaign verifies fix
4. QA campaign prevents regression

---

## 12. Tooling Requirements

**Profiling:**
- `cargo flamegraph` — CPU profiling
- `dhat` / `heaptrack` — Allocation profiling
- `perf` — Low-level CPU metrics
- RenderDoc — GPU profiling

**Analysis:**
- `cargo bloat` — Binary analysis
- `cargo udeps` — Unused dependencies
- `cargo outdated` — Dependency updates
- `cargo audit` — Security vulnerabilities
- `cargo build --timings` — Build profiling

---

---

# PART II: V2 ARCHITECTURE

---

## 13. Central Insight

**Code and tests are not two artifacts to keep in sync. They are two facets of the same stateful entity.**

Traditional model:
```
CODE ~~~~~~~ TEST
  ↑    "sync"   ↑
  └── can drift─┘
```

V2 model: A code unit has facets:
- **Implementation** — what it does
- **Contract** — what it promises
- **State** — lifecycle tracking
- **Verification** — derived from contract

---

## 14. The Two-Graph Model

### 14.1 Structure 1: Dependency Graph (DAG)

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│ module  │─────▶│ module  │─────▶│ module  │
│   A     │      │   B     │      │   C     │
└─────────┘      └─────────┘      └─────────┘
```

- **Nodes:** Code units (crate, module, file, function)
- **Edges:** "depends on" / "calls" / "imports"
- **Type:** DAG (directed acyclic graph)
- **Changes:** Only when code structure changes

### 14.2 Structure 2: Statechart (per Node)

```
┌──────────────────────────────────────────────┐
│                   KNOWN                       │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ TESTED  │──▶│  STALE  │──▶│ TESTED  │    │
│  │ GREEN   │   │         │   │   RED   │    │
│  └─────────┘   └─────────┘   └─────────┘    │
└──────────────────────────────────────────────┘
              │                 
              ▼                 
       ┌─────────────┐    ┌─────────────┐
       │ QA_APPROVED │    │ QUARANTINED │
       └─────────────┘    └─────────────┘
```

- **Type:** FSM (finite state machine)
- **Instance:** One per code unit
- **Changes:** On every event

### 14.3 The Magic: Graph + Statechart = Propagation

When B changes:
- B → `CHANGED`
- C (depends on B) → `STALE_DIRECT`
- D (depends on C) → `STALE_TRANSITIVE`

---

## 15. Code States

```rust
pub enum CodeState {
    // === INITIAL ===
    Unknown,        // Never analyzed
    Untouched,      // Analyzed but never tested
    
    // === TESTED ===
    TestedGreen,    // Tests pass
    TestedRed,      // Tests fail
    
    // === STALE ===
    StaleDirect,    // This code changed
    StaleTransitive,// A dependency changed
    StaleDeep,      // A dep of a dep changed
    
    // === QA ===
    QaApproved,     // Passed adversarial review
    Quarantined,    // Known broken, isolated
}
```

**Severity order:** TestedGreen (0) < QaApproved < Untouched < Unknown < StaleDeep < StaleTransitive < StaleDirect < TestedRed < Quarantined (7)

---

## 16. Code Events

```rust
pub enum CodeEvent {
    // === CHANGE ===
    SourceChanged,          // Any change
    SignatureChanged,       // Public API changed (propagates)
    BodyChanged,            // Internal only (no propagation)
    LayoutChanged,          // Memory layout (propagates to GPU mirrors!)
    
    // === DEPENDENCY ===
    DependencyChanged,      // Direct dep changed
    TransitiveDependencyChanged,  // N hops away
    
    // === TEST ===
    TestsPassed,
    TestsFailed,
    TestsSkipped,
    
    // === QA ===
    QaApproved,
    Quarantine,
    Release,
}
```

**Key insight:** `BodyChanged` does NOT propagate. If only internals change, dependents don't go stale (assuming tests are behavioral, not white-box).

---

## 17. Unified Code Substrate (7-Layer Architecture)

### Layer 0: File System
- `notify` crate for watching file changes
- Language detection from extension (.rs, .py, .wgsl)

### Layer 1: AST Parsers (Per Language)

| Language | Fast Parser | Full Parser | Purpose |
|----------|-------------|-------------|---------|
| Rust | tree-sitter | syn | Types, generics, macros |
| Python | tree-sitter | rustpython | Decorators, signatures |
| WGSL | tree-sitter | naga | **Struct layouts with offsets** |

### Layer 2: Unified Code Graph
```rust
pub struct CodeGraph {
    nodes: SlotMap<NodeId, CodeNode>,
    edges: SlotMap<EdgeId, CodeEdge>,
    // Indexes for fast lookup
    nodes_by_file: HashMap<PathBuf, Vec<NodeId>>,
    nodes_by_name: HashMap<QualifiedName, NodeId>,
    nodes_by_language: HashMap<Language, Vec<NodeId>>,
}
```

### Layer 3: Edge Kinds

| Edge | Meaning |
|------|---------|
| Contains | Parent has child (file→function) |
| Imports | A imports B |
| Calls | A calls B |
| Inherits | A inherits from B |
| Tests | Test A tests unit B |
| **MirrorsLayout** | WGSL struct mirrors Rust struct |
| UsesShader | Rust function uses WGSL shader |
| PyO3Call | Rust FFI calls Python |

### Layer 4: Cross-Language Boundary Detection
- Detect `#[pyfunction]`, `#[pyclass]` for PyO3
- Match Rust `#[repr(C)]` structs to WGSL structs by name
- **Validate layout match!** (catches alignment bugs)

### Layer 5: Statechart Binding (superstate)
- Each node gets a `CodeStateMachine` instance
- Uses superstate's `StateTree` for hierarchy

### Layer 6: Propagation Engine
- BFS traversal from changed node
- Marks dependents as STALE
- **Layout changes propagate to GPU mirrors**

### Layer 7: Query Interface
- `nodes_in_state(state)` — all nodes in state
- `needs_testing()` — all stale or untested
- `layout_mismatches()` — Rust↔WGSL alignment bugs
- `aggregate_state(parent)` — worst state of children

---

## 18. The Unified Code Unit

```rust
struct CodeUnit {
    // IMPLEMENTATION FACET
    source: Source,
    dependencies: Vec<UnitId>,
    signature: Signature,
    
    // CONTRACT FACET
    requires: Vec<Predicate>,    // Preconditions
    ensures: Vec<Predicate>,     // Postconditions
    invariants: Vec<Predicate>,  // Always-true
    properties: Vec<Property>,   // Algebraic laws
    
    // STATE FACET
    state: CodeState,
    history: Vec<(Timestamp, CodeState, CodeEvent)>,
    
    // VERIFICATION FACET — derived, not stored
    // verify() — generated from contract
    // tests() — generated from properties
}
```

### 18.1 Verification Is Derived

```rust
impl CodeUnit {
    fn verify(&self) -> VerificationResult {
        // 1. Check preconditions are satisfiable
        // 2. Check postconditions follow from pre + impl
        // 3. Check invariants hold across all reachable states
        // 4. Check properties via property-based testing
        //    (synth generates inputs satisfying `requires`)
    }
}
```

**Tests are not written separately — they're generated from the contract.**

---

## 19. superstate Integration

### 19.1 Hierarchical State Model

```rust
let tree = StateTreeBuilder::<CodeEvent>::new("crate")
    .root("crate", StateType::And)
        .child("gpu_driven", StateType::And)
            .child("build_indirect", StateType::And)
                .basic_child("cpu_build_indirect").end()
            .end()
        .end()
    .done()
    .build();
```

### 19.2 CTL Model Checking

```rust
// "No code reaches production without being GREEN"
let prop = CTLProperty::AG(
    Predicate::Implies(
        Predicate::InProduction,
        Predicate::StateIs(CodeState::QaApproved)
    )
);

verify_ctl(&prop, &reachability);
```

### 19.3 Coverage = State Coverage

Traditional: "80% of lines executed"
V2: "100% of reachable states verified"

---

## 20. synth Integration for V2

synth generates test inputs from contracts:

```rust
// Define schema from function signature
let schema = synth::Schema::Object {
    a: synth::Schema::Number { range: i32::MIN..i32::MAX },
    b: synth::Schema::Number { range: i32::MIN..i32::MAX },
};

// Add constraints from `requires`
let constraints = unit.requires.to_synth_constraints();

// Generate inputs satisfying preconditions
let inputs = synth::generate(schema, constraints, 1000);

// Check postconditions for all generated inputs
for input in inputs {
    let result = (unit.source)(input);
    for ens in &unit.ensures {
        assert!(ens.check(&input, &result));
    }
}
```

---

## 21. SuperSQLite Persistence Layer

### 21.1 Why SuperSQLite?

Single `brain.db` file with 15+ extensions:
- **Embedded** — In-process, ~0.1ms latency
- **Single file** — Copy = backup
- **Unified** — Graph, vector, events, cache in one system

### 21.2 Extension Mapping

| Extension | Equivalent | Our Use Case |
|-----------|------------|--------------|
| supersqlite_graph | Neo4j | Code dependency graph |
| supersqlite_streams | Kafka | Event log (CodeEvents) |
| supersqlite_bitemporal | SQL:2011 | State history queries |
| supersqlite_vector | Pinecone | Code embeddings search |
| supersqlite_memory | Redis | Fast state cache |
| supersqlite_timeseries | TimescaleDB | Metrics over time |

### 21.3 Core Tables

```sql
code_nodes       -- Every code unit in the graph
code_edges       -- Dependencies, test relationships
code_events      -- Append-only event log
code_state_history -- State transitions (bitemporal)
code_contracts   -- Extracted contract specifications
struct_layouts   -- Rust/WGSL layout info (alignment bugs!)
```

---

## 22. Workflow Integration (Event Engine)

### 22.1 Event Sources

| Source | Mechanism | Events |
|--------|-----------|--------|
| File Watcher | notify crate | File changes |
| Superfossil | commit hooks | Commits |
| CI Pipeline | test results | Test pass/fail |
| Manual | CLI/API | Force state |

### 22.2 Processing Pipeline

```
EVENT SOURCES → EVENT INGESTER → SUPERSQLITE
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
STATE UPDATER   PROPAGATION    NOTIFICATION
                  ENGINE         SERVICE
```

### 22.3 HarnessDaemon

Long-running background service:
- Watches filesystem
- Processes events
- Updates state machine
- Publishes notifications

---

## 23. Contract Language

### 23.1 Design Principles

| Principle | Decision |
|-----------|----------|
| Ceremony | Lightweight Rust attributes |
| Language | Rust-first (Python/WGSL later) |
| Primary Use | synth integration |
| Verification | All 4 levels |

### 23.2 Attribute Syntax

```rust
#[contract]
pub fn divide(a: i32, b: i32) -> i32 {
    #![requires(b != 0)]
    #![ensures(result * b == a)]
    a / b
}

#[contract]
#[layout(size = 144, align = 16)]
#[repr(C)]
pub struct ObjectData { ... }

#[contract]
pub fn add(a: i32, b: i32) -> i32 {
    #![property(commutative)]
    #![property(associative)]
    #![property(identity = 0)]
    a + b
}
```

### 23.3 Four Verification Levels

| Level | Tool | Generated From |
|-------|------|----------------|
| Runtime | `debug_assert!` | requires/ensures |
| Property | proptest/synth | properties + synth schema |
| Static | kani/miri | requires/ensures (symbolic) |
| Formal | creusot/prusti | full contract |

### 23.4 synth Schema Derivation

Contracts become synth schemas:
```rust
#![requires(x >= 0.0)]
#![requires(x <= 1000.0)]
```
→
```json
{ "x": { "range": { "low": 0.0, "high": 1000.0 } } }
```

---

## 24. Migration Guide

### 24.1 Current State

- **12,743 tests** (Rust: 2,200, Python: 8,500, Benchmarks: 200)
- No dependency tracking
- Run all tests on change
- No contract specification

### 24.2 Target State

- Code graph with statechart per unit
- Full dependency DAG
- Run only stale tests
- Contracts → synth → tests
- Rust ↔ Python ↔ WGSL unified

### 24.3 Migration Phases (6 Phases, ~1 Week)

| Phase | Duration | Action |
|-------|----------|--------|
| 1 Infrastructure | 1-2 days | Build harness crate, SuperSQLite |
| 2 Initial Parse | 1 day | Parse all code, build graph |
| 3 Test Mapping | 2-3 days | Map tests to nodes |
| 4 Baseline Run | 1 day | Run all tests, mark states |
| 5 Workflow Activation | 1 day | Start daemon, CI integration |
| 6 Contract Annotation | Ongoing | Add contracts to code |

### 24.4 Migration is ONE STEP

**Big-bang, not incremental.** V1 and V2 cannot coexist — dependency graph must be complete to work.

---

# PART III: V1 INFRASTRUCTURE

---

## 25. V1 Document Relationships

```
V1_ADVERSARIAL_REVIEW          V1_IMPROVEMENT_CAMPAIGNS
         │                              │
         │ "What to test"               │ "What to improve"
         │                              │
         └────────────┬─────────────────┘
                      │
                      ▼
           V1_TESTING_TOOLS
                      │
                      │ "How to do it"
                      │
                      ▼
              Testing Harness
              Custom Infrastructure
              Automated Pipelines
```

---

## 26. V1 Tool Ecosystem

### 14.1 Python → Rust Mapping

| Category | Python | Rust |
|----------|--------|------|
| Testing | `pytest` | `cargo test` |
| Coverage | `coverage.py` | `cargo-tarpaulin` |
| Property | `hypothesis` | `proptest` |
| Mocking | `unittest.mock` | `mockall` |
| Benchmarking | `pytest-benchmark` | `criterion` |
| Mutation | `mutmut` | `cargo-mutants` |
| Fuzzing | `atheris` | `cargo-fuzz` |
| Profiling | `cProfile` | `cargo flamegraph` |

### 14.2 Rust-Specific Tools

- **Snapshot testing:** `cargo-insta` — compare against saved "golden" output
- **Fixtures:** `rstest` — `#[fixture]`, `#[rstest]` with parameterized tests
- **Async testing:** `#[tokio::test]`
- **Doc tests:** Run with `cargo test --doc`

---

## 27. GPU-Specific Testing

### 15.1 GpuTestHarness

```rust
pub struct GpuTestHarness {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuTestHarness {
    pub fn new() -> Option<Self> { /* returns None if no GPU */ }
    pub fn run_compute<T: Pod>(&self, shader: &str, input: &[T]) -> Vec<T>;
}
```

### 15.2 CPU/GPU Parity Macro

```rust
parity_test!(
    test_frustum_cull_parity,
    cpu_frustum_cull,
    include_str!("shaders/frustum_cull.wgsl"),
    generate_test_objects(1000)
);
```

### 15.3 Shader Struct Validator

```rust
pub fn validate_struct_layout<T>(wgsl_source: &str, struct_name: &str);
```
**Catches Rust ↔ WGSL alignment mismatches** (the exact bug type we found).

---

## 28. Synth — Declarative Test Data Generator (V1)

**Location:** `/home/user/dev/USER/PROJECTS_VOID/synth/`

### 16.1 Why Synth?

| Feature | Description | Use Case |
|---------|-------------|----------|
| Schema-based | Define shape in JSON | Generate valid GPU objects |
| Constraint solving | Specify targets, synth computes params | "95% visible, avg LOD 2.5" |
| G-F-S loop | Generate → Falsify → Shrink | Property testing pattern |
| Gaussian Copula | Correlated multi-field generation | Spatial LOD correlation |
| Pipeline Models | Linear, DecisionTree, Logistic | Predict/invert constraints |

### 16.2 The G-F-S Loop

```
SCHEMA → CONSTRAINTS → GENERATE → FALSIFY → SHRINK
   │          │           │          │         │
   │          │           │          │         └─ Minimize failing case
   │          │           │          └─ Check property violation
   │          │           └─ Sample from schema
   │          └─ Population targets
   └─ Data shape definition
```

### 16.3 ObjectData Schema Example

```json
{
  "type": "array",
  "length": { "range": { "low": 100, "high": 100000 } },
  "content": {
    "mesh_index": { "type": "u32", "range": { "low": 0, "high": 99 } },
    "position": { "type": "array", "content": { "distribution": "normal" } },
    "lod_distances": { "type": "array", "length": 4 }
  }
}
```

### 16.4 Constraint File Example

```json
{
  "constraints": [
    { "type": "row_count", "collection": "objects", "count": 100000 },
    { "type": "percentage", "field": "flags", "target": 0.70 },
    { "type": "mean", "field": "mesh_index", "target": 50.0 }
  ]
}
```

**Translation:** "100K objects, ~70% visible, mesh indices averaging 50."

### 16.5 Integration Patterns

| Pattern | Speed | Flexibility |
|---------|-------|-------------|
| CLI Integration | Slow | High — full synth features |
| Library Integration | Medium | High — Rust API |
| Pre-generated Fixtures | Fast | Low — committed JSON files |

### 16.6 When to Use Synth

| Scenario | Use Synth? |
|----------|------------|
| Unit test edge case | NO — hand-craft |
| 100K object stress test | YES |
| Specific frustum test | NO — hand-craft |
| Statistical properties | YES |
| Regression test for bug | NO — capture exact case |
| Fuzz with valid data | YES — structured fuzzing |

---

## 29. V1 Testing Harness Architecture

### 17.1 Directory Structure

```
crates/renderer-backend/
├── tests/
│   ├── common/              # Shared utilities
│   │   ├── gpu_harness.rs
│   │   ├── fixtures.rs
│   │   ├── synth_bridge.rs
│   │   └── assertions.rs
│   ├── blackbox_*.rs        # Integration tests (external API)
│   └── whitebox_*.rs        # Internal tests (implementation)
├── test_schemas/            # Synth schemas
├── test_fixtures/           # Pre-generated data
├── benches/                 # Criterion benchmarks
└── fuzz/                    # Fuzz targets
```

### 17.2 Custom Assertions

```rust
assert_float_array_eq(actual, expected, epsilon);
assert_buffer_eq(device, queue, buffer, expected);
assert_draw_commands_eq(actual, expected);  // order-independent
```

### 17.3 Test Scenes

```rust
pub enum TestScene {
    Empty,
    SingleObject,
    ThousandObjects,
    StressTest,  // 100K objects
}
```

---

## 30. CI/CD Integration

### 18.1 GitHub Actions Jobs

- **test:** `cargo test --all-features`
- **coverage:** `cargo tarpaulin` → codecov
- **gpu-tests:** Self-hosted GPU runner
- **mutation:** `cargo mutants`

### 18.2 Pre-commit Hooks

```bash
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test --lib
```

---

## 31. Campaign → Tool Mapping

| QA Campaign | Primary Tool | Synth Role |
|-------------|--------------|------------|
| Mutation Testing | `cargo-mutants` | Generate diverse inputs |
| Boundary Testing | `proptest` | Constraint-driven edge cases |
| Fuzzing | `cargo-fuzz` | Structured fuzzing |
| Coverage | `cargo-tarpaulin` | High-coverage inputs |
| Statistical Validation | **synth** | Target distributions |
| Stress Testing | **synth** | 100K+ objects |

| Improvement Campaign | Primary Tool | Synth Role |
|---------------------|--------------|------------|
| Hot Path Analysis | `cargo flamegraph` | Realistic workloads |
| Allocation Elimination | `dhat` | Allocation-heavy cases |
| Performance Regression | `criterion` | Consistent benchmark data |
| Shader Performance | RenderDoc | GPU workloads |

---

*Principles: Trust but verify. Test the tests. Move fast, measure everything.*
