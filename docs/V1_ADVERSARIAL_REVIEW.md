# V1 Adversarial QA Review - renderer-backend

**Started:** 2026-06-04
**Status:** CAMPAIGN IN PROGRESS
**Verdict:** UNKNOWN - Tests run, quality unverified

## Executive Summary

We have **12,743+ tests that pass**. We do NOT know if they are good tests.

Passing tests mean nothing if:
- They test mocks instead of real code
- Assertions are trivial (e.g., `assert!(true)`)
- They only test the happy path
- They would still pass if the code was broken
- They're testing internal implementation, not observable behavior

## Campaign Log

### Campaign 1: Build Indirect GPU/CPU Parity (2026-06-04)

**Target:** `src/gpu_driven/build_indirect.rs` + `shaders/build_indirect.wgsl`

**Finding:** REAL BUG FOUND
- WGSL `vec4<f32>` requires 16-byte alignment
- Rust struct had `lod_distances` at offset 104
- GPU shader expected it at offset 112
- Result: GPU read wrong mesh indices for objects

**Test Quality Assessment:**
- `test_cpu_gpu_parity` - GOOD: Actually compares CPU vs GPU output
- `test_mixed_lod_levels` - GOOD: Tests real LOD selection logic
- Other GPU tests - GOOD: Actually run compute shaders and read back results

**Verdict:** These tests are REAL tests. They caught a real bug.

---

## Adversarial Review Checklist

For each module, we must verify:

### 1. Test Reality Check
- [ ] Tests use real implementations, not mocks
- [ ] Tests make network/GPU/filesystem calls where appropriate
- [ ] Tests verify observable behavior, not internal state

### 2. Assertion Quality
- [ ] Assertions check specific values, not just "didn't crash"
- [ ] Edge cases have explicit assertions
- [ ] Error conditions are tested with specific error types

### 3. Mutation Testing
- [ ] Introduce bugs intentionally
- [ ] Verify tests catch the bugs
- [ ] If tests still pass with bugs, tests are BAD

### 4. Coverage Analysis
- [ ] Critical paths have test coverage
- [ ] Error handling paths are tested
- [ ] Boundary conditions are tested

---

## Modules To Review

### Priority 1: GPU/Shader Code (High Risk)
| Module | Tests Exist | Tests Verified | Status |
|--------|-------------|----------------|--------|
| `gpu_driven/build_indirect.rs` | Yes | YES | VERIFIED GOOD |
| `gpu_driven/object_data.rs` | Yes | Partial | NEEDS REVIEW |
| `gpu_driven/frustum_cull_pipeline.rs` | Yes | No | NEEDS REVIEW |
| `gpu_driven/hiz_cull_pipeline.rs` | Yes | No | NEEDS REVIEW |
| `gpu_driven/lod_selection.rs` | ? | No | NEEDS REVIEW |
| `shaders/*.wgsl` | Integration | Partial | NEEDS REVIEW |

### Priority 2: Buffer/Memory Management (Medium Risk)
| Module | Tests Exist | Tests Verified | Status |
|--------|-------------|----------------|--------|
| `buffer_mapping.rs` | Yes | Partial | Fixed API bugs |
| `buffer_pool.rs` | Yes | Partial | Fixed Arc issues |
| `resources/buffer.rs` | Yes | No | NEEDS REVIEW |

### Priority 3: Frame Graph (Medium Risk)
| Module | Tests Exist | Tests Verified | Status |
|--------|-------------|----------------|--------|
| `frame_graph/mod.rs` | Yes | Partial | Fixed dead pass elimination |
| `frame_graph/compiler.rs` | Yes | No | NEEDS REVIEW |
| `frame_graph/barriers.rs` | Yes | No | NEEDS REVIEW |

### Priority 4: IK/Animation (Recently Added)
| Module | Tests Exist | Tests Verified | Status |
|--------|-------------|----------------|--------|
| `ik_core.rs` | Yes | No | NEEDS REVIEW |
| `ik_jacobian.rs` | Yes | No | NEEDS REVIEW |
| `ik_two_bone.rs` | Yes | No | NEEDS REVIEW |

---

## Red Flags To Investigate

### 1. Suspiciously High Pass Rates
- 12,743 tests, 0 failures is suspicious
- Real codebases have flaky tests, edge case failures
- Either tests are excellent OR tests are too easy

### 2. Test Naming Patterns
- `test_*_succeeds` - Does it test failure cases too?
- `test_*_basic` - What about complex cases?
- `test_default_*` - What about non-default configs?

### 3. Mock Usage
- How many tests use mocks vs real implementations?
- Are mocks testing the mock or the code?

### 4. GPU Tests Without GPU
- Do tests skip on headless systems?
- Are skipped tests silently passing?

---

## Adversarial Test Protocol

For each module under review:

```bash
# Step 1: Find tests
grep -r "fn test_" src/MODULE.rs tests/blackbox_MODULE.rs

# Step 2: Check for trivial assertions
grep -r "assert!(true)" tests/
grep -r "assert_eq!(1, 1)" tests/

# Step 3: Check for mock usage
grep -r "Mock\|mock\|Stub\|stub\|Fake\|fake" tests/

# Step 4: Mutation test - introduce a bug
# Change: `mesh_index: u32` to `mesh_index: u32 + 1`
# Run tests - if they pass, tests are BAD

# Step 5: Check error path coverage
grep -r "Err(" src/MODULE.rs | wc -l  # Count error returns
grep -r "test.*error\|test.*fail" tests/ | wc -l  # Count error tests
```

---

## Mutation Testing Campaign - Strategic Analysis

### The Core Question

> If I break this code, will a test fail?

If the answer is "no," then the tests are theater. They provide false confidence.

### Why Mutation Testing Matters Here

This is a **GPU renderer**. Bugs manifest as:
- Incorrect pixels on screen (visual corruption)
- Wrong draw calls issued (missing/duplicate geometry)
- Memory corruption (crashes, undefined behavior)
- Silent data corruption (looks fine, data is wrong)

The scariest bugs are **silent corruption**. The struct alignment bug we found was exactly this:
- Code ran without crashing
- Tests passed
- GPU read wrong data
- Would have caused incorrect rendering in production

### Mutation Categories for renderer-backend

#### Category A: Numeric Mutations (HIGH VALUE)
```rust
// Original
let index = object_idx * STRIDE;

// Mutant 1: Off-by-one
let index = object_idx * STRIDE + 1;

// Mutant 2: Wrong multiplier  
let index = object_idx * (STRIDE + 1);

// Mutant 3: Zero
let index = 0;
```

**Why this matters:** GPU shaders index into buffers. Off-by-one = read wrong object. Tests MUST catch this.

#### Category B: Conditional Mutations (MEDIUM VALUE)
```rust
// Original
if visible_count > 0 { dispatch(); }

// Mutant 1: Flip condition
if visible_count <= 0 { dispatch(); }

// Mutant 2: Remove condition
dispatch();

// Mutant 3: Change threshold
if visible_count > 1 { dispatch(); }
```

**Why this matters:** Edge cases at boundaries. Empty scenes, single object, max objects.

#### Category C: Data Flow Mutations (HIGH VALUE)
```rust
// Original
let mesh_id = object.mesh_index;

// Mutant 1: Use wrong field
let mesh_id = object.material_index;

// Mutant 2: Hardcode
let mesh_id = 0;

// Mutant 3: Use parameter instead of field
let mesh_id = object_idx as u32;
```

**Why this matters:** This is EXACTLY what the alignment bug caused. Wrong field read.

#### Category D: Return Value Mutations (MEDIUM VALUE)
```rust
// Original
fn is_visible(&self) -> bool {
    self.flags & VISIBLE != 0
}

// Mutant 1: Always true
fn is_visible(&self) -> bool { true }

// Mutant 2: Always false
fn is_visible(&self) -> bool { false }

// Mutant 3: Inverted
fn is_visible(&self) -> bool {
    self.flags & VISIBLE == 0
}
```

### Mutation Testing Strategy

#### Phase 1: Manual Spot Checks (NOW)
Pick 5 critical functions. Manually mutate them. Run tests. Log results.

**Targets:**
1. `cpu_build_indirect()` - Core draw generation
2. `ObjectData::mesh_index` access - Field we just fixed
3. `MeshData::index_count_for_lod()` - LOD selection
4. `compile()` in frame_graph - Pass elimination
5. `BufferReadback::wait_and_get_data()` - GPU readback

#### Phase 2: Automated Mutation (LATER)
Use `cargo-mutants` or similar tool to systematically mutate codebase.

```bash
# Install
cargo install cargo-mutants

# Run on specific module
cargo mutants --package renderer-backend -- gpu_driven/build_indirect.rs

# Interpret results
# - "caught" = test failed = GOOD
# - "missed" = test passed = BAD (test doesn't catch this bug)
# - "timeout" = infinite loop = test has coverage but slow
```

#### Phase 3: Mutation Score Tracking
Track mutation score per module:

| Module | Mutations | Caught | Missed | Score |
|--------|-----------|--------|--------|-------|
| build_indirect.rs | ? | ? | ? | ?% |
| object_data.rs | ? | ? | ? | ?% |
| frame_graph/mod.rs | ? | ? | ? | ?% |

**Target:** 80%+ mutation score for critical paths.

### Specific Concerns for This Codebase

#### 1. GPU/CPU Parity Tests Are Gold
The `test_cpu_gpu_parity` pattern is excellent. It runs the same logic on CPU and GPU and compares results. This catches:
- Shader bugs
- Struct alignment issues
- Endianness problems
- Precision differences

**Every GPU module should have a parity test.**

#### 2. "Headless Skip" Is Suspicious
Many GPU tests have:
```rust
let harness = match Harness::new() {
    Some(h) => h,
    None => {
        eprintln!("Skipping GPU test: no adapter");
        return;  // <-- SILENT PASS
    }
};
```

This means in CI without GPU, these tests **silently pass without running**.

**Question:** How many tests are "passing" by skipping?

#### 3. LOD Distance Tests
LOD selection is based on distance thresholds. Tests should verify:
- Object at distance 99 uses LOD 0
- Object at distance 101 uses LOD 1
- Boundary exactly at threshold

**Concern:** Are boundaries tested, or just "somewhere in the middle"?

#### 4. Buffer Size Edge Cases
Buffer pools, mapping, staging - all have edge cases:
- Zero-size buffer
- Max-size buffer
- Buffer larger than GPU limit
- Unaligned sizes

**Concern:** We fixed a zero-size mapping bug. Were there tests for this? Why didn't they catch it?

### Open Questions

1. **How many tests skip on headless?** Need to count.
2. **What's the actual coverage number?** Haven't run `cargo tarpaulin`.
3. **Are there property-based tests?** For randomized input testing.
4. **Are shaders tested, or just Rust?** WGSL has its own bugs.
5. **Integration vs unit test ratio?** Too many unit tests can miss integration bugs.

---

## Next Actions

1. **IMMEDIATE:** Manual mutation test on `cpu_build_indirect()` - verify tests catch off-by-one
2. **TODAY:** Count headless-skip tests - how many pass without GPU?
3. **THIS WEEK:** Run `cargo mutants` on gpu_driven/ modules
4. **ONGOING:** Log findings in this document

---

## Campaign Results

| Campaign | Target | Bugs Found | Test Quality |
|----------|--------|------------|--------------|
| #1 Build Indirect | GPU draw gen | 1 (alignment) | GOOD |
| #2 Buffer Mapping | Async readback | 2 (API misuse) | GOOD |
| #3 Frame Graph | Dead pass elim | 1 (filter bug) | GOOD |
| #4 | | | |

---

---

## QA Campaign Types

Beyond mutation testing, here are distinct campaign types for comprehensive coverage:

---

### Campaign Type: FUZZING
**Goal:** Throw garbage at the system, see what breaks.

**Targets:**
- GLTF/scene file loaders
- Shader source parsing
- Buffer data interpretation
- Configuration parsing

**Tools:** `cargo-fuzz`, `afl.rs`, `honggfuzz`

**What it catches:**
- Buffer overflows
- Panic on malformed input
- Integer overflows
- Infinite loops on crafted input

**Specific concerns for renderer:**
```rust
// What happens if GLTF says mesh has 1M vertices but buffer has 100?
// What happens if shader has unclosed brace?
// What happens if config JSON is truncated?
```

**Campaign questions:**
- [ ] Does loader validate before trusting file data?
- [ ] Are there size limits on allocations from untrusted input?
- [ ] Does bad input crash, panic, or return error gracefully?

---

### Campaign Type: BOUNDARY TESTING
**Goal:** Test at the edges of valid ranges.

**Boundaries to test:**

| Boundary | Values to Test |
|----------|----------------|
| Object count | 0, 1, 2, MAX_OBJECTS, MAX+1 |
| Buffer size | 0, 1, alignment-1, alignment, MAX |
| LOD distance | exactly at threshold, epsilon below/above |
| Texture dimensions | 1x1, power-of-two, non-power-of-two, MAX |
| Draw count | 0, 1, MAX_DRAWS, overflow |
| Workgroup dispatch | 0, 1, exactly fills, needs padding |

**What it catches:**
- Off-by-one errors
- Division by zero
- Overflow/underflow
- Fence-post errors

**Example test we should have:**
```rust
#[test]
fn test_lod_at_exact_threshold() {
    let threshold = 100.0;
    assert_eq!(select_lod(99.999), LOD_0);
    assert_eq!(select_lod(100.0), LOD_1);   // AT threshold
    assert_eq!(select_lod(100.001), LOD_1);
}
```

---

### Campaign Type: NEGATIVE TESTING
**Goal:** Verify error handling works.

**Negative cases to test:**
- Invalid GPU adapter (no GPU)
- Out of memory allocation
- Shader compilation failure
- Buffer mapping failure
- Texture creation failure
- Pipeline creation failure
- Invalid bind group layout

**What it catches:**
- Unhandled errors causing panic
- Wrong error type returned
- Error messages that are useless
- Error recovery that corrupts state

**Pattern to verify:**
```rust
// Every Result<T, E> should have tests for both Ok and Err paths
fn create_buffer(...) -> Result<Buffer, BufferError>;

#[test] fn test_create_buffer_success() { ... }
#[test] fn test_create_buffer_out_of_memory() { ... }
#[test] fn test_create_buffer_invalid_size() { ... }
#[test] fn test_create_buffer_unsupported_usage() { ... }
```

**Audit question:** For every `?` or `.unwrap()`, is there a test?

---

### Campaign Type: STATE MACHINE TESTING
**Goal:** Verify valid state transitions, reject invalid ones.

**State machines in renderer:**

```
Buffer States:
  UNMAPPED -> MAPPING_PENDING -> MAPPED_READ -> UNMAPPED
  UNMAPPED -> MAPPING_PENDING -> MAPPED_WRITE -> UNMAPPED
  
  Invalid: MAPPED_READ -> MAPPED_WRITE (can't switch mode while mapped)
  Invalid: UNMAPPED -> MAPPED_READ (must go through MAPPING_PENDING)

Pipeline States:
  CREATED -> BOUND -> DISPATCHED -> CREATED
  
  Invalid: CREATED -> DISPATCHED (must bind first)

Frame Graph States:
  BUILDING -> COMPILED -> EXECUTING -> BUILDING
  
  Invalid: BUILDING -> EXECUTING (must compile first)
```

**What it catches:**
- Use-after-free patterns
- Double-free patterns
- Operations in wrong order
- Missing initialization

---

### Campaign Type: CONCURRENCY TESTING
**Goal:** Find race conditions and deadlocks.

**Concurrent scenarios:**
- Multiple threads submitting to same queue
- Buffer mapping while GPU is reading
- Resource destruction while in use
- Simultaneous frame graph compilation

**Tools:** `loom`, `ThreadSanitizer`

**What it catches:**
- Data races
- Deadlocks
- Use-after-free across threads
- Missing synchronization

**Specific concerns:**
```rust
// Is this safe if called from multiple threads?
scene_data.add(object);  // Pushes to Vec
scene_data.upload(&queue);  // Writes to GPU

// What if thread A is adding while thread B uploads?
```

---

### Campaign Type: MEMORY TESTING
**Goal:** Find leaks, corruption, invalid access.

**What to check:**
- GPU buffer leaks (created but never destroyed)
- CPU memory leaks
- Use-after-free (buffer destroyed, still referenced)
- Double-free
- Out-of-bounds access

**Tools:** `Valgrind`, `AddressSanitizer`, `cargo-careful`

**Renderer-specific concerns:**
```rust
// Do we drop GPU resources when objects go out of scope?
{
    let buffer = device.create_buffer(...);
    // use buffer
}  // <- Is wgpu::Buffer dropped here? Is GPU memory freed?

// What about reference cycles?
struct Renderer {
    device: Arc<Device>,
    buffers: Vec<Buffer>,  // Buffers hold Arc<Device>?
}
```

---

### Campaign Type: PERFORMANCE REGRESSION
**Goal:** Catch performance degradations.

**Metrics to track:**
- Frame time (ms)
- Draw calls per frame
- GPU memory usage
- CPU time in hot paths
- Shader compilation time

**Approach:**
```rust
#[bench]
fn bench_build_indirect_10k_objects(b: &mut Bencher) {
    // Setup 10k objects
    b.iter(|| {
        cpu_build_indirect(&compacted, &objects, &lods, &meshes)
    });
}

// Fail if regression > 10%
```

**What it catches:**
- O(n²) accidentally introduced
- Unnecessary allocations in hot path
- Cache-unfriendly data layout changes
- Synchronous GPU stalls

---

### Campaign Type: VISUAL REGRESSION
**Goal:** Catch rendering bugs that produce wrong pixels.

**Approach:**
1. Render known scene to texture
2. Compare against golden reference image
3. Fail if difference > threshold

**What it catches:**
- Shader bugs
- Wrong transform calculations
- Incorrect blending
- Missing draw calls
- Z-fighting changes

**Challenge:** Requires actual GPU to render. Can't run headless.

**Alternative:** Compare draw command lists instead of pixels.
```rust
let commands = render_scene(&scene);
assert_eq!(commands, EXPECTED_COMMANDS);
```

---

### Campaign Type: API CONTRACT TESTING
**Goal:** Verify public API doesn't break.

**What to check:**
- Function signatures don't change
- Return types are stable
- Error types are stable
- Public struct fields don't change layout

**Approach:**
```rust
// Snapshot public API
// cargo public-api diff

// Or explicit contract tests
#[test]
fn api_contract_object_data_size() {
    assert_eq!(std::mem::size_of::<ObjectData>(), 144);
}

#[test]
fn api_contract_object_data_fields() {
    let obj = ObjectData::default();
    let _ = obj.mesh_index;  // Compile error if field removed
    let _ = obj.transform;
}
```

---

### Campaign Type: SHADER VALIDATION
**Goal:** Verify WGSL shaders are correct.

**Checks:**
- Shader compiles without errors
- Shader uses correct bind group layouts
- Struct layouts match Rust counterparts
- No undefined behavior in shader

**Approach:**
```rust
#[test]
fn test_shader_struct_matches_rust() {
    let wgsl = include_str!("shader.wgsl");
    
    // Parse WGSL, extract struct offsets
    // Compare against Rust struct offsets
    // FAIL if mismatch (like the bug we just fixed!)
}
```

**This is HIGH VALUE.** The alignment bug we fixed is exactly what this catches.

---

### Campaign Type: CHAOS ENGINEERING
**Goal:** Inject failures, verify graceful degradation.

**Faults to inject:**
- GPU device lost mid-frame
- Buffer allocation fails randomly
- Shader compilation timeout
- Queue submission fails
- Texture upload corrupted

**What it catches:**
- Crash on transient failure
- Leaked resources after error
- Corrupted state after recovery

---

### Campaign Type: COMPATIBILITY TESTING
**Goal:** Verify works across different environments.

**Dimensions:**
- GPU vendors: NVIDIA, AMD, Intel, Apple Silicon
- Backends: Vulkan, Metal, DX12, WebGPU
- Driver versions: old, current, beta
- OS: Linux, Windows, macOS

**What it catches:**
- Vendor-specific bugs
- Backend-specific behavior
- Driver workarounds needed

---

## Campaign Priority Matrix

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

## Conclusion

**Current Confidence Level:** MEDIUM

We have evidence that SOME tests are good (they caught real bugs). We do NOT have evidence that ALL tests are good. The adversarial review campaign must continue.

**Trust but verify. Test the tests.**
