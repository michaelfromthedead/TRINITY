# V1 Improvement Campaigns - renderer-backend

**Started:** 2026-06-04
**Purpose:** Make the code BETTER, not just verify it works.

---

## Philosophy

Testing asks: "Does it work?"
Improvement asks: "Can it work BETTER?"

This document tracks campaigns to actively improve the codebase across multiple dimensions.

---

## Campaign Categories

### Category A: PERFORMANCE OPTIMIZATION

Not "did it regress?" but "can we make it faster?"

---

#### Campaign: HOT PATH ANALYSIS

**Goal:** Identify and optimize the critical rendering path.

**Approach:**
```bash
# Profile with flamegraph
cargo flamegraph --bin renderer -- --scene demo.gltf

# Look for:
# - Functions taking >5% of frame time
# - Unexpected allocations in hot path
# - Lock contention
# - Cache misses
```

**Targets:**
| Hot Path | Current State | Optimization Opportunity |
|----------|---------------|-------------------------|
| Frame graph compile | ? ms | Cache compiled graph if unchanged |
| Buffer uploads | ? ms | Double-buffer to avoid stalls |
| Draw call generation | ? ms | GPU-side culling already done? |
| Command encoding | ? ms | Parallel encoding? |

**Metrics to track:**
- Frame time breakdown by phase
- Allocations per frame (should be ~0 in steady state)
- GPU idle time (CPU-bound indicator)
- CPU idle time (GPU-bound indicator)

---

#### Campaign: ALLOCATION ELIMINATION

**Goal:** Zero allocations in the hot path.

**Approach:**
```rust
// Use allocation profiler
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

// Find per-frame allocations
// Replace Vec with fixed-size arrays where possible
// Use arena allocators for temporary data
// Pool and reuse buffers
```

**Common offenders:**
```rust
// BAD: Allocates every frame
let commands: Vec<DrawCommand> = objects.iter().map(...).collect();

// GOOD: Reuse buffer
self.command_buffer.clear();
self.command_buffer.extend(objects.iter().map(...));

// BETTER: Pre-allocated fixed buffer
let commands: &mut [DrawCommand; MAX_DRAWS] = &mut self.command_buffer;
```

**Audit checklist:**
- [ ] `Vec::new()` in render loop?
- [ ] `String` operations in render loop?
- [ ] `Box::new()` in render loop?
- [ ] `clone()` in render loop?
- [ ] `collect()` in render loop?

---

#### Campaign: CACHE OPTIMIZATION

**Goal:** Improve CPU cache utilization.

**Techniques:**
```rust
// Data-oriented design: Struct of Arrays
// BAD: Array of Structs (AoS) - poor cache locality
struct Object { transform: Mat4, aabb: AABB, mesh_id: u32, ... }
objects: Vec<Object>

// GOOD: Struct of Arrays (SoA) - cache-friendly iteration
struct Objects {
    transforms: Vec<Mat4>,
    aabbs: Vec<AABB>,
    mesh_ids: Vec<u32>,
}
```

**Measurements:**
```bash
# Use perf to measure cache misses
perf stat -e cache-misses,cache-references cargo run

# Target: <5% cache miss rate on hot paths
```

---

#### Campaign: PARALLELIZATION

**Goal:** Utilize all CPU cores.

**Opportunities:**
| Task | Current | Parallel Opportunity |
|------|---------|---------------------|
| Frustum culling | Single-threaded? | Chunk objects across threads |
| Command encoding | Single-threaded? | Multiple encoders |
| Asset loading | Blocking? | Async/parallel loading |
| Shader compilation | Serial? | Parallel compile |

**Tools:** `rayon`, `tokio`, `crossbeam`

```rust
// Example: Parallel culling
use rayon::prelude::*;

let visible: Vec<usize> = objects
    .par_iter()  // Parallel iterator
    .enumerate()
    .filter(|(_, obj)| frustum.contains(obj.aabb))
    .map(|(i, _)| i)
    .collect();
```

---

### Category B: MEMORY OPTIMIZATION

---

#### Campaign: MEMORY FOOTPRINT REDUCTION

**Goal:** Use less RAM and VRAM.

**Analysis:**
```rust
// Print struct sizes
println!("ObjectData: {} bytes", std::mem::size_of::<ObjectData>());
println!("MeshData: {} bytes", std::mem::size_of::<MeshData>());

// Look for padding waste
// Look for oversized fields (u64 when u32 suffices)
// Look for redundant data
```

**Optimization patterns:**
```rust
// Pack booleans into bitflags
// BAD: 4 bools = 4 bytes (or 4 with padding!)
struct Object {
    visible: bool,
    casts_shadow: bool,
    static_: bool,
    selected: bool,
}

// GOOD: Bitflags = 1 byte
bitflags! {
    struct ObjectFlags: u8 {
        const VISIBLE = 0b0001;
        const CASTS_SHADOW = 0b0010;
        const STATIC = 0b0100;
        const SELECTED = 0b1000;
    }
}
```

**GPU memory audit:**
- [ ] Are we over-allocating buffer sizes?
- [ ] Are unused textures being freed?
- [ ] Are we using appropriate texture formats (RGBA8 vs RGBA16F)?
- [ ] Could we use texture compression?

---

#### Campaign: BUFFER POOLING

**Goal:** Reuse GPU buffers instead of create/destroy.

**Current state audit:**
```rust
// Are we doing this every frame?
let buffer = device.create_buffer(...);  // EXPENSIVE
// use buffer
drop(buffer);  // GPU memory freed

// Should we pool?
let buffer = buffer_pool.acquire(size);  // CHEAP
// use buffer
buffer_pool.release(buffer);  // Returns to pool
```

**Implementation:**
```rust
struct BufferPool {
    free_lists: HashMap<BufferSize, Vec<Buffer>>,
}

impl BufferPool {
    fn acquire(&mut self, size: u64) -> Buffer {
        // Round up to bucket size
        // Return from free list if available
        // Otherwise create new
    }
    
    fn release(&mut self, buffer: Buffer) {
        // Return to free list
    }
}
```

---

### Category C: CODE QUALITY

---

#### Campaign: COMPLEXITY REDUCTION

**Goal:** Simplify overly complex code.

**Metrics:**
```bash
# Cyclomatic complexity
cargo install cargo-bloat
cargo bloat --crates

# Lines per function (target: <50)
# Nesting depth (target: <4)
# Parameters per function (target: <5)
```

**Refactoring patterns:**
```rust
// BAD: Deep nesting
if a {
    if b {
        if c {
            if d {
                // actual code
            }
        }
    }
}

// GOOD: Early return
if !a { return; }
if !b { return; }
if !c { return; }
if !d { return; }
// actual code
```

**Targets:**
- [ ] Functions >100 lines
- [ ] Functions with >5 parameters
- [ ] Nesting >4 levels deep
- [ ] Files >500 lines

---

#### Campaign: DEAD CODE ELIMINATION

**Goal:** Remove code that's never executed.

**Tools:**
```bash
# Find unused functions
cargo +nightly udeps

# Compiler warnings
cargo build 2>&1 | grep "never used"

# Check feature flags
# Are there cfg(feature = "X") where X is never enabled?
```

**What to remove:**
- Unused functions
- Unused struct fields
- Unused enum variants
- Commented-out code
- TODO code that's been there for months
- Feature-flagged code for features that don't exist

---

#### Campaign: ERROR MESSAGE IMPROVEMENT

**Goal:** Make errors actionable.

**Bad error:**
```
Error: Buffer mapping failed
```

**Good error:**
```
Error: Buffer mapping failed
  Buffer: "vertex_staging_buffer" (1024 bytes)
  Requested: MAP_READ
  Reason: Buffer is currently mapped for writing
  Hint: Call unmap() before mapping with different mode
```

**Audit:**
```rust
// Find all error types
grep -r "enum.*Error" src/

// For each variant, check:
// - Does it include context?
// - Does it suggest a fix?
// - Does it include relevant values?
```

---

#### Campaign: API ERGONOMICS

**Goal:** Make the API easier to use correctly.

**Patterns:**
```rust
// BAD: Easy to use incorrectly
fn create_buffer(size: u64, usage: u32, mapped: bool) -> Buffer;
create_buffer(1024, 0x41, true);  // What do these mean?

// GOOD: Type-safe, self-documenting
fn create_buffer(desc: BufferDescriptor) -> Buffer;
create_buffer(BufferDescriptor {
    size: 1024,
    usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
    mapped_at_creation: true,
});

// BETTER: Builder pattern
Buffer::new(1024)
    .usage(BufferUsage::VERTEX | BufferUsage::COPY_DST)
    .mapped()
    .build(&device);
```

**Audit checklist:**
- [ ] Boolean parameters (should be enums)
- [ ] Multiple parameters of same type (easy to swap)
- [ ] Required vs optional parameters clear?
- [ ] Can invalid states be represented? (Make them unrepresentable)

---

### Category D: BUILD OPTIMIZATION

---

#### Campaign: COMPILE TIME REDUCTION

**Goal:** Faster iteration = faster development.

**Measurements:**
```bash
# Time full build
cargo clean && time cargo build

# Time incremental build
touch src/lib.rs && time cargo build

# Identify slow crates
cargo build --timings
```

**Techniques:**
```toml
# Cargo.toml
[profile.dev]
opt-level = 0
debug = 1  # Reduced debug info

[profile.dev.package."*"]
opt-level = 2  # Optimize dependencies, not your code
```

**Code changes:**
```rust
// Avoid heavy generics in hot headers
// Use impl Trait instead of generic <T: Trait>
// Split large crates into smaller ones
// Use dynamic linking in dev (controversial)
```

---

#### Campaign: BINARY SIZE REDUCTION

**Goal:** Smaller binaries, faster loads.

**Analysis:**
```bash
# What's in the binary?
cargo bloat --release

# Strip debug symbols
cargo build --release
strip target/release/renderer
```

**Techniques:**
```toml
[profile.release]
lto = true
codegen-units = 1
strip = true
panic = "abort"
```

---

### Category E: SHADER OPTIMIZATION

---

#### Campaign: SHADER PERFORMANCE

**Goal:** Faster GPU execution.

**Opportunities:**
| Shader | Optimization |
|--------|-------------|
| Frustum cull | Early-out on first plane failure |
| Build indirect | Coalesced memory access |
| PBR fragment | Reduce texture fetches |
| Light culling | Better workgroup utilization |

**Tools:**
- RenderDoc for GPU profiling
- Nsight Graphics (NVIDIA)
- PIX (Windows)
- GPU timestamps

**Patterns:**
```wgsl
// BAD: Divergent branches
if (object_idx % 2 == 0) {
    // half threads do this
} else {
    // half threads do that
}

// GOOD: Uniform branches (all threads same path)
if (params.mode == MODE_A) {
    // all threads do this
}
```

---

#### Campaign: SHADER VARIANT REDUCTION

**Goal:** Fewer shader permutations = faster compile, less memory.

**Analysis:**
```rust
// How many shader variants do we compile?
// For each boolean feature flag, we might 2x variants

// Features: SKINNED, ALPHA_TEST, SHADOWS, NORMAL_MAP
// Variants: 2^4 = 16 permutations!
```

**Techniques:**
- Use uber-shaders with dynamic branches (if GPU is fast enough)
- Group similar features
- Compile on demand, not ahead of time
- Cache compiled shaders to disk

---

### Category F: OBSERVABILITY

---

#### Campaign: LOGGING IMPROVEMENT

**Goal:** Better debugging when things go wrong.

**Levels:**
```rust
// ERROR: Something is broken
error!("Buffer mapping failed: {:?}", err);

// WARN: Something is suspicious
warn!("Frame took {}ms (target: 16ms)", elapsed);

// INFO: Lifecycle events
info!("Renderer initialized with {} objects", count);

// DEBUG: Detailed flow
debug!("Compiling shader: {}", path);

// TRACE: Everything
trace!("Object {} frustum test: {}", i, result);
```

**Structured logging:**
```rust
// BAD
info!("Rendering frame {} with {} objects", frame, count);

// GOOD: Machine-parseable
info!(frame = frame, objects = count, "Rendering frame");
```

---

#### Campaign: METRICS INSTRUMENTATION

**Goal:** Continuous performance visibility.

**Metrics to expose:**
```rust
// Counters
metrics.increment("draw_calls");
metrics.increment("triangles", mesh.triangle_count);

// Gauges
metrics.set("gpu_memory_used", bytes);
metrics.set("objects_visible", count);

// Histograms
metrics.histogram("frame_time_ms", elapsed);
metrics.histogram("cull_time_us", cull_elapsed);
```

**Integration:** Prometheus, StatsD, or custom dashboard.

---

### Category G: DEPENDENCY OPTIMIZATION

---

#### Campaign: DEPENDENCY AUDIT

**Goal:** Fewer, better dependencies.

**Analysis:**
```bash
# List all dependencies
cargo tree

# Find unused
cargo +nightly udeps

# Check for duplicates
cargo tree -d

# Check for outdated
cargo outdated

# Check for security issues
cargo audit
```

**Questions:**
- [ ] Do we need this dep, or can we inline 10 lines of code?
- [ ] Are we pulling in heavy deps for trivial features?
- [ ] Are deps maintained? Last update?
- [ ] Do deps have appropriate licenses?

---

## Improvement Priority Matrix

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

## Improvement Log

| Date | Campaign | Target | Before | After | Change |
|------|----------|--------|--------|-------|--------|
| | | | | | |

---

## Tooling Checklist

Required tools for improvement campaigns:

- [ ] `cargo flamegraph` - CPU profiling
- [ ] `dhat` or `heaptrack` - Allocation profiling
- [ ] `perf` - Low-level CPU metrics
- [ ] `cargo bloat` - Binary analysis
- [ ] `cargo udeps` - Unused dependencies
- [ ] `cargo outdated` - Dependency updates
- [ ] `cargo audit` - Security vulnerabilities
- [ ] RenderDoc - GPU profiling
- [ ] `cargo build --timings` - Build profiling

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Frame time (1K objects) | ? ms | <2 ms |
| Frame time (100K objects) | ? ms | <16 ms |
| Allocations per frame | ? | 0 |
| Build time (incremental) | ? s | <5 s |
| Binary size (release) | ? MB | <10 MB |
| GPU memory (1K objects) | ? MB | <100 MB |

---

## Relationship to QA Campaigns

| Improvement Campaign | Related QA Campaign |
|---------------------|---------------------|
| Hot Path Analysis | Performance Regression Testing |
| Allocation Elimination | Memory Testing |
| Parallelization | Concurrency Testing |
| Shader Performance | Visual Regression (correctness) |
| Error Messages | Negative Testing (error coverage) |

**The loop:**
1. QA campaign finds issue
2. Improvement campaign fixes it
3. QA campaign verifies fix
4. QA campaign prevents regression

---

**Move fast, measure everything, improve continuously.**
