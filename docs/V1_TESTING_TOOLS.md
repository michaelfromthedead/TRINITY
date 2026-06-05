# V1 Testing Tools & Harness - renderer-backend

**Started:** 2026-06-04
**Purpose:** Tools, harnesses, and infrastructure for QA and improvement campaigns.

---

## Document Relationships

```
V1_ADVERSARIAL_REVIEW.md    V1_IMPROVEMENT_CAMPAIGNS.md
         │                            │
         │  "What to test"            │  "What to improve"
         │                            │
         └──────────┬─────────────────┘
                    │
                    ▼
         V1_TESTING_TOOLS.md (this doc)
                    │
                    │  "How to do it"
                    │
                    ▼
            Testing Harness
            Custom Infrastructure
            Automated Pipelines
```

---

## Python → Rust Tool Mapping

For those coming from Python, here's what maps to what:

### Testing Basics

| Python | Rust | Notes |
|--------|------|-------|
| `pytest` | `cargo test` | Built into cargo, no install needed |
| `pytest test_foo.py` | `cargo test foo` | Filter by name |
| `pytest -k "pattern"` | `cargo test pattern` | Pattern matching |
| `pytest -x` | `cargo test -- --test-threads=1` | Stop on first failure (sort of) |
| `pytest -v` | `cargo test -- --nocapture` | Verbose output |
| `pytest --tb=short` | Default | Rust backtraces are shorter |
| `conftest.py` fixtures | `#[fixture]` from rstest | Or manual setup functions |

### Test Organization

| Python | Rust | Notes |
|--------|------|-------|
| `tests/` directory | `tests/` directory | Integration tests |
| `test_*.py` files | `*.rs` files in tests/ | Each file is separate binary |
| In-file tests | `#[cfg(test)] mod tests` | Unit tests in source file |
| `@pytest.mark.skip` | `#[ignore]` | Skip test |
| `@pytest.mark.parametrize` | `rstest` crate | Parameterized tests |

### Coverage

| Python | Rust | Notes |
|--------|------|-------|
| `coverage.py` | `cargo-tarpaulin` | Line coverage |
| `pytest-cov` | `cargo tarpaulin` | Integrated coverage |
| `coverage html` | `cargo tarpaulin --out Html` | HTML report |
| `.coveragerc` | `tarpaulin.toml` | Config file |

```bash
# Install
cargo install cargo-tarpaulin

# Run
cargo tarpaulin --out Html --output-dir coverage/

# Ignore test code itself
cargo tarpaulin --ignore-tests
```

### Property-Based Testing

| Python | Rust | Notes |
|--------|------|-------|
| `hypothesis` | `proptest` | Property-based testing |
| `@given(...)` | `proptest!` macro | Generate random inputs |
| `@example(...)` | `#[test]` | Explicit examples |

```rust
// Rust proptest example
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_lod_selection_never_panics(distance in 0.0f32..1000000.0) {
        let _ = select_lod(distance);  // Should never panic
    }
    
    #[test]
    fn test_lod_monotonic(d1 in 0.0f32..1000.0, d2 in 0.0f32..1000.0) {
        // Farther distance should never give higher detail LOD
        if d1 <= d2 {
            prop_assert!(select_lod(d1) <= select_lod(d2));
        }
    }
}
```

### Mocking

| Python | Rust | Notes |
|--------|------|-------|
| `unittest.mock` | `mockall` | Generate mock implementations |
| `@patch` | Trait-based dependency injection | Different pattern |
| `MagicMock` | `mock!` macro | Auto-generate mocks |

```rust
// Rust mocking with mockall
use mockall::automock;

#[automock]
trait GpuDevice {
    fn create_buffer(&self, size: u64) -> Buffer;
}

#[test]
fn test_with_mock_device() {
    let mut mock = MockGpuDevice::new();
    mock.expect_create_buffer()
        .returning(|size| Buffer::fake(size));
    
    // Use mock instead of real device
    render_frame(&mock);
}
```

**Note:** In GPU code, we often prefer real tests over mocks. Mocking the GPU defeats the purpose.

### Benchmarking

| Python | Rust | Notes |
|--------|------|-------|
| `pytest-benchmark` | `criterion` | Statistical benchmarking |
| `timeit` | `std::time::Instant` | Manual timing |
| `memory_profiler` | `dhat` | Allocation profiling |

```rust
// Rust criterion example
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_build_indirect(c: &mut Criterion) {
    let objects = setup_10k_objects();
    
    c.bench_function("build_indirect_10k", |b| {
        b.iter(|| cpu_build_indirect(&objects))
    });
}

criterion_group!(benches, bench_build_indirect);
criterion_main!(benches);
```

### Mutation Testing

| Python | Rust | Notes |
|--------|------|-------|
| `mutmut` | `cargo-mutants` | Mutation testing |
| `cosmic-ray` | `cargo-mutants` | Alternative |

```bash
# Install
cargo install cargo-mutants

# Run on module
cargo mutants -p renderer-backend -- src/gpu_driven/build_indirect.rs

# Results show "caught" (test failed = good) vs "missed" (test passed = bad)
```

### Fuzzing

| Python | Rust | Notes |
|--------|------|-------|
| `atheris` | `cargo-fuzz` | Coverage-guided fuzzing |
| `python-afl` | `afl.rs` | AFL fuzzer |

```rust
// Rust fuzzing target
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to parse data as GLTF, should not crash
    let _ = parse_gltf(data);
});
```

```bash
# Run fuzzer
cargo +nightly fuzz run fuzz_gltf_parser
```

### Linting & Static Analysis

| Python | Rust | Notes |
|--------|------|-------|
| `flake8` | `cargo clippy` | Linting |
| `pylint` | `cargo clippy` | More lints |
| `black` | `cargo fmt` | Formatting |
| `mypy` | Built-in | Rust is already typed |
| `bandit` | `cargo audit` | Security lints |

```bash
# Lint
cargo clippy -- -W clippy::all

# Format
cargo fmt

# Security audit
cargo audit
```

### Profiling

| Python | Rust | Notes |
|--------|------|-------|
| `cProfile` | `cargo flamegraph` | CPU profiling |
| `py-spy` | `perf` + flamegraph | Sampling profiler |
| `memory_profiler` | `dhat`, `heaptrack` | Memory profiling |
| `line_profiler` | Not common | Use flamegraph instead |

```bash
# CPU flamegraph
cargo install flamegraph
cargo flamegraph --bin my_app

# Memory profiling
cargo install cargo-instruments  # macOS
# Or use heaptrack on Linux
```

---

## Rust-Specific Testing Tools

### Snapshot Testing (like pytest-snapshot)

```bash
cargo install cargo-insta
```

```rust
use insta::assert_snapshot;

#[test]
fn test_draw_commands() {
    let commands = generate_draw_commands(&scene);
    assert_snapshot!(format!("{:#?}", commands));
}
```

Review snapshots:
```bash
cargo insta review
```

### Test Fixtures (like pytest fixtures)

```bash
cargo add rstest --dev
```

```rust
use rstest::{fixture, rstest};

#[fixture]
fn gpu_device() -> Device {
    create_test_device()
}

#[fixture]
fn sample_scene() -> Scene {
    load_scene("test_scene.gltf")
}

#[rstest]
fn test_render(gpu_device: Device, sample_scene: Scene) {
    // Both fixtures injected automatically
    render(&gpu_device, &sample_scene);
}

// Parameterized tests
#[rstest]
#[case(0, 0)]
#[case(1, 1)]
#[case(100, 100)]
#[case(10000, 10000)]
fn test_object_counts(#[case] input: usize, #[case] expected: usize) {
    assert_eq!(count_visible(input), expected);
}
```

### Async Testing

```rust
// For async code (not common in GPU code but useful for asset loading)
#[tokio::test]
async fn test_async_load() {
    let asset = load_asset("model.gltf").await;
    assert!(asset.is_ok());
}
```

### Doc Tests

```rust
/// Selects LOD level based on distance.
/// 
/// # Example
/// ```
/// use renderer_backend::select_lod;
/// assert_eq!(select_lod(50.0), 0);  // Close = high detail
/// assert_eq!(select_lod(500.0), 2); // Far = low detail
/// ```
pub fn select_lod(distance: f32) -> usize {
    // ...
}
```

Doc tests run with `cargo test` automatically!

---

## GPU-Specific Testing Tools

No Python equivalent exists for most of these.

### wgpu Test Harness

```rust
/// Standard test harness for GPU tests
pub struct GpuTestHarness {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuTestHarness {
    /// Create harness, returns None if no GPU available
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default())
        )?;
        
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
        ).ok()?;
        
        Some(Self { instance, adapter, device, queue })
    }
    
    /// Run compute shader and read back results
    pub fn run_compute<T: bytemuck::Pod>(
        &self,
        shader: &str,
        input: &[T],
    ) -> Vec<T> {
        // ... implementation
    }
}

// Usage in tests
#[test]
fn test_gpu_culling() {
    let harness = match GpuTestHarness::new() {
        Some(h) => h,
        None => {
            eprintln!("Skipping: no GPU");
            return;
        }
    };
    
    let result = harness.run_compute(CULL_SHADER, &objects);
    assert_eq!(result.len(), expected_visible);
}
```

### GPU/CPU Parity Test Pattern

```rust
/// Macro for CPU/GPU parity tests
macro_rules! parity_test {
    ($name:ident, $cpu_fn:expr, $gpu_shader:expr, $input:expr) => {
        #[test]
        fn $name() {
            let harness = GpuTestHarness::new().expect("Need GPU for parity test");
            
            let input = $input;
            let cpu_result = $cpu_fn(&input);
            let gpu_result = harness.run_compute($gpu_shader, &input);
            
            assert_eq!(
                cpu_result, gpu_result,
                "CPU/GPU mismatch!\nCPU: {:?}\nGPU: {:?}",
                cpu_result, gpu_result
            );
        }
    };
}

// Usage
parity_test!(
    test_frustum_cull_parity,
    cpu_frustum_cull,
    include_str!("shaders/frustum_cull.wgsl"),
    generate_test_objects(1000)
);
```

### Shader Struct Validator

```rust
/// Validates that Rust struct layout matches WGSL struct
pub fn validate_struct_layout<T>(wgsl_source: &str, struct_name: &str) 
where T: Sized
{
    let rust_size = std::mem::size_of::<T>();
    let wgsl_size = parse_wgsl_struct_size(wgsl_source, struct_name);
    
    assert_eq!(
        rust_size, wgsl_size,
        "Struct size mismatch for {}: Rust={}, WGSL={}",
        struct_name, rust_size, wgsl_size
    );
    
    // Also validate field offsets...
}

#[test]
fn test_object_data_layout_matches() {
    validate_struct_layout::<ObjectData>(
        include_str!("shaders/build_indirect.wgsl"),
        "ObjectData"
    );
}
```

---

## Synth Integration — Declarative Test Data Generation

We maintain a custom fork of [synth](../../../synth/) — a declarative data generator with
constraint-driven population synthesis. We integrate it into our testing harness for:

1. **Learning** — Build expertise in property-based testing and constraint solving
2. **Exercising synth** — Keep our fork alive and battle-tested
3. **Full-featured harness** — Generate complex, realistic test data at scale

### What Synth Provides

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Schema-based generation** | Define data shape in JSON | Generate valid GPU objects |
| **Constraint solving** | Specify population targets, synth computes params | "95% visible, avg LOD 2.5" |
| **G-F-S loop** | Generate → Falsify → Shrink | Property testing pattern |
| **Gaussian Copula** | Correlated multi-field generation | Spatial LOD correlation |
| **Pipeline Models** | Linear, DecisionTree, Logistic, Ensemble | Predict/invert constraints |
| **SQLite integration** | Import/export schemas and data | Persist test fixtures |

### Synth Location

```
/home/user/dev/USER/PROJECTS_VOID/synth/
├── synth/           # CLI and main crate
├── core/            # Schema model, compiler
├── constraint/      # G-F-S loop, copula, solver
├── gen/             # Generator primitives
└── test_macros/     # File-based test generation
```

### Architecture: The G-F-S Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONSTRAINT-DRIVEN SYNTHESIS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐                                              │
│   │   SCHEMA    │  "10K objects with mesh_index 0-99"          │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ CONSTRAINTS │  "mean(lod) = 2.5, 95% visible"              │
│   └──────┬──────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │  GENERATE   │────▶│   FALSIFY   │────▶│   SHRINK    │      │
│   │  (sample)   │     │  (check)    │     │ (minimize)  │      │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘      │
│          │                   │                   │              │
│          │◀──────────────────┴───────────────────┘              │
│          │         (adjust params, repeat)                      │
│          ▼                                                      │
│   ┌─────────────┐                                              │
│   │ TEST DATA   │  Valid objects satisfying all constraints    │
│   └─────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Renderer Test Data Schemas

#### ObjectData Schema

```json
{
  "type": "array",
  "length": { "type": "number", "subtype": "u64", "range": { "low": 100, "high": 100000 } },
  "content": {
    "type": "object",
    "mesh_index": {
      "type": "number",
      "subtype": "u32",
      "range": { "low": 0, "high": 99 }
    },
    "material_index": {
      "type": "number",
      "subtype": "u32",
      "range": { "low": 0, "high": 49 }
    },
    "position": {
      "type": "array",
      "length": { "type": "number", "constant": 3 },
      "content": {
        "type": "number",
        "subtype": "f32",
        "distributed": { "distribution": "normal", "mean": 0.0, "std_dev": 100.0 }
      }
    },
    "lod_distances": {
      "type": "array",
      "length": { "type": "number", "constant": 4 },
      "content": {
        "type": "number",
        "subtype": "f32",
        "range": { "low": 100.0, "high": 10000.0 }
      }
    },
    "flags": {
      "type": "number",
      "subtype": "u32",
      "range": { "low": 0, "high": 4095 }
    }
  }
}
```

Save as `test_schemas/object_data.json`, generate with:
```bash
cd /home/user/dev/USER/PROJECTS_VOID/synth
cargo run -- generate ../TRINITY/test_schemas/object_data --size 10000
```

#### Constraint File for Stress Testing

```json
{
  "constraints": [
    {
      "type": "row_count",
      "collection": "objects",
      "count": 100000
    },
    {
      "type": "percentage",
      "field": "objects.content.flags",
      "condition": "bitwise_and(value, 1) == 1",
      "target": 0.70,
      "tolerance": 0.05
    },
    {
      "type": "mean",
      "field": "objects.content.mesh_index",
      "target": 50.0,
      "tolerance": 5.0
    },
    {
      "type": "distribution",
      "field": "objects.content.position[0]",
      "distribution": "normal",
      "mean": 0.0,
      "std_dev": 100.0
    }
  ]
}
```

**Translation:** "Generate 100K objects where ~70% are visible (flag bit 0), mesh indices average around 50, and positions follow a normal distribution."

### Integration Pattern: Rust + Synth

#### Option 1: CLI Integration (Simplest)

```rust
use std::process::Command;

fn generate_test_objects(count: usize) -> Vec<ObjectData> {
    let output = Command::new("cargo")
        .current_dir("/home/user/dev/USER/PROJECTS_VOID/synth")
        .args([
            "run", "--release", "--",
            "generate", "../TRINITY/test_schemas/object_data",
            "--size", &count.to_string(),
            "--to", "-"  // stdout
        ])
        .output()
        .expect("synth generate failed");
    
    assert!(output.status.success(), "synth failed: {}", 
            String::from_utf8_lossy(&output.stderr));
    
    // Parse JSON output to ObjectData
    parse_synth_output(&output.stdout)
}

#[test]
fn test_culling_100k_objects() {
    let objects = generate_test_objects(100_000);
    let visible = frustum_cull(&objects, &test_frustum());
    
    // With ~70% visible constraint, we expect roughly 70K visible
    assert!(visible.len() > 60_000, "Expected ~70% visible");
    assert!(visible.len() < 80_000, "Expected ~70% visible");
}
```

#### Option 2: Library Integration (Tighter)

```rust
// In Cargo.toml
[dev-dependencies]
synth-core = { path = "../synth/core" }
synth-gen = { path = "../synth/gen" }

// In tests
use synth_core::{Namespace, Content};
use synth_gen::prelude::*;

fn generate_objects_inline(count: usize) -> Vec<ObjectData> {
    let schema = Content::Array {
        length: Box::new(Content::Number(NumberContent::U64(
            number::U64::Range(RangeStep::new(count as u64, count as u64, 1))
        ))),
        content: Box::new(Content::Object(/* ... */)),
    };
    
    let mut rng = rand::thread_rng();
    let generated = schema.generate(&mut rng);
    
    // Convert to ObjectData
    synth_to_object_data(generated)
}
```

#### Option 3: Pre-generated Fixtures (Fastest Tests)

```bash
# Generate once, commit to repo
cd synth
cargo run --release -- generate ../TRINITY/test_schemas/object_data \
    --size 10000 --seed 42 \
    --to ../TRINITY/test_fixtures/objects_10k.json

cargo run --release -- generate ../TRINITY/test_schemas/object_data \
    --size 100000 --seed 42 \
    --to ../TRINITY/test_fixtures/objects_100k.json
```

```rust
// In tests — load pre-generated fixtures
const OBJECTS_10K: &str = include_str!("../test_fixtures/objects_10k.json");

fn load_test_objects_10k() -> Vec<ObjectData> {
    serde_json::from_str(OBJECTS_10K).expect("parse fixture")
}
```

### Synth for Property Testing

Combine synth with the G-F-S pattern for property-based testing:

```rust
use synth_constraint::{falsify, shrink, ShrinkStrategy};

#[test]
fn property_culling_preserves_count() {
    for seed in 0..100 {
        // Generate with different seeds
        let objects = generate_test_objects_seeded(1000, seed);
        let frustum = random_frustum(seed);
        
        let visible = frustum_cull(&objects, &frustum);
        let culled = objects.len() - visible.len();
        
        // Property: visible + culled == total
        if visible.len() + culled != objects.len() {
            // FALSIFY: Found a violation
            let actual = hashmap!{ "visible" => visible.len() as f64 };
            let target = hashmap!{ "visible" => objects.len() as f64 };
            
            let result = falsify(&actual, &target, &hashmap!{});
            
            // SHRINK: Find minimal failing case
            let minimal = shrink(&actual, &target, &hashmap!{}, 
                                 ShrinkStrategy::ParameterRange);
            
            panic!("Property violated!\n{}\nMinimal case: {:?}", 
                   result.sensitivity.unwrap().primary_target.unwrap(),
                   minimal);
        }
    }
}
```

### Correlated Data with Gaussian Copula

Generate spatially-correlated LOD levels (nearby objects = similar LOD):

```rust
use synth_constraint::GaussianCopula;

fn generate_correlated_lods(positions: &[[f32; 3]], correlation: f64) -> Vec<u32> {
    // Create copula with position-LOD correlation
    let copula = GaussianCopula::bivariate(correlation).unwrap();
    
    let mut rng = rand::thread_rng();
    let mut lods = Vec::with_capacity(positions.len());
    
    for pos in positions {
        let [u_pos, u_lod] = copula.sample(&mut rng);
        
        // Map uniform to position influence
        let distance_from_origin = (pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]).sqrt();
        
        // Map uniform to LOD (0-3), biased by position
        let lod = ((u_lod * 4.0) as u32).min(3);
        lods.push(lod);
    }
    
    lods
}
```

### test_macros Integration

Use synth's `tmpl_ignore` macro to generate tests from schema files:

```rust
use test_macros::tmpl_ignore;

// Generate a validation test for every .json schema in test_schemas/
#[tmpl_ignore("./test_schemas/", exclude_dir = true, filter_extension = "json")]
#[test]
fn test_schema_PATH_IDENT_is_valid() {
    let schema_json = include_str!(PATH);
    let schema: synth_core::Content = serde_json::from_str(schema_json)
        .expect("schema should parse");
    
    // Validate schema is well-formed
    schema.validate().expect("schema should be valid");
}

// Generate a generation test for every schema
#[tmpl_ignore("./test_schemas/", exclude_dir = true, filter_extension = "json")]
#[test]
fn test_schema_PATH_IDENT_generates() {
    let schema_json = include_str!(PATH);
    let schema: synth_core::Content = serde_json::from_str(schema_json).unwrap();
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let result = schema.generate(&mut rng);
    
    assert!(result.is_some(), "schema should generate data");
}
```

### When to Use Synth vs Hand-Crafted Data

| Scenario | Use Synth? | Why |
|----------|------------|-----|
| Unit test edge case | NO | Hand-craft the specific case |
| "Does it crash with 100K objects?" | YES | Synth generates realistic scale |
| "Is culling correct for this frustum?" | NO | Hand-craft known-visible/hidden objects |
| "Statistical properties hold?" | YES | Synth can target distributions |
| "Regression test specific bug" | NO | Capture exact failing case |
| "Stress test memory pressure" | YES | Generate to constraint limits |
| "Fuzz with valid data" | YES | Synth = structured fuzzing |

### Synth Development Tasks

As we use synth, track issues and improvements:

| Task | Status | Notes |
|------|--------|-------|
| Create `test_schemas/` directory | TODO | ObjectData, MeshData, etc. |
| Add synth as workspace member | TODO | Or keep separate |
| Create CLI wrapper script | TODO | `./scripts/synth-gen.sh` |
| Pre-generate fixtures | TODO | 1K, 10K, 100K objects |
| Property test integration | TODO | Combine with proptest |
| CI integration | TODO | Generate in CI, cache fixtures |

### Synth Quick Reference

```bash
# Build synth
cd /home/user/dev/USER/PROJECTS_VOID/synth
cargo build --release

# Generate from schema
cargo run --release -- generate <namespace> --size <N>

# Generate to file
cargo run --release -- generate <namespace> --size <N> --to output.json

# Generate to SQLite
cargo run --release -- generate <namespace> --size <N> --to sqlite:output.db

# Import schema from SQLite
cargo run --release -- import my_ns --from sqlite:source.db

# Validate schema
cargo run --release -- validate <namespace>

# With constraints
cargo run --release -- generate <namespace> --constraints targets.json --size <N>

# Deterministic generation (for reproducible tests)
cargo run --release -- generate <namespace> --size <N> --seed 42
```

---

## Testing Harness Architecture

### Directory Structure

```
crates/renderer-backend/
├── src/
│   └── lib.rs                    # Main code
├── tests/
│   ├── common/                   # Shared test utilities
│   │   ├── mod.rs
│   │   ├── gpu_harness.rs        # GPU test harness
│   │   ├── fixtures.rs           # Test data generators
│   │   ├── synth_bridge.rs       # Synth integration helpers
│   │   └── assertions.rs         # Custom assertions
│   ├── blackbox_*.rs             # Integration tests (external API)
│   └── whitebox_*.rs             # Internal tests (implementation details)
├── test_schemas/                 # Synth schema definitions
│   ├── object_data.json          # ObjectData generation schema
│   ├── mesh_data.json            # MeshData generation schema
│   ├── draw_commands.json        # DrawCommand generation schema
│   └── constraints/              # Constraint files for different scenarios
│       ├── stress_100k.json      # 100K objects stress test
│       ├── edge_cases.json       # Boundary conditions
│       └── lod_distribution.json # LOD statistical targets
├── test_fixtures/                # Pre-generated test data (committed)
│   ├── objects_1k.json           # 1K objects, seed 42
│   ├── objects_10k.json          # 10K objects, seed 42
│   └── objects_100k.json         # 100K objects, seed 42
├── benches/
│   ├── criterion.rs              # Benchmark entry point
│   ├── build_indirect.rs         # Build indirect benchmarks
│   └── frame_graph.rs            # Frame graph benchmarks
└── fuzz/
    ├── Cargo.toml                # Fuzz target crate
    └── fuzz_targets/
        ├── gltf_parser.rs        # Fuzz GLTF parsing
        └── shader_parser.rs      # Fuzz shader parsing

# Synth library (sibling project)
../synth/
├── synth/                        # CLI crate
├── core/                         # Schema model
├── constraint/                   # G-F-S loop
├── gen/                          # Generators
└── test_macros/                  # tmpl_ignore macro
```

### Custom Assertions

```rust
// tests/common/assertions.rs

/// Assert two float arrays are approximately equal
pub fn assert_float_array_eq(actual: &[f32], expected: &[f32], epsilon: f32) {
    assert_eq!(actual.len(), expected.len(), "Array length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < epsilon,
            "Mismatch at index {}: {} vs {} (epsilon={})",
            i, a, e, epsilon
        );
    }
}

/// Assert buffer contents match expected
pub fn assert_buffer_eq<T: bytemuck::Pod + std::fmt::Debug + PartialEq>(
    device: &Device,
    queue: &Queue,
    buffer: &Buffer,
    expected: &[T],
) {
    let actual = read_buffer::<T>(device, queue, buffer);
    assert_eq!(actual, expected);
}

/// Assert draw commands match (order-independent)
pub fn assert_draw_commands_eq(
    actual: &[DrawCommand],
    expected: &[DrawCommand],
) {
    let mut actual_sorted = actual.to_vec();
    let mut expected_sorted = expected.to_vec();
    actual_sorted.sort_by_key(|c| c.first_instance);
    expected_sorted.sort_by_key(|c| c.first_instance);
    assert_eq!(actual_sorted, expected_sorted);
}
```

### Test Data Generators

```rust
// tests/common/fixtures.rs

/// Generate N random objects for testing
pub fn random_objects(count: usize, seed: u64) -> Vec<ObjectData> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    
    (0..count).map(|i| {
        ObjectData::new()
            .with_transform(random_transform(&mut rng))
            .with_aabb(random_aabb(&mut rng))
            .with_mesh(rng.gen_range(0..10))
            .with_lod_distances([100.0, 400.0, 1600.0, 6400.0])
    }).collect()
}

/// Generate objects in a grid pattern (deterministic)
pub fn grid_objects(nx: usize, ny: usize, nz: usize) -> Vec<ObjectData> {
    let mut objects = Vec::with_capacity(nx * ny * nz);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                objects.push(ObjectData::new()
                    .with_position([x as f32, y as f32, z as f32]));
            }
        }
    }
    objects
}

/// Standard test scenes
pub enum TestScene {
    Empty,
    SingleObject,
    ThousandObjects,
    StressTest,  // 100K objects
}

impl TestScene {
    pub fn objects(&self) -> Vec<ObjectData> {
        match self {
            Self::Empty => vec![],
            Self::SingleObject => vec![ObjectData::default()],
            Self::ThousandObjects => random_objects(1000, 42),
            Self::StressTest => random_objects(100_000, 42),
        }
    }
}
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        
      - name: Cache
        uses: Swatinem/rust-cache@v2
        
      - name: Run tests
        run: cargo test --all-features
        
      - name: Run clippy
        run: cargo clippy -- -D warnings
        
      - name: Check formatting
        run: cargo fmt -- --check

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Install tarpaulin
        run: cargo install cargo-tarpaulin
        
      - name: Generate coverage
        run: cargo tarpaulin --out Xml
        
      - name: Upload to codecov
        uses: codecov/codecov-action@v3

  gpu-tests:
    runs-on: [self-hosted, gpu]  # Need GPU runner!
    steps:
      - uses: actions/checkout@v4
      - name: Run GPU tests
        run: cargo test --features gpu-tests

  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install cargo-mutants
        run: cargo install cargo-mutants
      - name: Run mutation tests
        run: cargo mutants --package renderer-backend -- src/gpu_driven/
```

### Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

set -e

echo "Running cargo fmt..."
cargo fmt -- --check

echo "Running cargo clippy..."
cargo clippy -- -D warnings

echo "Running tests..."
cargo test --lib

echo "All checks passed!"
```

---

## Tool Installation Checklist

```bash
# Core testing
cargo install cargo-tarpaulin    # Coverage
cargo install cargo-mutants      # Mutation testing
cargo install cargo-insta        # Snapshot testing

# Performance
cargo install cargo-flamegraph   # CPU profiling
cargo install criterion          # Benchmarking (usually as dev-dep)

# Quality
cargo install cargo-audit        # Security audit
cargo install cargo-outdated     # Dependency updates
cargo install cargo-udeps        # Unused dependencies

# Fuzzing (requires nightly)
cargo +nightly install cargo-fuzz

# Optional
cargo install cargo-watch        # Auto-rerun on changes
cargo install cargo-nextest      # Faster test runner

# Synth (our declarative data generator)
cd /home/user/dev/USER/PROJECTS_VOID/synth
cargo build --release
# Add to PATH or create alias:
alias synth='cargo run --manifest-path /home/user/dev/USER/PROJECTS_VOID/synth/Cargo.toml --release --'
```

---

## Quick Reference

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_cpu_gpu_parity

# Specific module
cargo test gpu_driven::

# Integration tests only
cargo test --test blackbox_build_indirect

# With output
cargo test -- --nocapture

# Single-threaded (for debugging)
cargo test -- --test-threads=1

# Ignored tests
cargo test -- --ignored

# Doc tests only
cargo test --doc
```

### Coverage

```bash
# Generate HTML report
cargo tarpaulin --out Html --output-dir coverage/

# Just show percentage
cargo tarpaulin --print-rust-flags

# Specific package
cargo tarpaulin -p renderer-backend
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Specific benchmark
cargo bench build_indirect

# Save baseline
cargo bench -- --save-baseline main

# Compare to baseline
cargo bench -- --baseline main
```

### Mutation Testing

```bash
# Run on specific file
cargo mutants -- src/gpu_driven/build_indirect.rs

# Faster (less thorough)
cargo mutants --timeout 30

# Just list mutations
cargo mutants --list
```

---

## Mapping Campaigns to Tools

| QA Campaign | Primary Tool | Secondary Tool | Synth Role |
|-------------|--------------|----------------|------------|
| Mutation Testing | `cargo-mutants` | Manual mutation | Generate diverse inputs |
| Boundary Testing | `proptest` | `rstest` parametrized | Constraint-driven edge cases |
| Fuzzing | `cargo-fuzz` | `afl.rs` | Structured fuzzing (valid data) |
| Coverage | `cargo-tarpaulin` | `grcov` | Generate high-coverage inputs |
| Negative Testing | Standard `cargo test` | Custom assertions | — |
| State Machine | `proptest` state machines | Manual tests | Generate state sequences |
| Memory Testing | `dhat` / `heaptrack` | `miri` | Generate memory-pressure loads |
| Concurrency | `loom` | ThreadSanitizer | Generate concurrent workloads |
| **Statistical Validation** | **synth** | proptest | **Target distributions** |
| **Stress Testing** | **synth** | manual | **100K+ objects** |

| Improvement Campaign | Primary Tool | Secondary Tool | Synth Role |
|---------------------|--------------|----------------|------------|
| Hot Path Analysis | `cargo flamegraph` | `perf` | Generate realistic workloads |
| Allocation Elimination | `dhat` | `heaptrack` | Generate allocation-heavy cases |
| Performance Regression | `criterion` | Manual timing | Consistent benchmark data |
| Dead Code | `cargo-udeps` | Compiler warnings | — |
| Complexity | `cargo-bloat` | Manual review | — |
| Shader Performance | RenderDoc | Nsight Graphics | Generate GPU workloads |

---

## Next Steps

### Phase 1: Foundation (This Week)
1. [ ] Set up `tests/common/` directory with shared utilities
2. [ ] Create `GpuTestHarness` implementation
3. [ ] Add `validate_struct_layout` for all GPU structs
4. [ ] Install all tools from checklist

### Phase 2: Synth Integration (Next Week)
5. [ ] Build synth from source, verify it runs
6. [ ] Create `test_schemas/` directory in renderer-backend
7. [ ] Write `object_data.json` schema matching Rust struct
8. [ ] Write `mesh_data.json` schema matching Rust struct
9. [ ] Create `tests/common/synth_bridge.rs` for CLI integration
10. [ ] Generate first fixture: `objects_1k.json` with seed 42
11. [ ] Write test using synth-generated data

### Phase 3: Constraint-Driven Testing
12. [ ] Create constraint files for different scenarios:
    - [ ] `constraints/stress_100k.json` - Maximum load
    - [ ] `constraints/edge_cases.json` - Boundary conditions
    - [ ] `constraints/lod_distribution.json` - LOD targets
13. [ ] Integrate synth G-F-S loop with proptest
14. [ ] Create copula-based correlated data generator

### Phase 4: CI/CD
15. [ ] Configure CI with coverage and mutation testing
16. [ ] Add synth generation step to CI
17. [ ] Create first benchmark baseline
18. [ ] Set up performance regression detection

---

## Synth Learning Goals

As we integrate synth, we're also learning:

| Concept | Synth Implementation | Learning Value |
|---------|---------------------|----------------|
| Schema-first design | JSON schema → generated data | Declarative thinking |
| Constraint solving | Inverse problem: targets → params | Statistical reasoning |
| Property testing | G-F-S loop | Automated bug finding |
| Shrinking | Minimal failing case | Debugging discipline |
| Copulas | Correlated generation | Statistical dependence |
| Pipeline models | Linear, Tree, Ensemble | ML fundamentals |

---

## Success Criteria for Synth Integration

| Milestone | Metric | Target |
|-----------|--------|--------|
| Schemas defined | Count | 5+ schemas |
| Fixtures generated | Objects | 1K, 10K, 100K |
| Tests using synth | Count | 10+ tests |
| Constraint scenarios | Count | 5+ scenarios |
| G-F-S integrated | Tests | 3+ property tests |
| Bugs found via synth | Count | ≥1 (validation!) |

---

**Tools are worthless without discipline. Use them consistently.**

**Build the harness. Trust the harness. Improve the harness.**
