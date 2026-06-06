# How To: Getting Started with Trinity Harness

This guide walks you through integrating the harness into your project.

## Step 1: Add Dependency

```toml
# Cargo.toml
[dependencies]
trinity-harness = { path = "../trinity-harness" }
trinity-contracts = { path = "../trinity-contracts" }
```

## Step 2: Build the Code Graph

```rust
use trinity_harness::graph::GraphBuilder;

fn main() {
    // Scan your project
    let mut builder = GraphBuilder::new();
    let stats = builder.full_scan("./src");
    
    println!("Scanned {} nodes", stats.total_nodes);
    println!("  Rust: {}", stats.nodes_per_language.get(&Language::Rust).unwrap_or(&0));
    println!("  Python: {}", stats.nodes_per_language.get(&Language::Python).unwrap_or(&0));
    
    let graph = builder.build();
}
```

## Step 3: Map Tests to Code

```rust
use trinity_harness::graph::{ConventionMapper, create_test_edges};

// Auto-map using naming conventions
let mapper = ConventionMapper::new();
let (mappings, stats) = mapper.map_tests(&graph);

println!("Mapped {} tests", stats.tests_mapped);
println!("Unmapped: {}", stats.tests_unmapped);

// Add edges to graph
create_test_edges(&mut graph, &mappings);
```

### Custom Mappings (TOML)

Create `test_mappings.toml`:

```toml
[[mappings]]
test = "tests/integration/test_renderer.rs"
targets = ["src/renderer.rs", "src/gpu/*.rs"]

[[mappings]]
test = "tests/e2e/*.rs"
targets = ["src/core.rs"]
```

Load it:

```rust
use trinity_harness::graph::CombinedMapper;

let mapper = CombinedMapper::load_explicit("test_mappings.toml")?;
let (mappings, stats) = mapper.map_tests(&graph, Path::new("."));
```

## Step 4: Run Tests and Track State

```rust
use trinity_harness::runners::{ExecutorConfig, run_all_tests, StateTracker};

// Configure test execution
let config = ExecutorConfig::new(".")
    .cargo_timeout(300)   // 5 min
    .pytest_timeout(600); // 10 min

// Run tests
let result = run_all_tests(&config);

println!("Total: {}, Passed: {}, Failed: {}", 
    result.total, result.passed, result.failed);

// Track state
let mut tracker = StateTracker::new();
tracker.apply_results(&mapped_results);

let summary = tracker.summary();
println!("Health: {:.1}%", summary.health_percent());
```

## Step 5: Start the Daemon

```rust
use trinity_harness::daemon::{DaemonConfig, HarnessDaemon};

let config = DaemonConfig::new()
    .watch_path("./src")
    .watch_path("./tests")
    .poll_interval(1000);  // 1 second

let mut daemon = HarnessDaemon::new(config, graph);

// Set up callbacks
daemon.on_file_change(|path| {
    println!("Changed: {}", path);
});

daemon.on_state_change(|node_id, old, new| {
    println!("Node {} : {:?} -> {:?}", node_id.0, old, new);
});

// Run (blocks)
daemon.run()?;
```

## Step 6: Add Contracts to Your Code

```rust
use trinity_contracts::contract;

#[contract]
#[requires(divisor != 0)]
#[ensures(*result == dividend / divisor)]
pub fn safe_div(dividend: i32, divisor: i32) -> i32 {
    dividend / divisor
}
```

Inner attribute style:

```rust
#[contract]
pub fn sqrt(x: f64) -> f64 {
    #![requires(x >= 0.0)]
    #![ensures(*result >= 0.0)]
    x.sqrt()
}
```

## Step 7: Verify GPU Struct Layouts

```rust
use trinity_contracts::{assert_layout, LayoutSpec, WgslMirror, MirrorRegistry};

#[repr(C)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

// Compile-time check
assert_layout!(Vertex, size = 24, align = 4);

// Runtime registry for WGSL mirrors
let mut registry = MirrorRegistry::new();
registry.register(
    WgslMirror::new("Vertex", "Vertex")
        .layout(LayoutSpec::new().size(24).align(4))
);
```

## CLI Usage

```bash
# Start daemon
cargo run -p trinity-harness -- daemon

# Query stale tests
cargo run -p trinity-harness -- query needs-testing

# Run only stale tests
cargo run -p trinity-harness -- run-stale

# Update state from results file
cargo run -p trinity-harness -- update --results ./results.json
```

## Next Steps

- [ADDING_CONTRACTS.md](./ADDING_CONTRACTS.md) — Contract patterns and best practices
- [CI_INTEGRATION.md](./CI_INTEGRATION.md) — GitHub Actions setup
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) — Common issues and fixes
