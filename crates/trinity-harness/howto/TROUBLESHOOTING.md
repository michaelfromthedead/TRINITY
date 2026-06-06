# How To: Troubleshooting

Common issues and their solutions.

## Parse Errors

### "Failed to parse Rust file"

**Cause:** Invalid Rust syntax or unsupported language features.

**Solution:**
```rust
// Check if the file parses
use trinity_harness::parsers::RustParser;

let parser = RustParser::new();
match parser.parse(source) {
    Ok(units) => println!("Parsed {} units", units.len()),
    Err(e) => eprintln!("Parse error: {:?}", e),
}
```

### "Unknown type format" warning

**Cause:** Non-path type encountered (e.g., function pointers, references).

**Output:**
```
[rust_parser] Unexpected type format (non-path type) — returning placeholder
```

**Solution:** This is informational. The parser uses `"?"` as a placeholder for complex types. These are still tracked, just with less detail.

## Test Mapping Issues

### "Test has no targets"

**Cause:** Convention-based mapping couldn't find a matching code file.

**Check:**
```rust
use trinity_harness::graph::extract_unmapped;

let review = extract_unmapped(&mappings, &graph);
println!("{}", review.generate_report());
```

**Solutions:**
1. Rename test to match convention (`test_<name>.rs` → `<name>.rs`)
2. Add explicit mapping in `test_mappings.toml`:
   ```toml
   [[mappings]]
   test = "tests/my_special_test.rs"
   targets = ["src/actual_module.rs"]
   ```

### Tests mapped to wrong code

**Cause:** Name collision or overly broad convention match.

**Solution:** Use explicit mapping with precise paths:
```toml
[[mappings]]
test = "tests/unit/test_parser.rs"
targets = ["src/parser/mod.rs"]  # Not src/parser.rs
```

## State Tracking Issues

### "Node stuck in DIRTY state"

**Cause:** Tests haven't run since the code changed.

**Check:**
```rust
let dirty_nodes = tracker.nodes_in_state(NodeState::Dirty);
println!("Dirty: {:?}", dirty_nodes);
```

**Solution:** Run the stale tests:
```bash
trinity-harness run-stale
```

### "All nodes show UNTESTED"

**Cause:** Initial scan without baseline run.

**Solution:** Run full test suite once to establish baseline:
```rust
use trinity_harness::runners::{run_all_tests, ExecutorConfig};

let config = ExecutorConfig::new(".");
let result = run_all_tests(&config);
tracker.apply_results(&mapped_results);
```

## Daemon Issues

### "File watcher not detecting changes"

**Cause:** Watch path not set or polling too slow.

**Solution:**
```rust
let config = DaemonConfig::new()
    .watch_path("./src")     // Add all relevant paths
    .watch_path("./tests")
    .poll_interval(500);     // Faster polling (500ms)
```

### "Too many file events"

**Cause:** No debouncing, or IDE saving multiple times.

**Solution:** Increase debounce time:
```rust
use trinity_harness::daemon::Debouncer;

let debouncer = Debouncer::new(200);  // 200ms debounce
```

### "Daemon consuming high CPU"

**Cause:** Poll interval too short or watching too many files.

**Solution:**
1. Increase poll interval: `.poll_interval(2000)` (2 seconds)
2. Exclude directories: `.ignore_path("target")`, `.ignore_path("node_modules")`
3. Filter extensions: `.watch_extensions(&["rs", "py", "wgsl"])`

## Contract Issues

### "Precondition violated" panic

**Cause:** Function called with invalid arguments.

**Example:**
```
thread 'main' panicked at 'Precondition violated: x > 0'
```

**Solution:** Fix the caller to pass valid arguments, or relax the constraint if it's too strict.

### "Postcondition violated" panic

**Cause:** Function implementation doesn't satisfy the constraint.

**Example:**
```
thread 'main' panicked at 'Postcondition violated: *result >= 0'
```

**Solution:** Fix the function implementation or correct the postcondition.

### Layout assertion fails

**Cause:** Struct size/alignment doesn't match expectation.

**Example:**
```
error: size mismatch for Vertex
```

**Check actual layout:**
```rust
use trinity_contracts::layout::get_layout;

let (size, align) = get_layout::<Vertex>();
println!("Vertex: size={}, align={}", size, align);
```

**Solution:** Add padding or use `#[repr(C, align(N))]` to match expected layout.

## Performance Issues

### "Full scan takes too long"

**Cause:** Scanning too many files (e.g., `target/`, `node_modules/`).

**Solution:**
```rust
let mut builder = GraphBuilder::new();
builder.exclude_dir("target");
builder.exclude_dir("node_modules");
builder.exclude_dir(".git");
let stats = builder.full_scan("./");
```

### "Test execution timeout"

**Cause:** Default timeout exceeded.

**Solution:** Increase timeouts:
```rust
let config = ExecutorConfig::new(".")
    .cargo_timeout(1200)   // 20 minutes
    .pytest_timeout(3600); // 1 hour
```

## Database Issues

### "Failed to open state.db"

**Cause:** File permissions or corrupted database.

**Solution:**
1. Check permissions: `ls -la .harness/state.db`
2. Delete and rebuild: `rm .harness/state.db && trinity-harness scan`

### "Database locked"

**Cause:** Another process has the database open.

**Solution:**
1. Stop other harness processes
2. Check for stale locks: `fuser .harness/state.db`
3. Force unlock (use with caution): `rm .harness/state.db-wal .harness/state.db-shm`

## Getting Help

1. **Check logs:** Run with `RUST_LOG=debug` for verbose output
2. **Validate graph:** `trinity-harness query validate`
3. **Export state:** `trinity-harness export --format json > state.json`
4. **File an issue:** Include graph stats and reproduction steps
