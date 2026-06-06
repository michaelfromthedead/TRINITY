# Getting Started with Trinity Harness

## What This Does

You have 12,743 tests. Running them all takes 30 minutes. 

**The harness tracks which tests cover which code, so when you edit a file, it only runs the affected tests.**

```
Before:  Edit 1 file → Run all 12,743 tests → 30 minutes
After:   Edit 1 file → Run 50 affected tests → 30 seconds
```

## Quick Start (5 minutes)

### 1. Build the harness

```bash
cd /path/to/TRINITY
cargo build -p trinity-harness --release
```

### 2. Scan your codebase

```bash
./target/release/trinity-harness scan ./crates ./engine
```

This parses every `.rs`, `.py`, `.wgsl` file and figures out:
- What functions/structs exist
- Who calls whom
- Which test file tests which source file

Output:
```
Scanned 5,000 nodes
  Rust: 3,200
  Python: 1,500
  WGSL: 300
Mapped 800 tests to 2,000 code nodes
Unmapped: 50 tests (need manual mapping)
```

### 3. Run tests once to establish baseline

```bash
./target/release/trinity-harness run-all
```

This runs `cargo test` and `pytest`, then records which tests pass/fail:
```
Total: 12,743  Passed: 12,700  Failed: 43
Health: 99.7%
```

### 4. Start the daemon

```bash
./target/release/trinity-harness daemon
```

Now it watches your source files. When you edit something:

```
[daemon] File changed: crates/renderer/src/pipeline.rs
[daemon] Marking DIRTY: pipeline.rs (3 functions)
[daemon] Affected tests: test_pipeline.rs (12 tests)
[daemon] Running...
[daemon] 12/12 passed (1.2s)
[daemon] Health: 99.7%
```

**That's it.** Edit code, tests run automatically, only the ones that matter.

---

## How It Knows Which Tests to Run

### Automatic mapping (by naming convention)

| Test file | Maps to |
|-----------|---------|
| `tests/test_renderer.rs` | `src/renderer.rs` |
| `tests/blackbox_parser.rs` | `src/parser.rs` |
| `test_compute.py` | `compute.py` |

### Manual mapping (for edge cases)

If your test doesn't follow the convention, create `test_mappings.toml`:

```toml
[[mappings]]
test = "tests/integration/big_test.rs"
targets = ["src/renderer.rs", "src/gpu/*.rs"]
```

Then reload:
```bash
./target/release/trinity-harness reload
```

---

## Commands Reference

| Command | What it does |
|---------|--------------|
| `scan <paths...>` | Parse source files, build the graph |
| `run-all` | Run all tests, establish baseline |
| `daemon` | Watch files, auto-run affected tests |
| `query needs-testing` | List tests that need to run |
| `run-stale` | Run only the stale tests |
| `status` | Show current health (green/red/dirty counts) |

---

## Example Session

```bash
# Initial setup (once)
$ trinity-harness scan ./crates ./engine
$ trinity-harness run-all

# Daily work
$ trinity-harness daemon &

# You edit crates/renderer/src/shader.rs
# Terminal shows:
[daemon] shader.rs changed
[daemon] Running 8 affected tests...
[daemon] 8/8 passed (0.9s)

# You edit crates/renderer/src/buffer.rs (introduces a bug)
[daemon] buffer.rs changed
[daemon] Running 15 affected tests...
[daemon] 13/15 passed, 2 FAILED
[daemon] FAILED: test_buffer_allocation, test_buffer_resize

# You fix the bug
[daemon] buffer.rs changed
[daemon] Running 2 failed tests...
[daemon] 2/2 passed (0.3s)
[daemon] Health restored: 99.7%
```

---

## Advanced (optional)

These are extra features you don't need to get started:

- **Contracts** — Add `#[requires(x > 0)]` to functions, auto-generates tests
- **Layout checks** — Verify Rust struct sizes match GPU shader structs
- **CI integration** — GitHub Actions workflow that only runs affected tests

See the other howto guides for these.
