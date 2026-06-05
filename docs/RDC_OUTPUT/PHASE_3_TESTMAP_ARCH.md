# PHASE 3: Test Mapping — Architecture

**Duration:** 2-3 days
**Depends On:** Phase 2 (Code Graph)

---

## Overview

Map existing 12,743 tests to code nodes. Create "tests" edges in the graph.

## Mapping Strategies

### 3.1 Auto-Mapping Rules

| Pattern | Rule |
|---------|------|
| `tests/blackbox_foo.rs` | Tests `src/foo.rs` |
| `#[test] fn test_xyz()` in `src/mod.rs` | Tests same file |
| `tests/unit/test_bar.py` | Tests `engine/bar.py` |
| `class TestFoo` | Tests class `Foo` |

### 3.2 Manual Mapping File

For tests that don't follow convention:

```toml
# test_mappings.toml
[[mapping]]
test = "tests/integration/test_render_pipeline.py"
targets = [
    "engine/render/pipeline.py",
    "engine/render/stages/*.py"
]
```

### 3.3 Edge Creation

```rust
pub fn create_test_edge(test_node: NodeId, target_node: NodeId) {
    self.db.insert_edge(CodeEdge {
        from: test_node,
        to: target_node,
        kind: EdgeKind::Tests,
    });
}
```

## Acceptance Criteria

- [ ] Auto-mapping rules implemented
- [ ] Manual mapping file parsed
- [ ] All test files mapped to targets
- [ ] Tests edges created in graph
- [ ] Coverage report: % of code nodes with tests
