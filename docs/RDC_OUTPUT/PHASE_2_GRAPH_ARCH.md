# PHASE 2: Code Graph — Architecture

**Duration:** 1 day
**Depends On:** Phase 1 (Infrastructure)

---

## Overview

Parse all Rust, Python, and WGSL files. Build the unified code graph. All nodes start as UNKNOWN.

## Components

### 2.1 Full Scan

```rust
pub struct GraphBuilder {
    db: HarnessDb,
    parsers: ParserRegistry,
}

impl GraphBuilder {
    pub fn full_scan(&self, roots: &[PathBuf]) -> Result<ScanResult> {
        for root in roots {
            for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                let lang = Language::from_path(path);
                if lang != Language::Unknown {
                    self.parse_and_insert(path, lang)?;
                }
            }
        }
    }
}
```

### 2.2 Node Insertion

```rust
pub fn insert_node(&self, unit: &CodeUnit) -> Result<NodeId> {
    self.db.execute(
        "INSERT INTO code_nodes (...) VALUES (...)",
        params![...]
    )?;
    self.db.append_event(&CodeEvent::NodeCreated { node_id })?;
    Ok(node_id)
}
```

### 2.3 Edge Detection

Detect dependencies from AST:
- **Rust:** `use`, function calls, type references
- **Python:** `import`, function calls, class inheritance
- **Cross-language:** #[pyfunction], MirrorsLayout detection

## Acceptance Criteria

- [ ] Full scan of all source directories completes
- [ ] All Rust files parsed, nodes inserted
- [ ] All Python files parsed, nodes inserted
- [ ] All WGSL files parsed, nodes inserted
- [ ] Dependencies detected and edges created
- [ ] All nodes have state = UNKNOWN
