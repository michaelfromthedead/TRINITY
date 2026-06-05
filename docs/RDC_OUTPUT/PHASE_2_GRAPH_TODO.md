# PHASE 2: Code Graph — TODO

**Duration:** 1 day

---

## Tasks

### T-GRAPH-2.1: Implement GraphBuilder ✓
- [x] Create `graph/builder.rs`
- [x] Implement `full_scan()` with walkdir
- [x] Filter by language extension

### T-GRAPH-2.2: Parse all Rust files ✓
- [x] Scan `crates/` directory
- [x] Parse each .rs file
- [x] Insert nodes for functions, structs, modules

### T-GRAPH-2.3: Parse all Python files ✓
- [x] Scan `engine/` and `tests/` directories
- [x] Parse each .py file
- [x] Insert nodes for functions, classes, methods

### T-GRAPH-2.4: Parse all WGSL files ✓
- [x] Scan `crates/renderer-backend/shaders/`
- [x] Parse each .wgsl file
- [x] Insert nodes for structs (with layout!), functions, entry points

### T-GRAPH-2.5: Dependency detection ✓
- [x] Implement Rust dependency detection (use, calls, types)
- [x] Implement Python dependency detection (import, calls)
- [x] Create edges for dependencies

### T-GRAPH-2.6: Cross-language edges ✓
- [x] Detect PyO3 boundaries (#[pyfunction], #[pyclass])
- [x] Detect WGSL↔Rust struct mirrors (same name, #[repr(C)])
- [x] Create MirrorsLayout edges

### T-GRAPH-2.7: Validate graph
- [ ] Query node count by language
- [ ] Query edge count by type
- [ ] Verify no orphan nodes (except entry points)

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-GRAPH-2.1 | 1h | 2h | 4h |
| T-GRAPH-2.2 | 2h | 4h | 8h |
| T-GRAPH-2.3 | 2h | 4h | 8h |
| T-GRAPH-2.4 | 1h | 2h | 4h |
| T-GRAPH-2.5 | 2h | 4h | 8h |
| T-GRAPH-2.6 | 2h | 4h | 8h |
| T-GRAPH-2.7 | 1h | 2h | 4h |
| **Total** | **11h** | **22h** | **44h** |
