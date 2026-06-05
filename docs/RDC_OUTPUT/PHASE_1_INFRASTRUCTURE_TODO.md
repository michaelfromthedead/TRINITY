# PHASE 1: Infrastructure — TODO

**Duration:** 1-2 days

---

## Tasks

### T-HARNESS-1.1: Create crate skeleton ✓
- [x] Create `crates/trinity-harness/Cargo.toml`
- [x] Add dependencies: superrusqlite, syn, rustpython_parser, naga, tree-sitter-*
- [x] Create module structure (db.rs, parsers/, graph/, state/)

### T-HARNESS-1.2: SuperSQLite connection ✓
- [x] Implement `HarnessDb::open(path)`
- [x] Configure pragmas (WAL, cache)
- [x] Verify extensions loaded (`SELECT core_version()`) — deferred: SuperSQLite in DESIGN PHASE

### T-HARNESS-1.3: Database schema ✓
- [x] Create `schema.sql` with tables: code_nodes, code_edges, code_events, code_state_history, code_contracts, struct_layouts
- [x] Add indexes for common queries
- [x] Test schema creation on fresh database

### T-HARNESS-1.4: Rust parser ✓
- [x] Implement `RustParser` with syn + tree-sitter
- [x] Extract: functions, structs, enums, impls, modules
- [x] Compute hashes: full, signature, body, layout
- [x] Return `Vec<RustUnit>`

### T-HARNESS-1.5: Python parser
- [ ] Implement `PythonParser` with rustpython_parser + tree-sitter
- [ ] Extract: functions, classes, methods, imports
- [ ] Compute hashes
- [ ] Return `Vec<PythonUnit>`

### T-HARNESS-1.6: WGSL parser
- [ ] Implement `WgslParser` with naga + tree-sitter
- [ ] Extract: structs with member offsets, functions, entry points, bindings
- [ ] **Critical:** Capture struct layout (offset, size) for alignment checking
- [ ] Return `Vec<WgslUnit>`

### T-HARNESS-1.7: Unified CodeUnit
- [ ] Define `CodeUnit` enum spanning all languages
- [ ] Implement `ParserRegistry::parse_file()`
- [ ] Test on sample files from each language

### T-HARNESS-1.8: Basic tests
- [ ] Test schema creation
- [ ] Test parsing sample Rust file
- [ ] Test parsing sample Python file
- [ ] Test parsing sample WGSL file

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-HARNESS-1.1 | 1h | 2h | 4h |
| T-HARNESS-1.2 | 1h | 2h | 4h |
| T-HARNESS-1.3 | 2h | 4h | 8h |
| T-HARNESS-1.4 | 4h | 8h | 16h |
| T-HARNESS-1.5 | 2h | 4h | 8h |
| T-HARNESS-1.6 | 2h | 4h | 8h |
| T-HARNESS-1.7 | 2h | 4h | 8h |
| T-HARNESS-1.8 | 1h | 2h | 4h |
| **Total** | **15h** | **30h** | **60h** |
