# PHASE 1: Infrastructure — Architecture

**Duration:** 1-2 days
**Depends On:** Nothing (foundation)

---

## Overview

Build the foundation: harness crate skeleton, SuperSQLite connection, and multi-language parsers.

## Components

### 1.1 Harness Crate Structure

```
crates/trinity-harness/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── db.rs           # SuperSQLite connection
│   ├── schema.sql      # Database schema
│   ├── parsers/
│   │   ├── mod.rs
│   │   ├── rust.rs     # syn + tree-sitter
│   │   ├── python.rs   # rustpython + tree-sitter
│   │   └── wgsl.rs     # naga + tree-sitter
│   ├── graph/
│   │   ├── mod.rs
│   │   ├── nodes.rs
│   │   └── edges.rs
│   └── state/
│       ├── mod.rs
│       └── machine.rs  # superstate binding
```

### 1.2 SuperSQLite Setup

```rust
pub struct HarnessDb {
    conn: Connection,
}

impl HarnessDb {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
        "#)?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }
}
```

### 1.3 Parser Registry

```rust
pub struct ParserRegistry {
    rust: RustParser,
    python: PythonParser,
    wgsl: WgslParser,
}

impl ParserRegistry {
    pub fn parse_file(&self, path: &Path, source: &str, lang: Language) 
        -> Vec<CodeUnit>;
}
```

## Acceptance Criteria

- [ ] `crates/trinity-harness` exists with basic structure
- [ ] SuperSQLite connects and creates schema
- [ ] Rust parser extracts functions and structs
- [ ] Python parser extracts functions and classes
- [ ] WGSL parser extracts structs with offset info
- [ ] All parsers return unified `CodeUnit` type
