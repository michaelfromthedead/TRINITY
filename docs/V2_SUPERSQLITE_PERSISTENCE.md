# V2 SuperSQLite Persistence Layer

**Started:** 2026-06-04
**Status:** DESIGN PHASE
**Depends On:** V2_SUPERSTATE_VISION.md

---

## Executive Summary

The V2 Superstate Vision requires persistent storage for the code graph, state history, and event log. We will use **SuperSQLite** — a custom SQLite distribution with 15+ extensions — as our persistence layer.

SuperSQLite provides:
- **Graph storage** for code dependencies (supersqlite_graph)
- **Event sourcing** for code changes (supersqlite_streams)
- **Bitemporal queries** for state history (supersqlite_bitemporal)
- **Vector search** for semantic code search (supersqlite_vector)
- **In-memory cache** for fast state access (supersqlite_memory)
- **Time-series** for metrics over time (supersqlite_timeseries)

All in a single `brain.db` file with zero network overhead.

---

## Part 1: Why SuperSQLite?

### The Requirements

| Requirement | Solution |
|-------------|----------|
| Store code graph (nodes + edges) | supersqlite_graph |
| Track state changes over time | supersqlite_bitemporal |
| Event log for all changes | supersqlite_streams |
| Query "what was state at time X?" | Bitemporal AS OF queries |
| Find similar code semantically | supersqlite_vector |
| Fast in-memory state cache | supersqlite_memory |
| Metrics and observability | supersqlite_metrics |

### Why Not Just SQLite?

Plain SQLite lacks:
- Graph traversal algorithms (we'd have to implement them)
- Bitemporal queries (we'd have to build the temporal model)
- Event streaming primitives (we'd have to build the log)
- Vector similarity search (we'd need a separate system)

SuperSQLite bundles all of this into one coherent system.

### Why Not Postgres/Neo4j/Redis/Pinecone?

| System | Problem |
|--------|---------|
| PostgreSQL | Network overhead, server process, overkill for local tool |
| Neo4j | Heavy JVM, separate graph database, complexity |
| Redis | Separate process, no persistence guarantees by default |
| Pinecone | Cloud service, network latency, cost |

SuperSQLite is:
- **Embedded** — In-process, ~0.1ms latency
- **Single file** — `brain.db` is the entire state
- **Portable** — Copy the file, that's your backup
- **Unified** — Graph, vector, events, cache in one system

---

## Part 2: SuperSQLite Architecture

### Location

```
/home/user/dev/USER/PROJECTS_VOID/SQLITE/platform/
├── supersqlite/              # 15 extension crates
│   ├── supersqlite_core/     # Foundation (required by all)
│   ├── supersqlite_graph/    # Graph storage & algorithms
│   ├── supersqlite_streams/  # Event streaming
│   ├── supersqlite_bitemporal/ # Temporal queries
│   ├── supersqlite_vector/   # Vector similarity
│   ├── supersqlite_memory/   # In-memory cache
│   ├── supersqlite_timeseries/ # Time-series
│   └── ...
├── supersqlite-sys/          # Compiles SQLite + all extensions
├── superrusqlite/            # Rusqlite-compatible API
└── superfossil/              # Fossil VCS (not used for this)
```

### Extension Overview

| Extension | SQL Functions | Equivalent To | Our Use Case |
|-----------|:-------------:|---------------|--------------|
| supersqlite_core | 12 | systemd | Runtime, health, metrics foundation |
| supersqlite_graph | 41 | Neo4j | Code dependency graph |
| supersqlite_streams | 33 | Kafka | Event log (CodeEvents) |
| supersqlite_bitemporal | 8 | SQL:2011 | State history queries |
| supersqlite_vector | 25 | Pinecone | Code embeddings search |
| supersqlite_memory | 53 | Redis | Fast state cache |
| supersqlite_timeseries | 33 | TimescaleDB | Metrics over time |
| supersqlite_stats | 23 | pg_stat | Query statistics |
| supersqlite_metrics | 23 | Prometheus | Application metrics |

**Total: 300+ SQL functions available**

---

## Part 3: Dependency Setup

### Cargo.toml

```toml
[dependencies]
# SuperSQLite core (provides sqlite3 + all extensions)
supersqlite-sys = { path = "../SQLITE/platform/supersqlite-sys" }

# Rusqlite-compatible API
superrusqlite = { path = "../SQLITE/platform/superrusqlite" }

# Or if using rusqlite directly, patch it:
# [patch.crates-io]
# rusqlite = { path = "../SQLITE/platform/superrusqlite" }
```

### Connection Setup

```rust
use superrusqlite::{Connection, params, Result};

pub struct HarnessDb {
    conn: Connection,
}

impl HarnessDb {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        // Configure for performance
        conn.execute_batch(r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;  -- 64MB
            PRAGMA mmap_size = 268435456; -- 256MB
            PRAGMA foreign_keys = ON;
        "#)?;
        
        // Verify extensions loaded
        let version: String = conn.query_row(
            "SELECT core_version()", 
            [], 
            |row| row.get(0)
        )?;
        println!("SuperSQLite core: {}", version);
        
        // Initialize schema
        Self::init_schema(&conn)?;
        
        Ok(Self { conn })
    }
    
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(include_str!("schema.sql"))?;
        Ok(())
    }
}
```

---

## Part 4: Schema Design

### 4.1 Code Nodes Table

```sql
-- =============================================================================
-- CODE NODES — Every code unit in the graph
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_nodes (
    -- Primary key
    node_id TEXT PRIMARY KEY,
    
    -- Location
    file_path TEXT NOT NULL,
    span_start_line INTEGER NOT NULL,
    span_start_col INTEGER NOT NULL,
    span_end_line INTEGER NOT NULL,
    span_end_col INTEGER NOT NULL,
    
    -- Identity
    language TEXT NOT NULL CHECK (language IN ('rust', 'python', 'wgsl', 'toml', 'json')),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT,
    
    -- Hashes for change detection
    hash_full BLOB NOT NULL,
    hash_signature BLOB,
    hash_body BLOB,
    hash_layout BLOB,  -- For structs (catches alignment bugs)
    
    -- Current state (denormalized for fast queries)
    current_state TEXT NOT NULL DEFAULT 'unknown' CHECK (current_state IN (
        'unknown', 'untouched', 'changed',
        'tested_green', 'tested_red', 'tested_skipped',
        'stale_direct', 'stale_transitive', 'stale_deep',
        'qa_approved', 'qa_flagged', 'quarantined', 'deprecated'
    )),
    
    -- Hierarchy
    parent_id TEXT REFERENCES code_nodes(node_id) ON DELETE SET NULL,
    depth INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_tested_at TEXT,
    last_changed_at TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_file ON code_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_language ON code_nodes(language);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON code_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_state ON code_nodes(current_state);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON code_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_qualified ON code_nodes(qualified_name);
```

### Node Kind Enumeration

```sql
-- Rust node kinds
-- 'rust_crate', 'rust_module', 'rust_function', 'rust_struct', 
-- 'rust_enum', 'rust_impl', 'rust_trait', 'rust_macro',
-- 'rust_const', 'rust_static', 'rust_type_alias'

-- Python node kinds
-- 'python_package', 'python_module', 'python_class',
-- 'python_function', 'python_method', 'python_variable'

-- WGSL node kinds
-- 'wgsl_shader', 'wgsl_struct', 'wgsl_function',
-- 'wgsl_entry_point', 'wgsl_binding', 'wgsl_const'
```

### 4.2 Code Edges Table

```sql
-- =============================================================================
-- CODE EDGES — Dependencies and relationships between nodes
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_edges (
    edge_id TEXT PRIMARY KEY,
    from_node TEXT NOT NULL REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    to_node TEXT NOT NULL REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN (
        -- Structural
        'contains',
        -- Dependencies
        'imports', 'calls', 'references', 'inherits', 'implements',
        -- Testing
        'tests', 'tested_by',
        -- Cross-language
        'pyo3_call', 'pyo3_callback', 'uses_shader', 'mirrors_layout'
    )),
    
    -- Optional metadata
    metadata TEXT,  -- JSON
    
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    -- Prevent duplicate edges
    UNIQUE(from_node, to_node, kind)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_edges_from ON code_edges(from_node);
CREATE INDEX IF NOT EXISTS idx_edges_to ON code_edges(to_node);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON code_edges(kind);

-- Register with graph extension
SELECT graph_register('code', 'code_nodes', 'node_id', 'code_edges', 'from_node', 'to_node');
```

### 4.3 State History Table (Bitemporal)

```sql
-- =============================================================================
-- CODE STATE HISTORY — Full temporal history of all state changes
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_state_history (
    -- Identity
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    
    -- The state
    state TEXT NOT NULL,
    
    -- Bitemporal columns
    -- valid_time: when the state was actually true in the real world
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL DEFAULT '9999-12-31T23:59:59',
    
    -- system_time: when we recorded this fact
    system_from TEXT NOT NULL DEFAULT (datetime('now')),
    system_to TEXT NOT NULL DEFAULT '9999-12-31T23:59:59',
    
    -- What caused this transition
    caused_by_event_id INTEGER,
    caused_by_event_type TEXT,
    
    -- Previous state (for debugging)
    previous_state TEXT
);

-- Indexes for temporal queries
CREATE INDEX IF NOT EXISTS idx_history_node ON code_state_history(node_id);
CREATE INDEX IF NOT EXISTS idx_history_valid ON code_state_history(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_history_system ON code_state_history(system_from, system_to);
CREATE INDEX IF NOT EXISTS idx_history_state ON code_state_history(state);

-- Register with bitemporal extension
SELECT bitemporal_register(
    'code_state_history',
    'valid_from', 'valid_to',
    'system_from', 'system_to'
);
```

### 4.4 Event Log Table (Streams)

```sql
-- =============================================================================
-- CODE EVENTS — Append-only event log
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_events (
    -- Sequence number (monotonically increasing)
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Timestamp
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    
    -- Event identity
    event_type TEXT NOT NULL CHECK (event_type IN (
        -- Change events
        'source_changed', 'signature_changed', 'body_changed', 'layout_changed',
        -- Dependency events
        'dependency_changed', 'transitive_dependency_changed',
        -- Test events
        'tests_passed', 'tests_failed', 'tests_skipped',
        -- QA events
        'qa_approved', 'qa_flagged',
        -- Lifecycle events
        'quarantine', 'release', 'deprecated',
        -- System events
        'node_created', 'node_deleted', 'edge_created', 'edge_deleted',
        'full_reparse', 'propagation_complete'
    )),
    
    -- What node(s) are affected
    node_id TEXT,
    
    -- Event payload (JSON)
    payload TEXT,
    
    -- For idempotency (prevent duplicate events)
    idempotency_key TEXT UNIQUE,
    
    -- Correlation (for tracing event chains)
    correlation_id TEXT,
    causation_id INTEGER REFERENCES code_events(sequence)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_type ON code_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_node ON code_events(node_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON code_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON code_events(correlation_id);

-- Register with streams extension
SELECT stream_register('code_events', 'sequence', 'timestamp');
```

### 4.5 Contracts Table

```sql
-- =============================================================================
-- CODE CONTRACTS — Pre/post conditions and properties
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_contracts (
    node_id TEXT PRIMARY KEY REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    
    -- Preconditions (what callers must provide)
    requires TEXT,  -- JSON array of predicates
    
    -- Postconditions (what this promises to return)
    ensures TEXT,   -- JSON array of predicates
    
    -- Invariants (what must always hold)
    invariants TEXT, -- JSON array of predicates
    
    -- Properties (algebraic laws)
    properties TEXT, -- JSON array of properties
    
    -- Verification status
    last_verified_at TEXT,
    verification_result TEXT,  -- 'passed', 'failed', 'timeout', 'error'
    verification_details TEXT, -- JSON
    
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

### 4.6 Struct Layouts Table (Alignment Bug Detection)

```sql
-- =============================================================================
-- STRUCT LAYOUTS — For detecting Rust/WGSL alignment mismatches
-- =============================================================================

CREATE TABLE IF NOT EXISTS struct_layouts (
    node_id TEXT PRIMARY KEY REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    
    -- Layout details
    total_size INTEGER NOT NULL,
    alignment INTEGER NOT NULL,
    
    -- Members as JSON array
    -- [{name: "field", offset: 0, size: 4, type: "u32"}, ...]
    members TEXT NOT NULL,
    
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- View to find mismatches between Rust and WGSL structs
CREATE VIEW IF NOT EXISTS layout_mismatches AS
SELECT 
    rn.node_id AS rust_node_id,
    rn.name AS struct_name,
    rn.file_path AS rust_file,
    wn.node_id AS wgsl_node_id,
    wn.file_path AS wgsl_file,
    rs.total_size AS rust_size,
    ws.total_size AS wgsl_size,
    rs.alignment AS rust_align,
    ws.alignment AS wgsl_align,
    rs.members AS rust_members,
    ws.members AS wgsl_members
FROM code_nodes rn
JOIN code_nodes wn ON rn.name = wn.name
JOIN struct_layouts rs ON rn.node_id = rs.node_id
JOIN struct_layouts ws ON wn.node_id = ws.node_id
WHERE rn.language = 'rust' 
  AND wn.language = 'wgsl'
  AND rn.kind = 'rust_struct' 
  AND wn.kind = 'wgsl_struct'
  AND (rs.total_size != ws.total_size 
       OR rs.members != ws.members);
```

### 4.7 Code Embeddings Table (Vector Search)

```sql
-- =============================================================================
-- CODE EMBEDDINGS — Vector representations for semantic search
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_embeddings (
    node_id TEXT PRIMARY KEY REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    
    -- The embedding vector (stored as blob)
    embedding BLOB NOT NULL,
    
    -- Embedding model used
    model TEXT NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    dimensions INTEGER NOT NULL DEFAULT 384,
    
    -- What was embedded (for cache invalidation)
    source_hash BLOB NOT NULL,
    
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Register with vector extension
SELECT vec_register('code_embeddings', 'embedding', 384);

-- Create HNSW index for fast similarity search
SELECT vec_index_create('code_embeddings', 'embedding', 'hnsw', 384);
```

### 4.8 Test Results Table

```sql
-- =============================================================================
-- TEST RESULTS — Test execution history
-- =============================================================================

CREATE TABLE IF NOT EXISTS test_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- What was tested
    node_id TEXT NOT NULL REFERENCES code_nodes(node_id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    
    -- Result
    outcome TEXT NOT NULL CHECK (outcome IN ('passed', 'failed', 'skipped', 'error')),
    duration_ms INTEGER,
    
    -- Details
    error_message TEXT,
    stack_trace TEXT,
    stdout TEXT,
    stderr TEXT,
    
    -- When
    executed_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    -- Linked to event
    event_id INTEGER REFERENCES code_events(sequence)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_results_node ON test_results(node_id);
CREATE INDEX IF NOT EXISTS idx_results_outcome ON test_results(outcome);
CREATE INDEX IF NOT EXISTS idx_results_executed ON test_results(executed_at);
```

### 4.9 Consumer Cursors (for Stream Processing)

```sql
-- =============================================================================
-- STREAM CURSORS — Track where each consumer is in the event stream
-- =============================================================================

CREATE TABLE IF NOT EXISTS stream_cursors (
    cursor_name TEXT PRIMARY KEY,
    stream_name TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Pre-create cursors
INSERT OR IGNORE INTO stream_cursors (cursor_name, stream_name) VALUES
    ('propagation_engine', 'code_events'),
    ('state_updater', 'code_events'),
    ('test_runner', 'code_events'),
    ('notification_service', 'code_events');
```

---

## Part 5: Rust API Layer

### 5.1 Node Operations

```rust
use superrusqlite::{Connection, params, Result};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    pub node_id: String,
    pub file_path: String,
    pub language: Language,
    pub kind: String,
    pub name: String,
    pub qualified_name: Option<String>,
    pub span: Span,
    pub hashes: NodeHashes,
    pub current_state: CodeState,
    pub parent_id: Option<String>,
}

impl HarnessDb {
    /// Insert or update a code node
    pub fn upsert_node(&self, node: &CodeNode) -> Result<()> {
        self.conn.execute(r#"
            INSERT INTO code_nodes (
                node_id, file_path, language, kind, name, qualified_name,
                span_start_line, span_start_col, span_end_line, span_end_col,
                hash_full, hash_signature, hash_body, hash_layout,
                current_state, parent_id, updated_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6,
                ?7, ?8, ?9, ?10,
                ?11, ?12, ?13, ?14,
                ?15, ?16, datetime('now')
            )
            ON CONFLICT(node_id) DO UPDATE SET
                file_path = excluded.file_path,
                language = excluded.language,
                kind = excluded.kind,
                name = excluded.name,
                qualified_name = excluded.qualified_name,
                span_start_line = excluded.span_start_line,
                span_start_col = excluded.span_start_col,
                span_end_line = excluded.span_end_line,
                span_end_col = excluded.span_end_col,
                hash_full = excluded.hash_full,
                hash_signature = excluded.hash_signature,
                hash_body = excluded.hash_body,
                hash_layout = excluded.hash_layout,
                current_state = excluded.current_state,
                parent_id = excluded.parent_id,
                updated_at = datetime('now')
        "#, params![
            node.node_id,
            node.file_path,
            node.language.as_str(),
            node.kind,
            node.name,
            node.qualified_name,
            node.span.start_line,
            node.span.start_col,
            node.span.end_line,
            node.span.end_col,
            node.hashes.full,
            node.hashes.signature,
            node.hashes.body,
            node.hashes.layout,
            node.current_state.as_str(),
            node.parent_id,
        ])?;
        Ok(())
    }
    
    /// Get a node by ID
    pub fn get_node(&self, node_id: &str) -> Result<Option<CodeNode>> {
        self.conn.query_row(
            "SELECT * FROM code_nodes WHERE node_id = ?1",
            params![node_id],
            |row| Self::row_to_node(row),
        ).optional()
    }
    
    /// Get all nodes in a given state
    pub fn nodes_in_state(&self, state: CodeState) -> Result<Vec<CodeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM code_nodes WHERE current_state = ?1"
        )?;
        let nodes = stmt.query_map(params![state.as_str()], |row| {
            Self::row_to_node(row)
        })?.collect::<Result<Vec<_>>>()?;
        Ok(nodes)
    }
    
    /// Get all stale nodes
    pub fn stale_nodes(&self) -> Result<Vec<CodeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM code_nodes WHERE current_state IN ('stale_direct', 'stale_transitive', 'stale_deep')"
        )?;
        let nodes = stmt.query_map([], |row| {
            Self::row_to_node(row)
        })?.collect::<Result<Vec<_>>>()?;
        Ok(nodes)
    }
    
    /// Get children of a node
    pub fn children(&self, parent_id: &str) -> Result<Vec<CodeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM code_nodes WHERE parent_id = ?1 ORDER BY name"
        )?;
        let nodes = stmt.query_map(params![parent_id], |row| {
            Self::row_to_node(row)
        })?.collect::<Result<Vec<_>>>()?;
        Ok(nodes)
    }
}
```

### 5.2 Edge Operations

```rust
impl HarnessDb {
    /// Add an edge between nodes
    pub fn add_edge(&self, from: &str, to: &str, kind: EdgeKind) -> Result<String> {
        let edge_id = format!("edge_{}_{}", from, to);
        self.conn.execute(r#"
            INSERT OR IGNORE INTO code_edges (edge_id, from_node, to_node, kind)
            VALUES (?1, ?2, ?3, ?4)
        "#, params![edge_id, from, to, kind.as_str()])?;
        Ok(edge_id)
    }
    
    /// Get all edges from a node
    pub fn edges_from(&self, node_id: &str) -> Result<Vec<CodeEdge>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM code_edges WHERE from_node = ?1"
        )?;
        stmt.query_map(params![node_id], |row| Self::row_to_edge(row))?
            .collect()
    }
    
    /// Get all edges to a node
    pub fn edges_to(&self, node_id: &str) -> Result<Vec<CodeEdge>> {
        let mut stmt = self.conn.prepare(
            "SELECT * FROM code_edges WHERE to_node = ?1"
        )?;
        stmt.query_map(params![node_id], |row| Self::row_to_edge(row))?
            .collect()
    }
    
    /// Find dependents using graph extension
    pub fn dependents(&self, node_id: &str, depth: u32) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT node_id FROM graph_traverse('code', ?1, 'incoming', ?2)
        "#)?;
        stmt.query_map(params![node_id, depth], |row| row.get(0))?
            .collect()
    }
    
    /// Find dependencies using graph extension
    pub fn dependencies(&self, node_id: &str, depth: u32) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT node_id FROM graph_traverse('code', ?1, 'outgoing', ?2)
        "#)?;
        stmt.query_map(params![node_id, depth], |row| row.get(0))?
            .collect()
    }
    
    /// Find shortest path between nodes
    pub fn shortest_path(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT path FROM graph_shortest_path('code', ?1, ?2)
        "#)?;
        let path: String = stmt.query_row(params![from, to], |row| row.get(0))?;
        Ok(serde_json::from_str(&path)?)
    }
}
```

### 5.3 Event Operations (Streams)

```rust
impl HarnessDb {
    /// Append an event to the log
    pub fn append_event(&self, event: &CodeEvent) -> Result<i64> {
        let sequence: i64 = self.conn.query_row(r#"
            INSERT INTO code_events (event_type, node_id, payload, idempotency_key, correlation_id, causation_id)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            RETURNING sequence
        "#, params![
            event.event_type.as_str(),
            event.node_id,
            event.payload.as_ref().map(|p| serde_json::to_string(p).unwrap()),
            event.idempotency_key,
            event.correlation_id,
            event.causation_id,
        ], |row| row.get(0))?;
        
        Ok(sequence)
    }
    
    /// Read events from the stream
    pub fn read_events(&self, from_sequence: i64, limit: u32) -> Result<Vec<StoredEvent>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT * FROM code_events 
            WHERE sequence > ?1 
            ORDER BY sequence 
            LIMIT ?2
        "#)?;
        stmt.query_map(params![from_sequence, limit], |row| Self::row_to_event(row))?
            .collect()
    }
    
    /// Read events by type
    pub fn read_events_by_type(&self, event_type: &str, from: i64, limit: u32) -> Result<Vec<StoredEvent>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT * FROM code_events 
            WHERE event_type = ?1 AND sequence > ?2
            ORDER BY sequence 
            LIMIT ?3
        "#)?;
        stmt.query_map(params![event_type, from, limit], |row| Self::row_to_event(row))?
            .collect()
    }
    
    /// Get cursor position for a consumer
    pub fn get_cursor(&self, cursor_name: &str) -> Result<i64> {
        self.conn.query_row(
            "SELECT last_sequence FROM stream_cursors WHERE cursor_name = ?1",
            params![cursor_name],
            |row| row.get(0),
        )
    }
    
    /// Update cursor position
    pub fn set_cursor(&self, cursor_name: &str, sequence: i64) -> Result<()> {
        self.conn.execute(r#"
            UPDATE stream_cursors 
            SET last_sequence = ?2, updated_at = datetime('now')
            WHERE cursor_name = ?1
        "#, params![cursor_name, sequence])?;
        Ok(())
    }
}
```

### 5.4 State History Operations (Bitemporal)

```rust
impl HarnessDb {
    /// Record a state change
    pub fn record_state_change(
        &self,
        node_id: &str,
        new_state: CodeState,
        event_id: Option<i64>,
        event_type: Option<&str>,
    ) -> Result<()> {
        // Get current state
        let current: Option<String> = self.conn.query_row(
            "SELECT current_state FROM code_nodes WHERE node_id = ?1",
            params![node_id],
            |row| row.get(0),
        ).optional()?;
        
        let now = chrono::Utc::now().to_rfc3339();
        
        // Close the current state record
        self.conn.execute(r#"
            UPDATE code_state_history 
            SET valid_to = ?2
            WHERE node_id = ?1 AND valid_to = '9999-12-31T23:59:59'
        "#, params![node_id, now])?;
        
        // Insert new state record
        self.conn.execute(r#"
            INSERT INTO code_state_history (
                node_id, state, valid_from, previous_state,
                caused_by_event_id, caused_by_event_type
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#, params![
            node_id,
            new_state.as_str(),
            now,
            current,
            event_id,
            event_type,
        ])?;
        
        // Update denormalized current_state
        self.conn.execute(r#"
            UPDATE code_nodes 
            SET current_state = ?2, updated_at = datetime('now')
            WHERE node_id = ?1
        "#, params![node_id, new_state.as_str()])?;
        
        Ok(())
    }
    
    /// Query state at a specific time (AS OF)
    pub fn state_at(&self, node_id: &str, at_time: &str) -> Result<Option<String>> {
        self.conn.query_row(r#"
            SELECT state FROM code_state_history
            WHERE node_id = ?1
              AND valid_from <= ?2
              AND valid_to > ?2
            ORDER BY system_from DESC
            LIMIT 1
        "#, params![node_id, at_time], |row| row.get(0)).optional()
    }
    
    /// Query state history between times
    pub fn state_between(&self, node_id: &str, from: &str, to: &str) -> Result<Vec<StateHistoryEntry>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT * FROM code_state_history
            WHERE node_id = ?1
              AND valid_from < ?3
              AND valid_to > ?2
            ORDER BY valid_from
        "#)?;
        stmt.query_map(params![node_id, from, to], |row| Self::row_to_history(row))?
            .collect()
    }
    
    /// Get full state timeline for a node
    pub fn state_timeline(&self, node_id: &str) -> Result<Vec<StateHistoryEntry>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT * FROM code_state_history
            WHERE node_id = ?1
            ORDER BY valid_from
        "#)?;
        stmt.query_map(params![node_id], |row| Self::row_to_history(row))?
            .collect()
    }
}
```

### 5.5 Vector Search Operations

```rust
impl HarnessDb {
    /// Store code embedding
    pub fn store_embedding(&self, node_id: &str, embedding: &[f32], source_hash: &[u8]) -> Result<()> {
        let embedding_blob = bytemuck::cast_slice(embedding);
        self.conn.execute(r#"
            INSERT OR REPLACE INTO code_embeddings (node_id, embedding, source_hash)
            VALUES (?1, ?2, ?3)
        "#, params![node_id, embedding_blob, source_hash])?;
        Ok(())
    }
    
    /// Find similar code by embedding
    pub fn find_similar(&self, query_embedding: &[f32], limit: u32) -> Result<Vec<(String, f32)>> {
        let query_blob = bytemuck::cast_slice(query_embedding);
        let mut stmt = self.conn.prepare(r#"
            SELECT 
                node_id, 
                vec_distance_cosine(embedding, ?1) as distance
            FROM code_embeddings
            ORDER BY distance
            LIMIT ?2
        "#)?;
        stmt.query_map(params![query_blob, limit], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?.collect()
    }
    
    /// Find code similar to a given node
    pub fn find_similar_to(&self, node_id: &str, limit: u32) -> Result<Vec<(String, f32)>> {
        let mut stmt = self.conn.prepare(r#"
            SELECT 
                e2.node_id,
                vec_distance_cosine(e1.embedding, e2.embedding) as distance
            FROM code_embeddings e1
            JOIN code_embeddings e2 ON e1.node_id != e2.node_id
            WHERE e1.node_id = ?1
            ORDER BY distance
            LIMIT ?2
        "#)?;
        stmt.query_map(params![node_id, limit], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })?.collect()
    }
}
```

### 5.6 Layout Mismatch Detection

```rust
impl HarnessDb {
    /// Store struct layout
    pub fn store_layout(&self, node_id: &str, layout: &StructLayout) -> Result<()> {
        let members_json = serde_json::to_string(&layout.members)?;
        self.conn.execute(r#"
            INSERT OR REPLACE INTO struct_layouts (node_id, total_size, alignment, members)
            VALUES (?1, ?2, ?3, ?4)
        "#, params![node_id, layout.total_size, layout.alignment, members_json])?;
        Ok(())
    }
    
    /// Find all Rust/WGSL layout mismatches
    pub fn find_layout_mismatches(&self) -> Result<Vec<LayoutMismatch>> {
        let mut stmt = self.conn.prepare("SELECT * FROM layout_mismatches")?;
        stmt.query_map([], |row| {
            Ok(LayoutMismatch {
                rust_node_id: row.get(0)?,
                struct_name: row.get(1)?,
                rust_file: row.get(2)?,
                wgsl_node_id: row.get(3)?,
                wgsl_file: row.get(4)?,
                rust_size: row.get(5)?,
                wgsl_size: row.get(6)?,
                rust_align: row.get(7)?,
                wgsl_align: row.get(8)?,
                rust_members: serde_json::from_str(row.get_ref(9)?.as_str()?)?,
                wgsl_members: serde_json::from_str(row.get_ref(10)?.as_str()?)?,
            })
        })?.collect()
    }
}

#[derive(Debug)]
pub struct LayoutMismatch {
    pub rust_node_id: String,
    pub struct_name: String,
    pub rust_file: String,
    pub wgsl_node_id: String,
    pub wgsl_file: String,
    pub rust_size: u32,
    pub wgsl_size: u32,
    pub rust_align: u32,
    pub wgsl_align: u32,
    pub rust_members: Vec<LayoutMember>,
    pub wgsl_members: Vec<LayoutMember>,
}
```

---

## Part 6: Advanced Queries

### 6.1 Aggregate State Queries

```sql
-- State distribution for entire codebase
SELECT 
    current_state,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM code_nodes
GROUP BY current_state
ORDER BY count DESC;

-- State distribution by language
SELECT 
    language,
    current_state,
    COUNT(*) as count
FROM code_nodes
GROUP BY language, current_state
ORDER BY language, count DESC;

-- State distribution for a module
SELECT 
    current_state,
    COUNT(*) as count
FROM code_nodes
WHERE parent_id = 'module_gpu_driven'
GROUP BY current_state;

-- Find the "worst" state in each module
SELECT 
    parent_id as module,
    MAX(CASE current_state
        WHEN 'quarantined' THEN 7
        WHEN 'tested_red' THEN 6
        WHEN 'stale_direct' THEN 5
        WHEN 'stale_transitive' THEN 4
        WHEN 'stale_deep' THEN 3
        WHEN 'unknown' THEN 2
        WHEN 'untouched' THEN 1
        ELSE 0
    END) as worst_severity
FROM code_nodes
WHERE parent_id IS NOT NULL
GROUP BY parent_id
HAVING worst_severity > 0;
```

### 6.2 Impact Analysis Queries

```sql
-- What would be affected if I change this node?
-- Uses graph extension for transitive dependents
SELECT 
    n.node_id,
    n.name,
    n.file_path,
    n.current_state,
    t.depth
FROM graph_traverse('code', 'target_node_id', 'incoming', 10) t
JOIN code_nodes n ON t.node_id = n.node_id
ORDER BY t.depth, n.name;

-- Find all tests that test a given module
SELECT 
    n.node_id,
    n.name,
    n.file_path,
    e.kind
FROM code_edges e
JOIN code_nodes n ON e.from_node = n.node_id
WHERE e.to_node = 'module_id'
  AND e.kind = 'tests';

-- Find untested code
SELECT 
    n.node_id,
    n.name,
    n.file_path
FROM code_nodes n
LEFT JOIN code_edges e ON n.node_id = e.to_node AND e.kind = 'tests'
WHERE e.edge_id IS NULL
  AND n.kind IN ('rust_function', 'python_function')
  AND n.name NOT LIKE 'test_%';
```

### 6.3 Temporal Queries

```sql
-- What was the state of this module last week?
SELECT 
    node_id,
    state,
    valid_from,
    valid_to
FROM code_state_history
WHERE node_id LIKE 'module_gpu_driven%'
  AND valid_from <= datetime('now', '-7 days')
  AND valid_to > datetime('now', '-7 days');

-- State transitions in the last 24 hours
SELECT 
    h.node_id,
    n.name,
    h.previous_state,
    h.state,
    h.valid_from,
    h.caused_by_event_type
FROM code_state_history h
JOIN code_nodes n ON h.node_id = n.node_id
WHERE h.valid_from > datetime('now', '-24 hours')
ORDER BY h.valid_from DESC;

-- How long was this node in each state?
SELECT 
    state,
    SUM(
        julianday(COALESCE(
            CASE WHEN valid_to = '9999-12-31T23:59:59' THEN datetime('now') ELSE valid_to END,
            datetime('now')
        )) - julianday(valid_from)
    ) * 24 as hours_in_state
FROM code_state_history
WHERE node_id = 'target_node'
GROUP BY state
ORDER BY hours_in_state DESC;
```

### 6.4 Event Stream Queries

```sql
-- Recent events
SELECT * FROM code_events 
ORDER BY sequence DESC 
LIMIT 100;

-- Events for a specific node
SELECT * FROM code_events 
WHERE node_id = 'target_node'
ORDER BY sequence DESC;

-- Event type distribution
SELECT 
    event_type,
    COUNT(*) as count
FROM code_events
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY event_type
ORDER BY count DESC;

-- Follow event chain (causation)
WITH RECURSIVE event_chain AS (
    SELECT * FROM code_events WHERE sequence = 12345
    UNION ALL
    SELECT e.* FROM code_events e
    JOIN event_chain c ON e.causation_id = c.sequence
)
SELECT * FROM event_chain ORDER BY sequence;
```

### 6.5 Cross-Language Queries

```sql
-- Find all Python -> Rust boundaries (PyO3)
SELECT 
    py.name as python_func,
    py.file_path as python_file,
    rs.name as rust_func,
    rs.file_path as rust_file
FROM code_edges e
JOIN code_nodes py ON e.from_node = py.node_id
JOIN code_nodes rs ON e.to_node = rs.node_id
WHERE e.kind = 'pyo3_call';

-- Find all Rust -> WGSL shader usages
SELECT 
    rs.name as rust_func,
    rs.file_path as rust_file,
    wg.name as shader,
    wg.file_path as shader_file
FROM code_edges e
JOIN code_nodes rs ON e.from_node = rs.node_id
JOIN code_nodes wg ON e.to_node = wg.node_id
WHERE e.kind = 'uses_shader';

-- Find structs that exist in both Rust and WGSL
SELECT 
    r.name,
    r.file_path as rust_file,
    w.file_path as wgsl_file,
    CASE WHEN r.hash_layout = w.hash_layout THEN 'match' ELSE 'MISMATCH' END as layout_status
FROM code_nodes r
JOIN code_nodes w ON r.name = w.name
WHERE r.language = 'rust' AND r.kind = 'rust_struct'
  AND w.language = 'wgsl' AND w.kind = 'wgsl_struct';
```

---

## Part 7: Performance Considerations

### 7.1 WAL Mode

```sql
-- Enable Write-Ahead Logging for concurrent reads
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

**Benefits:**
- Multiple readers don't block writer
- Writers don't block readers
- Better crash recovery
- Faster commits

### 7.2 Cache Configuration

```sql
-- 64MB page cache (negative = KB, so -64000 = ~64MB)
PRAGMA cache_size = -64000;

-- 256MB memory-mapped I/O
PRAGMA mmap_size = 268435456;
```

### 7.3 Batch Operations

```rust
impl HarnessDb {
    /// Batch insert nodes (much faster than individual inserts)
    pub fn batch_upsert_nodes(&self, nodes: &[CodeNode]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(r#"
                INSERT OR REPLACE INTO code_nodes (...) VALUES (...)
            "#)?;
            for node in nodes {
                stmt.execute(params![...])?;
            }
        }
        tx.commit()?;
        Ok(())
    }
}
```

### 7.4 Index Strategy

```sql
-- Primary indexes (already created in schema)
-- idx_nodes_file, idx_nodes_state, idx_edges_from, idx_edges_to

-- Covering indexes for common queries
CREATE INDEX IF NOT EXISTS idx_nodes_state_file 
ON code_nodes(current_state, file_path);

CREATE INDEX IF NOT EXISTS idx_events_node_type 
ON code_events(node_id, event_type);
```

### 7.5 Expected Performance

| Operation | Expected Latency |
|-----------|------------------|
| Single node lookup | <1ms |
| Node insert/update | <1ms |
| Graph traversal (depth 3) | 5-20ms |
| Event append | <1ms |
| Event read (100) | <5ms |
| Vector search (10) | 10-50ms |
| State history query | <5ms |

**Throughput:**
- Batch inserts: 50,000-200,000 rows/second
- Event appends: 10,000-50,000 events/second
- Queries: 100,000+ reads/second

---

## Part 8: Event-Driven Architecture

### 8.1 Event Processing Loop

```rust
pub struct EventProcessor {
    db: HarnessDb,
    cursor_name: String,
}

impl EventProcessor {
    pub async fn run(&self) -> Result<()> {
        loop {
            let cursor = self.db.get_cursor(&self.cursor_name)?;
            let events = self.db.read_events(cursor, 100)?;
            
            if events.is_empty() {
                // No new events, sleep
                tokio::time::sleep(Duration::from_millis(100)).await;
                continue;
            }
            
            for event in &events {
                self.process_event(event)?;
            }
            
            // Update cursor
            if let Some(last) = events.last() {
                self.db.set_cursor(&self.cursor_name, last.sequence)?;
            }
        }
    }
    
    fn process_event(&self, event: &StoredEvent) -> Result<()> {
        match event.event_type.as_str() {
            "source_changed" => self.handle_source_changed(event)?,
            "tests_passed" => self.handle_tests_passed(event)?,
            "tests_failed" => self.handle_tests_failed(event)?,
            _ => {}
        }
        Ok(())
    }
    
    fn handle_source_changed(&self, event: &StoredEvent) -> Result<()> {
        if let Some(node_id) = &event.node_id {
            // Update state
            self.db.record_state_change(
                node_id,
                CodeState::StaleDirect,
                Some(event.sequence),
                Some(&event.event_type),
            )?;
            
            // Propagate to dependents
            let dependents = self.db.dependents(node_id, 10)?;
            for dep in dependents {
                self.db.append_event(&CodeEvent {
                    event_type: EventType::DependencyChanged,
                    node_id: Some(dep),
                    causation_id: Some(event.sequence),
                    ..Default::default()
                })?;
            }
        }
        Ok(())
    }
}
```

### 8.2 Pub/Sub for Real-Time Updates

```rust
use supersqlite_memory as mem;

impl HarnessDb {
    /// Publish state change notification
    pub fn publish_state_change(&self, node_id: &str, new_state: CodeState) -> Result<()> {
        let message = serde_json::json!({
            "node_id": node_id,
            "state": new_state.as_str(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        self.conn.execute(r#"
            SELECT mem_pubsub_publish('state_changes', ?1)
        "#, params![message.to_string()])?;
        Ok(())
    }
    
    /// Subscribe to state changes (for IDE integration)
    pub fn subscribe_state_changes(&self) -> Result<i64> {
        let sub_id: i64 = self.conn.query_row(
            "SELECT mem_pubsub_subscribe('state_changes')",
            [],
            |row| row.get(0),
        )?;
        Ok(sub_id)
    }
}
```

---

## Part 9: Backup and Recovery

### 9.1 Online Backup

```rust
impl HarnessDb {
    /// Create online backup
    pub fn backup(&self, dest_path: &str) -> Result<()> {
        self.conn.execute(
            &format!("VACUUM INTO '{}'", dest_path),
            [],
        )?;
        Ok(())
    }
}
```

### 9.2 Point-in-Time Recovery

```bash
# Using Litestream for continuous replication
litestream replicate brain.db s3://bucket/brain.db

# Restore to specific point in time
litestream restore -o brain-recovered.db -timestamp 2026-06-04T10:00:00Z s3://bucket/brain.db
```

### 9.3 Integrity Check

```rust
impl HarnessDb {
    pub fn check_integrity(&self) -> Result<bool> {
        let result: String = self.conn.query_row(
            "PRAGMA integrity_check",
            [],
            |row| row.get(0),
        )?;
        Ok(result == "ok")
    }
}
```

---

## Part 10: Integration with V2 Superstate

### 10.1 How It Fits Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           V2 SUPERSTATE VISION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        IN-MEMORY LAYER                                │  │
│  │                                                                       │  │
│  │   AST Parsers ──▶ CodeGraph ──▶ Statecharts ──▶ Propagation          │  │
│  │   (syn, naga,     (petgraph)    (superstate)   (BFS)                 │  │
│  │    rustpython)                                                        │  │
│  │                                                                       │  │
│  └────────────────────────────────────┬─────────────────────────────────┘  │
│                                       │                                    │
│                               SYNC    │   LOAD                             │
│                               (write) │   (read)                           │
│                                       │                                    │
│  ┌────────────────────────────────────▼─────────────────────────────────┐  │
│  │                      PERSISTENCE LAYER (this doc)                     │  │
│  │                                                                       │  │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │  │
│  │   │ code_nodes  │ │ code_edges  │ │ code_events │ │code_history │   │  │
│  │   │ (graph ext) │ │ (graph ext) │ │(streams ext)│ │(bitemp ext) │   │  │
│  │   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘   │  │
│  │                                                                       │  │
│  │                        supersqlite / brain.db                         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Sync Strategy

```rust
pub struct GraphSyncer {
    in_memory: CodeGraph,
    db: HarnessDb,
}

impl GraphSyncer {
    /// Load graph from database on startup
    pub fn load(&mut self) -> Result<()> {
        // Load all nodes
        let nodes = self.db.all_nodes()?;
        for node in nodes {
            self.in_memory.add_node(node.into());
        }
        
        // Load all edges
        let edges = self.db.all_edges()?;
        for edge in edges {
            self.in_memory.add_edge(edge.into());
        }
        
        Ok(())
    }
    
    /// Sync changes back to database
    pub fn sync(&self, changes: &[GraphChange]) -> Result<()> {
        let tx = self.db.transaction()?;
        
        for change in changes {
            match change {
                GraphChange::NodeAdded(node) => {
                    self.db.upsert_node(node)?;
                    self.db.append_event(&CodeEvent::node_created(node))?;
                }
                GraphChange::NodeRemoved(node_id) => {
                    self.db.delete_node(node_id)?;
                    self.db.append_event(&CodeEvent::node_deleted(node_id))?;
                }
                GraphChange::StateChanged { node_id, old, new } => {
                    self.db.record_state_change(node_id, *new, None, None)?;
                    self.db.append_event(&CodeEvent::state_changed(node_id, old, new))?;
                }
                // ... other changes
            }
        }
        
        tx.commit()?;
        Ok(())
    }
}
```

---

## Part 11: File Layout

```
TRINITY/
├── brain.db              # Main database (all state)
├── brain.db-wal          # Write-ahead log
├── brain.db-shm          # Shared memory
│
├── crates/
│   └── harness/          # The testing harness crate
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── db/
│           │   ├── mod.rs
│           │   ├── schema.sql    # Full schema (this doc)
│           │   ├── nodes.rs      # Node operations
│           │   ├── edges.rs      # Edge operations
│           │   ├── events.rs     # Event stream
│           │   ├── history.rs    # Bitemporal queries
│           │   └── vectors.rs    # Embedding search
│           ├── graph/
│           │   ├── mod.rs
│           │   └── ...
│           └── statechart/
│               ├── mod.rs
│               └── ...
│
└── docs/
    ├── V2_SUPERSTATE_VISION.md
    └── V2_SUPERSQLITE_PERSISTENCE.md  # This document
```

---

## Appendix A: Full Schema SQL

```sql
-- =============================================================================
-- TRINITY Testing Harness - SuperSQLite Schema
-- Version: 1.0.0
-- =============================================================================

-- Configuration
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA cache_size = -64000;

-- [Include all CREATE TABLE statements from Part 4]
-- [Include all CREATE INDEX statements]
-- [Include all CREATE VIEW statements]

-- Register extensions
SELECT graph_register('code', 'code_nodes', 'node_id', 'code_edges', 'from_node', 'to_node');
SELECT stream_register('code_events', 'sequence', 'timestamp');
SELECT bitemporal_register('code_state_history', 'valid_from', 'valid_to', 'system_from', 'system_to');
SELECT vec_register('code_embeddings', 'embedding', 384);
SELECT vec_index_create('code_embeddings', 'embedding', 'hnsw', 384);
```

---

## Appendix B: Extension Function Quick Reference

### Graph (supersqlite_graph)

```sql
graph_register(name, node_table, node_id_col, edge_table, from_col, to_col)
graph_node_create(graph, label, properties_json)
graph_edge_create(graph, from_id, to_id, type, properties_json)
graph_traverse(graph, start_node, direction, depth)  -- direction: 'outgoing'|'incoming'|'both'
graph_shortest_path(graph, from_node, to_node)
graph_pagerank(graph, iterations, damping)
graph_connected_components(graph)
```

### Streams (supersqlite_streams)

```sql
stream_register(table, sequence_col, timestamp_col)
stream_append(stream, type, payload, key, idempotency_key)
stream_read(stream, from_offset, limit)
stream_read_by_type(stream, type, from_offset, limit)
stream_read_by_key(stream, key, from_offset, limit)
stream_latest_offset(stream)
```

### Bitemporal (supersqlite_bitemporal)

```sql
bitemporal_register(table, valid_from, valid_to, system_from, system_to)
-- Then use AS OF in queries:
-- SELECT * FROM table WHERE ... AS OF '2026-06-01'
-- SELECT * FROM table WHERE ... BETWEEN '2026-06-01' AND '2026-06-02'
```

### Vector (supersqlite_vector)

```sql
vec_register(table, column, dimensions)
vec_index_create(table, column, type, dimensions)  -- type: 'hnsw'|'ivf'
vec_from_json(json_array)
vec_distance_L2(vec1, vec2)
vec_distance_cosine(vec1, vec2)
vec_search(table, column, query_vec, limit)
```

### Memory (supersqlite_memory)

```sql
mem_get(key)
mem_set(key, value, ttl_ms)
mem_del(key)
mem_incr(key, delta)
mem_pubsub_publish(channel, message)
mem_pubsub_subscribe(channel)
mem_lock_acquire(name, ttl_ms, owner)
mem_lock_release(name, owner)
```

---

## Appendix C: Migration Path

If you have existing data:

```sql
-- 1. Export from old format
-- 2. Create new schema
-- 3. Import with transformation

-- Example: Import from JSON files
INSERT INTO code_nodes (node_id, file_path, ...)
SELECT 
    json_extract(value, '$.id'),
    json_extract(value, '$.path'),
    ...
FROM json_each(readfile('nodes.json'));
```

---

**SuperSQLite gives us everything we need in one place. No network, no separate services, no complexity. Just a single file with 300+ functions ready to use.**
