-- =============================================================================
-- TRINITY Testing Harness - Database Schema
-- Version: 2.0.0
-- =============================================================================
--
-- ARCHITECTURE NOTE:
--   v1 (current): Plain rusqlite. Graph traversal in Rust (HashMap BFS).
--   v2 (planned): SuperSQLite. Graph traversal in SQL (graph_traverse()).
--                 See docs/V2_SUPERSQLITE_PERSISTENCE.md for migration.
--
-- Tables:
--   1. code_nodes        - Code units (functions, structs, classes) with hashes
--   2. code_edges        - Dependency relationships between nodes
--   3. code_events       - State change events (touched, tested, etc.)
--   4. code_state_history - Bitemporal state tracking
--   5. code_contracts    - Contract annotations (#[requires], #[ensures])
--   6. struct_layouts    - Struct memory layouts for cross-language alignment
--
-- Designed for rusqlite with future SuperSQLite upgrade path.
-- =============================================================================

-- =============================================================================
-- 1. CODE NODES - Every code unit in the graph
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_nodes (
    -- Primary key (opaque string ID for flexibility)
    node_id TEXT PRIMARY KEY,

    -- Location
    file_path TEXT NOT NULL,
    span_start_line INTEGER NOT NULL,
    span_start_col INTEGER NOT NULL DEFAULT 0,
    span_end_line INTEGER NOT NULL,
    span_end_col INTEGER NOT NULL DEFAULT 0,

    -- Identity
    language TEXT NOT NULL CHECK (language IN ('rust', 'python', 'wgsl', 'toml', 'json')),
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT,

    -- Hashes for change detection (stored as hex strings)
    hash_full TEXT NOT NULL,
    hash_signature TEXT,
    hash_body TEXT,
    hash_layout TEXT,

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

-- Indexes for code_nodes
CREATE INDEX IF NOT EXISTS idx_nodes_file ON code_nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_language ON code_nodes(language);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON code_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON code_nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_state ON code_nodes(current_state);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON code_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_qualified ON code_nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_nodes_state_file ON code_nodes(current_state, file_path);

-- =============================================================================
-- 2. CODE EDGES - Dependencies and relationships between nodes
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

    -- Optional metadata (JSON)
    metadata TEXT,

    created_at TEXT NOT NULL DEFAULT (datetime('now')),

    -- Prevent duplicate edges
    UNIQUE(from_node, to_node, kind)
);

-- Indexes for code_edges
CREATE INDEX IF NOT EXISTS idx_edges_from ON code_edges(from_node);
CREATE INDEX IF NOT EXISTS idx_edges_to ON code_edges(to_node);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON code_edges(kind);
CREATE INDEX IF NOT EXISTS idx_edges_from_kind ON code_edges(from_node, kind);
CREATE INDEX IF NOT EXISTS idx_edges_to_kind ON code_edges(to_node, kind);

-- =============================================================================
-- 3. CODE EVENTS - Append-only event log
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

-- Indexes for code_events
CREATE INDEX IF NOT EXISTS idx_events_type ON code_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_node ON code_events(node_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON code_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON code_events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_node_type ON code_events(node_id, event_type);

-- =============================================================================
-- 4. CODE STATE HISTORY - Full temporal history of all state changes
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
CREATE INDEX IF NOT EXISTS idx_history_node_valid ON code_state_history(node_id, valid_from, valid_to);

-- =============================================================================
-- 5. CODE CONTRACTS - Pre/post conditions and properties
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_contracts (
    node_id TEXT PRIMARY KEY REFERENCES code_nodes(node_id) ON DELETE CASCADE,

    -- Preconditions (what callers must provide) - JSON array of predicates
    requires TEXT,

    -- Postconditions (what this promises to return) - JSON array of predicates
    ensures TEXT,

    -- Invariants (what must always hold) - JSON array of predicates
    invariants TEXT,

    -- Properties (algebraic laws) - JSON array of properties
    properties TEXT,

    -- Verification status
    last_verified_at TEXT,
    verification_result TEXT CHECK (verification_result IS NULL OR verification_result IN (
        'passed', 'failed', 'timeout', 'error'
    )),
    verification_details TEXT,

    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- =============================================================================
-- 6. STRUCT LAYOUTS - For detecting Rust/WGSL alignment mismatches
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

-- Index for layout queries by size (useful for finding large structs)
CREATE INDEX IF NOT EXISTS idx_layouts_size ON struct_layouts(total_size);

-- =============================================================================
-- VIEWS
-- =============================================================================

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

-- View to find stale nodes
CREATE VIEW IF NOT EXISTS stale_nodes AS
SELECT *
FROM code_nodes
WHERE current_state IN ('stale_direct', 'stale_transitive', 'stale_deep');

-- View to find untested code
CREATE VIEW IF NOT EXISTS untested_nodes AS
SELECT n.*
FROM code_nodes n
LEFT JOIN code_edges e ON n.node_id = e.to_node AND e.kind = 'tests'
WHERE e.edge_id IS NULL
  AND n.kind IN ('rust_function', 'python_function', 'wgsl_function')
  AND n.name NOT LIKE 'test_%';

-- =============================================================================
-- LEGACY COMPATIBILITY
-- =============================================================================
-- These tables are kept for backward compatibility with existing code.
-- New code should use code_nodes and code_edges instead.

CREATE TABLE IF NOT EXISTS code_units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    language TEXT NOT NULL,
    unit_type TEXT NOT NULL,
    name TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES code_units(id),
    FOREIGN KEY (target_id) REFERENCES code_units(id)
);

CREATE INDEX IF NOT EXISTS idx_code_units_file ON code_units(file_path);
CREATE INDEX IF NOT EXISTS idx_code_units_language ON code_units(language);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
