//! Whitebox tests for trinity-harness database module.
//!
//! WHITEBOX coverage plan:
//!   - Path A: open_in_memory() creates valid connection with schema
//!   - Path B: schema.sql embedded correctly and creates expected tables
//!   - Path C: connection() returns working connection reference
//!   - Path D: code_units table has correct schema
//!   - Path E: edges table has correct schema with foreign keys
//!   - Path F: indices are created
//!   - Path G: open(path) creates file on disk (T-HARNESS-1.2)
//!   - Path H: open(path) enables WAL journal mode (T-HARNESS-1.2)
//!   - Path I: open(path) sets synchronous pragma (T-HARNESS-1.2)
//!   - Path J: open(path) sets cache_size pragma (T-HARNESS-1.2)
//!   - Path K: open(path) initializes schema (T-HARNESS-1.2)
//!   - Path L: open(path) returns error for invalid path (T-HARNESS-1.2)

use tempfile::TempDir;
use trinity_harness::HarnessDb;

#[test]
fn test_open_in_memory_creates_valid_db() {
    // Path A: open_in_memory() success path
    let result = HarnessDb::open_in_memory();
    assert!(result.is_ok(), "open_in_memory() should succeed");
}

#[test]
fn test_schema_creates_code_units_table() {
    // Path B: code_units table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Query table info to verify schema
    let mut stmt = conn
        .prepare("PRAGMA table_info(code_units)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(columns.contains(&"id".to_string()), "should have id column");
    assert!(
        columns.contains(&"file_path".to_string()),
        "should have file_path column"
    );
    assert!(
        columns.contains(&"language".to_string()),
        "should have language column"
    );
    assert!(
        columns.contains(&"unit_type".to_string()),
        "should have unit_type column"
    );
    assert!(
        columns.contains(&"name".to_string()),
        "should have name column"
    );
    assert!(
        columns.contains(&"start_line".to_string()),
        "should have start_line column"
    );
    assert!(
        columns.contains(&"end_line".to_string()),
        "should have end_line column"
    );
    assert!(
        columns.contains(&"content_hash".to_string()),
        "should have content_hash column"
    );
    assert!(
        columns.contains(&"created_at".to_string()),
        "should have created_at column"
    );
}

#[test]
fn test_schema_creates_edges_table() {
    // Path E: edges table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(edges)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(columns.contains(&"id".to_string()), "should have id column");
    assert!(
        columns.contains(&"source_id".to_string()),
        "should have source_id column"
    );
    assert!(
        columns.contains(&"target_id".to_string()),
        "should have target_id column"
    );
    assert!(
        columns.contains(&"edge_type".to_string()),
        "should have edge_type column"
    );
}

#[test]
fn test_schema_creates_indices() {
    // Path F: indices are created
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_code_units_file".to_string()),
        "should have file index"
    );
    assert!(
        indices.contains(&"idx_code_units_language".to_string()),
        "should have language index"
    );
    assert!(
        indices.contains(&"idx_edges_source".to_string()),
        "should have edges source index"
    );
    assert!(
        indices.contains(&"idx_edges_target".to_string()),
        "should have edges target index"
    );
}

#[test]
fn test_connection_returns_usable_reference() {
    // Path C: connection() returns working reference
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Should be able to execute queries on the connection
    let result: i64 = conn
        .query_row("SELECT 1 + 1", [], |row| row.get(0))
        .expect("simple query should work");

    assert_eq!(result, 2);
}

#[test]
fn test_can_insert_and_query_code_unit() {
    // Verify the schema is actually usable for its intended purpose
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    conn.execute(
        "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        ("src/main.rs", "rust", "function", "main", 1, 10, "abc123"),
    )
    .expect("insert should work");

    let name: String = conn
        .query_row(
            "SELECT name FROM code_units WHERE file_path = ?",
            ["src/main.rs"],
            |row| row.get(0),
        )
        .expect("query should work");

    assert_eq!(name, "main");
}

#[test]
fn test_can_insert_and_query_edge() {
    // Verify edges table is usable
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // First insert two code units
    conn.execute(
        "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        ("src/a.rs", "rust", "function", "caller", 1, 5, "hash1"),
    )
    .expect("insert should work");

    conn.execute(
        "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        ("src/b.rs", "rust", "function", "callee", 1, 5, "hash2"),
    )
    .expect("insert should work");

    // Insert edge
    conn.execute(
        "INSERT INTO edges (source_id, target_id, edge_type) VALUES (?1, ?2, ?3)",
        (1, 2, "calls"),
    )
    .expect("edge insert should work");

    let edge_type: String = conn
        .query_row(
            "SELECT edge_type FROM edges WHERE source_id = 1",
            [],
            |row| row.get(0),
        )
        .expect("query should work");

    assert_eq!(edge_type, "calls");
}

// =============================================================================
// T-HARNESS-1.2: File-based HarnessDb::open(path) tests
// =============================================================================

#[test]
fn test_open_creates_file_on_disk() {
    // Path G: open(path) creates a database file
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    assert!(!db_path.exists(), "file should not exist before open");

    let _db = HarnessDb::open(db_path_str).expect("open should succeed");

    assert!(db_path.exists(), "file should exist after open");
}

#[test]
fn test_open_enables_wal_journal_mode() {
    // Path H: open(path) sets PRAGMA journal_mode = WAL
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_wal.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    let journal_mode: String = conn
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .expect("pragma query should work");

    assert_eq!(journal_mode, "wal", "journal mode should be WAL");
}

#[test]
fn test_open_sets_synchronous_normal() {
    // Path I: open(path) sets PRAGMA synchronous = NORMAL
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_sync.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    let synchronous: i64 = conn
        .query_row("PRAGMA synchronous", [], |row| row.get(0))
        .expect("pragma query should work");

    // SQLite synchronous values: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
    assert_eq!(synchronous, 1, "synchronous should be NORMAL (1)");
}

#[test]
fn test_open_sets_cache_size() {
    // Path J: open(path) sets PRAGMA cache_size = -64000 (64MB)
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_cache.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    let cache_size: i64 = conn
        .query_row("PRAGMA cache_size", [], |row| row.get(0))
        .expect("pragma query should work");

    // Negative value means KB, so -64000 = 64000 KB = ~64MB
    assert_eq!(cache_size, -64000, "cache_size should be -64000 (64MB)");
}

#[test]
fn test_open_initializes_schema() {
    // Path K: open(path) creates tables from schema.sql
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_schema.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    // Verify code_units table exists
    let table_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='code_units'",
            [],
            |row| row.get(0),
        )
        .expect("query should work");
    assert_eq!(table_count, 1, "code_units table should exist");

    // Verify edges table exists
    let edges_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='edges'",
            [],
            |row| row.get(0),
        )
        .expect("query should work");
    assert_eq!(edges_count, 1, "edges table should exist");
}

#[test]
fn test_open_schema_allows_insert_query() {
    // Path K (extended): schema is usable for inserts and queries
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_insert.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    conn.execute(
        "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        ("src/lib.rs", "rust", "module", "lib", 1, 100, "hash123"),
    )
    .expect("insert should work in file-based db");

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM code_units", [], |row| row.get(0))
        .expect("count query should work");

    assert_eq!(count, 1, "should have inserted one row");
}

#[test]
fn test_open_invalid_path_returns_error() {
    // Path L: open() with invalid path returns error
    let result = HarnessDb::open("/nonexistent/deeply/nested/path/that/cannot/exist/db.sqlite");

    assert!(result.is_err(), "open with invalid path should return error");
}

#[test]
fn test_open_creates_wal_sidecar_files() {
    // Verify WAL mode creates the expected sidecar files (-wal, -shm)
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_wal_files.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    // Force WAL activity by doing a write
    conn.execute(
        "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        ("src/test.rs", "rust", "function", "test_fn", 1, 5, "xyz"),
    )
    .expect("insert should work");

    // WAL and SHM files should exist after a write
    let wal_path = tmp_dir.path().join("test_wal_files.db-wal");
    let shm_path = tmp_dir.path().join("test_wal_files.db-shm");

    assert!(
        wal_path.exists(),
        "WAL file should exist after write: {:?}",
        wal_path
    );
    assert!(
        shm_path.exists(),
        "SHM file should exist after write: {:?}",
        shm_path
    );
}

#[test]
fn test_open_reopen_persists_data() {
    // Verify data persists across open/close cycles
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_persist.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    // First open: insert data
    {
        let db = HarnessDb::open(db_path_str).expect("first open should succeed");
        let conn = db.connection();

        conn.execute(
            "INSERT INTO code_units (file_path, language, unit_type, name, start_line, end_line, content_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            ("src/persist.rs", "rust", "struct", "PersistMe", 10, 20, "persist_hash"),
        )
        .expect("insert should work");
    } // db dropped here, connection closed

    // Second open: verify data persists
    {
        let db = HarnessDb::open(db_path_str).expect("second open should succeed");
        let conn = db.connection();

        let name: String = conn
            .query_row(
                "SELECT name FROM code_units WHERE file_path = ?",
                ["src/persist.rs"],
                |row| row.get(0),
            )
            .expect("query should find persisted data");

        assert_eq!(name, "PersistMe", "data should persist across reopens");
    }
}

#[test]
fn test_open_indices_exist_for_file_based_db() {
    // Verify indices are created in file-based db (same as in-memory)
    let tmp_dir = TempDir::new().expect("failed to create temp dir");
    let db_path = tmp_dir.path().join("test_indices.db");
    let db_path_str = db_path.to_str().expect("valid utf8 path");

    let db = HarnessDb::open(db_path_str).expect("open should succeed");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_code_units_file".to_string()),
        "should have file index in file-based db"
    );
    assert!(
        indices.contains(&"idx_code_units_language".to_string()),
        "should have language index in file-based db"
    );
    assert!(
        indices.contains(&"idx_edges_source".to_string()),
        "should have edges source index in file-based db"
    );
    assert!(
        indices.contains(&"idx_edges_target".to_string()),
        "should have edges target index in file-based db"
    );
}

// =============================================================================
// T-HARNESS-1.3: New Schema Tables Tests
// =============================================================================
// Tests for the 6 new tables introduced in schema.sql v2.0.0:
//   - code_nodes: Code units with hashes and state tracking
//   - code_edges: Dependency relationships between nodes
//   - code_events: Append-only event log
//   - code_state_history: Bitemporal state tracking
//   - code_contracts: Contract annotations
//   - struct_layouts: Struct memory layouts

// -----------------------------------------------------------------------------
// code_nodes table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_code_nodes_table() {
    // T-HARNESS-1.3: code_nodes table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_nodes)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    // Primary key and location
    assert!(
        columns.contains(&"node_id".to_string()),
        "should have node_id column"
    );
    assert!(
        columns.contains(&"file_path".to_string()),
        "should have file_path column"
    );
    assert!(
        columns.contains(&"span_start_line".to_string()),
        "should have span_start_line column"
    );
    assert!(
        columns.contains(&"span_start_col".to_string()),
        "should have span_start_col column"
    );
    assert!(
        columns.contains(&"span_end_line".to_string()),
        "should have span_end_line column"
    );
    assert!(
        columns.contains(&"span_end_col".to_string()),
        "should have span_end_col column"
    );

    // Identity
    assert!(
        columns.contains(&"language".to_string()),
        "should have language column"
    );
    assert!(
        columns.contains(&"kind".to_string()),
        "should have kind column"
    );
    assert!(
        columns.contains(&"name".to_string()),
        "should have name column"
    );
    assert!(
        columns.contains(&"qualified_name".to_string()),
        "should have qualified_name column"
    );

    // Hashes
    assert!(
        columns.contains(&"hash_full".to_string()),
        "should have hash_full column"
    );
    assert!(
        columns.contains(&"hash_signature".to_string()),
        "should have hash_signature column"
    );
    assert!(
        columns.contains(&"hash_body".to_string()),
        "should have hash_body column"
    );
    assert!(
        columns.contains(&"hash_layout".to_string()),
        "should have hash_layout column"
    );

    // State
    assert!(
        columns.contains(&"current_state".to_string()),
        "should have current_state column"
    );

    // Hierarchy
    assert!(
        columns.contains(&"parent_id".to_string()),
        "should have parent_id column"
    );
    assert!(
        columns.contains(&"depth".to_string()),
        "should have depth column"
    );

    // Timestamps
    assert!(
        columns.contains(&"created_at".to_string()),
        "should have created_at column"
    );
    assert!(
        columns.contains(&"updated_at".to_string()),
        "should have updated_at column"
    );
    assert!(
        columns.contains(&"last_tested_at".to_string()),
        "should have last_tested_at column"
    );
    assert!(
        columns.contains(&"last_changed_at".to_string()),
        "should have last_changed_at column"
    );
}

#[test]
fn test_code_nodes_language_constraint() {
    // T-HARNESS-1.3: code_nodes language CHECK constraint works
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Valid language should work
    let valid_result = conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("node1", "src/main.rs", 1, 10, "rust", "function", "main", "abc123"),
    );
    assert!(valid_result.is_ok(), "valid language should work");

    // Invalid language should fail
    let invalid_result = conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("node2", "src/bad.txt", 1, 10, "invalid_lang", "function", "bad", "xyz789"),
    );
    assert!(invalid_result.is_err(), "invalid language should fail CHECK constraint");
}

#[test]
fn test_code_nodes_state_constraint() {
    // T-HARNESS-1.3: code_nodes current_state CHECK constraint works
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Valid state should work
    let valid_result = conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full, current_state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        ("node_state1", "src/test.rs", 1, 10, "rust", "function", "test", "hash1", "tested_green"),
    );
    assert!(valid_result.is_ok(), "valid state should work");

    // Invalid state should fail
    let invalid_result = conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full, current_state)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        ("node_state2", "src/bad.rs", 1, 10, "rust", "function", "bad", "hash2", "invalid_state"),
    );
    assert!(invalid_result.is_err(), "invalid state should fail CHECK constraint");
}

#[test]
fn test_code_nodes_indices() {
    // T-HARNESS-1.3: code_nodes indices exist
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_nodes_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_nodes_file".to_string()),
        "should have idx_nodes_file index"
    );
    assert!(
        indices.contains(&"idx_nodes_language".to_string()),
        "should have idx_nodes_language index"
    );
    assert!(
        indices.contains(&"idx_nodes_kind".to_string()),
        "should have idx_nodes_kind index"
    );
    assert!(
        indices.contains(&"idx_nodes_name".to_string()),
        "should have idx_nodes_name index"
    );
    assert!(
        indices.contains(&"idx_nodes_state".to_string()),
        "should have idx_nodes_state index"
    );
    assert!(
        indices.contains(&"idx_nodes_parent".to_string()),
        "should have idx_nodes_parent index"
    );
    assert!(
        indices.contains(&"idx_nodes_qualified".to_string()),
        "should have idx_nodes_qualified index"
    );
    assert!(
        indices.contains(&"idx_nodes_state_file".to_string()),
        "should have idx_nodes_state_file composite index"
    );
}

// -----------------------------------------------------------------------------
// code_edges table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_code_edges_table() {
    // T-HARNESS-1.3: code_edges table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_edges)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        columns.contains(&"edge_id".to_string()),
        "should have edge_id column"
    );
    assert!(
        columns.contains(&"from_node".to_string()),
        "should have from_node column"
    );
    assert!(
        columns.contains(&"to_node".to_string()),
        "should have to_node column"
    );
    assert!(
        columns.contains(&"kind".to_string()),
        "should have kind column"
    );
    assert!(
        columns.contains(&"metadata".to_string()),
        "should have metadata column"
    );
    assert!(
        columns.contains(&"created_at".to_string()),
        "should have created_at column"
    );
}

#[test]
fn test_code_edges_kind_constraint() {
    // T-HARNESS-1.3: code_edges kind CHECK constraint works
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // First create nodes for foreign key
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("src_node", "src/a.rs", 1, 10, "rust", "function", "caller", "hash1"),
    ).expect("insert source node");

    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("dst_node", "src/b.rs", 1, 10, "rust", "function", "callee", "hash2"),
    ).expect("insert dest node");

    // Valid edge kind should work
    let valid_result = conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("edge1", "src_node", "dst_node", "calls"),
    );
    assert!(valid_result.is_ok(), "valid edge kind 'calls' should work");

    // Test more valid kinds
    let pyo3_result = conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("edge2", "src_node", "dst_node", "pyo3_call"),
    );
    assert!(pyo3_result.is_ok(), "valid edge kind 'pyo3_call' should work");

    // Invalid edge kind should fail
    let invalid_result = conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("edge3", "src_node", "dst_node", "invalid_kind"),
    );
    assert!(invalid_result.is_err(), "invalid edge kind should fail CHECK constraint");
}

#[test]
fn test_code_edges_unique_constraint() {
    // T-HARNESS-1.3: code_edges UNIQUE(from_node, to_node, kind) constraint
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create nodes
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("uniq_src", "src/a.rs", 1, 10, "rust", "function", "fn_a", "hash1"),
    ).expect("insert source node");

    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("uniq_dst", "src/b.rs", 1, 10, "rust", "function", "fn_b", "hash2"),
    ).expect("insert dest node");

    // First edge should work
    conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("uniq_edge1", "uniq_src", "uniq_dst", "calls"),
    ).expect("first edge should work");

    // Duplicate (same from, to, kind) should fail
    let dup_result = conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("uniq_edge2", "uniq_src", "uniq_dst", "calls"),
    );
    assert!(dup_result.is_err(), "duplicate edge should fail UNIQUE constraint");

    // Same nodes with different kind should work
    let diff_kind = conn.execute(
        "INSERT INTO code_edges (edge_id, from_node, to_node, kind) VALUES (?1, ?2, ?3, ?4)",
        ("uniq_edge3", "uniq_src", "uniq_dst", "references"),
    );
    assert!(diff_kind.is_ok(), "same nodes with different kind should work");
}

#[test]
fn test_code_edges_indices() {
    // T-HARNESS-1.3: code_edges indices exist
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_edges_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_edges_from".to_string()),
        "should have idx_edges_from index"
    );
    assert!(
        indices.contains(&"idx_edges_to".to_string()),
        "should have idx_edges_to index"
    );
    assert!(
        indices.contains(&"idx_edges_kind".to_string()),
        "should have idx_edges_kind index"
    );
    assert!(
        indices.contains(&"idx_edges_from_kind".to_string()),
        "should have idx_edges_from_kind composite index"
    );
    assert!(
        indices.contains(&"idx_edges_to_kind".to_string()),
        "should have idx_edges_to_kind composite index"
    );
}

// -----------------------------------------------------------------------------
// code_events table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_code_events_table() {
    // T-HARNESS-1.3: code_events table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_events)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        columns.contains(&"sequence".to_string()),
        "should have sequence column"
    );
    assert!(
        columns.contains(&"timestamp".to_string()),
        "should have timestamp column"
    );
    assert!(
        columns.contains(&"event_type".to_string()),
        "should have event_type column"
    );
    assert!(
        columns.contains(&"node_id".to_string()),
        "should have node_id column"
    );
    assert!(
        columns.contains(&"payload".to_string()),
        "should have payload column"
    );
    assert!(
        columns.contains(&"idempotency_key".to_string()),
        "should have idempotency_key column"
    );
    assert!(
        columns.contains(&"correlation_id".to_string()),
        "should have correlation_id column"
    );
    assert!(
        columns.contains(&"causation_id".to_string()),
        "should have causation_id column"
    );
}

#[test]
fn test_code_events_event_type_constraint() {
    // T-HARNESS-1.3: code_events event_type CHECK constraint works
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Valid event type should work
    let valid_result = conn.execute(
        "INSERT INTO code_events (event_type, node_id) VALUES (?1, ?2)",
        ("source_changed", "some_node"),
    );
    assert!(valid_result.is_ok(), "valid event_type should work");

    // Test more valid event types
    let test_passed = conn.execute(
        "INSERT INTO code_events (event_type, node_id) VALUES (?1, ?2)",
        ("tests_passed", "another_node"),
    );
    assert!(test_passed.is_ok(), "tests_passed event_type should work");

    // Invalid event type should fail
    let invalid_result = conn.execute(
        "INSERT INTO code_events (event_type, node_id) VALUES (?1, ?2)",
        ("invalid_event", "bad_node"),
    );
    assert!(invalid_result.is_err(), "invalid event_type should fail CHECK constraint");
}

#[test]
fn test_code_events_sequence_autoincrement() {
    // T-HARNESS-1.3: code_events sequence is autoincrement
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    conn.execute(
        "INSERT INTO code_events (event_type) VALUES (?1)",
        ("node_created",),
    ).expect("first insert");

    conn.execute(
        "INSERT INTO code_events (event_type) VALUES (?1)",
        ("node_deleted",),
    ).expect("second insert");

    let sequences: Vec<i64> = conn
        .prepare("SELECT sequence FROM code_events ORDER BY sequence")
        .expect("prepare")
        .query_map([], |row| row.get(0))
        .expect("query")
        .filter_map(|r| r.ok())
        .collect();

    assert_eq!(sequences.len(), 2, "should have 2 events");
    assert!(sequences[1] > sequences[0], "sequence should auto-increment");
}

#[test]
fn test_code_events_idempotency_key_unique() {
    // T-HARNESS-1.3: code_events idempotency_key UNIQUE constraint
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    conn.execute(
        "INSERT INTO code_events (event_type, idempotency_key) VALUES (?1, ?2)",
        ("source_changed", "key_123"),
    ).expect("first insert with idempotency_key");

    let dup_result = conn.execute(
        "INSERT INTO code_events (event_type, idempotency_key) VALUES (?1, ?2)",
        ("source_changed", "key_123"),
    );
    assert!(dup_result.is_err(), "duplicate idempotency_key should fail");
}

#[test]
fn test_code_events_indices() {
    // T-HARNESS-1.3: code_events indices exist
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_events_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_events_type".to_string()),
        "should have idx_events_type index"
    );
    assert!(
        indices.contains(&"idx_events_node".to_string()),
        "should have idx_events_node index"
    );
    assert!(
        indices.contains(&"idx_events_timestamp".to_string()),
        "should have idx_events_timestamp index"
    );
    assert!(
        indices.contains(&"idx_events_correlation".to_string()),
        "should have idx_events_correlation index"
    );
    assert!(
        indices.contains(&"idx_events_node_type".to_string()),
        "should have idx_events_node_type composite index"
    );
}

// -----------------------------------------------------------------------------
// code_state_history table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_code_state_history_table() {
    // T-HARNESS-1.3: code_state_history table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_state_history)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    // Primary key
    assert!(
        columns.contains(&"history_id".to_string()),
        "should have history_id column"
    );
    assert!(
        columns.contains(&"node_id".to_string()),
        "should have node_id column"
    );
    assert!(
        columns.contains(&"state".to_string()),
        "should have state column"
    );

    // Bitemporal columns
    assert!(
        columns.contains(&"valid_from".to_string()),
        "should have valid_from column"
    );
    assert!(
        columns.contains(&"valid_to".to_string()),
        "should have valid_to column"
    );
    assert!(
        columns.contains(&"system_from".to_string()),
        "should have system_from column"
    );
    assert!(
        columns.contains(&"system_to".to_string()),
        "should have system_to column"
    );

    // Causation tracking
    assert!(
        columns.contains(&"caused_by_event_id".to_string()),
        "should have caused_by_event_id column"
    );
    assert!(
        columns.contains(&"caused_by_event_type".to_string()),
        "should have caused_by_event_type column"
    );
    assert!(
        columns.contains(&"previous_state".to_string()),
        "should have previous_state column"
    );
}

#[test]
fn test_code_state_history_bitemporal_defaults() {
    // T-HARNESS-1.3: code_state_history has correct bitemporal defaults
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create a node first for FK
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("hist_node", "src/hist.rs", 1, 10, "rust", "function", "hist_fn", "hash1"),
    ).expect("insert node");

    // Insert state history with only required fields
    conn.execute(
        "INSERT INTO code_state_history (node_id, state, valid_from) VALUES (?1, ?2, ?3)",
        ("hist_node", "tested_green", "2026-01-01T00:00:00"),
    ).expect("insert history");

    // Check defaults
    let (valid_to, system_to): (String, String) = conn
        .query_row(
            "SELECT valid_to, system_to FROM code_state_history WHERE node_id = ?",
            ["hist_node"],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("query should work");

    assert_eq!(valid_to, "9999-12-31T23:59:59", "valid_to should default to far future");
    assert_eq!(system_to, "9999-12-31T23:59:59", "system_to should default to far future");
}

#[test]
fn test_code_state_history_indices() {
    // T-HARNESS-1.3: code_state_history indices exist
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_history_%'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_history_node".to_string()),
        "should have idx_history_node index"
    );
    assert!(
        indices.contains(&"idx_history_valid".to_string()),
        "should have idx_history_valid index"
    );
    assert!(
        indices.contains(&"idx_history_system".to_string()),
        "should have idx_history_system index"
    );
    assert!(
        indices.contains(&"idx_history_state".to_string()),
        "should have idx_history_state index"
    );
    assert!(
        indices.contains(&"idx_history_node_valid".to_string()),
        "should have idx_history_node_valid composite index"
    );
}

// -----------------------------------------------------------------------------
// code_contracts table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_code_contracts_table() {
    // T-HARNESS-1.3: code_contracts table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(code_contracts)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    // Primary key (also FK to code_nodes)
    assert!(
        columns.contains(&"node_id".to_string()),
        "should have node_id column"
    );

    // Contract predicates (JSON)
    assert!(
        columns.contains(&"requires".to_string()),
        "should have requires column"
    );
    assert!(
        columns.contains(&"ensures".to_string()),
        "should have ensures column"
    );
    assert!(
        columns.contains(&"invariants".to_string()),
        "should have invariants column"
    );
    assert!(
        columns.contains(&"properties".to_string()),
        "should have properties column"
    );

    // Verification
    assert!(
        columns.contains(&"last_verified_at".to_string()),
        "should have last_verified_at column"
    );
    assert!(
        columns.contains(&"verification_result".to_string()),
        "should have verification_result column"
    );
    assert!(
        columns.contains(&"verification_details".to_string()),
        "should have verification_details column"
    );
    assert!(
        columns.contains(&"updated_at".to_string()),
        "should have updated_at column"
    );
}

#[test]
fn test_code_contracts_verification_result_constraint() {
    // T-HARNESS-1.3: code_contracts verification_result CHECK constraint
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create a node first for FK
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("contract_node", "src/contract.rs", 1, 10, "rust", "function", "contract_fn", "hash1"),
    ).expect("insert node");

    // Valid verification_result should work
    let valid_result = conn.execute(
        "INSERT INTO code_contracts (node_id, verification_result) VALUES (?1, ?2)",
        ("contract_node", "passed"),
    );
    assert!(valid_result.is_ok(), "valid verification_result 'passed' should work");

    // NULL verification_result should work
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("contract_node2", "src/contract2.rs", 1, 10, "rust", "function", "contract_fn2", "hash2"),
    ).expect("insert node2");

    let null_result = conn.execute(
        "INSERT INTO code_contracts (node_id, verification_result) VALUES (?1, NULL)",
        ("contract_node2",),
    );
    assert!(null_result.is_ok(), "NULL verification_result should work");

    // Invalid verification_result should fail
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("contract_node3", "src/contract3.rs", 1, 10, "rust", "function", "contract_fn3", "hash3"),
    ).expect("insert node3");

    let invalid_result = conn.execute(
        "INSERT INTO code_contracts (node_id, verification_result) VALUES (?1, ?2)",
        ("contract_node3", "invalid_result"),
    );
    assert!(invalid_result.is_err(), "invalid verification_result should fail CHECK constraint");
}

#[test]
fn test_code_contracts_json_storage() {
    // T-HARNESS-1.3: code_contracts can store JSON predicates
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create a node first for FK
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("json_node", "src/json.rs", 1, 10, "rust", "function", "json_fn", "hash1"),
    ).expect("insert node");

    let requires_json = r#"["x > 0", "y != null"]"#;
    let ensures_json = r#"["result >= 0"]"#;

    conn.execute(
        "INSERT INTO code_contracts (node_id, requires, ensures) VALUES (?1, ?2, ?3)",
        ("json_node", requires_json, ensures_json),
    ).expect("insert contract with JSON");

    let (requires, ensures): (String, String) = conn
        .query_row(
            "SELECT requires, ensures FROM code_contracts WHERE node_id = ?",
            ["json_node"],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("query should work");

    assert_eq!(requires, requires_json, "requires JSON should be stored correctly");
    assert_eq!(ensures, ensures_json, "ensures JSON should be stored correctly");
}

// -----------------------------------------------------------------------------
// struct_layouts table tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_struct_layouts_table() {
    // T-HARNESS-1.3: struct_layouts table exists with correct columns
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("PRAGMA table_info(struct_layouts)")
        .expect("pragma should work");

    let columns: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        columns.contains(&"node_id".to_string()),
        "should have node_id column"
    );
    assert!(
        columns.contains(&"total_size".to_string()),
        "should have total_size column"
    );
    assert!(
        columns.contains(&"alignment".to_string()),
        "should have alignment column"
    );
    assert!(
        columns.contains(&"members".to_string()),
        "should have members column"
    );
    assert!(
        columns.contains(&"updated_at".to_string()),
        "should have updated_at column"
    );
}

#[test]
fn test_struct_layouts_required_fields() {
    // T-HARNESS-1.3: struct_layouts requires total_size, alignment, members
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create a node first for FK
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("layout_node", "src/layout.rs", 1, 10, "rust", "rust_struct", "MyStruct", "hash1"),
    ).expect("insert node");

    // Complete insert should work
    let members_json = r#"[{"name": "x", "offset": 0, "size": 4, "type": "u32"}]"#;
    let valid_result = conn.execute(
        "INSERT INTO struct_layouts (node_id, total_size, alignment, members) VALUES (?1, ?2, ?3, ?4)",
        ("layout_node", 4, 4, members_json),
    );
    assert!(valid_result.is_ok(), "complete struct_layouts insert should work");

    // Missing required field should fail
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("layout_node2", "src/layout2.rs", 1, 10, "rust", "rust_struct", "MyStruct2", "hash2"),
    ).expect("insert node2");

    let missing_members = conn.execute(
        "INSERT INTO struct_layouts (node_id, total_size, alignment) VALUES (?1, ?2, ?3)",
        ("layout_node2", 4, 4),
    );
    assert!(missing_members.is_err(), "missing members should fail NOT NULL constraint");
}

#[test]
fn test_struct_layouts_json_members() {
    // T-HARNESS-1.3: struct_layouts can store complex JSON members
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    // Create a node first for FK
    conn.execute(
        "INSERT INTO code_nodes (node_id, file_path, span_start_line, span_end_line, language, kind, name, hash_full)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        ("json_layout", "src/gpu.rs", 1, 20, "rust", "rust_struct", "GpuUniform", "hash1"),
    ).expect("insert node");

    let members_json = r#"[
        {"name": "model_matrix", "offset": 0, "size": 64, "type": "mat4x4<f32>"},
        {"name": "view_matrix", "offset": 64, "size": 64, "type": "mat4x4<f32>"},
        {"name": "proj_matrix", "offset": 128, "size": 64, "type": "mat4x4<f32>"}
    ]"#;

    conn.execute(
        "INSERT INTO struct_layouts (node_id, total_size, alignment, members) VALUES (?1, ?2, ?3, ?4)",
        ("json_layout", 192, 16, members_json),
    ).expect("insert layout");

    let (total_size, alignment, members): (i64, i64, String) = conn
        .query_row(
            "SELECT total_size, alignment, members FROM struct_layouts WHERE node_id = ?",
            ["json_layout"],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("query should work");

    assert_eq!(total_size, 192, "total_size should be correct");
    assert_eq!(alignment, 16, "alignment should be correct");
    assert!(members.contains("model_matrix"), "members JSON should contain model_matrix");
}

#[test]
fn test_struct_layouts_size_index() {
    // T-HARNESS-1.3: struct_layouts has idx_layouts_size index
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_layouts_size'")
        .expect("query should work");

    let indices: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        indices.contains(&"idx_layouts_size".to_string()),
        "should have idx_layouts_size index"
    );
}

// -----------------------------------------------------------------------------
// Views tests
// -----------------------------------------------------------------------------

#[test]
fn test_schema_creates_views() {
    // T-HARNESS-1.3: schema creates expected views
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_master WHERE type = 'view'")
        .expect("query should work");

    let views: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .expect("query should work")
        .filter_map(|r| r.ok())
        .collect();

    assert!(
        views.contains(&"layout_mismatches".to_string()),
        "should have layout_mismatches view"
    );
    assert!(
        views.contains(&"stale_nodes".to_string()),
        "should have stale_nodes view"
    );
    assert!(
        views.contains(&"untested_nodes".to_string()),
        "should have untested_nodes view"
    );
}

// -----------------------------------------------------------------------------
// All tables summary test
// -----------------------------------------------------------------------------

#[test]
fn test_all_six_new_tables_exist() {
    // T-HARNESS-1.3: Summary test - all 6 new tables exist
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let expected_tables = [
        "code_nodes",
        "code_edges",
        "code_events",
        "code_state_history",
        "code_contracts",
        "struct_layouts",
    ];

    for table in &expected_tables {
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                [table],
                |row| row.get(0),
            )
            .expect("query should work");

        assert_eq!(count, 1, "table '{}' should exist", table);
    }
}

#[test]
fn test_all_schema_indices_count() {
    // T-HARNESS-1.3: Verify all expected indices are created
    let db = HarnessDb::open_in_memory().expect("db should open");
    let conn = db.connection();

    let index_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'",
            [],
            |row| row.get(0),
        )
        .expect("query should work");

    // Expected indices:
    // code_nodes: 8 (file, language, kind, name, state, parent, qualified, state_file)
    // code_edges: 5 (from, to, kind, from_kind, to_kind)
    // code_events: 5 (type, node, timestamp, correlation, node_type)
    // code_state_history: 5 (node, valid, system, state, node_valid)
    // struct_layouts: 1 (size)
    // code_units (legacy): 2 (file, language)
    // edges (legacy): 2 (source, target)
    // Total: 28
    assert!(
        index_count >= 28,
        "should have at least 28 indices, found {}",
        index_count
    );
}
