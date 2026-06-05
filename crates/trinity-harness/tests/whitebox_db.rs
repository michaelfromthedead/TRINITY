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
