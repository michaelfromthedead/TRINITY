//! Whitebox tests for trinity-harness database module.
//!
//! WHITEBOX coverage plan:
//!   - Path A: open_in_memory() creates valid connection with schema
//!   - Path B: schema.sql embedded correctly and creates expected tables
//!   - Path C: connection() returns working connection reference
//!   - Path D: code_units table has correct schema
//!   - Path E: edges table has correct schema with foreign keys
//!   - Path F: indices are created

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
